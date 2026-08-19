"""Export site-aware native-T30 SPEEDY cloud features from unified ARMBE data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from dinosaur.sigma_coordinates import SigmaCoordinates

from export_symbolic_features import (
    FEATURE_COLUMNS,
    _cloud_features,
    _forcing_batch,
    _one_step,
)
from jcm.physics.speedy.speedy_coords import compute_speedy_vertical_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData


UNIFIED_FEATURE_COLUMNS = (*FEATURE_COLUMNS, "fmask")


def _state(samples: xr.Dataset, indices: np.ndarray) -> PhysicsState:
    """Construct the batch-shaped physical state for selected cache rows."""
    return PhysicsState(
        temperature=jnp.asarray(samples["temperature"].values[indices]),
        specific_humidity=jnp.asarray(samples["specific_humidity"].values[indices]),
        u_wind=jnp.asarray(samples["u_wind"].values[indices]),
        v_wind=jnp.asarray(samples["v_wind"].values[indices]),
        geopotential=jnp.asarray(samples["geopotential"].values[indices]),
        normalized_surface_pressure=jnp.asarray(
            samples["normalized_surface_pressure"].values[indices]
        ),
        tracers={},
    )


def _site_diagnoser(terrain_config: dict, nlev: int, fsg: jax.Array):
    """Build a compiled diagnostic for one static T30 terrain column."""
    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(nlev),
        terrain=TerrainData.single_column(
            orog=float(terrain_config["orog_m"]),
            fmask=float(terrain_config["fmask"]),
            lfluxland=bool(terrain_config["lfluxland"]),
        ),
        dt_seconds=1800.0,
    )

    @jax.jit
    def diagnose_batch(state, forcing):
        predictions = jax.vmap(lambda sample, force: _one_step(scm, sample, force))(state, forcing)
        return jax.vmap(lambda prediction: _cloud_features(prediction, fsg))(predictions)

    return diagnose_batch


def augment_fmask(features_dir: str | Path, terrain_config: str | Path) -> None:
    """Add the static site-specific T30 land mask without rerunning diagnostics."""
    features_dir = Path(features_dir)
    terrain_by_site = json.loads(Path(terrain_config).read_text())["sites"]
    features_path = features_dir / "features.nc"
    with xr.open_dataset(features_path) as raw:
        table = raw.load()
    sites = np.asarray(table["site_facility"].values).astype(str)
    missing_sites = set(sites) - set(terrain_by_site)
    if missing_sites:
        raise ValueError(f"terrain config is missing sites: {sorted(missing_sites)}")
    table["fmask"] = (
        "sample", np.asarray([terrain_by_site[site]["fmask"] for site in sites])
    )
    table.to_netcdf(features_path, mode="w")
    labels = np.asarray(table["split"].values).astype(str)
    for split in ("train", "validation", "test"):
        mask = labels == split
        frame = pd.DataFrame({"target": table["target"].values[mask]})
        for name in UNIFIED_FEATURE_COLUMNS:
            frame[name] = table[name].values[mask]
        frame.to_csv(features_dir / f"{split}.csv", index=False)
    manifest_path = features_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_columns"] = list(UNIFIED_FEATURE_COLUMNS)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def export_features(
    cache: str | Path,
    terrain_config: str | Path,
    out_dir: str | Path,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> Path:
    """Run independent SPEEDY diagnostics using each site's nearest T30 cell."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided")
    cache = Path(cache)
    terrain_config = Path(terrain_config)
    out_dir = Path(out_dir)
    with xr.open_dataset(cache) as raw:
        samples = raw.load()
    if max_samples is not None:
        samples = samples.isel(sample=slice(max_samples))
    terrain_by_site = json.loads(terrain_config.read_text())["sites"]
    site_names = np.asarray(samples["site_facility"].values).astype(str)
    unknown_sites = set(site_names) - set(terrain_by_site)
    if unknown_sites:
        raise ValueError(f"terrain config is missing sites: {sorted(unknown_sites)}")

    nlev = int(samples.sizes["level"])
    _hsg, fsg, _dhs, _sigl, _gs, _gc, _wvi = compute_speedy_vertical_coords(nlev)
    fsg = jnp.asarray(fsg)
    names = (*FEATURE_COLUMNS, "cloudc", "cloudstr", "speedy_raw_sum")
    values = {name: np.empty(samples.sizes["sample"], dtype=float) for name in names}
    for site in sorted(set(site_names)):
        indices = np.flatnonzero(site_names == site)
        diagnose_batch = _site_diagnoser(terrain_by_site[site], nlev, fsg)
        for first in range(0, len(indices), batch_size):
            batch_indices = indices[first : first + batch_size]
            forcing = _forcing_batch(
                np.asarray(samples["surface_temperature"].values[batch_indices]),
                np.asarray(samples["time"].values[batch_indices]),
            )
            diagnostics = diagnose_batch(_state(samples, batch_indices), forcing)
            for name in names:
                values[name][batch_indices] = np.asarray(diagnostics[name])

    times = np.asarray(samples["time"].values)
    table = xr.Dataset(
        {
            "target": ("sample", np.asarray(samples["target"].values)),
            "split": ("sample", np.asarray(samples["split"].values)),
            "site_facility": ("sample", site_names),
            "year": ("sample", times.astype("datetime64[Y]").astype(int) + 1970),
            "fmask": ("sample", np.asarray([terrain_by_site[site]["fmask"] for site in site_names])),
            **{name: ("sample", value) for name, value in values.items()},
        },
        coords={"sample": np.asarray(samples["sample"].values), "time": ("sample", times)},
        attrs={
            "target": "ARMBECLDRAD tot_cld with qc_tot_cld == 0",
            "feature_semantics": (
                "same-time default SPEEDY diagnostics using each site's nearest native T30 "
                "terrain cell; rh_cloudc_max is the maximum RH over cloudc reference sigmas "
                "and PBL top; precip_mm_day is 86.4 times precnv plus precls"
            ),
            "raw_sum_semantics": "literal cloudc plus cloudstr; no clipping or overlap rule",
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_netcdf(out_dir / "features.nc")
    for split in ("train", "validation", "test"):
        mask = np.asarray(samples["split"].values).astype(str) == split
        frame = pd.DataFrame({"target": table["target"].values[mask]})
        for name in UNIFIED_FEATURE_COLUMNS:
            frame[name] = table[name].values[mask]
        frame.to_csv(out_dir / f"{split}.csv", index=False)
    manifest = {
        "cache": str(cache.resolve()),
        "terrain_config": str(terrain_config.resolve()),
        "samples": int(samples.sizes["sample"]),
        "feature_columns": list(UNIFIED_FEATURE_COLUMNS),
        "target": "ARMBECLDRAD tot_cld with qc_tot_cld == 0",
        "raw_sum_baseline": "cloudc + cloudstr without clipping or overlap correction",
        "split_counts": {
            split: int(np.sum(np.asarray(samples["split"].values).astype(str) == split))
            for split in ("train", "validation", "test")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--terrain-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--augment-fmask-only",
        action="store_true",
        help="add fmask to an existing output without rerunning SPEEDY diagnostics",
    )
    args = parser.parse_args(argv)
    if args.augment_fmask_only:
        augment_fmask(args.out_dir, args.terrain_config)
        print(f"added fmask to symbolic feature tables in {args.out_dir}")
        return 0
    out_dir = export_features(
        args.cache, args.terrain_config, args.out_dir, args.batch_size, args.max_samples
    )
    print(f"wrote symbolic feature tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
