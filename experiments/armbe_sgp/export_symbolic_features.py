"""Export SPEEDY cloud-diagnostic predictors for symbolic regression.

Each output row corresponds to one sample in the independent ARMBE diagnostic
cache. The predictors are the effective scalar inputs to SPEEDY's ``cloudc``
and ``cloudstr`` formulas, not arbitrary profile-vector features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pandas as pd
import xarray as xr
from dinosaur.sigma_coordinates import SigmaCoordinates

from armbe_io import SGP_OROG_M, build_forcing
from jcm.date import DateData
from jcm.physics.radiation.speedy_shortwave import (
    PBL_TOP_SIGMA,
    _CLOUDC_REF_SIGMAS,
    interp_to_sigma,
)
from jcm.physics.speedy.speedy_coords import compute_speedy_vertical_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData


FEATURE_COLUMNS = ("rh_cloudc_max", "precip_mm_day", "gse", "rh_lowest")


def _stack(tree_items):
    return jax.tree.map(lambda *values: jnp.stack(values), *tree_items)


def _one_step(scm: SingleColumnModel, state, forcing):
    state_step = jax.tree.map(lambda value: value[None, ...], state)
    forcing_step = jax.tree.map(lambda value: value[None, ...], forcing)
    return scm.run(state_step, forcing_steps=forcing_step)


def _scalar(value):
    return jnp.reshape(value, (value.shape[0], -1))[0, 0]


def _cloud_features(predictions, fsg: jax.Array) -> dict[str, jax.Array]:
    """Extract the scalar inputs and outputs of SPEEDY's cloud diagnostics."""
    humidity = predictions.physics_data["_humidity"]
    convection = predictions.physics_data["_convection"]
    condensation = predictions.physics_data["_condensation"]
    shortwave = predictions.physics_data["_shortwave_rad"]
    rh = humidity.rh[0]
    rh_candidates = jnp.stack(
        [interp_to_sigma(rh, fsg, PBL_TOP_SIGMA)]
        + [interp_to_sigma(rh, fsg, sigma) for sigma in _CLOUDC_REF_SIGMAS]
    )
    cloudc = _scalar(shortwave.cloudc)
    cloudstr = _scalar(shortwave.cloudstr)
    return {
        "rh_cloudc_max": jnp.max(rh_candidates),
        "precip_mm_day": 86.4 * (_scalar(convection.precnv) + _scalar(condensation.precls)),
        "gse": _scalar(shortwave.gse),
        "rh_lowest": jnp.ravel(rh[-1])[0],
        "cloudc": cloudc,
        "cloudstr": cloudstr,
        "speedy_raw_sum": cloudc + cloudstr,
    }


def _forcing_batch(surface_temperature: np.ndarray, times: np.ndarray):
    forcing_ds = xr.Dataset(
        {"temp_sfc": ("time", surface_temperature)},
        coords={"time": times},
    )
    forcing, _ = build_forcing(forcing_ds, times)
    dates = [
        DateData.set_date(
            jdt.to_datetime(np.datetime_as_string(time, unit="s")), 0, 1800.0, "gregorian"
        )
        for time in times
    ]
    return _stack([forcing.select(date, calendar="gregorian") for date in dates])


def export_features(
    cache: str | Path,
    out_dir: str | Path,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> Path:
    """Run default SPEEDY diagnostics and write split-preserving feature tables."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided")
    cache = Path(cache)
    out_dir = Path(out_dir)
    with xr.open_dataset(cache / "samples.nc") as raw:
        samples = raw.load()
    if max_samples is not None:
        samples = samples.isel(sample=slice(max_samples))
    config = json.loads((cache / "config.json").read_text())
    terrain_config = config["terrain"]
    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(int(config["nlev"])),
        terrain=TerrainData.single_column(
            orog=float(terrain_config.get("orog_m", SGP_OROG_M)),
            fmask=float(terrain_config["fmask"]),
            lfluxland=bool(terrain_config["lfluxland"]),
        ),
        dt_seconds=1800.0,
    )
    _hsg, fsg, _dhs, _sigl, _gs, _gc, _wvi = compute_speedy_vertical_coords(int(config["nlev"]))
    fsg = jnp.asarray(fsg)

    @jax.jit
    def diagnose_batch(state, forcing):
        predictions = jax.vmap(lambda sample, force: _one_step(scm, sample, force))(state, forcing)
        return jax.vmap(lambda prediction: _cloud_features(prediction, fsg))(predictions)

    feature_values = {name: [] for name in (*FEATURE_COLUMNS, "cloudc", "cloudstr", "speedy_raw_sum")}
    n_samples = samples.sizes["sample"]
    for first in range(0, n_samples, batch_size):
        last = min(first + batch_size, n_samples)
        state = PhysicsState(
            temperature=jnp.asarray(samples["temperature"].values[first:last]),
            specific_humidity=jnp.asarray(samples["specific_humidity"].values[first:last]),
            u_wind=jnp.asarray(samples["u_wind"].values[first:last]),
            v_wind=jnp.asarray(samples["v_wind"].values[first:last]),
            geopotential=jnp.asarray(samples["geopotential"].values[first:last]),
            normalized_surface_pressure=jnp.asarray(
                samples["normalized_surface_pressure"].values[first:last]
            ),
            tracers={},
        )
        forcing = _forcing_batch(
            np.asarray(samples["surface_temperature"].values[first:last]),
            np.asarray(samples["time"].values[first:last]),
        )
        diagnostics = diagnose_batch(state, forcing)
        for name in feature_values:
            feature_values[name].append(np.asarray(diagnostics[name]))
    values = {name: np.concatenate(parts) for name, parts in feature_values.items()}
    table = xr.Dataset(
        {
            "target": ("sample", np.asarray(samples["target"].values)),
            "year": ("sample", np.asarray(samples["year"].values)),
            "split": ("sample", np.asarray(samples["split"].values)),
            **{name: ("sample", value) for name, value in values.items()},
        },
        coords={
            "sample": np.asarray(samples["sample"].values),
            "time": ("sample", np.asarray(samples["time"].values)),
        },
        attrs={
            "target": "ARMBE tot_cld",
            "feature_semantics": (
                "same-time default SPEEDY diagnostics; rh_cloudc_max is the maximum RH "
                "over cloudc reference sigmas and PBL top; precip_mm_day is 86.4 times "
                "precnv plus precls"
            ),
            "raw_sum_semantics": "literal cloudc plus cloudstr; no clipping or overlap rule",
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_netcdf(out_dir / "features.nc")
    for split in ("train", "validation", "test"):
        mask = np.asarray(samples["split"].values).astype(str) == split
        frame = pd.DataFrame({"target": table["target"].values[mask]})
        for name in FEATURE_COLUMNS:
            frame[name] = table[name].values[mask]
        frame.to_csv(out_dir / f"{split}.csv", index=False)
    manifest = {
        "cache": str(cache.resolve()),
        "samples": int(n_samples),
        "feature_columns": list(FEATURE_COLUMNS),
        "target": "ARMBE tot_cld",
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)
    out_dir = export_features(args.cache, args.out_dir, args.batch_size, args.max_samples)
    print(f"wrote symbolic feature tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
