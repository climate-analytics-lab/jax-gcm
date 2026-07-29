"""Evaluate independent SPEEDY cloud diagnostics against simultaneous ARMBE cloud cover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr

from dinosaur.sigma_coordinates import SigmaCoordinates
from jcm.date import DateData
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData

from armbe_io import (
    SGP_LAT_DEG,
    SGP_LON_DEG,
    SGP_OROG_M,
    build_forcing,
    load_armbe,
    to_obs_targets,
    to_state_series,
)
from forecast_cache import load_config

DEFAULT_CONFIG = {
    "nlev": 8,
    "batch_size": 8,
    "target": {
        "observation": "cloud_fraction",
        "model": "shortwave_rad.cloudc",
    },
}


def resolved_config(config: dict, root: Path) -> dict:
    """Apply diagnostic defaults and resolve data paths."""
    target = {**DEFAULT_CONFIG["target"], **config.get("target", {})}
    resolved = {**DEFAULT_CONFIG, **config, "target": target}
    for key in ("atm", "cldrad"):
        if key not in resolved:
            raise ValueError(f"diagnostic config requires {key!r}")
        path = Path(resolved[key])
        resolved[key] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    if target != DEFAULT_CONFIG["target"]:
        raise ValueError(f"only target {DEFAULT_CONFIG['target']!r} is supported")
    if int(resolved["nlev"]) < 1:
        raise ValueError("nlev must be positive")
    if int(resolved["batch_size"]) < 1:
        raise ValueError("batch_size must be positive")
    return resolved


def _stack(tree_items):
    return jax.tree.map(lambda *values: jnp.stack(values), *tree_items)


def _one_step(scm: SingleColumnModel, state, forcing):
    """Diagnose one snapshot with no tracer or physics carry from another sample."""
    state_step = jax.tree.map(lambda value: value[None, ...], state)
    forcing_step = jax.tree.map(lambda value: value[None, ...], forcing)
    return scm.run(state_step, forcing_steps=forcing_step)


def _cloudc(predictions):
    term = predictions.physics_data["_shortwave_rad"]
    cloud = term.cloudc if hasattr(term, "cloudc") else term["cloudc"]
    return jnp.reshape(cloud, (cloud.shape[0], -1))[0, 0]


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Return RMSE over the finite, QC-passed comparison subset."""
    valid = np.asarray(mask, dtype=bool)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))


def run_diagnostic(config: dict, out_dir: Path) -> dict:
    """Run one independent diagnostic physics step for every valid ARMBE profile."""
    ds = load_armbe(config["atm"], config["cldrad"], config.get("start"), config.get("end"))
    states, times, meta = to_state_series(ds, nlev=int(config["nlev"]))
    if not states:
        raise ValueError("ARMBE contains no valid atmospheric profiles")
    retained_ds = ds.isel(time=meta["retained_indices"])
    target = np.asarray(to_obs_targets(ds, meta["retained_indices"])["cloud_fraction"])
    cloud_qc = np.asarray(ds["qc_tot_cld"].values)[meta["retained_indices"]]
    target_mask = np.isfinite(target) & (cloud_qc == 0)

    forcing, _ = build_forcing(retained_ds, times)
    dates = [
        DateData.set_date(
            jdt.to_datetime(np.datetime_as_string(time, unit="s")), 0, 1800.0, "gregorian"
        )
        for time in times
    ]
    forcing_at_time = _stack([forcing.select(date, calendar="gregorian") for date in dates])
    terrain = TerrainData.single_column(orog=SGP_OROG_M, fmask=1.0, lfluxland=True)
    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(int(config["nlev"])),
        lat_deg=SGP_LAT_DEG,
        lon_deg=SGP_LON_DEG,
        terrain=terrain,
        dt_seconds=1800.0,
        calendar="gregorian",
    )

    prediction_batches = []
    for first in range(0, len(states), int(config["batch_size"])):
        state_batch = _stack(states[first : first + int(config["batch_size"])])
        forcing_batch = jax.tree.map(
            lambda value: value[first : first + int(config["batch_size"])], forcing_at_time
        )
        predictions = jax.vmap(lambda state, force: _one_step(scm, state, force))(
            state_batch, forcing_batch
        )
        prediction_batches.append(np.asarray(jax.vmap(_cloudc)(predictions)))
    prediction = np.concatenate(prediction_batches)

    out_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "prediction": ("sample", prediction),
            "target": ("sample", target),
            "target_mask": ("sample", target_mask),
        },
        coords={"sample": np.arange(len(times)), "time": ("sample", times)},
    ).to_netcdf(out_dir / "cloud_pairs.nc")
    metrics = {
        "metric": "qc_masked_cloud_fraction_rmse",
        "rmse": masked_rmse(prediction, target, target_mask),
        "count": int(target_mask.sum()),
        "samples": int(len(times)),
        "dropped_profiles": int(meta["n_dropped"]),
        "semantics": "independent one-step diagnostics; no atmospheric, tracer, or physics carry",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest = {"config": config, "metrics": metrics, "outputs": {"pairs": "cloud_pairs.nc"}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML diagnostic configuration")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    metrics = run_diagnostic(resolved_config(load_config(config_path), config_path.parent), args.out_dir)
    print(f"cloud-fraction RMSE={metrics['rmse']:.4f} over {metrics['count']} QC-passed samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
