"""Build compact, config-driven ARMBE free-forecast caches offline.

The cache contains no ARM files.  It is intentionally model-neutral apart from
the ARMBE-to-SPEEDY state conversion: JEM-Cal adapters reconstruct JAX inputs
from ``windows.nc`` and validate the recipe before calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from armbe_io import load_armbe, pick, to_obs_targets, to_state_series

DEFAULT_TARGET = {
    "observation": "cloud_fraction",
    "model": "shortwave_rad.cloudc",
    "reduction": "trajectory",
}
DEFAULT_CONFIG = {
    "nlev": 8,
    "physics_dt_minutes": 30,
    "horizon_minutes": 360,
    "stride_minutes": 360,
    "observation_cadence_minutes": 360,
    "target": DEFAULT_TARGET,
}


def resolved_config(config: Mapping[str, Any], root: Path | None = None) -> dict:
    """Apply defaults and resolve input paths before they enter a cache."""
    root = root or Path.cwd()
    target = {**DEFAULT_TARGET, **config.get("target", {})}
    resolved = {**DEFAULT_CONFIG, **config, "target": target}
    for key in ("atm", "cldrad"):
        if resolved.get(key) is not None:
            path = Path(resolved[key])
            resolved[key] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    if "atm" not in resolved:
        raise ValueError("cache config requires an 'atm' ARMBE file or directory")
    if target["reduction"] != "trajectory":
        raise ValueError("only the trajectory target reduction is currently supported")
    timing_keys = (
        "physics_dt_minutes", "horizon_minutes", "stride_minutes", "observation_cadence_minutes"
    )
    for key in timing_keys:
        if float(resolved[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(resolved["horizon_minutes"]) % float(resolved["physics_dt_minutes"]):
        raise ValueError("horizon_minutes must be divisible by physics_dt_minutes")
    if float(resolved["horizon_minutes"]) < float(resolved["observation_cadence_minutes"]):
        raise ValueError("horizon_minutes must include at least one observation cadence")
    if float(resolved["stride_minutes"]) % float(resolved["observation_cadence_minutes"]):
        raise ValueError("stride_minutes must be a multiple of observation_cadence_minutes")
    resolved.update({
        key.replace("_minutes", "_seconds"): float(resolved[key]) * 60
        for key in timing_keys
    })
    return resolved


def cache_recipe(config: Mapping[str, Any], resolved_variables: Mapping[str, str | None]) -> dict:
    """Return the comparison definition that travels with this cache."""
    return {
        "version": 2,
        "comparison": "armbe-sgp-free-forecast",
        "target": dict(config["target"]),
        "physics_dt_seconds": float(config["physics_dt_seconds"]),
        "horizon_seconds": float(config["horizon_seconds"]),
        "stride_seconds": float(config["stride_seconds"]),
        "observation_cadence_seconds": float(config["observation_cadence_seconds"]),
        "nlev": int(config["nlev"]),
        "resolved_variables": dict(resolved_variables),
        "target_order": "window, profile evaluation lead",
    }


def build_cache(config: Mapping[str, Any], cache: str | Path, root: Path | None = None) -> Path:
    """Materialize free-forecast windows, targets, resolved config, and recipe."""
    config = resolved_config(config, root)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    ds = load_armbe(config["atm"], config.get("cldrad"), config.get("start"), config.get("end"))
    states, times, meta = to_state_series(ds, nlev=int(config["nlev"]))
    if not states:
        raise ValueError("ARMBE contains no valid atmospheric profiles")
    physics_dt = int(config["physics_dt_seconds"])
    horizon = int(config["horizon_seconds"])
    observation_cadence = int(config["observation_cadence_seconds"])
    stride = int(config["stride_seconds"])
    steps = horizon // physics_dt
    seconds = np.asarray(times).astype("datetime64[s]").astype(np.int64)
    # Profiles define independent initial conditions, not the SCM integration grid.
    observation_phases, observation_counts = np.unique(seconds % observation_cadence, return_counts=True)
    start_phases, start_counts = np.unique(seconds % stride, return_counts=True)
    observation_phase = observation_phases[np.argmax(observation_counts)]
    start_phase = start_phases[np.argmax(start_counts)]
    surface_name = pick(ds, "surface_temperature", required=False)
    if surface_name is not None:
        surface_temperature = np.asarray(ds[surface_name].values, dtype=float)
    else:
        surface_temperature = np.full(ds.sizes["time"], 295.0)
        surface_name = "constant_295_k"
    forcing_times = np.asarray(ds.time.values)
    forcing_seconds = forcing_times.astype("datetime64[s]").astype(np.int64)
    start_indices = np.flatnonzero(
        (seconds % observation_cadence == observation_phase)
        & (seconds % stride == start_phase)
        & (seconds + (steps - 1) * physics_dt <= forcing_seconds[-1])
    )
    if not len(start_indices):
        raise ValueError("no profile-selected starts have surface-temperature forcing through horizon")
    n_windows = len(start_indices)
    target_name = config["target"]["observation"]
    observations = to_obs_targets(ds, meta["retained_indices"])
    if target_name not in observations:
        raise KeyError(f"ARMBE does not provide target {target_name!r}")
    target = np.asarray(observations[target_name])
    lead_times = np.arange(observation_cadence, horizon + 1, observation_cadence, dtype=np.int64)
    target_values = np.zeros((n_windows, len(lead_times)), dtype=float)
    target_mask = np.zeros((n_windows, len(lead_times)), dtype=bool)
    by_time = {time: i for i, time in enumerate(times.astype("datetime64[s]"))}
    cloud_qc = np.asarray(ds["qc_tot_cld"].values) if "qc_tot_cld" in ds else None
    for window, start_index in enumerate(start_indices):
        for evaluation, lead in enumerate(lead_times):
            profile_index = by_time.get(times[start_index].astype("datetime64[s]") + np.timedelta64(lead, "s"))
            if profile_index is None:
                continue
            value = target[profile_index]
            source_index = meta["retained_indices"][profile_index]
            qc_good = target_name != "cloud_fraction" or (
                cloud_qc is not None and cloud_qc[source_index] == 0
            )
            if np.isfinite(value) and qc_good:
                target_values[window, evaluation] = value
                target_mask[window, evaluation] = True
    state_fields = ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential",
                    "normalized_surface_pressure")
    data = {
        name: (("window", "level") if np.asarray(getattr(states[0], name)).ndim else ("window",),
                np.stack([np.asarray(getattr(states[i], name)) for i in start_indices]))
        for name in state_fields
    }
    data["target"] = (("window", "evaluation"), target_values)
    data["target_mask"] = (("window", "evaluation"), target_mask)
    data["surface_temperature"] = (("forcing_time",), surface_temperature)
    windows = xr.Dataset(
        data,
        coords={
            "window": np.arange(n_windows), "level": np.arange(int(config["nlev"])),
            "evaluation": np.arange(len(lead_times)),
            "lead_time_seconds": ("evaluation", lead_times),
            "start_time": ("window", times[start_indices]),
            "forcing_time": forcing_times,
        },
    )
    windows["surface_temperature"].attrs["source_variable"] = surface_name
    windows.to_netcdf(cache / "windows.nc")
    (cache / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    recipe = cache_recipe(config, meta["resolved"])
    (cache / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format": "armbe-sgp-free-forecast-cache-v2",
        "config": config,
        "recipe": recipe,
        "retained_states": int(len(states)),
        "windows": n_windows,
        "dropped_states": int(meta["n_dropped"]),
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return cache


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration")
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    out = build_cache(json.loads(config_path.read_text()), args.cache, config_path.parent)
    print(f"wrote ARMBE forecast cache to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
