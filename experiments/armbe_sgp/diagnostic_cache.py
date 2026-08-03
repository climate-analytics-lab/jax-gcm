"""Build compact same-time ARMBE cloud-diagnostic caches for JEM-Cal.

The cache holds only converted SPEEDY column states, the contemporaneous land
surface temperature, and QC-passed ``tot_cld`` targets. ARM files are read only
while constructing the cache, never in the calibration loop.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from armbe_io import (
    InvalidArmbeData,
    SGP_OROG_M,
    load_armbe,
    pick,
    to_obs_targets,
    to_state_series,
)
from forecast_cache import UniqueKeyLoader
import yaml


DEFAULT_CONFIG = {
    "nlev": 8,
    "split_seed": 20260731,
    "validation_years": 4,
    "test_years": 3,
    "target": {
        "observation": "cloud_fraction",
        "operator": "cloudc_plus_cloudstr_raw",
    },
    "terrain": {"orog_m": SGP_OROG_M, "fmask": 1.0, "lfluxland": True},
}
_YEAR_PATTERN = re.compile(r"\.(\d{4})\d{4}\.")


def load_config(path: Path) -> dict:
    """Load a diagnostic-cache YAML configuration without duplicate keys."""
    config = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def resolved_config(config: Mapping, root: Path) -> dict:
    """Resolve paths and validate the fixed raw-sum diagnostic definition."""
    target = {**DEFAULT_CONFIG["target"], **config.get("target", {})}
    resolved = {**DEFAULT_CONFIG, **config, "target": target}
    for key in ("atm", "cldrad"):
        if key not in resolved:
            raise ValueError(f"diagnostic cache requires {key!r}")
        resolved[key] = str(_resolve_path(resolved[key], root))
    if target != DEFAULT_CONFIG["target"]:
        raise ValueError(
            "diagnostic cache currently supports only the literal "
            "cloudc_plus_cloudstr_raw cloud-cover operator"
        )
    if int(resolved["nlev"]) < 1:
        raise ValueError("nlev must be positive")
    for key in ("validation_years", "test_years"):
        if int(resolved[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return resolved


def discover_annual_files(directory: str | Path) -> dict[int, Path]:
    """Return one annual ARMBE file per year, preferring modern NetCDF files."""
    directory = Path(directory)
    candidates: dict[int, list[Path]] = {}
    for pattern in ("*.cdf", "*.nc"):
        for path in directory.glob(pattern):
            match = _YEAR_PATTERN.search(path.name)
            if match is not None:
                candidates.setdefault(int(match.group(1)), []).append(path)
    selected = {}
    for year, paths in candidates.items():
        selected[year] = sorted(paths, key=lambda path: (path.suffix != ".nc", path.name))[0]
    return selected


def assign_year_splits(
    years: Sequence[int], *, validation_years: int, test_years: int, seed: int
) -> dict[int, str]:
    """Assign whole years reproducibly to train, validation, and test splits."""
    years = sorted(set(map(int, years)))
    held_out = int(validation_years) + int(test_years)
    if len(years) <= held_out:
        raise ValueError("need at least one training year after validation and test assignment")
    shuffled = np.random.default_rng(seed).permutation(years)
    split = {int(year): "validation" for year in shuffled[:validation_years]}
    split.update({int(year): "test" for year in shuffled[validation_years:held_out]})
    split.update({int(year): "train" for year in shuffled[held_out:]})
    return split


def _surface_temperature(ds: xr.Dataset, indices: np.ndarray) -> tuple[np.ndarray, str]:
    """Return finite K-valued surface temperatures for retained profile samples."""
    name = pick(ds, "surface_temperature", required=False)
    if name is None:
        return np.full(len(indices), 295.0), "constant_295_k"
    values = np.asarray(ds[name].values, dtype=float)
    if np.nanmax(values) < 100.0:
        values = values + 273.15
    values = values[indices]
    fallback = float(np.nanmean(values)) if np.isfinite(values).any() else 295.0
    return np.nan_to_num(values, nan=fallback), name


def build_diagnostic_cache(config: Mapping, cache: str | Path, root: Path | None = None) -> Path:
    """Materialize QC-valid independent cloud-diagnostic samples into NetCDF."""
    root = root or Path.cwd()
    config = resolved_config(config, root)
    atm_files = discover_annual_files(config["atm"])
    cldrad_files = discover_annual_files(config["cldrad"])
    common_years = sorted(set(atm_files) & set(cldrad_files))
    if not common_years:
        raise ValueError("no paired annual ARMBEATM and ARMBECLDRAD files found")
    audit = []
    eligible_years = []
    for year in common_years:
        try:
            load_armbe([atm_files[year]], [cldrad_files[year]], str(year), str(year + 1))
        except (InvalidArmbeData, ValueError) as error:
            audit.append({
                "year": year,
                "atm": str(atm_files[year]),
                "cldrad": str(cldrad_files[year]),
                "status": "excluded",
                "reason": str(error),
            })
        else:
            eligible_years.append(year)
    splits = assign_year_splits(
        eligible_years,
        validation_years=int(config["validation_years"]),
        test_years=int(config["test_years"]),
        seed=int(config["split_seed"]),
    )

    records: dict[str, list[np.ndarray]] = {
        "temperature": [], "specific_humidity": [], "u_wind": [], "v_wind": [],
        "geopotential": [], "normalized_surface_pressure": [], "surface_temperature": [],
        "target": [], "time": [], "year": [], "split": [],
    }
    for year in eligible_years:
        ds = load_armbe([atm_files[year]], [cldrad_files[year]], str(year), str(year + 1))
        states, times, meta = to_state_series(ds, nlev=int(config["nlev"]))
        retained = np.asarray(meta["retained_indices"], dtype=int)
        targets = np.asarray(to_obs_targets(ds, retained)["cloud_fraction"], dtype=float)
        cloud_qc = np.asarray(ds["qc_tot_cld"].values)[retained]
        valid = np.isfinite(targets) & (cloud_qc == 0)
        surface_temperature, source_name = _surface_temperature(ds, retained)
        selected = np.flatnonzero(valid)
        for index in selected:
            state = states[index]
            for field in ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential"):
                records[field].append(np.asarray(getattr(state, field)))
            records["normalized_surface_pressure"].append(
                np.asarray(state.normalized_surface_pressure)
            )
            records["surface_temperature"].append(np.asarray(surface_temperature[index]))
            records["target"].append(np.asarray(targets[index]))
            records["time"].append(np.asarray(times[index], dtype="datetime64[ns]"))
            records["year"].append(np.asarray(year, dtype=np.int32))
            records["split"].append(np.asarray(splits[year], dtype="U10"))
        audit.append({
            "year": year,
            "atm": str(atm_files[year]),
            "cldrad": str(cldrad_files[year]),
            "profiles": int(len(states)),
            "qc_valid_samples": int(len(selected)),
            "surface_temperature_source": source_name,
            "status": "included",
        })

    if not records["target"]:
        raise ValueError("no QC-valid diagnostic samples found")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    nlev = int(config["nlev"])
    data = {
        field: (("sample", "level"), np.stack(records[field]))
        for field in ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential")
    }
    data.update({
        field: ("sample", np.asarray(records[field]))
        for field in ("normalized_surface_pressure", "surface_temperature", "target", "year", "split")
    })
    dataset = xr.Dataset(
        data,
        coords={
            "sample": np.arange(len(records["target"])),
            "level": np.arange(nlev),
            "time": ("sample", np.asarray(records["time"], dtype="datetime64[ns]")),
        },
    )
    dataset.to_netcdf(cache / "samples.nc")
    recipe = {
        "version": 1,
        "comparison": "armbe-sgp-independent-speedy-diagnostic",
        "target": config["target"],
        "semantics": "independent one-step diagnostics; fresh tracer and physics carry; raw sum is unclipped",
        "nlev": nlev,
        "split_seed": int(config["split_seed"]),
        "year_splits": {str(year): split for year, split in sorted(splits.items())},
    }
    manifest = {
        "format": "armbe-sgp-diagnostic-cache-v1",
        "config": config,
        "recipe": recipe,
        "audit": audit,
        "samples": int(len(records["target"])),
        "split_counts": {
            split: int(sum(value == split for value in records["split"]))
            for split in ("train", "validation", "test")
        },
    }
    (cache / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (cache / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return cache


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cache = build_diagnostic_cache(load_config(config_path), args.cache, config_path.parent)
    print(f"wrote diagnostic cache to {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
