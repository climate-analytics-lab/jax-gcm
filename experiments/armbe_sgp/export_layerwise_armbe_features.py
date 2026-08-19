"""Export standardized non-condensate Grundner-style ARMBE layer features.

This is a layer-cloud experiment, separate from the pooled column-total SPEEDY
experiment. It uses only high-resolution ARMBE pairs because pressure on the
height grid is required for the feature set.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import CubicSpline

from armbe_io import specific_humidity_from_dewpoint
from unified_cache import randomized_month_block_splits


BASE_FEATURES = (
    "wind_speed",
    "specific_humidity",
    "temperature",
    "pressure",
    "relative_humidity",
)
DERIVATIVE_FEATURES = tuple(
    f"d_{name}_dz" for name in BASE_FEATURES
) + tuple(f"d2_{name}_dz2" for name in BASE_FEATURES)
FEATURE_COLUMNS = (*BASE_FEATURES, *DERIVATIVE_FEATURES, "height_msl", "fmask", "surface_pressure")
SITE_BY_PREFIX = {"ena": "enaC1", "nsa": "nsaC1", "sgp": "sgpC1"}


def _site_and_year(path: Path) -> tuple[str, int]:
    match = re.match(r"([a-z]{3})armbeatmhiresC1\.c1\.(\d{4})", path.name)
    if not match or match.group(1) not in SITE_BY_PREFIX:
        raise ValueError(f"cannot determine supported site/year from {path.name}")
    return SITE_BY_PREFIX[match.group(1)], int(match.group(2))


def _interpolate_with_derivatives(
    source_height: np.ndarray, source: np.ndarray, target_height: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a profile and its cubic-spline derivatives onto target heights."""
    valid = np.isfinite(source_height) & np.isfinite(source)
    if valid.sum() < 4:
        missing = np.full(target_height.shape, np.nan)
        return missing, missing, missing
    height = source_height[valid]
    values = source[valid]
    order = np.argsort(height)
    height, values = height[order], values[order]
    unique = np.r_[True, np.diff(height) > 0.0]
    height, values = height[unique], values[unique]
    if len(height) < 4:
        missing = np.full(target_height.shape, np.nan)
        return missing, missing, missing
    spline = CubicSpline(height, values, extrapolate=False)
    return spline(target_height), spline(target_height, 1), spline(target_height, 2)


def _six_hour_indices(times: np.ndarray) -> np.ndarray:
    stamps = pd.DatetimeIndex(times)
    # High-resolution ARMBE stores ten-minute averages at five-minute centers.
    return np.flatnonzero((stamps.minute == 5) & np.isin(stamps.hour, (0, 6, 12, 18)))


def _append_file(
    atm_path: Path,
    cloud_path: Path,
    terrain: dict[str, float],
    layers_per_profile: int,
    rng: np.random.Generator,
) -> dict[str, list[np.ndarray]]:
    """Extract deterministic layer samples from one aligned high-resolution pair."""
    site, _year = _site_and_year(atm_path)
    with xr.open_dataset(atm_path) as raw_atm, xr.open_dataset(cloud_path) as raw_cloud:
        indices = _six_hour_indices(np.asarray(raw_cloud["time"].values))
        cloud = raw_cloud.isel(time=indices).load()
        atm = raw_atm.sel(time=cloud["time"]).load()
    cloud_height = np.asarray(cloud["height"].values, dtype=float)
    atm_height = np.asarray(atm["height"].values, dtype=float)
    altitude = float(np.asarray(atm["alt"].values))
    rows = {name: [] for name in ("target", *FEATURE_COLUMNS, "time", "site_facility")}
    for index, timestamp in enumerate(np.asarray(cloud["time"].values)):
        profiles = {
            "wind_speed": np.hypot(atm["u_wind_h"].values[index], atm["v_wind_h"].values[index]),
            "specific_humidity": specific_humidity_from_dewpoint(
                atm["dewpoint_h"].values[index], atm["pressure_h"].values[index]
            ),
            "temperature": atm["temperature_h"].values[index],
            "pressure": 100.0 * atm["pressure_h"].values[index],
            "relative_humidity": 0.01 * atm["relative_humidity_h"].values[index],
        }
        values = {}
        for name, profile in profiles.items():
            value, first, second = _interpolate_with_derivatives(atm_height, profile, cloud_height)
            values[name] = value
            values[f"d_{name}_dz"] = first
            values[f"d2_{name}_dz2"] = second
        values["height_msl"] = cloud_height + altitude
        values["fmask"] = np.full(len(cloud_height), terrain[site])
        values["surface_pressure"] = np.full(len(cloud_height), 100.0 * atm["pressure_sfc"].values[index])
        target = 0.01 * np.asarray(cloud["cld_frac"].values[index])
        valid = (np.asarray(cloud["qc_cld_frac"].values[index]) == 0) & np.isfinite(target)
        valid &= np.logical_and.reduce([np.isfinite(values[name]) for name in FEATURE_COLUMNS])
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        selected = rng.choice(candidates, size=min(layers_per_profile, len(candidates)), replace=False)
        rows["target"].append(target[selected])
        for name in FEATURE_COLUMNS:
            rows[name].append(values[name][selected])
        rows["time"].append(np.full(len(selected), timestamp, dtype="datetime64[ns]"))
        rows["site_facility"].append(np.full(len(selected), site, dtype="U8"))
    return rows


def _concatenate(parts: list[dict[str, list[np.ndarray]]]) -> dict[str, np.ndarray]:
    names = parts[0].keys()
    merged = {name: [value for part in parts for value in part[name]] for name in names}
    if not merged["target"]:
        raise ValueError("no QC-valid layer rows were available after interpolation")
    return {name: np.concatenate(values) for name, values in merged.items()}


def _split_labels(times: np.ndarray, sites: np.ndarray, seed: int) -> np.ndarray:
    labels = np.empty(len(times), dtype="U10")
    for site in np.unique(sites):
        rows = np.flatnonzero(sites == site)
        unique_times = np.unique(times[rows])
        time_labels = randomized_month_block_splits(unique_times, seed + sum(site.encode("ascii")))
        labels[rows] = time_labels[np.searchsorted(unique_times, times[rows])]
    return labels


def export_features(
    root: Path, terrain_path: Path, out_dir: Path, layers_per_profile: int, seed: int
) -> None:
    terrain = json.loads(terrain_path.read_text())["sites"]
    atm_paths = sorted(root.glob("**/*armbeatmhiresC1.c1/*.nc"))
    pairs = []
    for atm_path in atm_paths:
        site, _year = _site_and_year(atm_path)
        cloud_name = atm_path.name.replace("armbeatmhires", "armbecldradhires")
        cloud_dir = atm_path.parent.parent / atm_path.parent.name.replace("armbeatmhires", "armbecldradhires")
        cloud_path = cloud_dir / cloud_name
        if cloud_path.exists():
            pairs.append((atm_path, cloud_path, site))
    if not pairs:
        raise ValueError("no supported high-resolution ATM/CLDRAD pairs found")
    rng = np.random.default_rng(seed)
    parts = [
        _append_file(atm, cloud, {site: float(terrain[site]["fmask"])}, layers_per_profile, rng)
        for atm, cloud, site in pairs
    ]
    table = _concatenate(parts)
    labels = _split_labels(table["time"], table["site_facility"], seed)
    train = labels == "train"
    means = {name: float(np.mean(table[name][train])) for name in FEATURE_COLUMNS}
    stds = {name: float(np.std(table[name][train])) for name in FEATURE_COLUMNS}
    invalid = [name for name, std in stds.items() if not np.isfinite(std) or std == 0.0]
    if invalid:
        raise ValueError(f"zero or non-finite train standard deviation: {invalid}")
    features = np.column_stack([(table[name] - means[name]) / stds[name] for name in FEATURE_COLUMNS])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "features.npz",
        target=table["target"].astype(np.float32),
        features=features.astype(np.float32),
        feature_names=np.asarray(FEATURE_COLUMNS),
        split=labels,
        time=table["time"],
        site_facility=table["site_facility"],
    )
    search_rng = np.random.default_rng(seed)
    for split, limit in (("train", 5000), ("validation", 100_000), ("test", 100_000)):
        rows = np.flatnonzero(labels == split)
        selected = search_rng.choice(rows, size=min(limit, len(rows)), replace=False)
        frame = pd.DataFrame({"target": table["target"][selected]})
        for index, name in enumerate(FEATURE_COLUMNS):
            frame[name] = features[selected, index]
        frame.to_csv(out_dir / f"{split}.csv", index=False)
    manifest = {
        "experiment": "layerwise non-condensate ARMBE symbolic regression",
        "target": "high-resolution ARMBECLDRAD cld_frac / 100 with qc_cld_frac == 0",
        "source_pairs": [str(atm) for atm, _cloud, _site in pairs],
        "sites": sorted({site for _atm, _cloud, site in pairs}),
        "cadence": "six-hourly UTC samples from ten-minute high-resolution ARMBE records",
        "features": list(FEATURE_COLUMNS),
        "missing_features": ["qc", "qi", "d_qc_dz", "d2_qc_dz2", "d_qi_dz", "d2_qi_dz2"],
        "derivatives": "CubicSpline derivatives after interpolation onto CLDRAD height grid",
        "normalization": {"split": "train", "mean": means, "std": stds},
        "layers_per_profile": layers_per_profile,
        "seed": seed,
        "rows": {split: int(np.sum(labels == split)) for split in ("train", "validation", "test")},
        "csv_rows": {"train": 5000, "validation": min(100_000, int(np.sum(labels == "validation"))), "test": min(100_000, int(np.sum(labels == "test")))},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers-per-profile", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args(argv)
    export_features(args.root, args.terrain, args.out_dir, args.layers_per_profile, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
