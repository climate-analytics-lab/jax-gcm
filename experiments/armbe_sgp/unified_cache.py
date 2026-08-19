"""Build a pooled, site-aware ARMBEATM/ARMBECLDRAD column dataset.

The cache holds one QC-valid observation per ``(timestamp, site-facility)``.
Splits are chronological within each site-facility record by valid-sample count,
so each timestamp-location observation receives one vote after pooling.
It deliberately contains no terrain prescription: that is a separate scientific
choice required before running site-specific SPEEDY diagnostics.
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from armbe_io import InvalidArmbeData, load_armbe, pick, to_obs_targets, to_state_series


_DATE_PATTERN = re.compile(r"\.(\d{8})\.")
_ATM_MARKER = "armbeatm"
_CLOUD_MARKER = "armbecldrad"


def _file_by_start_date(directory: Path) -> dict[str, Path]:
    """Return one file per start date, preferring modern NetCDF over legacy CDF."""
    candidates: dict[str, list[Path]] = {}
    for path in directory.glob("*.*"):
        if path.suffix not in {".nc", ".cdf"}:
            continue
        match = _DATE_PATTERN.search(path.name)
        if match is not None:
            candidates.setdefault(match.group(1), []).append(path)
    return {
        date: sorted(paths, key=lambda path: (path.suffix != ".nc", path.name))[0]
        for date, paths in candidates.items()
    }


def discover_paired_files(root: str | Path) -> list[tuple[str, Path, Path]]:
    """Discover standard-resolution ARMBEATM/ARMBECLDRAD file pairs in an order."""
    root = Path(root)
    pairs = []
    for atm_dir in sorted(root.glob(f"*{_ATM_MARKER}*.c1")):
        if "hires" in atm_dir.name:
            continue
        cloud_dir = root / atm_dir.name.replace(_ATM_MARKER, _CLOUD_MARKER)
        if not cloud_dir.is_dir():
            continue
        atm_files = _file_by_start_date(atm_dir)
        cloud_files = _file_by_start_date(cloud_dir)
        site_facility = atm_dir.name.replace(_ATM_MARKER, "").removesuffix(".c1")
        pairs.extend(
            (site_facility, atm_files[date], cloud_files[date])
            for date in sorted(set(atm_files) & set(cloud_files))
        )
    return pairs


def chronological_splits(
    times: np.ndarray, train_fraction: float = 0.7, validation_fraction: float = 0.2
) -> np.ndarray:
    """Assign chronological train, validation, and test splits by sample count."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    values = np.asarray(times, dtype="datetime64[ns]").astype("int64")
    if not len(values):
        raise ValueError("cannot split an empty time array")
    if len(values) < 3:
        raise ValueError("a site-facility record needs at least three timestamps")
    order = np.argsort(values, kind="stable")
    n_train = int(np.floor(len(values) * train_fraction))
    n_validation = int(np.floor(len(values) * validation_fraction))
    labels = np.empty(len(values), dtype="U10")
    labels[order[:n_train]] = "train"
    labels[order[n_train : n_train + n_validation]] = "validation"
    labels[order[n_train + n_validation :]] = "test"
    return labels


def randomized_month_block_splits(times: np.ndarray, seed: int) -> np.ndarray:
    """Randomly split whole year-month blocks, stratified by calendar month.

    A block is never shared between splits. Within a calendar-month stratum,
    multi-year records are divided as closely as possible to 70/20/10. A
    short campaign may have only one occurrence of a calendar month, so its
    singleton blocks are assigned randomly and balance is enforced only at the
    site-facility level.
    """
    values = np.asarray(times, dtype="datetime64[ns]")
    if len(values) < 3:
        raise ValueError("a site-facility record needs at least three timestamps")
    blocks = values.astype("datetime64[M]")
    month = (blocks.astype("int64") % 12).astype(int)
    rng = np.random.default_rng(seed)
    labels = np.empty(len(values), dtype="U10")
    fractions = np.asarray([0.7, 0.2, 0.1])
    names = np.asarray(["train", "validation", "test"])
    for calendar_month in range(12):
        block_values = np.unique(blocks[month == calendar_month])
        n_blocks = len(block_values)
        if not n_blocks:
            continue
        if n_blocks == 1:
            block_labels = rng.choice(names, size=1, p=fractions)
        else:
            block_weights = np.asarray([np.sum(blocks == block) for block in block_values])
            target = fractions * block_weights.sum()
            current = np.zeros(3)
            block_labels = np.empty(n_blocks, dtype="U10")
            for index in rng.permutation(n_blocks):
                scores = []
                for candidate in range(3):
                    prospective = current.copy()
                    prospective[candidate] += block_weights[index]
                    scores.append(np.sum((prospective - target) ** 2))
                best = np.flatnonzero(np.isclose(scores, np.min(scores)))
                candidate = int(rng.choice(best))
                block_labels[index] = names[candidate]
                current[candidate] += block_weights[index]
            if n_blocks >= 3:
                for missing in names[~np.isin(names, block_labels)]:
                    donor_candidates = np.flatnonzero(
                        [np.sum(block_labels == candidate) > 1 for candidate in names]
                    )
                    donor = names[int(rng.choice(donor_candidates))]
                    movable = np.flatnonzero(block_labels == donor)
                    move = int(rng.choice(movable))
                    block_labels[move] = missing
        for block, label in zip(block_values, block_labels, strict=True):
            labels[blocks == block] = label

    # Ensure each site-facility contributes to every split when it has enough
    # month blocks. This adjustment is only needed for short campaign records.
    for label in names:
        if not np.any(labels == label):
            donor = names[np.argmax([np.sum(labels == candidate) for candidate in names])]
            donor_blocks = np.unique(blocks[labels == donor])
            replacement = rng.choice(donor_blocks)
            labels[blocks == replacement] = label
    return labels


def _surface_temperature(ds: xr.Dataset, indices: np.ndarray) -> tuple[np.ndarray, str]:
    """Return finite Kelvin-valued surface temperatures for retained profile samples."""
    name = pick(ds, "surface_temperature", required=False)
    if name is None:
        return np.full(len(indices), 295.0), "constant_295_k"
    values = np.asarray(ds[name].values, dtype=float)
    if np.nanmax(values) < 100.0:
        values = values + 273.15
    values = values[indices]
    fallback = float(np.nanmean(values)) if np.isfinite(values).any() else 295.0
    return np.nan_to_num(values, nan=fallback), name


def _scalar_metadata(ds: xr.Dataset, name: str) -> float:
    if name not in ds.variables:
        return float("nan")
    value = np.asarray(ds[name].values)
    return float(value) if value.ndim == 0 else float("nan")


def _append(chunks: dict[str, list[np.ndarray]], key: str, values: Iterable) -> None:
    chunks[key].append(np.asarray(values))


def build_unified_cache(
    order_root: str | Path,
    output: str | Path,
    nlev: int = 8,
    split_policy: str = "chronological",
    split_seed: int = 20260731,
) -> Path:
    """Convert all locally available paired standard ARMBE records into one cache."""
    if nlev < 1:
        raise ValueError("nlev must be positive")
    if split_policy not in {"chronological", "random_month_blocks"}:
        raise ValueError("split_policy must be 'chronological' or 'random_month_blocks'")
    pairs = discover_paired_files(order_root)
    if not pairs:
        raise ValueError(f"no paired standard ARMBE files found in {order_root}")

    fields = (
        "temperature",
        "specific_humidity",
        "u_wind",
        "v_wind",
        "geopotential",
        "normalized_surface_pressure",
        "surface_temperature",
        "target",
        "time",
        "site_facility",
        "latitude",
        "longitude",
        "altitude_m",
    )
    chunks = {field: [] for field in fields}
    audit = []
    for site_facility, atm_path, cloud_path in pairs:
        try:
            ds = load_armbe([atm_path], [cloud_path])
            states, times, meta = to_state_series(ds, nlev=nlev)
            retained = np.asarray(meta["retained_indices"], dtype=int)
            targets = np.asarray(to_obs_targets(ds, retained)["cloud_fraction"], dtype=float)
            cloud_qc = np.asarray(ds["qc_tot_cld"].values)[retained]
            selected = np.flatnonzero(np.isfinite(targets) & (cloud_qc == 0))
            surface_temperature, temperature_source = _surface_temperature(ds, retained)
        except (InvalidArmbeData, KeyError, ValueError) as error:
            audit.append({
                "site_facility": site_facility,
                "atm": str(atm_path),
                "cldrad": str(cloud_path),
                "status": "excluded",
                "reason": str(error),
            })
            continue

        if not len(selected):
            audit.append({
                "site_facility": site_facility,
                "atm": str(atm_path),
                "cldrad": str(cloud_path),
                "profiles": int(len(states)),
                "qc_valid_samples": 0,
                "surface_temperature_source": temperature_source,
                "status": "excluded",
                "reason": "no finite tot_cld samples with qc_tot_cld == 0",
            })
            continue

        for field in ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential"):
            _append(chunks, field, np.stack([np.asarray(getattr(states[i], field)) for i in selected]))
        _append(
            chunks,
            "normalized_surface_pressure",
            [np.asarray(states[i].normalized_surface_pressure) for i in selected],
        )
        _append(chunks, "surface_temperature", surface_temperature[selected])
        _append(chunks, "target", targets[selected])
        _append(chunks, "time", times[selected].astype("datetime64[ns]"))
        _append(chunks, "site_facility", np.full(len(selected), site_facility, dtype="U16"))
        for field, name in (("latitude", "lat"), ("longitude", "lon"), ("altitude_m", "alt")):
            _append(chunks, field, np.full(len(selected), _scalar_metadata(ds, name)))
        audit.append({
            "site_facility": site_facility,
            "atm": str(atm_path),
            "cldrad": str(cloud_path),
            "profiles": int(len(states)),
            "qc_valid_samples": int(len(selected)),
            "surface_temperature_source": temperature_source,
            "status": "included",
        })

    if not chunks["target"]:
        raise ValueError("no QC-valid paired ARMBE samples found")
    values = {field: np.concatenate(parts) for field, parts in chunks.items()}
    split = np.empty(len(values["target"]), dtype="U10")
    split_boundaries = {}
    seasonal_sample_counts = {}
    for site_facility in sorted(set(values["site_facility"].astype(str))):
        indices = np.flatnonzero(values["site_facility"] == site_facility)
        site_times = values["time"][indices]
        if split_policy == "chronological":
            site_split = chronological_splits(site_times)
        else:
            site_seed = split_seed + zlib.crc32(site_facility.encode("ascii"))
            site_split = randomized_month_block_splits(site_times, site_seed)
        split[indices] = site_split
        split_boundaries[site_facility] = {
            "first_timestamp": str(site_times.min()),
            "last_timestamp": str(site_times.max()),
        }
        month = (site_times.astype("datetime64[M]").astype("int64") % 12 + 1).astype(int)
        seasonal_sample_counts[site_facility] = {
            str(calendar_month): {
                name: int(np.sum((month == calendar_month) & (site_split == name)))
                for name in ("train", "validation", "test")
            }
            for calendar_month in range(1, 13)
        }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = {
        field: (("sample", "level"), values[field])
        for field in ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential")
    }
    data.update({
        field: ("sample", values[field])
        for field in (
            "normalized_surface_pressure",
            "surface_temperature",
            "target",
            "site_facility",
            "latitude",
            "longitude",
            "altitude_m",
        )
    })
    data["split"] = ("sample", split)
    dataset = xr.Dataset(
        data,
        coords={
            "sample": np.arange(len(values["target"])),
            "level": np.arange(nlev),
            "time": ("sample", values["time"]),
        },
        attrs={
            "target": "ARMBECLDRAD tot_cld with qc_tot_cld == 0",
            "sample_semantics": "one QC-valid timestamp-site_facility observation per sample",
            "terrain_status": "not prescribed; site terrain is required before SPEEDY diagnostics",
        },
    )
    dataset.to_netcdf(output / "samples.nc")
    manifest = {
        "format": "armbe-unified-paired-cache-v1",
        "source_order": str(Path(order_root).resolve()),
        "nlev": nlev,
        "split_policy": {
            "type": (
                "per-site-facility chronological sample count"
                if split_policy == "chronological"
                else "per-site-facility randomized year-month blocks stratified by calendar month"
            ),
            "train_fraction": 0.7,
            "validation_fraction": 0.2,
            "test_fraction": 0.1,
            "seed": split_seed if split_policy == "random_month_blocks" else None,
            "boundaries": split_boundaries,
            "seasonal_sample_counts": seasonal_sample_counts,
        },
        "samples": int(len(values["target"])),
        "site_sample_counts": {
            site: int(np.sum(values["site_facility"] == site))
            for site in sorted(set(values["site_facility"].astype(str)))
        },
        "split_counts": {name: int(np.sum(split == name)) for name in ("train", "validation", "test")},
        "audit": audit,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nlev", type=int, default=8)
    parser.add_argument(
        "--split-policy", choices=("chronological", "random_month_blocks"), default="chronological"
    )
    parser.add_argument("--split-seed", type=int, default=20260731)
    args = parser.parse_args(argv)
    output = build_unified_cache(
        args.order_root, args.output, args.nlev, args.split_policy, args.split_seed
    )
    print(f"wrote unified ARMBE cache to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
