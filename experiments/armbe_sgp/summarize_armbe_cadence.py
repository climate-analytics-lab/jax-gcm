"""Summarize temporal semantics and empirical availability of standard ARMBE variables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import xarray as xr


FAMILIES = ("armbeatm", "armbecldrad", "armbeland")


def standard_files(root: Path) -> list[tuple[str, Path]]:
    """Discover downloaded standard-resolution ARMBE files by product family."""
    files = []
    for family in FAMILIES:
        for path in root.glob(f"*{family}*.c1/*"):
            if "hires" not in str(path) and path.suffix in {".nc", ".cdf"}:
                files.append((family, path))
    return sorted(files, key=lambda item: str(item[1]))


def decode_primary_time(ds: xr.Dataset) -> np.ndarray:
    """Decode ARMBE's primary time coordinate without decoding invalid auxiliaries."""
    time = ds["time"]
    return np.asarray(
        xr.coding.times.decode_cf_datetime(
            time.values,
            units=time.attrs["units"],
            calendar=time.attrs.get("calendar", "standard"),
        ),
        dtype="datetime64[ns]",
    )


def available_by_time(values: np.ndarray) -> np.ndarray:
    """Return whether each time row contains at least one usable value."""
    values = np.asarray(values)
    if values.dtype.kind in "SUO":
        valid = np.asarray([str(value).strip() not in {"", "nan", "None"} for value in values.flat])
        valid = valid.reshape(values.shape)
    elif values.dtype.kind == "M":
        valid = ~np.isnat(values)
    else:
        valid = np.isfinite(values)
    if valid.ndim == 1:
        return valid
    return np.any(valid, axis=tuple(range(1, valid.ndim)))


def temporal_semantics(name: str, attrs: dict, family: str, dimensions: tuple[str, ...]) -> str:
    """Classify what one time record represents without inferring undocumented operators."""
    text = " ".join(
        str(attrs.get(key, "")) for key in ("long_name", "description", "comment")
    ).lower()
    lower_name = name.lower()
    source = str(attrs.get("source_comment", "")).lower()
    if name in {"time", "time_bounds", "time_offset", "time_frac"}:
        return "time_coordinate_auxiliary"
    if lower_name.startswith("qc_") or "quality check" in text:
        return "quality_flag_for_hourly_cell"
    if lower_name.startswith("source_") or "source flag" in text:
        return "source_or_provenance_flag"
    if lower_name.startswith("stdev_") or "standard deviation" in text:
        return "within_hour_standard_deviation"
    if "hourly mean" in text or "hourly average" in text:
        return "one_hour_mean"
    profile_dimensions = {"p", "z", "pressure", "height"}
    if family == "armbeatm" and (
        "sonde" in source or bool(profile_dimensions & set(dimensions))
    ) and "nwp" not in lower_name:
        return "sounding_associated_hourly_cell"
    if "nwp" in lower_name:
        return "nwp_analysis_on_hourly_grid"
    if family in {"armbeatm", "armbeland"}:
        return "one_hour_mean"
    return "hourly_product_value_operator_not_explicit"


def cadence_label(minutes: int | None) -> str:
    if minutes is None:
        return "none"
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def summarize(root: Path) -> list[dict]:
    """Scan all files and return one record per exact family-variable name."""
    records: dict[tuple[str, str], dict] = {}
    for family, path in standard_files(root):
        with xr.open_dataset(path, decode_times=False) as ds:
            times = decode_primary_time(ds)
            site = path.name[:3]
            for name, variable in ds.variables.items():
                if "time" not in variable.dims:
                    continue
                key = (family, name)
                record = records.setdefault(
                    key,
                    {
                        "family": family,
                        "variable": name,
                        "long_names": set(),
                        "units": set(),
                        "dimensions": set(),
                        "source_comments": set(),
                        "semantics": set(),
                        "sites": set(),
                        "files_present": 0,
                        "time_slots": 0,
                        "available_slots": 0,
                        "intervals": Counter(),
                    },
                )
                attrs = variable.attrs
                record["long_names"].add(str(attrs.get("long_name", "")))
                record["units"].add(str(attrs.get("units", "")))
                record["dimensions"].add(" ".join(variable.dims))
                record["source_comments"].add(str(attrs.get("source_comment", "")))
                record["semantics"].add(
                    temporal_semantics(name, attrs, family, variable.dims)
                )
                record["sites"].add(site)
                record["files_present"] += 1
                record["time_slots"] += len(times)
                available = available_by_time(variable.values)
                record["available_slots"] += int(np.sum(available))
                available_times = times[available]
                if len(available_times) > 1:
                    minutes = np.diff(available_times).astype("timedelta64[m]").astype(int)
                    record["intervals"].update(minutes.tolist())

    output = []
    for record in records.values():
        intervals = record.pop("intervals")
        common = intervals.most_common(5)
        dominant = common[0][0] if common else None
        available = record["available_slots"]
        slots = record["time_slots"]
        output.append(
            {
                **{
                    key: "; ".join(sorted(value - {""})) if isinstance(value, set) else value
                    for key, value in record.items()
                },
                "availability_percent": 100.0 * available / slots if slots else 0.0,
                "dominant_available_interval": cadence_label(dominant),
                "dominant_interval_percent": (
                    100.0 * common[0][1] / sum(intervals.values()) if common else 0.0
                ),
                "common_available_intervals": "; ".join(
                    f"{cadence_label(minutes)}:{count}" for minutes, count in common
                ),
            }
        )
    return sorted(output, key=lambda item: (item["family"], item["variable"]))


def write_csv(path: Path, records: list[dict]) -> None:
    """Write the complete per-variable inventory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def write_markdown(path: Path, records: list[dict]) -> None:
    """Write a human-readable inventory while retaining every exact variable name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard ARMBE Variable Cadence",
        "",
        "This inventory covers every time-dependent variable in the downloaded standard",
        "`armbeatm`, `armbecldrad`, and `armbeland` files from order `267892`.",
        "High-resolution products are excluded. All products use one-hour time cells;",
        "`dominant interval` describes spacing between finite cells, not an averaging",
        "window. `Availability` is the finite-cell fraction within files containing that",
        "variable, pooled across sites and years. For profile variables, a cell is",
        "available when at least one vertical level is finite.",
        "",
        "Temporal semantics:",
        "",
        "- `one_hour_mean`: a one-hour aggregate, documented by the variable or product.",
        "- `sounding_associated_hourly_cell`: a radiosonde profile placed in its one-hour",
        "  cell; sparse availability does not imply a multi-hour mean.",
        "- `nwp_analysis_on_hourly_grid`: an NWP analysis field, not an observed hourly mean.",
        "- `within_hour_standard_deviation`: spread of native samples within the hour.",
        "- `quality_flag_for_hourly_cell` and `source_or_provenance_flag`: ancillary values.",
        "- `hourly_product_value_operator_not_explicit`: hourly-grid field whose exact",
        "  averaging operator is not stated consistently across all downloaded versions.",
        "",
    ]
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        lines.extend(
            [
                f"## {family.upper()}",
                "",
                "| Variable | Temporal semantics | Availability | Dominant interval | Common intervals | Sites |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for record in family_records:
            lines.append(
                f"| `{record['variable']}` | `{record['semantics']}` | "
                f"{record['availability_percent']:.1f}% | "
                f"{record['dominant_available_interval']} | "
                f"{record['common_available_intervals'] or 'none'} | "
                f"{record['sites']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    records = summarize(args.root)
    write_csv(args.output, records)
    if args.markdown_output:
        write_markdown(args.markdown_output, records)
    if args.summary_output:
        summary = {
            "scope": "Standard-resolution ARMBE files; high-resolution products excluded.",
            "variables": len(records),
            "families": {
                family: {
                    "variables": sum(record["family"] == family for record in records),
                    "semantics": dict(
                        sorted(
                            Counter(
                                record["semantics"]
                                for record in records
                                if record["family"] == family
                            ).items()
                        )
                    ),
                }
                for family in FAMILIES
            },
        }
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"variables": len(records), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
