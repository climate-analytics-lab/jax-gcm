"""Process staged MICROBASE days into resumable hourly and paired caches."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from collocate_microbase_pilot import (
    DEFAULT_ATM,
    DEFAULT_CLDRAD,
    _sha256,
    build_hourly_cache,
)


FILENAME_DATE = re.compile(r"\.(\d{8})\.\d{6}\.nc$")
PROCESSING_SCHEMA_VERSION = 4


def _file_day(path: Path) -> date:
    match = FILENAME_DATE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse ARM date from {path.name}")
    return date.fromisoformat(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
    )


def verify_day_outputs(hourly_path: Path, paired_path: Path, day: str) -> None:
    """Reopen a reduced day and reject incomplete or misaligned output."""
    start = np.datetime64(day)
    end = start + np.timedelta64(1, "D")
    with xr.open_dataset(hourly_path) as hourly, xr.open_dataset(paired_path) as paired:
        if hourly.sizes != {"time": 24, "height": 596}:
            raise ValueError(f"Unexpected hourly dimensions: {dict(hourly.sizes)}")
        if paired.sizes.get("height") != 596 or paired.sizes.get("time", 0) < 1:
            raise ValueError(f"Unexpected paired dimensions: {dict(paired.sizes)}")
        if not np.all((hourly.time.values >= start) & (hourly.time.values < end)):
            raise ValueError("Hourly output contains timestamps outside its source day")
        if not np.all(np.isin(paired.time.values, hourly.time.values)):
            raise ValueError("Paired atmospheric times are absent from hourly output")
        if hourly.armbe_cloud_fraction.count().item() == 0:
            raise ValueError("Hourly output contains an empty cloud-fraction variable")
        liquid_valid = np.isfinite(hourly.liquid_water_concentration.values)
        ice_valid = np.isfinite(hourly.ice_water_concentration.values)
        pair_valid = hourly.condensate_pair_count.values > 0
        if not (
            np.array_equal(liquid_valid, ice_valid)
            and np.array_equal(liquid_valid, pair_valid)
        ):
            raise ValueError("Hourly condensate values disagree with retrieval-pair counts")
        atmospheric_valid = paired.atmospheric_valid.values.astype(bool)
        if not np.any(atmospheric_valid):
            raise ValueError("Paired output has no sounding-supported atmospheric cells")
        for name in (
            "temperature",
            "relative_humidity",
            "dewpoint",
            "pressure",
            "air_density",
            "specific_humidity",
        ):
            if not np.all(np.isfinite(paired[name].values[atmospheric_valid])):
                raise ValueError(f"Paired output has invalid {name} in supported cells")
        model_sample_valid = paired.model_sample_valid.values.astype(bool)
        for name in ("qc", "qi", "armbe_cloud_fraction"):
            if not np.all(np.isfinite(paired[name].values[model_sample_valid])):
                raise ValueError(f"Paired output has invalid {name} in model samples")
        for profile in paired.pressure.values:
            finite = profile[np.isfinite(profile)]
            if finite.size and not np.all(np.diff(finite) < 0.0):
                raise ValueError("Paired pressure must decrease with height")


def _completed(day_dir: Path, source: Path, day: str) -> bool:
    hourly_path = day_dir / "microbase_hourly.nc"
    paired_path = day_dir / "observed_atmosphere_paired.nc"
    manifest_path = day_dir / "manifest.json"
    if not (hourly_path.exists() and paired_path.exists() and manifest_path.exists()):
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("processing_schema_version") != PROCESSING_SCHEMA_VERSION:
        return False
    source_record = manifest.get("sources", {}).get("microbase", {})
    if source_record.get("size_bytes") != source.stat().st_size:
        return False
    verify_day_outputs(hourly_path, paired_path, day)
    return True


def _upgrade_existing_days(output: Path) -> None:
    """Add schema-4 validity masks to verified schema-3 reduced outputs."""
    for day_dir in sorted(path for path in output.iterdir() if path.is_dir()):
        manifest_path = day_dir / "manifest.json"
        paired_path = day_dir / "observed_atmosphere_paired.nc"
        if not (manifest_path.exists() and paired_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("processing_schema_version") != 3:
            continue
        with xr.open_dataset(paired_path) as source:
            paired = source.load()
        paired["model_sample_valid"] = (
            paired.atmospheric_valid
            & (paired.condensate_pair_count > 0)
            & paired.liquid_water_concentration.notnull()
            & paired.ice_water_concentration.notnull()
            & (paired.armbe_cloud_fraction_qc == 0)
            & paired.armbe_cloud_fraction.notnull()
        )
        paired.model_sample_valid.attrs.update(
            description=(
                "True where atmosphere, condensate pair, and primary ARMBE "
                "target are all valid"
            )
        )
        temporary = paired_path.with_suffix(".nc.tmp")
        paired.to_netcdf(temporary)
        temporary.replace(paired_path)
        verify_day_outputs(day_dir / "microbase_hourly.nc", paired_path, day_dir.name)
        manifest["processing_schema_version"] = PROCESSING_SCHEMA_VERSION
        record = manifest["outputs"]["observed_atmosphere_paired"]
        record["size_bytes"] = paired_path.stat().st_size
        record["sha256"] = _sha256(paired_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"upgraded verified cache {day_dir.name} to schema 4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbase-dir", type=Path, required=True)
    parser.add_argument("--atmosphere", type=Path, default=DEFAULT_ATM)
    parser.add_argument("--cldrad", type=Path, default=DEFAULT_CLDRAD)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 6, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2018, 7, 1))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--delete-raw-after-verify",
        action="store_true",
        help="delete each MICROBASE source only after reduced outputs pass validation",
    )
    args = parser.parse_args(argv)
    if args.end <= args.start:
        parser.error("--end must be later than --start")

    candidates = sorted(args.microbase_dir.glob("sgpmicrobaseC1.c1.*.nc"))
    selected = [path for path in candidates if args.start <= _file_day(path) < args.end]
    if not selected:
        parser.error("no MICROBASE files found in the requested date range")
    args.output.mkdir(parents=True, exist_ok=True)
    _upgrade_existing_days(args.output)

    shared_sources = {
        "armbeatm": {
            "path": str(args.atmosphere.resolve()),
            "size_bytes": args.atmosphere.stat().st_size,
            "sha256": _sha256(args.atmosphere),
        },
        "armbecldrad": {
            "path": str(args.cldrad.resolve()),
            "size_bytes": args.cldrad.stat().st_size,
            "sha256": _sha256(args.cldrad),
        },
    }
    processed = 0
    skipped = 0
    with (
        xr.open_dataset(args.atmosphere) as atmosphere,
        xr.open_dataset(args.cldrad) as cldrad,
    ):
        for source in selected:
            source_day = _file_day(source).isoformat()
            day_dir = args.output / source_day
            day_dir.mkdir(parents=True, exist_ok=True)
            if _completed(day_dir, source, source_day):
                print(f"skip verified {source_day}")
                skipped += 1
                continue

            print(f"process {source_day}: {source.name}")
            with xr.open_dataset(source) as microbase:
                hourly, paired, audit = build_hourly_cache(
                    microbase, atmosphere, cldrad, source_day
                )
                hourly_path = day_dir / "microbase_hourly.nc"
                paired_path = day_dir / "observed_atmosphere_paired.nc"
                hourly.to_netcdf(hourly_path)
                paired.to_netcdf(paired_path)
            verify_day_outputs(hourly_path, paired_path, source_day)

            audit["sources"] = {
                "microbase": {
                    "path": str(source.resolve()),
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                },
                **shared_sources,
            }
            audit["processing_schema_version"] = PROCESSING_SCHEMA_VERSION
            audit["model_sample_valid_cells"] = int(
                np.count_nonzero(paired.model_sample_valid.values)
            )
            audit["outputs"] = {
                "hourly": {
                    "path": str(hourly_path.resolve()),
                    "size_bytes": hourly_path.stat().st_size,
                    "sha256": _sha256(hourly_path),
                },
                "observed_atmosphere_paired": {
                    "path": str(paired_path.resolve()),
                    "size_bytes": paired_path.stat().st_size,
                    "sha256": _sha256(paired_path),
                },
            }
            (day_dir / "manifest.json").write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n"
            )
            verify_day_outputs(hourly_path, paired_path, source_day)
            if args.delete_raw_after_verify:
                source.unlink()
                print(f"deleted verified raw source {source.name}")
            processed += 1

    verified_days = sorted(
        path.parent.name
        for path in args.output.glob("*/manifest.json")
        if json.loads(path.read_text()).get("processing_schema_version")
        == PROCESSING_SCHEMA_VERSION
    )
    month_manifest = {
        "start": args.start.isoformat(),
        "end_exclusive": args.end.isoformat(),
        "discovered_days": [_file_day(path).isoformat() for path in selected],
        "verified_output_days": verified_days,
        "processed": processed,
        "skipped_verified": skipped,
        "raw_cleanup_enabled": args.delete_raw_after_verify,
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "shared_sources": shared_sources,
    }
    (args.output / "month_manifest.json").write_text(
        json.dumps(month_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"done: {processed} processed, {skipped} already verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
