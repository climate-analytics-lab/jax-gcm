"""Inventory local ARMBE payloads and classify experiment outputs.

This audit reads filenames, file sizes, archive-order ``file_list.txt`` files,
and optional catalog JSON. It does not open NetCDF payloads. Duplicate payloads
are only called byte-identical when ``--hash-duplicates`` is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


PAYLOAD_SUFFIXES = {".cdf", ".nc"}
ARM_FILENAME = re.compile(
    r"^(?P<datastream>[A-Za-z0-9]+\.[a-z0-9]+)\."
    r"(?P<date>\d{8})\.(?P<time>\d{6})\.(?:cdf|nc)$"
)


def parse_arm_filename(path: Path) -> dict[str, str] | None:
    """Parse the datastream and timestamp encoded in an ARM filename."""
    match = ARM_FILENAME.match(path.name)
    if match is None:
        return None
    datastream = match.group("datastream")
    site_match = re.match(r"^(?P<site>[a-z]{3})(?P<product>.+?)(?P<facility>[A-Z]\d)\.", datastream)
    if site_match is None:
        return None
    return {
        "datastream": datastream,
        "site": site_match.group("site"),
        "product": site_match.group("product"),
        "facility": site_match.group("facility"),
        "date": match.group("date"),
        "time": match.group("time"),
    }


def payload_files(root: Path) -> list[Path]:
    """Return local NetCDF/CDF payloads below a root."""
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in PAYLOAD_SUFFIXES)


def summarize_collection(root: Path, paths: list[Path]) -> dict:
    """Summarize stream and temporal coverage for one raw-data collection."""
    streams: dict[str, dict] = {}
    unparsed = []
    grouped: dict[str, list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    for path in paths:
        parsed = parse_arm_filename(path)
        if parsed is None:
            unparsed.append(str(path.relative_to(root)))
            continue
        grouped[parsed["datastream"]].append((path, parsed))

    for datastream, entries in sorted(grouped.items()):
        parsed = entries[0][1]
        dates = sorted(item[1]["date"] for item in entries)
        streams[datastream] = {
            "site": parsed["site"],
            "facility": parsed["facility"],
            "product": parsed["product"],
            "files": len(entries),
            "bytes": sum(item[0].stat().st_size for item in entries),
            "first_file_date": dates[0],
            "last_file_date": dates[-1],
            "extensions": sorted({item[0].suffix.lower() for item in entries}),
        }
    return {
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "datastreams": streams,
        "unparsed_payloads": unparsed,
    }


def audit_order(collection_root: Path, paths: list[Path]) -> dict | None:
    """Compare an archive order's payloads with its supplied file list."""
    manifests = list(collection_root.rglob("file_list.txt"))
    if len(manifests) != 1:
        return None
    expected = {
        "/".join(line.strip().split("/")[-2:])
        for line in manifests[0].read_text().splitlines()
        if line.strip()
    }
    present = {
        f"{path.parent.name}/{path.name}"
        for path in paths
    }
    return {
        "manifest": str(manifests[0].relative_to(collection_root)),
        "expected_files": len(expected),
        "present_files": len(present),
        "missing": sorted(expected - present),
        "unexpected": sorted(present - expected),
        "complete": expected == present,
    }


def sha256(path: Path) -> str:
    """Hash one payload without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_groups(data_root: Path, paths: list[Path], hash_duplicates: bool) -> list[dict]:
    """Find repeated datastream filenames across raw-data collections."""
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        parsed = parse_arm_filename(path)
        if parsed is not None:
            grouped[(parsed["datastream"], path.name)].append(path)

    duplicates = []
    for (datastream, filename), matches in sorted(grouped.items()):
        collections = {path.relative_to(data_root).parts[0] for path in matches}
        if len(matches) < 2 or len(collections) < 2:
            continue
        sizes = [path.stat().st_size for path in matches]
        item = {
            "datastream": datastream,
            "filename": filename,
            "paths": [str(path.relative_to(data_root)) for path in matches],
            "same_size": len(set(sizes)) == 1,
            "byte_identical": None,
        }
        if hash_duplicates and item["same_size"]:
            hashes = [sha256(path) for path in matches]
            item["sha256"] = hashes[0] if len(set(hashes)) == 1 else None
            item["byte_identical"] = len(set(hashes)) == 1
        duplicates.append(item)
    return duplicates


def classify_output(path: Path) -> str:
    """Classify a top-level experiment output by its role."""
    name = path.name
    if name.startswith("cache_"):
        return "observational_or_model_cache"
    if name.startswith("symbolic_features_"):
        return "feature_export_and_search"
    if name.startswith("calibration_"):
        return "calibration_run"
    if name.startswith("diagnostic_"):
        return "diagnostic_run"
    if name.startswith("evaluation_"):
        return "evaluation_run"
    if name.startswith(("hindcast_", "real_", "scm_run")):
        return "hindcast_or_scm_intermediate"
    if path.suffix.lower() == ".png":
        return "plot"
    return "other"


def summarize_outputs(outputs_root: Path, exclusions: set[Path] | None = None) -> dict:
    """Summarize top-level processed artifacts without inspecting payloads."""
    exclusions = {path.resolve() for path in exclusions or set()}
    artifacts = []
    for path in sorted(outputs_root.iterdir()):
        files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else [path]
        files = [item for item in files if item.resolve() not in exclusions]
        if not files:
            continue
        artifacts.append(
            {
                "path": path.name,
                "classification": classify_output(path),
                "files": len(files),
                "bytes": sum(item.stat().st_size for item in files),
            }
        )
    return {
        "files": sum(item["files"] for item in artifacts),
        "bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def compare_catalog(catalog_path: Path, local_streams: set[str]) -> dict:
    """Compare local streams with ARMBE-family records in a catalog inventory."""
    catalog = json.loads(catalog_path.read_text())
    records = [
        record
        for record in catalog["datastreams"]
        if record["instrument_code"].lower().startswith("armbe")
    ]
    catalog_streams = {record["name"] for record in records}
    product_codes = {record["instrument_code"] for record in records}
    local_product_codes = {
        parsed["product"]
        for stream in local_streams
        if (parsed := parse_arm_filename(Path(f"{stream}.20000101.000000.nc"))) is not None
    }
    return {
        "catalog_retrieved_at": catalog.get("retrieved_at"),
        "catalog_armbe_datastreams": len(catalog_streams),
        "catalog_armbe_product_codes": sorted(product_codes),
        "local_datastreams_in_catalog": len(local_streams & catalog_streams),
        "local_datastreams_not_in_catalog": sorted(local_streams - catalog_streams),
        "catalog_datastreams_not_local": sorted(catalog_streams - local_streams),
        "local_product_codes": sorted(local_product_codes),
        "all_catalog_armbe_datastreams_local": catalog_streams <= local_streams,
        "all_catalog_armbe_product_codes_local": product_codes <= local_product_codes,
    }


def build_audit(
    data_root: Path,
    outputs_root: Path,
    catalog_path: Path | None = None,
    hash_duplicates: bool = False,
    output_exclusions: set[Path] | None = None,
) -> dict:
    """Build the complete local audit document."""
    collections = {}
    all_payloads = payload_files(data_root)
    for collection_root in sorted(path for path in data_root.iterdir() if path.is_dir()):
        paths = payload_files(collection_root)
        summary = summarize_collection(collection_root, paths)
        order = audit_order(collection_root, paths)
        if order is not None:
            summary["archive_order"] = order
        collections[collection_root.name] = summary

    local_streams = {
        parsed["datastream"]
        for path in all_payloads
        if (parsed := parse_arm_filename(path)) is not None
    }
    audit = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "scope": "Local filesystem metadata; NetCDF contents were not inspected.",
        "raw": {
            "root": str(data_root),
            "files": len(all_payloads),
            "bytes": sum(path.stat().st_size for path in all_payloads),
            "collections": collections,
            "duplicate_filename_groups": duplicate_groups(
                data_root, all_payloads, hash_duplicates
            ),
        },
        "processed": summarize_outputs(outputs_root, output_exclusions),
    }
    if catalog_path is not None:
        audit["catalog_comparison"] = compare_catalog(catalog_path, local_streams)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--outputs-root", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--catalog", type=Path, help="JSON from inventory_arm_datastreams.py")
    parser.add_argument("--hash-duplicates", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(
        args.data_root,
        args.outputs_root,
        args.catalog,
        args.hash_duplicates,
        {args.output},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "raw_files": audit["raw"]["files"],
                "raw_bytes": audit["raw"]["bytes"],
                "processed_files": audit["processed"]["files"],
                "processed_bytes": audit["processed"]["bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
