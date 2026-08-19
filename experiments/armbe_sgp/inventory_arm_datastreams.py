"""Create a metadata-only inventory from ARM Data Discovery's datastream feed.

The public feed lists datastream names and coverage metadata. It neither queries
file inventories nor downloads ARM observations, so an ARM access token is not
needed. The resulting JSON can be filtered locally before using ARM Live Data
for authenticated file retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen


DEFAULT_ENDPOINT = "https://adc.arm.gov/arm-armint-api/datastream/entity/list"
DEFAULT_SITES_ENDPOINT = "https://metadata-api.svcs.arm.gov/sites"


def fetch_catalog(endpoint: str) -> list[dict]:
    """Fetch the public Data Discovery datastream catalog."""
    with urlopen(endpoint, timeout=120) as response:  # noqa: S310, public fixed default endpoint
        records = json.load(response)
    if not isinstance(records, list):
        raise ValueError("ARM catalog response is not a JSON array.")
    return records


def fetch_sites(endpoint: str) -> dict[str, dict]:
    """Fetch the public ARM site names used to preserve deployment provenance."""
    with urlopen(endpoint, timeout=120) as response:  # noqa: S310, public fixed default endpoint
        records = json.load(response)
    if not isinstance(records, list):
        raise ValueError("ARM site response is not a JSON array.")
    return {record["site_code"].lower(): record for record in records}


def deployment_type(site_name: str | None) -> str:
    """Classify only explicit mobile/campaign labels, retaining uncertainty otherwise."""
    site_name = site_name or ""
    if "Off-Site Campaign" in site_name:
        return "off_site_campaign"
    if "Mobile Facility" in site_name:
        return "mobile_facility"
    return "fixed_or_other"


def add_site_provenance(records: list[dict], sites: dict[str, dict]) -> None:
    """Attach source site names and conservative deployment categories in place."""
    for record in records:
        site = sites.get(record["siteCode"].lower(), {})
        record["siteName"] = site.get("site_name")
        record["siteDeployment"] = deployment_type(record["siteName"])


def iso_time(milliseconds: int | None) -> str | None:
    """Convert ARM's epoch-millisecond coverage field to an ISO timestamp."""
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=UTC).isoformat()


def support_tier(variants: list[dict], anchor_sites: set[str]) -> tuple[str, list[str]]:
    """Classify support relative to durable reference observatories."""
    sites = {item["siteCode"].lower() for item in variants}
    anchors = sorted(sites & anchor_sites)
    if anchors:
        return "anchor_supported", anchors
    deployments = {item["siteDeployment"] for item in variants}
    if deployments <= {"mobile_facility", "off_site_campaign"}:
        return "mobile_or_campaign_only", []
    return "other_fixed_support", []


def summarize(records: list[dict], anchor_sites: set[str]) -> list[dict]:
    """Group datastream variants by ARM's product/instrument code."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["instrumentCode"]].append(record)
    summaries = []
    for code, variants in sorted(grouped.items()):
        tier, anchors = support_tier(variants, anchor_sites)
        summaries.append(
            {
                "instrument_code": code,
                "datastream_count": len(variants),
                "datastream_names": sorted(item["name"] for item in variants),
                "instrument_classes": sorted(
                    {item.get("instrumentClassCode") or "unknown" for item in variants}
                ),
                "sites": sorted({item["siteCode"] for item in variants}),
                "deployment_types": sorted({item["siteDeployment"] for item in variants}),
                "data_levels": sorted({item["dataLevelCode"] for item in variants}),
                "support_tier": tier,
                "anchor_sites": anchors,
            }
        )
    return summaries


def summarize_classes(records: list[dict]) -> list[dict]:
    """Group products by ARM's compact instrument-class taxonomy."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record.get("instrumentClassCode") or "unknown"].append(record)
    summaries = []
    for code, variants in sorted(grouped.items()):
        instrument_codes = sorted({item["instrumentCode"] for item in variants})
        summaries.append(
            {
                "instrument_class_code": code,
                "datastream_count": len(variants),
                "instrument_code_count": len(instrument_codes),
                "instrument_codes_preview": ";".join(instrument_codes[:20]),
                "instrument_codes_truncated": len(instrument_codes) > 20,
                "sites": ";".join(sorted({item["siteCode"] for item in variants})),
                "deployment_types": ";".join(sorted({item["siteDeployment"] for item in variants})),
                "data_levels": ";".join(sorted({item["dataLevelCode"] for item in variants})),
            }
        )
    return summaries


def write_class_csv(path: Path, classes: list[dict]) -> None:
    """Write a spreadsheet-friendly, one-row-per-instrument-class catalog."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "instrument_class_code",
                "datastream_count",
                "instrument_code_count",
                "instrument_codes_preview",
                "instrument_codes_truncated",
                "sites",
                "deployment_types",
                "data_levels",
            ),
        )
        writer.writeheader()
        writer.writerows(classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--sites-endpoint", default=DEFAULT_SITES_ENDPOINT)
    parser.add_argument(
        "--anchor-site",
        action="append",
        default=None,
        help="Durable reference site, repeatable (defaults to SGP, NSA, ENA)",
    )
    parser.add_argument("--site", action="append", help="Keep only this site code, repeatable")
    parser.add_argument("--facility", action="append", help="Keep only this facility code, repeatable")
    parser.add_argument("--available-only", action="store_true", help="Exclude catalog records without data")
    parser.add_argument("--visible-only", action="store_true", help="Exclude hidden catalog records")
    parser.add_argument("--output", type=Path, required=True, help="JSON inventory path")
    parser.add_argument(
        "--class-summary-output",
        type=Path,
        help="Optional CSV with one row per ARM instrument class",
    )
    args = parser.parse_args()

    records = fetch_catalog(args.endpoint)
    add_site_provenance(records, fetch_sites(args.sites_endpoint))
    anchor_sites = {site.lower() for site in args.anchor_site or ["sgp", "nsa", "ena"]}
    if args.site:
        sites = {site.lower() for site in args.site}
        records = [record for record in records if record["siteCode"].lower() in sites]
    if args.facility:
        facilities = {facility.upper() for facility in args.facility}
        records = [record for record in records if record["facilityCode"].upper() in facilities]
    if args.available_only:
        records = [record for record in records if record["dataAvailable"] == "Y"]
    if args.visible_only:
        records = [record for record in records if record["visible"] == "Y"]

    inventory = {
        "endpoint": args.endpoint,
        "retrieved_at": datetime.now(tz=UTC).isoformat(),
        "filters": {
            "sites": args.site or [],
            "facilities": args.facility or [],
            "available_only": args.available_only,
            "visible_only": args.visible_only,
            "anchor_sites": sorted(anchor_sites),
        },
        "counts": {
            "datastreams": len(records),
            "instrument_codes": len({record["instrumentCode"] for record in records}),
            "instrument_classes": len(
                {record.get("instrumentClassCode") or "unknown" for record in records}
            ),
        },
        "instrument_classes": summarize_classes(records),
        "products": summarize(records, anchor_sites),
        "datastreams": [
            {
                "name": record["name"],
                "instrument_code": record["instrumentCode"],
                "instrument_class_code": record.get("instrumentClassCode") or "unknown",
                "site": record["siteCode"],
                "site_name": record["siteName"],
                "site_deployment": record["siteDeployment"],
                "facility": record["facilityCode"],
                "data_level": record["dataLevelCode"],
                "source_class_code": record["sourceClassCode"],
                "visible": record["visible"] == "Y",
                "data_available": record["dataAvailable"] == "Y",
                "retired": record["retired"] == "Y",
                "pre_release": record["preRelease"],
                "start_time": iso_time(record.get("startDate")),
                "end_time": iso_time(record.get("endDate")),
            }
            for record in sorted(records, key=lambda item: item["name"])
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    if args.class_summary_output:
        write_class_csv(args.class_summary_output, inventory["instrument_classes"])
    print(json.dumps(inventory["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
