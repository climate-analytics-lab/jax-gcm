"""Tests for the local ARMBE filesystem audit."""

import json
from pathlib import Path

import audit_local_armbe


def write_payload(path: Path, content: bytes = b"payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_parse_arm_filename() -> None:
    parsed = audit_local_armbe.parse_arm_filename(
        Path("enaarmbeatmhiresC1.c1.20240101.000500.nc")
    )
    assert parsed == {
        "datastream": "enaarmbeatmhiresC1.c1",
        "site": "ena",
        "product": "armbeatmhires",
        "facility": "C1",
        "date": "20240101",
        "time": "000500",
    }


def test_build_audit_checks_orders_duplicates_and_catalog(tmp_path: Path) -> None:
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    stream = "sgparmbeatmC1.c1"
    filename = f"{stream}.20230101.003000.nc"
    order = data / "order-1" / "archive" / "1"
    write_payload(order / stream / filename)
    (order / "file_list.txt").write_text(f"/archive/1/{stream}/{filename}\n")
    write_payload(data / "copy" / stream / filename)
    write_payload(outputs / "cache_example" / "samples.nc")

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "retrieved_at": "2026-08-13T00:00:00+00:00",
                "datastreams": [
                    {"name": stream, "instrument_code": "armbeatm"},
                    {"name": "sgparmbelandC1.c1", "instrument_code": "armbeland"},
                    {"name": "sgpqcradC1.c1", "instrument_code": "qcrad"},
                ],
            }
        )
    )

    audit = audit_local_armbe.build_audit(data, outputs, catalog, hash_duplicates=True)

    assert audit["raw"]["collections"]["order-1"]["archive_order"]["complete"]
    duplicate = audit["raw"]["duplicate_filename_groups"][0]
    assert duplicate["byte_identical"] is True
    assert audit["processed"]["artifacts"][0]["classification"] == "observational_or_model_cache"
    comparison = audit["catalog_comparison"]
    assert comparison["local_datastreams_in_catalog"] == 1
    assert not comparison["all_catalog_armbe_product_codes_local"]


def test_summarize_outputs_excludes_report_itself(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    write_payload(outputs / "cache_example" / "samples.nc")
    report = outputs / "audit.json"
    report.write_text("old report")

    summary = audit_local_armbe.summarize_outputs(outputs, {report})

    assert summary["files"] == 1
    assert [item["path"] for item in summary["artifacts"]] == ["cache_example"]
