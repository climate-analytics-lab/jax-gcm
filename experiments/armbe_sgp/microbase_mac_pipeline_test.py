"""Tests for the restartable Mac-local MICROBASE pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from microbase_mac_pipeline import (
    DEFAULT_COMPARE_RTOL,
    Ledger,
    cleanup_month,
    compare_days,
    cookie_config,
    package_month,
    reduce_day,
    remote_publish,
    safe_member_name,
    sha256,
)


DAY = "2011-06-24"
MONTH = "2011-06"


def write_day(root: Path, source_hash: str) -> Path:
    """Create a minimal schema-valid reduced day with a complete manifest."""
    day_dir = root / DAY
    day_dir.mkdir(parents=True)
    time = np.datetime64(DAY) + np.arange(24) * np.timedelta64(1, "h")
    height = np.arange(596, dtype=float)
    hourly = xr.Dataset(
        {
            "armbe_cloud_fraction": (("time", "height"), np.zeros((24, 596))),
            "liquid_water_concentration": (("time", "height"), np.zeros((24, 596))),
            "ice_water_concentration": (("time", "height"), np.zeros((24, 596))),
        },
        coords={"time": time, "height": height},
    )
    paired = hourly.isel(time=[5, 11, 17, 23]).copy()
    pressure = np.broadcast_to(np.linspace(100000.0, 10000.0, 596), (4, 596)).copy()
    for name, value in {
        "temperature": np.full((4, 596), 280.0),
        "relative_humidity": np.full((4, 596), 0.5),
        "dewpoint": np.full((4, 596), 270.0),
        "pressure": pressure,
        "air_density": pressure / (287.04 * 280.0),
        "specific_humidity": np.full((4, 596), 0.005),
        "qc": np.zeros((4, 596)),
        "qi": np.zeros((4, 596)),
    }.items():
        paired[name] = (("time", "height"), value)
    paired["atmospheric_valid"] = (("time", "height"), np.ones((4, 596), dtype=bool))
    paired["model_sample_valid"] = (("time", "height"), np.ones((4, 596), dtype=bool))
    hourly_path = day_dir / "microbase_hourly.nc"
    paired_path = day_dir / "observed_atmosphere_paired.nc"
    hourly.to_netcdf(hourly_path)
    paired.to_netcdf(paired_path)
    manifest = {
        "day": DAY,
        "processing_schema_version": 4,
        "sources": {"microbase": {"sha256": source_hash}},
        "outputs": {
            "hourly": {
                "path": hourly_path.name,
                "size_bytes": hourly_path.stat().st_size,
                "sha256": sha256(hourly_path),
            },
            "observed_atmosphere_paired": {
                "path": paired_path.name,
                "size_bytes": paired_path.stat().st_size,
                "sha256": sha256(paired_path),
            },
        },
    }
    (day_dir / "manifest.json").write_text(json.dumps(manifest))
    return day_dir


def prepare_ledger(tmp_path: Path) -> tuple[dict, Ledger, Path]:
    workspace = tmp_path / "workspace"
    config = {"workspace": workspace}
    ledger = Ledger(workspace / "pipeline.sqlite3")
    filename = "sgpmicrobaseC1.c1.20110624.000000.nc"
    raw = workspace / "raw" / "2011" / MONTH / filename
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"raw fixture")
    raw_hash = sha256(raw)
    ledger.ensure_source(DAY, filename, "https://example", MONTH)
    ledger.set_scope(MONTH, date.fromisoformat(DAY), date.fromisoformat(DAY) + timedelta(days=1), True)
    day_dir = write_day(workspace / "reduced", raw_hash)
    ledger.transition(DAY, "downloaded", raw_path=str(raw), raw_size=raw.stat().st_size,
                      raw_sha256=raw_hash)
    ledger.transition(DAY, "processing")
    ledger.transition(DAY, "processed_verified", output_path=str(day_dir),
                      day_manifest_sha256=sha256(day_dir / "manifest.json"))
    return config, ledger, raw


def test_science_import_does_not_load_jax_or_dinosaur():
    code = (
        "import sys; import collocate_microbase_pilot; "
        "assert 'jax' not in sys.modules; assert 'dinosaur' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parent)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_ledger_records_transitions(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.ensure_source(DAY, "file.nc", "https://example/file.nc", MONTH)
    ledger.transition(DAY, "downloading", attempts=1)
    ledger.transition(DAY, "downloaded", raw_size=123)
    assert ledger.source(DAY)["state"] == "downloaded"
    assert ledger.source(DAY)["attempts"] == 1
    assert ledger.connection.execute("SELECT COUNT(*) FROM event").fetchone()[0] == 2
    with pytest.raises(ValueError, match="Invalid state"):
        ledger.transition(DAY, "not-a-state")
    ledger.close()


def test_archive_publish_and_receipt_gated_cleanup(tmp_path: Path):
    config, ledger, raw = prepare_ledger(tmp_path)
    archive = package_month(config, ledger, MONTH)
    sidecar = archive.with_suffix(".tar.sha256.json")
    uploaded = archive.with_suffix(".tar.part")
    shutil.copyfile(archive, uploaded)
    publish_root = tmp_path / "published"
    receipt = tmp_path / "receipt.json"
    remote_publish(uploaded, sidecar, publish_root, receipt)
    accepted = json.loads(receipt.read_text())
    assert accepted["archive_sha256"] == sha256(archive)
    published = publish_root / "schema4" / "sgp" / "C1" / "2011" / MONTH
    assert (published / DAY / "microbase_hourly.nc").is_file()

    with pytest.raises(RuntimeError, match="server-verified"):
        cleanup_month(config, ledger, MONTH)
    local_receipt = config["workspace"] / "receipt.json"
    shutil.copyfile(receipt, local_receipt)
    ledger.update_batch(MONTH, "server_verified", receipt_path=str(local_receipt),
                        receipt_sha256=sha256(local_receipt))
    ledger.transition(DAY, "uploading")
    ledger.transition(DAY, "server_verified")
    assert cleanup_month(config, ledger, MONTH) == 1
    assert not raw.exists()
    assert ledger.source(DAY)["state"] == "raw_deleted"
    ledger.close()


def test_partial_range_cannot_use_canonical_month_package(tmp_path: Path):
    config, ledger, _ = prepare_ledger(tmp_path)
    ledger.connection.execute(
        "UPDATE batch SET canonical_month = 0 WHERE month = ?", (MONTH,)
    )
    with pytest.raises(RuntimeError, match="Partial/benchmark"):
        package_month(config, ledger, MONTH)
    ledger.close()


def test_reduction_rejects_raw_changed_after_download(tmp_path: Path):
    config, ledger, raw = prepare_ledger(tmp_path)
    ledger.connection.execute(
        "UPDATE source SET state = 'downloaded', output_path = NULL WHERE day = ?", (DAY,)
    )
    raw.write_bytes(b"changed raw fixture")
    with pytest.raises(ValueError, match="changed after download"):
        reduce_day(config, ledger.source(DAY), ledger, tmp_path / "atm.nc", tmp_path / "cloud.nc")
    ledger.close()


def test_cleanup_rejects_tampered_receipt(tmp_path: Path):
    config, ledger, _ = prepare_ledger(tmp_path)
    archive = package_month(config, ledger, MONTH)
    receipt = config["workspace"] / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "receipt_schema_version": 1,
                "site": "sgp",
                "facility": "C1",
                "month": MONTH,
                "archive_sha256": sha256(archive),
                "archive_size_bytes": archive.stat().st_size,
            }
        )
    )
    ledger.update_batch(MONTH, "server_verified", receipt_path=str(receipt),
                        receipt_sha256=sha256(receipt))
    ledger.transition(DAY, "uploading")
    ledger.transition(DAY, "server_verified")
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt changed"):
        cleanup_month(config, ledger, MONTH)
    ledger.close()


def test_cookie_config_rejects_newlines():
    with pytest.raises(ValueError, match="control character"):
        with cookie_config("secret\noutput = bad"):
            pass


def test_compare_days_detects_changed_science(tmp_path: Path):
    source_hash = "a" * 64
    reference = write_day(tmp_path / "reference", source_hash)
    candidate = tmp_path / "candidate" / DAY
    shutil.copytree(reference, candidate)
    compare_days(candidate, reference, rtol=1e-12, atol=1e-12)
    assert DEFAULT_COMPARE_RTOL == 1e-6
    with xr.open_dataset(candidate / "microbase_hourly.nc") as source:
        changed = source.load()
    changed["armbe_cloud_fraction"][0, 0] = 0.5
    changed.to_netcdf(candidate / "microbase_hourly.nc")
    with pytest.raises(AssertionError):
        compare_days(candidate, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "valid"),
    [("sgp/C1/file.nc", True), ("/etc/passwd", False), ("sgp/../secret", False)],
)
def test_safe_member_name(name: str, valid: bool):
    assert safe_member_name(name) is valid
