"""Restartable SGP MICROBASE download, reduction, and publication pipeline."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import tomllib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
import xarray as xr

from collocate_microbase_pilot import build_hourly_cache
from microbase_physics import constants_record
from process_microbase_month import PROCESSING_SCHEMA_VERSION, verify_day_outputs


PIPELINE_VERSION = 1
DEFAULT_COMPARE_RTOL = 1e-6
FILENAME = re.compile(r"^sgpmicrobaseC1\.c1\.(\d{8})\.000000\.nc$")
VALID_STATES = {
    "discovered",
    "downloading",
    "downloaded",
    "processing",
    "processed_verified",
    "packaged",
    "uploading",
    "server_verified",
    "raw_deleted",
    "failed_retryable",
    "failed_manual_review",
}
TRANSITIONS = {
    "discovered": {"downloading", "downloaded", "failed_retryable", "failed_manual_review"},
    "downloading": {"downloaded", "failed_retryable", "failed_manual_review"},
    "downloaded": {"processing", "failed_retryable", "failed_manual_review"},
    "processing": {"processed_verified", "failed_retryable", "failed_manual_review"},
    "processed_verified": {"packaged", "failed_retryable", "failed_manual_review"},
    "packaged": {"uploading", "failed_retryable", "failed_manual_review"},
    "uploading": {"server_verified", "failed_retryable", "failed_manual_review"},
    "failed_retryable": {"downloading", "downloaded", "processing", "uploading", "failed_manual_review"},
    "server_verified": {"raw_deleted", "failed_manual_review"},
    "raw_deleted": set(),
    "failed_manual_review": set(),
}


def utc_now() -> str:
    """Return a stable UTC timestamp for manifests and ledger events."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON and atomically publish it at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def resolve_path(value: str, config_path: Path) -> Path:
    """Resolve a configured path relative to its TOML file."""
    expanded = Path(value).expanduser()
    return expanded if expanded.is_absolute() else (config_path.parent / expanded).resolve()


def load_config(path: Path) -> dict[str, Any]:
    """Load TOML and resolve local filesystem paths."""
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_path"] = path.resolve()
    config["workspace"] = resolve_path(config["workspace"], path)
    remote = config.get("remote", {})
    if "identity" in remote:
        remote["identity"] = str(resolve_path(remote["identity"], path))
    for companion in config.get("companions", {}).values():
        companion["atmosphere"] = str(resolve_path(companion["atmosphere"], path))
        companion["cldrad"] = str(resolve_path(companion["cldrad"], path))
    return config


class Ledger:
    """SQLite-backed state and event ledger."""

    def __init__(self, path: Path):
        """Open or initialize the ledger at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS source (
                id INTEGER PRIMARY KEY,
                day TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                url TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                raw_path TEXT,
                raw_size INTEGER,
                raw_sha256 TEXT,
                output_path TEXT,
                day_manifest_sha256 TEXT,
                batch_month TEXT NOT NULL,
                last_error TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batch (
                month TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                archive_path TEXT,
                archive_size INTEGER,
                archive_sha256 TEXT,
                receipt_path TEXT,
                receipt_sha256 TEXT,
                canonical_month INTEGER NOT NULL DEFAULT 0,
                scope_start TEXT,
                scope_end TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event (
                id INTEGER PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                day TEXT,
                month TEXT,
                old_state TEXT,
                new_state TEXT,
                message TEXT NOT NULL
            );
            """
        )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(batch)")
        }
        for name, declaration in {
            "canonical_month": "INTEGER NOT NULL DEFAULT 0",
            "scope_start": "TEXT",
            "scope_end": "TEXT",
        }.items():
            if name not in columns:
                self.connection.execute(f"ALTER TABLE batch ADD COLUMN {name} {declaration}")

    def close(self) -> None:
        self.connection.close()

    def ensure_source(self, day: str, filename: str, url: str, month: str) -> None:
        now = utc_now()
        with self.connection:
            self.connection.execute(
                """INSERT OR IGNORE INTO source
                   (day, filename, url, state, batch_month, updated_at)
                   VALUES (?, ?, ?, 'discovered', ?, ?)""",
                (day, filename, url, month, now),
            )
            self.connection.execute(
                """INSERT OR IGNORE INTO batch (month, state, updated_at)
                   VALUES (?, 'open', ?)""",
                (month, now),
            )
        existing = self.source(day)
        identity = (existing["filename"], existing["url"], existing["batch_month"])
        if identity != (filename, url, month):
            raise ValueError(f"Ledger identity differs from configuration for {day}")

    def set_scope(self, month: str, start: date, end: date, canonical: bool) -> None:
        """Record whether a batch is eligible for canonical monthly publication."""
        batch = self.batch(month)
        existing = (batch["scope_start"], batch["scope_end"])
        requested = (start.isoformat(), end.isoformat())
        if existing != (None, None) and existing != requested:
            raise ValueError(f"Ledger scope {existing} differs from requested scope {requested}")
        with self.connection:
            self.connection.execute(
                """UPDATE batch SET canonical_month = ?, scope_start = ?, scope_end = ?,
                   updated_at = ? WHERE month = ?""",
                (int(canonical), *requested, utc_now(), month),
            )

    def source(self, day: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM source WHERE day = ?", (day,)
        ).fetchone()
        if row is None:
            raise KeyError(day)
        return row

    def sources(self, month: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM source WHERE batch_month = ? ORDER BY day", (month,)
            )
        )

    def batch(self, month: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM batch WHERE month = ?", (month,)
        ).fetchone()
        if row is None:
            raise KeyError(month)
        return row

    def transition(self, day: str, new_state: str, **fields: Any) -> None:
        if new_state not in VALID_STATES:
            raise ValueError(f"Invalid state {new_state}")
        current = self.source(day)
        if new_state not in TRANSITIONS[current["state"]]:
            raise ValueError(f"Illegal transition {current['state']} -> {new_state}")
        assignments = ["state = ?", "updated_at = ?"]
        values: list[Any] = [new_state, utc_now()]
        for key, value in fields.items():
            if key not in {
                "raw_path", "raw_size", "raw_sha256", "output_path",
                "day_manifest_sha256", "last_error", "attempts",
            }:
                raise ValueError(f"Invalid source field {key}")
            assignments.append(f"{key} = ?")
            values.append(value)
        values.append(day)
        with self.connection:
            self.connection.execute(
                f"UPDATE source SET {', '.join(assignments)} WHERE day = ?", values
            )
            self.connection.execute(
                """INSERT INTO event
                   (occurred_at, day, month, old_state, new_state, message)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (utc_now(), day, current["batch_month"], current["state"], new_state,
                 fields.get("last_error") or new_state),
            )

    def update_batch(self, month: str, state: str, **fields: Any) -> None:
        current_state = self.batch(month)["state"]
        allowed = {
            "archive_path", "archive_size", "archive_sha256",
            "receipt_path", "receipt_sha256",
        }
        assignments = ["state = ?", "updated_at = ?"]
        values: list[Any] = [state, utc_now()]
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Invalid batch field {key}")
            assignments.append(f"{key} = ?")
            values.append(value)
        values.append(month)
        with self.connection:
            self.connection.execute(
                f"UPDATE batch SET {', '.join(assignments)} WHERE month = ?", values
            )
            self.connection.execute(
                """INSERT INTO event
                   (occurred_at, month, old_state, new_state, message)
                   VALUES (?, ?, ?, ?, ?)""",
                (utc_now(), month, current_state, state, state),
            )


@contextlib.contextmanager
def ledger_for(config: dict[str, Any]) -> Iterator[Ledger]:
    ledger = Ledger(config["workspace"] / "pipeline.sqlite3")
    try:
        yield ledger
    finally:
        ledger.close()


def month_days(month: str) -> list[date]:
    """Return every calendar day in ``YYYY-MM``."""
    start = date.fromisoformat(f"{month}-01")
    next_month = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return [start + timedelta(days=i) for i in range((next_month - start).days)]


def raw_filename(day: date) -> str:
    return f"sgpmicrobaseC1.c1.{day:%Y%m%d}.000000.nc"


def companion_paths(config: dict[str, Any], year: int) -> tuple[Path, Path]:
    values = config.get("companions", {}).get(str(year))
    if values is None:
        raise ValueError(f"No companion paths configured for {year}")
    atmosphere = Path(values["atmosphere"])
    cldrad = Path(values["cldrad"])
    for path in (atmosphere, cldrad):
        if not path.is_file():
            raise FileNotFoundError(path)
    return atmosphere, cldrad


def validate_raw(path: Path, expected_day: date, minimum_bytes: int) -> None:
    """Reject login pages, truncated files, and incompatible SGP products."""
    if path.stat().st_size < minimum_bytes:
        raise ValueError(f"Raw file is too small: {path.stat().st_size} bytes")
    if FILENAME.fullmatch(path.name.replace(".part", "")) is None:
        raise ValueError(f"Unexpected raw filename {path.name}")
    with xr.open_dataset(path, engine="netcdf4") as source:
        if source.sizes.get("height") != 596:
            raise ValueError(f"Expected 596 heights, got {source.sizes.get('height')}")
        required = {
            "retrieval_flag", "liquid_water_content", "ice_water_content",
            "qc_liquid_water_content", "qc_ice_water_content", "precip_flag",
        }
        missing = sorted(required - set(source.variables))
        if missing:
            raise ValueError(f"Missing MICROBASE variables: {missing}")
        if source.liquid_water_content.attrs.get("units") != "g m-3":
            raise ValueError("Unexpected liquid-water units")
        values = source.time.values
        if values.size == 0 or str(values[0])[:10] != expected_day.isoformat():
            raise ValueError("MICROBASE timestamps do not match requested day")


def cookie_config(cookie: str) -> Iterator[Path]:
    """Yield a domain-scoped cookie jar without exposing the token in argv."""
    @contextlib.contextmanager
    def manager() -> Iterator[Path]:
        if any(character in cookie for character in ("\r", "\n", "\t", "\0")):
            raise ValueError("THREDDS cookie contains a forbidden control character")
        value = cookie.removeprefix("_oauth2_proxy=")
        if not value:
            raise ValueError("THREDDS cookie is empty")
        fd, name = tempfile.mkstemp(prefix="microbase-curl-", text=True)
        path = Path(name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write("# Netscape HTTP Cookie File\n")
                stream.write(
                    "thredds-ui.svcs.arm.gov\tFALSE\t/\tTRUE\t0\t"
                    f"_oauth2_proxy\t{value}\n"
                )
            yield path
        finally:
            path.unlink(missing_ok=True)
    return manager()


def download_source(config: dict[str, Any], row: sqlite3.Row, ledger: Ledger) -> Path:
    """Resume and validate one authenticated THREDDS download."""
    day = date.fromisoformat(row["day"])
    raw_dir = config["workspace"] / "raw" / f"{day.year}" / f"{day:%Y-%m}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = raw_dir / row["filename"]
    partial = final.with_suffix(final.suffix + ".part")
    minimum = int(config["download"].get("expected_minimum_bytes", 600_000_000))
    free_required = int(config["download"].get("minimum_free_bytes", 0))
    if shutil.disk_usage(config["workspace"]).free < free_required:
        raise RuntimeError(f"Free space is below the configured {free_required} byte reserve")
    staged_limit = int(config["download"].get("maximum_staged_raw_bytes", 0))
    staged_bytes = sum(
        path.stat().st_size
        for path in (config["workspace"] / "raw").glob("**/*")
        if path.name.endswith((".nc", ".nc.part"))
        if path.is_file()
    )
    if staged_limit and staged_bytes + minimum > staged_limit:
        raise RuntimeError("Raw staging limit would be exceeded by another download")
    if final.exists():
        validate_raw(final, day, minimum)
        digest = sha256(final)
        ledger.transition(row["day"], "downloaded", raw_path=str(final),
                          raw_size=final.stat().st_size, raw_sha256=digest)
        return final
    cookie = os.environ.get(config["download"].get("cookie_env", "THREDDS_COOKIE"))
    if not cookie:
        raise RuntimeError("THREDDS_COOKIE is not set")
    ledger.transition(row["day"], "downloading", attempts=row["attempts"] + 1)
    with cookie_config(cookie) as cookie_jar:
        command = [
            "curl", "--cookie", str(cookie_jar), "--fail", "--location",
            "--proto-redir", "=https",
            "--continue-at", "-", "--retry", "10", "--retry-all-errors",
            "--retry-delay", "10", "--connect-timeout", "30", "--speed-limit",
            "1024", "--speed-time", "120", "--output", str(partial),
            "--write-out", "%{url_effective}", row["url"],
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    if result.stdout.strip() != row["url"]:
        raise RuntimeError("THREDDS redirected away from the requested file")
    validate_raw(partial, day, minimum)
    partial.replace(final)
    digest = sha256(final)
    ledger.transition(row["day"], "downloaded", raw_path=str(final),
                      raw_size=final.stat().st_size, raw_sha256=digest,
                      last_error=None)
    return final


def output_day_dir(config: dict[str, Any], day: date) -> Path:
    return (
        config["workspace"] / "reduced" / "schema4" / "sgp" / "C1"
        / str(day.year) / f"{day:%Y-%m}" / day.isoformat()
    )


def verify_manifest(
    day_dir: Path,
    expected_source_hash: str | None = None,
    expected_day: str | None = None,
) -> dict[str, Any]:
    """Verify a complete day and all hashes recorded by its manifest."""
    manifest_path = day_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("processing_schema_version") != PROCESSING_SCHEMA_VERSION:
        raise ValueError("Unexpected processing schema")
    verify_day_outputs(
        day_dir / "microbase_hourly.nc",
        day_dir / "observed_atmosphere_paired.nc",
        expected_day or day_dir.name,
    )
    if expected_source_hash and manifest["sources"]["microbase"]["sha256"] != expected_source_hash:
        raise ValueError("Source checksum differs from the day manifest")
    for record in manifest["outputs"].values():
        if record["path"] not in {"microbase_hourly.nc", "observed_atmosphere_paired.nc"}:
            raise ValueError(f"Unexpected output path {record['path']}")
        path = day_dir / record["path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Output checksum mismatch: {path}")
    return manifest


def reduce_day(
    config: dict[str, Any], row: sqlite3.Row, ledger: Ledger,
    atmosphere: Path, cldrad: Path,
) -> Path:
    """Reduce one raw day into an atomically published verified directory."""
    day = date.fromisoformat(row["day"])
    destination = output_day_dir(config, day)
    if destination.exists():
        verify_manifest(destination, row["raw_sha256"])
        ledger.transition(row["day"], "processed_verified", output_path=str(destination),
                          day_manifest_sha256=sha256(destination / "manifest.json"))
        return destination
    ledger.transition(row["day"], "processing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        source = Path(row["raw_path"])
        if source.stat().st_size != row["raw_size"] or sha256(source) != row["raw_sha256"]:
            raise ValueError(f"Raw source changed after download verification: {source}")
        with (
            xr.open_dataset(source, engine="netcdf4") as microbase,
            xr.open_dataset(atmosphere, engine="netcdf4") as atmosphere_ds,
            xr.open_dataset(cldrad, engine="netcdf4") as cldrad_ds,
        ):
            hourly, paired, audit = build_hourly_cache(
                microbase, atmosphere_ds, cldrad_ds, row["day"]
            )
            hourly_path = temporary / "microbase_hourly.nc"
            paired_path = temporary / "observed_atmosphere_paired.nc"
            hourly.to_netcdf(hourly_path, engine="netcdf4")
            paired.to_netcdf(paired_path, engine="netcdf4")
        verify_day_outputs(hourly_path, paired_path, row["day"])
        audit.update(
            processing_schema_version=PROCESSING_SCHEMA_VERSION,
            pipeline_version=PIPELINE_VERSION,
            physical_constants=constants_record(),
            model_sample_valid_cells=int(np.count_nonzero(paired.model_sample_valid.values)),
            sources={
                "microbase": {"path": source.name, "size_bytes": source.stat().st_size,
                              "sha256": row["raw_sha256"]},
                "armbeatm": {"path": atmosphere.name, "size_bytes": atmosphere.stat().st_size,
                             "sha256": sha256(atmosphere)},
                "armbecldrad": {"path": cldrad.name, "size_bytes": cldrad.stat().st_size,
                                "sha256": sha256(cldrad)},
            },
            outputs={
                "hourly": {"path": hourly_path.name, "size_bytes": hourly_path.stat().st_size,
                           "sha256": sha256(hourly_path)},
                "observed_atmosphere_paired": {
                    "path": paired_path.name, "size_bytes": paired_path.stat().st_size,
                    "sha256": sha256(paired_path),
                },
            },
        )
        atomic_json(temporary / "manifest.json", audit)
        verify_manifest(temporary, row["raw_sha256"], row["day"])
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    ledger.transition(row["day"], "processed_verified", output_path=str(destination),
                      day_manifest_sha256=sha256(destination / "manifest.json"),
                      last_error=None)
    return destination


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def package_month(config: dict[str, Any], ledger: Ledger, month: str) -> Path:
    """Build and verify a deterministic uncompressed monthly archive."""
    rows = ledger.sources(month)
    batch = ledger.batch(month)
    if not batch["canonical_month"]:
        raise RuntimeError("Partial/benchmark ranges cannot be packaged as canonical months")
    if not rows or any(row["state"] not in {"processed_verified", "packaged", "server_verified", "raw_deleted"} for row in rows):
        raise RuntimeError("Every discovered source must be processed before packaging")
    package_dir = config["workspace"] / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    archive = package_dir / f"sgp-C1-microbase-{month}-schema4.tar"
    temporary = archive.with_suffix(".tar.part")
    batch_manifest = {
        "batch_schema_version": 1,
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "site": "sgp",
        "facility": "C1",
        "month": month,
        "created_at": utc_now(),
        "physical_constants": constants_record(),
        "days": [],
    }
    files: list[tuple[Path, str]] = []
    for row in rows:
        day_dir = Path(row["output_path"])
        verify_manifest(day_dir, row["raw_sha256"])
        relative_root = f"sgp/C1/{month}/{row['day']}"
        day_record = {"day": row["day"], "manifest_sha256": sha256(day_dir / "manifest.json")}
        batch_manifest["days"].append(day_record)
        for name in ("manifest.json", "microbase_hourly.nc", "observed_atmosphere_paired.nc"):
            files.append((day_dir / name, f"{relative_root}/{name}"))
    manifest_path = package_dir / f".{month}.batch_manifest.json"
    atomic_json(manifest_path, batch_manifest)
    files.append((manifest_path, f"sgp/C1/{month}/batch_manifest.json"))
    with tarfile.open(temporary, "w") as archive_file:
        for source, name in sorted(files, key=lambda item: item[1]):
            info = archive_file.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                archive_file.addfile(info, stream)
    with tarfile.open(temporary, "r") as archive_file:
        names = archive_file.getnames()
        if names != sorted(name for _, name in files) or not all(safe_member_name(name) for name in names):
            raise ValueError("Archive member verification failed")
    temporary.replace(archive)
    sidecar = {
        "archive": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "month": month,
    }
    atomic_json(archive.with_suffix(".tar.sha256.json"), sidecar)
    ledger.update_batch(month, "packaged", archive_path=str(archive),
                        archive_size=archive.stat().st_size, archive_sha256=sidecar["sha256"])
    for row in rows:
        if row["state"] == "processed_verified":
            ledger.transition(row["day"], "packaged")
    manifest_path.unlink()
    return archive


def ssh_base(remote: dict[str, Any]) -> list[str]:
    return ["ssh", "-p", str(remote["port"]), "-i", remote["identity"],
            f"{remote['user']}@{remote['host']}"]


def upload_month(config: dict[str, Any], ledger: Ledger, month: str) -> Path:
    """Upload a package, publish remotely, and verify its acceptance receipt."""
    batch = ledger.batch(month)
    archive = Path(batch["archive_path"])
    sidecar = archive.with_suffix(".tar.sha256.json")
    remote = config["remote"]
    remote_stage = f"{remote['staging_root'].rstrip('/')}/{archive.name}"
    subprocess.run(
        ssh_base(remote) + [shlex.join(["mkdir", "-p", remote["staging_root"]])],
        check=True,
    )
    if sha256(archive) != batch["archive_sha256"]:
        raise ValueError("Packaged archive changed before upload")
    ledger.update_batch(month, "uploading")
    for row in ledger.sources(month):
        if row["state"] == "packaged":
            ledger.transition(row["day"], "uploading")
    rsync_shell = shlex.join(
        ["ssh", "-p", str(remote["port"]), "-i", remote["identity"]]
    )
    destination = f"{remote['user']}@{remote['host']}:{remote_stage}.part"
    if shutil.which("rsync"):
        subprocess.run(
            ["rsync", "--partial", "--append-verify", "-e", rsync_shell,
             str(archive), destination], check=True
        )
    else:
        subprocess.run(
            ["scp", "-P", str(remote["port"]), "-i", remote["identity"],
             str(archive), destination], check=True
        )
    subprocess.run(
        ["scp", "-P", str(remote["port"]), "-i", remote["identity"], str(sidecar),
         f"{remote['user']}@{remote['host']}:{remote_stage}.sha256.json"], check=True
    )
    remote_python = remote.get(
        "python",
        "/data/MOSAIC/tools/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12",
    )
    remote_script = remote.get(
        "pipeline_script",
        "/data/MOSAIC/jax-gcm/experiments/armbe_sgp/microbase_mac_pipeline.py",
    )
    remote_pythonpath = remote.get(
        "pythonpath",
        "/data/MOSAIC/jax-gcm:/data/MOSAIC/.venv/lib/python3.12/site-packages",
    )
    receipt_remote = f"{remote_stage}.receipt.json"
    remote_command = [
        "env", f"PYTHONPATH={remote_pythonpath}", remote_python, remote_script,
        "remote-publish",
        "--archive-part", f"{remote_stage}.part",
        "--sidecar", f"{remote_stage}.sha256.json",
        "--publish-root", remote["publish_root"],
        "--receipt", receipt_remote,
    ]
    subprocess.run(ssh_base(remote) + [shlex.join(remote_command)], check=True)
    receipt = config["workspace"] / "receipts" / f"{archive.name}.receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-P", str(remote["port"]), "-i", remote["identity"],
         f"{remote['user']}@{remote['host']}:{receipt_remote}", str(receipt)], check=True
    )
    accepted = json.loads(receipt.read_text())
    if accepted["archive_sha256"] != batch["archive_sha256"] or accepted["month"] != month:
        raise ValueError("Remote receipt does not match the local package")
    ledger.update_batch(month, "server_verified", receipt_path=str(receipt),
                        receipt_sha256=sha256(receipt))
    for row in ledger.sources(month):
        ledger.transition(row["day"], "server_verified")
    cleanup_remote = shlex.join(
        ["rm", "-f", remote_stage, f"{remote_stage}.sha256.json", receipt_remote]
    )
    subprocess.run(ssh_base(remote) + [cleanup_remote], check=True)
    return receipt


def remote_publish(archive_part: Path, sidecar_path: Path, publish_root: Path, receipt: Path) -> None:
    """Verify, extract, and atomically publish an uploaded monthly archive."""
    sidecar = json.loads(sidecar_path.read_text())
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", sidecar.get("month", "")):
        raise ValueError("Invalid sidecar month")
    if sidecar.get("archive") != archive_part.name.removesuffix(".part"):
        raise ValueError("Sidecar archive name mismatch")
    if archive_part.stat().st_size != sidecar["size_bytes"] or sha256(archive_part) != sidecar["sha256"]:
        raise ValueError("Uploaded archive checksum mismatch")
    month = sidecar["month"]
    final_archive = archive_part.with_suffix("")
    archive_part.replace(final_archive)
    publish = publish_root / "schema4" / "sgp" / "C1" / month[:4] / month
    publish.parent.mkdir(parents=True, exist_ok=True)
    temporary = publish.parent / f".{month}.{os.getpid()}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    try:
        with tarfile.open(final_archive, "r") as source:
            for member in source.getmembers():
                if not safe_member_name(member.name) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member {member.name}")
            source.extractall(temporary, filter="data")
        month_root = temporary / "sgp" / "C1" / month
        batch_manifest_path = month_root / "batch_manifest.json"
        batch_manifest = json.loads(batch_manifest_path.read_text())
        if batch_manifest["month"] != month:
            raise ValueError("Batch manifest month mismatch")
        seen_days: set[str] = set()
        for day in batch_manifest["days"]:
            day_value = day["day"]
            parsed_day = date.fromisoformat(day_value)
            if parsed_day.strftime("%Y-%m") != month or day_value in seen_days:
                raise ValueError(f"Invalid or duplicate batch day {day_value}")
            seen_days.add(day_value)
            day_dir = month_root / day_value
            if sha256(day_dir / "manifest.json") != day["manifest_sha256"]:
                raise ValueError(f"Day manifest mismatch for {day['day']}")
            verify_manifest(day_dir)
        if publish.exists():
            existing = publish / "acceptance_receipt.json"
            if not existing.exists() or json.loads(existing.read_text())["archive_sha256"] != sidecar["sha256"]:
                raise FileExistsError(f"Conflicting published month {publish}")
            atomic_json(receipt, json.loads(existing.read_text()))
            return
        accepted = {
            "receipt_schema_version": 1,
            "accepted_at": utc_now(),
            "site": "sgp",
            "facility": "C1",
            "month": month,
            "archive": sidecar["archive"],
            "archive_size_bytes": sidecar["size_bytes"],
            "archive_sha256": sidecar["sha256"],
            "day_count": len(batch_manifest["days"]),
            "published_path": str(publish),
            "pipeline_version": PIPELINE_VERSION,
            "host": platform.node(),
        }
        atomic_json(month_root / "acceptance_receipt.json", accepted)
        month_root.replace(publish)
        atomic_json(receipt, accepted)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def cleanup_month(config: dict[str, Any], ledger: Ledger, month: str) -> int:
    """Delete exact verified raw paths only after a matching remote receipt."""
    batch = ledger.batch(month)
    if batch["state"] != "server_verified" or not batch["receipt_path"]:
        raise RuntimeError("Cleanup requires a server-verified batch receipt")
    receipt_path = Path(batch["receipt_path"])
    if sha256(receipt_path) != batch["receipt_sha256"]:
        raise ValueError("Acceptance receipt changed after verification")
    receipt = json.loads(receipt_path.read_text())
    expected_receipt = {
        "receipt_schema_version": 1,
        "site": "sgp",
        "facility": "C1",
        "month": month,
        "archive_sha256": batch["archive_sha256"],
        "archive_size_bytes": batch["archive_size"],
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ValueError("Receipt/archive mismatch")
    raw_root = (config["workspace"] / "raw").resolve()
    deleted = 0
    for row in ledger.sources(month):
        if row["state"] == "raw_deleted":
            continue
        if row["state"] != "server_verified":
            raise RuntimeError(f"{row['day']} is not server verified")
        raw = Path(row["raw_path"])
        if not raw.parent.resolve().is_relative_to(raw_root) or raw.name != row["filename"]:
            raise ValueError(f"Refusing to delete raw path outside the staging layout: {raw}")
        if not raw.exists():
            raise FileNotFoundError(f"Verified raw file disappeared before cleanup: {raw}")
        if raw.stat().st_size != row["raw_size"] or sha256(raw) != row["raw_sha256"]:
            raise ValueError(f"Raw file changed before cleanup: {raw}")
        raw.unlink()
        ledger.transition(row["day"], "raw_deleted")
        deleted += 1
    ledger.update_batch(month, "raw_deleted")
    return deleted


def compare_days(candidate: Path, reference: Path, rtol: float, atol: float) -> None:
    """Compare scientific arrays while ignoring platform-specific NetCDF bytes."""
    for name in ("microbase_hourly.nc", "observed_atmosphere_paired.nc"):
        with xr.open_dataset(candidate / name) as actual, xr.open_dataset(reference / name) as expected:
            if set(actual.variables) != set(expected.variables):
                raise AssertionError(f"Variable mismatch in {name}")
            for variable in expected.variables:
                if actual[variable].dims != expected[variable].dims:
                    raise AssertionError(f"Dimension mismatch in {name}:{variable}")
                for attribute in ("units", "method", "interpolation", "description"):
                    if actual[variable].attrs.get(attribute) != expected[variable].attrs.get(attribute):
                        raise AssertionError(f"Attribute mismatch in {name}:{variable}:{attribute}")
                left = actual[variable].values
                right = expected[variable].values
                if left.dtype.kind in "biuOSUMm" or right.dtype.kind in "biuOSUMm":
                    np.testing.assert_array_equal(left, right, err_msg=f"{name}:{variable}")
                else:
                    np.testing.assert_allclose(left, right, rtol=rtol, atol=atol,
                                               equal_nan=True, err_msg=f"{name}:{variable}")


def initialize_month(
    config: dict[str, Any], ledger: Ledger, month: str,
    start: date | None = None, end: date | None = None,
) -> None:
    base_url = config["download"]["base_url"].rstrip("/")
    selected = month_days(month)
    if start is not None:
        selected = [day for day in selected if day >= start]
    if end is not None:
        selected = [day for day in selected if day < end]
    if not selected:
        raise ValueError("The requested date range contains no days in the month")
    for day in selected:
        filename = raw_filename(day)
        ledger.ensure_source(day.isoformat(), filename, f"{base_url}/{filename}", month)
    full_month = selected == month_days(month)
    ledger.set_scope(month, selected[0], selected[-1] + timedelta(days=1), full_month)


def run_month(
    config: dict[str, Any], month: str, stop_after: str,
    start: date | None = None, end: date | None = None,
) -> None:
    """Run one SGP calendar month through the requested durable stage."""
    with ledger_for(config) as ledger:
        initialize_month(config, ledger, month, start, end)
        batch_state = ledger.batch(month)["state"]
        if batch_state == "raw_deleted":
            return
        if batch_state == "server_verified":
            if stop_after == "raw_deleted":
                cleanup_month(config, ledger, month)
            return
        year = int(month[:4])
        atmosphere, cldrad = companion_paths(config, year)
        for initial_row in ledger.sources(month):
            current = date.fromisoformat(initial_row["day"])
            row = ledger.source(initial_row["day"])
            try:
                if row["state"] in {"discovered", "downloading", "failed_retryable"}:
                    download_source(config, row, ledger)
                    row = ledger.source(current.isoformat())
                if stop_after == "downloaded":
                    continue
                if row["state"] in {"downloaded", "processing"}:
                    reduce_day(config, row, ledger, atmosphere, cldrad)
            except Exception as error:
                current_row = ledger.source(current.isoformat())
                ledger.transition(current.isoformat(), "failed_retryable",
                                  last_error=f"{type(error).__name__}: {error}",
                                  attempts=current_row["attempts"])
                raise
        if stop_after == "downloaded" or stop_after == "processed_verified":
            return
        batch_state = ledger.batch(month)["state"]
        if batch_state not in {"packaged", "uploading"}:
            package_month(config, ledger, month)
        if stop_after == "packaged":
            return
        upload_month(config, ledger, month)
        if stop_after == "server_verified":
            return
        cleanup_month(config, ledger, month)


def doctor(config: dict[str, Any]) -> None:
    """Check the local environment without exposing credentials."""
    workspace = config["workspace"]
    workspace.mkdir(parents=True, exist_ok=True)
    companion_checks = {
        year: {
            kind: Path(value).is_file()
            for kind, value in paths.items()
        }
        for year, paths in config.get("companions", {}).items()
    }
    remote = config.get("remote", {})
    identity_exists = Path(remote["identity"]).is_file() if remote.get("identity") else None
    with tempfile.NamedTemporaryFile(suffix=".nc", dir=workspace) as temporary:
        xr.Dataset({"value": ("sample", [1.0])}).to_netcdf(
            temporary.name, engine="netcdf4"
        )
        with xr.open_dataset(temporary.name, engine="netcdf4") as smoke:
            netcdf_roundtrip = smoke.value.item() == 1.0
    checks = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "netcdf4_engine": "netcdf4" in xr.backends.list_engines(),
        "curl": shutil.which("curl"),
        "ssh": shutil.which("ssh"),
        "rsync": shutil.which("rsync"),
        "workspace": str(workspace),
        "free_bytes": shutil.disk_usage(workspace).free,
        "constants": constants_record(),
        "companions": companion_checks,
        "remote_identity_exists": identity_exists,
        "netcdf_roundtrip": netcdf_roundtrip,
    }
    companions_valid = all(all(kinds.values()) for kinds in companion_checks.values())
    if (
        not checks["netcdf4_engine"]
        or not checks["curl"]
        or not checks["ssh"]
        or not checks["netcdf_roundtrip"]
        or not companions_valid
        or identity_exists is False
    ):
        raise RuntimeError(f"Environment check failed: {checks}")
    print(json.dumps(checks, indent=2, sort_keys=True))


def status(config: dict[str, Any], month: str | None) -> None:
    with ledger_for(config) as ledger:
        query = "SELECT batch_month, state, COUNT(*) AS count, SUM(COALESCE(raw_size, 0)) AS bytes FROM source"
        values: tuple[Any, ...] = ()
        if month:
            query += " WHERE batch_month = ?"
            values = (month,)
        query += " GROUP BY batch_month, state ORDER BY batch_month, state"
        print(json.dumps([dict(row) for row in ledger.connection.execute(query, values)],
                         indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "status", "run", "package", "upload", "cleanup"):
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name != "doctor":
            child.add_argument("--month", required=name != "status")
    commands.choices["run"].add_argument(
        "--stop-after",
        choices=("downloaded", "processed_verified", "packaged", "server_verified", "raw_deleted"),
        default="raw_deleted",
    )
    commands.choices["run"].add_argument("--start", type=date.fromisoformat)
    commands.choices["run"].add_argument("--end", type=date.fromisoformat)
    compare = commands.add_parser("compare")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--rtol", type=float, default=DEFAULT_COMPARE_RTOL)
    compare.add_argument("--atol", type=float, default=1e-12)
    remote = commands.add_parser("remote-publish")
    remote.add_argument("--archive-part", type=Path, required=True)
    remote.add_argument("--sidecar", type=Path, required=True)
    remote.add_argument("--publish-root", type=Path, required=True)
    remote.add_argument("--receipt", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "compare":
        compare_days(args.candidate, args.reference, args.rtol, args.atol)
        print("scientific arrays match")
        return 0
    if args.command == "remote-publish":
        remote_publish(args.archive_part, args.sidecar, args.publish_root, args.receipt)
        return 0
    config = load_config(args.config)
    if args.command == "doctor":
        doctor(config)
    elif args.command == "status":
        status(config, args.month)
    elif args.command == "run":
        run_month(config, args.month, args.stop_after, args.start, args.end)
    else:
        with ledger_for(config) as ledger:
            if args.command == "package":
                package_month(config, ledger, args.month)
            elif args.command == "upload":
                upload_month(config, ledger, args.month)
            elif args.command == "cleanup":
                print(f"deleted {cleanup_month(config, ledger, args.month)} raw files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
