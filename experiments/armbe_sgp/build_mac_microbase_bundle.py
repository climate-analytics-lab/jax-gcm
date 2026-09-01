"""Build the credential-free Apple Silicon MICROBASE handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "dist"
BUNDLE_NAME = "microbase-sgp-mac-apple-silicon-20260825"

CODE_FILES = (
    "microbase_mac_pipeline.py",
    "microbase_physics.py",
    "collocate_microbase_pilot.py",
    "process_microbase_month.py",
    "microbase_mac_pipeline_test.py",
    "collocate_microbase_pilot_test.py",
    "process_microbase_month_test.py",
    "requirements-microbase-mac.txt",
    "microbase_campaign.example.toml",
    "microbase_benchmark.toml",
    "setup_microbase_mac.sh",
    "run_microbase_mac_benchmark.sh",
    "verify_microbase_bundle.py",
    "MAC_MICROBASE_README_FIRST.md",
    "MAC_MICROBASE_CHATGPT_HANDOFF.md",
    "MAC_LOCAL_MICROBASE_PIPELINE_PLAN.md",
)

ATMOSPHERE = (
    HERE / "data/order-267892/ftp.archive.arm.gov/fisherm1/267892/"
    "sgparmbeatmC1.c1/sgparmbeatmC1.c1.20110101.000000.cdf"
)
CLDRAD = (
    HERE / "data/order-267892/ftp.archive.arm.gov/fisherm1/267892/"
    "sgparmbecldradC1.c1/sgparmbecldradC1.c1.20110101.003000.nc"
)
REFERENCE = HERE / "outputs/echam_layer_cloud_2011"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def manifest(root: Path) -> dict:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "bundle_schema_version": 1,
        "bundle": root.name,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contains_credentials": False,
        "contains_raw_microbase": False,
        "files": files,
    }


def build(output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    for required in (ATMOSPHERE, CLDRAD, REFERENCE):
        if not required.exists():
            raise FileNotFoundError(required)
    with tempfile.TemporaryDirectory(prefix="microbase-bundle-") as temporary_name:
        temporary = Path(temporary_name)
        root = temporary / BUNDLE_NAME
        root.mkdir()
        for name in CODE_FILES:
            copy_file(HERE / name, root / name)
        copy_file(
            HERE / "microbase_campaign.example.toml",
            root / "microbase_campaign.toml",
        )
        copy_file(ATMOSPHERE, root / "benchmark/companions" / ATMOSPHERE.name)
        copy_file(CLDRAD, root / "benchmark/companions" / CLDRAD.name)
        for day in range(24, 31):
            source = REFERENCE / f"2011-06-{day:02d}"
            destination = root / "benchmark/reference" / source.name
            shutil.copytree(source, destination)
        copy_file(
            REFERENCE / "month_manifest.json",
            root / "benchmark/reference/month_manifest.json",
        )
        bundle_manifest = manifest(root)
        (root / "BUNDLE_MANIFEST.json").write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
        )
        archive = output / f"{BUNDLE_NAME}.tar.gz"
        partial = archive.with_suffix(".tar.gz.part")
        with tarfile.open(partial, "w:gz", compresslevel=6) as tar:
            for path in sorted(root.rglob("*")):
                tar.add(
                    path,
                    arcname=f"{BUNDLE_NAME}/{path.relative_to(root)}",
                    recursive=False,
                )
        partial.replace(archive)
    sidecar = Path(f"{archive}.sha256.json")
    sidecar.write_text(
        json.dumps(
            {
                "archive": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return archive, sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archive, sidecar = build(args.output)
    print(archive)
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
