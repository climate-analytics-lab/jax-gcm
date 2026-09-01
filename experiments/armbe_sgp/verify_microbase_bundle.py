"""Verify every allowlisted file in an extracted MICROBASE Mac bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {record["path"] for record in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise SystemExit(
            f"bundle member mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for record in manifest["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise SystemExit(f"bundle checksum mismatch: {record['path']}")
    print(f"verified {len(expected)} bundle files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
