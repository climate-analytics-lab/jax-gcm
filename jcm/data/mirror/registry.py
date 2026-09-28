"""Build ``registry.json`` for the upload tree (sha256 + size per file)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def build_registry(root: str, base: dict | None = None,
                   paths=None) -> dict:
    """Hash files under ``root``; entries override those of ``base``.

    ``base`` is the published registry: its entries for files not re-hashed
    are kept, so ``registry.json`` still covers the whole mirror. ``paths``
    (relative to ``root``) names the files to hash — the ones a build wrote;
    ``None`` hashes the whole tree.
    """
    reg = {"repo": "climate-analytics-lab/jax-gcm-data",
           "files": dict((base or {}).get("files", {}))}
    root_p = Path(root)
    if paths is None:
        files = [p for p in sorted(root_p.rglob("*"))
                 if p.is_file() and p.name != "registry.json"]
    else:
        files = [root_p / rel for rel in sorted(paths)]
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        reg["files"][str(p.relative_to(root_p))] = {
            "sha256": h.hexdigest(), "size": p.stat().st_size}
    return reg


def write_registry(root: str, base: dict | None = None, paths=None) -> str:
    """Write ``root/registry.json`` (merged onto ``base``, see build_registry)."""
    out = os.path.join(root, "registry.json")
    with open(out, "w") as f:
        json.dump(build_registry(root, base, paths), f, indent=1,
                  sort_keys=True)
    return out


if __name__ == "__main__":
    import sys
    print(write_registry(sys.argv[1]))
