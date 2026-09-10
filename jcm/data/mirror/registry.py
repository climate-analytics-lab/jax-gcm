"""Build ``registry.json`` for the upload tree (sha256 + size per file)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def build_registry(root: str) -> dict:
    reg = {"repo": "climate-analytics-lab/jax-gcm-data", "files": {}}
    root_p = Path(root)
    for p in sorted(root_p.rglob("*")):
        if not p.is_file() or p.name == "registry.json":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        rel = str(p.relative_to(root_p))
        reg["files"][rel] = {"sha256": h.hexdigest(),
                             "size": p.stat().st_size}
    return reg


def write_registry(root: str) -> str:
    out = os.path.join(root, "registry.json")
    with open(out, "w") as f:
        json.dump(build_registry(root), f, indent=1, sort_keys=True)
    return out


if __name__ == "__main__":
    import sys
    print(write_registry(sys.argv[1]))
