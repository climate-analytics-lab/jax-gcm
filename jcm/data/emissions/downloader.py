"""Host-agnostic fetch for emission source files.

Deliberately *not* coupled to ESGF or any single provider: prescribed-emission
files are large and live in many places (a local path on an HPC scratch disk, an
HTTP(S) URL, an object-store bucket, a future self-hosted mirror — see the
follow-up issue on hosting a compressed 0.5° CEDS mirror). This helper resolves
any of those to a local path, with optional sha256 verification and on-disk
caching, so the rest of the pipeline only ever sees a file path.

It is a thin wrapper over :mod:`pooch` (already a dependency via xarray's I/O
extra) plus a trivial local-path passthrough.
"""

from __future__ import annotations

import os
import shutil


def fetch(source: str, *, known_hash: str | None = None,
          cache_dir: str | None = None) -> str:
    """Resolve ``source`` to a local file path.

    Args:
        source: a local filesystem path **or** an ``http(s)://`` / ``ftp://`` /
            ``doi:`` URL understood by pooch.
        known_hash: optional ``"sha256:…"`` (or bare hex) to verify the file;
            ``None`` skips verification (and, for a remote file, accepts whatever
            is downloaded — fine for trusted sources, but pass a hash for
            reproducibility).
        cache_dir: directory for downloaded files; defaults to pooch's OS cache
            under ``jcm_emissions`` (override with ``$JCM_EMISSIONS_CACHE``).

    Returns:
        Absolute path to the local file.

    """
    # Local files are returned as-is (after an existence check) so callers can
    # point straight at e.g. the on-disk CESM CEDS files.
    if os.path.exists(source):
        return os.path.abspath(source)

    import pooch

    cache_dir = cache_dir or os.environ.get(
        "JCM_EMISSIONS_CACHE",
        os.path.join(pooch.os_cache("jcm_emissions")),
    )
    return pooch.retrieve(url=source, known_hash=known_hash, path=cache_dir)


def stage(source: str, dest: str, *, known_hash: str | None = None) -> str:
    """Fetch ``source`` and copy it to ``dest`` (e.g. into a run directory)."""
    local = fetch(source, known_hash=known_hash)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.abspath(local) != os.path.abspath(dest):
        shutil.copyfile(local, dest)
    return os.path.abspath(dest)
