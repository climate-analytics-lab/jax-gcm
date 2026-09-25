"""Fetch boundary-condition bundles from the Hugging Face data mirror.

The mirror (dataset ``climate-analytics-lab/jax-gcm-data``) hosts
per-grid bundles assembled by ``jcm.data.mirror`` — terrain, forcing,
ozone, oxidants, emissions, DMS and dust files for the supported grids
(t63, t106, t127, t255 at L47/L95; ne30pg3 native columns). ``registry.json`` at the
dataset root lists every file with its sha256.

Every read is pinned to one dataset commit, so every machine reads the same
bytes whatever its cache holds:

- the commit is :data:`MIRROR_REVISION`, or ``$JCM_MIRROR_REVISION`` when set
  (a full 40-hex commit sha; a branch name is refused);
- a file cached at that commit is used with no network access;
- otherwise it is downloaded at that commit, and on a node that cannot reach
  the Hub the error prints the prefetch command.

Files resolve to a plain filesystem path, so they can be passed straight to
Hydra overrides::

    from jcm.data.remote import bundle_file
    terrain = bundle_file("t63", "terrain.nc")
    ozone = bundle_file("t63_l47", "ozone_pd.nc")

This module deliberately has no intra-package (``jcm``) import:
``tools/benchmark.py`` loads it by file path so that prefetching does not
initialise a JAX backend before the free-GPU gate.
"""

from __future__ import annotations

import os
import re

DEFAULT_REPO = "climate-analytics-lab/jax-gcm-data"

#: The dataset commit every mirror read resolves against: the upload of
#: 2026-09-24 04:13:00 UTC, which carries the conservatively remapped
#: t63/t106 ``emissions_{pd,pi}`` bundles of #889. It lives here, not in the
#: generated ``mirror_manifest.json``, because the manifest is rewritten by
#: ``build_mirror --stage manifest``. Bumping it changes every mirror input a
#: run reads, so it is a reviewed one-line change: ``build_mirror --stage
#: upload`` prints the new commit, and band files drawn at the old one are
#: regenerated in the same PR.
MIRROR_REVISION = "6d3506d7c957f24526c157f917043c8c40c8bc42"

#: Environment variable overriding :data:`MIRROR_REVISION` for a process.
REVISION_ENV = "JCM_MIRROR_REVISION"

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def mirror_revision() -> str:
    """Return the commit this process reads the mirror at.

    ``$JCM_MIRROR_REVISION`` when set, else :data:`MIRROR_REVISION`. Only a
    full commit sha is accepted: a branch such as ``main`` moves, so two
    machines (or two jobs of one run) could read different files under it.

    Raises:
        ValueError: the override is not a 40-hex commit sha.

    """
    value = os.environ.get(REVISION_ENV, "").strip()
    if not value:
        return MIRROR_REVISION
    if not _COMMIT_SHA.match(value):
        raise ValueError(
            f"{REVISION_ENV}={value!r} is not a commit sha. Mirror reads are "
            "pinned to an immutable commit; resolve a branch to one with\n"
            "  python -c \"from huggingface_hub import HfApi; print(HfApi()"
            f".dataset_info('{DEFAULT_REPO}', revision='{value}').sha)\"\n"
            f"and set {REVISION_ENV} to the printed sha.")
    return value


def revision_source() -> str:
    """Return ``"env"`` when ``$JCM_MIRROR_REVISION`` is set, else ``"pinned"``."""
    return "env" if os.environ.get(REVISION_ENV, "").strip() else "pinned"


def is_transport_failure(exc: BaseException) -> bool:
    """Whether ``exc`` means the Hub could not be reached, not that it said no.

    Decides only which error to show: "not cached, prefetch" or "not on the
    mirror". ``huggingface_hub`` often wraps the reason (an uncached download
    under ``HF_HUB_OFFLINE=1`` raises ``LocalEntryNotFoundError`` caused by
    the offline error), so the cause chain is walked and the first HTTP status
    or connection error decides; an HTTP 4xx is an answer, 429/5xx are not.
    """
    try:
        from huggingface_hub.errors import (
            HfHubHTTPError, LocalEntryNotFoundError)
    except ImportError:                             # pragma: no cover
        HfHubHTTPError = LocalEntryNotFoundError = ()
    local_miss, seen, link = False, set(), exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        if HfHubHTTPError and isinstance(link, HfHubHTTPError):
            status = getattr(getattr(link, "response", None),
                             "status_code", None)
            return status is None or status == 429 or status >= 500
        if isinstance(link, (ConnectionError, TimeoutError)):
            return True
        if LocalEntryNotFoundError and isinstance(link,
                                                  LocalEntryNotFoundError):
            local_miss = True
        link = link.__cause__ or link.__context__
    return local_miss


def fetch(path: str, repo_id: str = DEFAULT_REPO) -> str:
    """Resolve one mirror file at :func:`mirror_revision` to a local path.

    The Hugging Face cache stores each file under the commit it was
    downloaded at, so only a copy at this commit is accepted: a cache holding
    an older copy re-fetches rather than silently reading it. A cached file is
    returned with no network access at all, which is what lets a warm cache
    serve internet-less compute nodes.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as e:                        # pragma: no cover
        raise ImportError(
            "Fetching remote boundary conditions needs huggingface_hub: "
            "pip install huggingface_hub") from e
    revision = mirror_revision()
    common = dict(repo_id=repo_id, repo_type="dataset", filename=path,
                  revision=revision)
    try:
        return hf_hub_download(**common, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    try:
        return hf_hub_download(**common)
    except Exception as e:
        if not is_transport_failure(e):
            raise FileNotFoundError(
                f"hf://{path} could not be fetched at mirror revision "
                f"{revision}: {type(e).__name__}: {e}") from e
        # The prefetch must use the same revision, or it warms a cache entry
        # this read will not accept.
        env = (f"{REVISION_ENV}={revision} "
               if revision != MIRROR_REVISION else "")
        raise FileNotFoundError(
            f"hf://{path} at mirror revision {revision} is not in the local "
            f"Hugging Face cache and could not be downloaded "
            f"({type(e).__name__}). Compute nodes have no internet — "
            f"prefetch on a login node first:\n"
            f"  {env}python -c \"from jcm.data.remote import fetch; "
            f"fetch('{path}')\"") from e


def bundle_file(grid: str, name: str, repo_id: str = DEFAULT_REPO) -> str:
    """Resolve ``bundles/<grid>/<name>`` to a local path (see :func:`fetch`).

    ``grid`` is a Gaussian grid (``t63``, ``t106``, ``t127``, ``t255``), one
    of those with a level suffix (``t63_l47``, ``t255_l95``, …), or
    ``ne30pg3`` — level-suffixed grids hold the level-resolved products
    (ozone, oxidants). ``jcm/data/mirror_manifest.json`` lists what exists.
    """
    return fetch(f"bundles/{grid}/{name}", repo_id=repo_id)
