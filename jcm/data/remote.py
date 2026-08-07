"""Fetch boundary-condition bundles from the Hugging Face data mirror.

The mirror (dataset ``climate-analytics-lab/jax-gcm-data``) hosts
per-grid bundles assembled by ``jcm.data.mirror`` — terrain, forcing,
ozone, oxidants, emissions, DMS and dust files for the supported grids
(t63, t106 at L47/L95; ne30pg3 native columns). ``registry.json`` at the
dataset root lists every file with its sha256.

Files download once into the local Hugging Face cache and resolve to a
plain filesystem path, so they can be passed straight to Hydra
overrides::

    from jcm.data.remote import bundle_file
    terrain = bundle_file("t63", "terrain.nc")
    ozone = bundle_file("t63_l47", "ozone_pd.nc")
"""

from __future__ import annotations

DEFAULT_REPO = "climate-analytics-lab/jax-gcm-data"


def fetch(path: str, repo_id: str = DEFAULT_REPO,
          revision: str | None = None) -> str:
    """Download (or reuse from cache) one file from the data mirror."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:                        # pragma: no cover
        raise ImportError(
            "Fetching remote boundary conditions needs huggingface_hub: "
            "pip install huggingface_hub") from e
    return hf_hub_download(repo_id=repo_id, repo_type="dataset",
                           filename=path, revision=revision)


def bundle_file(grid: str, name: str, repo_id: str = DEFAULT_REPO,
                revision: str | None = None) -> str:
    """Resolve ``bundles/<grid>/<name>`` to a local path.

    ``grid`` is one of ``t63``, ``t106``, ``t63_l47``, ``t63_l95``,
    ``t106_l47``, ``t106_l95``, ``ne30pg3`` — level-suffixed grids hold
    the level-resolved products (ozone, oxidants).
    """
    return fetch(f"bundles/{grid}/{name}", repo_id=repo_id,
                 revision=revision)
