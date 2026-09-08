"""Read-side view of the packaged mirror manifest (``mirror_manifest.json``).

The manifest is the data-driven source of truth for what the project data mirror
publishes: per product, which grids / vertical levels carry it, its yearly
coverage (or ``None`` for a climatology/static file), the mirror-relative path
template, and its time-alignment kind. It is generated data-side by
``jcm.data.mirror.build_mirror.stage_manifest`` from the ``PUBLISHED_*`` sets in
:mod:`jcm.data.bundle_names` plus that module's declarative product table, so the
availability knowledge the resolver consults cannot drift from the build.

This module is deliberately free of any intra-package (``jcm``) import — the same
invariant :mod:`jcm.data.bundle_names` and :mod:`jcm.data.yearly_files` maintain —
so it can be loaded in isolation (``spec_from_file_location``) by
``tools/benchmark.py``'s pre-GPU prefetch, which must not import ``jcm`` (that
initialises a JAX backend and preallocates the GPU before the free-card gate).
The manifest JSON sits beside this file and is read by ``__file__`` sibling, so a
file-path load resolves it the same way a package import does.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Packaged manifest, beside this module. Resolved via ``__file__`` so it works
#: under both a normal package import and a benchmark-style file-path load.
_DEFAULT_PATH = Path(__file__).resolve().with_name("mirror_manifest.json")


def load_manifest(path=None) -> dict:
    """Load and return the manifest dict (defaults to the packaged JSON)."""
    return json.loads(Path(path or _DEFAULT_PATH).read_text())


def product(manifest: dict, name: str) -> dict:
    """One product's record, raising a clear error on an unknown name."""
    try:
        return manifest["products"][name]
    except KeyError:
        raise KeyError(
            f"no mirror product {name!r}; known: "
            f"{sorted(manifest['products'])}") from None


def product_for_key(manifest: dict, key: str):
    """Return the product ``forcing.<key>=auto`` resolves to, or ``None``.

    Exactly one product per forcing key is flagged ``auto`` (the present-day
    climatology member — e.g. ``emissions_pd`` for ``emissions_file``); the
    transient/PI siblings share the key but are opt-in via an explicit path.
    """
    for name, rec in manifest["products"].items():
        if rec.get("auto") and rec.get("key") == key:
            return name
    return None


def is_published(manifest: dict, name: str, grid: str, nlev=None,
                 vertical: str = "hybrid") -> bool:
    """Whether the mirror publishes ``name`` for this (grid, nlev, vertical).

    Generalises :func:`jcm.data.bundle_names.bundle_is_published` over the whole
    product table: the ``grids`` list must carry the grid (a grid-free product —
    ``grids is None`` — publishes on any grid), and a level-dependent product
    (``levels`` set) additionally requires ``nlev`` in that list AND a matching
    ``vertical`` (the level-resolved bundles are on hybrid-level pressures, so a
    sigma grid sharing a published (grid, nlev) must NOT pull one). Level-free
    products ignore ``nlev``/``vertical``. ``staged`` is deliberately NOT checked
    here — a declared-but-unstaged product is "published" in the schema sense so
    the resolver can give a precise not-yet-staged error; use
    ``product(...)['staged']`` for availability.
    """
    rec = product(manifest, name)
    grids = rec["grids"]
    if grids is not None and grid not in grids:
        return False
    if rec["levels"] is not None:
        if nlev is None or int(nlev) not in rec["levels"]:
            return False
        if vertical != rec["vertical"]:
            return False
    return True


def bundle_path(manifest: dict, name: str, grid: str = None, nlev=None) -> str:
    """Mirror-relative path for a product, filling ``{grid}``/``{nlev}``.

    Returns the path *without* an ``hf://`` scheme (the resolver prepends it).
    Raises if a needed template field is missing.
    """
    tmpl = product(manifest, name)["path"]
    fields = {}
    if "{grid}" in tmpl:
        if grid is None:
            raise ValueError(f"product {name!r} path {tmpl!r} needs a grid")
        fields["grid"] = grid
    if "{nlev}" in tmpl:
        if nlev is None:
            raise ValueError(f"product {name!r} path {tmpl!r} needs nlev")
        fields["nlev"] = int(nlev)
    return tmpl.format(**fields)


def coverage(manifest: dict, name: str):
    """Return the inclusive ``[first, last]`` yearly coverage, or ``None``."""
    return product(manifest, name)["coverage"]
