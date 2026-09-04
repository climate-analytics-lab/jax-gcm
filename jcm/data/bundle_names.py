"""Data-mirror bundle path naming for prescribed forcing inputs.

The mirror lays each grid's boundary data out as
``bundles/<grid>[_l<nlev>]/<product>`` (see
``docs/source/design/data_mirror.md``). This module is the single source of
truth for that layout and is deliberately free of any intra-package (``jcm``)
import, so it can be loaded in isolation:

* :mod:`jcm.runners` imports it normally to resolve the ``auto`` emission
  inputs at model-build time.
* ``tools/benchmark.py`` loads it **by file path** (exactly as it loads
  ``jcm/data/remote.py``) to enumerate those same auto inputs for its
  pre-GPU prefetch **without importing** ``jcm`` — importing the package
  initialises a JAX backend, which preallocates the GPU before the free-card
  gate and makes the harness look like a tenant to its own gate (a six-job
  sweep died that way; see ``tools/benchmark.py:_hf_fetch``).

Keeping the naming convention here means the build-time resolver and the
prefetch enumerator cannot drift apart.
"""

from __future__ import annotations

#: Grid tokens (``"t63"``) for which the data mirror actually publishes bundles.
#: Single source of truth for the mirror's published-grid whitelist:
#: ``jcm/data/mirror/build_mirror.py`` imports this to drive its build loop (it
#: adds the per-grid Gaussian latitude count), and :func:`jcm.runners.
#: _resolve_one_emission_input` consults it so ``auto`` resolves to ``None``
#: (rather than eagerly fetching a non-existent ``bundles/<grid>/*.nc`` and
#: aborting) on any grid the mirror does not carry. That restores the null,
#: emission-free behaviour automatically for every non-mirrored grid — a
#: ``physics=echam-jam grid=echam_t42_l8_sigma`` run then composes with online
#: sources only, without hand-nulling the four emission keys.
PUBLISHED_GRIDS = frozenset({"t63", "t106"})

#: Vertical layer counts for which the mirror publishes the level-resolved
#: (``bundles/<grid>_l<nlev>/``) products. ``build_mirror.stage_bundles`` builds
#: the oxidant (and ozone) files only at these layer counts — ``t63``/``t106`` ×
#: ``l47``/``l95`` — so a *level-dependent* ``auto`` key on a published
#: horizontal grid but an unpublished layer count (e.g. ``t63_l8``) has no
#: bundle and must resolve to None rather than compose a nonexistent
#: ``hf://bundles/t63_l8/oxidants_pd.nc`` (F2). Level-FREE products
#: (emissions/dms/dust, published one-per-grid) ignore the layer count.
PUBLISHED_LEVELS = frozenset({47, 95})

#: Vertical-coordinate families for which the mirror publishes the level-resolved
#: products. ``build_mirror``/``interpolate_ozone`` interpolate the oxidant (and
#: ozone) files onto the model's **hybrid**-level centre pressures at a reference
#: surface pressure — the file's level k is a specific hybrid pressure, mapped
#: one-to-one onto model level k with no re-interpolation. That mapping is only
#: correct on a matching hybrid grid: a *sigma* grid whose (token, nlev) happen to
#: coincide with a published hybrid bundle (e.g. ``grid=echam_t42_l8_sigma``
#: ``grid.spectral_truncation=63 grid.layers=47`` → ``t63``/``l47``) would pull the
#: hybrid bundle and silently wire its stratospheric pressures onto unrelated sigma
#: levels — and :func:`jcm.forcing.validate_oxidant_levels` deliberately skips the
#: hyam/hybm coefficient cross-check for ``SigmaCoordinates``, so nothing else
#: catches it. So a LEVEL-dependent ``auto`` key additionally requires a hybrid
#: vertical; level-FREE products (emissions/dms/dust, purely horizontal) ignore it.
PUBLISHED_VERTICALS = frozenset({"hybrid"})

#: Prescribed-emission forcing keys that honour the ``auto`` convention. Value
#: is ``(subdir_suffix, filename)``: ``""`` is the horizontal bundle
#: (``bundles/<grid>/``), ``"_l{nlev}"`` the level-resolved one
#: (``bundles/<grid>_l<nlev>/``) — matching the mirror layout in
#: ``docs/source/design/data_mirror.md``. The ``{nlev}`` here is an *internal*
#: format token that :func:`emission_bundle_path` fills from the model grid; it
#: is not a user-facing path template (``auto`` is the only grid-portable
#: mechanism — see :func:`jcm.runners._resolve_one_emission_input`). The
#: presence of ``{nlev}`` in the suffix is also the single source of truth for
#: which keys are LEVEL-dependent — :func:`bundle_is_published` reads it so the
#: published-set knowledge lives in one place.
#: ``emissions_pd``/``oxidants_pd`` are the present-day climatology members.
EMISSION_AUTO_BUNDLES = {
    "emissions_file": ("", "emissions_pd.nc"),
    "dms_file": ("", "dms.nc"),
    "dust_file": ("", "dust.nc"),
    "oxidants_file": ("_l{nlev}", "oxidants_pd.nc"),
}


def bundle_is_published(key: str, token: str, nlev, vertical: str = "hybrid") -> bool:
    """Whether the mirror publishes the ``auto`` bundle for ``key`` on this grid.

    The horizontal ``token`` must be a :data:`PUBLISHED_GRIDS` member for every
    product. A LEVEL-dependent product (its subdir carries ``{nlev}`` — only
    ``oxidants_file`` today) additionally requires ``nlev`` to be a
    :data:`PUBLISHED_LEVELS` member *and* ``vertical`` (the coordinate family,
    ``"hybrid"``/``"sigma"``) to be a :data:`PUBLISHED_VERTICALS` member: the
    level-resolved bundles are interpolated onto hybrid-level pressures and are
    silently wrong when mapped level-for-level onto a sigma grid that merely
    shares the same (token, nlev). Level-FREE products (empty subdir:
    emissions/dms/dust) are purely horizontal and ignore both ``nlev`` and
    ``vertical``. Level-dependence is inferred from the subdir template in
    :data:`EMISSION_AUTO_BUNDLES`, so the published set and the path layout
    cannot drift. Shared by :func:`jcm.runners._emission_auto_resolves_to_none`
    (the build-time resolver + its warning predicate) and
    ``tools/benchmark.py``'s prefetch enumerator so all three agree on which
    ``auto`` keys have a real bundle (F2). ``vertical`` defaults to ``"hybrid"``
    so level-free callers (and hybrid grids) need not pass it.
    """
    if token not in PUBLISHED_GRIDS:
        return False
    subdir, _ = EMISSION_AUTO_BUNDLES[key]
    if "{nlev}" in subdir:
        return int(nlev) in PUBLISHED_LEVELS and vertical in PUBLISHED_VERTICALS
    return True


def grid_token(spectral_truncation) -> str:
    """Mirror grid token (``"t63"``) for a spectral truncation.

    Matches the relation ``utils.get_coords`` uses (truncation =
    ``total_wavenumbers - 2``); :func:`jcm.runners._grid_token` derives the
    truncation from a built ``coords`` and defers here so no hand-maintained
    table can go stale.
    """
    return f"t{int(spectral_truncation)}"


def emission_bundle_path(key: str, token: str, nlev) -> str:
    """``hf://`` path of the ``auto`` bundle for one emission ``key``."""
    subdir, name = EMISSION_AUTO_BUNDLES[key]
    return f"hf://bundles/{token}{subdir.format(nlev=int(nlev))}/{name}"


def auto_emission_bundle_paths(token: str, nlev) -> dict:
    """``{key: hf-path}`` for all four ``auto`` emission bundles on a grid."""
    return {key: emission_bundle_path(key, token, nlev)
            for key in EMISSION_AUTO_BUNDLES}
