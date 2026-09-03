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

#: Prescribed-emission forcing keys that honour the ``auto`` convention. Value
#: is ``(subdir_suffix, filename)``: ``""`` is the horizontal bundle
#: (``bundles/<grid>/``), ``"_l{nlev}"`` the level-resolved one
#: (``bundles/<grid>_l<nlev>/``) — matching the mirror layout in
#: ``docs/source/design/data_mirror.md``. The ``{nlev}`` here is an *internal*
#: format token that :func:`emission_bundle_path` fills from the model grid; it
#: is not a user-facing path template (``auto`` is the only grid-portable
#: mechanism — see :func:`jcm.runners._resolve_one_emission_input`).
#: ``emissions_pd``/``oxidants_pd`` are the present-day climatology members.
EMISSION_AUTO_BUNDLES = {
    "emissions_file": ("", "emissions_pd.nc"),
    "dms_file": ("", "dms.nc"),
    "dust_file": ("", "dust.nc"),
    "oxidants_file": ("_l{nlev}", "oxidants_pd.nc"),
}


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
