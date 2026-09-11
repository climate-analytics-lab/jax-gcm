"""Forcing-side assembly of a :class:`jcm.forcing.ForcingData` from config.

The single home for composing a boundary-forcing struct out of a resolved
``forcing`` config + model ``coords``: the ``auto`` emission resolution, the
per-product ``{year}``/``hf://`` resolution (incl. ``_resolve_data_path``'s
provenance-recording fetch and the packaged/mirror auto-ozone/terrain
discovery), the merge-compatibility guard, and the ozone / emissions / dms /
dust / oxidant / MACv2 attach chain. Both the CLI door
(:func:`jcm.runners.build_forcing`, a thin config-unpacker) and the Python door
(:meth:`jcm.forcing.ForcingData.from_bundles`) drive :func:`build_forcing`
here, so the two provably agree pytree-for-pytree (#751; the equivalence test
in ``forcing_test``).

This module is the shared engine and depends only on :mod:`jcm.forcing` /
:mod:`jcm.data` — never on the CLI adapter (:mod:`jcm.runners`). The runner
holds the cfg dispatch plus the pySES column-sampling branch (deliberately not
unified — it delegates to ``attach_jam_forcing`` but shares the resolution
helpers here); tests that stub the resolution helpers patch them on THIS
module, which reaches both doors.
"""

from __future__ import annotations

import logging

from jcm import provenance
from jcm.data import input_resolution as ir
from jcm.data import mirror_manifest as mm
from jcm.forcing import expand_yearly_files as _expand_years

logger = logging.getLogger(__name__)

#: The prescribed-emission forcing keys that honour the ``auto`` convention:
#: ``auto`` resolves each to its per-grid HF bundle (via the mirror manifest)
#: when a prognostic-aerosol (JAM) package is active. The manifest is the
#: read-side single source of truth for whether a given key's bundle exists on a
#: grid (:func:`mm.is_published`); this is only the set of keys to iterate.
_EMISSION_AUTO_KEYS = ("emissions_file", "dms_file", "dust_file",
                       "dust_preferential_file", "dust_soil_types_file",
                       "dust_regions_file", "dust_roughness_file",
                       "oxidants_file")

#: The Tegen dust inputs that must arrive together: ``mo_ham_dust.f90`` aborts
#: without any of them, and running with the soil textures, preferential sources
#: or tuning regions missing would silently emit an all-coarse, untuned flux.
#: The roughness map is NOT here — the Fortran reads it only when
#: ``ndurough = 0``, so it is the one channel with a real opt-out.
_DUST_REQUIRED_KEYS = ("dust_preferential_file", "dust_soil_types_file",
                       "dust_regions_file")


# ---------------------------------------------------------------------------
# small config/path helpers (shared by the attach chain)
# ---------------------------------------------------------------------------

def _resolve_data_path(path):
    """Resolve a boundary-file path from config.

    ``hf://<path-in-dataset>`` fetches (or reuses from the local HF cache)
    the file from the project data mirror via :mod:`jcm.data.remote`, e.g.
    ``hf://bundles/t63/terrain.nc``. Anything else passes through
    unchanged. Fetch on a login/head node first — compute nodes usually
    have no internet, but a warm cache needs none.

    The ``hf://`` scheme is parsed in ONE place —
    :func:`jcm.data.input_resolution._fetch_path` — this wrapper adds the
    provenance recording (fetched pairs and plain local paths) and recurses
    element-wise over lists so each element records its own provenance.
    """
    if ir._is_seq(path):
        # emissions_file may be a list of paths (incl. Hydra ListConfig).
        # Mappings/bytes fail _is_seq and pass through untouched — iterating
        # them would silently turn a mis-typed config into a list of keys/ints.
        return [_resolve_data_path(p) for p in path]

    def _fetch(rel):
        from jcm.data.remote import fetch
        resolved = fetch(rel)
        provenance.record_input(f"hf://{rel}", resolved)
        return resolved

    resolved = ir._fetch_path(path, _fetch)
    if isinstance(resolved, str) and resolved == path:
        provenance.record_input(path)   # plain path; hf:// recorded in _fetch
    return resolved


def _resolve_auto_ozone(coords):
    """Find an ozone climatology matching the model grid.

    Two-stage discovery: (1) the packaged ``ozone_packaged`` product
    (``jcm/data/bc/*/ozone.nc``) shape-matched on (nlev, nlat, nlon) via the one
    packaged-product mechanism (:func:`jcm.data.input_resolution.
    resolve_packaged`); (2) the data mirror's per-grid ``bundles/<grid>_l<nlev>/
    ozone_pd.nc`` (cache-first fetch — works offline once cached; the loader
    rejects any grid mismatch). Grid identity is then fully validated by
    ``OzoneClimatology.from_file``. Returns ``None`` when neither stage finds a
    file — the caller warns and falls back to the analytic profile, whose ~7.6×
    tropospheric ozone column biases clear-sky OLR ~12 W/m² low.

    Auto resolves to ``None`` on a **sigma** grid: every ozone product (packaged
    and mirror alike) is written by ``jcm.data.bc.interpolate_ozone`` onto the
    model's *hybrid*-level centre pressures and mapped level-for-level, so a
    sigma grid that merely shares a published (token, nlev) would wire
    stratospheric-pressure ozone onto unrelated sigma levels — the same silent
    corruption the oxidant gate rejects (the manifest's hybrid-only verticals).
    ``OzoneClimatology.from_file`` only cross-checks shape and lat/lon, not the
    vertical coordinate, so nothing downstream would catch it.
    """
    if _vertical_kind(coords) != "hybrid":
        logger.warning(
            "forcing.ozone_file=auto on a %s-vertical grid — the packaged and "
            "mirror ozone products are interpolated onto hybrid-level pressures "
            "and would be mapped level-for-level onto unrelated sigma levels. "
            "Falling back to the ANALYTIC ozone profile; supply an on-grid "
            "forcing.ozone_file (built for this vertical grid) for production "
            "radiation.",
            _vertical_kind(coords),
        )
        return None
    nlon, nlat = (int(v) for v in coords.horizontal.nodal_shape)
    nlev = int(coords.nodal_shape[0])
    packaged = ir.resolve_packaged(mm.load_manifest(), "ozone_packaged",
                                   nlev=nlev, nlat=nlat, nlon=nlon)
    if packaged is not None:
        return packaged
    token = _grid_token(coords)
    from jcm.data.remote import bundle_file
    try:
        return str(bundle_file(f"{token}_l{nlev}", "ozone_pd.nc"))
    except Exception as e:  # noqa: BLE001 — degrade, but LOUDLY
        # Warning, not info: the analytic-profile fallback biases
        # clear-sky OLR ~12 W/m² and the generic no-packaged-file
        # warning downstream does not mention the failed mirror fetch.
        logger.warning(
            "auto-ozone: mirror fetch bundles/%s_l%d/ozone_pd.nc failed "
            "(%s); falling back to the analytic ozone profile.",
            token, nlev, e,
        )
    return None


def _resolve_auto_terrain(coords):
    """Native-grid terrain path for ``terrain.kind: auto``.

    Terrain must be NATIVE to the model grid: horizontally interpolating
    a coarser file breaks the Lott-Miller SSO sub-grid orography fields
    (shape mismatch inside the column vmap). Stages: the packaged
    ``terrain_packaged`` product (``jcm/data/bc/*/terrain.nc``) shape-matched on
    (nlat, nlon) via :func:`jcm.data.input_resolution.resolve_packaged`, then the
    mirror's ``bundles/<grid>/terrain.nc``. Raises when neither exists, because a
    silently substituted terrain corrupts the run.
    """
    nlon, nlat = (int(v) for v in coords.horizontal.nodal_shape)
    packaged = ir.resolve_packaged(mm.load_manifest(), "terrain_packaged",
                                   nlat=nlat, nlon=nlon)
    if packaged is not None:
        return packaged
    token = _grid_token(coords)
    from jcm.data.remote import bundle_file
    try:
        return str(bundle_file(token, "terrain.nc"))
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"terrain.kind=auto: no packaged terrain matches "
            f"({nlon}x{nlat}) and the mirror fetch of "
            f"bundles/{token}/terrain.nc failed: {e}"
        ) from e


def _forcing_products(file_spec, years, available):
    """Split a forcing spec into per-product, year-expanded file sets.

    Thin adapter over :func:`jcm.data.input_resolution.forcing_products` — the
    single home for the list-into-products split + ``{year}`` coverage clamp.
    """
    return ir.forcing_products(file_spec, years, available)


def _open_forcing_dataset(path):
    """Open one product's file(s) as a single xarray dataset.

    A product may be a list of yearly files (a ``{year}`` expansion) — those
    share one time axis and are concatenated with
    ``open_mfdataset(combine="by_coords")``. A scalar path opens directly.
    """
    import xarray as xr
    if isinstance(path, (list, tuple)):
        paths = [str(p) for p in path]
        return (xr.open_mfdataset(paths, combine="by_coords") if len(paths) > 1
                else xr.open_dataset(paths[0]))
    return xr.open_dataset(str(path))


def _product_available_years(forcing_cfg, key: str):
    """Per-product source coverage, falling back to ``available_years``.

    A preset can mix yearly products with different coverages (e.g.
    ``forcing_era5`` surface files run to 2024 while the FZJ ``ozone_amip``
    and CEDS ``emissions_amip`` bundles both end in 2022); a per-product
    override (``ozone_available_years`` / ``emissions_available_years`` /
    ``oxidants_available_years``) keeps each pattern's expansion inside the
    files that actually exist — for run dates beyond it, the time lookup
    clamps to the last sample. ``key`` selects the override; any product
    without one shares the top-level ``available_years``.
    """
    avail = forcing_cfg.get(key, None)
    return avail if avail is not None else forcing_cfg.get(
        "available_years", None)


def _model_latlon_deg(coords):
    """Model nodal latitudes/longitudes in degrees (dinosaur stores radians)."""
    import numpy as np
    lat_deg = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
    lon_deg = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
    return lat_deg, lon_deg


def _ensure_parent_forcing(forcing, coords):
    """Build the aquaplanet parent ``ForcingData`` when ``kind: default``.

    Same rationale as ``_attach_ozone``: ``default_forcing`` preserves the
    cos²-latitude SST climatology that ``ForcingData.zeros`` would silently
    replace with a uniform 288.15 K placeholder.
    """
    if forcing is not None:
        return forcing
    from jcm.forcing import default_forcing
    return default_forcing(coords.horizontal)


def _grid_token(coords) -> str:
    """Mirror grid token (``"t63"``) for the model's horizontal grid.

    Derived from the spectral resolution (truncation =
    ``total_wavenumbers - 2``, the same relation ``utils.get_coords``
    uses), so no hand-maintained table can go stale; whether the mirror
    actually carries the grid is decided by the fetch itself.
    """
    return f"t{int(coords.horizontal.total_wavenumbers) - 2}"


def _vertical_kind(coords) -> str:
    """Coordinate family (``"hybrid"``/``"sigma"``) of the model's vertical grid.

    Gates the level-resolved ``auto`` products (oxidants, ozone), which the data
    mirror interpolates onto hybrid-level pressures and maps level-for-level: a
    sigma grid must NOT pull a hybrid bundle that merely shares its (token, nlev).
    Uses the same ``HybridCoordinates`` test as
    :func:`jcm.forcing.validate_oxidant_levels` so the availability gate and the
    (sigma-skipping) coefficient validator classify a grid identically.
    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    return "hybrid" if isinstance(coords.vertical, HybridCoordinates) else "sigma"


# ---------------------------------------------------------------------------
# auto emission resolution
# ---------------------------------------------------------------------------

def _emission_auto_resolves_to_none(key, coords, jam, is_pyses) -> bool:
    """Whether ``auto`` for one emission ``key`` resolves to None — pure classify.

    ``auto`` yields None (no bundle) on the pySES path, for a non-JAM package
    (neither consumes prescribed emissions), or when the mirror does not publish
    THIS key's bundle for the model grid — a key-specific manifest lookup
    (:func:`jcm.data.mirror_manifest.is_published`): the horizontal token
    must be a published grid, and the level-dependent ``oxidants_file`` bundle
    additionally requires a published layer count AND a hybrid vertical (the
    bundle is on hybrid-level pressures — mapping it level-for-level onto a
    sigma grid is silently wrong), so a published-horizontal / unpublished-level
    combo (e.g. ``t63_l8``) OR a sigma grid that merely shares a published
    (token, nlev) (e.g. ``echam_t42_l8_sigma`` at ``t63``/``l47``) nulls
    ``oxidants_file`` while the level-free emissions/dms/dust keys still resolve
    (F2). No fetch, no side
    effects. Shared by the build-time resolver
    (:func:`_resolve_one_emission_input`) and the config-trap warner
    (:func:`jcm.runners.warn_on_config_traps`) so the None-decision the warning
    reasons about can never drift from the one the build actually applies. The
    pySES / non-JAM cases return before ``_grid_token`` so pySES native coords
    are never asked for a spectral token.
    """
    if is_pyses or not jam:
        return True
    # Publication gating is the manifest lookup ``mm.is_published`` over the
    # whole product table — the read-side single source of truth (#751).
    manifest = mm.load_manifest()
    return not mm.is_published(
        manifest, mm.product_for_key(manifest, key),
        _grid_token(coords), int(coords.nodal_shape[0]), _vertical_kind(coords))


def _resolve_one_emission_input(value, key, coords, jam, is_pyses):
    """Resolve one prescribed-emission forcing value (``auto`` / explicit).

    * ``auto`` → the per-grid HF bundle, resolved + eager-fetched *now* (cold
      cache fails loudly at build time) by :func:`jcm.data.input_resolution.
      resolve_input` — but only on the spectral path with a JAM package active;
      pySES native grids are not the spectral-token bundles and a non-JAM
      package consumes no emissions, so both give ``None`` (the early gate keeps
      pySES native coords from being asked for a spectral token). Fetch is
      routed through ``_resolve_data_path`` so the hf:// scheme + provenance are
      honoured. ``auto`` is the only grid-portable mechanism; there is no
      user-facing ``{grid}``/``{nlev}`` template.
    * explicit null (``None``/``""``/``"null"``) → ``None`` (opt-out).
    * an explicit path / ``hf://`` URL → returned unchanged for the attach
      helper's own ``_resolve_data_path`` to fetch. A ``{year}`` pattern is
      left intact for :func:`_expand_years` (there is no ``{grid}``/``{nlev}``
      substitution — use ``auto`` to let a config follow the grid).
    """
    if value == "auto":
        if _emission_auto_resolves_to_none(key, coords, jam, is_pyses):
            # Non-mirrored grid / unpublished level / pySES / non-JAM: resolve
            # ``auto`` to None so the run falls back to the emission-free
            # baseline instead of aborting on a 404. ``warn_on_config_traps``
            # surfaces the silent degrade (same None-decision), not a log here.
            return None
        r = ir.resolve_input(
            key, "auto", grid_token=_grid_token(coords),
            nlev=int(coords.nodal_shape[0]), vertical=_vertical_kind(coords),
            fetch=lambda rel: _resolve_data_path("hf://" + rel))
        return r.paths[0] if not r.is_none else None
    if value in (None, "", "null", "none"):
        return None
    return value


def _resolve_emission_inputs(forcing_cfg, cfg, coords, is_pyses):
    """Concretise the ``auto`` emission keys on ``forcing_cfg``.

    Returns ``forcing_cfg`` with the four prescribed-emission keys resolved (see
    :func:`_resolve_one_emission_input`). ``auto`` is keyed off whether the
    composed physics is the prognostic-aerosol (JAM) module — the only package
    that consumes prescribed emissions.

    Pure resolution only: the silent degrade when a JAM run's ``auto`` keys
    null on a non-mirrored grid (or the pySES path) is surfaced by
    :func:`jcm.runners.warn_on_config_traps`, which folds it into ONE config-trap
    warning with the zero-emission finding rather than an invisible info log
    here (F2).
    """
    from omegaconf import OmegaConf

    jam = str((cfg.get("physics", {}) or {}).get("aerosol_module", "")) == "jam"
    updates = {
        key: _resolve_one_emission_input(
            forcing_cfg.get(key, None), key, coords, jam, is_pyses)
        for key in _EMISSION_AUTO_KEYS
    }
    return OmegaConf.merge(forcing_cfg, updates)


# ---------------------------------------------------------------------------
# the shared engine (both doors call this)
# ---------------------------------------------------------------------------

def build_forcing(cfg, coords):
    """Build a spectral-grid ``ForcingData`` from ``cfg`` — the forcing engine.

    Unpacks ``cfg.forcing``, resolves its ``auto`` emission keys, then runs the
    shared spectral assembly (:func:`assemble_spectral_forcing`). Both the CLI
    door (:func:`jcm.runners.build_forcing`, which delegates here for every
    non-pySES dycore) and the Python door
    (:meth:`jcm.forcing.ForcingData.from_bundles`) go through this one entry, so
    they provably agree pytree-for-pytree (#751). The pySES column backend is
    dispatched adapter-side (in the runner) so this engine never depends on it.

    ``kind: default`` yields ``None`` (``Model.run`` falls back to the aquaplanet
    ``default_forcing``); ``kind: from_file`` loads a netCDF boundary file. The
    four prescribed-emission keys default to ``auto`` — a JAM package composes
    the per-grid HF bundles by itself while non-JAM packages leave them empty;
    ``forcing.<key>=null`` opts out.
    """
    _forcing_cfg = cfg.get("forcing", None)
    if _forcing_cfg is not None:
        _forcing_cfg = _resolve_emission_inputs(
            _forcing_cfg, cfg, coords, is_pyses=False)
    return assemble_spectral_forcing(_forcing_cfg, coords)


def assemble_spectral_forcing(forcing_cfg, coords):
    """Assemble a spectral-grid ``ForcingData`` from a RESOLVED forcing config.

    ``forcing_cfg`` has already had its ``auto`` emission keys concretised (see
    :func:`_resolve_emission_inputs`). ``kind: default`` returns ``None`` (the
    caller falls back to ``default_forcing``); ``kind: from_file`` loads the
    surface netCDF, then the ozone / emissions / dms / dust / oxidant / MACv2
    attach chain runs. The single engine both :func:`jcm.runners.build_forcing`
    (spectral path) and :meth:`jcm.forcing.ForcingData.from_bundles` call, so
    they agree pytree-for-pytree.
    """
    if forcing_cfg is None or forcing_cfg.kind == "default":
        forcing = None
    elif forcing_cfg.kind == "from_file":
        from jcm.forcing import ForcingData
        files = _expand_years(forcing_cfg.file, forcing_cfg.get("years", None),
                              forcing_cfg.get("available_years", None))
        forcing = ForcingData.from_file(
            _resolve_data_path(files), coords=coords,
            align_mode=str(forcing_cfg.get("align", "auto")))
    else:
        raise ValueError(f"Unknown forcing.kind={forcing_cfg.kind!r}")
    forcing = _attach_ozone(forcing, forcing_cfg, coords)
    forcing = _attach_emissions(forcing, forcing_cfg, coords)
    forcing = _attach_dms(forcing, forcing_cfg, coords)
    forcing = _attach_dust(forcing, forcing_cfg, coords)
    forcing = _attach_oxidants(forcing, forcing_cfg, coords)
    forcing = _attach_macv2_weights(forcing, forcing_cfg, coords)
    return forcing


# ---------------------------------------------------------------------------
# ozone
# ---------------------------------------------------------------------------

def _attach_ozone(forcing, forcing_cfg, coords):
    """Load the ozone climatology and attach to ``forcing``.

    ``ozone_file: auto`` (the shipped default) resolves a packaged
    climatology matching the grid via ``_resolve_auto_ozone``; no match
    degrades to the analytic ozone profile with a warning. An explicit
    path is loaded strictly (errors on any mismatch). ``null`` disables
    the climatology silently (analytic profile, no warning).

    When ``forcing`` is ``None`` (``kind: default``) and an ozone file IS
    given, build the parent struct via ``default_forcing(...)`` so the
    aquaplanet cos²-latitude SST climatology is preserved — using
    ``ForcingData.zeros`` here would silently swap it for the uniform
    288.15 K placeholder, materially changing the boundary conditions
    for any run configured with only ``ozone_file``.
    """
    if forcing_cfg is None:
        return forcing
    ozone_file = _resolve_data_path(_expand_years(
        forcing_cfg.get("ozone_file", None),
        forcing_cfg.get("years", None),
        _product_available_years(forcing_cfg, "ozone_available_years")))
    if isinstance(ozone_file, (list, tuple)):
        ozone_file = [str(p) for p in ozone_file]
    if ozone_file in (None, "", "null"):
        provenance.record_fact("ozone_source", "analytic (no ozone_file)")
        return forcing
    if ozone_file == "auto":
        ozone_file = _resolve_auto_ozone(coords)
        if ozone_file is None:
            provenance.record_fact(
                "ozone_source", "analytic (auto found no packaged match)")
            logging.warning(
                "forcing.ozone_file=auto: no packaged jcm/data/bc/*/ozone.nc "
                "matches this grid — falling back to the ANALYTIC ozone "
                "profile, whose ~7.6x tropospheric ozone column biases "
                "clear-sky OLR low by ~12 W/m2. Prepare a climatology with "
                "jcm.data.bc.interpolate_ozone for production radiation."
            )
            return forcing
        logging.info("forcing.ozone_file=auto resolved to %s", ozone_file)
    import numpy as np

    from jcm.forcing import default_forcing
    from jcm.ozone_climatology import OzoneClimatology
    nlon, nlat = coords.horizontal.nodal_shape
    nlev = coords.nodal_shape[0]
    # Pass the model's lat/lon (degrees) so the loader catches files
    # with the right shape but flipped/shifted grids — same N points,
    # wrong column mapping, would otherwise wire ozone into the wrong
    # latitudes silently. Dinosaur stores both in radians.
    lat_deg = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
    lon_deg = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
    climatology = OzoneClimatology.from_file(
        ozone_file,
        nlon=int(nlon), nlat=int(nlat), nlev=int(nlev),
        lat_deg=lat_deg, lon_deg=lon_deg,
    )
    provenance.record_fact("ozone_source", f"prescribed:{ozone_file}")
    provenance.record_input(ozone_file)
    if forcing is None:
        forcing = default_forcing(coords.horizontal)
    return forcing.copy(ozone_climatology=climatology)


# ---------------------------------------------------------------------------
# prescribed aerosol emissions
# ---------------------------------------------------------------------------

def _merge_disjoint_emissions(acc, acc_src, new, path):
    """Merge one emission product's variables into ``acc``, rejecting overlaps.

    Emission products are contractually **disjoint**: each ``emissions_file``
    list entry contributes distinct ``emis_<sector>_<species>`` /
    ``aero_emis_<tracer>`` variables (see :func:`_attach_emissions`). Two
    products claiming the SAME variable is ambiguous user intent, so raise a
    build-time ``ValueError`` naming the colliding variable(s) and both
    products — the runner will neither silently keep the last (``dict.update``
    is last-one-wins, the F1 defect) nor invent an additive merge (summing two
    products for one sector/species would silently double-count). ``acc_src``
    maps each already-merged variable to the product that supplied it, purely
    to name both sides of a collision.
    """
    collisions = [v for v in new if v in acc]
    if collisions:
        detail = "; ".join(
            f"{v!r} (from {acc_src[v]!r} and {str(path)!r})"
            for v in collisions)
        raise ValueError(
            "forcing.emissions_file: duplicate emission variable(s) across "
            f"products — {detail}. Each list entry must contribute DISTINCT "
            "emission variables (different sectors/species); two products "
            "claiming the same variable is ambiguous, so the runner neither "
            "silently keeps one nor sums them (which would double-count). "
            "Remove the overlap, or merge the overlapping products offline "
            "into a single file."
        )
    acc.update(new)
    acc_src.update({v: str(path) for v in new})


def _attach_emissions(forcing, forcing_cfg, coords):
    """Attach prescribed aerosol emissions from ``cfg.forcing.emissions_file``.

    No-op when unset. ``emissions_file`` may be a single path or a **list** of
    paths (e.g. one file for biomass burning and one for the rest). Each list
    element is a **product** opened and time-aligned on its own (see
    :func:`_forcing_products`), so a transient ``{year}`` product and a 12-month
    climatology can be mixed without outer-joining their disjoint time axes; the
    per-variable ``TimeSeries`` leaves from all products merge into one
    ``ForcingData``, each keeping its own time axis. The fields auto-route by
    content: variables named ``emis_<sector>_<species>`` drive the bulk /
    in-model-speciated path (``anthropogenic_emissions``); ``aero_emis_<tracer>``
    variables drive the CAM6-faithful pre-speciated path
    (``prescribed_aerosol_emissions``). A file may carry either or both. The
    fields must already be on the model horizontal
    grid — this does **not** regrid (use :mod:`jcm.data.emissions.prepare`
    first); a grid mismatch raises rather than silently zeroing (the emission
    terms fall back to zero on a size mismatch, which from the CLI would look
    like the file "did nothing"). Like ozone, when ``kind: default`` supplies no
    parent struct one is built via ``default_forcing`` so the aquaplanet SST
    climatology is preserved.

    The matching emission term must also be in the physics package (e.g.
    ``physics=echam-jam``) for the fields to be consumed.
    """
    if forcing_cfg is None:
        return forcing
    raw = forcing_cfg.get("emissions_file", None)
    if raw in (None, "", "null"):
        return forcing
    years = forcing_cfg.get("years", None)
    # Per-product coverage: a preset may run its SURFACE forcing past the
    # emissions series' end (``forcing=era5`` runs to 2024 but the mirror's
    # ``emissions_amip`` bundle ends 2022), so honour an
    # ``emissions_available_years`` override — the same mechanism ozone uses —
    # and only fall back to the shared ``available_years`` when it is unset.
    # Without this, ``emissions_file=.../{year}.nc`` would over-expand into
    # never-built 2023/2024 files following the transient-warning's advice.
    available = _product_available_years(
        forcing_cfg, "emissions_available_years")

    from jcm.forcing import (
        default_forcing,
        read_anthropogenic_emissions,
        read_prescribed_aerosol_emissions,
        validate_emissions_grid,
    )

    # Read each product independently and merge the per-variable TimeSeries
    # leaves. Each product keeps its own time axis / align_mode, so mixing a
    # transient product with a climatology does not force one shared axis.
    #
    # This per-product merge is meaningful HERE but not for oxidants (contrast
    # _attach_oxidants, which handles a list as ONE product): emission products
    # carry DISJOINT variables (different ``emis_<sector>_<species>`` /
    # ``aero_emis_<tracer>`` sets), so merging across products unions genuinely
    # distinct keys. That disjointness is a contract, not a hope:
    # ``_merge_disjoint_emissions`` REJECTS a variable supplied by two products
    # (F1) rather than let a plain ``dict.update`` silently keep the last.
    # Oxidant files must each carry the IDENTICAL four gases, so an analogous
    # update would be pure last-one-wins.
    anthro: dict = {}
    speciated: dict = {}
    # Provenance for the disjoint-merge check: which product supplied each
    # already-merged variable, for a precise collision message (F1).
    anthro_src: dict = {}
    speciated_src: dict = {}
    for product in _forcing_products(raw, years, available):
        path = _resolve_data_path(product)
        if path in (None, "", "null"):
            continue
        ds = _open_forcing_dataset(path)
        try:
            a = read_anthropogenic_emissions(ds)
            s = read_prescribed_aerosol_emissions(ds)
        finally:
            ds.close()
        if a is None and s is None:
            raise ValueError(
                f"forcing.emissions_file {path!r} has no emissions variables: "
                "expected ``emis_<sector>_<species>`` (bulk) or "
                "``aero_emis_<tracer>`` (pre-speciated). See the emissions-file "
                "contract in docs/design/jam.md."
            )
        if a:
            _merge_disjoint_emissions(anthro, anthro_src, a, path)
        if s:
            _merge_disjoint_emissions(speciated, speciated_src, s, path)
    if not anthro and not speciated:
        return forcing
    validate_emissions_grid({**anthro, **speciated}, coords, raw)
    if forcing is None:
        forcing = default_forcing(coords.horizontal)
    return forcing.copy(anthropogenic_emissions=anthro or None,
                        prescribed_aerosol_emissions=speciated or None)


def _reject_year_pattern(value, key):
    """Fail loudly on a ``{year}`` pattern for a climatology-only forcing key.

    ``dms_file`` and the ``dust_*`` keys are single-file WRAP_YEAR climatologies
    or static maps: the
    data mirror publishes NO transient (yearly) DMS or dust product, and their
    readers (:func:`jcm.forcing.read_dms_seawater` /
    :func:`jcm.forcing.read_dust_source`) do not expand ``{year}``. A pattern
    here would otherwise reach ``xr.open_dataset`` as a literal-brace path and
    die with a cryptic file-not-found; reject it up front, naming the reason and
    that only ``emissions_file`` / ``oxidants_file`` accept yearly patterns.
    Returns ``value`` unchanged when it is not a pattern.
    """
    if isinstance(value, str) and "{year}" in value:
        raise ValueError(
            f"forcing.{key}={value!r} contains a {{year}} pattern, but {key} "
            "is a climatology-only single file: the data mirror publishes no "
            "transient (yearly) DMS/dust product and its reader does not expand "
            "{year}. Only forcing.emissions_file and forcing.oxidants_file "
            "accept yearly patterns; give dms_file/dust_*_file a single "
            "12-month climatology or static map (or 'auto'/'null').")
    return value


# ---------------------------------------------------------------------------
# natural-emission climatologies (dms / dust)
# ---------------------------------------------------------------------------

def _attach_dms(forcing, forcing_cfg, coords):
    """Attach the seawater-DMS climatology from ``cfg.forcing.dms_file``.

    No-op when unset. Loads a HAMMOZ-style monthly ``DMS_sea (time, lat, lon)``
    climatology (nmol/L, converted to kg/m³ — see
    :func:`jcm.forcing.read_dms_seawater`) as a ``WRAP_YEAR`` ``TimeSeries``
    on ``forcing.dms_seawater``, which :class:`DmsEmissions` consumes. The
    file must already be on the model horizontal grid; lat/lon values are
    validated (a descending-latitude file is flipped) and a mismatch raises —
    the term otherwise falls back to zero on a size mismatch, which from the
    CLI would look like the file "did nothing". Needs a JAM physics package
    (e.g. ``physics=echam-jam``) for the field to be consumed.
    """
    if forcing_cfg is None:
        return forcing
    path = _resolve_data_path(
        _reject_year_pattern(forcing_cfg.get("dms_file", None), "dms_file"))
    if path in (None, "", "null"):
        return forcing
    import xarray as xr

    from jcm.forcing import read_dms_seawater
    lat_deg, lon_deg = _model_latlon_deg(coords)
    with xr.open_dataset(str(path)) as ds:
        ts = read_dms_seawater(ds, lat_deg=lat_deg, lon_deg=lon_deg)
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(dms_seawater=ts)


def _dust_path(forcing_cfg, key):
    """Resolve one dust forcing key, or ``None`` when unset/opted out."""
    path = _resolve_data_path(
        _reject_year_pattern(forcing_cfg.get(key, None), key))
    return None if path in (None, "", "null") else str(path)


def _attach_dust(forcing, forcing_cfg, coords):
    """Attach the five Tegen/HAMMOZ dust inputs (#802).

    No-op when ``dust_file`` is unset. Otherwise loads the monthly effective-LAI
    potential-source climatology as a ``WRAP_YEAR`` ``TimeSeries`` (the Fortran
    steps it by month start, never interpolates) plus the static preferential
    sources, nine soil-texture fractions and categorical tuning regions, all of
    which ``mo_ham_dust.f90`` treats as mandatory — a missing one raises rather
    than letting the scheme run on an all-coarse, untuned soil. The monthly
    satellite roughness map is optional: it is consumed only on the
    ``ndurough = 0`` sensitivity path. Grid handling as in :func:`_attach_dms`.
    """
    if forcing_cfg is None:
        return forcing
    path = _dust_path(forcing_cfg, "dust_file")
    if path is None:
        return forcing
    import xarray as xr

    from jcm.forcing import (read_dust_preferential, read_dust_regions,
                             read_dust_roughness, read_dust_soil_types,
                             read_dust_source)
    lat_deg, lon_deg = _model_latlon_deg(coords)
    companions = {key: _dust_path(forcing_cfg, key)
                  for key in _DUST_REQUIRED_KEYS}
    missing = sorted(k for k, v in companions.items() if v is None)
    if missing:
        raise ValueError(
            f"forcing.dust_file is set but {missing} resolved to nothing. The "
            "Tegen scheme needs the preferential sources, the nine soil-texture "
            "fractions and the tuning regions alongside the potential-source "
            "map; without them it would emit an untuned, all-coarse-soil flux. "
            "Set them (or 'auto'), or disable dust with forcing.dust_file=null.")
    fields = {}
    with xr.open_dataset(path) as ds:
        fields["dust_source"] = read_dust_source(
            ds, lat_deg=lat_deg, lon_deg=lon_deg)
    readers = {"dust_preferential_file": ("dust_preferential",
                                          read_dust_preferential),
               "dust_soil_types_file": ("dust_soil_types", read_dust_soil_types),
               "dust_regions_file": ("dust_regions", read_dust_regions)}
    for key, (field, reader) in readers.items():
        with xr.open_dataset(companions[key]) as ds:
            fields[field] = reader(ds, lat_deg=lat_deg, lon_deg=lon_deg)
    roughness = _dust_path(forcing_cfg, "dust_roughness_file")
    if roughness is not None:
        with xr.open_dataset(roughness) as ds:
            fields["dust_roughness"] = read_dust_roughness(
                ds, lat_deg=lat_deg, lon_deg=lon_deg)
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(**fields)


# ---------------------------------------------------------------------------
# oxidants (shared spectral + pySES resolution)
# ---------------------------------------------------------------------------

def _resolve_oxidant_paths(forcing_cfg):
    """Resolve ``forcing.oxidants_file`` into the single oxidant product's files.

    Shared by the spectral (:func:`_attach_oxidants`) and pySES
    (``jcm.runners._build_pyses_forcing``) paths so ``{year}`` expansion,
    ``hf://`` resolution and the uniform-time-axis validation cannot drift
    between the two — a forked pySES bypass would hand a literal ``{year}``
    brace path (or an unfetched ``hf://`` URL) straight to
    ``xr.open_dataset``. A ``{year}`` pattern expands to the product's yearly
    files; an explicit list is taken verbatim as that one product's file set;
    either way the set is opened together (``open_mfdataset``, by-coords)
    downstream, so :func:`_assert_uniform_time_axis` runs here to reject
    a mixed set up front. Returns a non-empty list of string paths, or ``None``
    when ``oxidants_file`` is unset/empty.
    """
    if forcing_cfg is None:
        return None
    raw = forcing_cfg.get("oxidants_file", None)
    if raw in (None, "", "null"):
        return None
    from omegaconf import ListConfig
    # Per-product coverage, resolved HERE so both the spectral and pySES paths
    # (which share this helper) clamp a transient oxidant ``{year}`` pattern to
    # the same range and cannot drift. The mirror publishes no transient
    # oxidants product, but a user bringing their own series whose coverage
    # differs from the surface forcing's sets ``oxidants_available_years``; it
    # falls back to the shared ``available_years`` when unset.
    files = _resolve_data_path(_expand_years(
        raw, forcing_cfg.get("years", None),
        _product_available_years(forcing_cfg, "oxidants_available_years")))
    if isinstance(files, (list, tuple, ListConfig)):
        paths = [str(p) for p in files
                 if str(p) not in ("", "null", "none", "None")]
    elif files in (None, "", "null"):
        return None
    else:
        paths = [str(files)]
    if not paths:
        return None
    # Oxidants are ONE product (its ``{year}`` expansion or explicit file set),
    # so the whole list is a single product for the guard: a lone climatology
    # or one transient product's yearly files are uniform, while a file set
    # straddling integer-month and datetime axes is rejected as malformed.
    _assert_uniform_time_axis([paths], config_key="forcing.oxidants_file")
    return paths


def _resolve_pyses_emission_paths(forcing_cfg):
    """Resolve ``forcing.emissions_file`` into a flat file list for pySES.

    The pySES column backend opens ALL emission files as ONE combined dataset
    (``jcm.dycore.pyses.forcing.attach_jam_forcing`` → a single
    ``open_mfdataset``, by-coords), so every element — a scalar path, an
    ``hf://`` URL, or a ``{year}`` pattern — is expanded and the result
    FLATTENED into one path list for that single open. A ``{year}`` element
    becomes its product's yearly files (via :func:`_forcing_products`, the same
    expansion the spectral path uses), so both a scalar ``{year}`` pattern and a
    ``{year}`` element *inside a list* resolve to real yearly files rather than a
    literal-brace path reaching ``xr.open_dataset``.

    This deliberately differs from the spectral :func:`_attach_emissions`, which
    opens each list element as an INDEPENDENT product with its own time axis
    (per-product ``TimeSeries`` alignment) and can therefore mix a transient
    ``{year}`` product with a 12-month climatology in one list. pySES has no
    per-product alignment machinery — one combined open along a single time
    axis — so a mixed climatology+transient list cannot be aligned and is
    rejected up front by :func:`_assert_uniform_time_axis` (the honest
    semantics here), exactly as the shared oxidant path does. Returns a
    non-empty list of string paths, or ``None`` when ``emissions_file`` is
    unset/empty.
    """
    if forcing_cfg is None:
        return None
    raw = forcing_cfg.get("emissions_file", None)
    if raw in (None, "", "null"):
        return None
    years = forcing_cfg.get("years", None)
    available = _product_available_years(
        forcing_cfg, "emissions_available_years")
    # Keep each list element / ``{year}`` expansion as its own product for the
    # uniform-time-axis guard (the time axis is a per-product property — see
    # :func:`_product_time_axis`), then flatten to the single path list
    # ``attach_jam_forcing`` opens by-coords.
    products: list[list[str]] = []
    for product in _forcing_products(raw, years, available):
        resolved = _resolve_data_path(product)
        if isinstance(resolved, (list, tuple)):
            files = [str(p) for p in resolved
                     if str(p) not in ("", "null", "none", "None")]
        elif resolved not in (None, "", "null"):
            files = [str(resolved)]
        else:
            files = []
        if files:
            products.append(files)
    if not products:
        return None
    _assert_uniform_time_axis(products, config_key="forcing.emissions_file")
    return [p for product in products for p in product]


def _attach_oxidants(forcing, forcing_cfg, coords):
    """Attach the oxidant climatology from ``cfg.forcing.oxidants_file``.

    No-op when unset. Loads a HAMMOZ/MACC-style monthly
    ``OH/NO3/O3/H2O2_VMR_avrg (time, mlev, lat, lon)`` mole-fraction
    climatology on ECHAM hybrid model levels into ``forcing.oxidant_vmr`` as
    ``WRAP_YEAR`` ``TimeSeries`` leaves; :class:`PrescribedOxidants` converts
    VMR → molec cm⁻³ in-term where T and p are available. The file's levels
    are mapped **one-to-one** onto the model levels: the level count is
    asserted in :func:`jcm.forcing.read_oxidant_vmr`, and when the model runs
    hybrid vertical coordinates the file's ``hyam``/``hybm`` are additionally
    cross-checked against the model's coefficients here, so a file on
    different 47 levels can't be wired in silently. Horizontal grid handling
    as in :func:`_attach_dms`.

    Supports the same yearly-expansion the anthropogenic emissions path does, so
    the year-matched transient oxidant product recommended for a transient run
    (``oxidants_file=.../{year}.nc`` with ``forcing.years``) actually loads: a
    ``{year}`` pattern expands to one file per year, which are concatenated
    along the time axis (``open_mfdataset``, by-coords) and read with ``auto``
    alignment — a single 12-month climatology stays ``WRAP_YEAR`` while a
    multi-year axis becomes ``BY_DATE``. The level-for-level vertical mapping
    is unchanged (the yearly files share the model's hybrid grid).

    Oxidants are handled as **one product**, unlike the per-product emissions
    path. A user-supplied **list** ``oxidants_file`` means the yearly files of a
    *single* oxidant product (exactly what ``{year}`` expansion produces): the
    whole file set is opened together (``open_mfdataset``, by-coords) along one
    time axis and read once. This differs deliberately from
    :func:`_attach_emissions`, whose per-product merge is meaningful because
    emission products carry **disjoint** variables (different sectors/species).
    :func:`jcm.forcing.read_oxidant_vmr` instead requires *every* oxidant file
    to carry **all four** gases (oh/no3/o3/h2o2), so distinct products fully
    overlap — a per-product ``dict.update`` would be pure last-one-wins and
    silently keep only the final file. Genuinely incompatible members in one
    file set (e.g. an integer-month climatology mixed with datetime transients)
    are rejected up front by :func:`_assert_uniform_time_axis` rather
    than left to ``open_mfdataset`` to NaN-fill or clash cryptically.
    """
    if forcing_cfg is None:
        return forcing
    paths = _resolve_oxidant_paths(forcing_cfg)
    if not paths:
        return forcing

    import xarray as xr

    from jcm.forcing import read_oxidant_vmr, validate_oxidant_levels
    lat_deg, lon_deg = _model_latlon_deg(coords)
    nlev = int(coords.nodal_shape[0])

    # ``data_vars="minimal"`` so only the time-dependent VMR fields are
    # concatenated across a multi-year set: the static ``hyam``/``hybm`` hybrid
    # coefficients carry no time axis and must stay 1-D (the default
    # ``data_vars="all"`` would stack them to a spurious ``(nfiles, mlev)`` that
    # breaks ``read_oxidant_vmr`` / ``validate_oxidant_levels``).
    ds = (xr.open_mfdataset(paths, combine="by_coords", data_vars="minimal")
          if len(paths) > 1 else xr.open_dataset(paths[0]))
    ref = paths if len(paths) > 1 else paths[0]
    try:
        mapping = read_oxidant_vmr(ds, nlev=nlev, lat_deg=lat_deg,
                                   lon_deg=lon_deg, align_mode="auto")
        validate_oxidant_levels(ds, coords, ref)
    finally:
        ds.close()
    if not mapping:
        return forcing
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(oxidant_vmr=mapping)


# ---------------------------------------------------------------------------
# merge-compatibility guard (thin adapters over the engine)
# ---------------------------------------------------------------------------

def _product_time_axis(product):
    """One product's combined time axis (thin adapter over the engine).

    See :func:`jcm.data.input_resolution.product_time_axis` — the single home
    for the merge-key computation shared with the Python door.
    """
    return ir.product_time_axis(product)


def _assert_uniform_time_axis(products, *, config_key) -> None:
    """Reject a product set that cannot share one by-coords time axis.

    Thin adapter over :func:`jcm.data.input_resolution.assert_uniform_time_axis`
    — the merge-compatibility rule lives once in the engine (#750/#751).
    """
    ir.assert_uniform_time_axis(products, config_key=config_key)


# ---------------------------------------------------------------------------
# MACv2-SP plume weights
# ---------------------------------------------------------------------------

def _attach_macv2_weights(forcing, forcing_cfg, coords):
    """Attach time-varying MACv2-SP plume weights from ``forcing.macv2_file``.

    No-op when unset. Loads the ``year_weight``/``ann_cycle`` scalings from a
    ``MACv2.0-SP_v1.nc`` file (via :func:`jcm.forcing.read_macv2_weights`) onto
    ``forcing.aerosol_year_weight`` / ``aerosol_ann_cycle`` — the fields the
    MACv2-SP aerosol term reads for per-year amplitude and the seasonal cycle.
    Without a file these default to all-ones (perpetual year-2005 amplitude, no
    seasonal cycle); ``forcing=macv2_sp`` sets the key. ``macv2_file=auto`` (that
    config's default) resolves to the repo-packaged SPv2.1 file
    (:func:`jcm.forcing.packaged_macv2_path`); an explicit path overrides. The
    weights are plume-indexed and grid-independent, so no regridding is needed.
    """
    if forcing_cfg is None:
        return forcing
    raw = forcing_cfg.get("macv2_file", None)
    if raw in (None, "", "null"):
        return forcing
    from jcm.forcing import packaged_macv2_path
    path = _resolve_data_path(
        packaged_macv2_path() if raw == "auto" else raw)
    from jcm.forcing import read_macv2_weights
    year_weight, ann_cycle = read_macv2_weights(str(path))
    forcing = _ensure_parent_forcing(forcing, coords)
    return forcing.copy(aerosol_year_weight=year_weight,
                        aerosol_ann_cycle=ann_cycle)
