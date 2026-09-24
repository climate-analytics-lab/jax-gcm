"""End-to-end mirror build driver (NCAR Glade or DKRZ Levante).

Reproduces every artifact in the Hugging Face dataset from the sources in
``SOURCES.md``::

    python -m jcm.data.mirror.build_mirror --stage all
    python -m jcm.data.mirror.build_mirror --stage sso,bundles

Source roots come from :mod:`jcm.data.mirror.sites` (auto-detected, or
``JCM_MIRROR_SITE``). ``--grids t127,t255`` restricts every stage to a subset
of the published grids — how a new grid is added without rebuilding (or
re-uploading) the existing ones. A site without the RDA ERA5 archive or the raw
input4MIPs emissions streams (Levante) first runs ``--stage pull``, which
fetches the published Tier A products and ``registry.json`` from the mirror so
the per-grid bundles regrid from exactly the data the other grids were built
from::

    python -m jcm.data.mirror.build_mirror --grids t127,t255 \
        --stage pull,sso,ozone,aux,dust,bundles,manifest,registry

Stages: ``pull`` (Tier A + registry from the published mirror), ``sso``,
``era5``, ``ozone``, ``emissions`` (fat-node PBS job
recommended — see ``--help``), ``aux`` (dms/oxidants via
``tools/prep_jam_aux_inputs.py``), ``dust`` (the five Tegen/HAMMOZ
dust inputs), ``bundles``, ``amip`` (yearly
transient forcing/emissions/ozone, ``--years first,last`` — issue #610),
``era5-transient`` (yearly all-ERA5 forcing incl. transient land —
issue #629), ``registry``, ``upload``
(push to the HF dataset; needs ``hf auth login`` with write access).
Outputs land in ``$JCM_MIRROR_ROOT`` (default: the site's scratch ``hf_mirror``):
Tier A under ``build/``, the HF-shaped tree under ``upload/``.

Run from a repo checkout's own directory (``python -m ...`` with the
worktree as cwd): an editable-installed jcm elsewhere shadows
``PYTHONPATH`` and hides this package.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from jcm.data.mirror import sites

#: The mirror's published sets — the source of truth the manifest is generated
#: from (``build_manifest`` expands the product table over these, and the
#: read-side ``mirror_manifest`` view carries them at top level for the resolver
#: + the benchmark prefetch). Declared beside the build loop so the declaration
#: and what actually gets staged cannot drift.
#: ``PUBLISHED_GRIDS`` is the Gaussian-grid whitelist; ``PUBLISHED_LEVELS`` the
#: layer counts carrying level-resolved (oxidant/ozone) bundles; and level-
#: resolved products are only correct on a ``PUBLISHED_VERTICALS`` (hybrid) grid
#: (they are on hybrid-level pressures — a sigma grid sharing a (token, nlev)
#: must not pull one).
#: t127/t255 are ECHAM's own T127/T255 grids: *supported, not validated* — every
#: climatological/static input exists for them, but they are not release-matrix
#: members and nothing is tuned for them (docs/source/science/configurations.md).
PUBLISHED_GRIDS = frozenset({"t63", "t106", "t127", "t255"})
PUBLISHED_LEVELS = frozenset({47, 95})
PUBLISHED_VERTICALS = frozenset({"hybrid"})
#: The grids that also carry the yearly ``{year}`` transient series
#: (``forcing_amip``/``emissions_amip``/``ozone_amip``/``forcing_era5``), a
#: subset of :data:`PUBLISHED_GRIDS`. Transient bundles are tens of GB per grid
#: and are staged per grid deliberately; t127/t255 have not been (#888).
TRANSIENT_GRIDS = frozenset({"t63", "t106"})

# Per-grid Gaussian latitude count. The *set* of published grids is
# :data:`PUBLISHED_GRIDS`; this dict only adds each grid's ``nlat``. Missing an
# entry for a published grid raises loudly below rather than silently skipping.
_NLAT = {"t63": 96, "t106": 160, "t127": 192, "t255": 384}
GRIDS = {grid: _NLAT[grid] for grid in sorted(PUBLISHED_GRIDS)}
assert TRANSIENT_GRIDS <= PUBLISHED_GRIDS

# Column (non-Gaussian) grids that carry a terrain bundle only.
_COLUMN_GRIDS = {"ne30pg3": None}

#: Declarative product table the manifest is generated from (``build_manifest``
#: expands ``{grid}``/``{nlev}`` over the ``PUBLISHED_*`` sets). Fields: ``path``
#: template; ``grids`` (``gaussian``/``gaussian+column``/``transient`` =
#: :data:`TRANSIENT_GRIDS`/``None``=grid-free);
#: ``levels`` (bool = level-resolved on a published vertical); ``coverage``
#: ([first,last] transient series, else ``None``); ``alignment``
#: (transient/climatology/static); ``key`` (forcing knob); ``auto`` (the product
#: ``forcing.<key>=auto`` picks); ``staged`` (on the mirror today).
_MANIFEST_PRODUCTS: tuple[dict, ...] = (
    {"name": "terrain", "path": "bundles/{grid}/terrain.nc",
     "grids": "gaussian+column", "levels": False, "coverage": None,
     "alignment": "static", "key": "terrain_file", "auto": True,
     "staged": True},
    {"name": "forcing_pd", "path": "bundles/{grid}/forcing_pd.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "file", "auto": False,
     "staged": True},
    {"name": "forcing_pi", "path": "bundles/{grid}/forcing_pi.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "file", "auto": False,
     "staged": True},
    {"name": "emissions_pd", "path": "bundles/{grid}/emissions_pd.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "emissions_file", "auto": True,
     "staged": True},
    {"name": "emissions_pi", "path": "bundles/{grid}/emissions_pi.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "emissions_file", "auto": False,
     "staged": True},
    {"name": "dms", "path": "bundles/{grid}/dms.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "dms_file", "auto": True,
     "staged": True},
    # The five Tegen/HAMMOZ dust inputs (#802). They replace the CAM
    # erodibility product that used to sit at ``bundles/{grid}/dust.nc``; that
    # name is retired rather than reused so no warm cache can resolve a CAM
    # geomorphic map into a scheme that expects an effective-LAI fraction.
    {"name": "dust_potential_sources",
     "path": "bundles/{grid}/dust_potential_sources.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "dust_file", "auto": True,
     "staged": True},
    {"name": "dust_preferential_sources",
     "path": "bundles/{grid}/dust_preferential_sources.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "static", "key": "dust_preferential_file", "auto": True,
     "staged": True},
    {"name": "dust_soil_types", "path": "bundles/{grid}/dust_soil_types.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "static", "key": "dust_soil_types_file", "auto": True,
     "staged": True},
    {"name": "dust_regions", "path": "bundles/{grid}/dust_regions.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "static", "key": "dust_regions_file", "auto": True,
     "staged": True},
    {"name": "dust_surface_roughness",
     "path": "bundles/{grid}/dust_surface_roughness.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "dust_roughness_file", "auto": True,
     "staged": True},
    {"name": "ozone_pd", "path": "bundles/{grid}_l{nlev}/ozone_pd.nc",
     "grids": "gaussian", "levels": True, "coverage": None,
     "alignment": "climatology", "key": "ozone_file", "auto": True,
     "staged": True},
    {"name": "ozone_pi", "path": "bundles/{grid}_l{nlev}/ozone_pi.nc",
     "grids": "gaussian", "levels": True, "coverage": None,
     "alignment": "climatology", "key": "ozone_file", "auto": False,
     "staged": True},
    {"name": "oxidants_pd", "path": "bundles/{grid}_l{nlev}/oxidants_pd.nc",
     "grids": "gaussian", "levels": True, "coverage": None,
     "alignment": "climatology", "key": "oxidants_file", "auto": True,
     "staged": True},
    {"name": "oxidants_pi", "path": "bundles/{grid}_l{nlev}/oxidants_pi.nc",
     "grids": "gaussian", "levels": True, "coverage": None,
     "alignment": "climatology", "key": "oxidants_file", "auto": False,
     "staged": True},
    # Yearly transient series. ``coverage`` is the span ACTUALLY on the mirror,
    # NOT the wider raw-source span: the resolver pads a requested range against
    # it and must only name year files that exist. Verified 2026-09-07 against
    # HfApi().list_repo_files — forcing/emissions/ozone AMIP all hold 1950-2022
    # (contiguous), era5 1979-2024. A partial ``--stage amip --years`` build
    # narrows this further via the staging sidecar (see ``build_manifest`` /
    # ``_record_staged_coverage``); ``--stage manifest --verify-remote`` re-checks
    # it against the live mirror.
    {"name": "forcing_amip", "path": "bundles/{grid}/forcing_amip/{year}.nc",
     "grids": "transient", "levels": False, "coverage": [1950, 2022],
     "alignment": "transient", "key": "file", "auto": False, "staged": True},
    {"name": "emissions_amip",
     "path": "bundles/{grid}/emissions_amip/{year}.nc", "grids": "transient",
     "levels": False, "coverage": [1950, 2022], "alignment": "transient",
     "key": "emissions_file", "auto": False, "staged": True},
    {"name": "ozone_amip",
     "path": "bundles/{grid}_l{nlev}/ozone_amip/{year}.nc",
     "grids": "transient", "levels": True, "coverage": [1950, 2022],
     "alignment": "transient", "key": "ozone_file", "auto": False,
     "staged": True},
    {"name": "forcing_era5", "path": "bundles/{grid}/forcing_era5/{year}.nc",
     "grids": "transient", "levels": False, "coverage": [1979, 2024],
     "alignment": "transient", "key": "file", "auto": False, "staged": True},
    # Packaged products (``source: "packaged"``): boundary files shipped in the
    # wheel under ``jcm/data/bc`` (path relative to the ``jcm`` package), NOT on
    # the HF mirror. The engine resolves them via
    # ``jcm.data.input_resolution.resolve_packaged``: a direct file for the
    # grid-independent MACv2-SP plumes; a shape-keyed ``*`` glob for the ozone /
    # terrain climatologies, which have BOTH a packaged variant (tried first, on
    # the grid it was built for) and the mirrored ``ozone_pd`` / ``terrain``
    # variant above (the fallback). ``grids``/``coverage`` are ``None`` (grid-
    # free / no {grid} expansion — the glob spans the packaged dirs itself).
    {"name": "macv2_sp", "path": "data/bc/SPv2.1_18502023_CMIP7.nc",
     "source": "packaged", "grids": None, "levels": False, "coverage": None,
     "alignment": "static", "key": "macv2_file", "auto": False, "staged": True},
    {"name": "ozone_packaged", "path": "data/bc/*/ozone.nc",
     "source": "packaged", "grids": None, "levels": True, "coverage": None,
     "alignment": "climatology", "key": "ozone_file", "auto": False,
     "staged": True},
    # The packaged surface climatologies: ``data/bc/t63/forcing.nc`` (the pySES
    # backend's default ``forcing.file`` and the T63 regression fixtures) and
    # the SPEEDY T30 ``data/bc/t30/clim/forcing.nc`` the getting-started guide
    # loads. Declared here so ``align: auto`` resolves them from their recorded
    # kind rather than from their time axis (#884).
    {"name": "forcing_packaged", "path": "data/bc/*/forcing.nc",
     "source": "packaged", "grids": None, "levels": False, "coverage": None,
     "alignment": "climatology", "key": "file", "auto": False,
     "staged": True},
    {"name": "forcing_packaged_t30", "path": "data/bc/t30/clim/forcing.nc",
     "source": "packaged", "grids": None, "levels": False, "coverage": None,
     "alignment": "climatology", "key": "file", "auto": False,
     "staged": True},
    {"name": "terrain_packaged", "path": "data/bc/*/terrain.nc",
     "source": "packaged", "grids": None, "levels": False, "coverage": None,
     "alignment": "static", "key": "terrain_file", "auto": False,
     "staged": True},
)
SITE = sites.current()
NE30_TOPO = SITE.ne30_topo
GRAV = 9.80665

ROOT = Path(os.environ.get("JCM_MIRROR_ROOT", SITE.default_root))
BUILD = ROOT / "build"
UPLOAD = ROOT / "upload"
GMTED = ROOT / "sources" / "gmted" / "mn30_grd"

#: Sidecar the transient staging steps maintain in the build tree, recording the
#: year range each ``{year}``-series product was ACTUALLY staged over (a per-run
#: ``--years`` choice — the default ``--stage amip`` does not stage the whole
#: source series). ``build_manifest`` folds this in so the coverage the resolver
#: pads against names files that exist on the mirror, not the wider source span.
#: It lives under ``build/`` (not ``upload/``), so it is never pushed to HF; it
#: persists between a staging run and a later ``--stage manifest`` on the same
#: machine. Absent (e.g. an in-repo ``--stage manifest``), coverage falls back to
#: the declared full-mirror span in :data:`_MANIFEST_PRODUCTS` (which equals the
#: full staged range, so the committed no-sidecar manifest already matches HF).
_STAGED_COVERAGE_PATH = BUILD / "staged_coverage.json"


def _load_staged_coverage(path: Path = None) -> dict:
    """Return ``{product_name: [first, last]}`` from the sidecar, or ``{}``."""
    path = path or _STAGED_COVERAGE_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _record_staged_coverage(products, first: int, last: int,
                            path: Path = None) -> None:
    """Merge ``[first, last]`` into the sidecar for each named product.

    Called once per transient staging run. An append that overlaps or is
    adjacent to the recorded span (issue #610 stages contiguous years without
    rewriting history) widens it by ``min``/``max``. A *disjoint* append — one
    that would leave a gap of never-staged years between the two ranges — is
    rejected loudly: ``coverage`` is a single ``[first, last]``, so unioning
    across the gap would advertise year files the mirror never staged and send
    the resolver after them. The operator must fill the gap or stage the
    disjoint ranges into separate mirrors (split coverage the manifest cannot
    represent). Consistent with the manifest's loud-not-silent rule.
    """
    path = path or _STAGED_COVERAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = _load_staged_coverage(path)
    for name in products:
        prev = staged.get(name)
        if prev is None:
            staged[name] = [int(first), int(last)]
            continue
        # Inclusive integer year ranges are contiguous when unioned iff they
        # overlap or touch: max(starts) <= min(ends) + 1. Otherwise the years
        # in (min(ends), max(starts)) were never staged.
        if max(prev[0], first) > min(prev[1], last) + 1:
            gap_lo, gap_hi = min(prev[1], last) + 1, max(prev[0], first) - 1
            raise ValueError(
                f"{name}: staged range [{first}, {last}] is disjoint from the "
                f"recorded [{prev[0]}, {prev[1]}] — years {gap_lo}-{gap_hi} "
                f"would be missing. Fill the gap, or stage the disjoint ranges "
                f"into separate mirrors.")
        staged[name] = [int(min(prev[0], first)), int(max(prev[1], last))]
    path.write_text(json.dumps(staged, indent=2, sort_keys=True) + "\n")


#: The grids this run builds (``--grids``); ``None`` = every published grid plus
#: the column grids. Set from the command line in :func:`main`.
_SELECTED: frozenset | None = None


def _grids(transient: bool = False) -> dict:
    """``{grid: nlat}`` of the Gaussian grids this run builds."""
    pool = TRANSIENT_GRIDS if transient else PUBLISHED_GRIDS
    return {g: n for g, n in GRIDS.items()
            if g in pool and (_SELECTED is None or g in _SELECTED)}


#: Bundle products the ``bundles``/``amip`` stages can (re)build; ``--products``
#: narrows them (e.g. re-deriving only ``emissions`` after a regridding change,
#: without re-publishing files whose content did not change).
BUNDLE_PRODUCTS = ("terrain", "forcing", "emissions", "dms", "ozone",
                   "oxidants")
_PRODUCTS: frozenset | None = None


def _want(product: str) -> bool:
    """Whether this run (re)builds bundle ``product`` (see ``--products``)."""
    return _PRODUCTS is None or product in _PRODUCTS


def _column_selected() -> bool:
    """Whether this run's grid selection includes the column grid (ne30pg3)."""
    return _SELECTED is None or any(g in _SELECTED for g in _COLUMN_GRIDS)


def _column_requested() -> bool:
    """Whether ``--grids`` names the column grid explicitly."""
    return _SELECTED is not None and any(g in _SELECTED for g in _COLUMN_GRIDS)


def _column_buildable() -> bool:
    """Whether the ne30pg3 part of sso/bundles runs here.

    Selected and its CESM topography on this site. Without ``--grids`` a site
    lacking the topography (Levante) builds the Gaussian grids and skips only
    ne30pg3; naming ne30pg3 in ``--grids`` there is refused in check_sources.
    """
    return _column_selected() and NE30_TOPO is not None


def _partial_build() -> bool:
    """Whether this run's upload tree covers only part of the mirror.

    True for a ``--grids`` or ``--products`` build, and for any build whose
    Tier A came from ``--stage pull`` (published already, never re-staged).
    Such a tree must merge its registry onto the published one.
    """
    return (_SELECTED is not None or _PRODUCTS is not None
            or _pulled_tier_a())


def _truncation(grid: str) -> int:
    """``"t127"`` -> 127 — the relation ``jcm.forcing_assembly`` uses."""
    return int(grid[1:])


def stage_sso() -> None:
    """SSO statistics for the Gaussian grids + the native ne30pg3 bundle.

    The ne30 file is assembled directly to its final (Tier B) form: land
    mask from the CESM topo ``LANDFRAC`` (the GMTED validity mask is a
    placeholder — GMTED stores oceans as elevation 0), SSO fields zeroed
    over ocean, and exact GLL-node orography from ``PHIS_gll``.
    """
    import xarray as xr

    from jcm.data.regridding import gaussian_latlon
    from jcm.data.mirror.sso import column_grid_sso, gaussian_grid_sso

    out = BUILD / "sso"
    out.mkdir(parents=True, exist_ok=True)
    for grid, nlat in _grids().items():
        lats, lons = gaussian_latlon(nlat)
        fields = gaussian_grid_sso(str(GMTED), lats, lons)
        xr.Dataset({k: (("lat", "lon"), v) for k, v in fields.items()},
                   coords={"lat": lats, "lon": lons}
                   ).to_netcdf(out / f"sso_gmted2010_{grid}.nc")
        print("sso:", grid, flush=True)

    if not _column_buildable():
        if _column_selected():
            print("sso: skipping ne30pg3 — no CESM ne30 topography on site "
                  f"{SITE.name!r}", flush=True)
        return
    topo = xr.open_dataset(NE30_TOPO)
    fields = column_grid_sso(str(GMTED), topo.lat.values, topo.lon.values)
    # fractional LANDFRAC as lsm (fmask is consumed fractionally); zero
    # SSO only below 10% land so islands keep orography but open-ocean
    # cells drop shoreline-step DEM artifacts
    frac = np.clip(topo.LANDFRAC.values, 0.0, 1.0)
    keep = frac >= 0.1
    ds = xr.Dataset(coords={"lat": ("ncol", topo.lat.values),
                            "lon": ("ncol", topo.lon.values)})
    ds["lsm"] = ("ncol", frac)
    for name in ("orog", "orostd", "orosig", "orogam", "orothe",
                 "oropic", "oroval"):
        ds[name] = ("ncol", np.where(keep, fields[name], 0.0))
    ds["orog_gll"] = ("ncol_gll", topo.PHIS_gll.values / GRAV)
    ds["lat_gll"] = ("ncol_gll", topo.lat_gll.values)
    ds["lon_gll"] = ("ncol_gll", topo.lon_gll.values)
    ds.attrs = {"source": "GMTED2010 30arcsec + CESM ne30 topo (LANDFRAC, "
                          "PHIS_gll)"}
    ds.to_netcdf(out / "sso_gmted2010_ne30pg3.nc")
    print("sso: ne30pg3", flush=True)


def stage_era5() -> None:
    from jcm.data.mirror.era5_land import build_climatology

    ds = build_climatology()
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    BUILD.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(BUILD / "era5_land_climo_2005-2014_0p25.nc", encoding=enc)
    print("era5: done", flush=True)


def stage_ozone() -> None:
    from jcm.data.bc.interpolate_ozone import interpolate_ozone
    from jcm.data.mirror.ozone import load_pd, load_pi, regrid_climatology
    from jcm.data.regridding import gaussian_latlon

    out = BUILD / "ozone"
    out.mkdir(parents=True, exist_ok=True)
    for era, loader in (("pi1850", load_pi), ("pd2005-2014", load_pd)):
        da = loader()
        for grid, nlat in _grids().items():
            ds = regrid_climatology(da, *gaussian_latlon(nlat))
            assert np.isfinite(ds.O3.values).all(), f"NaN in {era}/{grid}"
            plev = out / f"ozone_fzj_cmip7_{era}_{grid}_plev.nc"
            ds.to_netcdf(plev)
            for nlev in (47, 95):
                interpolate_ozone(
                    plev, out / f"ozone_fzj_cmip7_{era}_{grid}_l{nlev}.nc",
                    nlev)
            print("ozone:", era, grid, flush=True)


def stage_emissions() -> None:
    """Multi-GB streaming — run inside a PBS job, not a login node."""
    from jcm.data.mirror.emissions import (SPECIES, build_store,
                                           load_bb_species,
                                           load_ceds_species)
    # A --stage pull symlink points at the climatology-only copy; writing
    # through it would find every species "already built" and skip. Replace
    # it with a real store (the pulled copy stays under build/pulled).
    for name in ("ceds_anthro.zarr", "bb4cmip7.zarr"):
        if (BUILD / name).is_symlink():
            (BUILD / name).unlink()
    build_store(load_ceds_species, SPECIES, str(BUILD / "ceds_anthro.zarr"),
                "CEDS-CMIP-2025-04-18 (input4MIPs CMIP7), sector-summed, "
                "0.5 deg")
    build_store(load_bb_species, SPECIES, str(BUILD / "bb4cmip7.zarr"),
                "DRES-CMIP-BB4CMIP7-2-0 (input4MIPs CMIP7), 0.25 deg")


def stage_aux() -> None:
    """DMS/oxidant matrix via tools/prep_jam_aux_inputs.py."""
    tool = Path(__file__).resolve().parents[3] / "tools" / \
        "prep_jam_aux_inputs.py"
    out = BUILD / "aux"
    for year in (1850, 2005):
        for nlev in (47, 95):
            for trunc in map(_truncation, _grids()):
                subprocess.run(
                    [sys.executable, str(tool), "--year", str(year),
                     "--nlevels", str(nlev), "--oxid-source", "waccm",
                     "--outdir", str(out), "--target-truncation",
                     str(trunc)],
                    check=True)


def stage_dust() -> None:
    """Build the five Tegen/HAMMOZ dust bundles (#802) from the HAMMOZ pool.

    Native at T63/T127/T255, conservatively remapped from the finest native
    file elsewhere (t106), region mask regenerated — see mirror/dust.py.
    """
    from jcm.data.mirror.dust import DUST_PRODUCTS, build_dust_product

    for grid, nlat in _grids().items():
        d = UPLOAD / "bundles" / grid
        d.mkdir(parents=True, exist_ok=True)
        for name in DUST_PRODUCTS:
            build_dust_product(name, nlat, d / f"{name}.nc")
            print("dust:", grid, name, flush=True)
        # The retired CAM erodibility product: drop it from any upload tree
        # built before #802 so a re-upload cannot re-register the old name.
        (d / "dust.nc").unlink(missing_ok=True)


def stage_bundles() -> None:
    from jcm.data.mirror.bundles import (build_emissions_nc, build_forcing,
                                         build_terrain)
    from jcm.data.regridding import gaussian_latlon

    era5 = BUILD / "era5_land_climo_2005-2014_0p25.nc"
    for grid, nlat in _grids().items():
        lats, lons = gaussian_latlon(nlat)
        d = UPLOAD / "bundles" / grid
        d.mkdir(parents=True, exist_ok=True)
        if _want("terrain"):
            build_terrain(str(BUILD / "sso" / f"sso_gmted2010_{grid}.nc"),
                          str(era5), str(d / "terrain.nc"))
        for era in ("pd", "pi"):
            if _want("forcing"):
                build_forcing(str(era5), era, lats, lons,
                              str(d / f"forcing_{era}.nc"))
            if _want("emissions"):
                build_emissions_nc(str(BUILD / "ceds_anthro.zarr"),
                                   str(BUILD / "bb4cmip7.zarr"), era, lats,
                                   lons, str(d / f"emissions_{era}.nc"))

    trunc = {grid: _truncation(grid) for grid in _grids()}
    for grid in _grids():
        for nlev in (47, 95):
            d = UPLOAD / "bundles" / f"{grid}_l{nlev}"
            d.mkdir(parents=True, exist_ok=True)
            for era, tag in (("pi", "pi1850"), ("pd", "pd2005-2014")):
                if _want("ozone"):
                    shutil.copy(BUILD / "ozone" /
                                f"ozone_fzj_cmip7_{tag}_{grid}_l{nlev}.nc",
                                d / f"ozone_{era}.nc")
            for era, year in (("pi", 1850), ("pd", 2005)):
                if _want("oxidants"):
                    shutil.copy(
                        BUILD / "aux" / f"oxidants_waccm_echam_l{nlev}_"
                        f"{year}_t{trunc[grid]}.nc",
                        d / f"oxidants_{era}.nc")
        g = UPLOAD / "bundles" / grid
        if _want("dms"):
            shutil.copy(BUILD / "aux" /
                        f"dms_lana2011_climo_t{trunc[grid]}.nc", g / "dms.nc")

    if not _column_buildable() or not _want("terrain"):
        print("bundles: done", flush=True)
        return
    d = UPLOAD / "bundles" / "ne30pg3"
    d.mkdir(parents=True, exist_ok=True)
    # terrain.nc, matching the Gaussian bundles: the file is the fully
    # assembled terrain (LANDFRAC lsm + orog_gll), and the old sso.nc
    # name invited grabbing a raw SSO product instead (#596)
    shutil.copy(BUILD / "sso" / "sso_gmted2010_ne30pg3.nc",
                d / "terrain.nc")
    # a rerun over a pre-#596 upload tree must not re-register the trap
    # file under its old name
    (d / "sso.nc").unlink(missing_ok=True)
    print("bundles: done", flush=True)


#: Year range for ``--stage amip`` (inclusive); set from ``--years``.
_AMIP_YEARS: tuple[int, int] = (1950, 2022)


def stage_amip() -> None:
    """Yearly transient AMIP bundles (issue #610).

    Per grid and year: ``forcing_amip/<year>.nc`` (tosbcs/siconcbcs +
    land climatology + CR GHGs), ``emissions_amip/<year>.nc`` (transient
    CEDS/BB slices) and ``<grid>_l{47,95}/ozone_amip/<year>.nc`` (FZJ
    monthly on model levels). Not part of ``--stage all`` — run
    explicitly with ``--stage amip --years 1950,2022``. Needs the
    ``era5`` and ``emissions`` stage outputs in ``build/``.
    """
    from jcm.data.bc.interpolate_ozone import interpolate_ozone
    from jcm.data.mirror.amip_yearly import (_TIME_ENC, build_emissions_year,
                                             build_forcing_year,
                                             load_ozone_year,
                                             regrid_ozone_year)
    from jcm.data.regridding import gaussian_latlon

    first, last = _AMIP_YEARS
    if not _grids(transient=True):
        return
    if not any(_want(p) for p in ("forcing", "emissions", "ozone")):
        print("amip: --products names no yearly series (forcing, emissions, "
              "ozone); nothing to build", flush=True)
        return
    era5 = BUILD / "era5_land_climo_2005-2014_0p25.nc"
    scratch = BUILD / "ozone_amip"
    scratch.mkdir(parents=True, exist_ok=True)
    for grid, nlat in _grids(transient=True).items():
        lats, lons = gaussian_latlon(nlat)
        g = UPLOAD / "bundles" / grid
        (g / "forcing_amip").mkdir(parents=True, exist_ok=True)
        (g / "emissions_amip").mkdir(parents=True, exist_ok=True)
        for nlev in (47, 95):
            (UPLOAD / "bundles" / f"{grid}_l{nlev}"
             / "ozone_amip").mkdir(parents=True, exist_ok=True)
        for year in range(first, last + 1):
            if _want("forcing"):
                build_forcing_year(str(era5), year, lats, lons,
                                   str(g / "forcing_amip" / f"{year}.nc"))
            if _want("emissions"):
                build_emissions_year(str(BUILD / "ceds_anthro.zarr"),
                                     str(BUILD / "bb4cmip7.zarr"), year,
                                     lats, lons,
                                     str(g / "emissions_amip" / f"{year}.nc"))
            if _want("ozone"):
                plev = scratch / f"ozone_{grid}_{year}_plev.nc"
                regrid_ozone_year(load_ozone_year(year), lats,
                                  lons).to_netcdf(plev, encoding=_TIME_ENC)
                for nlev in (47, 95):
                    interpolate_ozone(
                        plev,
                        UPLOAD / "bundles" / f"{grid}_l{nlev}" / "ozone_amip"
                        / f"{year}.nc", nlev)
            print("amip:", grid, year, flush=True)
    # Record the span actually staged for each series built, so the manifest
    # coverage names files that exist, not the wider source series.
    _record_staged_coverage(
        tuple(f"{p}_amip" for p in ("forcing", "emissions", "ozone")
              if _want(p)), first, last)


def stage_era5_transient() -> None:
    """Yearly all-ERA5 transient bundles (issue #629).

    Per year: Tier A intermediates under ``build/`` — the 6-hourly
    SST/ice reduction (``era5_sstice/<year>.nc``, the expensive part:
    ~80 GB streamed per year) and the 13-month land means
    (``era5_land_transient/<year>.nc``) — then per grid
    ``bundles/<grid>/forcing_era5/<year>.nc``. Intermediates are
    written atomically and reused, so a killed job resumes where it
    stopped. Not part of ``--stage all``; run explicitly, e.g.
    ``--stage era5-transient --years 1979,2024``. Needs the ``era5``
    stage climatology in ``build/``.
    """
    import xarray as xr

    from jcm.data.mirror.era5_yearly import (build_forcing_year,
                                             build_land_year,
                                             build_sstice_year)
    from jcm.data.regridding import gaussian_latlon

    first, last = _AMIP_YEARS
    if not _grids(transient=True):
        return
    clim = BUILD / "era5_land_climo_2005-2014_0p25.nc"
    tiers = {"era5_sstice": build_sstice_year,
             "era5_land_transient": build_land_year}
    for name, builder in tiers.items():
        (BUILD / name).mkdir(parents=True, exist_ok=True)
    for year in range(first, last + 1):
        for name, builder in tiers.items():
            path = BUILD / name / f"{year}.nc"
            if path.exists():
                continue
            ds = builder(year)
            enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            tmp = path.with_suffix(".tmp.nc")
            ds.to_netcdf(tmp, encoding=enc)
            tmp.rename(path)
            print("era5-transient:", name, year, flush=True)
    for grid, nlat in _grids(transient=True).items():
        lats, lons = gaussian_latlon(nlat)
        g = UPLOAD / "bundles" / grid / "forcing_era5"
        g.mkdir(parents=True, exist_ok=True)
        for year in range(first, last + 1):
            build_forcing_year(
                str(clim), str(BUILD / "era5_sstice" / f"{year}.nc"),
                year, lats, lons, str(g / f"{year}.nc"),
                land=xr.open_dataset(
                    BUILD / "era5_land_transient" / f"{year}.nc"))
    _record_staged_coverage(("forcing_era5",), first, last)


def build_manifest(staged_coverage: dict = None) -> dict:
    """Assemble the mirror-manifest dict from :data:`_MANIFEST_PRODUCTS`.

    Expands each row's ``{grid}``/``{nlev}`` template against the published grid
    (:data:`GRIDS` + column grids), level (:data:`PUBLISHED_LEVELS`) and vertical
    (:data:`PUBLISHED_VERTICALS`) sets so the availability knowledge the resolver
    consults is generated, never hand-listed. The published sets (:data:`
    PUBLISHED_GRIDS` etc.) are declared here so this and the read-side manifest
    view cannot drift.

    ``coverage`` for a ``{year}``-series product is the ACTUAL staged span when
    the staging sidecar records it (``staged_coverage``, default
    :data:`_STAGED_COVERAGE_PATH`), else the declared full-mirror span in
    :data:`_MANIFEST_PRODUCTS` (which equals the committed mirror's range). The
    distinction matters because the resolver pads a requested range by a year on
    each side clipped to ``coverage`` (for the ``by_date_interp`` bracket): if
    that over-advertised while ``--stage amip --years 1950,2000`` staged only a
    subset, the pad would fetch a year file that was never built. Regenerating
    the manifest right after a partial staging (same machine, sidecar present)
    keeps the two honest; an in-repo regeneration with no sidecar reproduces the
    committed full-mirror-coverage manifest unchanged.
    """
    from jcm.data.remote import DEFAULT_REPO

    if staged_coverage is None:
        staged_coverage = _load_staged_coverage()
    gaussian = sorted(GRIDS)
    products = {}
    for row in _MANIFEST_PRODUCTS:
        if row["grids"] == "gaussian":
            grids = gaussian
        elif row["grids"] == "transient":
            grids = sorted(TRANSIENT_GRIDS)
        elif row["grids"] == "gaussian+column":
            grids = gaussian + sorted(_COLUMN_GRIDS)
        else:  # None -> grid-free single file
            grids = None
        products[row["name"]] = {
            "path": row["path"],
            # ``source`` selects the resolver: "mirror" (default) fetches the HF
            # bundle, "packaged" reads a wheel-shipped file (resolve_packaged).
            "source": row.get("source", "mirror"),
            "grids": grids,
            "levels": sorted(PUBLISHED_LEVELS) if row["levels"] else None,
            "vertical": (sorted(PUBLISHED_VERTICALS)[0]
                         if row["levels"] else None),
            "coverage": staged_coverage.get(row["name"], row["coverage"]),
            "alignment": row["alignment"],
            "key": row["key"],
            "auto": row["auto"],
            "staged": row["staged"],
        }
    return {
        "schema_version": 1,
        "repo": DEFAULT_REPO,
        "grids": {**GRIDS, **_COLUMN_GRIDS},
        "levels": sorted(PUBLISHED_LEVELS),
        "verticals": sorted(PUBLISHED_VERTICALS),
        "products": products,
    }


#: Packaged manifest path (``jcm/data/mirror_manifest.json``), refreshed in-repo
#: by ``stage_manifest`` so a mirror change regenerates it mechanically.
_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "mirror_manifest.json"


def stage_manifest() -> None:
    """(Re)generate the packaged ``jcm/data/mirror_manifest.json``.

    Pure metadata — no Glade source — so it runs anywhere (see ``check_sources``).
    """
    _MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    print("manifest:", _MANIFEST_PATH, flush=True)


def _product_variants(rec: dict):
    """Concrete ``(grid, nlev)`` variants a product declares.

    ``grid`` ranges over ``rec['grids']`` (``None`` for a grid-free single file,
    which yields the lone ``(None, None)`` variant); ``nlev`` over
    ``rec['levels']`` when the product is level-resolved, else the single
    ``None``. Each is a distinct artifact the mirror is expected to hold —
    verification checks every one rather than pooling them, so a complete grid
    cannot mask a missing sibling variant.
    """
    for grid in rec["grids"] or (None,):
        for nlev in rec["levels"] or (None,):
            yield (grid, nlev)


def _variant_label(variant) -> str:
    """``(grid, nlev)`` -> the label used in drift keys, e.g. ``t63`` / ``t63_l47``.

    A grid-free variant (``(None, None)``) has no per-variant suffix, so the
    product name alone is the key (see ``_variant_key``).
    """
    grid, nlev = variant
    if grid is None:
        return ""
    return grid if nlev is None else f"{grid}_l{nlev}"


def _variant_key(name: str, variant) -> str:
    """Drift-dict key for a product variant: ``name[label]``, or ``name`` grid-free."""
    label = _variant_label(variant)
    return f"{name}[{label}]" if label else name


def remote_transient_coverage(files, manifest: dict) -> dict:
    """Sorted list of staged years per ``{year}`` product **and grid/level variant**.

    Pure (no network): matches each mirror-relative path in ``files`` against the
    manifest's ``{year}`` path templates, capturing ``{grid}``/``{nlev}`` so the
    years are resolved per concrete ``(grid, nlev)`` variant (``nlev`` is ``None``
    for grid-only products). Returns ``{product: {(grid, nlev): [years...]}}`` with
    the years sorted ascending; a variant with no staged year is simply absent
    from its inner dict. The FULL year list (not a reduced ``[first, last]`` span)
    is returned so :func:`verify_remote_coverage` can catch an interior hole — a
    variant that keeps its endpoints but drops a middle year. Split out so the
    path-to-years logic is testable without hitting the Hub, and kept per-variant
    because pooling all variants under the product would let a complete grid hide
    a missing one.
    """
    import re
    from collections import defaultdict

    years: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    patterns = {
        name: re.compile(
            "^" + re.escape(rec["path"]).replace(r"\{grid\}", r"(?P<grid>[^/]+)")
            .replace(r"\{nlev\}", r"(?P<nlev>\d+)")
            .replace(r"\{year\}", r"(?P<year>\d{4})") + "$")
        for name, rec in manifest["products"].items() if "{year}" in rec["path"]}
    for f in files:
        for name, pat in patterns.items():
            m = pat.match(f)
            if m:
                gd = m.groupdict()
                variant = (gd["grid"],
                           int(gd["nlev"]) if gd.get("nlev") else None)
                years[name][variant].add(int(gd["year"]))
                break
    return {name: {variant: sorted(ys)
                   for variant, ys in years.get(name, {}).items()}
            for name in patterns}


def _bounded_years(years, cap: int = 12):
    """Bounded rendering of a missing-year list for drift output.

    All years when few; otherwise the first ``cap`` plus a ``"+N more"`` sentinel,
    so a large contiguous hole stays readable in the reported drift.
    """
    years = list(years)
    if len(years) <= cap:
        return years
    return [*years[:cap], f"+{len(years) - cap} more"]


def verify_remote_coverage(manifest: dict = None, repo_id: str = None) -> dict:
    """Cross-check every declared mirror artifact against the live mirror.

    Lists the dataset repo (``HfApi().list_repo_files``) once and checks two
    things, returning a drift dict (empty == no drift) whose entries the CLI
    prints and exits non-zero on:

    * **Transient ``{year}`` series** — the real staged years per product **and
      grid/level variant** (:func:`remote_transient_coverage`) vs the manifest
      ``coverage``. Checked for CONTIGUITY, not just endpoints: a variant whose
      ``[first, last]`` match but which drops an interior year is flagged with an
      extra ``"missing"`` list (bounded for large holes) — otherwise clients
      trust the contiguous span and later request a file that is not there.
    * **Static / climatology artifacts** — every ``staged: true`` product with no
      ``{year}`` template must have each declared ``(grid, nlev)`` variant file
      present on the mirror; a missing one reports ``"remote": None``. Without
      this, a ``staged`` flag could advertise a file the resolver then 404s on.
      Unstaged products are skipped — their absence is intentional and the
      resolver already gives a precise not-yet-published error.

    Checking each variant — not one pooled span per product — is what keeps a
    complete grid from masking a missing sibling (e.g. a full t63 series hiding an
    absent t106). Network-only; driven by ``--stage manifest --verify-remote``.
    """
    from huggingface_hub import HfApi

    from jcm.data import mirror_manifest as mm
    from jcm.data.remote import DEFAULT_REPO

    if manifest is None:
        manifest = build_manifest()
    repo_id = repo_id or manifest.get("repo", DEFAULT_REPO)
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    file_set = set(files)
    coverage = remote_transient_coverage(files, manifest)
    drift = {}
    for name, rec in manifest["products"].items():
        # Packaged products ship in the wheel, not on the mirror — nothing to
        # cross-check against list_repo_files.
        if rec.get("source") == "packaged":
            continue
        if "{year}" in rec["path"]:
            declared = rec["coverage"]
            remote_variants = coverage.get(name, {})
            for variant in _product_variants(rec):
                years = remote_variants.get(variant)
                key = _variant_key(name, variant)
                if years is None:
                    drift[key] = {"manifest": declared, "remote": None}
                    continue
                span = [years[0], years[-1]]
                # Interior holes: years absent WITHIN the actual span (endpoint
                # truncation is already conveyed by span != declared).
                missing = sorted(set(range(span[0], span[1] + 1)) - set(years))
                if span == declared and not missing:
                    continue
                entry = {"manifest": declared, "remote": span}
                if missing:
                    entry["missing"] = _bounded_years(missing)
                drift[key] = entry
        elif rec["staged"]:
            # Static/climatology artifact: every declared variant file must exist.
            for variant in _product_variants(rec):
                grid, nlev = variant
                rel = mm.bundle_path(manifest, name, grid, nlev)
                if rel not in file_set:
                    drift[_variant_key(name, variant)] = {
                        "manifest": rel, "remote": None}
    return drift


def stage_registry() -> None:
    """Stage Tier A into the upload tree and write ``registry.json``.

    A full build (every grid and product, Tier A built here) stages Tier A
    and writes the registry from its own tree, so a file it no longer produces
    drops out. A partial build (``_partial_build``: ``--grids``,
    ``--products`` or pulled Tier A) stages no Tier A — it is what the new
    bundles were regridded *from*, already published, and restaging a locally
    rebuilt copy would republish GB of unchanged data — and merges its hashes
    onto the pulled published registry, since a registry built from a partial
    tree alone would drop every other file's entry. SSO statistics (the
    terrain product's Tier A) are staged for the selected grids when terrain is
    among the products built.
    """
    from jcm.data.mirror.registry import write_registry

    partial = _partial_build()
    if not partial:
        for name in ("ceds_anthro.zarr", "bb4cmip7.zarr",
                     "era5_land_climo_2005-2014_0p25.nc"):
            src, dst = BUILD / name, UPLOAD / "products" / name
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst) if src.is_dir() else shutil.copy(
                    src, dst)
    if _want("terrain"):
        sso_dst = UPLOAD / "products" / "sso"
        sso_dst.mkdir(parents=True, exist_ok=True)
        for f in (BUILD / "sso").glob("*.nc"):
            if (_SELECTED is not None
                    and f.stem.rsplit("_", 1)[-1] not in _SELECTED):
                continue
            dst = sso_dst / f.name
            # staging may have hardlinked build -> upload already
            if not (dst.exists() and dst.samefile(f)):
                shutil.copy(f, dst)
    base = None
    if partial:
        if not _REMOTE_REGISTRY.exists():
            sys.exit("registry: a partial build (--grids / --products / pulled "
                     "Tier A) must merge onto the published registry.json — "
                     "run --stage pull first.")
        base = json.loads(_REMOTE_REGISTRY.read_text())
    print(write_registry(str(UPLOAD), base=base), flush=True)


def _stage_sources(site: sites.Site = None) -> dict[str, tuple]:
    """``{stage: ((label, path | None), ...)}`` — what each stage reads on ``site``.

    Checked so a wrong machine or an unmounted filesystem fails in seconds with
    a clear list, not hours in with an obscure I/O error. ``None`` is a source
    the site does not provide at all. Paths under the mirror root (``build/``)
    are produced by earlier stages and are checked just before their consumer
    runs (see :func:`check_sources`), so a one-shot ``--stage pull,...,bundles``
    on a fresh root is not refused up front.
    """
    site = site or SITE
    i4m = f"{site.input4mips}/CMIP7/CMIP"
    rda = site.rda
    era5_moda = f"{rda}/d633001/e5.moda.an.sfc" if rda else None
    hammoz = site.hammoz
    dust = ([("ECHAM-HAMMOZ pool", f"{hammoz}/{rel}")
             for rel in _dust_source_files()]
            if hammoz else [("ECHAM-HAMMOZ pool", None)])
    oxid = [(f"WACCM CCMI REFC1 oxidants {d}-{d + 9}",
             f"{site.waccm_oxidants}/oxid_ozone_WACCM_CCMI_REFC1_"
             f"f.e11.FWTREFC1.{d}-{d + 9}.f19_f19.ccmi34.001_monthly.nc")
            for d in (1850, 2000)]
    era5_climo = ("Tier A ERA5 land climatology",
                  str(BUILD / "era5_land_climo_2005-2014_0p25.nc"))
    ceds = ("Tier A CEDS store", str(BUILD / "ceds_anthro.zarr"))
    pcmdi = ("PCMDI AMIP SST/ice (input4MIPs)", f"{i4m}/PCMDI/PCMDI-AMIP-1-1-10")
    bb = ("Tier A BB4CMIP7 store", str(BUILD / "bb4cmip7.zarr"))
    return {
        "pull": (),
        "sso": (("GMTED2010 DEM", str(GMTED)),
                *((("CESM ne30 topography", site.ne30_topo),)
                  if _column_requested()
                  or (_column_selected() and site.ne30_topo) else ())),
        "era5": (("RDA ERA5 monthly means", era5_moda),),
        "ozone": (("FZJ ozone (input4MIPs)", f"{i4m}/FZJ/FZJ-CMIP-ozone-1-0"),),
        "emissions": (
            ("CEDS (input4MIPs)", f"{i4m}/PNNL-JGCRI/CEDS-CMIP-2025-04-18"),
            ("BB4CMIP7 (input4MIPs)", f"{i4m}/DRES/DRES-CMIP-BB4CMIP7-2-0")),
        "aux": (("Lana DMS (CESM inputdata)",
                 f"{site.cesm_inputdata}/atm/cam/chem/ocnexch/"
                 "Csw_DMS_Lana2011_f09f09_1750_2100_20200717a.nc"), *oxid),
        "dust": tuple(dust),
        "bundles": tuple(
            src for product, srcs in (
                ("terrain", (era5_climo,
                             ("SSO statistics", str(BUILD / "sso")))),
                ("forcing", (pcmdi, era5_climo)),
                ("emissions", (ceds, bb)),
                ("ozone", (("ozone stage output", str(BUILD / "ozone")),)),
                ("oxidants", (("aux stage output", str(BUILD / "aux")),)),
                ("dms", (("aux stage output", str(BUILD / "aux")),)))
            if _want(product) for src in srcs),
        "amip": tuple(
            src for product, srcs in (
                ("forcing", (pcmdi, era5_climo,
                             ("CR-CMIP GHGs (input4MIPs)",
                              f"{i4m}/CR/CR-CMIP-1-0-0"))),
                ("emissions", (ceds, bb)),
                ("ozone", (("FZJ ozone (input4MIPs)",
                            f"{i4m}/FZJ/FZJ-CMIP-ozone-1-0"),)))
            if _want(product) for src in srcs),
        "era5-transient": (
            ("RDA ERA5 6-hourly analyses",
             f"{rda}/d633000/e5.oper.an.sfc" if rda else None),
            ("RDA ERA5 monthly means", era5_moda),
            ("CR-CMIP GHGs (input4MIPs)", f"{i4m}/CR/CR-CMIP-1-0-0"),
            era5_climo),
        "registry": (("upload tree", str(UPLOAD)),),
    }


def _dust_source_files() -> list[str]:
    """Every pool-relative HAMMOZ file the dust stage may read.

    Not narrowed by ``--grids``: a grid HAMMOZ does not ship reads the finest
    native file, so checking the whole set is the simple, safe superset.
    """
    from jcm.data.mirror.dust import NATIVE_SOURCES

    return sorted({rel for table in NATIVE_SOURCES.values()
                   for product in table.values()
                   for rel, _ in product.values()})


#: Where to point a user whose site lacks a source, by source label prefix.
_UNAVAILABLE_HINTS = {
    "RDA ERA5": "--stage pull fetches the published ERA5 Tier A instead",
    "ECHAM-HAMMOZ": ("build dust where the HAMMOZ pool is mounted (Levante) "
                     "or set JCM_HAMMOZ_DIR to a copy of it"),
    "CESM ne30": "exclude ne30pg3 with --grids",
}


def _unavailable(stage_names) -> dict[str, list[str]]:
    """``{stage: [labels]}`` of sources this site does not provide at all."""
    table = _stage_sources()
    out = {}
    for name in stage_names:
        labels = [label for label, p in table.get(name, ()) if p is None]
        if labels:
            out[name] = labels
    return out


def check_sources(stage_names, *, include_build: bool = False) -> None:
    """Fail when the sources for the requested stages are absent here.

    Up front (``include_build=False``) only external sources are checked;
    each stage re-checks with ``include_build=True`` just before it runs, when
    the build-tree outputs of earlier stages must exist. Source-free stages
    (``manifest``; ``pull`` — network) run anywhere.
    """
    unavailable = _unavailable(stage_names)
    if unavailable:
        lines = []
        for stage, labels in sorted(unavailable.items()):
            for label in labels:
                hint = next((h for k, h in _UNAVAILABLE_HINTS.items()
                             if label.startswith(k)), "")
                lines.append(f"{stage}: {label}" + (f" — {hint}" if hint else ""))
        sys.exit(f"Site {SITE.name!r} does not provide (see "
                 "jcm/data/mirror/sites.py):\n  " + "\n  ".join(lines))
    table = _stage_sources()

    def produced_here(p) -> bool:
        path = Path(p)
        return path.is_relative_to(BUILD) or path.is_relative_to(UPLOAD)

    missing = [f"{name}: {label} ({p})" for name in stage_names
               for label, p in table.get(name, ())
               if (include_build or not produced_here(p))
               and not Path(p).exists()]
    if missing:
        sys.exit(f"Missing sources on site {SITE.name!r} "
                 "(see jcm/data/mirror/SOURCES.md):\n  "
                 + "\n  ".join(sorted(set(missing))))
    if ("amip" in stage_names and include_build and _want("emissions")
            and _pulled_emissions()):
        sys.exit("amip: build/ceds_anthro.zarr is the --stage pull copy, which "
                 "carries only the PI/PD climatology arrays; the yearly slices "
                 "would read unfilled chunks. Build the stores with --stage "
                 "emissions (or pull them whole) first.")


def _pulled(names) -> bool:
    """Whether any of the build-tree Tier A ``names`` is a --stage pull copy."""
    pulled = (BUILD / "pulled").resolve()
    return any((BUILD / n).is_symlink()
               and pulled in (BUILD / n).resolve().parents for n in names)


def _pulled_emissions() -> bool:
    """Whether the build tree's emissions stores are the partial pulled copies."""
    return _pulled(("ceds_anthro.zarr", "bb4cmip7.zarr"))


def _pulled_tier_a() -> bool:
    """Whether any build-tree Tier A product came from --stage pull."""
    return _pulled(("ceds_anthro.zarr", "bb4cmip7.zarr",
                    "era5_land_climo_2005-2014_0p25.nc"))


#: Tier A products the per-grid bundles regrid from. ``stage_pull`` fetches
#: them from the published mirror; for the emissions stores only the PI/PD
#: climatology arrays (``*_clim``) plus coordinates and metadata are needed.
_TIER_A_PULL = (
    "products/era5_land_climo_2005-2014_0p25.nc",
    "products/ceds_anthro.zarr/zarr.json",
    "products/ceds_anthro.zarr/*_clim/**",
    "products/bb4cmip7.zarr/zarr.json",
    "products/bb4cmip7.zarr/*_clim/**",
    *(f"products/{store}/{coord}/**"
      for store in ("ceds_anthro.zarr", "bb4cmip7.zarr")
      for coord in ("lat", "lon", "month", "time")),
    "registry.json",
)

#: The published ``registry.json`` as pulled, merged into by ``stage_registry``
#: when the upload tree holds only some grids (a ``--grids`` build).
_REMOTE_REGISTRY = BUILD / "remote_registry.json"


def stage_pull() -> None:
    """Fetch the published Tier A products and ``registry.json`` into ``build/``.

    For a site that cannot rebuild Tier A (no RDA ERA5, no raw input4MIPs
    emission streams) and for any ``--grids`` build that adds a grid: the new
    bundles then regrid from exactly the Tier A data the published grids were
    built from, and the registry is merged rather than rewritten.
    """
    from huggingface_hub import snapshot_download

    from jcm.data.remote import DEFAULT_REPO

    stage = BUILD / "pulled"
    snapshot_download(repo_id=DEFAULT_REPO, repo_type="dataset",
                      allow_patterns=list(_TIER_A_PULL), local_dir=str(stage))
    for name in ("era5_land_climo_2005-2014_0p25.nc", "ceds_anthro.zarr",
                 "bb4cmip7.zarr"):
        dst = BUILD / name
        if not dst.exists():
            dst.symlink_to(stage / "products" / name)
    shutil.copy(stage / "registry.json", _REMOTE_REGISTRY)
    print("pull: done", flush=True)


def stage_upload() -> None:
    """Push the upload tree to the HF dataset (needs a write token).

    Retries transient backend failures: the xet upload pipeline has
    aborted mid-transfer with TimeoutError("error decoding response
    body") on a 44k-file push — uploads are resumable, so committed
    files are skipped on the next attempt.
    """
    import time

    from huggingface_hub import HfApi

    from jcm.data.remote import DEFAULT_REPO

    api = HfApi()
    last = None
    for attempt in range(1, 6):
        print(f"upload attempt {attempt}", flush=True)
        try:
            api.upload_folder(repo_id=DEFAULT_REPO, repo_type="dataset",
                              folder_path=str(UPLOAD),
                              commit_message="Mirror update via "
                                             "build_mirror --stage upload")
            print("upload: done", flush=True)
            return
        except Exception as e:                      # noqa: BLE001
            last = e
            print(f"upload attempt {attempt} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            time.sleep(60)
    raise RuntimeError("upload failed after 5 attempts") from last


STAGES = {"pull": stage_pull, "sso": stage_sso, "era5": stage_era5, "ozone": stage_ozone,
          "emissions": stage_emissions, "aux": stage_aux, "dust": stage_dust,
          "bundles": stage_bundles, "amip": stage_amip,
          "era5-transient": stage_era5_transient,
          "manifest": stage_manifest,
          "registry": stage_registry, "upload": stage_upload}

#: Heavy / source-gated opt-in stages excluded from ``--stage all``: multi-GB
#: builds or the push to the remote.
_NOT_IN_ALL = ("pull", "amip", "era5-transient", "upload")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    help="comma-separated stage list, or 'all' "
                         f"({', '.join(STAGES)}; 'all' excludes "
                         f"{', '.join(_NOT_IN_ALL)})")
    ap.add_argument("--grids", default=None,
                    help="comma-separated subset of the published grids to "
                         f"build ({', '.join(sorted({**GRIDS, **_COLUMN_GRIDS}))}"
                         "); default all")
    ap.add_argument("--products", default=None,
                    help="comma-separated subset of the bundle products the "
                         f"bundles/amip stages build ({', '.join(BUNDLE_PRODUCTS)}"
                         "); default all")
    ap.add_argument("--years", default="1950,2022",
                    help="inclusive year range for --stage amip, "
                         "e.g. 1950,2022")
    ap.add_argument("--verify-remote", action="store_true",
                    help="after staging, cross-check the manifest against the "
                         "live mirror (list_repo_files): transient coverage "
                         "per variant + existence of every staged static "
                         "artifact; exit non-zero on any drift")
    args = ap.parse_args()
    global _AMIP_YEARS, _SELECTED, _PRODUCTS
    first, last = (int(y) for y in args.years.split(","))
    _AMIP_YEARS = (first, last)
    if args.grids:
        _SELECTED = frozenset(args.grids.split(","))
        unknown = sorted(_SELECTED - set(GRIDS) - set(_COLUMN_GRIDS))
        if unknown:
            sys.exit(f"Unknown grid(s) {unknown}; published: "
                     f"{', '.join(sorted({**GRIDS, **_COLUMN_GRIDS}))}")
    if args.products:
        _PRODUCTS = frozenset(args.products.split(","))
        unknown = sorted(_PRODUCTS - set(BUNDLE_PRODUCTS))
        if unknown:
            sys.exit(f"Unknown product(s) {unknown}; valid: "
                     f"{', '.join(BUNDLE_PRODUCTS)}")
    names = ([n for n in STAGES if n not in _NOT_IN_ALL]
             if args.stage == "all" else args.stage.split(","))
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s) {unknown}; valid: {', '.join(STAGES)}")
    if args.stage == "all":
        # 'all' means everything THIS site can build; an explicitly named stage
        # whose sources are absent still fails in check_sources below.
        for name, labels in sorted(_unavailable(names).items()):
            hints = sorted({h for k, h in _UNAVAILABLE_HINTS.items()
                            for label in labels if label.startswith(k)})
            print(f"skipping stage {name}: site {SITE.name!r} does not "
                  f"provide {', '.join(labels)}"
                  + (f" ({'; '.join(hints)})" if hints else ""), flush=True)
            names.remove(name)
    transient = [n for n in names if n in ("amip", "era5-transient")]
    if transient and _SELECTED is not None \
            and not TRANSIENT_GRIDS <= _SELECTED:
        # Staged coverage is recorded per product, not per grid: a partial
        # transient build would advertise year files on the grids it skipped.
        sys.exit(f"{transient} must build every transient grid "
                 f"({', '.join(sorted(TRANSIENT_GRIDS))}) — include them all "
                 "in --grids or omit --grids.")
    BUILD.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    check_sources(names)
    for name in names:
        print(f"=== stage: {name} ===", flush=True)
        check_sources([name], include_build=True)
        STAGES[name]()
    if args.verify_remote:
        print("=== verify-remote ===", flush=True)
        drift = verify_remote_coverage()
        for name, d in sorted(drift.items()):
            print(f"DRIFT {name}: manifest={d['manifest']} "
                  f"remote={d['remote']}", flush=True)
        if drift:
            sys.exit("manifest disagrees with the live mirror")
        print("verify-remote: manifest matches the mirror "
              "(coverage + static existence)", flush=True)


if __name__ == "__main__":
    main()
