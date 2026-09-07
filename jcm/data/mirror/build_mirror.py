"""End-to-end mirror build driver (runs on NCAR Glade only).

Reproduces every artifact in the Hugging Face dataset from the sources in
``SOURCES.md``::

    python -m jcm.data.mirror.build_mirror --stage all
    python -m jcm.data.mirror.build_mirror --stage sso,bundles

Stages: ``sso``, ``era5``, ``ozone``, ``emissions`` (fat-node PBS job
recommended — see ``--help``), ``aux`` (dms/dust/oxidants via
``tools/prep_jam_aux_inputs.py``), ``bundles``, ``amip`` (yearly
transient forcing/emissions/ozone, ``--years first,last`` — issue #610),
``era5-transient`` (yearly all-ERA5 forcing incl. transient land —
issue #629), ``registry``, ``upload``
(push to the HF dataset; needs ``hf auth login`` with write access).
Outputs land in ``$JCM_MIRROR_ROOT`` (default ``$SCRATCH/hf_mirror``):
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

from jcm.data.bundle_names import (PUBLISHED_GRIDS, PUBLISHED_LEVELS,
                                   PUBLISHED_VERTICALS)

# Per-grid Gaussian latitude count. The *set* of published grids is owned by
# ``jcm.data.bundle_names.PUBLISHED_GRIDS`` (the whitelist the runner's ``auto``
# resolver and the benchmark prefetch both consult) so the build loop here and
# that resolver cannot drift; this dict only adds each grid's ``nlat``. Missing
# an entry for a published grid raises loudly below rather than silently
# skipping it.
_NLAT = {"t63": 96, "t106": 160}
GRIDS = {grid: _NLAT[grid] for grid in sorted(PUBLISHED_GRIDS)}

# Column (non-Gaussian) grids that carry a terrain bundle only.
_COLUMN_GRIDS = {"ne30pg3": None}

#: Declarative product table the manifest is generated from (``build_manifest``
#: expands ``{grid}``/``{nlev}`` over the ``PUBLISHED_*`` sets). Fields: ``path``
#: template; ``grids`` (``gaussian``/``gaussian+column``/``None``=grid-free);
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
    {"name": "dust", "path": "bundles/{grid}/dust.nc",
     "grids": "gaussian", "levels": False, "coverage": None,
     "alignment": "climatology", "key": "dust_file", "auto": True,
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
     "grids": "gaussian", "levels": False, "coverage": [1950, 2022],
     "alignment": "transient", "key": "file", "auto": False, "staged": True},
    {"name": "emissions_amip",
     "path": "bundles/{grid}/emissions_amip/{year}.nc", "grids": "gaussian",
     "levels": False, "coverage": [1950, 2022], "alignment": "transient",
     "key": "emissions_file", "auto": False, "staged": True},
    {"name": "ozone_amip",
     "path": "bundles/{grid}_l{nlev}/ozone_amip/{year}.nc",
     "grids": "gaussian", "levels": True, "coverage": [1950, 2022],
     "alignment": "transient", "key": "ozone_file", "auto": False,
     "staged": True},
    {"name": "forcing_era5", "path": "bundles/{grid}/forcing_era5/{year}.nc",
     "grids": "gaussian", "levels": False, "coverage": [1979, 2024],
     "alignment": "transient", "key": "file", "auto": False, "staged": True},
    # MACv2-SP simple-plume file (Stevens et al. 2017; WDCC MACv2_SP_v1),
    # grid/level-free; staged=false until stage_macv2 uploads it.
    {"name": "macv2_sp", "path": "macv2_sp/MACv2.0-SP_v1.nc", "grids": None,
     "levels": False, "coverage": None, "alignment": "static",
     "key": "macv2_file", "auto": True, "staged": False},
)
NE30_TOPO = ("/glade/campaign/cesm/cesmdata/inputdata/atm/cam/topo/se/"
             "ne30np4_gmted2010_modis_bedmachine_nc3000_Laplace0100_"
             "noleak_greenlndantarcsgh30fac2.50_20250825.nc")
GRAV = 9.80665

#: Local MACv2.0-SP parameter file for ``stage_macv2`` — download once from
#: WDCC (https://doi.org/10.1594/WDCC/MACv2_SP_v1) into ``sources/macv2``.
MACV2_SRC = Path(os.environ.get(
    "JCM_MIRROR_ROOT",
    f"/glade/derecho/scratch/{os.environ.get('USER', '')}/hf_mirror")
) / "sources" / "macv2" / "MACv2.0-SP_v1.nc"

ROOT = Path(os.environ.get(
    "JCM_MIRROR_ROOT",
    f"/glade/derecho/scratch/{os.environ.get('USER', '')}/hf_mirror"))
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
    for grid, nlat in GRIDS.items():
        lats, lons = gaussian_latlon(nlat)
        fields = gaussian_grid_sso(str(GMTED), lats, lons)
        xr.Dataset({k: (("lat", "lon"), v) for k, v in fields.items()},
                   coords={"lat": lats, "lon": lons}
                   ).to_netcdf(out / f"sso_gmted2010_{grid}.nc")
        print("sso:", grid, flush=True)

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
        for grid, nlat in GRIDS.items():
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
    build_store(load_ceds_species, SPECIES, str(BUILD / "ceds_anthro.zarr"),
                "CEDS-CMIP-2025-04-18 (input4MIPs CMIP7), sector-summed, "
                "0.5 deg")
    build_store(load_bb_species, SPECIES, str(BUILD / "bb4cmip7.zarr"),
                "DRES-CMIP-BB4CMIP7-2-0 (input4MIPs CMIP7), 0.25 deg")


def stage_aux() -> None:
    """DMS/dust/oxidant matrix via tools/prep_jam_aux_inputs.py."""
    tool = Path(__file__).resolve().parents[3] / "tools" / \
        "prep_jam_aux_inputs.py"
    out = BUILD / "aux"
    for year in (1850, 2005):
        for nlev in (47, 95):
            for trunc in (63, 106):
                subprocess.run(
                    [sys.executable, str(tool), "--year", str(year),
                     "--nlevels", str(nlev), "--oxid-source", "waccm",
                     "--outdir", str(out), "--target-truncation",
                     str(trunc)],
                    check=True)


def stage_bundles() -> None:
    from jcm.data.mirror.bundles import (build_emissions_nc, build_forcing,
                                         build_terrain)
    from jcm.data.regridding import gaussian_latlon

    era5 = BUILD / "era5_land_climo_2005-2014_0p25.nc"
    for grid, nlat in GRIDS.items():
        lats, lons = gaussian_latlon(nlat)
        d = UPLOAD / "bundles" / grid
        d.mkdir(parents=True, exist_ok=True)
        build_terrain(str(BUILD / "sso" / f"sso_gmted2010_{grid}.nc"),
                      str(era5), str(d / "terrain.nc"))
        for era in ("pd", "pi"):
            build_forcing(str(era5), era, lats, lons,
                          str(d / f"forcing_{era}.nc"))
            build_emissions_nc(str(BUILD / "ceds_anthro.zarr"),
                               str(BUILD / "bb4cmip7.zarr"), era, lats,
                               lons, str(d / f"emissions_{era}.nc"))

    trunc = {"t63": 63, "t106": 106}
    for grid in GRIDS:
        for nlev in (47, 95):
            d = UPLOAD / "bundles" / f"{grid}_l{nlev}"
            d.mkdir(parents=True, exist_ok=True)
            for era, tag in (("pi", "pi1850"), ("pd", "pd2005-2014")):
                shutil.copy(BUILD / "ozone" /
                            f"ozone_fzj_cmip7_{tag}_{grid}_l{nlev}.nc",
                            d / f"ozone_{era}.nc")
            for era, year in (("pi", 1850), ("pd", 2005)):
                shutil.copy(
                    BUILD / "aux" /
                    f"oxidants_waccm_echam_l{nlev}_{year}_t{trunc[grid]}.nc",
                    d / f"oxidants_{era}.nc")
        g = UPLOAD / "bundles" / grid
        shutil.copy(BUILD / "aux" / f"dms_lana2011_climo_t{trunc[grid]}.nc",
                    g / "dms.nc")
        shutil.copy(BUILD / "aux" /
                    f"dust_erodibility_cam_f05_t{trunc[grid]}.nc",
                    g / "dust.nc")

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
    era5 = BUILD / "era5_land_climo_2005-2014_0p25.nc"
    scratch = BUILD / "ozone_amip"
    scratch.mkdir(parents=True, exist_ok=True)
    for grid, nlat in GRIDS.items():
        lats, lons = gaussian_latlon(nlat)
        g = UPLOAD / "bundles" / grid
        (g / "forcing_amip").mkdir(parents=True, exist_ok=True)
        (g / "emissions_amip").mkdir(parents=True, exist_ok=True)
        for nlev in (47, 95):
            (UPLOAD / "bundles" / f"{grid}_l{nlev}"
             / "ozone_amip").mkdir(parents=True, exist_ok=True)
        for year in range(first, last + 1):
            build_forcing_year(str(era5), year, lats, lons,
                               str(g / "forcing_amip" / f"{year}.nc"))
            build_emissions_year(str(BUILD / "ceds_anthro.zarr"),
                                 str(BUILD / "bb4cmip7.zarr"), year, lats,
                                 lons,
                                 str(g / "emissions_amip" / f"{year}.nc"))
            plev = scratch / f"ozone_{grid}_{year}_plev.nc"
            regrid_ozone_year(load_ozone_year(year), lats,
                              lons).to_netcdf(plev, encoding=_TIME_ENC)
            for nlev in (47, 95):
                interpolate_ozone(
                    plev,
                    UPLOAD / "bundles" / f"{grid}_l{nlev}" / "ozone_amip"
                    / f"{year}.nc", nlev)
            print("amip:", grid, year, flush=True)
    # Record the span actually staged (all three amip series share it) so the
    # manifest coverage names files that exist, not the wider source series.
    _record_staged_coverage(("forcing_amip", "emissions_amip", "ozone_amip"),
                            first, last)


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
    for grid, nlat in GRIDS.items():
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


def stage_macv2() -> None:
    """Stage the MACv2.0-SP simple-plume parameter file (grid/level-free).

    Copies ``MACv2.0-SP_v1.nc`` (Stevens et al. 2017; WDCC ``MACv2_SP_v1``,
    https://doi.org/10.1594/WDCC/MACv2_SP_v1) — a single small file, one plume
    geometry + ``year_weight``/``ann_cycle`` scalings, no per-grid variant — from
    ``MACV2_SRC`` to the manifest's ``macv2_sp/`` product path. Flip that
    product's ``staged`` flag to ``True`` and regenerate the manifest once the
    upload lands.
    """
    dst = UPLOAD / "macv2_sp" / "MACv2.0-SP_v1.nc"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(MACV2_SRC, dst)
    print("macv2:", dst, flush=True)


def build_manifest(staged_coverage: dict = None) -> dict:
    """Assemble the mirror-manifest dict from :data:`_MANIFEST_PRODUCTS`.

    Expands each row's ``{grid}``/``{nlev}`` template against the published grid
    (:data:`GRIDS` + column grids), level (:data:`PUBLISHED_LEVELS`) and vertical
    (:data:`PUBLISHED_VERTICALS`) sets so the availability knowledge the resolver
    consults is generated, never hand-listed. The published sets stay owned by
    ``jcm.data.bundle_names`` so this and the runner's ``auto`` gate cannot drift.

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
        elif row["grids"] == "gaussian+column":
            grids = gaussian + sorted(_COLUMN_GRIDS)
        else:  # None -> grid-free single file
            grids = None
        products[row["name"]] = {
            "path": row["path"],
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
    """Concrete ``(grid, nlev)`` variants a ``{year}``-series product declares.

    ``grid`` ranges over ``rec['grids']`` (every transient product is Gaussian);
    ``nlev`` over ``rec['levels']`` when the product is level-resolved, else the
    single ``None`` variant. Each is a distinct series the mirror is expected to
    hold in full — verification checks every one rather than pooling them, so a
    complete grid cannot mask a missing sibling variant.
    """
    for grid in rec["grids"]:
        for nlev in rec["levels"] or (None,):
            yield (grid, nlev)


def _variant_label(variant) -> str:
    """``(grid, nlev)`` -> the label used in drift keys, e.g. ``t63`` / ``t63_l47``."""
    grid, nlev = variant
    return grid if nlev is None else f"{grid}_l{nlev}"


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
    """Cross-check every transient variant's manifest coverage against the mirror.

    Lists the dataset repo (``HfApi().list_repo_files``), derives the real staged
    years per ``{year}`` product **and grid/level variant**
    (:func:`remote_transient_coverage`) and returns
    ``{"product[variant]": {"manifest": [...], "remote": [...] | None}}`` for every
    declared variant whose coverage disagrees with the mirror (empty == no drift).
    A variant absent from the mirror reports ``"remote": None``. Coverage is
    checked for CONTIGUITY, not just endpoints: a variant whose ``[first, last]``
    match but which drops an interior year is flagged with an extra ``"missing"``
    list (bounded for large holes) — otherwise clients trust the contiguous span
    and later request a file that is not there. Checking each variant — not one
    pooled span per product — is what keeps a complete grid from masking a missing
    sibling (e.g. a full t63 series hiding an absent t106). Network-only; driven by
    ``--stage manifest --verify-remote``.
    """
    from huggingface_hub import HfApi

    from jcm.data.remote import DEFAULT_REPO

    if manifest is None:
        manifest = build_manifest()
    repo_id = repo_id or manifest.get("repo", DEFAULT_REPO)
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    coverage = remote_transient_coverage(files, manifest)
    drift = {}
    for name, rec in manifest["products"].items():
        if "{year}" not in rec["path"]:
            continue
        declared = rec["coverage"]
        remote_variants = coverage.get(name, {})
        for variant in _product_variants(rec):
            years = remote_variants.get(variant)
            key = f"{name}[{_variant_label(variant)}]"
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
    return drift


def stage_registry() -> None:
    from jcm.data.mirror.registry import write_registry

    for name in ("ceds_anthro.zarr", "bb4cmip7.zarr",
                 "era5_land_climo_2005-2014_0p25.nc"):
        src, dst = BUILD / name, UPLOAD / "products" / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src,
                                                                       dst)
    sso_dst = UPLOAD / "products" / "sso"
    sso_dst.mkdir(parents=True, exist_ok=True)
    for f in (BUILD / "sso").glob("*.nc"):
        dst = sso_dst / f.name
        # staging may have hardlinked build -> upload already
        if not (dst.exists() and dst.samefile(f)):
            shutil.copy(f, dst)
    print(write_registry(str(UPLOAD)), flush=True)


#: Source paths each stage streams from — checked up front so a wrong
#: machine or an unmounted filesystem fails in seconds with a clear list,
#: not hours in with an obscure I/O error.
_STAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "sso": (str(GMTED), NE30_TOPO),
    "era5": ("/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc",),
    "ozone": ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
              "CMIP7/CMIP/FZJ/FZJ-CMIP-ozone-1-0",),
    "emissions": ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
                  "CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18",
                  "/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
                  "CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0"),
    "aux": ("/glade/campaign/cesm/cesmdata/inputdata/atm/cam/dst",
            "/glade/p/cesmdata/cseg/inputdata/atm/cam/ozone"),
    "bundles": (str(BUILD),),
    "macv2": (str(MACV2_SRC),),
    "amip": ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10",
             "/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/FZJ/FZJ-CMIP-ozone-1-0",
             "/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/CR/CR-CMIP-1-0-0",
             str(BUILD / "ceds_anthro.zarr"),
             str(BUILD / "era5_land_climo_2005-2014_0p25.nc")),
    "era5-transient": (
        "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc",
        "/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc",
        "/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
        "CMIP7/CMIP/CR/CR-CMIP-1-0-0",
        str(BUILD / "era5_land_climo_2005-2014_0p25.nc")),
    "registry": (str(UPLOAD),),
}


def check_sources(stage_names) -> None:
    """Fail fast when the Glade sources for the requested stages are absent.

    Source-free stages (``manifest`` — pure metadata) skip the Glade guard so
    the packaged manifest can be regenerated on any machine.
    """
    if not any(_STAGE_SOURCES.get(name) for name in stage_names):
        return
    if not Path("/glade").is_dir():
        sys.exit("This builder streams NCAR Glade source data — /glade is "
                 "not mounted here. Run it on Derecho/Casper (see "
                 "jcm/data/mirror/SOURCES.md).")
    missing = [p for name in stage_names
               for p in _STAGE_SOURCES.get(name, ())
               if not Path(p).exists()]
    if missing:
        sys.exit("Missing source paths (see jcm/data/mirror/SOURCES.md):\n  "
                 + "\n  ".join(sorted(set(missing))))


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


STAGES = {"sso": stage_sso, "era5": stage_era5, "ozone": stage_ozone,
          "emissions": stage_emissions, "aux": stage_aux,
          "bundles": stage_bundles, "amip": stage_amip,
          "era5-transient": stage_era5_transient, "macv2": stage_macv2,
          "manifest": stage_manifest,
          "registry": stage_registry, "upload": stage_upload}

#: Heavy / source-gated opt-in stages excluded from ``--stage all``. ``macv2``
#: needs the one-off WDCC download (``MACV2_SRC``); the others are multi-GB or
#: push to the remote.
_NOT_IN_ALL = ("amip", "era5-transient", "macv2", "upload")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    help="comma-separated stage list, or 'all' "
                         f"({', '.join(STAGES)}; 'all' excludes "
                         f"{', '.join(_NOT_IN_ALL)})")
    ap.add_argument("--years", default="1950,2022",
                    help="inclusive year range for --stage amip, "
                         "e.g. 1950,2022")
    ap.add_argument("--verify-remote", action="store_true",
                    help="after staging, cross-check every transient product's "
                         "manifest coverage against the live mirror "
                         "(list_repo_files) and exit non-zero on any drift")
    args = ap.parse_args()
    global _AMIP_YEARS
    first, last = (int(y) for y in args.years.split(","))
    _AMIP_YEARS = (first, last)
    names = ([n for n in STAGES if n not in _NOT_IN_ALL]
             if args.stage == "all" else args.stage.split(","))
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s) {unknown}; valid: {', '.join(STAGES)}")
    check_sources(names)
    for name in names:
        print(f"=== stage: {name} ===", flush=True)
        STAGES[name]()
    if args.verify_remote:
        print("=== verify-remote ===", flush=True)
        drift = verify_remote_coverage()
        for name, d in sorted(drift.items()):
            print(f"DRIFT {name}: manifest={d['manifest']} "
                  f"remote={d['remote']}", flush=True)
        if drift:
            sys.exit("manifest coverage disagrees with the live mirror")
        print("verify-remote: manifest coverages match the mirror", flush=True)


if __name__ == "__main__":
    main()
