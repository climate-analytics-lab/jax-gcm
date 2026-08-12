"""End-to-end mirror build driver (runs on NCAR Glade only).

Reproduces every artifact in the Hugging Face dataset from the sources in
``SOURCES.md``::

    python -m jcm.data.mirror.build_mirror --stage all
    python -m jcm.data.mirror.build_mirror --stage sso,bundles

Stages: ``sso``, ``era5``, ``ozone``, ``emissions`` (fat-node PBS job
recommended — see ``--help``), ``aux`` (dms/dust/oxidants via
``tools/prep_jam_aux_inputs.py``), ``bundles``, ``amip`` (yearly
transient forcing/emissions/ozone, ``--years first,last`` — issue #610),
``registry``, ``upload``
(push to the HF dataset; needs ``hf auth login`` with write access).
Outputs land in ``$JCM_MIRROR_ROOT`` (default ``$SCRATCH/hf_mirror``):
Tier A under ``build/``, the HF-shaped tree under ``upload/``.

Run from a repo checkout's own directory (``python -m ...`` with the
worktree as cwd): an editable-installed jcm elsewhere shadows
``PYTHONPATH`` and hides this package.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

GRIDS = {"t63": 96, "t106": 160}
NE30_TOPO = ("/glade/campaign/cesm/cesmdata/inputdata/atm/cam/topo/se/"
             "ne30np4_gmted2010_modis_bedmachine_nc3000_Laplace0100_"
             "noleak_greenlndantarcsgh30fac2.50_20250825.nc")
GRAV = 9.80665

ROOT = Path(os.environ.get(
    "JCM_MIRROR_ROOT",
    f"/glade/derecho/scratch/{os.environ.get('USER', '')}/hf_mirror"))
BUILD = ROOT / "build"
UPLOAD = ROOT / "upload"
GMTED = ROOT / "sources" / "gmted" / "mn30_grd"


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
    "amip": ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10",
             "/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/FZJ/FZJ-CMIP-ozone-1-0",
             "/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/CR/CR-CMIP-1-0-0",
             str(BUILD / "ceds_anthro.zarr"),
             str(BUILD / "era5_land_climo_2005-2014_0p25.nc")),
    "registry": (str(UPLOAD),),
}


def check_sources(stage_names) -> None:
    """Fail fast when the Glade sources for the requested stages are absent."""
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
          "registry": stage_registry, "upload": stage_upload}

#: Heavy opt-in stages excluded from ``--stage all``.
_NOT_IN_ALL = ("amip", "upload")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    help="comma-separated stage list, or 'all' "
                         f"({', '.join(STAGES)}; 'all' excludes "
                         f"{', '.join(_NOT_IN_ALL)})")
    ap.add_argument("--years", default="1950,2022",
                    help="inclusive year range for --stage amip, "
                         "e.g. 1950,2022")
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


if __name__ == "__main__":
    main()
