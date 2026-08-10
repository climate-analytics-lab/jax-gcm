"""Assemble the per-grid bundles from the Tier A products.

Layout under $SCRATCH/hf_mirror/upload/ mirrors the intended HF dataset:

    bundles/<grid>/terrain.nc
    bundles/<grid>/forcing_{pi,pd}.nc
    bundles/<grid>/emissions_{pi,pd}.nc
    bundles/<grid>_l{47,95}/ozone_{pi,pd}.nc      (level-resolved)
    bundles/<grid>_l{47,95}/oxidants_{1850,2014}.nc
    bundles/<grid>/dms.nc, dust.nc
    bundles/ne30pg3/sso.nc                        (native-column SSO)
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/glade/derecho/scratch/duncanwp/tmp/jam-fixes-dev")
from jcm.data.mirror.bundles import (build_emissions_nc, build_forcing,
                                     build_terrain, gaussian_latlon)

B = "/glade/derecho/scratch/duncanwp/hf_mirror/build"
UP = "/glade/derecho/scratch/duncanwp/hf_mirror/upload/bundles"
ERA5 = f"{B}/era5_land_climo_2005-2014_0p25.nc"

GRIDS = {"t63": 96, "t106": 160}

for grid, nlat in GRIDS.items():
    lats, lons = gaussian_latlon(nlat)
    d = f"{UP}/{grid}"
    os.makedirs(d, exist_ok=True)
    build_terrain(f"{B}/sso/sso_gmted2010_{grid}.nc", ERA5, f"{d}/terrain.nc")
    for era in ("pd", "pi"):
        build_forcing(ERA5, era, lats, lons, f"{d}/forcing_{era}.nc")
        build_emissions_nc(f"{B}/ceds_anthro.zarr", f"{B}/bb4cmip7.zarr",
                           era, lats, lons, f"{d}/emissions_{era}.nc")

# level-resolved + aux products already built by their own drivers: copy in.
trunc = {"t63": 63, "t106": 106}
for grid in GRIDS:
    for nlev in (47, 95):
        d = f"{UP}/{grid}_l{nlev}"
        os.makedirs(d, exist_ok=True)
        for era, tag in (("pi", "pi1850"), ("pd", "pd2005-2014")):
            shutil.copy(f"{B}/ozone/ozone_fzj_cmip7_{tag}_{grid}_l{nlev}.nc",
                        f"{d}/ozone_{era}.nc")
        for year in (1850, 2014):
            src = f"{B}/aux/oxidants_cam_echam_l{nlev}_{year}_t{trunc[grid]}.nc"
            shutil.copy(src, f"{d}/oxidants_{year}.nc")
    shutil.copy(f"{B}/aux/dms_lana2011_climo_t{trunc[grid]}.nc",
                f"{UP}/{grid}/dms.nc")
    shutil.copy(f"{B}/aux/dust_erodibility_cam_f05_t{trunc[grid]}.nc",
                f"{UP}/{grid}/dust.nc")

os.makedirs(f"{UP}/ne30pg3", exist_ok=True)
shutil.copy(f"{B}/sso/sso_gmted2010_ne30pg3.nc", f"{UP}/ne30pg3/sso.nc")
print("BUNDLES_DONE", flush=True)
