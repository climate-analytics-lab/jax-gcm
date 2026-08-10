"""Build PI/PD ozone climatologies on T63+T106, then L47/L95 model-level files."""
import sys

import numpy as np

sys.path.insert(0, "/glade/derecho/scratch/duncanwp/tmp/jam-fixes-dev")
from jcm.data.bc.interpolate_ozone import interpolate_ozone
from jcm.data.mirror.ozone import load_pd, load_pi, regrid_climatology

OUT = "/glade/derecho/scratch/duncanwp/hf_mirror/build/ozone"
import os
os.makedirs(OUT, exist_ok=True)


def gaussian_lats(nlat):
    return np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))


GRIDS = {"t63": (gaussian_lats(96), np.arange(192) * 360.0 / 192),
         "t106": (gaussian_lats(160), np.arange(320) * 360.0 / 320)}

for era, loader in (("pi1850", load_pi), ("pd2005-2014", load_pd)):
    da = loader()
    print(era, "loaded", dict(da.sizes), flush=True)
    for grid, (lats, lons) in GRIDS.items():
        ds = regrid_climatology(da, lats, lons)
        assert np.isfinite(ds.O3.values).all(), f"NaNs in {era}/{grid}"
        plev_path = f"{OUT}/ozone_fzj_cmip7_{era}_{grid}_plev.nc"
        ds.to_netcdf(plev_path)
        for nlev in (47, 95):
            out = f"{OUT}/ozone_fzj_cmip7_{era}_{grid}_l{nlev}.nc"
            interpolate_ozone(plev_path, out, nlev)
            print("wrote", out, flush=True)
