"""Build SSO products for T63, T106 (Gaussian) and ne30pg3 (columns).

Writes netCDF to $SCRATCH/hf_mirror/build/sso/ and prints validation
correlations (T63 vs packaged ECHAM terrain, ne30 vs CESM topo).
"""
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, "/glade/derecho/scratch/duncanwp/tmp/jam-fixes-dev")
from jcm.data.mirror.sso import column_grid_sso, gaussian_grid_sso

DEM = "/glade/derecho/scratch/duncanwp/hf_mirror/sources/gmted/mn30_grd"
OUT = "/glade/derecho/scratch/duncanwp/hf_mirror/build/sso"
TOPO = ("/glade/campaign/cesm/cesmdata/inputdata/atm/cam/topo/se/"
        "ne30np4_gmted2010_modis_bedmachine_nc3000_Laplace0100_noleak_"
        "greenlndantarcsgh30fac2.50_20250825.nc")
REF = "/glade/u/home/duncanwp/jax-gcm/jcm/data/bc/t63/terrain.nc"

ATTRS = dict(
    source="GMTED2010 mean 30 arc-sec (USGS EROS)",
    method="exact pixel binning; Lott & Miller (1997) gradient tensor",
    note=("lsm is DEM-validity placeholder; mask with ERA5 invariant "
          "land fraction at bundle assembly"),
)


def gaussian_lats(nlat):
    return np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))


def write_gaussian(fields, lats, lons, path):
    ds = xr.Dataset(
        {k: (("lat", "lon"), v) for k, v in fields.items()},
        coords={"lat": lats, "lon": lons}, attrs=ATTRS)
    ds.to_netcdf(path)
    print("wrote", path)


def corr(a, b, mask):
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


import os
os.makedirs(OUT, exist_ok=True)

# ---- T63 (96x192): build + validate against packaged ECHAM terrain ----
t0 = time.time()
lats63 = gaussian_lats(96)
lons63 = np.arange(192) * 360.0 / 192
f63 = gaussian_grid_sso(DEM, lats63, lons63)
print(f"T63 pass: {time.time()-t0:.0f}s")
ref = xr.open_dataset(REF)
# packaged file may be (lon, lat) and N->S; align to ours (lat asc, lon).
flip = ref.lat.values[0] > ref.lat.values[-1]


def ref_field(name):
    v = ref[name].transpose("lat", "lon").values
    return v[::-1] if flip else v


land = ref_field("lsm") > 0.5
for name in ("orog", "orostd", "oropic", "oroval", "orogam"):
    print(f"  corr {name}: {corr(f63[name], ref_field(name), land):.3f}")
write_gaussian(f63, lats63, lons63, f"{OUT}/sso_gmted2010_t63.nc")

# ---- T106 (160x320) ----
t0 = time.time()
lats106 = gaussian_lats(160)
lons106 = np.arange(320) * 360.0 / 320
f106 = gaussian_grid_sso(DEM, lats106, lons106)
print(f"T106 pass: {time.time()-t0:.0f}s")
write_gaussian(f106, lats106, lons106, f"{OUT}/sso_gmted2010_t106.nc")

# ---- ne30pg3 columns: build + cross-check against CESM topo ----
topo = xr.open_dataset(TOPO)
t0 = time.time()
fne = column_grid_sso(DEM, topo.lat.values, topo.lon.values)
print(f"ne30 pass: {time.time()-t0:.0f}s")
phis_m = topo.PHIS.values / 9.80665
landfrac = topo.LANDFRAC.values > 0.5
print(f"  corr orog vs PHIS/g (land): {corr(fne['orog'], phis_m, landfrac):.3f}")
print(f"  corr orostd vs SGH (land): {corr(fne['orostd'], topo.SGH.values, landfrac):.3f}")
ds = xr.Dataset(
    {k: (("ncol",), v) for k, v in fne.items()},
    coords={"lat": ("ncol", topo.lat.values),
            "lon": ("ncol", topo.lon.values)}, attrs=ATTRS)
ds.to_netcdf(f"{OUT}/sso_gmted2010_ne30pg3.nc")
print("wrote", f"{OUT}/sso_gmted2010_ne30pg3.nc")
