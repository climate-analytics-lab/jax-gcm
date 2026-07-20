"""First-NaN forensics for the T63 SL sulfate blow-up.

Reads the per-step trajectory from run_dinosaur_sl_smoke --save-nc and
reports, for the first field/step where NaN appears: the step, the cells
(level / lat / lon), and the co-located state (orography proxy via
geopotential, T, humidity, cloud water, g_so2, sulfate masses, mode numbers)
one step BEFORE the NaN — i.e. the conditions that produced it.
"""

from __future__ import annotations

import sys

import numpy as np
import xarray as xr

path = sys.argv[1] if len(sys.argv) > 1 else (
    "/glade/derecho/scratch/duncanwp/jam_runs/sl_forensics/sl_t63_24steps.nc")
ds = xr.open_dataset(path)
print(f"{path}: {dict(ds.sizes)}")

watch = [v for v in ("m_so4_ait", "m_so4_acc", "m_so4_cor", "g_h2so4",
                     "g_so2", "n_ait", "temperature") if v in ds]
first_nan = {}
for v in watch:
    arr = ds[v].values  # (time, ...)
    for t in range(arr.shape[0]):
        if np.isnan(arr[t]).any():
            first_nan[v] = t
            break
print("first NaN step per field:", first_nan or "NONE")
if not first_nan:
    sys.exit(0)

v0 = min(first_nan, key=first_nan.get)
t0 = first_nan[v0]
print(f"\nearliest: {v0} at step {t0}")
arr = ds[v0].values[t0]
idx = np.argwhere(np.isnan(arr))
print(f"NaN cells: {len(idx)}; first 5 (level, lon, lat indices): {idx[:5].tolist()}")

# Levels are surface-first? dinosaur to_xarray keeps TOA-first physics frame —
# report the raw indices plus coordinates if present.
lev_i, lon_i, lat_i = idx[0]
lat = float(ds["lat"].values[lat_i]) if "lat" in ds else float("nan")
lon = float(ds["lon"].values[lon_i]) if "lon" in ds else float("nan")
print(f"first cell: level-idx {lev_i}, lon {lon:.1f}E, lat {lat:.1f}N")

tprev = max(t0 - 1, 0)
print(f"\nstate at step {tprev} (pre-NaN), same cell:")
for v in ("temperature", "specific_humidity", "g_so2", "g_h2so4",
          "m_so4_ait", "m_so4_acc", "n_ait", "n_acc",
          "geopotential", "aerosol_optical_depth"):
    if v not in ds:
        continue
    a = ds[v].values[tprev]
    val = a[(lev_i, lon_i, lat_i)] if a.ndim == 3 else a[(lon_i, lat_i)]
    print(f"  {v:24s} {float(val):.6e}")

# Column profile of the offender one step before, to see where in the column
# it lives (surface source vs model-top).
if v0 in ds and ds[v0].values.ndim == 4:
    col = ds[v0].values[tprev][:, lon_i, lat_i]
    print(f"\n{v0} column at pre-NaN step (TOA-first): min {np.nanmin(col):.3e} "
          f"max {np.nanmax(col):.3e}")
    kmax = int(np.nanargmax(np.abs(col)))
    print(f"  |max| at level-idx {kmax} of {col.size}")
# How many cells NaN per step (growth curve).
growth = [int(np.isnan(ds[v0].values[t]).sum()) for t in range(ds.sizes["time"])]
print(f"\n{v0} NaN-cell growth by step: {growth}")
