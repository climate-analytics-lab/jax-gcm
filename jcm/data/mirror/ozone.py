"""FZJ CMIP7 ozone (vmro3) -> per-grid climatology files.

Source: FZJ-CMIP-ozone-1-0, 66 pressure levels (1000 -> 1e-4 hPa),
1.9 deg x 2.5 deg, mole/mole. Two climatologies:

* PI — the provided 1850 12-month climatology file, used directly.
* PD — 2005-2014 monthly mean from the 2000-2022 transient file.

The output of :func:`regrid_climatology` is a ``(time, plev, lat, lon)``
``O3`` file with ``plev`` in Pa — exactly the input contract of
``jcm.data.bc.interpolate_ozone``, which then produces the model-level
L47/L95 files at bundle assembly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/CMIP7/"
         "CMIP/FZJ/FZJ-CMIP-ozone-1-0/atmos")
PI_CLIM = (f"{_ROOT}/monC/vmro3/gn/v20250904/"
           "vmro3_input4MIPs_ozone_CMIP_FZJ-CMIP-ozone-1-0_gn_"
           "185001-185012-clim.nc")
PD_TRANSIENT = (f"{_ROOT}/mon/vmro3/gn/v20250904/"
                "vmro3_input4MIPs_ozone_CMIP_FZJ-CMIP-ozone-1-0_gn_"
                "200001-202212.nc")


def load_pi() -> xr.DataArray:
    return xr.open_dataset(PI_CLIM, decode_times=False).vmro3


def load_pd(years=(2005, 2014)) -> xr.DataArray:
    ds = xr.open_dataset(PD_TRANSIENT)
    da = ds.vmro3.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    clim = da.groupby("time.month").mean("time").rename(month="time")
    return clim.transpose("time", "plev", "lat", "lon")


def regrid_climatology(da: xr.DataArray, lats: np.ndarray,
                       lons: np.ndarray) -> xr.Dataset:
    """Bilinear regrid to a Gaussian grid; returns interpolate_ozone input."""
    from jcm.data.regridding import interp_to

    out = interp_to(da, lats, lons)
    plev_pa = out.plev.values * 100.0          # source file is hPa
    ds = xr.Dataset({"O3": (("time", "plev", "lat", "lon"),
                            out.transpose("time", "plev", "lat", "lon").values,
                            {"units": "mole mole-1"})},
                    coords={"time": np.arange(1, 13), "plev": plev_pa,
                            "lat": lats, "lon": lons})
    ds.plev.attrs["units"] = "Pa"
    ds.attrs["source"] = "FZJ-CMIP-ozone-1-0 (input4MIPs CMIP7)"
    return ds
