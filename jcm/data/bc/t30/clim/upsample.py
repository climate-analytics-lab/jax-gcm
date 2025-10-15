import numpy as np
import xarray as xr
from jcm.geometry import get_coords
import argparse
from pathlib import Path

def pad_1st_axis(arr):
    arr = np.swapaxes(arr, 0, 1)
    nan_rows = np.all(np.isnan(arr), axis=tuple(range(1, arr.ndim)))
    num_leading, num_trailing = np.argmax(~nan_rows), np.argmax(~nan_rows[::-1])
    arr_valid = arr[num_leading: arr.shape[0] - num_trailing]
    arr_padded = np.pad(arr_valid, pad_width=[(num_leading, num_trailing)] + [(0, 0)] * (arr.ndim - 1), mode='edge')
    return np.swapaxes(arr_padded, 0, 1)

def clamp_to_valid_ranges(ds):
    ds['stl'] = np.maximum(0, ds['stl'])
    ds['icec'] = ds['icec'].clip(0.0, 1.0)
    ds['sst'] = np.maximum(0, ds['sst'])
    ds['snowd'] = np.maximum(0, ds['snowd'])
    ds['soilw_am'] = ds['soilw_am'].clip(0.0, 1.0)
    # skipping orog to avoid clamping valid areas below sea level, but this might cause problems at edges
    ds['lsm'] = ds['lsm'].clip(0.0, 1.0)
    ds['alb'] = ds['alb'].clip(0.0, 1.0)
    return ds

def main(target_resolution):
    grid = get_coords(spectral_truncation=target_resolution).horizontal
    ds_boundaries = xr.open_dataset(Path(__file__).parent / 'boundaries_daily.nc')
    da_orog = xr.open_dataarray(Path(__file__).parent / 'orography.nc')

    # Pad longitude so edge values are handled correctly
    lon = ds_boundaries['lon'].values
    ds_boundaries_pad = xr.concat([
        ds_boundaries.assign_coords(lon=lon - 360),
        ds_boundaries,
        ds_boundaries.assign_coords(lon=lon + 360)
    ], dim='lon')
    da_orog_pad = xr.concat([
        da_orog.assign_coords(lon=lon - 360),
        da_orog,
        da_orog.assign_coords(lon=lon + 360)
    ], dim='lon')

    # Interpolate to new grid
    ds_boundaries_interp = ds_boundaries_pad.interp(
        lat=grid.latitudes * 180 / np.pi,
        lon=grid.longitudes * 180 / np.pi,
        method="linear"
    )
    da_orog_interp = da_orog_pad.interp(
        lat=grid.latitudes * 180 / np.pi,
        lon=grid.longitudes * 180 / np.pi,
        method="linear"
    )

    # Fill missing data at latitude extremes by padding with nearest non-nan values
    for var in ds_boundaries_interp.data_vars:
        ds_boundaries_interp[var].values = pad_1st_axis(ds_boundaries_interp[var].values)
    da_orog_interp.values = pad_1st_axis(da_orog_interp.values)

    ds_boundaries_interp = clamp_to_valid_ranges(ds_boundaries_interp)

    ds_boundaries_interp.to_netcdf(Path(__file__).parent / f'./boundaries_daily_t{target_resolution}.nc')
    da_orog_interp.to_netcdf(Path(__file__).parent / f'./orography_t{target_resolution}.nc')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale boundaries file to target horizontal spatial resolution.")
    parser.add_argument("target_resolution", type=int, help="Target horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425)")
    args = parser.parse_args()
    main(args.target_resolution)