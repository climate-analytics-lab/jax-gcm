import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from jcm.geometry import get_coords

def create_boundaries_daily():
    """
    Generate boundaries_daily.nc by interpolating boundaries.nc to daily frequency.
    """
    cwd = Path(__file__).resolve().parent

    ds_monthly = xr.open_dataset(cwd / "t30/clim/boundaries.nc")

    time_vars = [var for var in ds_monthly.data_vars if 'time' in ds_monthly[var].dims]
    non_time_vars = [var for var in ds_monthly.data_vars if 'time' not in ds_monthly[var].dims]

    # pad monthly data with dec/jan of adjacent years
    pad_n = 1
    previous_year_padding = [ds_monthly[time_vars].isel(time=i) for i in range(12 - pad_n, 12)]
    next_year_padding = [ds_monthly[time_vars].isel(time=i) for i in range(pad_n)]
    extended_monthly_time_vars = xr.concat(previous_year_padding + [ds_monthly[time_vars]] + next_year_padding, dim='time')
    extended_time = pd.date_range(start=f'1980-{13-pad_n:02}-01', end=f'1982-{pad_n:02}-01', freq='MS')
    extended_monthly_time_vars['time'] = extended_time

    daily_time_vars = extended_monthly_time_vars.resample(time='1D').interpolate('linear')
    daily_time_vars = daily_time_vars.sel(time=slice('1981-01-01', '1981-12-31'))
    ds_daily = xr.merge([daily_time_vars, ds_monthly[non_time_vars]])

    output_file = cwd / "t30/clim/boundaries_daily.nc"
    ds_daily.to_netcdf(output_file)
    ds_monthly.close()
    print(f"Generated {output_file.name}")

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
    ds['snowc'] = np.maximum(0, ds['snowc'])
    ds['soilw_am'] = ds['soilw_am'].clip(0.0, 1.0)
    # skipping orog to avoid clamping valid areas below sea level, but this might cause problems at edges
    ds['lsm'] = ds['lsm'].clip(0.0, 1.0)
    ds['alb'] = ds['alb'].clip(0.0, 1.0)
    return ds

def upsample_ds(ds, target_resolution):
    grid = get_coords(spectral_truncation=target_resolution).horizontal
    
    # Pad longitude so edge values are handled correctly
    lon = ds['lon'].values
    ds_pad = xr.concat([
        ds.assign_coords(lon=lon - 360),
        ds,
        ds.assign_coords(lon=lon + 360)
    ], dim='lon')
    
    # Interpolate to new grid
    ds_interp = ds_pad.interp(
        lat=grid.latitudes * 180 / np.pi,
        lon=grid.longitudes * 180 / np.pi,
        method="linear"
    )

    # Fill missing data at latitude extremes by padding with nearest non-nan values
    if isinstance(ds_interp, xr.DataArray):
        ds_interp.values = pad_1st_axis(ds_interp.values)
    else:
        for var in ds_interp.data_vars:
            ds_interp[var].values = pad_1st_axis(ds_interp[var].values)

    return ds_interp

def interpolate(target_resolution):
    boundaries_output_file = Path(__file__).parent / f"boundaries_daily_t{target_resolution}.nc"
    if boundaries_output_file.exists():
        print(f"{boundaries_output_file.name} already exists.")
    else:
        boundaries_input_file = Path(__file__).parent / "t30/clim/boundaries_daily.nc"
        if not boundaries_input_file.exists():
            create_boundaries_daily()
        print(f"Interpolating boundaries_daily.nc to T{target_resolution} resolution...")
        ds_boundaries = xr.open_dataset(boundaries_input_file)
        ds_boundaries_interp = upsample_ds(ds_boundaries, target_resolution)
        ds_boundaries_interp = clamp_to_valid_ranges(ds_boundaries_interp)
        ds_boundaries_interp.to_netcdf(boundaries_output_file)
        ds_boundaries.close()
        print(f"Generated {boundaries_output_file.name}")

    orography_output_file = Path(__file__).parent / f"orography_t{target_resolution}.nc"
    if orography_output_file.exists():
        print(f"{orography_output_file.name} already exists.")
        return
    print(f"Interpolating orography.nc to T{target_resolution} resolution...")
    da_orog = xr.open_dataarray(Path(__file__).parent / 't30/clim/orography.nc')
    da_orog_interp = upsample_ds(da_orog, target_resolution)
    da_orog_interp.to_netcdf(orography_output_file)
    da_orog.close()
    
def main(argv=None) -> int:
    """
    CLI entrypoint. Parse argv and call `interpolate`.

    Args:
        argv (list[str] | None): list of command-line args (not including program name).
                                 If None, uses sys.argv[1:].
    Returns:
        int: exit code (0 = success, non-zero = failure)
    """
    parser = argparse.ArgumentParser(
        description="Upscale boundaries file to target horizontal spatial resolution."
    )
    valid_res = [21, 31, 42, 85, 106, 119, 170, 213, 340, 425]
    parser.add_argument(
        "target_resolution",
        type=int,
        choices=valid_res,
        help=f"Target horizontal resolution (choices: {valid_res})"
    )

    # let argparse handle argument errors (it raises SystemExit on bad args)
    args = parser.parse_args(argv) # uses sys.argv[1:] if argv is None

    try:
        interpolate(args.target_resolution)
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())