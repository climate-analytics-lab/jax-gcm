import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from jcm.utils import VALID_TRUNCATIONS, get_coords

def _pad_1st_axis(arr):
    arr = np.swapaxes(arr, 0, 1)
    nan_rows = np.all(np.isnan(arr), axis=tuple(range(1, arr.ndim)))
    num_leading, num_trailing = np.argmax(~nan_rows), np.argmax(~nan_rows[::-1])
    arr_valid = arr[num_leading: arr.shape[0] - num_trailing]
    arr_padded = np.pad(arr_valid, pad_width=[(num_leading, num_trailing)] + [(0, 0)] * (arr.ndim - 1), mode='edge')
    return np.swapaxes(arr_padded, 0, 1)

def interpolate_to_daily(ds_monthly: xr.Dataset) -> xr.Dataset:
    # validate that time coordinate is monthly
    time = pd.DatetimeIndex(ds_monthly["time"].values)
    if len(time) != 12:
        raise ValueError(f"'time' has {len(time)} entries, expected 12 monthly timestamps.")
    elif pd.infer_freq(time) not in ("MS", "M"):
        raise ValueError(f"Timestamps do not have a monthly frequency")

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
    return xr.merge([daily_time_vars, ds_monthly[non_time_vars]])

def _upsample_ds(ds: xr.Dataset, target_resolution: int) -> xr.Dataset:
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
    for var in ds_interp.data_vars:
        ds_interp[var].values = _pad_1st_axis(ds_interp[var].values)

    return ds_interp

def upsample_forcings_ds(ds: xr.Dataset, target_resolution: int) -> xr.Dataset:
    ds_interp = _upsample_ds(ds, target_resolution)
    for v in ds_interp.data_vars:
        ds_interp[v] = ds_interp[v].clip(min=0.)
    for v in ['icec', 'soilw_am', 'alb']:
        ds_interp[v] = ds_interp[v].clip(max=1.)
    return ds_interp

def upsample_terrain_ds(ds: xr.Dataset, target_resolution: int) -> xr.Dataset:
    ds_interp = _upsample_ds(ds, target_resolution)
    ds_interp['lsm'] = ds_interp['lsm'].clip(0.0, 1.0)
    # not clamping orog to avoid erasing real areas below sea level, but this might allow bad extrapolated values at the extreme latitudes
    return ds_interp

def interpolate(target_resolution):
    cwd = Path(__file__).resolve().parent
    forcing_original_file = cwd / "t30/clim/forcing.nc"
    forcing_daily_file = cwd / "t30/clim/forcing_daily.nc"
    forcing_upscaled_file = cwd / f"forcing_t{target_resolution}.nc"

    if forcing_upscaled_file.exists():
        print(f"{forcing_upscaled_file.name} already exists.")

    else:
        if not forcing_daily_file.exists():
            print(f"Interpolating {forcing_original_file.name} to daily resolution...")
            with xr.open_dataset(forcing_original_file) as ds_monthly:
                ds_daily = interpolate_to_daily(ds_monthly)
                ds_daily.to_netcdf(forcing_daily_file)
            print(f"Generated {forcing_daily_file.name}")

        print(f"Interpolating {forcing_daily_file.name} to T{target_resolution} resolution...")
        with xr.open_dataset(forcing_daily_file) as ds_forcing:
            ds_forcing_interp = upsample_forcings_ds(ds_forcing, target_resolution)
            ds_forcing_interp.to_netcdf(forcing_upscaled_file)
        print(f"Generated {forcing_upscaled_file.name}")


    terrain_original_file = cwd / "t30/clim/terrain.nc"
    terrain_upscaled_file = cwd / f"terrain_t{target_resolution}.nc"

    if terrain_upscaled_file.exists():
        print(f"{terrain_upscaled_file.name} already exists.")
        return
    
    print(f"Interpolating {terrain_original_file.name} to T{target_resolution} resolution...")
    with xr.open_dataset(terrain_original_file) as ds_terrain:
        ds_terrain_interp = upsample_terrain_ds(ds_terrain, target_resolution)
        ds_terrain_interp.to_netcdf(terrain_upscaled_file)
    print(f"Generated {terrain_upscaled_file.name}")

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
        description="Upscale forcing file to target horizontal spatial resolution."
    )
    parser.add_argument(
        "target_resolution",
        type=int,
        choices=list(VALID_TRUNCATIONS),
        help=f"Target horizontal resolution (choices: {VALID_TRUNCATIONS})"
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