import xarray as xr
import pandas as pd
import sys
from pathlib import Path

def main(argv=None):
    """
    Main entrypoint for interpolate CLI and for importable use.

    Args:
        argv (list|None): list of command-line args (not including program name).
                          If None, uses sys.argv[1:].
    Returns:
        int: exit code (0 = success)
    """
    if argv is None:
        argv = sys.argv[1:]

    try:
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

        ds_daily.to_netcdf(cwd / "t30/clim/boundaries_daily.nc")
        ds_monthly.close()
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())