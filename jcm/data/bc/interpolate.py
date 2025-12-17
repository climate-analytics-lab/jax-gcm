import xarray as xr
import numpy as np
import pandas as pd
from importlib import resources
import argparse
from pathlib import Path
from jcm.utils import VALID_TRUNCATIONS, get_coords

def interpolate_to_daily(ds_monthly: xr.Dataset) -> xr.Dataset:
    """Interpolate monthly forcing data to daily resolution using linear interpolation.

    Pads the monthly data with December/January from adjacent years to enable smooth
    interpolation across year boundaries, then resamples to daily frequency.

    Args:
        ds_monthly: Dataset with monthly time resolution (12 time steps).
                   Must have a 'time' dimension with monthly frequency.

    Returns:
        Dataset interpolated to daily resolution (365 time steps).

    Raises:
        ValueError: If the dataset doesn't have exactly 12 monthly timestamps.

    """
    # validate that time coordinate is monthly
    time = pd.DatetimeIndex(ds_monthly["time"].values)
    if len(time) != 12:
        raise ValueError(f"'time' has {len(time)} entries, expected 12 monthly timestamps.")
    elif pd.infer_freq(time) not in ("MS", "M"):
        raise ValueError("Timestamps do not have a monthly frequency")

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
    f"""Upsample a dataset to a target spectral resolution using linear interpolation.

    Pads the dataset at the poles (latitude) and periodically in longitude to enable
    interpolation to grid points outside the original domain, then interpolates to the
    target grid using linear interpolation.

    Args:
        ds: Dataset to upsample with 'lat' and 'lon' dimensions.
        target_resolution: Target spectral truncation number. Must be one of: {VALID_TRUNCATIONS}.

    Returns:
        Dataset interpolated to the target resolution grid.

    """
    grid = get_coords(spectral_truncation=target_resolution).horizontal

    # Pad latitude with extra rows at poles so data can be interpolated to higher latitudes than exist in T30 grid
    south_pole = ds.isel(lat=0).mean(dim="lon", keep_attrs=True)
    north_pole = ds.isel(lat=-1).mean(dim="lon", keep_attrs=True)
    ds_pad = xr.concat([
        south_pole.expand_dims(lon=ds.lon, lat=[-90]).transpose(*ds.dims),
        ds,
        north_pole.expand_dims(lon=ds.lon, lat=[90]).transpose(*ds.dims),
    ], dim="lat")

    # Pad longitude to enforce periodicity
    lon = ds_pad['lon'].values
    ds_pad = xr.concat([
        ds_pad.assign_coords(lon=lon - 360),
        ds_pad,
        ds_pad.assign_coords(lon=lon + 360)
    ], dim='lon')
    
    # Interpolate to new grid
    ds_interp = ds_pad.interp(
        lat=grid.latitudes * 180 / np.pi,
        lon=grid.longitudes * 180 / np.pi,
        method="linear"
    )

    return ds_interp

def upsample_forcings_ds(ds: xr.Dataset, target_resolution: int) -> xr.Dataset:
    f"""Upsample forcing data to target resolution with physical constraints.

    Interpolates the forcing dataset to the target resolution and applies physical
    constraints: all variables are clipped to non-negative values, and fractional
    variables (ice concentration, soil moisture, albedo) are clipped to [0, 1].

    Args:
        ds: Forcing dataset to upsample (should contain variables like icec, soilw_am, alb).
        target_resolution: Target spectral truncation number. Must be one of: {VALID_TRUNCATIONS}.

    Returns:
        Upsampled forcing dataset with physical constraints applied.

    """
    ds_interp = _upsample_ds(ds, target_resolution)
    for v in ds_interp.data_vars:
        ds_interp[v] = ds_interp[v].clip(min=0.)
    for v in ['icec', 'soilw_am', 'alb']:
        ds_interp[v] = ds_interp[v].clip(max=1.)
    return ds_interp

def upsample_terrain_ds(ds: xr.Dataset, target_resolution: int) -> xr.Dataset:
    f"""Upsample terrain data to target resolution with physical constraints.

    Interpolates the terrain dataset to the target resolution and clips the land-sea
    mask to [0, 1]. Orography is not clipped to preserve real areas below sea level,
    though this may allow bad extrapolated values at extreme latitudes.

    Args:
        ds: Terrain dataset to upsample (should contain 'lsm' and 'orog').
        target_resolution: Target spectral truncation number. Must be one of: {VALID_TRUNCATIONS}.

    Returns:
        Upsampled terrain dataset with land-sea mask clipped to [0, 1].

    """
    ds_interp = _upsample_ds(ds, target_resolution)
    ds_interp['lsm'] = ds_interp['lsm'].clip(0.0, 1.0)
    # not clamping orog to avoid erasing real areas below sea level, but this might allow bad extrapolated values at the extreme latitudes
    return ds_interp

def interpolate(target_resolution, output_dir=None):
    f"""Interpolate T30 forcing and terrain data to target resolution and save to output directory.

    Reads the original T30 resolution forcing and terrain data from package resources,
    interpolates them to the target spectral resolution, and writes the output files.
    Skips generation if output files already exist to avoid overwriting.

    The function generates three output files:
    - forcing_daily.nc: Intermediate daily forcing data (reused across resolutions)
    - forcing_t{{target_resolution}}.nc: Forcing data at target resolution
    - terrain_t{{target_resolution}}.nc: Terrain data at target resolution

    Args:
        target_resolution: Target spectral truncation number. Must be one of: {VALID_TRUNCATIONS}.
        output_dir: Directory to write output files. If None, uses current working directory.

    """
    # Read source files from package resources
    bc_dir = resources.files('jcm.data.bc')
    forcing_original_file = bc_dir / "t30/clim/forcing.nc"
    terrain_original_file = bc_dir / "t30/clim/terrain.nc"

    # Write output files to current working directory (or specified output_dir)
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    forcing_daily_file = output_dir / "forcing_daily.nc"
    forcing_upscaled_file = output_dir / f"forcing_t{target_resolution}.nc"
    terrain_upscaled_file = output_dir / f"terrain_t{target_resolution}.nc"

    # Generate forcing at target resolution
    if forcing_upscaled_file.exists():
        print(f"{forcing_upscaled_file} already exists, skipping forcing interpolation.")
    else:
        # Generate daily forcing if needed
        if not forcing_daily_file.exists():
            print(f"Interpolating {forcing_original_file} to daily resolution...")
            with xr.open_dataset(forcing_original_file) as ds_monthly:
                ds_daily = interpolate_to_daily(ds_monthly)
                ds_daily.to_netcdf(forcing_daily_file)
            print(f"Generated {forcing_daily_file}")
        else:
            print(f"{forcing_daily_file} already exists, skipping daily forcing generation.")

        print(f"Interpolating {forcing_daily_file} to T{target_resolution} resolution...")
        with xr.open_dataset(forcing_daily_file) as ds_forcing:
            ds_forcing_interp = upsample_forcings_ds(ds_forcing, target_resolution)
            ds_forcing_interp.to_netcdf(forcing_upscaled_file)
        print(f"Generated {forcing_upscaled_file}")

    # Generate terrain at target resolution
    if terrain_upscaled_file.exists():
        print(f"{terrain_upscaled_file} already exists, skipping terrain interpolation.")
    else:
        print(f"Interpolating {terrain_original_file} to T{target_resolution} resolution...")
        with xr.open_dataset(terrain_original_file) as ds_terrain:
            ds_terrain_interp = upsample_terrain_ds(ds_terrain, target_resolution)
            ds_terrain_interp.to_netcdf(terrain_upscaled_file)
        print(f"Generated {terrain_upscaled_file}")

def main(argv=None) -> int:
    """CLI entrypoint. Parse argv and call `interpolate`.

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