"""Build dated prescribed land-temperature forcing from control snapshots.

This preserves absolute timestamps and the recorded diurnal cycle. It does
not regrid, average over days, or replay a control year on different dates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def build_archive(paths, output, *, control_sha):
    """Write time/lat/lon skin temperatures, retaining control provenance."""
    variable = 'surface.land_surface_temperature'
    frames, provenance = [], []
    for path in paths:
        with xr.open_dataset(path) as source:
            if variable not in source:
                raise ValueError(f'{path}: missing {variable}; use skin-temperature snapshots.')
            temperature = source[variable]
            if set(temperature.dims) != {'time', 'lat', 'lon'}:
                raise ValueError(f'{path}: expected time/lat/lon snapshots on the model grid.')
            if temperature.attrs.get('units') != 'K':
                raise ValueError(f'{path}: skin temperatures must declare units=K.')
            frames.append(temperature.load())
            provenance.append({'path': str(path), 'attributes': dict(source.attrs)})
    if not frames:
        raise ValueError('At least one control snapshot file is required.')
    temperature = xr.concat(frames, dim='time', join='exact').sortby('time')
    time = temperature.time.values
    if (not np.issubdtype(time.dtype, np.datetime64)
            or len(time) < 2 or np.any(np.isnat(time))):
        raise ValueError('At least two decoded Gregorian timestamps are required.')
    if np.any(np.diff(time) <= np.timedelta64(0, 's')):
        raise ValueError('Snapshot timestamps must be distinct; remove overlapping files.')
    if not np.all(np.isfinite(temperature.values)) or np.any(temperature.values <= 0):
        raise ValueError('Snapshot temperatures must be finite and positive kelvin.')
    if not control_sha:
        raise ValueError('Record the source control run commit with control_sha.')
    dataset = temperature.rename('land_skin_temperature').to_dataset()
    dataset.attrs.update(
        control_sha=control_sha,
        source_snapshots=json.dumps(provenance, default=str),
        recipe='Copy instantaneous control-run skin temperatures; no averaging or regridding.',
        time_coverage_start=str(time[0]), time_coverage_end=str(time[-1]),
        coverage_days=float((time[-1] - time[0]) / np.timedelta64(1, 'D')),
        alignment='by_date_interp', persistence='strict',
    )
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new archive path.')
    dataset.to_netcdf(output)
    return output


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='+', help='Control-run snapshot netCDF files.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--control-sha', required=True, help='Commit used by the control run.')
    args = parser.parse_args()
    build_archive(args.snapshots, args.output, control_sha=args.control_sha)


if __name__ == '__main__':
    main()
