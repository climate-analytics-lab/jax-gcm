"""Extract native SPEEDY T30 terrain values for a unified ARMBE cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def nearest_t30_site_terrain(cache: str | Path, terrain_file: str | Path) -> dict:
    """Return one native T30 terrain prescription per cache site-facility."""
    with xr.open_dataset(cache) as raw_cache, xr.open_dataset(terrain_file) as terrain:
        cache_data = raw_cache.load()
        terrain_data = terrain.load()

    required_cache = {"site_facility", "latitude", "longitude"}
    missing_cache = required_cache - set(cache_data.variables)
    if missing_cache:
        raise ValueError(f"cache is missing required variables: {sorted(missing_cache)}")
    required_terrain = {"orog", "lsm"}
    missing_terrain = required_terrain - set(terrain_data.variables)
    if missing_terrain:
        raise ValueError(f"terrain file is missing required variables: {sorted(missing_terrain)}")

    site_names = np.asarray(cache_data["site_facility"].values).astype(str)
    values = {}
    for site in sorted(set(site_names)):
        rows = np.flatnonzero(site_names == site)
        latitude = np.asarray(cache_data["latitude"].values)[rows]
        longitude = np.asarray(cache_data["longitude"].values)[rows]
        station_coordinates = np.unique(np.column_stack((latitude, longitude)), axis=0)
        # The packaged T30 terrain uses 0--360 degrees longitude and Gaussian latitudes.
        selected_cells = []
        for station_latitude, station_longitude in station_coordinates:
            cell = terrain_data.sel(
                lon=float(station_longitude) % 360.0,
                lat=float(station_latitude),
                method="nearest",
            )
            selected_cells.append((float(cell["lat"].values), float(cell["lon"].values)))
        if len(set(selected_cells)) != 1:
            raise ValueError(f"site {site!r} resolves to multiple native T30 cells")
        t30_latitude, t30_longitude = selected_cells[0]
        cell = terrain_data.sel(lon=t30_longitude, lat=t30_latitude)
        fmask = float(cell["lsm"].values)
        values[site] = {
            "station_coordinates": [
                {"latitude": float(lat), "longitude_0_360": float(lon) % 360.0}
                for lat, lon in station_coordinates
            ],
            "t30_latitude": t30_latitude,
            "t30_longitude_0_360": t30_longitude,
            "orog_m": float(cell["orog"].values),
            "fmask": fmask,
            # TerrainData.from_file enables land surface fluxes for the T30 package.
            # This is a global SPEEDY switch, not a cell-specific terrain field.
            "lfluxland": True,
        }
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--terrain-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    values = nearest_t30_site_terrain(args.cache, args.terrain_file)
    payload = {
        "source": {
            "cache": str(args.cache.resolve()),
            "terrain_file": str(args.terrain_file.resolve()),
            "selection": "nearest native T30 Gaussian-grid cell",
            "lfluxland": "true, matching TerrainData.from_file default for packaged T30 terrain",
        },
        "sites": values,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
