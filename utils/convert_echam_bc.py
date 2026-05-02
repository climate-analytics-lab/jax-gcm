#!/usr/bin/env python3
r"""Convert ECHAM-style boundary-condition NetCDFs into JCM-canonical layout.

Run once per ECHAM dataset (e.g. T63 AMIP climatology). The output files
plug into the standard ``terrain=from_file`` and ``forcing=from_file``
config groups, so the runtime path no longer needs to know about ECHAM
naming or axis conventions.

Differences from the JCM-canonical NetCDF layout that this script fixes:

1. **Variable names.** ECHAM surface files use uppercase
   (``OROMEA``, ``OROSTD``, …, ``SLM``/``SLF``); JCM uses lowercase
   (``orog``, ``orostd``, …, ``lsm``). When both ``SLF`` (fractional)
   and ``SLM`` (binary) are present, ``SLF`` wins.
2. **Axis order.** ECHAM files write ``(time, lat, lon)`` /
   ``(lat, lon)``; JCM expects ``(lon, lat, time)`` /
   ``(lon, lat)``.
3. **Latitude direction.** ECHAM goes north-to-south (descending);
   dinosaur grids go south-to-north (ascending).
4. **AMIP unit conventions.** ECHAM AMIP sea-ice is written in percent;
   JCM uses fraction in [0, 1].
5. **Monthly timestamps.** ECHAM AMIP timestamps the monthly mean at
   the month centre (e.g. ``1979-01-16T12:00``); the JCM time
   interpolator wants month-start (``1979-01-01``) so pandas detects
   monthly frequency.
6. **Coordinate alignment across companion files.** SST/SIC/surface
   files are written on the same Gaussian latitudes but disagree in
   the last few bits (file → xarray round-trip). xarray would mask
   the mismatched rows as NaN when combined; we force one canonical
   set of (lat, lon) coords.

Usage::

    python utils/convert_echam_bc.py \\
        --surface T63GR15_jan_surf.nc \\
        --sst T63_amipsst_1979-2008_mean.nc \\
        --sic T63_amipsic_1979-2008_mean.nc \\
        --out-dir jcm/data/bc/t63/

Produces, in ``--out-dir``:

- ``terrain.nc`` — ``orog``, ``lsm``, plus the six SSO descriptors if
  present in the surface file (``orostd``/``orosig``/``orogam``/
  ``orothe``/``oropic``/``oroval``).
- ``forcing.nc`` — ``sst``, ``icec``, ``stl``, ``alb``, ``soilw_am``,
  ``snowc`` on a 12-month axis.

Either ``--surface`` (terrain only) or ``--sst`` + ``--sic`` +
``--surface`` (terrain + forcing) is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


# ECHAM uppercase → JCM lowercase. ``SLF`` (fractional, 0..1) is preferred
# over ``SLM`` (binary 0/1) when both are present.
_ECHAM_TO_JCM_NAMES = {
    "SLM": "lsm",
    "SLF": "lsm",
    "OROMEA": "orog",
    "OROSTD": "orostd",
    "OROSIG": "orosig",
    "OROGAM": "orogam",
    "OROTHE": "orothe",
    "OROPIC": "oropic",
    "OROVAL": "oroval",
}

_SSO_NAMES = ("orostd", "orosig", "orogam", "orothe", "oropic", "oroval")


def _normalize(ds: xr.Dataset) -> xr.Dataset:
    """Translate uppercase + (lat, lon) + descending-lat → JCM canonical."""
    if "SLF" in ds and "SLM" in ds:
        ds = ds.drop_vars("SLM")
    rename = {k: v for k, v in _ECHAM_TO_JCM_NAMES.items() if k in ds}
    if rename:
        ds = ds.rename(rename)
    if "lat" in ds.dims and "lon" in ds.dims:
        ds = ds.transpose("lon", "lat", ...)
        if float(ds["lat"][0]) > float(ds["lat"][-1]):
            ds = ds.isel(lat=slice(None, None, -1))
    return ds


def _build_terrain(surface_ds: xr.Dataset) -> xr.Dataset:
    if "orog" not in surface_ds or "lsm" not in surface_ds:
        raise ValueError(
            "Surface file is missing OROMEA (→orog) or SLM/SLF (→lsm)."
        )
    out = xr.Dataset({
        "orog": surface_ds["orog"].astype("float32"),
        "lsm": surface_ds["lsm"].astype("float32"),
    })
    for name in _SSO_NAMES:
        if name in surface_ds:
            out[name] = surface_ds[name].astype("float32")
    return out


def _build_forcing(
    sst_ds: xr.Dataset,
    sic_ds: xr.Dataset,
    surface_ds: xr.Dataset,
) -> xr.Dataset:
    # Force shared (lat, lon) coords from SST so xarray doesn't mask rows
    # where the three files disagree in the last few bits.
    sic_ds = sic_ds.assign_coords(lat=sst_ds["lat"], lon=sst_ds["lon"])
    surface_ds = surface_ds.assign_coords(lat=sst_ds["lat"], lon=sst_ds["lon"])

    sic = sic_ds["sic"].clip(0.0, 100.0) * 0.01

    time = sst_ds["time"]
    ones_t = xr.ones_like(time, dtype="float32")
    ws = surface_ds["WS"] if "WS" in surface_ds else surface_ds["lsm"] * 0.0
    sn = surface_ds["SN"] if "SN" in surface_ds else surface_ds["lsm"] * 0.0
    alb = surface_ds["ALB"] if "ALB" in surface_ds else surface_ds["lsm"] * 0.0

    soilw_t = (ws * ones_t).transpose("lon", "lat", "time")
    snowc_t = (sn * ones_t).transpose("lon", "lat", "time")
    # Land surface T proxy = SST (extrapolated over land in the AMIP file).
    stl_t = sst_ds["sst"].transpose("lon", "lat", "time")

    ds = xr.Dataset({
        "sst":      sst_ds["sst"].transpose("lon", "lat", "time").astype("float32"),
        "icec":     sic.transpose("lon", "lat", "time").astype("float32"),
        "stl":      stl_t.astype("float32"),
        "soilw_am": soilw_t.astype("float32"),
        "snowc":    snowc_t.astype("float32"),
        "alb":      alb.transpose("lon", "lat").astype("float32"),
    })
    snapped = pd.to_datetime(ds["time"].values).to_period("M").to_timestamp()
    return ds.assign_coords(time=snapped)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", required=True, type=Path,
                   help="ECHAM surface file (e.g. T63GR15_jan_surf.nc)")
    p.add_argument("--sst", type=Path,
                   help="AMIP monthly SST file (omit for terrain-only)")
    p.add_argument("--sic", type=Path,
                   help="AMIP monthly sea-ice file (omit for terrain-only)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory (created if missing)")
    args = p.parse_args()

    if (args.sst is None) != (args.sic is None):
        p.error("--sst and --sic must be given together")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    surface_ds = _normalize(xr.open_dataset(args.surface))

    terrain_path = args.out_dir / "terrain.nc"
    _build_terrain(surface_ds).to_netcdf(terrain_path)
    print(f"wrote {terrain_path}")

    if args.sst is not None:
        sst_ds = _normalize(xr.open_dataset(args.sst))
        sic_ds = _normalize(xr.open_dataset(args.sic))
        forcing_path = args.out_dir / "forcing.nc"
        _build_forcing(sst_ds, sic_ds, surface_ds).to_netcdf(forcing_path)
        print(f"wrote {forcing_path}")


if __name__ == "__main__":
    main()
