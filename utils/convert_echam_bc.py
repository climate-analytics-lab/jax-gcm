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
        --land-init ic_land_soil_T63GR15_1976.nc \\
        --out-dir jcm/data/bc/t63/

Produces, in ``--out-dir``:

- ``terrain.nc`` — ``orog``, ``lsm``, plus the six SSO descriptors if
  present in the surface file (``orostd``/``orosig``/``orogam``/
  ``orothe``/``oropic``/``oroval``).
- ``forcing.nc`` — ``sst``, ``icec``, ``stl``, ``alb``, ``soilw_am``,
  ``snowc`` on a 12-month axis.

Either ``--surface`` (terrain only) or ``--sst`` + ``--sic`` +
``--surface`` (terrain + forcing) is required.

If ``--land-init`` is provided (the standard ECHAM
``ic_land_soil_T63GR15_*.nc`` JSBACH initial-conditions file), the
``stl`` field uses the **real monthly land surface temperature
climatology** (variable ``surf_temp``, 12-month) and the soil moisture
fields use ``init_moist`` / ``layer_moist`` rather than the AMIP-SST
extrapolation. Without ``--land-init``, ``stl`` falls back to the
AMIP SST extrapolation (note: gives ~+30 K bias over Antarctic
plateau and similar over the Tibetan/Andean plateaus, requiring the
lapse-rate workaround in ``echam_physics.apply_surface``). With it,
the workaround is unnecessary.
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
    land_ds: xr.Dataset | None = None,
) -> xr.Dataset:
    # Force shared (lat, lon) coords from SST so xarray doesn't mask rows
    # where the three files disagree in the last few bits.
    sic_ds = sic_ds.assign_coords(lat=sst_ds["lat"], lon=sst_ds["lon"])
    surface_ds = surface_ds.assign_coords(lat=sst_ds["lat"], lon=sst_ds["lon"])
    if land_ds is not None:
        land_ds = land_ds.assign_coords(lat=sst_ds["lat"], lon=sst_ds["lon"])

    sic = sic_ds["sic"].clip(0.0, 100.0) * 0.01

    time = sst_ds["time"]
    ones_t = xr.ones_like(time, dtype="float32")
    alb = surface_ds["ALB"] if "ALB" in surface_ds else surface_ds["lsm"] * 0.0

    if land_ds is not None and "surf_temp" in land_ds:
        # ``surf_temp`` is the JSBACH monthly land surface temperature
        # climatology (12, lat, lon). Use it directly instead of the
        # AMIP SST extrapolation. Re-stamp the time axis to the SST
        # months so the JCM forcing interpolator sees a single time
        # dimension.
        stl_src = land_ds["surf_temp"].transpose("time", "lat", "lon")
        # Map ``land_ds.time`` (12 months) onto ``sst_ds.time`` (also 12
        # months for AMIP climatology). They have identical length but
        # different epoch encodings.
        stl_t = stl_src.assign_coords(time=time).transpose("lon", "lat", "time")
    else:
        # Fallback: AMIP SST extrapolated over land. Gives ~+30 K bias
        # at high orography (Tibetan / Antarctic plateau) — needs the
        # lapse-rate workaround in ``apply_surface``.
        stl_t = sst_ds["sst"].transpose("lon", "lat", "time")

    if land_ds is not None and "init_moist" in land_ds:
        # JSBACH-initialised soil wetness (m). Single-time field broadcast
        # across the 12-month axis. ``layer_moist`` is also available
        # (5 soil layers) but JCM's surface scheme currently only uses
        # the column-integrated soilw_am.
        ws = land_ds["init_moist"]
    else:
        ws = surface_ds["WS"] if "WS" in surface_ds else surface_ds["lsm"] * 0.0
    if land_ds is not None and "snow" in land_ds:
        sn = land_ds["snow"]
    else:
        sn = surface_ds["SN"] if "SN" in surface_ds else surface_ds["lsm"] * 0.0
    soilw_t = (ws * ones_t).transpose("lon", "lat", "time")
    snowc_t = (sn * ones_t).transpose("lon", "lat", "time")

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
    p.add_argument("--land-init", type=Path,
                   help="JSBACH land initial-conditions file "
                        "(e.g. ic_land_soil_T63GR15_1976.nc) — provides the "
                        "real monthly land T climatology + initial soil "
                        "moisture. Optional but strongly recommended; without "
                        "it ``stl`` falls back to AMIP-SST extrapolation.")
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
        # JSBACH IC file has time encoded with a pre-1582 ``1-1-1`` reference
        # date that xarray decodes via cftime by default; ask for non-decoded
        # times so the SST monthly axis substitution below stays simple.
        land_ds = (
            _normalize(xr.open_dataset(args.land_init, decode_times=False))
            if args.land_init is not None
            else None
        )
        forcing_path = args.out_dir / "forcing.nc"
        _build_forcing(sst_ds, sic_ds, surface_ds, land_ds).to_netcdf(forcing_path)
        print(f"wrote {forcing_path}")


if __name__ == "__main__":
    main()
