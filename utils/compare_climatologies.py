"""Compare climatology between two simulation outputs.

Used to verify a JAX physics change doesn't break climatology — specifically
zonal-mean temperature, humidity, surface fluxes, and precipitation. Reports
mean / max-abs diff per field and a few key climate metrics (tropical
precipitation, polar T, etc.).

Example::

    python utils/compare_climatologies.py \\
        icon_t85_47_bd_30d.nc icon_t85_47_louis_30d.nc \\
        --label-a businger_dyer --label-b echam_louis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def _last_n_days_mean(ds: xr.Dataset, var: str, days: int = 10) -> xr.DataArray:
    """Time-mean of the last ``days`` of the timeseries (climatology proxy)."""
    if var not in ds:
        return None
    da = ds[var]
    n_keep = min(days, da.sizes.get("time", 1))
    return da.isel(time=slice(-n_keep, None)).mean(dim="time")


def _surface_2d_zonal(da: xr.DataArray) -> np.ndarray:
    """Reduce a (time-mean) DataArray to a 1D zonal-mean profile along lat.

    For 3D fields (level, lat, lon), pick the bottom level (last index)
    first, then take the zonal mean. For 2D (lat, lon) fields just take
    the zonal mean directly.
    """
    if da is None:
        return None
    if "level" in da.dims:
        da = da.isel(level=-1)
    if "longitude" in da.dims:
        return da.mean(dim="longitude").values
    if "lon" in da.dims:
        return da.mean(dim="lon").values
    return da.values


def _get_lat(ds: xr.Dataset) -> np.ndarray:
    for k in ("latitude", "lat"):
        if k in ds.coords:
            return ds[k].values
    return None


def _summary_stats(da: xr.DataArray, scale: float = 1.0, name: str = "") -> str:
    if da is None:
        return f"  {name}: <missing>"
    arr = da.values * scale
    return (f"  {name}: min={float(np.nanmin(arr)):.4g}  "
            f"max={float(np.nanmax(arr)):.4g}  "
            f"mean={float(np.nanmean(arr)):.4g}  "
            f"NaNs={int(np.isnan(arr).sum())}")


def _print_zonal_diff(label_a, label_b, var, dsA, dsB, lat,
                      scale=1.0, units="", days=10):
    da_a = _last_n_days_mean(dsA, var, days=days)
    da_b = _last_n_days_mean(dsB, var, days=days)
    if da_a is None or da_b is None:
        return
    za_s = _surface_2d_zonal(da_a) * scale
    zb_s = _surface_2d_zonal(da_b) * scale
    if za_s.shape != zb_s.shape:
        print(f"  {var}: shape mismatch {za_s.shape} vs {zb_s.shape}")
        return
    diff_s = zb_s - za_s
    print(f"\n  ``{var}`` ({units}, last {days}-day zonal-mean, "
          f"bottom level if 3-D) — {label_a} vs {label_b}")
    bands = [(-90, -60, "S polar"), (-60, -30, "S mid"),
             (-30, 30, "tropics"), (30, 60, "N mid"), (60, 90, "N polar")]
    for lo, hi, name in bands:
        sel = (lat >= lo) & (lat <= hi)
        if not sel.any():
            continue
        print(f"    {name:>9s} ({lo:>+3.0f}..{hi:>+3.0f}°)  "
              f"{label_a}={float(np.nanmean(za_s[sel])):+10.4g}  "
              f"{label_b}={float(np.nanmean(zb_s[sel])):+10.4g}  "
              f"Δ={float(np.nanmean(diff_s[sel])):+10.4g}")


def _check_precip(label_a, label_b, dsA, dsB, lat, days=10):
    """Tropics-vs-polar precipitation sanity check."""
    print(f"\n--- Precipitation climatology ({label_a} vs {label_b}) ---")
    for var in ("convection.precip_conv", "clouds.precip_rain",
                "clouds.precip_snow", "precip_conv", "precip_rain"):
        if var in dsA and var in dsB:
            _print_zonal_diff(label_a, label_b, var, dsA, dsB, lat,
                              scale=86400.0, units="mm/day", days=days)


def _check_temperature(label_a, label_b, dsA, dsB, lat, days=10):
    print(f"\n--- Temperature climatology ({label_a} vs {label_b}) ---")
    if "temperature" in dsA and "temperature" in dsB:
        _print_zonal_diff(label_a, label_b, "temperature", dsA, dsB, lat,
                          units="K", days=days)


def _check_surface_fluxes(label_a, label_b, dsA, dsB, lat, days=10):
    print(f"\n--- Surface fluxes ({label_a} vs {label_b}) ---")
    candidates = [
        ("surface.sensible_heat_flux", "W/m²", 1.0),
        ("surface.latent_heat_flux", "W/m²", 1.0),
        ("vertical_diffusion.surface_friction_velocity", "m/s", 1.0),
        ("vdiff.surface_friction_velocity", "m/s", 1.0),
    ]
    for var, units, scale in candidates:
        if var in dsA and var in dsB:
            _print_zonal_diff(label_a, label_b, var, dsA, dsB, lat,
                              scale=scale, units=units, days=days)


def _check_state_summary(label, ds):
    print(f"\n--- {label} state summary (final) ---")
    if "temperature" in ds:
        print(_summary_stats(ds["temperature"].isel(time=-1), name="T (K)"))
    if "specific_humidity" in ds:
        # netCDF saves q in g/kg by convention here
        units = ds["specific_humidity"].attrs.get("units", "g/kg")
        scale = 1.0 if units == "g/kg" else 1000.0
        print(_summary_stats(ds["specific_humidity"].isel(time=-1),
                             scale=scale, name="q (g/kg)"))
    if "u_wind" in ds:
        print(_summary_stats(ds["u_wind"].isel(time=-1), name="u (m/s)"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file_a")
    p.add_argument("file_b")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--days", type=int, default=10,
                   help="Use the last N days for climatology")
    args = p.parse_args()

    pa, pb = Path(args.file_a), Path(args.file_b)
    if not pa.exists():
        sys.exit(f"missing: {pa}")
    if not pb.exists():
        sys.exit(f"missing: {pb}")
    dsA = xr.open_dataset(pa)
    dsB = xr.open_dataset(pb)
    lat = _get_lat(dsA)
    if lat is None:
        sys.exit("no latitude coord")

    print(f"=== {args.label_a} ({pa.name}) vs {args.label_b} ({pb.name}) ===")
    print(f"  ntime A = {dsA.sizes.get('time')}, B = {dsB.sizes.get('time')}")

    _check_state_summary(args.label_a, dsA)
    _check_state_summary(args.label_b, dsB)
    _check_temperature(args.label_a, args.label_b, dsA, dsB, lat, args.days)
    _check_precip(args.label_a, args.label_b, dsA, dsB, lat, args.days)
    _check_surface_fluxes(args.label_a, args.label_b, dsA, dsB, lat, args.days)


if __name__ == "__main__":
    main()
