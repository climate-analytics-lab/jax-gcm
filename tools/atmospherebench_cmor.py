r"""Post-process JCM output into AtmosphereBench / CMIP6 CMOR NetCDF files.

Reads one or more JCM netCDF files — or a JCM run directory containing
``*_monthly_YYYY-MM.nc`` and/or ``*_day*.nc`` chunk files — and emits **one
variable per file** following the AtmosphereBench submission layout and CMIP6
CMOR naming convention::

    <out>/<experiment>/<member>/          # member experiments: amip, lgm, amip-hist
        <var>_Amon_<model>_<experiment>_<member>_<grid>_<YYYYMM>-<YYYYMM>.nc
        <var>_day_<model>_<experiment>_<member>_<grid>_<YYYYMMDD>-<YYYYMMDD>.nc
        sftlf_fx_<model>_<experiment>_<member>_<grid>.nc
        areacella_fx_<model>_<experiment>_<member>_<grid>.nc

    <out>/<experiment>/                   # single-member experiments (amip-4xCO2, amip-p4K, ...)
        <var>_Amon_<model>_<experiment>_<member>_<grid>_<YYYYMM>-<YYYYMM>.nc
        ...

Pass ``--flat`` to write directly into ``<out>`` (useful for ``atmospherebench dev``).

Examples
--------
Convert a completed JCM run directory (auto-detecting monthly and daily files)::

    python tools/atmospherebench_cmor.py /path/to/jcm-run \
        --model JCM-ECHAM-1M --experiment amip --member r1i1p1f1 \
        --out submission/

Convert explicit monthly files only::

    python tools/atmospherebench_cmor.py run_monthly_1979-*.nc \
        --model JCM-ECHAM-1M --experiment amip --freq mon --out submission/

"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import re
import sys
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import xarray as xr

# Allow running directly as ``python tools/atmospherebench_cmor.py`` without
# requiring ``PYTHONPATH=.``.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jcm.constants as c
from jcm.analysis import (
    area_weights,
    column_burden,
    to_pressure_levels,
    validate_pressure_levels,
)

#: WMO standard gravity [m s-2] for CMIP6 geopotential height ``zg = Phi / g0``
#: (distinct from ``jcm.constants.grav`` used in hydrostatic column integrals).
STANDARD_GRAVITY = 9.80665

#: Experiments whose AtmosphereBench layout places files under ``<exp>/<member>/``.
MEMBER_EXPERIMENTS = frozenset({"amip", "lgm", "amip-hist"})

#: Standard CMIP6 19-level pressure grid [hPa] for monthly 3D fields (covers
#: 850/700/200 hPa and the full tropospheric/stratospheric column for MSU).
DEFAULT_MONTHLY_PLEV_HPA: tuple[float, ...] = (
    1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0,
    200.0, 150.0, 100.0, 70.0, 50.0, 30.0, 20.0, 10.0, 5.0, 1.0,
)

#: Compact pressure levels [hPa] for daily 3D fields (geostrophic balance at
#: 850 hPa and MJO diagnostics at 850/200 hPa).
DEFAULT_DAILY_PLEV_HPA: tuple[float, ...] = (850.0, 200.0)

#: Default daily variable set requested by AtmosphereBench.
DEFAULT_DAILY_VARS: tuple[str, ...] = (
    "pr", "tas", "tasmax", "tasmin", "rlut", "ua", "zg",
)

#: Fixed (time-invariant) CMOR grid fields.
FX_VARS = frozenset({"sftlf", "areacella"})

#: 3D CMOR variables that live on ``plev`` when exported.
PLEV_VARS = frozenset({"ta", "hus", "hur", "ua", "va", "zg"})

#: Source variable(s) needed before calling ``to_pressure_levels`` for each 3D CMOR var.
_PLEV_SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ta": ("temperature",),
    "hus": ("specific_humidity",),
    "hur": ("relative_humidity",),
    "ua": ("u_wind",),
    "va": ("v_wind",),
    "zg": ("height_full", "geopotential"),
}

_VARIANT_RE = re.compile(r"^r\d+i\d+p\d+f\d+$")
_MONTHLY_FILE_RE = re.compile(r"^(?P<prefix>.+)_monthly_\d{4}-\d{2}\.nc$")
_CHUNK_FILE_RE = re.compile(r"^(?P<prefix>.+)_day(?P<day>\d+(?:\.\d+)?)\.nc$")

#: CF metadata for every CMOR variable this converter can emit:
#: ``cmor_name -> (units, standard_name, long_name)``.
CMOR_META: dict[str, tuple[str, str, str]] = {
    "tas": ("K", "air_temperature", "Near-Surface Air Temperature"),
    "tasmax": ("K", "air_temperature", "Daily Maximum Near-Surface Air Temperature"),
    "tasmin": ("K", "air_temperature", "Daily Minimum Near-Surface Air Temperature"),
    "ts": ("K", "surface_temperature", "Surface Temperature"),
    "pr": ("kg m-2 s-1", "precipitation_flux", "Precipitation"),
    "evspsbl": ("kg m-2 s-1", "water_evapotranspiration_flux", "Evaporation Including Sublimation and Transpiration"),
    "hfls": ("W m-2", "surface_upward_latent_heat_flux", "Surface Upward Latent Heat Flux"),
    "hfss": ("W m-2", "surface_upward_sensible_heat_flux", "Surface Upward Sensible Heat Flux"),
    "rsdt": ("W m-2", "toa_incoming_shortwave_flux", "TOA Incident Shortwave Radiation"),
    "rsut": ("W m-2", "toa_outgoing_shortwave_flux", "TOA Outgoing Shortwave Radiation"),
    "rlut": ("W m-2", "toa_outgoing_longwave_flux", "TOA Outgoing Longwave Radiation"),
    "rsutcs": ("W m-2", "toa_outgoing_shortwave_flux_assuming_clear_sky", "TOA Outgoing Clear-Sky Shortwave Radiation"),
    "rlutcs": ("W m-2", "toa_outgoing_longwave_flux_assuming_clear_sky", "TOA Outgoing Clear-Sky Longwave Radiation"),
    "rsds": ("W m-2", "surface_downwelling_shortwave_flux_in_air", "Surface Downwelling Shortwave Radiation"),
    "rsus": ("W m-2", "surface_upwelling_shortwave_flux_in_air", "Surface Upwelling Shortwave Radiation"),
    "rlds": ("W m-2", "surface_downwelling_longwave_flux_in_air", "Surface Downwelling Longwave Radiation"),
    "rlus": ("W m-2", "surface_upwelling_longwave_flux_in_air", "Surface Upwelling Longwave Radiation"),
    "clt": ("%", "cloud_area_fraction", "Total Cloud Cover Percentage"),
    "prw": ("kg m-2", "atmosphere_mass_content_of_water_vapor", "Water Vapor Path"),
    "ps": ("Pa", "surface_air_pressure", "Surface Air Pressure"),
    "psl": ("Pa", "air_pressure_at_mean_sea_level", "Sea Level Pressure"),
    "huss": ("1", "specific_humidity", "Near-Surface Specific Humidity"),
    "uas": ("m s-1", "eastward_wind", "Eastward Near-Surface Wind"),
    "vas": ("m s-1", "northward_wind", "Northward Near-Surface Wind"),
    "ta": ("K", "air_temperature", "Air Temperature"),
    "hus": ("1", "specific_humidity", "Specific Humidity"),
    "hur": ("%", "relative_humidity", "Relative Humidity"),
    "ua": ("m s-1", "eastward_wind", "Eastward Wind"),
    "va": ("m s-1", "northward_wind", "Northward Wind"),
    "zg": ("m", "geopotential_height", "Geopotential Height"),
    "sftlf": ("%", "land_area_fraction", "Percentage of the Grid Cell Occupied by Land (Including Lakes)"),
    "areacella": ("m2", "cell_area", "Grid-Cell Area for Atmospheric Grid Variables"),
}

#: CF ``positive`` direction attribute for radiative and turbulent fluxes.
_FLUX_POSITIVE: dict[str, str] = {
    "rsdt": "down",
    "rsut": "up",
    "rlut": "up",
    "rsutcs": "up",
    "rlutcs": "up",
    "rsds": "down",
    "rsus": "up",
    "rlds": "down",
    "rlus": "up",
    "hfls": "up",
    "hfss": "up",
}

#: Direct JCM -> CMOR mappings with fallback priority (first present key wins):
#: ``cmor_name -> ((jcm_name, scale), ...)``.
_DIRECT_SOURCES: dict[str, tuple[tuple[str, float], ...]] = {
    "ts": (("surface.surface_temperature", 1.0), ("surface.skin_temperature", 1.0)),
    "evspsbl": (
        ("surface.evaporation", 1.0),
        ("surface_exchange.evaporation", 1.0),
        ("vertical_diffusion.surface_evaporation", 1.0),
    ),
    "hfls": (("surface.latent_heat_flux", 1.0), ("surface_exchange.latent_heat_flux", 1.0)),
    "hfss": (
        ("surface.sensible_heat_flux_dry", 1.0),
        ("surface.sensible_heat_flux", 1.0),
        ("surface_exchange.sensible_heat_flux", 1.0),
    ),
    "rsdt": (("radiation.toa_sw_down", 1.0),),
    "rsut": (("radiation.toa_sw_up", 1.0),),
    "rlut": (("radiation.toa_lw_up", 1.0), ("longwave_rad.ftop", 1.0)),
    "rsutcs": (("radiation.toa_sw_up_clear", 1.0), ("clouds.toa_sw_up_clear", 1.0)),
    "rlutcs": (("radiation.toa_lw_up_clear", 1.0), ("clouds.toa_lw_up_clear", 1.0)),
    "rsds": (("radiation.surface_sw_down", 1.0), ("surface_exchange.downward_shortwave_flux", 1.0)),
    "rsus": (("radiation.surface_sw_up", 1.0),),
    "rlds": (("radiation.surface_lw_down", 1.0), ("surface_exchange.downward_longwave_flux", 1.0)),
    "rlus": (("radiation.surface_lw_up", 1.0),),
    "ps": (("surface_pressure", 1.0), ("surface_exchange.surface_pressure", 1.0)),
    "psl": (("aerocom_psl", 1.0),),
    "huss": (("surface.humidity_2m", 1.0),),
    "uas": (("vertical_diffusion.wind_10m_u", 1.0), ("aerocom_uas", 1.0)),
    "vas": (("vertical_diffusion.wind_10m_v", 1.0), ("aerocom_vas", 1.0)),
    "tasmax": (("tasmax", 1.0),),
    "tasmin": (("tasmin", 1.0),),
}


def _is_percent_units(da: xr.DataArray) -> bool:
    units = str(da.attrs.get("units", "")).strip().lower()
    return units in ("%", "percent")


def _extract_tas(ds: xr.Dataset) -> xr.DataArray | None:
    for key in ("surface.temperature_2m", "aerocom_tas"):
        if key in ds.data_vars:
            return ds[key]
    if "temperature" in ds.data_vars and "level" in ds["temperature"].dims:
        temp = ds["temperature"]
        if "pressure_full" in ds.data_vars and "level" in ds["pressure_full"].dims:
            # Pick the highest-pressure (lowest-altitude) level regardless of
            # whether the vertical axis is surface-first or TOA-first.
            mean_p = ds["pressure_full"].mean(
                [d for d in ds["pressure_full"].dims if d != "level"]
            )
            sfc_idx = int(np.argmax(np.asarray(mean_p.values)))
        else:
            sfc_idx = 0
        warnings.warn(
            "Falling back to lowest-model-level 'temperature' for CMOR 'tas' "
            "because neither 'surface.temperature_2m' nor 'aerocom_tas' is present; "
            "this approximates 2 m air temperature.",
            UserWarning,
            stacklevel=2,
        )
        return temp.isel(level=sfc_idx, drop=True)
    return None


def _extract_clt(ds: xr.Dataset) -> xr.DataArray | None:
    for key in (
        "clouds.total_cloud_cover",
        "aerocom_clt",
        "radiation.total_cloud_cover",
        "shortwave_rad.cloudc",
    ):
        if key in ds.data_vars:
            da = ds[key]
            if not _is_percent_units(da):
                da = da * 100.0
            return da.clip(0.0, 100.0)
    return None


def _extract_pr(ds: xr.Dataset) -> xr.DataArray | None:
    if "surface_exchange.precipitation" in ds.data_vars:
        return ds["surface_exchange.precipitation"]
    echam_parts = [
        ds[k]
        for k in ("clouds.precip_rain", "clouds.precip_snow", "convection.precip_conv")
        if k in ds.data_vars
    ]
    if echam_parts:
        out = echam_parts[0]
        for part in echam_parts[1:]:
            out = out + part
        return out
    speedy_parts = [
        ds[k]
        for k in ("condensation.precls", "convection.precnv")
        if k in ds.data_vars
    ]
    if speedy_parts:
        out = speedy_parts[0]
        for part in speedy_parts[1:]:
            out = out + part
        # SPEEDY stores precipitation in g m-2 s-1 (= 1e-3 kg m-2 s-1).
        return out * 1e-3
    return None


def _extract_prw(ds: xr.Dataset) -> xr.DataArray | None:
    if "aerocom_prw" in ds.data_vars:
        return ds["aerocom_prw"]
    if "specific_humidity" not in ds.data_vars or "level" not in ds["specific_humidity"].dims:
        return None
    if "pressure_thickness" not in ds.data_vars and "pressure_half" not in ds.data_vars:
        return None
    return column_burden(ds, "specific_humidity")


def _extract_sftlf(ds: xr.Dataset) -> xr.DataArray | None:
    # Prefer dynamic discovery via ``surface_type="land"`` (works for both
    # ECHAM's 3-tile ``(water, sea_ice, land)`` and SPEEDY's 2-tile ``(sea, land)``).
    da: xr.DataArray | None = None
    for name in sorted(ds.data_vars):
        if str(name).startswith("surface_exchange.tile_fraction."):
            if ds[name].attrs.get("surface_type") == "land":
                da = ds[name]
                break
    if da is None:
        for key in ("surface_exchange.tile_fraction.2", "vertical_diffusion.surface_fraction.2"):
            if key in ds.data_vars:
                da = ds[key]
                break
    if da is None:
        return None
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    if not _is_percent_units(da):
        da = da * 100.0
    return da.clip(0.0, 100.0)


def _compute_areacella(ds: xr.Dataset) -> xr.DataArray | None:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    w = area_weights(lat)
    w_norm = w / w.sum()
    total_area_m2 = 4.0 * np.pi * c.rearth**2
    cell_area_by_lat = w_norm.values * (total_area_m2 / lon.size)
    area_2d = np.broadcast_to(cell_area_by_lat[:, None], (lat.size, lon.size)).astype(np.float64)
    return xr.DataArray(area_2d, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


_DERIVED_EXTRACTORS: dict[str, Callable[[xr.Dataset], xr.DataArray | None]] = {
    "tas": _extract_tas,
    "clt": _extract_clt,
    "pr": _extract_pr,
    "prw": _extract_prw,
}


def _check_token(flag: str, value: str) -> str:
    if not value or "_" in value or "/" in value:
        raise ValueError(
            f"{flag}={value!r}: CMOR filename tokens must be non-empty and "
            "must not contain '_' or '/' (use '-' instead)."
        )
    return value


def _check_member(member: str) -> str:
    if not _VARIANT_RE.match(member):
        raise ValueError(
            f"--member={member!r}: expected CMIP variant label like 'r1i1p1f1'."
        )
    return member


def _date_ymd(val) -> tuple[int, int, int]:
    """Extract ``(year, month, day)`` from a numpy datetime64, Timestamp, or cftime scalar."""
    if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        return int(val.year), int(val.month), int(val.day)
    if isinstance(val, np.datetime64):
        s = str(np.datetime_as_string(val, unit="D"))
        y, m, d = s.split("-")
        return int(y), int(m), int(d)
    raise TypeError(f"Unsupported time coordinate value: {val!r} ({type(val).__name__})")


def _get_time_bounds(ds: xr.Dataset) -> xr.DataArray | None:
    """Return the CF time-bounds variable from ``ds`` (data_vars or coords), if present."""
    if "time" not in ds.coords:
        return None
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    if bounds_name in ds:
        return ds[bounds_name]
    if bounds_name != "time_bounds" and "time_bounds" in ds:
        return ds["time_bounds"]
    return None


def _timedelta_days(delta) -> float:
    """Convert a numpy timedelta64 or datetime.timedelta to float days."""
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s")) / 86400.0
    if isinstance(delta, datetime.timedelta):
        return delta.total_seconds() / 86400.0
    raise TypeError(f"Unsupported timedelta type: {type(delta).__name__}")


def _median_step_days(ds: xr.Dataset, time_bounds: xr.DataArray | None) -> float | None:
    """Estimate the median sampling/averaging interval [days] of ``ds['time']``."""
    if time_bounds is not None and time_bounds.ndim == 2 and time_bounds.shape[1] == 2:
        tb = time_bounds.values
        widths = [_timedelta_days(tb[i, 1] - tb[i, 0]) for i in range(tb.shape[0])]
        if widths:
            return float(np.median(widths))
    if "time" in ds.coords and ds.sizes.get("time", 0) >= 2:
        tvals = ds.coords["time"].values
        diffs = [_timedelta_days(tvals[i + 1] - tvals[i]) for i in range(len(tvals) - 1)]
        if diffs:
            return float(np.median(diffs))
    return None


def _resample_subdaily_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Aggregate a sub-daily JCM Dataset into daily means plus ``tasmax``/``tasmin``."""
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    drop_vars = [v for v in (bounds_name, "time_bounds") if v in ds]
    ds_no_bounds = ds.drop_vars(drop_vars) if drop_vars else ds

    # Extract raw sub-daily near-surface temperature before averaging so we can
    # compute daily tasmax and tasmin.
    tas_raw = _extract_tas(ds_no_bounds)
    daily = ds_no_bounds.resample(time="1D").mean(keep_attrs=True)
    if tas_raw is not None:
        if "tasmax" not in daily.data_vars:
            daily["tasmax"] = tas_raw.resample(time="1D").max(keep_attrs=True)
        if "tasmin" not in daily.data_vars:
            daily["tasmin"] = tas_raw.resample(time="1D").min(keep_attrs=True)

    tvals = daily.coords["time"].values
    if len(tvals) > 0:
        one_day = (
            np.timedelta64(1, "D")
            if isinstance(tvals[0], np.datetime64)
            else datetime.timedelta(days=1)
        )
        tb = np.stack([tvals, tvals + one_day], axis=-1)
        daily["time_bounds"] = xr.DataArray(tb, dims=("time", "bounds"))
        daily["time"].attrs["bounds"] = "time_bounds"
    return daily


def _time_stamp(
    ds: xr.Dataset,
    table: str,
    time_bounds: xr.DataArray | None = None,
) -> str:
    """Format ``<start>-<end>`` CMOR date token from ``ds['time']`` (or ``time_bounds``)."""
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        raise ValueError("Dataset has no non-empty 'time' coordinate.")
    if time_bounds is not None and time_bounds.ndim == 2 and time_bounds.shape[1] == 2:
        t0 = time_bounds.values[0, 0]
        t1 = time_bounds.values[-1, 0]
    else:
        t0 = ds.coords["time"].values[0]
        t1 = ds.coords["time"].values[-1]
    y0, m0, d0 = _date_ymd(t0)
    y1, m1, d1 = _date_ymd(t1)
    if table == "Amon":
        return f"{y0:04d}{m0:02d}-{y1:04d}{m1:02d}"
    return f"{y0:04d}{m0:02d}{d0:02d}-{y1:04d}{m1:02d}{d1:02d}"


def cmor_filename(
    var: str,
    table: str,
    model: str,
    experiment: str,
    member: str,
    grid_label: str,
    date_stamp: str | None = None,
) -> str:
    """Return the CMIP6 / AtmosphereBench CMOR filename."""
    if table == "fx" or not date_stamp:
        return f"{var}_{table}_{model}_{experiment}_{member}_{grid_label}.nc"
    return f"{var}_{table}_{model}_{experiment}_{member}_{grid_label}_{date_stamp}.nc"


def _cell_methods(cmor: str, table: str) -> str:
    """Return the CF-1.8 ``cell_methods`` string for ``(cmor, table)``."""
    if table == "fx":
        return "area: sum" if cmor == "areacella" else "area: mean"
    if cmor == "tasmax":
        return "area: mean time: maximum"
    if cmor == "tasmin":
        return "area: mean time: minimum"
    return "area: time: mean"


def _standardize_da(da: xr.DataArray, cmor: str, table: str) -> xr.DataArray:
    """Transpose to CMOR dim order, sort lat ascending, drop extra coords, set CF attrs."""
    # Drop JCM auxiliary coordinates that are not dimensions of this variable.
    extra_coords = [c for c in da.coords if c not in da.dims]
    if extra_coords:
        da = da.drop_vars(extra_coords)

    ordered_dims = [d for d in ("time", "plev", "lat", "lon") if d in da.dims]
    remaining = [d for d in da.dims if d not in ordered_dims]
    da = da.transpose(*ordered_dims, *remaining)

    if "lat" in da.dims and da.sizes["lat"] > 1:
        lat_vals = da.coords["lat"].values
        if lat_vals[0] > lat_vals[-1]:
            da = da.isel(lat=slice(None, None, -1))

    units, std_name, long_name = CMOR_META[cmor]
    da = da.rename(cmor)
    attrs: dict[str, str] = {
        "units": units,
        "standard_name": std_name,
        "long_name": long_name,
        "cell_methods": _cell_methods(cmor, table),
    }
    if cmor in _FLUX_POSITIVE:
        attrs["positive"] = _FLUX_POSITIVE[cmor]
    da.attrs = attrs
    return da


def _set_coord_cf_attrs(out: xr.Dataset) -> None:
    """Ensure every CMOR coordinate carries the CF attributes AtmosphereBench expects."""
    if "time" in out.coords:
        # Remove encoding-owned keys from attrs so xarray.to_netcdf does not
        # fail with "failed to prevent overwriting existing key units in attrs".
        out["time"].attrs.pop("units", None)
        out["time"].attrs.pop("calendar", None)
        out["time"].attrs.setdefault("standard_name", "time")
        out["time"].attrs.setdefault("long_name", "time")
        out["time"].attrs["axis"] = "T"
        if "time_bounds" in out.data_vars:
            out["time"].attrs["bounds"] = "time_bounds"
            out["time_bounds"].attrs.pop("units", None)
            out["time_bounds"].attrs.pop("calendar", None)
    if "lat" in out.coords:
        out["lat"].attrs["standard_name"] = "latitude"
        out["lat"].attrs["units"] = "degrees_north"
        out["lat"].attrs["axis"] = "Y"
        out["lat"].attrs.setdefault("long_name", "latitude")
    if "lon" in out.coords:
        out["lon"].attrs["standard_name"] = "longitude"
        out["lon"].attrs["units"] = "degrees_east"
        out["lon"].attrs["axis"] = "X"
        out["lon"].attrs.setdefault("long_name", "longitude")
    if "plev" in out.coords:
        out["plev"].attrs["standard_name"] = "air_pressure"
        out["plev"].attrs["units"] = "Pa"
        out["plev"].attrs["positive"] = "down"
        out["plev"].attrs["axis"] = "Z"
        out["plev"].attrs.setdefault("long_name", "pressure")


def _extract_single_plev_var(
    ds: xr.Dataset,
    cmor: str,
    plevs_hpa: Sequence[float],
    *,
    subset_preinterp: bool = False,
) -> xr.DataArray | None:
    """Extract and pressure-interpolate one 3D CMOR variable onto ``plevs_hpa`` [hPa]."""
    src: str | None = None
    for candidate in _PLEV_SOURCE_CANDIDATES[cmor]:
        if candidate in ds.data_vars:
            src = candidate
            break
    if src is None:
        return None

    da_src = ds[src]
    if "level" in da_src.dims:
        if "pressure_full" not in ds.data_vars:
            warnings.warn(
                f"Skipping 3D CMOR variable {cmor!r} (source {src!r}): "
                "'pressure_full' is required for vertical interpolation onto pressure levels.",
                UserWarning,
                stacklevel=2,
            )
            return None
        interp_ds = to_pressure_levels(ds[[src, "pressure_full"]], list(plevs_hpa))
        da = interp_ds[src]
    elif "plev" in da_src.dims:
        da = da_src
        plev_units = str(da.coords["plev"].attrs.get("units", "")).strip().lower()
        if plev_units in ("hpa", "mb", "millibar"):
            da = da.assign_coords(plev=da.coords["plev"].values.astype(np.float64) * 100.0)
        if subset_preinterp:
            target_pa = np.asarray(plevs_hpa, dtype=np.float64) * 100.0
            avail_pa = np.asarray(da.coords["plev"].values, dtype=np.float64)
            keep_idx = [
                int(np.argmin(np.abs(avail_pa - p)))
                for p in target_pa
                if np.min(np.abs(avail_pa - p)) <= 1e-3 * p
            ]
            if keep_idx:
                da = da.isel(plev=keep_idx)
            else:
                warnings.warn(
                    f"Pre-interpolated {cmor!r} has no pressure levels matching "
                    f"requested {list(plevs_hpa)} hPa; keeping available levels.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        return None

    if cmor == "hur" and not _is_percent_units(da):
        da = da * 100.0
    elif src == "geopotential":
        da = da / STANDARD_GRAVITY
    return da


def _time_encoding(out: xr.Dataset) -> dict[str, dict]:
    """Build NetCDF time encoding preserving cftime calendars when present."""
    if "time" not in out.coords:
        return {}
    t_enc: dict[str, str] = {"units": "days since 1850-01-01", "dtype": "float64"}
    cal = out["time"].encoding.get("calendar")
    if cal is None and out.sizes.get("time", 0) > 0:
        cal = getattr(out["time"].values[0], "calendar", None)
    if cal:
        t_enc["calendar"] = str(cal)
    enc: dict[str, dict] = {"time": t_enc}
    if "time_bounds" in out.data_vars:
        enc["time_bounds"] = dict(t_enc)
    return enc


def _write_single_var(
    da: xr.DataArray,
    *,
    cmor: str,
    table: str,
    model: str,
    experiment: str,
    member: str,
    grid_label: str,
    frequency: str,
    outdir: pathlib.Path,
    date_stamp: str | None = None,
    time_bounds: xr.DataArray | None = None,
    dry_run: bool = False,
) -> str:
    """Write one standardized CMOR variable NetCDF file atomically and return its filename."""
    std_da = _standardize_da(da, cmor, table)
    fname = cmor_filename(cmor, table, model, experiment, member, grid_label, date_stamp)
    if dry_run:
        return fname

    data_vars: dict[str, xr.DataArray] = {cmor: std_da}
    if time_bounds is not None and "time" in std_da.dims:
        tb = time_bounds
        extra = [c for c in tb.coords if c not in tb.dims]
        if extra:
            tb = tb.drop_vars(extra)
        data_vars["time_bounds"] = tb

    out = xr.Dataset(data_vars)
    if "time" in std_da.coords and "calendar" in std_da.coords["time"].encoding:
        out["time"].encoding["calendar"] = std_da.coords["time"].encoding["calendar"]
    _set_coord_cf_attrs(out)
    out.attrs = {
        "Conventions": "CF-1.8",
        "model_id": model,
        "source_id": model,
        "experiment_id": experiment,
        "variant_label": member,
        "table_id": table,
        "variable_id": cmor,
        "realm": "atmos",
        "frequency": frequency,
        "grid_label": grid_label,
        "comment": "Produced by tools/atmospherebench_cmor.py from JCM output.",
    }
    encoding: dict[str, dict] = {cmor: {"zlib": True, "complevel": 4}}
    encoding.update(_time_encoding(out))

    target = outdir / fname
    tmp_path = outdir / f".{fname}.tmp.{os.getpid()}"
    try:
        out.to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return fname


def _extract_non_plev_var(ds: xr.Dataset, cmor: str) -> xr.DataArray | None:
    if cmor in _DERIVED_EXTRACTORS:
        return _DERIVED_EXTRACTORS[cmor](ds)
    for src, scale in _DIRECT_SOURCES.get(cmor, ()):
        if src in ds.data_vars:
            da = ds[src]
            return da * scale if scale != 1.0 else da
    return None


def _slice_years(
    ds: xr.Dataset,
    start_year: int | None,
    end_year: int | None,
) -> xr.Dataset:
    if start_year is None and end_year is None:
        return ds
    if "time" not in ds.coords:
        raise ValueError("Cannot apply --start-year/--end-year: dataset has no 'time' coordinate.")
    years = ds["time"].dt.year
    mask = xr.ones_like(years, dtype=bool)
    if start_year is not None:
        mask = mask & (years >= start_year)
    if end_year is not None:
        mask = mask & (years <= end_year)
    sliced = ds.isel(time=np.nonzero(np.asarray(mask.values))[0])
    if sliced.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Year filter [{start_year}, {end_year}] left zero timesteps."
        )
    return sliced


def convert_dataset(
    ds: xr.Dataset,
    *,
    model: str,
    experiment: str,
    member: str = "r1i1p1f1",
    table: str = "Amon",
    grid_label: str = "gn",
    outdir: pathlib.Path,
    variables: Sequence[str] | None = None,
    plevs_hpa: Sequence[float] | None = None,
    write_fx: bool = True,
    fx_vars_filter: Sequence[str] | None = None,
    resample_daily: bool = True,
    start_year: int | None = None,
    end_year: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Convert one JCM Dataset into AtmosphereBench CMOR files; return written filenames."""
    _check_token("--model", model)
    _check_token("--experiment", experiment)
    _check_token("--grid-label", grid_label)
    _check_member(member)
    if table not in ("Amon", "day"):
        raise ValueError(f"Unsupported CMOR table {table!r}; expected 'Amon' or 'day'.")

    explicit_plevs = plevs_hpa is not None
    effective_plevs = (
        validate_pressure_levels(plevs_hpa).tolist()
        if plevs_hpa is not None
        else list(DEFAULT_DAILY_PLEV_HPA if table == "day" else DEFAULT_MONTHLY_PLEV_HPA)
    )

    frequency = "mon" if table == "Amon" else "day"
    explicit_vars = variables is not None
    if variables is None:
        if table == "day":
            target_vars = list(DEFAULT_DAILY_VARS)
        else:
            target_vars = [v for v in CMOR_META if v not in ("sftlf", "areacella", "tasmax", "tasmin")]
    else:
        unknown = [v for v in variables if v not in CMOR_META]
        if unknown:
            raise ValueError(f"Unknown CMOR variable(s): {', '.join(unknown)}")
        target_vars = list(dict.fromkeys(variables))

    ds = _slice_years(ds, start_year, end_year)
    time_bounds = _get_time_bounds(ds)

    step_days = _median_step_days(ds, time_bounds)
    if table == "Amon" and step_days is not None and step_days < 25.0:
        raise ValueError(
            f"Expected monthly data for table='Amon', but median time step is "
            f"{step_days:.2f} days. Pass --freq day for daily/sub-daily inputs."
        )
    if table == "day" and step_days is not None:
        if step_days > 1.5:
            raise ValueError(
                f"Expected daily or sub-daily data for table='day', but median "
                f"time step is {step_days:.2f} days."
            )
        if step_days < 0.9:
            if not resample_daily:
                raise ValueError(
                    f"Input for table='day' has sub-daily cadence ({step_days * 24:.2f} h). "
                    "Enable daily resampling (--resample-daily) to aggregate to daily means "
                    "and tasmax/tasmin."
                )
            print(
                f"INFO tools/atmospherebench_cmor.py: resampling sub-daily input "
                f"({step_days * 24:.2f} h cadence) to daily means (plus tasmax/tasmin).",
                file=sys.stderr,
            )
            ds = _resample_subdaily_to_daily(ds)
            time_bounds = _get_time_bounds(ds)

    if not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
    date_stamp = _time_stamp(ds, table, time_bounds=time_bounds)

    written: list[str] = []
    missing: list[str] = []

    # Stream one variable at a time to keep peak memory low on long runs.
    non_fx_targets = sorted(v for v in target_vars if v not in FX_VARS)
    for cmor in non_fx_targets:
        if cmor in PLEV_VARS:
            da = _extract_single_plev_var(
                ds, cmor, effective_plevs, subset_preinterp=explicit_plevs
            )
        else:
            da = _extract_non_plev_var(ds, cmor)
        if da is None:
            missing.append(cmor)
            continue
        written.append(
            _write_single_var(
                da,
                cmor=cmor,
                table=table,
                model=model,
                experiment=experiment,
                member=member,
                grid_label=grid_label,
                frequency=frequency,
                outdir=outdir,
                date_stamp=date_stamp,
                time_bounds=time_bounds,
                dry_run=dry_run,
            )
        )

    if write_fx:
        if fx_vars_filter is not None:
            fx_targets = sorted(v for v in fx_vars_filter if v in FX_VARS)
        elif explicit_vars:
            fx_targets = sorted(v for v in target_vars if v in FX_VARS)
        else:
            fx_targets = sorted(FX_VARS)

        fx_extractors: dict[str, Callable[[xr.Dataset], xr.DataArray | None]] = {
            "sftlf": _extract_sftlf,
            "areacella": _compute_areacella,
        }
        for cmor in fx_targets:
            da = fx_extractors[cmor](ds)
            if da is None:
                missing.append(cmor)
                continue
            written.append(
                _write_single_var(
                    da,
                    cmor=cmor,
                    table="fx",
                    model=model,
                    experiment=experiment,
                    member=member,
                    grid_label=grid_label,
                    frequency="fx",
                    outdir=outdir,
                    date_stamp=None,
                    time_bounds=None,
                    dry_run=dry_run,
                )
            )

    if missing and explicit_vars:
        warnings.warn(
            f"Requested CMOR variable(s) not available in input ({table}): "
            f"{', '.join(sorted(missing))}",
            UserWarning,
            stacklevel=2,
        )

    return written


def discover_inputs(
    paths: Sequence[pathlib.Path],
    freq: str = "auto",
    prefix: str | None = None,
) -> dict[str, list[pathlib.Path]]:
    """Resolve input files or a JCM run directory into ``{'Amon': [...], 'day': [...]}``."""
    if len(paths) == 1 and paths[0].is_dir():
        run_dir = paths[0]
        monthly_matches = [
            (p, m)
            for p in sorted(run_dir.glob("*_monthly_*.nc"))
            if (m := _MONTHLY_FILE_RE.match(p.name)) is not None
        ]
        chunk_matches = [
            (p, m)
            for p in run_dir.glob("*_day*.nc")
            if (m := _CHUNK_FILE_RE.match(p.name)) is not None
        ]
        if prefix is not None:
            monthly_matches = [(p, m) for p, m in monthly_matches if m.group("prefix") == prefix]
            chunk_matches = [(p, m) for p, m in chunk_matches if m.group("prefix") == prefix]
        else:
            prefixes = {m.group("prefix") for _, m in monthly_matches + chunk_matches}
            if len(prefixes) > 1:
                raise ValueError(
                    f"Multiple JCM run prefixes found in {run_dir}: "
                    f"{', '.join(sorted(prefixes))}. Pass --prefix or explicit file paths."
                )

        monthly = [p for p, _ in monthly_matches]
        chunks = [
            p
            for p, _ in sorted(chunk_matches, key=lambda item: float(item[1].group("day")))
        ]
        out: dict[str, list[pathlib.Path]] = {}
        if freq in ("auto", "both", "mon") and monthly:
            out["Amon"] = monthly
        if freq in ("auto", "both", "day") and chunks:
            out["day"] = chunks
        if freq == "both" and ("Amon" not in out or "day" not in out):
            missing_tables = [t for t in ("Amon", "day") if t not in out]
            raise FileNotFoundError(
                f"--freq both requested, but missing {', '.join(missing_tables)} files in {run_dir}"
            )
        if not out:
            raise FileNotFoundError(
                f"No *_monthly_YYYY-MM.nc or *_day*.nc files found in {run_dir}"
            )
        return out

    files = [pathlib.Path(p) for p in paths]
    dirs = [str(p) for p in files if p.is_dir()]
    if dirs:
        raise ValueError(
            f"Pass either a single run directory or explicit NetCDF files, not both: {', '.join(dirs)}"
        )
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")
    if freq == "day":
        return {"day": files}
    if freq == "mon":
        return {"Amon": files}

    monthly_explicit = [p for p in files if _MONTHLY_FILE_RE.match(p.name)]
    chunk_explicit = sorted(
        (p for p in files if _CHUNK_FILE_RE.match(p.name)),
        key=lambda p: float(_CHUNK_FILE_RE.match(p.name).group("day")),  # type: ignore[union-attr]
    )
    if monthly_explicit and chunk_explicit and len(monthly_explicit) + len(chunk_explicit) == len(files):
        return {"Amon": monthly_explicit, "day": chunk_explicit}
    if freq == "both":
        raise FileNotFoundError(
            "--freq both requires both *_monthly_YYYY-MM.nc and *_day*.nc inputs."
        )
    if len(chunk_explicit) == len(files):
        return {"day": chunk_explicit}
    return {"Amon": files}


def resolve_output_dir(
    out_root: pathlib.Path,
    experiment: str,
    member: str,
    *,
    flat: bool = False,
) -> pathlib.Path:
    """Return the AtmosphereBench submission directory for ``(experiment, member)``."""
    if flat:
        return out_root
    if experiment in MEMBER_EXPERIMENTS:
        return out_root / experiment / member
    return out_root / experiment


def _open_inputs(files: Sequence[pathlib.Path]) -> xr.Dataset:
    if len(files) == 1:
        return xr.open_dataset(files[0])
    try:
        return xr.open_mfdataset(
            [str(p) for p in files],
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
    except ImportError:
        # Fallback when dask is not installed (e.g. standalone jax-gcm runtime image).
        datasets = [xr.open_dataset(p) for p in files]
        try:
            combined = xr.concat(
                datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )
            return combined.load()
        finally:
            for d in datasets:
                d.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="JCM run directory or one or more JCM output NetCDF files",
    )
    ap.add_argument("--model", required=True, help="Model name for CMOR filenames (no underscores)")
    ap.add_argument(
        "--experiment",
        default="amip",
        help="AtmosphereBench experiment name (e.g. amip, amip-4xCO2, amip-p4K; default: %(default)s)",
    )
    ap.add_argument(
        "--member",
        default="r1i1p1f1",
        help="CMIP variant label (default: %(default)s)",
    )
    ap.add_argument(
        "--grid-label",
        default="gn",
        help="CMOR grid label (default: %(default)s)",
    )
    ap.add_argument(
        "--freq",
        default="auto",
        choices=("auto", "both", "mon", "day"),
        help="Which frequencies to export (default: %(default)s)",
    )
    ap.add_argument(
        "--prefix",
        help="Optional JCM filename prefix when a directory contains multiple runs",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("atmospherebench_submit"),
        help="Submission root directory (default: %(default)s)",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Write NetCDF files directly into --out instead of <out>/<experiment>[/<member>]/",
    )
    ap.add_argument(
        "--variables",
        help="Optional comma-separated subset of CMOR variables to export",
    )
    ap.add_argument(
        "--plevs",
        help="Comma-separated pressure levels in hPa for monthly 3D fields (default: 19 CMIP6 levels)",
    )
    ap.add_argument(
        "--daily-plevs",
        help="Comma-separated pressure levels in hPa for daily 3D fields (default: 850,200)",
    )
    ap.add_argument(
        "--start-year",
        type=int,
        help="Optional first calendar year (inclusive) to include",
    )
    ap.add_argument(
        "--end-year",
        type=int,
        help="Optional last calendar year (inclusive) to include",
    )
    ap.add_argument(
        "--resample-daily",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample sub-daily chunk files to daily means plus tasmax/tasmin (default: enabled)",
    )
    ap.add_argument(
        "--no-fx",
        action="store_true",
        help="Skip writing fixed grid fields (sftlf, areacella)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print filenames that would be written without writing NetCDF files",
    )
    return ap


def _parse_plev_arg(raw: str | None, flag: str) -> list[float] | None:
    if raw is None:
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError(f"{flag} must be a non-empty comma-separated list of pressures in hPa.")
    try:
        vals = [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(f"{flag}={raw!r}: could not parse pressure levels as floats.") from exc
    return validate_pressure_levels(vals).tolist()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    variables = (
        [v.strip() for v in args.variables.split(",") if v.strip()]
        if args.variables
        else None
    )
    monthly_plevs = _parse_plev_arg(args.plevs, "--plevs")
    daily_plevs = _parse_plev_arg(args.daily_plevs, "--daily-plevs")

    grouped = discover_inputs(args.inputs, freq=args.freq, prefix=args.prefix)
    outdir = resolve_output_dir(args.out, args.experiment, args.member, flat=args.flat)

    all_written: list[str] = []
    remaining_fx: set[str] = set()
    if not args.no_fx:
        if variables is not None:
            remaining_fx = {v for v in variables if v in FX_VARS}
        else:
            remaining_fx = set(FX_VARS)

    for table in ("Amon", "day"):
        if table not in grouped:
            continue
        with _open_inputs(grouped[table]) as ds:
            written = convert_dataset(
                ds,
                model=args.model,
                experiment=args.experiment,
                member=args.member,
                table=table,
                grid_label=args.grid_label,
                outdir=outdir,
                variables=variables,
                plevs_hpa=monthly_plevs if table == "Amon" else daily_plevs,
                write_fx=bool(remaining_fx),
                fx_vars_filter=sorted(remaining_fx) if remaining_fx else None,
                resample_daily=args.resample_daily,
                start_year=args.start_year,
                end_year=args.end_year,
                dry_run=args.dry_run,
            )
            for fname in written:
                var_name = fname.split("_", 1)[0]
                remaining_fx.discard(var_name)
            all_written.extend(written)

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(all_written)} file(s) to {outdir}")
    for fname in all_written:
        print(" ", fname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
