"""Shared JCM-to-CMOR variable mapping, CF metadata, and NetCDF writer.

Both ``tools/aerocom_cmor.py`` (AeroCom phase-4 / ACI submissions) and
``tools/atmospherebench_cmor.py`` (CMIP6 / AtmosphereBench submissions) build
on this module so variable mappings, fallback priorities, physical constants,
and CF coordinate metadata live in one place and cannot drift apart.

Design choices
--------------
* **Single source priority for ``clt``:** Both exporters read total cloud cover
  in the order ``clouds.total_cloud_cover`` -> ``aerocom_clt`` ->
  ``radiation.total_cloud_cover`` -> ``shortwave_rad.cloudc``, preferring the
  model's instantaneous maximum-random overlap cover over McICA sub-column
  noise (see :mod:`jcm.analysis`). Each protocol converts to its requested unit
  (fraction ``"1"`` for AeroCom, percent ``"%"`` for CMIP6 / AtmosphereBench).
* **Live physical constants:** Earth radius (``c.rearth``) and gravity
  (``c.grav``) are read from :mod:`jcm.constants` on attribute access so
  ``areacella``, ``prw``, and ``zg`` stay consistent with the model's sphere
  and honour :func:`jcm.constants.set_constants`.
* **CF coordinates from :mod:`jcm.cf_metadata`:** Standard ``lat``/``lon``/
  ``time`` coordinate attributes come from :func:`jcm.cf_metadata.apply_cf_attributes`
  rather than being re-tabled here.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import re
import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

import jcm.constants as c
from jcm.analysis import area_weights, column_burden, to_pressure_levels
from jcm.cf_metadata import apply_cf_attributes
from jcm.physics.surface.echam.surface_types import DIAG_TILE_NAMES

#: Land tile index on ECHAM's ``("water", "sea_ice", "land")`` surface-tile axis.
ECHAM_LAND_TILE_INDEX: int = DIAG_TILE_NAMES.index("land")

#: Fixed (time-invariant) CMOR grid fields.
FX_VARS = frozenset({"sftlf", "areacella"})

#: 3D CMOR variables that live on ``plev`` when pressure-interpolated.
PLEV_VARS = frozenset({"ta", "hus", "hur", "ua", "va", "zg"})

#: Source variable(s) in fallback priority order for each 3D CMOR variable.
PLEV_SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ta": ("temperature",),
    "hus": ("specific_humidity",),
    "hur": ("relative_humidity",),
    "ua": ("u_wind",),
    "va": ("v_wind",),
    "zg": ("height_full", "geopotential"),
}

#: Regex for CMIP6 variant labels such as ``r1i1p1f1``.
VARIANT_RE = re.compile(r"^r\d+i\d+p\d+f\d+$")

#: CF metadata for core CMOR variables:
#: ``cmor_name -> (cmip6_units, standard_name, long_name)``.
CMOR_META: dict[str, tuple[str, str, str]] = {
    "tas": ("K", "air_temperature", "Near-Surface Air Temperature"),
    "tasmax": ("K", "air_temperature", "Daily Maximum Near-Surface Air Temperature"),
    "tasmin": ("K", "air_temperature", "Daily Minimum Near-Surface Air Temperature"),
    "ts": ("K", "surface_temperature", "Surface Temperature"),
    "pr": ("kg m-2 s-1", "precipitation_flux", "Precipitation"),
    "evspsbl": (
        "kg m-2 s-1",
        "water_evapotranspiration_flux",
        "Evaporation Including Sublimation and Transpiration",
    ),
    "hfls": ("W m-2", "surface_upward_latent_heat_flux", "Surface Upward Latent Heat Flux"),
    "hfss": ("W m-2", "surface_upward_sensible_heat_flux", "Surface Upward Sensible Heat Flux"),
    "rsdt": ("W m-2", "toa_incoming_shortwave_flux", "TOA Incident Shortwave Radiation"),
    "rsut": ("W m-2", "toa_outgoing_shortwave_flux", "TOA Outgoing Shortwave Radiation"),
    "rlut": ("W m-2", "toa_outgoing_longwave_flux", "TOA Outgoing Longwave Radiation"),
    "rsutcs": (
        "W m-2",
        "toa_outgoing_shortwave_flux_assuming_clear_sky",
        "TOA Outgoing Clear-Sky Shortwave Radiation",
    ),
    "rlutcs": (
        "W m-2",
        "toa_outgoing_longwave_flux_assuming_clear_sky",
        "TOA Outgoing Clear-Sky Longwave Radiation",
    ),
    "rsds": (
        "W m-2",
        "surface_downwelling_shortwave_flux_in_air",
        "Surface Downwelling Shortwave Radiation",
    ),
    "rsus": (
        "W m-2",
        "surface_upwelling_shortwave_flux_in_air",
        "Surface Upwelling Shortwave Radiation",
    ),
    "rlds": (
        "W m-2",
        "surface_downwelling_longwave_flux_in_air",
        "Surface Downwelling Longwave Radiation",
    ),
    "rlus": (
        "W m-2",
        "surface_upwelling_longwave_flux_in_air",
        "Surface Upwelling Longwave Radiation",
    ),
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
    "sftlf": (
        "%",
        "land_area_fraction",
        "Percentage of the Grid Cell Occupied by Land (Including Lakes)",
    ),
    "areacella": ("m2", "cell_area", "Grid-Cell Area for Atmospheric Grid Variables"),
}

#: Additional CF standard names used by AeroCom / COSP diagnostics beyond
#: :data:`CMOR_META`.
_EXTRA_CF_STANDARD_NAMES: dict[str, str] = {
    "wap": "lagrangian_tendency_of_air_pressure",
    "w500": "lagrangian_tendency_of_air_pressure",
    "w700": "lagrangian_tendency_of_air_pressure",
    "rho": "air_density",
    "cl": "cloud_area_fraction_in_atmosphere_layer",
    "clw": "mass_fraction_of_cloud_liquid_water_in_air",
    "cli": "mass_fraction_of_cloud_ice_in_air",
    "lwp": "atmosphere_mass_content_of_cloud_liquid_water",
    "iwp": "atmosphere_mass_content_of_cloud_ice",
    "cltcalipso": "cloud_area_fraction",
    "cllcalipso": "cloud_area_fraction_in_atmosphere_layer",
    "clmcalipso": "cloud_area_fraction_in_atmosphere_layer",
    "clhcalipso": "cloud_area_fraction_in_atmosphere_layer",
    "cltmodis": "cloud_area_fraction",
    "cltisccp": "cloud_area_fraction",
    "dew2": "dew_point_temperature",
    "prsn": "snowfall_flux",
    "ptp": "tropopause_air_pressure",
    "rsutnoa": "toa_outgoing_shortwave_flux",
    "rlutnoa": "toa_outgoing_longwave_flux",
    "clwmodis": "liquid_water_cloud_area_fraction",
    "climodis": "ice_cloud_area_fraction",
    "tauwmodis": "atmosphere_optical_thickness_due_to_cloud",
    "tauimodis": "atmosphere_optical_thickness_due_to_cloud",
    "reffclwmodis": "effective_radius_of_cloud_liquid_water_particles",
    "reffclimodis": "effective_radius_of_cloud_ice_particles",
    "lwpmodis": "atmosphere_mass_content_of_cloud_liquid_water",
    "iwpmodis": "atmosphere_mass_content_of_cloud_ice",
    "od550aer": "atmosphere_optical_thickness_due_to_ambient_aerosol_particles",
    "abs550aer": "atmosphere_absorption_optical_thickness_due_to_ambient_aerosol_particles",
    "od550so4": "atmosphere_optical_thickness_due_to_sulfate_ambient_aerosol_particles",
    "od550bc": "atmosphere_optical_thickness_due_to_black_carbon_ambient_aerosol_particles",
    "od550oa": "atmosphere_optical_thickness_due_to_particulate_organic_matter_ambient_aerosol_particles",
    "od550ss": "atmosphere_optical_thickness_due_to_sea_salt_ambient_aerosol_particles",
    "od550dust": "atmosphere_optical_thickness_due_to_dust_ambient_aerosol_particles",
    "od550aerh2o": "atmosphere_optical_thickness_due_to_water_in_ambient_aerosol_particles",
    "ec355aer": "volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles",
    "ang4487aer": "angstrom_exponent_of_ambient_aerosol_in_air",
}

#: Unified CF standard-name table shared by AeroCom and AtmosphereBench writers.
CF_STANDARD_NAMES: dict[str, str] = {
    **{cmor: meta[1] for cmor, meta in CMOR_META.items()},
    **_EXTRA_CF_STANDARD_NAMES,
}

#: CF ``positive`` direction attribute for radiative and turbulent fluxes.
FLUX_POSITIVE: dict[str, str] = {
    "rsdt": "down",
    "rsut": "up",
    "rlut": "up",
    "rsutcs": "up",
    "rlutcs": "up",
    "rsutnoa": "up",
    "rlutnoa": "up",
    "rsutcsnoa": "up",
    "rlutcsnoa": "up",
    "rsds": "down",
    "rsus": "up",
    "rlds": "down",
    "rlus": "up",
    "hfls": "up",
    "hfss": "up",
}

#: Canonical source priority for total cloud cover (highest priority first).
CLT_SOURCES: tuple[str, ...] = (
    "clouds.total_cloud_cover",
    "aerocom_clt",
    "radiation.total_cloud_cover",
    "shortwave_rad.cloudc",
)

#: Canonical direct 2 m air-temperature sources (highest priority first).
TAS_SOURCES: tuple[str, ...] = (
    "surface.temperature_2m",
    "aerocom_tas",
)

#: Direct JCM -> CMOR source mappings with fallback priority (first present wins):
#: ``cmor_name -> ((jcm_name, scale), ...)``.
DIRECT_SOURCES: dict[str, tuple[tuple[str, float], ...]] = {
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


def validate_token(flag: str, value: str, *, max_len: int | None = None) -> str:
    """Validate a CMOR / AeroCom filename token (no ``_``, no ``/``, optional max length)."""
    if not value or "_" in value or "/" in value:
        raise ValueError(
            f"{flag}={value!r}: filename tokens must be non-empty and "
            "must not contain '_' or '/' (use '-' instead)."
        )
    if max_len is not None and len(value) > max_len:
        raise ValueError(
            f"{flag}={value!r}: token is capped at {max_len} characters "
            f"(got {len(value)})."
        )
    return value


def validate_variant_label(member: str) -> str:
    """Validate a CMIP6 variant label such as ``r1i1p1f1``."""
    if not VARIANT_RE.match(member):
        raise ValueError(
            f"--member={member!r}: expected CMIP variant label like 'r1i1p1f1'."
        )
    return member


def is_percent_units(da: xr.DataArray) -> bool:
    """Return True if ``da`` already declares percent units."""
    units = str(da.attrs.get("units", "")).strip().lower()
    return units in ("%", "percent")


def extract_tas(
    ds: xr.Dataset,
    *,
    allow_lowest_level_fallback: bool = True,
) -> xr.DataArray | None:
    """Extract 2 m air temperature ``tas``, optionally falling back to lowest model level."""
    for key in TAS_SOURCES:
        if key in ds.data_vars:
            return ds[key]
    if (
        allow_lowest_level_fallback
        and "temperature" in ds.data_vars
        and "level" in ds["temperature"].dims
    ):
        temp = ds["temperature"]
        if "pressure_full" in ds.data_vars and "level" in ds["pressure_full"].dims:
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


def extract_clt(ds: xr.Dataset, *, units: str = "%") -> xr.DataArray | None:
    """Extract total cloud cover ``clt`` using the shared source priority.

    Returns values in ``[0, 100]`` when ``units="%"`` (CMIP6 / AtmosphereBench)
    or ``[0, 1]`` when ``units="1"`` (AeroCom).
    """
    for key in CLT_SOURCES:
        if key in ds.data_vars:
            da = ds[key]
            in_pct = is_percent_units(da)
            if units == "%":
                if not in_pct:
                    da = da * 100.0
                return da.clip(0.0, 100.0)
            if units == "1":
                if in_pct:
                    da = da / 100.0
                return da.clip(0.0, 1.0)
            raise ValueError(f"Unsupported clt units {units!r}; expected '%' or '1'.")
    return None


def extract_pr(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract total surface precipitation flux ``pr`` [kg m-2 s-1]."""
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


def extract_prw(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract column water vapor path ``prw`` [kg m-2]."""
    if "aerocom_prw" in ds.data_vars:
        return ds["aerocom_prw"]
    if "specific_humidity" not in ds.data_vars or "level" not in ds["specific_humidity"].dims:
        return None
    if "pressure_thickness" not in ds.data_vars and "pressure_half" not in ds.data_vars:
        return None
    return column_burden(ds, "specific_humidity")


def extract_sftlf(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract land area fraction ``sftlf`` [%], preferring ``surface_type='land'`` metadata."""
    da: xr.DataArray | None = None
    for name in sorted(ds.data_vars):
        if (
            str(name).startswith("surface_exchange.tile_fraction.")
            and ds[name].attrs.get("surface_type") == "land"
        ):
            da = ds[name]
            break
    if da is None:
        for key in (
            f"surface_exchange.tile_fraction.{ECHAM_LAND_TILE_INDEX}",
            f"vertical_diffusion.surface_fraction.{ECHAM_LAND_TILE_INDEX}",
        ):
            if key in ds.data_vars:
                da = ds[key]
                break
    if da is None:
        return None
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    if not is_percent_units(da):
        da = da * 100.0
    return da.clip(0.0, 100.0)


def compute_areacella(ds: xr.Dataset) -> xr.DataArray | None:
    """Compute exact spherical grid-cell areas ``areacella`` [m2] using ``jcm.constants.rearth``."""
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


DERIVED_EXTRACTORS: dict[str, Callable[[xr.Dataset], xr.DataArray | None]] = {
    "tas": extract_tas,
    "clt": extract_clt,
    "pr": extract_pr,
    "prw": extract_prw,
}


def shared_source_entries(
    vert_by_cmor: Mapping[str, str],
    *,
    units_overrides: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, str, str, float, float]]:
    """Build a first-present-wins ``jcm_src -> (cmor, units, vert, scale, offset)`` table.

    Covers every CMOR variable in ``vert_by_cmor`` using the canonical fallback
    priority in :data:`DIRECT_SOURCES`, :data:`PLEV_SOURCE_CANDIDATES`,
    :data:`CLT_SOURCES`, and :data:`TAS_SOURCES` so downstream exporters share
    the exact same source priority without duplicating mappings.
    """
    overrides = units_overrides or {}
    out: dict[str, tuple[str, str, str, float, float]] = {}

    def _add(cmor: str, src: str, scale: float) -> None:
        units = overrides.get(cmor, CMOR_META[cmor][0])
        out[src] = (cmor, units, vert_by_cmor[cmor], scale, 0.0)

    for cmor in vert_by_cmor:
        if cmor == "clt":
            scale = 1.0 if overrides.get("clt") == "1" else 100.0
            for src in CLT_SOURCES:
                _add(cmor, src, scale)
        elif cmor == "tas":
            for src in TAS_SOURCES:
                _add(cmor, src, 1.0)
        elif cmor == "prw":
            _add(cmor, "aerocom_prw", 1.0)
        elif cmor in PLEV_SOURCE_CANDIDATES:
            for src in PLEV_SOURCE_CANDIDATES[cmor]:
                if cmor == "hur":
                    scale = 100.0
                elif src == "geopotential":
                    scale = 1.0 / c.grav
                else:
                    scale = 1.0
                _add(cmor, src, scale)
        elif cmor in DIRECT_SOURCES:
            for src, scale in DIRECT_SOURCES[cmor]:
                _add(cmor, src, scale)
    return out


def extract_direct_variable(ds: xr.Dataset, cmor: str) -> xr.DataArray | None:
    """Extract a 2D/derived CMOR variable from ``ds`` using the shared mappings."""
    if cmor in DERIVED_EXTRACTORS:
        return DERIVED_EXTRACTORS[cmor](ds)
    for src, scale in DIRECT_SOURCES.get(cmor, ()):
        if src in ds.data_vars:
            da = ds[src]
            return da * scale if scale != 1.0 else da
    return None


def extract_plev_variable(
    ds: xr.Dataset,
    cmor: str,
    plevs_hpa: Sequence[float],
    *,
    subset_preinterp: bool = False,
) -> xr.DataArray | None:
    """Extract and pressure-interpolate one 3D CMOR variable onto ``plevs_hpa`` [hPa]."""
    src: str | None = None
    for candidate in PLEV_SOURCE_CANDIDATES[cmor]:
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

    if cmor == "hur" and not is_percent_units(da):
        da = da * 100.0
    elif src == "geopotential":
        da = da / c.grav
    return da


def date_ymd(val) -> tuple[int, int, int]:
    """Extract ``(year, month, day)`` from a numpy datetime64, Timestamp, or cftime scalar."""
    if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        return int(val.year), int(val.month), int(val.day)
    if isinstance(val, np.datetime64):
        s = str(np.datetime_as_string(val, unit="D"))
        y, m, d = s.split("-")
        return int(y), int(m), int(d)
    raise TypeError(f"Unsupported time coordinate value: {val!r} ({type(val).__name__})")


def get_time_bounds(ds: xr.Dataset) -> xr.DataArray | None:
    """Return the CF time-bounds variable from ``ds`` (data_vars or coords), if present."""
    if "time" not in ds.coords:
        return None
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    if bounds_name in ds:
        return ds[bounds_name]
    if bounds_name != "time_bounds" and "time_bounds" in ds:
        return ds["time_bounds"]
    return None


def timedelta_days(delta) -> float:
    """Convert a numpy timedelta64 or datetime.timedelta to float days."""
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s")) / 86400.0
    if isinstance(delta, datetime.timedelta):
        return delta.total_seconds() / 86400.0
    raise TypeError(f"Unsupported timedelta type: {type(delta).__name__}")


def median_step_days(ds: xr.Dataset, time_bounds: xr.DataArray | None) -> float | None:
    """Estimate the median sampling/averaging interval [days] of ``ds['time']``."""
    if time_bounds is not None and time_bounds.ndim == 2 and time_bounds.shape[1] == 2:
        tb = time_bounds.values
        widths = [timedelta_days(tb[i, 1] - tb[i, 0]) for i in range(tb.shape[0])]
        if widths:
            return float(np.median(widths))
    if "time" in ds.coords and ds.sizes.get("time", 0) >= 2:
        tvals = ds.coords["time"].values
        diffs = [timedelta_days(tvals[i + 1] - tvals[i]) for i in range(len(tvals) - 1)]
        if diffs:
            return float(np.median(diffs))
    return None


def cell_methods(cmor: str, table: str) -> str:
    """Return the CF-1.8 ``cell_methods`` string for ``(cmor, table)``."""
    if table == "fx":
        return "area: sum" if cmor == "areacella" else "area: mean"
    if cmor == "tasmax":
        return "area: mean time: maximum"
    if cmor == "tasmin":
        return "area: mean time: minimum"
    return "area: time: mean"


def standardize_dataarray(
    da: xr.DataArray,
    cmor: str,
    *,
    table: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    long_name: str | None = None,
    comment: str | None = None,
    sort_lat: bool = True,
    transpose_cmor: bool = True,
    drop_extra_coords: bool = True,
) -> xr.DataArray:
    """Standardize dimensions, latitude order, and CF variable attributes."""
    if drop_extra_coords:
        extra_coords = [c_name for c_name in da.coords if c_name not in da.dims]
        if extra_coords:
            da = da.drop_vars(extra_coords)

    if transpose_cmor:
        ordered_dims = [d for d in ("time", "plev", "lat", "lon") if d in da.dims]
        remaining = [d for d in da.dims if d not in ordered_dims]
        da = da.transpose(*ordered_dims, *remaining)

    if sort_lat and "lat" in da.dims and da.sizes["lat"] > 1:
        lat_vals = da.coords["lat"].values
        if lat_vals[0] > lat_vals[-1]:
            da = da.isel(lat=slice(None, None, -1))

    meta = CMOR_META.get(cmor)
    resolved_units = units if units is not None else (meta[0] if meta else da.attrs.get("units"))
    resolved_std = (
        standard_name
        if standard_name is not None
        else CF_STANDARD_NAMES.get(cmor)
    )
    resolved_long = long_name if long_name is not None else (meta[2] if meta else cmor)

    da = da.rename(cmor)
    attrs: dict[str, str] = {}
    if resolved_units is not None:
        attrs["units"] = resolved_units
    if resolved_std is not None:
        attrs["standard_name"] = resolved_std
    attrs["long_name"] = resolved_long
    if table is not None:
        attrs["cell_methods"] = cell_methods(cmor, table)
    if cmor in FLUX_POSITIVE:
        attrs["positive"] = FLUX_POSITIVE[cmor]
    if comment is not None:
        attrs["comment"] = comment
    da.attrs = attrs
    return da


def apply_cmor_coord_attrs(out: xr.Dataset) -> None:
    """Stamp CF coordinate attributes using :func:`jcm.cf_metadata.apply_cf_attributes`."""
    apply_cf_attributes(out)
    if "time" in out.coords:
        # Remove encoding-owned keys from attrs so xarray.to_netcdf does not
        # fail with "failed to prevent overwriting existing key units in attrs".
        out["time"].attrs.pop("units", None)
        out["time"].attrs.pop("calendar", None)
        if out.sizes.get("time", 0) > 0 and hasattr(out["time"].values[0], "year"):
            # cftime coordinates are datetime-like even though their numpy dtype is object.
            out["time"].attrs["standard_name"] = "time"
            out["time"].attrs["long_name"] = "time"
            out["time"].attrs["axis"] = "T"
        if "time_bounds" in out.data_vars:
            out["time"].attrs["bounds"] = "time_bounds"
            out["time_bounds"].attrs.pop("units", None)
            out["time_bounds"].attrs.pop("calendar", None)
    if "plev" in out.coords:
        out["plev"].attrs.setdefault("standard_name", "air_pressure")
        out["plev"].attrs.setdefault("long_name", "pressure")
        out["plev"].attrs["units"] = "Pa"
        out["plev"].attrs.setdefault("positive", "down")
        out["plev"].attrs.setdefault("axis", "Z")


def time_encoding(out: xr.Dataset) -> dict[str, dict]:
    """Build NetCDF time encoding preserving cftime calendars when present."""
    if "time" not in out.coords:
        return {}
    tvals = out["time"].values
    is_datetime_like = np.issubdtype(np.asarray(tvals).dtype, np.datetime64) or (
        len(tvals) > 0 and hasattr(tvals[0], "year")
    )
    if not is_datetime_like:
        return {}
    t_enc: dict[str, str] = {"units": "days since 1850-01-01", "dtype": "float64"}
    cal = out["time"].encoding.get("calendar")
    if cal is None and len(tvals) > 0:
        cal = getattr(tvals[0], "calendar", None)
    if cal:
        t_enc["calendar"] = str(cal)
    enc: dict[str, dict] = {"time": t_enc}
    if "time_bounds" in out.data_vars:
        enc["time_bounds"] = dict(t_enc)
    return enc


def write_netcdf_atomic(
    ds: xr.Dataset,
    target: pathlib.Path,
    *,
    encoding: Mapping[str, dict] | None = None,
) -> None:
    """Write ``ds`` to ``target`` via a sibling temp file and atomic ``os.replace``."""
    tmp_path = target.parent / f".{target.name}.tmp.{os.getpid()}"
    try:
        ds.to_netcdf(tmp_path, encoding=dict(encoding) if encoding else None)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_cmor_variable(
    da: xr.DataArray,
    *,
    cmor: str,
    target: pathlib.Path,
    global_attrs: Mapping[str, str],
    table: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    long_name: str | None = None,
    comment: str | None = None,
    time_bounds: xr.DataArray | None = None,
    compress: bool = True,
    sort_lat: bool = True,
    transpose_cmor: bool = True,
    drop_extra_coords: bool = True,
    dry_run: bool = False,
) -> str:
    """Standardize and atomically write one CMOR variable NetCDF file."""
    std_da = standardize_dataarray(
        da,
        cmor,
        table=table,
        units=units,
        standard_name=standard_name,
        long_name=long_name,
        comment=comment,
        sort_lat=sort_lat,
        transpose_cmor=transpose_cmor,
        drop_extra_coords=drop_extra_coords,
    )
    if dry_run:
        return target.name

    data_vars: dict[str, xr.DataArray] = {cmor: std_da}
    if time_bounds is not None and "time" in std_da.dims:
        tb = time_bounds
        extra = [c_name for c_name in tb.coords if c_name not in tb.dims]
        if extra:
            tb = tb.drop_vars(extra)
        data_vars["time_bounds"] = tb

    out = xr.Dataset(data_vars)
    if "time" in std_da.coords and "calendar" in std_da.coords["time"].encoding:
        out["time"].encoding["calendar"] = std_da.coords["time"].encoding["calendar"]
    apply_cmor_coord_attrs(out)
    out.attrs.update(global_attrs)

    encoding: dict[str, dict] = {}
    if compress:
        encoding[cmor] = {"zlib": True, "complevel": 4}
    encoding.update(time_encoding(out))
    write_netcdf_atomic(out, target, encoding=encoding)
    return target.name
