"""Shared JCM-to-CMOR variable mapping, CF metadata, and NetCDF writer.

Both ``tools/aerocom_cmor.py`` (AeroCom phase-4 / ACI submissions) and
``tools/atmospherebench_cmor.py`` (CMIP6 / AtmosphereBench submissions) build
on this module so variable mappings, fallback priorities, physical constants,
and CF coordinate metadata live in one place and cannot drift apart.

Design choices
--------------
* **One source table:** :data:`SOURCES` lists, for every CMOR variable, the
  JCM fields that can supply it in fallback priority order (first present
  wins). Both exporters read it, so they always resolve the same field.
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
  ``time``/``plev`` coordinate attributes come from
  :func:`jcm.cf_metadata.apply_cf_attributes` rather than being re-tabled here.
* **Composable writer:** :func:`cmor_variable` stamps the CF variable
  attributes, :func:`to_cmip_layout` applies the CMIP dimension/latitude
  layout, and :func:`write_cmor_variable` writes the file. Exporters that
  submit on the native grid (AeroCom) simply skip the layout step.
"""

from __future__ import annotations

import datetime
import functools
import operator
import os
import pathlib
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

import jcm.constants as c
from jcm.analysis import cell_area, column_burden, to_pressure_levels
from jcm.cf_metadata import apply_cf_attributes
from jcm.physics.surface.echam.surface_types import DIAG_TILE_NAMES

#: Land tile index on ECHAM's ``("water", "sea_ice", "land")`` surface-tile axis.
_ECHAM_LAND_TILE_INDEX: int = DIAG_TILE_NAMES.index("land")

#: Fixed (time-invariant) CMOR grid fields.
FX_VARS = frozenset({"sftlf", "areacella"})

#: 3D CMOR variables that live on ``plev`` when pressure-interpolated.
PLEV_VARS = frozenset({"ta", "hus", "hur", "ua", "va", "zg"})

#: Regex for CMIP6 variant labels such as ``r1i1p1f1``.
_VARIANT_RE = re.compile(r"^r\d+i\d+p\d+f\d+$")

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
_FLUX_POSITIVE: dict[str, str] = {
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

#: JCM source field(s) for each CMOR variable, highest priority first (first
#: present wins). Unit conversions are not tabled here: see :func:`to_cmor_units`.
SOURCES: dict[str, tuple[str, ...]] = {
    "tas": ("surface.temperature_2m", "aerocom_tas"),
    "ts": ("surface.surface_temperature", "surface.skin_temperature"),
    "evspsbl": (
        "surface.evaporation",
        "surface_exchange.evaporation",
        "vertical_diffusion.surface_evaporation",
    ),
    "hfls": ("surface.latent_heat_flux", "surface_exchange.latent_heat_flux"),
    "hfss": (
        "surface.sensible_heat_flux_dry",
        "surface.sensible_heat_flux",
        "surface_exchange.sensible_heat_flux",
    ),
    "rsdt": ("radiation.toa_sw_down",),
    "rsut": ("radiation.toa_sw_up",),
    "rlut": ("radiation.toa_lw_up", "longwave_rad.ftop"),
    "rsutcs": ("radiation.toa_sw_up_clear", "clouds.toa_sw_up_clear"),
    "rlutcs": ("radiation.toa_lw_up_clear", "clouds.toa_lw_up_clear"),
    "rsds": ("radiation.surface_sw_down", "surface_exchange.downward_shortwave_flux"),
    "rsus": ("radiation.surface_sw_up",),
    "rlds": ("radiation.surface_lw_down", "surface_exchange.downward_longwave_flux"),
    "rlus": ("radiation.surface_lw_up",),
    "prw": ("aerocom_prw",),
    "ps": ("surface_pressure", "surface_exchange.surface_pressure"),
    "psl": ("aerocom_psl",),
    "huss": ("surface.humidity_2m",),
    "uas": ("vertical_diffusion.wind_10m_u", "aerocom_uas"),
    "vas": ("vertical_diffusion.wind_10m_v", "aerocom_vas"),
    "tasmax": ("tasmax",),
    "tasmin": ("tasmin",),
    "clt": (
        "clouds.total_cloud_cover",
        "aerocom_clt",
        "radiation.total_cloud_cover",
        "shortwave_rad.cloudc",
    ),
    "ta": ("temperature",),
    "hus": ("specific_humidity",),
    "hur": ("relative_humidity",),
    "ua": ("u_wind",),
    "va": ("v_wind",),
    "zg": ("height_full", "geopotential"),
}

#: Sources whose JCM -> CMOR factor depends on a model constant. Evaluated on
#: every use so overrides applied via :func:`jcm.constants.set_constants` after
#: module import are honoured.
_LIVE_SCALES: dict[str, Callable[[], float]] = {
    "geopotential": lambda: 1.0 / c.grav,
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
    if not _VARIANT_RE.match(member):
        raise ValueError(
            f"--member={member!r}: expected CMIP variant label like 'r1i1p1f1'."
        )
    return member


# --- unit conversion ------------------------------------------------------


def _is_percent(da: xr.DataArray) -> bool:
    """Return True if ``da`` already declares percent units."""
    return str(da.attrs.get("units", "")).strip().lower() in ("%", "percent")


def _to_percent(da: xr.DataArray) -> xr.DataArray:
    """Convert a fraction to percent, leaving a field that already declares percent alone."""
    return da if _is_percent(da) else da * 100.0


def _source_scale(src: str, default: float = 1.0) -> float:
    """Return the multiplicative JCM -> CMOR factor for ``src`` (live constants win)."""
    live = _LIVE_SCALES.get(src)
    return live() if live is not None else default


def to_cmor_units(
    da: xr.DataArray,
    src: str,
    units: str,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> xr.DataArray:
    """Convert the JCM source field ``src`` to the CMOR ``units`` requested.

    * ``units == "%"`` is a fraction -> percent conversion; a source that
      already declares percent units is passed through, and ``scale`` is not
      consulted.
    * A source in :data:`_LIVE_SCALES` (``geopotential`` -> height) is scaled
      by the live model constant.
    * Anything else is ``da * scale``.

    ``offset`` is added last in every case.
    """
    if units == "%":
        da = _to_percent(da)
    else:
        scale = _source_scale(src, scale)
        if scale != 1.0:
            da = da * scale
    if offset != 0.0:
        da = da + offset
    return da


# --- source lookup and derived variables ----------------------------------


def _first_present(ds: xr.Dataset, names: Iterable[str]) -> str | None:
    """Return the first of ``names`` that is a data variable of ``ds``."""
    return next((name for name in names if name in ds.data_vars), None)


def _sum_present(ds: xr.Dataset, names: Iterable[str]) -> xr.DataArray | None:
    """Sum whichever of ``names`` are data variables of ``ds`` (None if none are)."""
    parts = [ds[name] for name in names if name in ds.data_vars]
    return functools.reduce(operator.add, parts) if parts else None


def _lookup_direct(ds: xr.Dataset, cmor: str) -> xr.DataArray | None:
    src = _first_present(ds, SOURCES.get(cmor, ()))
    if src is None:
        return None
    return to_cmor_units(ds[src], src, CMOR_META[cmor][0])


def extract_tas(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract 2 m air temperature ``tas``, falling back to the lowest model level."""
    direct = _lookup_direct(ds, "tas")
    if direct is not None:
        return direct
    if "temperature" not in ds.data_vars or "level" not in ds["temperature"].dims:
        return None
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
    return ds["temperature"].isel(level=sfc_idx, drop=True)


def extract_clt(ds: xr.Dataset, *, units: str = "%") -> xr.DataArray | None:
    """Extract total cloud cover ``clt`` using the shared source priority.

    Returns values in ``[0, 100]`` when ``units="%"`` (CMIP6 / AtmosphereBench)
    or ``[0, 1]`` when ``units="1"`` (AeroCom).
    """
    src = _first_present(ds, SOURCES["clt"])
    if src is None:
        return None
    da = ds[src]
    if units == "%":
        return _to_percent(da).clip(0.0, 100.0)
    if units == "1":
        return (da / 100.0 if _is_percent(da) else da).clip(0.0, 1.0)
    raise ValueError(f"Unsupported clt units {units!r}; expected '%' or '1'.")


def extract_pr(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract total surface precipitation flux ``pr`` [kg m-2 s-1]."""
    if "surface_exchange.precipitation" in ds.data_vars:
        return ds["surface_exchange.precipitation"]
    echam = _sum_present(
        ds, ("clouds.precip_rain", "clouds.precip_snow", "convection.precip_conv")
    )
    if echam is not None:
        return echam
    speedy = _sum_present(ds, ("condensation.precls", "convection.precnv"))
    if speedy is not None:
        # SPEEDY stores precipitation in g m-2 s-1 (= 1e-3 kg m-2 s-1).
        return speedy * 1e-3
    return None


def extract_prw(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract column water vapor path ``prw`` [kg m-2]."""
    direct = _lookup_direct(ds, "prw")
    if direct is not None:
        return direct
    if "specific_humidity" not in ds.data_vars or "level" not in ds["specific_humidity"].dims:
        return None
    if "pressure_thickness" not in ds.data_vars and "pressure_half" not in ds.data_vars:
        return None
    return column_burden(ds, "specific_humidity")


def extract_sftlf(ds: xr.Dataset) -> xr.DataArray | None:
    """Extract land area fraction ``sftlf`` [%], preferring ``surface_type='land'`` metadata."""
    src = next(
        (
            name
            for name in sorted(ds.data_vars)
            if str(name).startswith("surface_exchange.tile_fraction.")
            and ds[name].attrs.get("surface_type") == "land"
        ),
        None,
    ) or _first_present(
        ds,
        (
            f"surface_exchange.tile_fraction.{_ECHAM_LAND_TILE_INDEX}",
            f"vertical_diffusion.surface_fraction.{_ECHAM_LAND_TILE_INDEX}",
        ),
    )
    if src is None:
        return None
    da = ds[src]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    return _to_percent(da).clip(0.0, 100.0)


def _areacella(ds: xr.Dataset) -> xr.DataArray | None:
    """Compute spherical grid-cell areas ``areacella`` [m2] via :func:`jcm.analysis.cell_area`."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None
    return cell_area(ds)


#: CMOR variables that need more than a renamed (and rescaled) source field.
_DERIVED_EXTRACTORS: dict[str, Callable[[xr.Dataset], xr.DataArray | None]] = {
    "tas": extract_tas,
    "clt": extract_clt,
    "pr": extract_pr,
    "prw": extract_prw,
    "sftlf": extract_sftlf,
    "areacella": _areacella,
}


def shared_source_entries(
    vert_by_cmor: Mapping[str, str],
    *,
    units_overrides: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, str, str, float, float]]:
    """Build a first-present-wins ``jcm_src -> (cmor, units, vert, scale, offset)`` table.

    Covers every CMOR variable in ``vert_by_cmor`` using the canonical fallback
    priority in :data:`SOURCES`, so downstream exporters share the exact same
    source priority without duplicating mappings. ``scale`` records the
    conversion :func:`to_cmor_units` applies for the entry's units.
    """
    overrides = units_overrides or {}
    out: dict[str, tuple[str, str, str, float, float]] = {}
    for cmor, vert in vert_by_cmor.items():
        units = overrides.get(cmor, CMOR_META[cmor][0])
        for src in SOURCES[cmor]:
            scale = 100.0 if units == "%" else _source_scale(src)
            out[src] = (cmor, units, vert, scale, 0.0)
    return out


def extract_direct_variable(ds: xr.Dataset, cmor: str) -> xr.DataArray | None:
    """Extract a 2D, derived, or fixed (``fx``) CMOR variable from ``ds``."""
    if cmor in _DERIVED_EXTRACTORS:
        return _DERIVED_EXTRACTORS[cmor](ds)
    return _lookup_direct(ds, cmor)


def extract_plev_variable(
    ds: xr.Dataset,
    cmor: str,
    plevs_hpa: Sequence[float],
    *,
    subset_preinterp: bool = False,
) -> xr.DataArray | None:
    """Extract and pressure-interpolate one 3D CMOR variable onto ``plevs_hpa`` [hPa]."""
    src = _first_present(ds, SOURCES[cmor])
    if src is None:
        return None

    da = ds[src]
    if "level" in da.dims:
        if "pressure_full" not in ds.data_vars:
            warnings.warn(
                f"Skipping 3D CMOR variable {cmor!r} (source {src!r}): "
                "'pressure_full' is required for vertical interpolation onto pressure levels.",
                UserWarning,
                stacklevel=2,
            )
            return None
        da = to_pressure_levels(ds[[src, "pressure_full"]], list(plevs_hpa))[src]
    elif "plev" in da.dims:
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

    return to_cmor_units(da, src, CMOR_META[cmor][0])


# --- time axis helpers ----------------------------------------------------


def _is_datetime_like(values) -> bool:
    """Return True for a datetime64 or cftime (object dtype) array of time values."""
    arr = np.asarray(values)
    return np.issubdtype(arr.dtype, np.datetime64) or (
        arr.size > 0 and hasattr(arr.flat[0], "year")
    )


def date_ymd(val) -> tuple[int, int, int]:
    """Extract ``(year, month, day)`` from a numpy datetime64, Timestamp, or cftime scalar."""
    if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        return int(val.year), int(val.month), int(val.day)
    if isinstance(val, np.datetime64):
        y, m, d = str(np.datetime_as_string(val, unit="D")).split("-")
        return int(y), int(m), int(d)
    raise TypeError(f"Unsupported time coordinate value: {val!r} ({type(val).__name__})")


def time_bounds_names(ds: xr.Dataset) -> list[str]:
    """Names of the CF time-bounds variables present in ``ds`` (the declared one first)."""
    declared = ds["time"].attrs.get("bounds", "time_bounds")
    return [name for name in dict.fromkeys((declared, "time_bounds")) if name in ds]


def get_time_bounds(ds: xr.Dataset) -> xr.DataArray | None:
    """Return the well-formed ``(time, 2)`` CF time-bounds variable of ``ds``, if any."""
    if "time" not in ds.coords:
        return None
    names = time_bounds_names(ds)
    if not names:
        return None
    bounds = ds[names[0]]
    return bounds if bounds.ndim == 2 and bounds.shape[1] == 2 else None


def _timedelta_days(delta) -> float:
    """Convert a numpy timedelta64 or datetime.timedelta to float days."""
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s")) / 86400.0
    if isinstance(delta, datetime.timedelta):
        return delta.total_seconds() / 86400.0
    raise TypeError(f"Unsupported timedelta type: {type(delta).__name__}")


def median_step_days(ds: xr.Dataset, time_bounds: xr.DataArray | None) -> float | None:
    """Estimate the median sampling/averaging interval [days] of ``ds['time']``.

    ``time_bounds`` is the result of :func:`get_time_bounds`; interval widths
    are preferred over the spacing of the time stamps when it is available.
    """
    if time_bounds is not None and time_bounds.shape[0] > 0:
        edges = time_bounds.values
        steps = edges[:, 1] - edges[:, 0]
    elif "time" in ds.coords and ds.sizes.get("time", 0) >= 2:
        steps = np.diff(ds.coords["time"].values)
    else:
        return None
    return float(np.median([_timedelta_days(step) for step in steps]))


# --- CF variable metadata, layout, and writer -----------------------------


def _cell_methods(cmor: str, table: str) -> str:
    """Return the CF-1.8 ``cell_methods`` string for ``(cmor, table)``."""
    if table == "fx":
        return "area: sum" if cmor == "areacella" else "area: mean"
    if cmor == "tasmax":
        return "area: mean time: maximum"
    if cmor == "tasmin":
        return "area: mean time: minimum"
    return "area: time: mean"


def cmor_variable(
    da: xr.DataArray,
    cmor: str,
    *,
    table: str | None = None,
    units: str | None = None,
    long_name: str | None = None,
    comment: str | None = None,
) -> xr.DataArray:
    """Rename ``da`` to ``cmor`` and replace its attributes with the CF variable attributes.

    ``units`` and ``long_name`` default to the :data:`CMOR_META` entry (the
    long name falls back to ``cmor`` itself for variables outside that table).
    ``standard_name`` is only written where :data:`CF_STANDARD_NAMES` has a
    verified mapping, and ``cell_methods`` only when a CMOR ``table`` is given.
    """
    meta = CMOR_META.get(cmor)
    if units is None and meta is not None:
        units = meta[0]
    if long_name is None:
        long_name = meta[2] if meta is not None else cmor

    attrs: dict[str, str] = {}
    if units is not None:
        attrs["units"] = units
    if cmor in CF_STANDARD_NAMES:
        attrs["standard_name"] = CF_STANDARD_NAMES[cmor]
    attrs["long_name"] = long_name
    if table is not None:
        attrs["cell_methods"] = _cell_methods(cmor, table)
    if cmor in _FLUX_POSITIVE:
        attrs["positive"] = _FLUX_POSITIVE[cmor]
    if comment is not None:
        attrs["comment"] = comment

    da = da.rename(cmor)
    da.attrs = attrs
    return da


def _drop_aux_coords(da: xr.DataArray) -> xr.DataArray:
    """Drop every coordinate of ``da`` that is not one of its dimensions."""
    return da.drop_vars([name for name in da.coords if name not in da.dims])


def to_cmip_layout(da: xr.DataArray) -> xr.DataArray:
    """Apply the CMIP layout: no auxiliary coords, ``(time, plev, lat, lon)``, south-to-north."""
    da = _drop_aux_coords(da)
    leading = [d for d in ("time", "plev", "lat", "lon") if d in da.dims]
    da = da.transpose(*leading, *(d for d in da.dims if d not in leading))
    if "lat" in da.dims and da.sizes["lat"] > 1:
        lat = da.coords["lat"].values
        if lat[0] > lat[-1]:
            da = da.isel(lat=slice(None, None, -1))
    return da


def _apply_coord_attrs(out: xr.Dataset) -> None:
    """Stamp CF coordinate attributes using :func:`jcm.cf_metadata.apply_cf_attributes`."""
    apply_cf_attributes(out)
    if "time" not in out.coords:
        return
    time = out["time"]
    # Remove encoding-owned keys from attrs so xarray.to_netcdf does not
    # fail with "failed to prevent overwriting existing key units in attrs".
    time.attrs.pop("units", None)
    time.attrs.pop("calendar", None)
    if "time" in out.sizes and _is_datetime_like(time.values):
        # ``apply_cf_attributes`` only recognises datetime64; a cftime axis is
        # datetime-like too even though its numpy dtype is object.
        time.attrs.update(standard_name="time", long_name="time", axis="T")
    if "time_bounds" in out.data_vars:
        time.attrs["bounds"] = "time_bounds"
        out["time_bounds"].attrs.pop("units", None)
        out["time_bounds"].attrs.pop("calendar", None)


def _time_encoding(out: xr.Dataset) -> dict[str, dict]:
    """Build NetCDF time encoding preserving cftime calendars when present."""
    if "time" not in out.coords:
        return {}
    tvals = out["time"].values
    if not _is_datetime_like(tvals):
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
    target: pathlib.Path,
    global_attrs: Mapping[str, str],
    *,
    time_bounds: xr.DataArray | None = None,
    compress: bool = True,
    dry_run: bool = False,
) -> str:
    """Atomically write one named CMOR variable to ``target``; return the file name.

    ``da`` is written as given (see :func:`cmor_variable` and
    :func:`to_cmip_layout`), with CF coordinate attributes and time encoding
    added. ``dry_run`` skips the write and only reports the file name.
    """
    if dry_run:
        return target.name

    data_vars: dict[str, xr.DataArray] = {da.name: da}
    if time_bounds is not None and "time" in da.dims:
        data_vars["time_bounds"] = _drop_aux_coords(time_bounds)

    out = xr.Dataset(data_vars)
    if "time" in da.coords and "calendar" in da.coords["time"].encoding:
        out["time"].encoding["calendar"] = da.coords["time"].encoding["calendar"]
    _apply_coord_attrs(out)
    out.attrs.update(global_attrs)

    encoding: dict[str, dict] = {}
    if compress:
        encoding[da.name] = {"zlib": True, "complevel": 4}
    encoding.update(_time_encoding(out))
    write_netcdf_atomic(out, target, encoding=encoding)
    return target.name
