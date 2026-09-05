"""ERA5-on-the-model-grid loading for the bias-correction trainers.

The offline notebook carried this regrid inline; online training needs the same preprocessing
at 6-hourly cadence (the online targets live at sub-daily lead times), so the
shared piece is promoted here rather than copied a third time. Notebook 07
keeps its inline copy deliberately: it documents the offline run exactly as it was
executed.

Everything returns **model-state units**: u, v in m/s, T in K, and specific
humidity in g/kg. ERA5 stores q in kg/kg; the conversion happens here, once,
because feeding kg/kg into a g/kg state silently relaxes the model toward
~1/1000 of the intended humidity (the kg/kg-vs-g/kg units bug).

The regrid is intentionally simple, matching the offline run: horizontal interpolation
onto the model lon/lat (in degrees), and a log-pressure vertical remap onto
``sigma * p0`` with a single reference surface pressure. Per-column surface
pressure is a later refinement.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

from jcm.data.bc.interpolate_ozone import vertical_interp_log_p

# NOTE the trailing date: this store ends on 10 January 2023, so 2023 holds
# only 40 six-hourly slots and the last COMPLETE year is 2022. Requesting a
# full year beyond that returns a short axis rather than an error, which is
# why era5_cache.build_year checks the slot count explicitly.
WB2_URL = ("gs://weatherbench2/datasets/era5/"
           "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")
VARS_3D = ("u_wind", "v_wind", "temperature", "specific_humidity")
P0 = 1.0e5  # Pa, reference surface pressure for the sigma -> pressure map

# The WB2 zarr is 6-hourly; cadences are expressed in these source steps.
_SOURCE_HOURS = 6

# `Model.__init__` defaults to SPEEDY's `365_day` calendar, so a model year is
# exactly this many days and 29 February does not exist.
DAYS_PER_MODEL_YEAR = 365


def cadence_slicing(cadence_hours: int, n_days: int) -> tuple[int, int]:
    """Slot arithmetic for sampling the 6-hourly source at ``cadence_hours``.

    Kept as a pure function because notebook 08's target-slot indexing
    depends on it lining up with the loaded time axis: a change here that
    is not mirrored there mis-pairs rollout ends with ERA5 targets.

    Returns:
        ``(stride, n_slices)``: take every ``stride``-th source slot,
        ``n_slices`` of them covering ``n_days``.

    """
    if cadence_hours % _SOURCE_HOURS:
        raise ValueError(f"cadence_hours must be a multiple of {_SOURCE_HOURS}")
    stride = cadence_hours // _SOURCE_HOURS
    per_day = 24 // _SOURCE_HOURS
    return stride, n_days * per_day


def era5_ds_to_model_grid(ds: xr.Dataset, coords) -> dict[str, np.ndarray]:
    """Regrid an ERA5-convention dataset onto the model grid.

    Args:
        ds: dataset with variables :data:`VARS_3D` on
            ``(time, level, lat, lon)``; ``level`` in hPa, ``lat``/``lon`` in
            degrees, humidity in kg/kg (raw ERA5 conventions).
        coords: model CoordinateSystem.

    Returns:
        ``{name: (ntime, nlev, nlon, nlat) ndarray}`` in model-state units.

    """
    lat_deg = np.degrees(np.asarray(coords.horizontal.latitudes))
    lon_deg = np.degrees(np.asarray(coords.horizontal.longitudes))

    # Longitude is periodic (0 == 360), but xarray's interp treats it as a
    # bounded axis and would linearly EXTRAPOLATE past the last source
    # longitude, corrupting the model columns near the seam (e.g. the T31
    # 356.25 column against WB2's last source lon 354.375). Append a cyclic
    # point carrying the first column's values at lon0 + 360 so the seam
    # interpolates by wrapping. fill_value stays for latitude, where the
    # Gaussian grid can poke marginally past the source cell centers.
    src = ds[list(VARS_3D)]
    wrap = src.isel(lon=0).assign_coords(lon=float(src.lon[0]) + 360.0)
    src = xr.concat([src, wrap], dim="lon")
    horiz = src.interp({"lat": lat_deg, "lon": lon_deg},
                       kwargs={"fill_value": "extrapolate"})

    p_src = np.asarray(ds.level.values) * 100.0          # hPa -> Pa
    p_tgt = np.asarray(coords.vertical.centers) * P0     # sigma * p0
    fields = {}
    for v in VARS_3D:
        a = np.asarray(horiz[v].transpose("time", "level", "lon", "lat").values)
        fields[v] = vertical_interp_log_p(a, p_src, p_tgt)

    # kg/kg -> g/kg, the PhysicsState convention (see module docstring).
    fields["specific_humidity"] = fields["specific_humidity"] * 1000.0
    return fields


def times_to_seconds(times) -> np.ndarray:
    """Time coordinate values -> seconds since 1970, the TimeSeries clock."""
    seconds = (pd.DatetimeIndex(np.asarray(times))
               - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
    return np.asarray(seconds, dtype=float)


def _resolve_era5_url(url):
    """Caller arg wins; else the JCM_ERA5_URL env var; else the default store."""
    return url if url is not None else os.environ.get("JCM_ERA5_URL", WB2_URL)


def _open_kwargs(url):
    """gs:// needs the anonymous gcsfs token; a local path must NOT get it."""
    if url.startswith("gs://"):
        return {"consolidated": True, "storage_options": {"token": "anon"}}
    return {"consolidated": None, "storage_options": None}


def _open_era5(url: str | None) -> xr.Dataset:
    """Open the source store and rename its axes/fields to model conventions."""
    url = _resolve_era5_url(url)
    era5 = xr.open_zarr(url, **_open_kwargs(url))
    return era5.rename({"latitude": "lat", "longitude": "lon",
                        "u_component_of_wind": "u_wind",
                        "v_component_of_wind": "v_wind"})


def drop_leap_day(ds: xr.Dataset) -> xr.Dataset:
    """Drop 29 February so a calendar year carries exactly 365 days.

    The model runs SPEEDY's ``365_day`` calendar (``Model.__init__``'s
    default), which has no 29 February, while ERA5 is true Gregorian. Keeping
    the leap day would put every later sample of a leap year one day out of
    seasonal phase against the model's day-of-year: model day 59 is 1 March,
    but the 60th ERA5 slot of a leap year is 29 February. Nothing errors --
    the labels are just silently shifted -- so this is dropped at load time
    rather than left to each caller.
    """
    t = ds["time"].dt
    keep = np.asarray(~((t.month == 2) & (t.day == 29)))
    return ds.isel(time=np.flatnonzero(keep))


def model_clock_seconds(start_year: int, n_samples: int,
                        cadence_hours: int) -> np.ndarray:
    """Sample times on the model's 365-day clock, in seconds since 1970.

    :func:`jcm.date.absolute_seconds_since_epoch` returns *real elapsed*
    seconds and is calendar-agnostic, so a ``365_day`` run advances exactly
    86400 s per model day regardless of leap years. Labelling the samples with
    their true Gregorian timestamps would therefore drift against the model
    clock by one day per leap year, and a ``BY_DATE`` nudging or forcing
    lookup would ``searchsorted`` into the wrong slot -- the same class of
    silent mis-pairing as the 24 h offline label lag.

    Counting uniformly from 1 January of ``start_year`` keeps file and model
    on one clock. In a non-leap year this is identical to the real timestamps,
    so single-year runs (2001, 2002) are unaffected.
    """
    start = (pd.Timestamp(f"{start_year}-01-01")
             - pd.Timestamp("1970-01-01")).total_seconds()
    return start + np.arange(n_samples, dtype=float) * cadence_hours * 3600.0


def check_year_is_complete(n_got: int, n_days: int, cadence_hours: int,
                           year: int) -> None:
    """Raise if the source returned fewer samples than ``n_days`` needs.

    The WeatherBench2 store ends part way through its final year, so asking
    for a full year of 2023 quietly returns ten days. Nothing downstream
    errors on that: a nudged run would still execute all 365 days with a
    target that ran out, clamping to its last slot, and a stitched multi-year
    axis would silently stop being uniform. Both surface much later as
    unexplainable results, so the length is checked at the point of loading.
    """
    want = n_days * (24 // cadence_hours)
    if n_got != want:
        raise ValueError(
            f"{year}: source has {n_got} slots at {cadence_hours} h cadence, "
            f"expected {want} for {n_days} days. The store does not cover all "
            f"of {year}; drop it from the span.")


def _year_fields(era5: xr.Dataset, coords, year: int, n_days: int,
                 cadence_hours: int) -> dict[str, np.ndarray]:
    """Regrid ``n_days`` from 1 January of ``year``, leap day removed."""
    stride, n_slices = cadence_slicing(cadence_hours, n_days)
    sub = drop_leap_day(era5.sel(time=str(year)))
    sub = sub.isel(time=slice(0, n_slices, stride))
    check_year_is_complete(sub.sizes["time"], n_days, cadence_hours, year)
    return era5_ds_to_model_grid(sub, coords)


def load_era5(coords, year: int, n_days: int, *,
              cadence_hours: int = 6, url: str | None = None):
    """Stream ERA5 u, v, T, q for one year onto the model grid.

    29 February is dropped and the returned axis is the model's 365-day clock;
    see :func:`drop_leap_day` and :func:`model_clock_seconds` for why. Any
    year is therefore safe to request, not only non-leap ones.

    Args:
        coords: model CoordinateSystem.
        year: calendar year to load.
        n_days: number of days from 1 January (at most 365).
        cadence_hours: sampling cadence; must be a multiple of the source's
            6 h (6 for online targets, 24 reproduces the offline daily slices).
        url: zarr store; defaults to the JCM_ERA5_URL env var, else the
            64x32 WeatherBench2 store (anonymous). Local paths open without
            the gcsfs anon token.

    Returns:
        ``(fields, time_seconds)`` where ``fields[name]`` is
        ``(nt, nlev, nlon, nlat)`` in model-state units and ``time_seconds``
        is the sample axis in seconds since 1970.

    """
    fields = _year_fields(_open_era5(url), coords, year, n_days, cadence_hours)
    n = next(iter(fields.values())).shape[0]
    return fields, model_clock_seconds(year, n, cadence_hours)


def load_era5_span(coords, start_year: int, n_days: int, *,
                   cadence_hours: int = 6, url: str | None = None):
    """Stream ERA5 across consecutive years onto the model grid.

    :func:`load_era5` covers one calendar year, but a 450-day eval crosses a
    year boundary and multi-year training spans many. Each year contributes
    exactly :data:`DAYS_PER_MODEL_YEAR` days once the leap day is dropped,
    which is what makes the single continuous 365-day clock correct across the
    whole span -- stitching true Gregorian timestamps instead would leave a
    step at every leap year.

    Everything is concatenated in memory, so this suits spans of a few years.
    A two-decade span is ~17 GB on the T31 grid and needs an on-disk cache
    instead.

    Args:
        coords: model CoordinateSystem.
        start_year: calendar year of the first sample (1 January).
        n_days: total days to load, spilling into later years as needed.
        cadence_hours: sampling cadence, as :func:`load_era5`.
        url: zarr store, as :func:`load_era5`.

    Returns:
        ``(fields, time_seconds)``, same layout as :func:`load_era5`.

    """
    era5 = _open_era5(url)
    parts, remaining, year = [], int(n_days), int(start_year)
    while remaining > 0:
        take = min(remaining, DAYS_PER_MODEL_YEAR)
        parts.append(_year_fields(era5, coords, year, take, cadence_hours))
        remaining -= take
        year += 1

    fields = {k: np.concatenate([p[k] for p in parts], axis=0)
              for k in parts[0]}
    n = next(iter(fields.values())).shape[0]
    return fields, model_clock_seconds(start_year, n, cadence_hours)
