import jax.numpy as jnp
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes, CoordinateSystem
from jcm.utils import VALID_TRUNCATIONS, VALID_NODAL_SHAPES, validate_ds
from jcm.data.bc.interpolate import interpolate_to_daily, upsample_forcings_ds
from jcm.date import (
    DateData,
    DEFAULT_CALENDAR,
    absolute_seconds_since_epoch,
    days_per_year,
)

# `TimeSeries.align_mode` constants. Stored as ints rather than strings so the
# struct stays a clean JAX pytree (string fields can't ride through `jit`).
WRAP_YEAR = 0   # index by `floor(date.tyear * n_time) % n_time` — climatology mode
BY_DATE = 1     # index by absolute time, using `time_seconds` as the lookup axis

# Default scalar CO2 mixing ratio (ppmv) when no time series is supplied. 360
# ppmv is SPEEDY's reference 1990s baseline, which the legacy `ablco2_ref`
# constant was tuned against — keeping the default at 360 ppmv means runs that
# do not pass a CO2 forcing reproduce SPEEDY's pre-`increase_co2` behavior.
DEFAULT_CO2_VMR_PPMV = 360.0


# ---------------------------------------------------------------------------
# Leaf wrappers
# ---------------------------------------------------------------------------


@tree_math.struct
class TimeSeries:
    """A time-varying forcing leaf.

    `values` carries the data with a time axis at index 0; `time_seconds`
    is a 1-D coordinate (seconds since `MODEL_EPOCH`) used by `BY_DATE`
    indexing. `align_mode` is `WRAP_YEAR` or `BY_DATE`. The Model collapses
    every `TimeSeries` leaf to its current-step slice via
    `ForcingData.select(date)` before handing the forcing to physics, so
    physics terms always see the leading-time axis already removed.
    """

    values: jnp.ndarray
    time_seconds: jnp.ndarray
    align_mode: jnp.ndarray   # int scalar, stored as a 0-d jnp array


def make_time_series(values, time_seconds, align_mode=BY_DATE):
    """Build a `TimeSeries` leaf with the given alignment mode."""
    return TimeSeries(
        values=jnp.asarray(values),
        time_seconds=jnp.asarray(time_seconds),
        align_mode=jnp.asarray(align_mode, dtype=jnp.int32),
    )


@tree_math.struct
class SolarGeometry:
    """Per-step solar/orbital geometry derived from `DateData`.

    Populated by `ForcingData.select(date)`, consumed by radiation schemes.
    Carrying it on `forcing` lets physics keep its `(state, forcing, terrain)`
    signature and stop reading `DateData` directly.
    """

    tyear: jnp.ndarray            # fractional year [0, 1) — SPEEDY shortwave
    orbital_phase: jnp.ndarray    # 2π × fraction-of-year, jax_solar convention
    synodic_phase: jnp.ndarray    # 2π × fraction-of-day,   jax_solar convention

    @classmethod
    def zero(cls):
        """Build a null SolarGeometry for placeholder / static `ForcingData` objects."""
        zero = jnp.zeros((), dtype=jnp.float32)
        return cls(tyear=zero, orbital_phase=zero, synodic_phase=zero)


# ---------------------------------------------------------------------------
# ForcingData
# ---------------------------------------------------------------------------


@tree_math.struct
class ForcingData:
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray # sea ice concentration (or TimeSeries thereof)
    snowc_am: jnp.ndarray # snow cover (used to be snowcl_ob in fortran - but one day of that was snowc_am)
    soilw_am: jnp.ndarray # soil moisture (used to be soilwcl_ob in fortran - but one day of that was soilw_am)
    stl_am: jnp.ndarray # temperature over land
    sea_surface_temperature: jnp.ndarray # SST, should come from sea_model.py or some default value

    # CO2 volume mixing ratio (ppmv). Scalar for fixed-CO2 runs; TimeSeries for
    # historical / scenario forcing. Replaces the old date-driven `ablco2`
    # ramp under `ForcingParameters.increase_co2` (#285).
    co2_vmr: jnp.ndarray

    # Aerosol temporal forcing (MACv2-SP plume weights). Today these are
    # placeholder 1-D `(nplumes,)` arrays; the multi-axis version will land
    # in the MACv2-SP fix PR (#437).
    aerosol_year_weight: jnp.ndarray
    aerosol_ann_cycle: jnp.ndarray

    # Solar/orbital geometry. Absent on user-built `ForcingData` (left as a
    # null SolarGeometry); populated by `select(date)` on every step.
    solar: SolarGeometry

    @classmethod
    def zeros(cls,nodal_shape,
              alb0=None,sice_am=None,snowc_am=None,
              soilw_am=None,stl_am=None,sea_surface_temperature=None,
              co2_vmr=None,
              aerosol_year_weight=None,aerosol_ann_cycle=None,
              solar=None,
              nplumes=9):
        # Land + SST temperatures default to ~15 °C — a sensible global
        # mean surface temperature — so that ``ForcingData.zeros(...)``
        # yields a physically plausible state when no forcing file is
        # supplied and the surface flux scheme isn't presented with an
        # unphysical ΔT against the atmosphere.
        T_default = 288.15
        return cls(
            alb0=alb0 if alb0 is not None else jnp.zeros((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.zeros((nodal_shape)),
            snowc_am=snowc_am if snowc_am is not None else jnp.zeros((nodal_shape)),
            soilw_am=soilw_am if soilw_am is not None else jnp.zeros((nodal_shape)),
            stl_am=stl_am if stl_am is not None else jnp.full(nodal_shape, T_default),
            sea_surface_temperature=sea_surface_temperature if sea_surface_temperature is not None else jnp.full(nodal_shape, T_default),
            co2_vmr=co2_vmr if co2_vmr is not None else jnp.array(DEFAULT_CO2_VMR_PPMV),
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else jnp.ones(nplumes),
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else jnp.ones(nplumes),
            solar=solar if solar is not None else SolarGeometry.zero(),
        )

    @classmethod
    def ones(cls,nodal_shape,
             alb0=None,sice_am=None,snowc_am=None,
             soilw_am=None,stl_am=None,sea_surface_temperature=None,
             co2_vmr=None,
             aerosol_year_weight=None,aerosol_ann_cycle=None,
             solar=None,
             nplumes=9):
        return cls(
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)),
            snowc_am=snowc_am if snowc_am is not None else jnp.ones((nodal_shape)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)),
            stl_am =stl_am if stl_am is not None else jnp.ones((nodal_shape)),
            sea_surface_temperature=sea_surface_temperature if sea_surface_temperature is not None else jnp.ones((nodal_shape)),
            co2_vmr=co2_vmr if co2_vmr is not None else jnp.array(DEFAULT_CO2_VMR_PPMV),
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else jnp.ones(nplumes),
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else jnp.ones(nplumes),
            solar=solar if solar is not None else SolarGeometry.zero(),
        )

    @classmethod
    def from_file(cls, filename: str, coords: CoordinateSystem = None,
                  align_mode: str = "auto"):
        """Initialize forcing data from a netCDF file.

        Thin wrapper around `from_dataset`: opens `filename` with xarray
        and delegates. See `from_dataset` for argument semantics.
        """
        import xarray as xr
        return cls.from_dataset(xr.open_dataset(filename), coords=coords,
                                align_mode=align_mode)

    @classmethod
    def from_dataset(cls, ds, coords: CoordinateSystem = None,
                     align_mode: str = "auto"):
        """Initialize forcing data from an in-memory xarray Dataset.

        Time-varying variables are wrapped as `TimeSeries` leaves so the
        Model can pre-slice them per step via `select(date)`. Static
        variables (`alb`) stay as bare 2-D arrays.

        Args:
            ds: An `xarray.Dataset` carrying the expected forcing fields.
            coords: CoordinateSystem to upscale to. If None, the dataset's
                native nodal shape is used.
            align_mode: "auto" (default) chooses `wrap_year` for files that
                cover at most one calendar year and `by_date` for longer
                spans; pass `"wrap_year"` or `"by_date"` to force the
                choice. `wrap_year` indexes the time axis by fraction of
                year (climatology mode); `by_date` aligns by absolute
                model date.

        """
        expected_structure = {
            "stl":      ("lon", "lat", "time"),
            "icec":     ("lon", "lat", "time"),
            "sst":      ("lon", "lat", "time"),
            "alb":      ("lon", "lat"),
            "soilw_am": ("lon", "lat", "time"),
            "snowc":    ("lon", "lat", "time"),
        }

        validate_ds(ds, expected_structure)
        # the spectral resolution is total wavenumbers - 2
        target_resolution = coords.horizontal.total_wavenumbers - 2 if coords is not None else None

        if target_resolution is None:
            ix, il, n_times = ds['stl'].shape
            if (ix, il) not in VALID_NODAL_SHAPES:
                raise ValueError(f"Invalid nodal shape: {(ix, il)}. Must be one of: {VALID_NODAL_SHAPES}.")
            # No assumption that n_times == 365 — multi-year files welcome.
            # FIXME: Consider validating lat/lon values here - would have to construct a coords object to get expected values though
        elif target_resolution not in VALID_TRUNCATIONS:
            raise ValueError(f"Invalid target resolution: {target_resolution}. Must be one of: {VALID_TRUNCATIONS}.")
        else:
            ds = upsample_forcings_ds(interpolate_to_daily(ds), grid=coords.horizontal)

        # Build the shared time axis (seconds since MODEL_EPOCH) for every
        # time-varying variable in this file, plus the alignment mode.
        time_seconds = _time_axis_seconds_from_ds(ds)
        resolved_align_mode = _resolve_align_mode(align_mode, ds)

        def _ts(values):
            """Wrap an `(lon, lat, time)` array as a `TimeSeries` leaf with
            time as the leading axis (matching `_select_time_series`'s
            convention).
            """
            arr = jnp.asarray(values)
            arr = jnp.moveaxis(arr, -1, 0)  # (time, lon, lat)
            return make_time_series(arr, time_seconds, align_mode=resolved_align_mode)

        # annual-mean surface albedo (no time axis)
        alb0 = jnp.asarray(ds["alb"])

        # sea ice concentration
        sice_am = _ts(ds["icec"])

        # snow depth (clip implausible values, same as before)
        snowc_raw = jnp.asarray(ds["snowc"])
        snowc_valid = (0.0 <= snowc_raw) & (snowc_raw <= 20000.0)
        snowc_clean = jnp.where(snowc_valid, snowc_raw, 0.0)
        snowc_am = _ts(snowc_clean)

        # soil moisture
        soilw_am = _ts(ds["soilw_am"])

        stl_am = _ts(ds["stl"])

        # Prescribed SSTs
        sea_surface_temperature = _ts(ds["sst"])

        # Optional CO2: if the netCDF includes it, treat as a scalar (per-time)
        # series; otherwise keep the default scalar from `ForcingData.zeros`.
        co2_vmr = None
        if "co2" in ds.data_vars:
            co2_arr = jnp.asarray(ds["co2"])
            if co2_arr.ndim == 0:
                co2_vmr = co2_arr
            else:
                co2_vmr = make_time_series(co2_arr, time_seconds, align_mode=resolved_align_mode)

        return cls.zeros(
            nodal_shape=alb0.shape,
            alb0=alb0, sice_am=sice_am, snowc_am=snowc_am, stl_am=stl_am,
            soilw_am=soilw_am, sea_surface_temperature=sea_surface_temperature,
            co2_vmr=co2_vmr,
        )

    def copy(self,alb0=None,
             sice_am=None,snowc_am=None,soilw_am=None, stl_am=None,
             sea_surface_temperature=None,
             co2_vmr=None,
             aerosol_year_weight=None,aerosol_ann_cycle=None,
             solar=None):
        return ForcingData(
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowc_am=snowc_am if snowc_am is not None else self.snowc_am,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            stl_am =stl_am if stl_am is not None else self.stl_am,
            sea_surface_temperature=sea_surface_temperature if sea_surface_temperature is not None else self.sea_surface_temperature,
            co2_vmr=co2_vmr if co2_vmr is not None else self.co2_vmr,
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else self.aerosol_year_weight,
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else self.aerosol_ann_cycle,
            solar=solar if solar is not None else self.solar,
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(jnp.any, self))

    def select(self, date: DateData, calendar: str = DEFAULT_CALENDAR) -> "ForcingData":
        """Collapse every `TimeSeries` leaf to the current step's slice and
        populate `solar` from `date`.

        Static fields pass through unchanged. Returns a new `ForcingData`
        whose every leaf is the shape physics expects (no leading time axis).
        """
        sliced = _slice_time_series_leaves(self, date, calendar=calendar)
        return sliced.copy(solar=_solar_from_date(date, calendar=calendar))


# ---------------------------------------------------------------------------
# Time selection helpers
# ---------------------------------------------------------------------------


def _time_axis_seconds_from_ds(ds) -> jnp.ndarray:
    """Convert a netCDF dataset's `time` coordinate to seconds since
    `MODEL_EPOCH` (1970-01-01 UTC). Returned as a 1-D float array.
    """
    import pandas as pd
    import numpy as np
    times = pd.DatetimeIndex(ds["time"].values)
    epoch = pd.Timestamp("1970-01-01")
    delta = (times - epoch).total_seconds().to_numpy()
    return jnp.asarray(np.asarray(delta, dtype=float))


def _resolve_align_mode(align_mode: str, ds) -> int:
    """Pick `WRAP_YEAR` vs `BY_DATE` from a string spec ("auto"/"wrap_year"/"by_date").

    `auto` chooses `wrap_year` when the file's time span fits in a single
    year (climatology) and `by_date` otherwise.
    """
    if align_mode == "wrap_year":
        return WRAP_YEAR
    if align_mode == "by_date":
        return BY_DATE
    if align_mode != "auto":
        raise ValueError(
            f"Unknown align_mode {align_mode!r}; expected 'auto', 'wrap_year', or 'by_date'"
        )
    # Auto-detect: if the time axis spans <= ~1.05 years, treat as climatology.
    import pandas as pd
    times = pd.DatetimeIndex(ds["time"].values)
    if len(times) <= 1:
        return WRAP_YEAR
    span_days = (times[-1] - times[0]).days
    return WRAP_YEAR if span_days <= 380 else BY_DATE


def _slice_time_series_leaves(forcing: ForcingData, date: DateData, calendar: str) -> ForcingData:
    """Return `forcing` with every `TimeSeries` leaf replaced by its slice
    at `date`. Non-`TimeSeries` leaves are passed through unchanged.
    """
    def slice_leaf(leaf):
        if isinstance(leaf, TimeSeries):
            return _select_time_series(leaf, date, calendar=calendar)
        return leaf

    return tree_util.tree_map(
        slice_leaf,
        forcing,
        is_leaf=lambda x: isinstance(x, TimeSeries),
    )


def _select_time_series(ts: TimeSeries, date: DateData, calendar: str) -> jnp.ndarray:
    """Index `ts.values` along the leading time axis at `date`."""
    n_time = ts.values.shape[0]
    if n_time == 0:
        # Defensive — shouldn't happen, but a 0-length axis would NaN downstream
        return ts.values

    # Both branches have to produce the same shape, which they do (scalar idx).
    idx_wrap = _wrap_year_index(n_time, date, calendar=calendar)
    idx_date = _by_date_index(ts.time_seconds, date)

    idx = jnp.where(ts.align_mode == BY_DATE, idx_date, idx_wrap)
    idx = jnp.clip(idx, 0, n_time - 1)
    return jnp.take(ts.values, idx, axis=0)


def _wrap_year_index(n_time: int, date: DateData, calendar: str) -> jnp.ndarray:
    """Climatological wrap: split the year evenly into `n_time` bins."""
    # `date.tyear` is already in [0, 1) (mod-1 by construction in date.py).
    idx = jnp.floor(date.tyear * n_time).astype(jnp.int32) % n_time
    return idx


def _by_date_index(time_seconds: jnp.ndarray, date: DateData) -> jnp.ndarray:
    """Date-aligned: nearest `time_seconds` entry at-or-before `date`."""
    target = absolute_seconds_since_epoch(date.dt)
    # `searchsorted(side='right') - 1` puts us at the entry whose timestamp
    # is the latest one <= target, which is the natural piecewise-constant
    # left interpretation of the forcing axis.
    raw = jnp.searchsorted(time_seconds, target, side='right') - 1
    return jnp.clip(raw, 0, time_seconds.shape[0] - 1).astype(jnp.int32)


def _solar_from_date(date: DateData, calendar: str) -> SolarGeometry:
    """Build a `SolarGeometry` from a `DateData`, parameterized by calendar."""
    dpy = days_per_year(calendar)
    # `jax_solar`-style phases. Replicates the math in
    # `OrbitalTime.from_datetime(when, days_per_year=dpy)`:
    #   fraction_of_day  = dt.delta.seconds / 86400
    #   fraction_of_year = ((dt.delta.days + fraction_of_day) / dpy) % 1
    #   orbital_phase    = 2π * fraction_of_year
    #   synodic_phase    = 2π * fraction_of_day
    fraction_of_day = date.dt.delta.seconds / 86400.0
    days_total = date.dt.delta.days + fraction_of_day
    fraction_of_year = (days_total / dpy) % 1.0
    two_pi = 2.0 * jnp.pi
    return SolarGeometry(
        tyear=jnp.asarray(date.tyear, dtype=jnp.float32),
        orbital_phase=jnp.asarray(two_pi * fraction_of_year, dtype=jnp.float32),
        synodic_phase=jnp.asarray(two_pi * fraction_of_day, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def _fixed_ssts(grid: HorizontalGridTypes) -> jnp.ndarray:
    """Return an array of SSTs with simple cos^2 profile from 300K at the equator to 273K at 60 degrees latitude.
    Obtained from Neale, R.B. and Hoskins, B.J. (2000),
    "A standard test for AGCMs including their physical parametrizations: I: the proposal."
    Atmosph. Sci. Lett., 1: 101-107. https://doi.org/10.1006/asle.2000.0022
    """
    lat = grid.latitudes
    sst_profile = jnp.where(jnp.abs(lat) < jnp.pi/3, 27*jnp.cos(3*lat/2)**2, 0) + 273.15
    return jnp.tile(sst_profile, (grid.nodal_shape[0], 1))

def default_forcing(
    grid: HorizontalGridTypes,
) -> ForcingData:
    """Initialize the default forcing data with prescribed SSTs"""
    sea_surface_temperature = _fixed_ssts(grid)

    return ForcingData.zeros(
        nodal_shape=grid.nodal_shape,sea_surface_temperature=sea_surface_temperature,
    )
