import jax
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
        """A null SolarGeometry for placeholder/static `ForcingData` objects."""
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
    def from_file(cls, filename: str, coords: CoordinateSystem = None):
        """Initialize forcing data from a file.

        Args:
            filename: Path to the forcing data file

        Returns:
            ForcingData: Time-varying forcing data

        """
        import xarray as xr

        # Read forcing data from file
        ds = xr.open_dataset(filename)

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
            if n_times != 365:
                raise ValueError(f"Expected 365 time steps, got {n_times}.")
            # FIXME: Consider validating lat/lon values here - would have to construct a coords object to get expected values though
        elif target_resolution not in VALID_TRUNCATIONS:
            raise ValueError(f"Invalid target resolution: {target_resolution}. Must be one of: {VALID_TRUNCATIONS}.")
        else:
            ds = upsample_forcings_ds(interpolate_to_daily(ds), grid=coords.horizontal)

        # annual-mean surface albedo
        alb0 = jnp.asarray(ds["alb"])

        # sea ice concentration
        sice_am = jnp.asarray(ds["icec"])

        # snow depth
        snowc_am = jnp.asarray(ds["snowc"])
        snowc_valid = (0.0 <= snowc_am) & (snowc_am <= 20000.0)
        # assert jnp.all(snowc_valid | (fmask[:,:,jnp.newaxis] == 0.0)) # FIXME: need to change the forcing.nc file so this passes
        snowc_am = jnp.where(snowc_valid, snowc_am, 0.0)

        # soil moisture
        soilw_am = jnp.asarray(ds["soilw_am"])

        stl_am = jnp.asarray(ds["stl"])

        # Prescribe SSTs
        sea_surface_temperature = jnp.asarray(ds["sst"])

        return cls.zeros(
            nodal_shape=alb0.shape,
            alb0=alb0, sice_am=sice_am, snowc_am=snowc_am,stl_am=stl_am,
            soilw_am=soilw_am, sea_surface_temperature=sea_surface_temperature
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
