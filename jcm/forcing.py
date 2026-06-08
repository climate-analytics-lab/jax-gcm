import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes, CoordinateSystem
from jcm.utils import VALID_TRUNCATIONS, VALID_NODAL_SHAPES, validate_ds
from jcm.data.bc.interpolate import interpolate_to_daily, upsample_forcings_ds
from jcm.date import (
    DateData,
    DEFAULT_CALENDAR,
    absolute_seconds_since_epoch,
)
from jcm.ozone_climatology import OzoneClimatology


# Sentinel for ``ForcingData.copy(nudging_target=...)`` so the field can be
# explicitly cleared by passing ``None`` (which would otherwise fall back to
# ``self.nudging_target`` under a naive ``x if x is not None else self.x``).
_UNSET = object()


def _empty_ozone_climatology() -> OzoneClimatology:
    """Sentinel ``OzoneClimatology`` used when no file is provided.

    Indirected through a helper so a future forcing-extension can
    swap the default without touching every ``zeros``/``ones``/``copy``
    call site.
    """
    return OzoneClimatology.empty()


def _validate_bc_fields(ds) -> None:
    """One-time sanity check on a loaded forcing dataset.

    Catches common authoring mistakes in the boundary-condition NetCDF
    (wrong units, AMIP-SST extrapolation over land, NaN holes, fields
    flipped in sign) before they manifest as a multi-day NaN once
    inside the JIT'd integration. Hard violations (out-of-range or
    non-finite) raise ``ValueError``; soft violations (the JSBACH-vs-
    AMIP heuristic) emit a warning and continue.

    The expected ranges below assume SI units throughout: temperatures
    in K, fractions in [0, 1], snow as a depth in mm. The downstream
    physics code assumes these conventions without re-checking.
    """
    # Hard ranges: any value outside these is a strong indication of a
    # unit error or corrupt input.
    HARD_RANGES = {
        "stl":  (180.0, 350.0),  # K — Antarctic plateau winter ≈ 180 K, hottest desert summer ≈ 330 K
        # AMIP-style SST files commonly carry the under-ice
        # temperature of the underlying water (down to ~220 K in
        # extreme Antarctic-pack winters); some authoring conventions
        # also extrapolate below freezing in fill regions. The lower
        # bound here is loose enough to admit real-world climatologies
        # while still catching unit errors (a Celsius file would have
        # values near 0).
        "sst":  (220.0, 320.0),  # K
        "icec": (0.0,   1.0),    # fraction
        "alb":  (0.0,   1.0),    # fraction
        "soilw_am": (0.0, 5.0),  # kg/m^2 (column-integrated soil water)
        "snowc":    (0.0, 20000.0),  # mm snow depth (we clip > 20000 to 0 anyway, but reject negatives)
    }
    for name, (lo, hi) in HARD_RANGES.items():
        if name not in ds.data_vars:
            continue
        arr = np.asarray(ds[name].values)
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(
                f"Forcing field '{name}' has {n_bad} non-finite values; "
                f"the integration would NaN as soon as the affected step is reached."
            )
        amin, amax = float(np.min(arr)), float(np.max(arr))
        if amin < lo or amax > hi:
            raise ValueError(
                f"Forcing field '{name}' is out of physical range "
                f"[{lo}, {hi}]: actual range [{amin:.3g}, {amax:.3g}]. "
                f"Check the units of the source NetCDF."
            )

    # Heuristic: the AMIP-SST-extrapolated stl from convert_echam_bc.py
    # without ``--land-init`` is ~stl ≈ sst everywhere, which gives a
    # large positive bias over high orography (e.g. +30 K over the
    # Tibetan plateau in DJF). If we can detect that case, warn — the
    # run will still launch, but multi-day stability over real terrain
    # has historically required the JSBACH-derived file.
    if "stl" in ds.data_vars and "sst" in ds.data_vars:
        stl = np.asarray(ds["stl"].values)
        sst = np.asarray(ds["sst"].values)
        if stl.shape == sst.shape:
            diff = np.abs(stl - sst)
            # If 99% of points have |stl - sst| < 1 K, the land field
            # is almost certainly the SST extrapolation (a real land
            # climatology has a 10-30 K spread relative to local SST
            # over continental interiors).
            if float(np.percentile(diff, 99)) < 1.0:
                warnings.warn(
                    "Forcing 'stl' is within 1 K of 'sst' for ≥99% of grid "
                    "points — this looks like the AMIP-SST extrapolation "
                    "produced by ``convert_echam_bc.py`` without "
                    "``--land-init``. Multi-day runs over real terrain "
                    "have historically NaN'd from the resulting +30 K "
                    "bias over high orography (Tibetan / Antarctic "
                    "plateaus). Regenerate the BC file with the JSBACH "
                    "initial-conditions file (e.g. "
                    "``ic_land_soil_T63GR15_*.nc``) to use the real "
                    "land surface temperature climatology.",
                    UserWarning,
                    stacklevel=3,
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

# Default scalar CH4 mixing ratio (ppmv) when no time series is supplied.
# 1.9 ppmv ≈ early-2020s tropospheric mean (CH4 has roughly doubled since
# pre-industrial); previously hardcoded inside ``EchamBoundaryConditions``
# as ``1900.0e-3`` ppmv. Issue #347.
DEFAULT_CH4_VMR_PPMV = 1.9


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

    # CH4 volume mixing ratio (ppmv). Scalar for fixed-CH4 runs; TimeSeries
    # for historical / scenario forcing. Was previously hardcoded inside
    # ``EchamBoundaryConditions``; promoted to forcing in #347.
    ch4_vmr: jnp.ndarray

    # Aerosol temporal forcing (MACv2-SP plume weights). Today these are
    # placeholder 1-D `(nplumes,)` arrays; the multi-axis version will land
    # in the MACv2-SP fix PR (#437).
    aerosol_year_weight: jnp.ndarray
    aerosol_ann_cycle: jnp.ndarray

    # Solar/orbital geometry. Absent on user-built `ForcingData` (left as a
    # null SolarGeometry); populated by `select(date)` on every step.
    solar: SolarGeometry

    # Pre-computed climatological ozone profile (annual mean today;
    # ``select(date)`` will eventually slice monthly / scenario-year as
    # needed). Empty sentinel when no climatology file is provided, in
    # which case downstream radiation falls back to an analytical
    # profile (see :class:`jcm.physics.chemistry.OzoneClimatology`).
    ozone_climatology: OzoneClimatology

    # Optional nudging reference fields. Each can be a static array or a
    # :class:`TimeSeries` leaf; ``ForcingData.select`` slices the whole
    # struct, so :class:`jcm.nudging.NudgingTerm` sees a target that has
    # already been collapsed for the current step — no date plumbing into
    # the physics path. Default ``None`` for runs without nudging.
    nudging_target: Any = None

    @classmethod
    def zeros(cls,nodal_shape,
              alb0=None,sice_am=None,snowc_am=None,
              soilw_am=None,stl_am=None,sea_surface_temperature=None,
              co2_vmr=None,
              aerosol_year_weight=None,aerosol_ann_cycle=None,
              solar=None,
              ozone_climatology=None,
              ch4_vmr=None,
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
            ch4_vmr=ch4_vmr if ch4_vmr is not None else jnp.array(DEFAULT_CH4_VMR_PPMV),
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else jnp.ones(nplumes),
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else jnp.ones(nplumes),
            solar=solar if solar is not None else SolarGeometry.zero(),
            ozone_climatology=(
                ozone_climatology if ozone_climatology is not None
                else _empty_ozone_climatology()
            ),
        )

    @classmethod
    def ones(cls,nodal_shape,
             alb0=None,sice_am=None,snowc_am=None,
             soilw_am=None,stl_am=None,sea_surface_temperature=None,
             co2_vmr=None,
             aerosol_year_weight=None,aerosol_ann_cycle=None,
             solar=None,
             ozone_climatology=None,
             ch4_vmr=None,
             nplumes=9):
        return cls(
            alb0=alb0 if alb0 is not None else jnp.ones((nodal_shape)),
            sice_am=sice_am if sice_am is not None else jnp.ones((nodal_shape)),
            snowc_am=snowc_am if snowc_am is not None else jnp.ones((nodal_shape)),
            soilw_am=soilw_am if soilw_am is not None else jnp.ones((nodal_shape)),
            stl_am =stl_am if stl_am is not None else jnp.ones((nodal_shape)),
            sea_surface_temperature=sea_surface_temperature if sea_surface_temperature is not None else jnp.ones((nodal_shape)),
            co2_vmr=co2_vmr if co2_vmr is not None else jnp.array(DEFAULT_CO2_VMR_PPMV),
            ch4_vmr=ch4_vmr if ch4_vmr is not None else jnp.array(DEFAULT_CH4_VMR_PPMV),
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else jnp.ones(nplumes),
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else jnp.ones(nplumes),
            solar=solar if solar is not None else SolarGeometry.zero(),
            ozone_climatology=(
                ozone_climatology if ozone_climatology is not None
                else _empty_ozone_climatology()
            ),
        )

    @classmethod
    def from_file(cls, filename: str, coords: CoordinateSystem = None,
                  align_mode: str = "auto", validate: bool = True):
        """Initialize forcing data from a netCDF file.

        Thin wrapper around `from_dataset`: opens `filename` with xarray
        and delegates. See `from_dataset` for argument semantics. The
        ``validate`` flag forwards to `from_dataset` (default ``True``;
        pass ``False`` to bypass the BC sanity check, e.g. for synthetic
        test fixtures).
        """
        import xarray as xr
        return cls.from_dataset(xr.open_dataset(filename), coords=coords,
                                align_mode=align_mode, validate=validate)

    @classmethod
    def from_dataset(cls, ds, coords: CoordinateSystem = None,
                     align_mode: str = "auto", validate: bool = True):
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
        # Sanity-check the loaded BC values once on the host before
        # entering the JIT pipeline. Raises on hard violations (units,
        # NaN, out-of-physical-range), warns on the AMIP-SST
        # extrapolation heuristic — see docstring. ``validate=False``
        # is for synthetic test fixtures that intentionally use
        # zero-filled or out-of-range data to exercise the time/shape
        # plumbing.
        if validate:
            _validate_bc_fields(ds)
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
        elif ds["stl"].shape[:2] == coords.horizontal.nodal_shape:
            # Source already at target resolution — skip the lat/lon interp
            # pipeline (which can introduce NaN through pole padding when
            # lat values match exactly). Only do the monthly -> daily time
            # interpolation for a 12-month climatology; native daily or
            # multi-year axes are passed through to the TimeSeries/BY_DATE
            # alignment unchanged (interpolate_to_daily requires exactly 12
            # monthly timestamps and would otherwise raise).
            if _is_monthly_climatology(ds):
                ds = interpolate_to_daily(ds)
        else:
            base = interpolate_to_daily(ds) if _is_monthly_climatology(ds) else ds
            ds = upsample_forcings_ds(base, grid=coords.horizontal)

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

        # Sea-ice concentration. Clip to [0, 1] — spectral interpolation
        # of a near-zero field can leave float-precision negatives (~1e-18),
        # which downstream scheme guards (e.g. ``sqrt(1 - sice)``) treat
        # as NaNs.
        sice_am = _ts(jnp.clip(jnp.asarray(ds["icec"]), 0.0, 1.0))

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
             solar=None,
             ozone_climatology=None,
             ch4_vmr=None,
             nudging_target=_UNSET):
        # ``nudging_target`` uses an ``_UNSET`` sentinel because ``None`` is
        # the natural value for "no nudging target wired" — falling back to
        # ``self.nudging_target`` only when the caller didn't supply the
        # kwarg lets ``.copy(nudging_target=None)`` *clear* the field.
        return ForcingData(
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowc_am=snowc_am if snowc_am is not None else self.snowc_am,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            stl_am =stl_am if stl_am is not None else self.stl_am,
            sea_surface_temperature=sea_surface_temperature if sea_surface_temperature is not None else self.sea_surface_temperature,
            co2_vmr=co2_vmr if co2_vmr is not None else self.co2_vmr,
            ch4_vmr=ch4_vmr if ch4_vmr is not None else self.ch4_vmr,
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else self.aerosol_year_weight,
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else self.aerosol_ann_cycle,
            solar=solar if solar is not None else self.solar,
            ozone_climatology=(
                ozone_climatology if ozone_climatology is not None
                else self.ozone_climatology
            ),
            nudging_target=(
                nudging_target if nudging_target is not _UNSET
                else self.nudging_target
            ),
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


def _is_monthly_climatology(ds) -> bool:
    """Return ``True`` if ``ds`` has a 12-step (monthly-climatology) time axis.

    ``interpolate_to_daily`` only accepts exactly 12 monthly timestamps (it pads
    with adjacent-year Dec/Jan and raises otherwise). Native daily or multi-year
    boundary files therefore must skip it and flow straight to the
    ``TimeSeries``/``BY_DATE`` alignment. This mirrors ``interpolate_to_daily``'s
    own contract (a length check), so a same-grid file is only treated as a
    monthly climatology when it actually has 12 timesteps.
    """
    return "time" in ds.dims and ds.sizes.get("time") == 12


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
    # `date.tyear(calendar)` is in [0, 1) by construction in date.py.
    idx = jnp.floor(date.tyear(calendar) * n_time).astype(jnp.int32) % n_time
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
    """Build a `SolarGeometry` from a `DateData`, parameterized by calendar.

    Calendar-aware fraction of year (Gregorian honours leap years; see
    `fraction_of_year_elapsed`). The orbital phase tracks the same
    fraction so the solar declination matches the actual day-of-year
    (#410). `synodic_phase` is fraction-of-day × 2π, calendar-independent.
    """
    fraction_of_day = date.dt.delta.seconds / 86400.0
    tyear = date.tyear(calendar)
    two_pi = 2.0 * jnp.pi
    return SolarGeometry(
        tyear=jnp.asarray(tyear, dtype=jnp.float32),
        orbital_phase=jnp.asarray(two_pi * tyear, dtype=jnp.float32),
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
