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
BY_DATE_INTERP = 2  # as BY_DATE, but linearly interpolate between samples.
                    # The right mode for AMIP mid-month boundary values
                    # (PCMDI ``tosbcs`` is constructed so that linear
                    # interpolation reconstructs the observed monthly means).

# Default scalar CO2 mixing ratio (ppmv) when no time series is supplied.
# 420 ppmv is the value the ECHAM/RRTMGP physics was calibrated against (it was
# previously hard-coded in ``ChemistryParameters``); SPEEDY's ``ablco2`` simply
# scales linearly against its own reference, so this single forcing default now
# drives every backend's CO2.
DEFAULT_CO2_VMR_PPMV = 420.0

# Default scalar CH4 mixing ratio (ppmv) when no time series is supplied.
# 1.9 ppmv ≈ early-2020s tropospheric mean (CH4 has roughly doubled since
# pre-industrial); previously hardcoded inside ``EchamBoundaryConditions``
# as ``1900.0e-3`` ppmv. Issue #347.
DEFAULT_CH4_VMR_PPMV = 1.9

# Default scalar N2O mixing ratio (ppmv) when no time series is supplied.
# 0.327 ppmv (327 ppbv) matches the value RRTMGP previously took from its
# ``vmr_global_means.json`` fallback, so prescribing N2O from the forcing here
# preserves the calibrated radiative effect while removing the silent fallback.
DEFAULT_N2O_VMR_PPMV = 0.327


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

    # N2O volume mixing ratio (ppmv). Scalar for fixed-N2O runs; TimeSeries for
    # historical / scenario forcing. Prescribed here so RRTMGP no longer falls
    # back silently to its ``vmr_global_means.json`` value.
    n2o_vmr: jnp.ndarray

    # Aerosol temporal forcing (MACv2-SP plume weights): year_weight is
    # `(nplumes,)` (CEDS amplitude relative to 2005 for the current year),
    # ann_cycle is `(nfeatures, nplumes)` (per-feature weekly cycle for the
    # current date). The all-ones defaults mean "perpetual year-2005
    # amplitude, no seasonal cycle" — a documented convenience, NOT the
    # historical forcing; real time series come from MACv2.0-SP_v1.nc via
    # the notebook-06 TimeSeries recipe (piecewise-constant per year /
    # per 1/52-year bin, mind the _FillValue masking beyond 2016).
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

    # Prescribed natural-aerosol emission surface fields (or TimeSeries
    # thereof), read from the forcing file when present and ``None`` otherwise
    # — the JAM emission terms fall back to zero on a ``None`` field, so DMS /
    # dust emission is simply inert until the field is supplied.
    dms_seawater: Any = None   # seawater DMS concentration kg/m³ (DmsEmissions)
    dust_source: Any = None    # dust source / erodibility 0–1 (DustEmissions)

    # Prescribed oxidant volume mixing ratios for the JAM sulfur chemistry
    # (#496 follow-up): a mapping ``{"oh"|"no3"|"o3"|"h2o2": TimeSeries}`` of
    # mole-fraction fields on the **model levels**, each shaped
    # ``(time, nlev, lon, lat)`` (see :func:`read_oxidant_vmr`).
    # ``PrescribedOxidants`` converts VMR → molec cm⁻³ in-term, where the
    # current T and p are available; kept as one dict-valued field (like
    # ``anthropogenic_emissions``) so ``select(date)`` slices the per-species
    # ``TimeSeries`` leaves like any other forcing leaf. ``None`` ⇒ the term
    # keeps its analytic interim proxies.
    oxidant_vmr: Any = None

    # Prescribed anthropogenic aerosol emissions (#498). A single mapping
    # ``{emis_<sector>_<species>: array | TimeSeries}`` of **bulk** per-super-
    # sector surface mass fluxes [kg/m²/s] on the model grid (so2 as SO₂ mass;
    # bc/oc as carbon mass — see the emissions-file contract in
    # ``.claude/aerosol_emissions_plan.md``). ``None`` ⇒ no anthropogenic
    # emission. Kept as one dict-valued field rather than a field per
    # (sector, species) so new channels need no struct change; ``select(date)``
    # slices the per-channel ``TimeSeries`` leaves like any other forcing leaf.
    anthropogenic_emissions: Any = None

    # Prescribed *already-speciated* aerosol emissions (#498), the CAM6/MAM4-
    # faithful counterpart to ``anthropogenic_emissions``. A mapping
    # ``{tracer_name: array | TimeSeries}`` keyed by the tracer each field feeds
    # (e.g. ``m_so4_acc``, ``m_bc_pcm``, ``n_pcm``, ``g_so2``); 2-D fields are
    # surface fluxes, 3-D ``(lev, …)`` fields are per-model-level volume fluxes
    # (see :class:`PreSpeciatedEmissions`). ``None`` ⇒ no prescribed emission.
    prescribed_aerosol_emissions: Any = None

    @classmethod
    def zeros(cls,nodal_shape,
              alb0=None,sice_am=None,snowc_am=None,
              soilw_am=None,stl_am=None,sea_surface_temperature=None,
              co2_vmr=None,
              aerosol_year_weight=None,aerosol_ann_cycle=None,
              solar=None,
              ozone_climatology=None,
              ch4_vmr=None,
              n2o_vmr=None,
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
            n2o_vmr=n2o_vmr if n2o_vmr is not None else jnp.array(DEFAULT_N2O_VMR_PPMV),
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else jnp.ones(nplumes),
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else jnp.ones((2, nplumes)),
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
             n2o_vmr=None,
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
            n2o_vmr=n2o_vmr if n2o_vmr is not None else jnp.array(DEFAULT_N2O_VMR_PPMV),
            aerosol_year_weight=aerosol_year_weight if aerosol_year_weight is not None else jnp.ones(nplumes),
            aerosol_ann_cycle=aerosol_ann_cycle if aerosol_ann_cycle is not None else jnp.ones((2, nplumes)),
            solar=solar if solar is not None else SolarGeometry.zero(),
            ozone_climatology=(
                ozone_climatology if ozone_climatology is not None
                else _empty_ozone_climatology()
            ),
        )

    @classmethod
    def from_file(cls, filename, coords: CoordinateSystem = None,
                  align_mode: str = "auto", validate: bool = True):
        """Initialize forcing data from one or more netCDF files.

        Thin wrapper around `from_dataset`: opens `filename` with xarray
        and delegates. A list/tuple of paths (e.g. the yearly transient
        AMIP bundles, issue #610) is concatenated along ``time`` in
        chronological order; the merged span then drives the ``auto``
        alignment detection, so a multi-year sequence aligns ``by_date``.
        The ``validate`` flag forwards to `from_dataset` (default ``True``;
        pass ``False`` to bypass the BC sanity check, e.g. for synthetic
        test fixtures).
        """
        import xarray as xr
        if isinstance(filename, (list, tuple)):
            ds = xr.open_mfdataset(
                [str(f) for f in filename], combine="by_coords",
                data_vars="minimal", coords="minimal", compat="override",
            ).sortby("time").load()
        else:
            ds = xr.open_dataset(filename)
        return cls.from_dataset(ds, coords=coords,
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
                spans; pass `"wrap_year"`, `"by_date"` or
                `"by_date_interp"` to force the choice. `wrap_year` indexes
                the time axis by fraction of year (climatology mode);
                `by_date` aligns by absolute model date (piecewise
                constant); `by_date_interp` additionally interpolates
                linearly between samples — required for AMIP mid-month
                boundary values (``tosbcs``) to reconstruct monthly means.

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

        # Resolve the alignment mode from the *raw* time axis, before any
        # monthly -> daily interpolation: a single-year transient file (12
        # mid-month steps, ``align_mode="by_date_interp"``) must keep its
        # real dates — ``interpolate_to_daily`` is a climatology transform
        # and only applies when the file actually wraps the year.
        resolved_align_mode = _resolve_align_mode(align_mode, ds)
        is_wrap_year_monthly = (resolved_align_mode == WRAP_YEAR
                                and _is_monthly_climatology(ds))

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
            if is_wrap_year_monthly:
                ds = interpolate_to_daily(ds)
        else:
            base = interpolate_to_daily(ds) if is_wrap_year_monthly else ds
            ds = upsample_forcings_ds(base, grid=coords.horizontal)

        # Build the shared time axis (seconds since MODEL_EPOCH) for every
        # time-varying variable in this file.
        time_seconds = _time_axis_seconds_from_ds(ds)

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

        # Optional well-mixed GHG scalars (CO2/CH4/N2O): if the netCDF includes
        # one, treat it as a scalar (per-time) series; otherwise keep the default
        # from `ForcingData.zeros`. Radiation reads CO2 and N2O straight from the
        # forcing (and CH4 via the chemistry seed), so reading every prescribed
        # gas here — not just CO2 — is what lets a scenario file actually drive
        # them rather than silently fall back to the default.
        def _optional_ghg(name):
            if name not in ds.data_vars:
                return None
            arr = jnp.asarray(ds[name])
            if arr.ndim == 0:
                return arr
            return make_time_series(arr, time_seconds, align_mode=resolved_align_mode)

        co2_vmr = _optional_ghg("co2")
        ch4_vmr = _optional_ghg("ch4")
        n2o_vmr = _optional_ghg("n2o")

        return cls.zeros(
            nodal_shape=alb0.shape,
            alb0=alb0, sice_am=sice_am, snowc_am=snowc_am, stl_am=stl_am,
            soilw_am=soilw_am, sea_surface_temperature=sea_surface_temperature,
            co2_vmr=co2_vmr, ch4_vmr=ch4_vmr, n2o_vmr=n2o_vmr,
        )

    def copy(self,alb0=None,
             sice_am=None,snowc_am=None,soilw_am=None, stl_am=None,
             sea_surface_temperature=None,
             co2_vmr=None,
             aerosol_year_weight=None,aerosol_ann_cycle=None,
             solar=None,
             ozone_climatology=None,
             ch4_vmr=None,
             n2o_vmr=None,
             nudging_target=_UNSET,
             dms_seawater=None,
             dust_source=None,
             oxidant_vmr=None,
             anthropogenic_emissions=None,
             prescribed_aerosol_emissions=None):
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
            n2o_vmr=n2o_vmr if n2o_vmr is not None else self.n2o_vmr,
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
            dms_seawater=dms_seawater if dms_seawater is not None else self.dms_seawater,
            dust_source=dust_source if dust_source is not None else self.dust_source,
            oxidant_vmr=oxidant_vmr if oxidant_vmr is not None else self.oxidant_vmr,
            anthropogenic_emissions=(
                anthropogenic_emissions if anthropogenic_emissions is not None
                else self.anthropogenic_emissions
            ),
            prescribed_aerosol_emissions=(
                prescribed_aerosol_emissions
                if prescribed_aerosol_emissions is not None
                else self.prescribed_aerosol_emissions
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

    Handles both numpy ``datetime64`` axes (standard/proleptic-Gregorian) and
    ``cftime`` axes from non-standard calendars — the CESM emission files use a
    ``365_day`` (noleap) calendar, which xarray decodes to ``cftime`` objects
    that pandas can't ingest.

    Both kinds are placed on the **same Gregorian clock the model runs on** by
    aligning on the *nominal* calendar date ``(year, month, day, …)``. This is
    deliberate: the ``BY_DATE`` lookup target is
    :func:`jcm.date.absolute_seconds_since_epoch`, which is built from
    ``jax_datetime`` and is leap-aware Gregorian (the model has no real noleap
    clock — see #449). Converting a ``365_day`` axis with *noleap day-counting*
    (e.g. ``cftime.date2num(..., calendar='365_day')``) would instead drift
    against that target by the accumulated leap days (~7 days by 2000, growing
    every leap year), so ``searchsorted`` would pick the wrong slice and corrupt
    multi-year prescribed-emissions runs. Mapping each cftime date by its
    calendar components onto the Gregorian epoch keeps file and model on one
    clock; noleap dates are always valid Gregorian dates (no 29 Feb), so the
    mapping is exact.
    """
    import numpy as np
    import pandas as pd
    vals = np.asarray(ds["time"].values)
    if vals.dtype == object:
        # cftime objects (a non-standard calendar like 365_day). Reinterpret
        # each by its (y, m, d, h, m, s) components on the Gregorian clock —
        # NOT by the file calendar's day count — so the axis matches the model's
        # leap-aware lookup target (see docstring).
        import datetime as _dt
        flat = np.ravel(vals)
        py_dates = [
            _dt.datetime(d.year, d.month, d.day,
                         getattr(d, "hour", 0), getattr(d, "minute", 0),
                         getattr(d, "second", 0))
            for d in flat
        ]
        idx = pd.DatetimeIndex(py_dates) if py_dates else pd.DatetimeIndex([])
        delta = (idx - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
    else:
        # datetime64, or plain numeric (pandas interprets the latter as ns).
        delta = (pd.DatetimeIndex(vals)
                 - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
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
    if align_mode == "by_date_interp":
        return BY_DATE_INTERP
    if align_mode != "auto":
        raise ValueError(
            f"Unknown align_mode {align_mode!r}; expected 'auto', 'wrap_year', "
            "'by_date', or 'by_date_interp'"
        )
    # Auto-detect: if the time axis spans <= ~1.05 years, treat as climatology.
    # Reuse the calendar-aware seconds conversion so non-standard (cftime)
    # calendars work here too.
    import numpy as np
    seconds = np.asarray(_time_axis_seconds_from_ds(ds))
    if seconds.size <= 1:
        return WRAP_YEAR
    span_days = (seconds[-1] - seconds[0]) / 86400.0
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

    # Every branch has to produce the same shape, which they do (scalar idx).
    idx_wrap = _wrap_year_index(n_time, date, calendar=calendar)
    idx_date = _by_date_index(ts.time_seconds, date)

    idx = jnp.where(ts.align_mode == WRAP_YEAR, idx_wrap, idx_date)
    idx = jnp.clip(idx, 0, n_time - 1)
    stepped = jnp.take(ts.values, idx, axis=0)
    if n_time < 2:
        return stepped

    # BY_DATE_INTERP: linear interpolation between the bracketing samples,
    # clamped to the end values outside the axis. `align_mode` is traced, so
    # both the stepped and interpolated values are computed and selected with
    # `where` (cheap: one extra gather + fma per leaf per step).
    lo = jnp.clip(idx_date, 0, n_time - 2)
    t_lo = jnp.take(ts.time_seconds, lo)
    t_hi = jnp.take(ts.time_seconds, lo + 1)
    target = absolute_seconds_since_epoch(date.dt)
    frac = jnp.clip((target - t_lo) / jnp.maximum(t_hi - t_lo, 1e-9), 0.0, 1.0)
    interp = ((1.0 - frac) * jnp.take(ts.values, lo, axis=0)
              + frac * jnp.take(ts.values, lo + 1, axis=0))
    return jnp.where(ts.align_mode == BY_DATE_INTERP, interp, stepped)


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

def read_anthropogenic_emissions(ds, align_mode: str = "auto"):
    """Build the ``ForcingData.anthropogenic_emissions`` mapping from a dataset.

    Reads every ``emis_<sector>_<species>`` variable (the emissions-file
    contract — see ``.claude/aerosol_emissions_plan.md``) and wraps each as a
    ``TimeSeries`` leaf (time-varying) or a bare array (static), keyed by the
    variable name. Returns ``None`` when the dataset carries no such variable,
    so the result can be passed straight to ``ForcingData.copy``::

        ds = xr.open_dataset(emissions_file)
        forcing = forcing.copy(anthropogenic_emissions=read_anthropogenic_emissions(ds))

    The fields must already be on the model horizontal grid: this does **no**
    regridding (use :mod:`jcm.data.emissions.prepare` to conservatively remap a
    source-grid file first). Monthly data uses the same wrap-year/by-date time
    alignment as the other forcing fields, so a 12-month climatology wraps and a
    multi-year axis aligns by date.
    """
    emis_names = [str(v) for v in ds.data_vars if str(v).startswith("emis_")]
    if not emis_names:
        return None
    has_time = any("time" in ds[n].dims for n in emis_names)
    time_seconds = _time_axis_seconds_from_ds(ds) if has_time else None
    mode = _resolve_align_mode(align_mode, ds) if has_time else BY_DATE
    out: dict[str, Any] = {}
    for name in emis_names:
        da = ds[name]
        if "time" in da.dims:
            # Lead with time so `_select_time_series` can index axis 0; keep the
            # remaining (horizontal) axes in their file order — the term ravels
            # them to (ncols,).
            others = [d for d in da.dims if d != "time"]
            arr = jnp.asarray(da.transpose("time", *others).values)
            out[name] = make_time_series(arr, time_seconds, align_mode=mode)
        else:
            out[name] = jnp.asarray(da.values)
    return out


def read_prescribed_aerosol_emissions(ds, align_mode: str = "auto"):
    """Build ``ForcingData.prescribed_aerosol_emissions`` from a dataset.

    Reads every ``aero_emis_<tracer>`` variable (the already-speciated emissions
    contract — see ``.claude/aerosol_emissions_plan.md``), keyed by the bare
    tracer name (``aero_emis_m_so4_acc`` → ``m_so4_acc``), and wraps each as a
    ``TimeSeries`` leaf (time-varying) or a bare array (static). Returns ``None``
    when no such variable is present. Fields may be 2-D (``lon, lat`` surface) or
    3-D (``lev, lon, lat`` volume); the non-time axes are kept in file order
    (``lev`` before the horizontal), which :class:`PreSpeciatedEmissions`
    reshapes to ``(nlev, ncols)``. Fields must already be on the model grid (no
    regridding here — use :mod:`jcm.data.emissions.prepare`).
    """
    prefix = "aero_emis_"
    names = [str(v) for v in ds.data_vars if str(v).startswith(prefix)]
    if not names:
        return None
    has_time = any("time" in ds[n].dims for n in names)
    time_seconds = _time_axis_seconds_from_ds(ds) if has_time else None
    mode = _resolve_align_mode(align_mode, ds) if has_time else BY_DATE
    out: dict[str, Any] = {}
    for name in names:
        da = ds[name]
        key = name[len(prefix):]
        if "time" in da.dims:
            others = [d for d in da.dims if d != "time"]
            arr = jnp.asarray(da.transpose("time", *others).values)
            out[key] = make_time_series(arr, time_seconds, align_mode=mode)
        else:
            out[key] = jnp.asarray(da.values)
    return out


# ---------------------------------------------------------------------------
# Natural-emission / oxidant climatology readers (HAMMOZ-style files)
# ---------------------------------------------------------------------------

# DMS molar mass [kg/mol] — same value as MAM4-JAX / the JAM gas registry
# (``jcm.physics.aerosol.jam.gas_species.GAS_SPECIES['dms']``); duplicated here
# as a plain constant so the core forcing module doesn't import physics.
_DMS_MOLAR_MASS_KG = 0.0621324

# nmol/L → kg/m³:  1 nmol/L = 1e-9 mol / 1e-3 m³ = 1e-6 mol/m³, × M_DMS.
_NMOL_PER_L_TO_KG_M3 = 1.0e-6 * _DMS_MOLAR_MASS_KG

# Unit strings accepted for the seawater DMS field. The Lana et al. (2011)
# HAMMOZ file uses ``nanomol l-1``.
_DMS_NMOL_UNITS = {"nanomol l-1", "nmol l-1", "nmol/l", "nanomol/l"}
_DMS_KG_M3_UNITS = {"kg m-3", "kg/m3", "kg m^-3"}

_LATLON_TOL_DEG = 1e-3


def _orient_to_model_grid(da, lat_deg=None, lon_deg=None, name=""):
    """Reorient a ``(..., lat, lon)`` DataArray to the model's ``(..., lon, lat)``.

    The HAMMOZ/ECHAM climatology files store fields ``(time[, lev], lat, lon)``
    with *descending* latitude (N→S), while the model's nodal layout is
    ``(lon, lat)`` with dinosaur's *ascending* Gaussian latitudes (S→N). This
    helper (a) validates the file's lat/lon values against the model's when
    given (same N points but a flipped/shifted axis would otherwise wire the
    field into the wrong columns silently — the same failure mode the ozone
    loader guards against), flipping a descending latitude axis to match,
    and (b) transposes so the trailing axes are ``(lon, lat)``, matching the
    raveled-column order the physics terms use.

    Returns a numpy array with dims ``(*others, lon, lat)`` where ``others``
    preserves the file order of the remaining dims (e.g. ``time``, ``mlev``).
    """
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(
            f"{name or da.name}: expected 'lat' and 'lon' dims, got {da.dims}."
        )
    file_lat = np.asarray(da["lat"].values, dtype=float)
    if lat_deg is not None:
        lat_deg = np.asarray(lat_deg, dtype=float)
        if np.allclose(file_lat, lat_deg, atol=_LATLON_TOL_DEG):
            pass
        elif np.allclose(file_lat[::-1], lat_deg, atol=_LATLON_TOL_DEG):
            # N→S file on a S→N model grid — flip to model orientation.
            da = da.isel(lat=slice(None, None, -1))
        else:
            raise ValueError(
                f"{name or da.name}: file latitudes "
                f"[{file_lat[0]:.3f}..{file_lat[-1]:.3f}] match the model grid "
                f"[{lat_deg[0]:.3f}..{lat_deg[-1]:.3f}] neither directly nor "
                "flipped. Regrid the file onto the model Gaussian grid first."
            )
    elif file_lat.size > 1 and file_lat[0] > file_lat[-1]:
        # No model grid to validate against (e.g. unit tests) — still
        # normalise to ascending latitude, the model convention.
        da = da.isel(lat=slice(None, None, -1))
    if lon_deg is not None:
        file_lon = np.mod(np.asarray(da["lon"].values, dtype=float), 360.0)
        lon_deg = np.mod(np.asarray(lon_deg, dtype=float), 360.0)
        if not np.allclose(file_lon, lon_deg, atol=_LATLON_TOL_DEG):
            raise ValueError(
                f"{name or da.name}: file longitudes "
                f"[{file_lon[0]:.3f}..{file_lon[-1]:.3f}] don't match the "
                f"model grid [{lon_deg[0]:.3f}..{lon_deg[-1]:.3f}]. Regrid "
                "the file onto the model Gaussian grid first."
            )
    others = [d for d in da.dims if d not in ("lat", "lon")]
    return np.asarray(da.transpose(*others, "lon", "lat").values)


def read_dms_seawater(ds, lat_deg=None, lon_deg=None, var_name="DMS_sea",
                      align_mode: str = "wrap_year"):
    """Read a seawater-DMS climatology into a ``ForcingData.dms_seawater`` leaf.

    Expects the HAMMOZ ``emiss_fields_dms_sea_monthly_T63.nc`` layout: a
    ``DMS_sea (time, lat, lon)`` monthly climatology of the Lana et al. (2011)
    surface-ocean DMS concentration in **nmol/L**. :class:`DmsEmissions`
    multiplies the field directly by the piston velocity [m/s] and treats the
    product as a mass flux [kg-DMS/m²/s], so the concentration is converted to
    **kg-DMS/m³** here (1 nmol/L = 1e-6 mol/m³ × 0.0621324 kg/mol ≈
    6.213e-8 kg/m³); a file already in ``kg m-3`` passes through. Any other
    (or missing) ``units`` attribute raises rather than guessing — a wrong
    unit would silently scale the global DMS source by ~1e7.

    Returned as a monthly ``WRAP_YEAR`` :class:`TimeSeries` shaped
    ``(time, lon, lat)`` on the model orientation (see
    :func:`_orient_to_model_grid`).
    """
    if var_name not in ds.data_vars:
        raise ValueError(
            f"DMS file has no {var_name!r} variable; found "
            f"{sorted(map(str, ds.data_vars))}."
        )
    da = ds[var_name]
    units = str(da.attrs.get("units", "")).strip().lower()
    arr = _orient_to_model_grid(da, lat_deg, lon_deg, name=var_name)
    # The HAMMOZ file marks land / no-data cells with ``_FillValue = 0``,
    # which xarray decodes to NaN (~30% of cells). Missing seawater DMS means
    # no emission, so map non-finite → 0 rather than let NaN reach the flux.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if units in _DMS_NMOL_UNITS:
        arr = arr * _NMOL_PER_L_TO_KG_M3
    elif units in _DMS_KG_M3_UNITS:
        pass
    else:
        raise ValueError(
            f"{var_name}: unrecognised units {units!r}; expected a seawater "
            "concentration in 'nanomol l-1' (converted to kg/m³ here) or "
            "'kg m-3'."
        )
    return make_time_series(
        arr, _time_axis_seconds_from_ds(ds), _resolve_align_mode(align_mode, ds)
    )


def read_dust_source(ds, lat_deg=None, lon_deg=None, var_name="pot_source",
                     align_mode: str = "wrap_year"):
    """Read a dust-source/erodibility climatology for ``ForcingData.dust_source``.

    Handles both layouts :class:`DustEmissions` accepts:

    * The HAMMOZ ``dust_potential_sources_T63.nc`` monthly climatology,
      ``pot_source (time, lat, lon)`` (Tegen 2002 potential-dust-source
      fraction) — returned as a monthly ``WRAP_YEAR`` :class:`TimeSeries`
      shaped ``(time, lon, lat)``.
    * A **static** potential-source / erodibility map with no time axis,
      ``pot_source (lat, lon)`` — returned as a bare ``(lon, lat)`` array.
      ``ForcingData.select`` passes non-``TimeSeries`` leaves through
      untouched, so the same field reaches ``DustEmissions`` every step.

    :class:`DustEmissions`' contract is a dimensionless erodibility in
    **[0, 1]**, so values are clipped — the file encodes missing cells as
    ``-1`` (its ``missing`` attribute), which the clip maps to zero (no
    source), and interpolation overshoot above 1 is capped.
    """
    if var_name not in ds.data_vars:
        raise ValueError(
            f"Dust-source file has no {var_name!r} variable; found "
            f"{sorted(map(str, ds.data_vars))}."
        )
    arr = _orient_to_model_grid(ds[var_name], lat_deg, lon_deg, name=var_name)
    # Missing cells (NaN after decode, or the raw ``-1`` marker) mean "no dust
    # source"; NaN would pass straight through ``clip``, so zero it first.
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    if "time" not in ds[var_name].dims:
        # Static (lat, lon) map → bare (lon, lat) array. No time axis to
        # build a TimeSeries from, and DustEmissions reads a 2-D field
        # directly.
        return jnp.asarray(arr)
    return make_time_series(
        arr, _time_axis_seconds_from_ds(ds), _resolve_align_mode(align_mode, ds)
    )


# Oxidant-file variable → ``ForcingData.oxidant_vmr`` key. The MACC/HAMMOZ
# ``ham_oxidants_monthly_*.nc`` files also carry ``NO2_VMR_avrg``, which the
# JAM sulfur chemistry doesn't consume — it is deliberately not read.
_OXIDANT_VAR_MAP = {
    "OH_VMR_avrg": "oh",
    "NO3_VMR_avrg": "no3",
    "O3_VMR_avrg": "o3",
    "H2O2_VMR_avrg": "h2o2",
}


def read_oxidant_vmr(ds, nlev: int, lat_deg=None, lon_deg=None,
                     align_mode: str = "wrap_year"):
    """Read a monthly oxidant climatology for ``ForcingData.oxidant_vmr``.

    Expects the HAMMOZ/MACC ``ham_oxidants_monthly_T63L47_macc.nc`` layout:
    ``OH/NO3/O3/H2O2_VMR_avrg (time, mlev, lat, lon)`` mole fractions
    [mole/mole] on ECHAM hybrid model levels (``hyam``/``hybm``/``p0``
    present), levels ordered **top→bottom** — the same ordering as the model
    state (index ``-1`` = surface).

    The fields are kept as **VMR** (not converted to molec cm⁻³): the number
    density conversion needs the instantaneous T and p, which only the
    :class:`~jcm.physics.aerosol.jam.chemistry.oxidants.PrescribedOxidants`
    term has, so the conversion happens in-term. The vertical is mapped
    **level-for-level** onto the model levels under the documented assumption
    that the file is already on the model's hybrid grid (e.g. T63L47 with
    ``grid=echam_t63_l47_hybrid``); ``nlev`` is asserted here, and
    ``runners._attach_oxidants`` additionally cross-checks ``hyam``/``hybm``
    against the model's hybrid coefficients. A bottom-up level axis
    (decreasing ``hybm``) raises.

    Returns ``{"oh"|"no3"|"o3"|"h2o2": TimeSeries}`` with values shaped
    ``(time, nlev, lon, lat)`` on the model orientation, ``WRAP_YEAR`` by
    default.
    """
    missing = [v for v in _OXIDANT_VAR_MAP if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Oxidant file is missing {missing}; expected all of "
            f"{sorted(_OXIDANT_VAR_MAP)} (found "
            f"{sorted(map(str, ds.data_vars))})."
        )
    # The level dim is whatever remains once time/lat/lon are accounted for
    # (``mlev`` in the HAMMOZ files, ``lev`` elsewhere).
    sample = ds[next(iter(_OXIDANT_VAR_MAP))]
    lev_dims = [d for d in sample.dims if d not in ("time", "lat", "lon")]
    if len(lev_dims) != 1:
        raise ValueError(
            f"Oxidant variables must be (time, lev, lat, lon); got dims "
            f"{sample.dims}."
        )
    nlev_file = int(ds.sizes[lev_dims[0]])
    if nlev_file != nlev:
        raise ValueError(
            f"Oxidant file has {nlev_file} levels but the model has {nlev}. "
            "The file must already be on the model's hybrid levels (e.g. the "
            "T63L47 file with grid=echam_t63_l47_hybrid) — no vertical "
            "interpolation is done here."
        )
    if "hybm" in ds:
        hybm = np.asarray(ds["hybm"].values, dtype=float)
        if hybm.size > 1 and hybm[0] > hybm[-1]:
            raise ValueError(
                "Oxidant file levels are ordered bottom→top (hybm decreasing) "
                "but the model expects top→bottom (surface last). Flip the "
                "level axis of the file."
            )
    time_seconds = _time_axis_seconds_from_ds(ds)
    mode = _resolve_align_mode(align_mode, ds)
    out: dict[str, Any] = {}
    for var, key in _OXIDANT_VAR_MAP.items():
        arr = _orient_to_model_grid(ds[var], lat_deg, lon_deg, name=var)
        # Defensive: a fill-value cell decoded to NaN means "no data" — treat
        # as zero oxidant rather than let NaN poison the sulfur chemistry.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = make_time_series(arr, time_seconds, align_mode=mode)
    return out


def default_forcing(
    grid: HorizontalGridTypes,
) -> ForcingData:
    """Initialize the default forcing data with prescribed SSTs"""
    sea_surface_temperature = _fixed_ssts(grid)

    return ForcingData.zeros(
        nodal_shape=grid.nodal_shape,sea_surface_temperature=sea_surface_temperature,
    )
