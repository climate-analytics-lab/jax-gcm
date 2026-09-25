import warnings
from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
from jax import tree_util
from dinosaur.coordinate_systems import HorizontalGridTypes, CoordinateSystem
from jcm.utils import VALID_TRUNCATIONS, VALID_NODAL_SHAPES, validate_ds
from jcm.data.bc.interpolate import interpolate_to_daily, upsample_forcings_ds
# ``{year}`` pattern expansion lives in the import-free engine
# :mod:`jcm.data.input_resolution` so ``tools/benchmark.py`` can load it by file
# path (jcm-free, before its GPU gate) and share this single source of truth;
# forcing.py imports JAX/dinosaur/``jcm`` at module top and so cannot itself be
# that shared leaf. Re-exported here — its historical home — for the runner and
# tests (``from jcm.forcing import expand_yearly_files``).
from jcm.data.input_resolution import expand_yearly_files as expand_yearly_files
from jcm.date import (
    DateData,
    gregorian_ymd_from_days,
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
        # Relative soil wetness ws/wsmx — a fraction by construction, so
        # anything outside [0, 1] means it was written as a volumetric content
        # or a water depth instead (#787).
        "soilw_rel": (0.0, 1.0),
        # Static land-cover fractions for the ECHAM land albedo (#672), and
        # the land share the conditional fields are regridded with.
        "forest": (0.0, 1.0),
        "glac": (0.0, 1.0),
        "lsm": (0.0, 1.0),
        # Snow cover in the SPEEDY convention SWE/sd2sc, whose min(1, .) is
        # the cover fraction: the mirror bundles and the packaged T63 file
        # store it already clipped to [0, 1]; the SPEEDY T30 file stores the
        # unclipped ratio (up to ~170). Reject negatives and absurd values.
        "snowc":    (0.0, 20000.0),
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
WRAP_YEAR = 0   # climatology replayed every year, by real calendar position
BY_DATE = 1     # piecewise-constant lookup on exact datetimes
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

    `values` carries the data with a time axis at index 0; `times` is its exact
    :class:`jax_datetime.Datetime` coordinate. `align_mode` distinguishes
    dated samples (``BY_DATE`` / ``BY_DATE_INTERP``) from a climatology
    replayed every year (``WRAP_YEAR``). The Model collapses
    every `TimeSeries` leaf to its current-step slice via
    `ForcingData.select(date)` before handing the forcing to physics, so
    physics terms always see the leading-time axis already removed.
    """

    values: jnp.ndarray
    times: jdt.Datetime
    align_mode: jnp.ndarray   # int scalar, stored as a 0-d jnp array


def make_time_series(values, times, align_mode=BY_DATE):
    """Build a validated `TimeSeries` with a strictly increasing time axis."""
    values = jnp.asarray(values)
    raw_times = np.asarray(times) if not isinstance(times, jdt.Datetime) else None
    if raw_times is not None and np.issubdtype(raw_times.dtype, np.datetime64):
        if np.any(np.isnat(raw_times)):
            raise ValueError("TimeSeries times cannot contain NaT")
        whole_seconds = raw_times.astype("datetime64[s]")
        if np.any((raw_times - whole_seconds) != np.timedelta64(0, "s")):
            raise ValueError("TimeSeries times must have whole-second precision")
    times = jdt.to_datetime(times)
    if values.shape[0] != times.delta.days.shape[0]:
        raise ValueError(
            f"values has {values.shape[0]} records but times has "
            f"{times.delta.days.shape[0]}"
        )
    # Forcing is loaded on the host. Reject duplicate or decreasing labels
    # here so exact left-hold/interpolation lookup remains unambiguous in JIT.
    day = np.asarray(times.delta.days, dtype=np.int64)
    second = np.asarray(times.delta.seconds, dtype=np.int64)
    key = day * 86400 + second
    if key.size > 1 and np.any(np.diff(key) <= 0):
        raise ValueError("TimeSeries times must be strictly increasing")
    mode_value = int(np.asarray(align_mode))
    host_dates = np.asarray(times.to_datetime64()).astype("datetime64[s]")
    months = host_dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    # A WRAP_YEAR table is selected by calendar position (see
    # ``_select_time_series``), so its labels must be in calendar order.
    is_monthly = mode_value == WRAP_YEAR and host_dates.size == 12
    is_daily = mode_value == WRAP_YEAR and host_dates.size in (365, 366)
    if is_monthly and not np.array_equal(
            months, np.arange(1, 13)):
        raise ValueError(
            "monthly climatology must contain January through December in order"
        )
    if is_daily:
        month_starts = host_dates.astype("datetime64[M]")
        days = (host_dates.astype("datetime64[D]") - month_starts).astype(int) + 1
        nominal_keys = months * 32 + days
        if (host_dates.size not in (365, 366)
                or months[0] != 1 or days[0] != 1
                or months[-1] != 12 or days[-1] != 31
                or np.any(np.diff(nominal_keys) <= 0)):
            raise ValueError(
                "daily climatology must contain ordered Jan 1 through Dec 31 "
                "nominal dates"
            )
    return TimeSeries(
        values=values,
        times=times,
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
# Canonical mirror-bundle composition (ForcingData.from_bundles)
# ---------------------------------------------------------------------------

# Surface epoch -> its mirror surface-bundle product.
_SURFACE_PRODUCTS = {
    "pd": "forcing_pd", "pi": "forcing_pi",
    "amip": "forcing_amip", "era5": "forcing_era5",
}

# Era consistency (F1): the surface epoch selects the ancillary epoch, so a PI
# surface is never paired with present-day ozone/emissions/oxidants (and vice
# versa). Only ozone/emissions/oxidants carry a PI variant; dms/dust are
# epoch-free (one product each, always "auto"). amip/era5 pair with the
# present-day *climatology* ancillaries — a transient ozone_amip/emissions_amip
# pairing is a documented deferral (the mirror stages those products, but wiring
# them per-year is future work), recorded here so every surface's choice is
# explicit rather than implicit in "auto".
_SURFACE_ANCILLARY_EPOCH = {
    None: "pd", "pd": "pd", "pi": "pi", "amip": "pd", "era5": "pd",
}

# The ancillary keys whose product carries an epoch (dms/dust do not).
_EPOCH_ANCILLARY_KEYS = ("ozone_file", "emissions_file", "oxidants_file")


# ---------------------------------------------------------------------------
# ForcingData
# ---------------------------------------------------------------------------


def land_snow_cover(snow_cover, glacier_fraction=None):
    """Snow-covered share of the land from ``snowc`` and ``glac``.

    ``snow_cover`` is the snow-covered fraction of the non-glacier land
    (clipped to [0, 1]); glaciers are fully snow covered, so the total is
    ``g + (1 - g)·s``. ``glacier_fraction=None`` (a product without the
    glacier map) reads as no glacier. Broadcasting-native.
    """
    s = jnp.clip(snow_cover, 0.0, 1.0)
    if glacier_fraction is None:
        return s
    g = jnp.clip(glacier_fraction, 0.0, 1.0)
    return g + (1.0 - g) * s


def land_wetness(soil_wetness, glacier_fraction=None, snow_cover=0.0):
    """Evaporative wetness of the whole land tile.

    ``soil_wetness`` (``soilw_am``) and ``snow_cover`` (``snowc``) describe
    the non-glacier land. The glacier share and, when passed, the
    snow-covered share evaporate at the potential rate, the rest at the soil
    wetness: ``g + (1 - g)·(s + (1 - s)·w)``. The ECHAM vertical diffusion
    passes the snow cover (JSBACH's ``qsat_fact``); SPEEDY, whose bulk
    formula has no snow term in the soil availability, passes only the
    glacier. With neither it is the soil wetness itself. Broadcasting-native.
    """
    s = jnp.clip(snow_cover, 0.0, 1.0)
    wet = s + (1.0 - s) * soil_wetness
    if glacier_fraction is None:
        return wet
    g = jnp.clip(glacier_fraction, 0.0, 1.0)
    return g + (1.0 - g) * wet



@tree_math.struct
class ForcingData:
    alb0: jnp.ndarray # bare-land annual mean albedo (ix,il)

    sice_am: jnp.ndarray # sea ice concentration (or TimeSeries thereof)
    snowc_am: jnp.ndarray # snow cover SWE/sd2sc; min(1, .) is the cover fraction (SPEEDY snowcl_ob; ECHAM land albedo, #672)
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
    # historical / scenario forcing. Prescribed here so RRTMGP does not fall
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

    # Relative soil wetness in [0, 1] — ECHAM's ``ws/wsmx`` semantics: soil
    # water as a fraction of the soil's own field capacity, so 1 means a
    # saturated soil on any texture. Built from ERA5's 0-7 cm volumetric
    # content divided by the HTESSEL field capacity of that cell's soil type
    # (``jcm.data.mirror.bundles.translate_land``), which is the layer and the
    # normalisation the saltation threshold is defined against.
    #
    # Deliberately ``None`` by default rather than a zeros field: zero means
    # "bone dry, never saturated", which is exactly the answer a *missing*
    # field would otherwise fake, and :class:`DustEmissions` must be able to
    # tell the two apart so it can warn. Forcing files written before #787 do
    # not carry it and leave it ``None``.
    #
    # Distinct from ``soilw_am``, which is SPEEDY's vegetation-weighted
    # root-zone *availability index* (capped at field capacity by
    # construction, blending the 7-28 cm layer) and stays the field SPEEDY's
    # land evaporation reads.
    soilw_rel: Any = None

    # Static land-cover fractions of the land part of a cell, read by the
    # ECHAM land albedo (JSBACH ``update_land_surface_fast``, #672):
    # ``forest_fraction`` masks the snow albedo under a canopy and
    # ``glacier_fraction`` switches to the glacier albedo. Built into the
    # mirror bundles from ERA5 high-vegetation cover ``cvh`` and the
    # permanent-snow ice-sheet mask (``jcm.data.mirror.bundles``). ``None``
    # (bundles built before #672) means "no forest, no glacier", which is
    # what ECHAM computes from zero maps — the albedo then falls back to the
    # snow-free background ``alb0`` plus open snow.
    #
    # Land-surface convention shared by every product, regrid and consumer
    # (``jcm.data.regridding.CONDITIONAL_FIELDS``): the terrain ``lsm`` is
    # the land share of the cell; ``glac`` (``glacier_fraction``) is the
    # glacier share of the LAND; ``forest`` (``forest_fraction``), ``snowc``
    # (``snowc_am``) and ``alb`` (``alb0``) describe the NON-glacier land
    # (JSBACH's tiling: a glacier tile, and vegetation / seasonal snow /
    # background albedo on the others), as do ``soilw_am`` / ``soilw_rel``
    # (the glacier counts as fully wet); ``stl`` is conditional on the land. Consumers combine them once: total snow cover
    # of the land ``g + (1 - g)·min(1, s)`` (:func:`land_snow_cover`),
    # effective forest ``(1 - g)·f``, the background albedo on the
    # non-glacier tile only (``jcm.physics.surface.echam.albedo``; SPEEDY's
    # ``alb0 + S·(albsn - alb0)`` with the whole-land snow cover S is the
    # same tile average, a glacier being fully snow covered), and the whole
    # land's wetness :func:`land_wetness`.
    forest_fraction: Any = None
    glacier_fraction: Any = None

    # Prescribed natural-aerosol emission surface fields (or TimeSeries
    # thereof), read from the forcing file when present and ``None`` otherwise
    # — the JAM emission terms fall back to zero on a ``None`` field, so DMS /
    # dust emission is simply inert until the field is supplied.
    dms_seawater: Any = None   # seawater DMS concentration kg/m³ (DmsEmissions)
    # The five Tegen/HAMMOZ dust inputs (#802). ``dust_source`` is the monthly
    # effective-LAI erodible fraction (the gate AND a linear factor on the flux);
    # ``dust_preferential`` the paleolake area fraction that swaps in soil type
    # 10; ``dust_soil_types`` a mapping of the nine prescribed texture area
    # fractions; ``dust_regions`` the integer 1–8 tuning index; and
    # ``dust_roughness`` the monthly satellite roughness [cm], read only on the
    # ``ndurough = 0`` sensitivity path. All but the first are static.
    dust_source: Any = None
    dust_preferential: Any = None
    dust_soil_types: Any = None
    dust_regions: Any = None
    dust_roughness: Any = None

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

    # Externally prescribed turbulent surface fluxes for forced mode
    # (jax-gcm#301): a coupler (or the CLI's
    # ``forcing.prescribed_surface_flux`` block) supplies the fluxes and the
    # forced surface path delivers them INSTEAD of the package's own bulk /
    # implicit surface exchange. Each is a 2-D ``(ix, il)`` map or a
    # ``TimeSeries`` thereof (``select(date)`` slices them like any other
    # leaf). Units and signs follow the surface-exchange coupling contract
    # (``docs/source/design/surface_exchange.md`` — the SAME convention the
    # published ``SurfaceExchange`` struct uses, so a coupler can feed back
    # exactly what it read): sensible heat [W/m²] and evaporation [kg/m²/s]
    # positive UP (surface → atmosphere); stress [N/m²] positive DOWN (the
    # eastward/northward momentum flux INTO the surface). ``None`` (the
    # default) means "not prescribed" — the forced-mode terms raise a
    # pointed error rather than silently applying zero fluxes.
    prescribed_sensible_heat_flux: Any = None
    prescribed_evaporation: Any = None
    prescribed_stress_u: Any = None
    prescribed_stress_v: Any = None
    # The ``(n, 2)`` per-sample ``[start, end]`` intervals (an exact
    # ``jax_datetime.Datetime``) a date-aligned prescribed-flux archive
    # declares it covers, from its CF ``time_bnds``. Kept per interval, not collapsed to
    # an envelope, so a gap the file declares stays a gap. ``None``: no
    # bounds, so coverage comes from the end samples' cadence. Read only by
    # the run-start coverage check (:func:`by_date_coverage_error`).
    prescribed_flux_time_bounds: Any = None

    @classmethod
    def zeros(cls,nodal_shape,
              alb0=None,sice_am=None,snowc_am=None,
              soilw_am=None,stl_am=None,sea_surface_temperature=None,
              soilw_rel=None,
              forest_fraction=None,
              glacier_fraction=None,
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
            # No zeros default: absence has to stay distinguishable from a dry
            # soil (see the field's declaration).
            soilw_rel=soilw_rel,
            forest_fraction=forest_fraction,
            glacier_fraction=glacier_fraction,
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
             soilw_rel=None,
             forest_fraction=None,
             glacier_fraction=None,
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
            # Left absent unless asked for: an all-ones relative wetness is a
            # globally saturated soil, which would silently switch dust
            # emission off in every test built from ``ones``.
            soilw_rel=soilw_rel,
            forest_fraction=forest_fraction,
            glacier_fraction=glacier_fraction,
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
        chronological order. ``align_mode="auto"`` resolves only when
        ``filename`` is a data-mirror or packaged product (its manifest
        ``alignment``); for any other file it raises and the mode must be
        given explicitly — see :func:`resolve_align` (#884).
        The ``validate`` flag forwards to `from_dataset` (default ``True``;
        pass ``False`` to bypass the BC sanity check, e.g. for synthetic
        test fixtures).
        """
        import xarray as xr
        align_mode = resolve_align(align_mode, paths=filename,
                                   config_key="forcing.align")
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
    def from_bundles(cls, coords, *, aerosol=None, surface="pd", years=None,
                     fetch=None):
        """Build the canonical mirror-bundle forcing set for a composition.

        The Python counterpart of the CLI's ``forcing=…`` + ``auto`` defaults:
        it composes the surface bundle (``surface`` ∈
        ``"pd"``/``"pi"``/``"amip"``/``"era5"``/``None``), ozone, and — for
        ``aerosol="jam"`` — the emission/dms/dust/oxidant set, then routes the
        composed config through the SAME engine
        (:func:`jcm.forcing_assembly.build_forcing`, which the CLI door
        ``jcm.runners.build_forcing`` also delegates to), so the CLI and Python
        doors provably agree (#751; see the
        equivalence test in ``forcing_test``). The emission-family config-trap
        warnings fire here from the shared home too. ``aerosol="macv2sp"`` wires
        the repo-packaged MACv2-SP file (:func:`packaged_macv2_path`) into
        ``forcing.macv2_file`` so the real plume weights are attached — the file
        is resolution-invariant and shipped in the wheel, so it needs no mirror
        fetch. Unpublished-grid / sigma degradations (``auto`` → nothing) mirror
        the CLI for the other products; ``auto`` ozone likewise mirrors the CLI
        in RAISING on a hybrid grid it cannot resolve (#774). ``fetch``
        (default: the HF cache) pre-resolves the composed surface bundle via the
        engine; the ``auto`` products use the cache.

        Era consistency (F1): the surface epoch selects the ancillary epoch, so
        an 1870s PI surface is not silently paired with present-day ancillaries.
        The pairing is explicit for every surface (dms/dust are epoch-free):

        =========  ====================================================
        surface    ozone / emissions / oxidants
        =========  ====================================================
        ``pd``     present-day (``*_pd``, via ``auto``)
        ``pi``     pre-industrial (``ozone_pi``/``emissions_pi``/``oxidants_pi``)
        ``amip``   present-day climatology (transient pairing deferred)
        ``era5``   present-day climatology (transient pairing deferred)
        ``None``   present-day (aquaplanet surface, ``*_pd`` ancillaries)
        =========  ====================================================
        """
        from omegaconf import OmegaConf
        from dinosaur.hybrid_coordinates import HybridCoordinates

        from jcm import forcing_assembly as fa
        from jcm import runners
        from jcm.data import input_resolution as ir
        from jcm.data import mirror_manifest as mm

        manifest = mm.load_manifest()
        grid_token = fa._grid_token(coords)
        nlev = int(coords.nodal_shape[0])
        vertical = ("hybrid" if isinstance(coords.vertical, HybridCoordinates)
                    else "sigma")

        if aerosol not in (None, "jam", "macv2sp"):
            raise ValueError(
                f"aerosol={aerosol!r}; expected None, 'jam' or 'macv2sp'.")
        if surface is not None and surface not in _SURFACE_PRODUCTS:
            raise ValueError(
                f"surface={surface!r}; expected 'pd'/'pi'/'amip'/'era5'/None.")

        macv2_file = None
        if aerosol == "macv2sp":
            # The MACv2-SP file is repo-packaged (resolution-invariant single
            # file), so wire its packaged path straight into forcing.macv2_file;
            # build_forcing then attaches the real weights instead of the all-ones
            # default (F2). No mirror fetch or staging gate.
            macv2_file = packaged_macv2_path()

        forcing_dict = {
            "ozone_file": "auto", "emissions_file": "auto", "dms_file": "auto",
            "dust_file": "auto", "dust_preferential_file": "auto",
            "dust_soil_types_file": "auto", "dust_regions_file": "auto",
            "dust_roughness_file": "auto",
            "oxidants_file": "auto", "align": "auto",
            "macv2_file": macv2_file,
            "years": years, "available_years": None,
            "ozone_available_years": None, "emissions_available_years": None,
            "oxidants_available_years": None,
        }

        # Pin the era-consistent ancillary epoch (F1). "pd" is the manifest
        # auto=True product, so it stays "auto" (keeping the silent-degrade on an
        # unpublished grid + the JAM-gating that "auto" gives). A non-pd epoch
        # pins the explicit ``*_<epoch>`` bundle, except on a grid where that
        # product is unpublished — there it falls back to "auto" so it degrades
        # to None exactly as the pd product would, not a 404. Emissions/oxidants
        # are JAM-only (a non-JAM package consumes neither), so their epoch is
        # pinned only for aerosol="jam"; ozone feeds radiation on every config,
        # so its epoch is always pinned.
        epoch = _SURFACE_ANCILLARY_EPOCH[surface]
        if epoch != "pd":
            keys = (_EPOCH_ANCILLARY_KEYS if aerosol == "jam"
                    else ("ozone_file",))
            for key in keys:
                product = f"{key[:-len('_file')]}_{epoch}"
                if mm.is_published(manifest, product, grid_token, nlev,
                                   vertical):
                    forcing_dict[key] = "hf://" + mm.bundle_path(
                        manifest, product, grid_token, nlev)

        if surface is None:
            forcing_dict["kind"] = "default"
        else:
            product = _SURFACE_PRODUCTS[surface]
            forcing_dict["kind"] = "from_file"
            if mm.product(manifest, product)["alignment"] == "transient":
                if years is None:
                    raise ValueError(
                        f"surface={surface!r} is a transient (per-year) bundle "
                        "— pass years=[first, last].")
                forcing_dict["align"] = "by_date_interp"
                forcing_dict["available_years"] = mm.coverage(manifest, product)
            file_spec = "hf://" + mm.bundle_path(manifest, product,
                                                 grid_token, nlev)
            if fetch is not None:
                sr = ir.resolve_input(
                    "file", file_spec, grid_token=grid_token, nlev=nlev,
                    vertical=vertical, years=years,
                    available=forcing_dict["available_years"],
                    manifest=manifest, fetch=fetch)
                file_spec = (list(sr.paths) if len(sr.paths) > 1
                             else sr.paths[0])
                # The fetched path may not name the product any more, so carry
                # the alignment ``resolve_input`` decided on the ORIGINAL spec
                # (#884): a path substitution never changes the alignment.
                if (forcing_dict["align"] == "auto"
                        and sr.alignment in (ir.WRAP_YEAR, ir.BY_DATE)):
                    forcing_dict["align"] = sr.alignment
            forcing_dict["file"] = file_spec

        physics_dict = {"aerosol_module": "jam"} if aerosol == "jam" else {}
        cfg = OmegaConf.create(
            {"forcing": forcing_dict, "physics": physics_dict})
        # Drive the forcing-side engine directly — the SAME engine the CLI door
        # (``runners.build_forcing``) delegates to — so the two doors provably
        # agree without this module depending on the runner's build (#751).
        forcing = fa.build_forcing(cfg, coords)
        # Same emission-family traps the CLI door fires (from the shared home).
        runners.warn_emission_config_traps(
            has_jam=(aerosol == "jam"), is_pyses=False, is_scm=False,
            forcing_cfg=cfg.forcing, coords=coords, forcing=forcing)
        return forcing

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
            align_mode: `"wrap_year"`, `"by_date"` or `"by_date_interp"`.
                `wrap_year` replays a climatology every model year by real
                calendar position (a twelve-record file is expanded to daily
                records with periodic Dec/Jan interpolation, then selected by
                nominal month/day); `by_date` aligns by absolute model date
                (piecewise constant); `by_date_interp` additionally
                interpolates linearly between samples — required for AMIP
                mid-month boundary values (``tosbcs``) to reconstruct
                monthly means. An in-memory dataset is not a mirror product,
                so the default `"auto"` raises (:func:`resolve_align`,
                #884); :meth:`from_file` resolves `auto` for mirror/packaged
                files.

        """
        expected_structure = {
            "stl":      ("lon", "lat", "time"),
            "icec":     ("lon", "lat", "time"),
            "sst":      ("lon", "lat", "time"),
            "alb":      ("lon", "lat"),
            "soilw_am": ("lon", "lat", "time"),
            "snowc":    ("lon", "lat", "time"),
        }
        # Optional (#787): only bundles built with the relative-wetness
        # derivation carry it, but when it IS there its axis order is checked
        # like any other field — a (time, lon, lat) file would otherwise be
        # silently transposed by the TimeSeries wrapper.
        if "soilw_rel" in ds.data_vars:
            expected_structure["soilw_rel"] = ("lon", "lat", "time")
        # Optional static land-cover maps for the ECHAM land albedo (#672).
        for name in ("forest", "glac", "lsm"):
            if name in ds.data_vars:
                expected_structure[name] = ("lon", "lat")

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
        resolved_align_mode = align_mode_code(resolve_align(
            align_mode, config_key="forcing.align"))
        is_wrap_year_monthly = (resolved_align_mode == WRAP_YEAR
                                and _is_monthly_climatology(ds))
        if is_wrap_year_monthly:
            # Surface climatologies are continuous boundary conditions. Expand
            # them with periodic Dec/Jan interpolation regardless of whether
            # horizontal regridding is also needed. The interpolation works on
            # a Gregorian month-start axis, so a climatology whose axis is
            # not datetime64 — a numeric month index, or cftime labels in an
            # idealised calendar (360_day, noleap year 0) that pandas cannot
            # hold — is first put on the nominal month labels WRAP_YEAR uses
            # anyway (``_wrap_year_times``). One interpolation path then
            # serves every axis kind; a datetime64 axis is interpolated on its
            # own year exactly as before (a leap-year source keeps Feb 29).
            if not np.issubdtype(np.asarray(ds["time"].values).dtype,
                                 np.datetime64):
                nominal = np.asarray(
                    _wrap_year_times(ds).to_datetime64()).astype("datetime64[ns]")
                ds = ds.assign_coords(time=("time", nominal))
            ds = interpolate_to_daily(ds)

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
            pass
        else:
            ds = upsample_forcings_ds(ds, grid=coords.horizontal)

        # Build the shared time axis (exact Gregorian dates for a
        # date-aligned file; nominal calendar labels for a climatology) for
        # every time-varying variable in this file.
        times = _times_for_mode(ds, resolved_align_mode)

        def _ts(values):
            """Wrap an `(lon, lat, time)` array as a `TimeSeries` leaf with
            time as the leading axis (matching `_select_time_series`'s
            convention).
            """
            arr = jnp.asarray(values)
            arr = jnp.moveaxis(arr, -1, 0)  # (time, lon, lat)
            return make_time_series(arr, times, align_mode=resolved_align_mode)

        # annual-mean surface albedo (no time axis)
        alb0 = jnp.asarray(ds["alb"])

        # Sea-ice concentration. Clip to [0, 1] — spectral interpolation
        # of a near-zero field can leave float-precision negatives (~1e-18),
        # which downstream scheme guards (e.g. ``sqrt(1 - sice)``) treat
        # as NaNs.
        sice_am = _ts(jnp.clip(jnp.asarray(ds["icec"]), 0.0, 1.0))

        # Snow cover ``SWE/sd2sc`` (consumers take ``min(1, .)`` as the cover
        # fraction); implausible values are zeroed.
        snowc_raw = jnp.asarray(ds["snowc"])
        snowc_valid = (0.0 <= snowc_raw) & (snowc_raw <= 20000.0)
        snowc_clean = jnp.where(snowc_valid, snowc_raw, 0.0)
        snowc_am = _ts(snowc_clean)

        # soil moisture
        soilw_am = _ts(ds["soilw_am"])

        # Relative soil wetness (ws/wsmx), optional: bundles built before #787
        # carry only ``soilw_am``. Absent stays ``None`` so the consumer can
        # warn rather than read a fabricated dry soil.
        soilw_rel = (_ts(ds["soilw_rel"]) if "soilw_rel" in ds.data_vars
                     else None)

        # Static land-cover fractions (#672), optional like ``soilw_rel``:
        # absent on bundles built before the ECHAM land albedo read them.
        # Clipped for the same interpolation-noise reason as ``icec``.
        forest_fraction = (jnp.clip(jnp.asarray(ds["forest"]), 0.0, 1.0)
                           if "forest" in ds.data_vars else None)
        glacier_fraction = (jnp.clip(jnp.asarray(ds["glac"]), 0.0, 1.0)
                            if "glac" in ds.data_vars else None)

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
            return make_time_series(arr, times, align_mode=resolved_align_mode)

        co2_vmr = _optional_ghg("co2")
        ch4_vmr = _optional_ghg("ch4")
        n2o_vmr = _optional_ghg("n2o")

        return cls.zeros(
            nodal_shape=alb0.shape,
            alb0=alb0, sice_am=sice_am, snowc_am=snowc_am, stl_am=stl_am,
            soilw_am=soilw_am, sea_surface_temperature=sea_surface_temperature,
            soilw_rel=soilw_rel,
            forest_fraction=forest_fraction,
            glacier_fraction=glacier_fraction,
            co2_vmr=co2_vmr, ch4_vmr=ch4_vmr, n2o_vmr=n2o_vmr,
        )

    def copy(self,alb0=None,
             sice_am=None,snowc_am=None,soilw_am=None, stl_am=None,
             sea_surface_temperature=None,
             soilw_rel=None,
             forest_fraction=None,
             glacier_fraction=None,
             co2_vmr=None,
             aerosol_year_weight=None,aerosol_ann_cycle=None,
             solar=None,
             ozone_climatology=None,
             ch4_vmr=None,
             n2o_vmr=None,
             nudging_target=_UNSET,
             dms_seawater=None,
             dust_source=None,
             dust_preferential=None,
             dust_soil_types=None,
             dust_regions=None,
             dust_roughness=None,
             oxidant_vmr=None,
             anthropogenic_emissions=None,
             prescribed_aerosol_emissions=None,
             prescribed_sensible_heat_flux=None,
             prescribed_evaporation=None,
             prescribed_stress_u=None,
             prescribed_stress_v=None,
             prescribed_flux_time_bounds=None):
        # ``nudging_target`` uses an ``_UNSET`` sentinel because ``None`` is
        # the natural value for "no nudging target wired" — falling back to
        # ``self.nudging_target`` only when the caller didn't supply the
        # kwarg lets ``.copy(nudging_target=None)`` *clear* the field.
        return ForcingData(
            alb0=alb0 if alb0 is not None else self.alb0,
            sice_am=sice_am if sice_am is not None else self.sice_am,
            snowc_am=snowc_am if snowc_am is not None else self.snowc_am,
            soilw_am = soilw_am if soilw_am is not None else self.soilw_am,
            soilw_rel=soilw_rel if soilw_rel is not None else self.soilw_rel,
            forest_fraction=(forest_fraction if forest_fraction is not None
                             else self.forest_fraction),
            glacier_fraction=(glacier_fraction if glacier_fraction is not None
                              else self.glacier_fraction),
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
            dust_preferential=(dust_preferential if dust_preferential is not None
                               else self.dust_preferential),
            dust_soil_types=(dust_soil_types if dust_soil_types is not None
                             else self.dust_soil_types),
            dust_regions=(dust_regions if dust_regions is not None
                          else self.dust_regions),
            dust_roughness=(dust_roughness if dust_roughness is not None
                            else self.dust_roughness),
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
            prescribed_sensible_heat_flux=(
                prescribed_sensible_heat_flux
                if prescribed_sensible_heat_flux is not None
                else self.prescribed_sensible_heat_flux
            ),
            prescribed_evaporation=(
                prescribed_evaporation
                if prescribed_evaporation is not None
                else self.prescribed_evaporation
            ),
            prescribed_stress_u=(
                prescribed_stress_u
                if prescribed_stress_u is not None
                else self.prescribed_stress_u
            ),
            prescribed_stress_v=(
                prescribed_stress_v
                if prescribed_stress_v is not None
                else self.prescribed_stress_v
            ),
            prescribed_flux_time_bounds=(
                prescribed_flux_time_bounds
                if prescribed_flux_time_bounds is not None
                else self.prescribed_flux_time_bounds
            ),
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(jnp.any, self))

    def select(self, date: DateData) -> "ForcingData":
        """Collapse every `TimeSeries` leaf to the current step's slice and
        populate `solar` from `date`.

        Static fields pass through unchanged. Returns a new `ForcingData`
        whose every leaf is the shape physics expects (no leading time axis).
        """
        sliced = _slice_time_series_leaves(self, date)
        return sliced.copy(solar=_solar_from_date(date))


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


def _time_axis_from_ds(ds) -> jdt.Datetime:
    """Convert a decoded netCDF time coordinate to exact Gregorian dates.

    Handles both numpy ``datetime64`` axes (standard/proleptic-Gregorian) and
    ``cftime`` axes from non-standard calendars — the CESM emission files use a
    ``365_day`` (noleap) calendar, which xarray decodes to ``cftime`` objects
    that pandas can't ingest.

    Both kinds are placed on the **same Gregorian clock the model runs on** by
    aligning on the *nominal* calendar date ``(year, month, day, …)``. This is
    deliberate: ``BY_DATE`` compares this axis directly with the model's
    leap-aware Gregorian ``jax_datetime`` clock. Converting a ``365_day`` axis
    with *noleap day-counting*
    (e.g. ``cftime.date2num(..., calendar='365_day')``) would instead drift
    against that target by the accumulated leap days (~7 days by 2000, growing
    every leap year), so ``searchsorted`` would pick the wrong slice and corrupt
    multi-year prescribed-emissions runs. Mapping each cftime date by its
    calendar components onto the Gregorian epoch keeps file and model on one
    clock; noleap dates are always valid Gregorian dates (no 29 Feb), so the
    mapping is exact.
    """
    import numpy as np
    vals = np.asarray(ds["time"].values)
    if vals.dtype == object:
        # cftime objects (a non-standard calendar like 365_day). Reinterpret
        # each by its (y, m, d, h, m, s) components on the Gregorian clock —
        # NOT by the file calendar's day count — so the axis matches the model's
        # leap-aware lookup target (see docstring).
        import datetime as _dt
        flat = np.ravel(vals)
        py_dates = []
        for d in flat:
            calendar = getattr(d, "calendar", "standard")
            if calendar not in ("standard", "gregorian", "proleptic_gregorian",
                                "365_day", "noleap"):
                raise ValueError(
                    f"Unsupported forcing calendar {calendar!r}; only Gregorian "
                    "and nominal noleap dates map to the model clock."
                )
            try:
                if getattr(d, "microsecond", 0):
                    raise ValueError("sub-second precision is unsupported")
                py_dates.append(_dt.datetime(
                    d.year, d.month, d.day, getattr(d, "hour", 0),
                    getattr(d, "minute", 0), getattr(d, "second", 0),
                    getattr(d, "microsecond", 0)))
            except ValueError as exc:
                raise ValueError(
                    f"Forcing date {d!r} is not a valid Gregorian date"
                ) from exc
        vals = np.asarray(py_dates, dtype="datetime64[us]")
    else:
        if not np.issubdtype(vals.dtype, np.datetime64):
            raise ValueError("forcing 'time' must decode to datetime values")
    if np.any(np.isnat(vals)):
        raise ValueError("forcing 'time' contains NaT")
    whole_seconds = vals.astype("datetime64[s]")
    if np.any((vals - whole_seconds) != np.timedelta64(0, "s")):
        raise ValueError("forcing dates must have whole-second precision")
    return jdt.Datetime.from_datetime64(vals)


def _host_epoch_seconds(times: jdt.Datetime) -> np.ndarray:
    """Exact integer seconds since 1970-01-01 of a concrete ``times`` axis.

    Host-side only (run-window and coverage checks): the day/second pair is
    combined in int64, so the result is exact to the second at any date —
    unlike float32 epoch seconds, which resolve only ~2 minutes today.
    """
    days = np.asarray(times.delta.days, dtype=np.int64)
    seconds = np.asarray(times.delta.seconds, dtype=np.int64)
    return days * 86400 + seconds


#: Reference years for a climatology's nominal labels: a non-leap year for
#: twelve monthly or 365 daily records, a leap year for a 366-record table.
_CLIMATOLOGY_YEAR = 2001
_CLIMATOLOGY_LEAP_YEAR = 2000


def _wrap_year_times(ds) -> jdt.Datetime:
    """Nominal labels for a ``WRAP_YEAR`` (climatology) leaf.

    A climatology is replayed every model year, so its records need only
    their position in the year, never an absolute date: WRAP_YEAR selects a
    twelve-record table by the model clock's calendar month, a 365/366-record
    table by nominal month/day, and any other length by equal fractions of
    the year (:func:`_select_time_series`). The labels are therefore built on
    a reference year from the decoded month (and day) — read through
    xarray's ``.dt`` accessor, which handles ``cftime`` calendars — and the
    source year is never converted to a Gregorian date. That keeps a
    climatology stamped with idealised calendar dates (CF ``noleap`` year 0,
    a ``360_day`` calendar) loadable. A numeric (index) axis is read as
    ordinal positions: months for twelve records, consecutive days for
    365/366. Other lengths get evenly spaced informational labels.
    :func:`make_time_series` then validates the month/day order.
    """
    raw = np.asarray(ds["time"].values).reshape(-1)
    n = int(raw.size)
    year = _CLIMATOLOGY_LEAP_YEAR if n == 366 else _CLIMATOLOGY_YEAR
    start = np.datetime64(f"{year:04d}-01-01", "s")
    numeric = np.issubdtype(raw.dtype, np.number)
    if not numeric and not _is_datetime_axis(raw):
        raise ValueError(
            f"climatology 'time' coordinate (dtype {raw.dtype}) is neither "
            "decoded dates nor a numeric index axis.")
    if n == 12:
        months = (np.arange(1, 13) if numeric
                  else np.asarray(ds["time"].dt.month, dtype=np.int64))
        return jdt.Datetime.from_datetime64(np.asarray(
            [f"{year:04d}-{int(m):02d}-01" for m in months],
            dtype="datetime64[s]"))
    if n in (365, 366):
        if numeric:
            return jdt.Datetime.from_datetime64(
                start + np.arange(n) * np.timedelta64(1, "D"))
        months = np.asarray(ds["time"].dt.month, dtype=np.int64)
        days = np.asarray(ds["time"].dt.day, dtype=np.int64)
        try:
            labels = np.asarray(
                [f"{year:04d}-{int(m):02d}-{int(d):02d}"
                 for m, d in zip(months, days)], dtype="datetime64[s]")
        except ValueError as exc:
            raise ValueError(
                f"daily climatology with {n} records has a nominal date that "
                f"does not exist in a {n}-day Gregorian year") from exc
        return jdt.Datetime.from_datetime64(labels)
    step = np.timedelta64(365 * 86400 // max(n, 1), "s")
    return jdt.Datetime.from_datetime64(start + np.arange(n) * step)


def _times_for_mode(ds, mode: int) -> jdt.Datetime:
    """Return the exact ``times`` axis a leaf of alignment ``mode`` needs.

    Only date-aligned modes decode the file's times onto the Gregorian model
    clock (:func:`_time_axis_from_ds`); ``WRAP_YEAR`` gets the nominal
    labels of :func:`_wrap_year_times`.
    """
    if mode == WRAP_YEAR:
        return _wrap_year_times(ds)
    return _time_axis_from_ds(ds)


#: The explicit time-alignment modes a config/API may name, and their codes.
ALIGN_MODES = {"wrap_year": WRAP_YEAR, "by_date": BY_DATE,
               "by_date_interp": BY_DATE_INTERP}


def resolve_align(align_mode, *, paths=None, config_key: str = "forcing.align",
                  transient: str = "by_date") -> str:
    """Resolve a time-alignment spec to an explicit mode name — THE one rule.

    Every time-resolved forcing input (surface boundary file, ozone, emissions,
    oxidants, prescribed surface fluxes, on both backends) resolves its
    ``align`` spec here, so all paths share one rule (#884):

    - ``wrap_year`` / ``by_date`` / ``by_date_interp`` are returned as given.
    - ``auto`` resolves ONLY from the data mirror manifest: when ``paths`` are a
      mirror or packaged product (:func:`jcm.data.input_resolution.
      manifest_alignment_for_paths`), its recorded ``alignment`` decides —
      ``climatology`` → ``wrap_year``, ``transient`` → ``transient`` (default
      ``by_date``; a loader whose transient products are mid-month means passes
      ``by_date_interp``).
    - ``auto`` for anything else — a user file, an in-memory dataset — RAISES.
      jcm does not guess whether a file is a climatology: a one-year transient
      archive and a monthly climatology have identical time axes, and replaying
      the former every year is a silent error.

    Args:
        align_mode: ``auto`` | ``wrap_year`` | ``by_date`` | ``by_date_interp``.
        paths: The file(s) being read (path, ``hf://`` URL or list), or
            ``None`` when there is no file (an in-memory dataset).
        config_key: The knob the user sets to declare the mode, for the error.
        transient: The mode a manifest ``transient`` product resolves to.

    Returns:
        The explicit mode name (a key of :data:`ALIGN_MODES`).

    """
    align_mode = str(align_mode)
    if align_mode in ALIGN_MODES:
        return align_mode
    if align_mode != "auto":
        raise ValueError(
            f"{config_key}={align_mode!r}: unknown time alignment; expected "
            f"'auto' or one of {tuple(ALIGN_MODES)}.")
    kind = None
    if paths is not None:
        from jcm.data import input_resolution as ir
        kind = ir.manifest_alignment_for_paths(paths)
    mode = manifest_mode_for_kind(kind, transient)
    if mode is not None:
        return mode
    what = ("an in-memory dataset" if paths is None
            else f"{paths!r} (not a data-mirror or packaged product)")
    raise ValueError(
        f"{config_key}=auto cannot resolve the time alignment of {what}: set "
        f"{config_key} to wrap_year (a climatology replayed every model year), "
        "by_date (aligned on its absolute timestamps) or by_date_interp "
        "(interpolated between them) — jcm does not guess whether a file is a "
        "climatology (#884). 'auto' resolves only data-mirror / packaged "
        "products, whose kind the manifest records.")


def manifest_mode_for_kind(kind, transient: str = "by_date"):
    """Map a manifest ``alignment`` kind to an explicit mode (``None`` if none).

    ``climatology`` → ``wrap_year``, ``transient`` → ``transient``; ``static``
    or unknown → ``None``. The single mapping :func:`resolve_align` and the
    spec-time declarations (:func:`declare_manifest_align`) share.
    """
    if kind == "climatology":
        return "wrap_year"
    if kind == "transient":
        return transient
    return None


def declare_manifest_align(align_mode, spec, transient: str = "by_date"):
    """Turn ``auto`` into the explicit mode of a manifest product ``spec``.

    Call this on the ORIGINAL input spec (an ``hf://`` URL / ``{year}``
    pattern / packaged path), BEFORE any fetch or cache callback substitutes a
    local path: a fetched file may live anywhere, so its path can no longer
    name the product, and a path substitution must never change an alignment
    decision (#884). Returns ``align_mode`` unchanged when it is already
    explicit (or a per-product list) or when ``spec`` is not a manifest
    product — the reader then applies :func:`resolve_align` as usual (and
    raises for a timed user file under ``auto``). Never raises.
    """
    if not isinstance(align_mode, str) or align_mode != "auto" or spec is None:
        return align_mode
    from jcm.data import input_resolution as ir
    mode = manifest_mode_for_kind(ir.manifest_alignment_for_paths(spec),
                                  transient)
    return mode if mode is not None else align_mode


def _times_and_mode(ds, align_mode, config_key):
    """Return ``(times, mode_code)`` for a single-product reader."""
    mode = align_mode_code(resolve_align(align_mode, config_key=config_key))
    return _times_for_mode(ds, mode), mode


def align_mode_code(align_mode: str) -> int:
    """Return the ``TimeSeries.align_mode`` code of an explicit mode name."""
    try:
        return ALIGN_MODES[str(align_mode)]
    except KeyError:
        raise ValueError(
            f"align_mode={align_mode!r} is not an explicit mode; expected one "
            f"of {tuple(ALIGN_MODES)} (resolve 'auto' with resolve_align "
            "first).") from None


def _slice_time_series_leaves(forcing: ForcingData, date: DateData) -> ForcingData:
    """Return `forcing` with every `TimeSeries` leaf replaced by its slice
    at `date`. Non-`TimeSeries` leaves are passed through unchanged.
    """
    def slice_leaf(leaf):
        if isinstance(leaf, TimeSeries):
            return _select_time_series(leaf, date)
        return leaf

    return tree_util.tree_map(
        slice_leaf,
        forcing,
        is_leaf=lambda x: isinstance(x, TimeSeries),
    )


def _select_time_series(ts: TimeSeries, date: DateData) -> jnp.ndarray:
    """Index `ts.values` along the leading time axis at `date`."""
    n_time = ts.values.shape[0]
    if n_time == 0:
        # Defensive — shouldn't happen, but a 0-length axis would NaN downstream
        return ts.values

    # Every branch has to produce the same shape, which they do (scalar idx).
    # WRAP_YEAR (a climatology replayed every model year) selects by real
    # calendar position; the table length is static, so the rule is chosen
    # at trace time: twelve records are January..December held from the 1st
    # of each Gregorian month (#805); 365/366 records are a nominal-date
    # daily table (a 365-record table holds Feb 28 through Feb 29); any
    # other length (e.g. MACv2-SP's weekly annual cycle) keeps equal
    # fractions of the actual year. ``make_time_series`` validated the
    # month/day order of the first two layouts.
    if n_time == 12:
        idx_wrap = _monthly_climatology_index(date)
    elif n_time in (365, 366):
        idx_wrap = _daily_climatology_index(ts.times, date)
    else:
        idx_wrap = jnp.floor(date.tyear() * n_time).astype(jnp.int32) % n_time
    idx_date = _by_date_index(ts.times, date)
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
    t_lo = jdt.Datetime(jdt.Timedelta(
        days=jnp.take(ts.times.delta.days, lo),
        seconds=jnp.take(ts.times.delta.seconds, lo)))
    t_hi = jdt.Datetime(jdt.Timedelta(
        days=jnp.take(ts.times.delta.days, lo + 1),
        seconds=jnp.take(ts.times.delta.seconds, lo + 1)))
    elapsed = date.dt - t_lo
    interval = t_hi - t_lo
    # Cast before multiplying: Timedelta stores int32 days and a century-scale
    # out-of-range query would otherwise overflow before the fraction clamps.
    elapsed_seconds = elapsed.days.astype(jnp.float32) * 86400.0 + elapsed.seconds
    interval_seconds = interval.days.astype(jnp.float32) * 86400.0 + interval.seconds
    frac = jnp.clip(elapsed_seconds / jnp.maximum(interval_seconds, 1), 0.0, 1.0)
    interp = ((1.0 - frac) * jnp.take(ts.values, lo, axis=0)
              + frac * jnp.take(ts.values, lo + 1, axis=0))
    return jnp.where(ts.align_mode == BY_DATE_INTERP, interp, stepped)


def _monthly_climatology_index(date: DateData) -> jnp.ndarray:
    """Select a Gregorian calendar month (January is record zero)."""
    _, month, _ = gregorian_ymd_from_days(date.dt.delta.days)
    return (month - 1).astype(jnp.int32)


def _daily_climatology_index(times: jdt.Datetime, date: DateData) -> jnp.ndarray:
    """Select the latest nominal month/day; missing Feb 29 holds Feb 28."""
    _, months, days = gregorian_ymd_from_days(times.delta.days)
    _, month, day = gregorian_ymd_from_days(date.dt.delta.days)
    keys = months * 32 + days
    key = month * 32 + day
    return jnp.clip(jnp.searchsorted(keys, key, side="right") - 1,
                    0, times.delta.days.shape[0] - 1).astype(jnp.int32)


def _by_date_index(times: jdt.Datetime, date: DateData) -> jnp.ndarray:
    """Date-aligned: nearest exact datetime entry at-or-before `date`."""
    # `searchsorted(side='right') - 1` puts us at the entry whose timestamp
    # is the latest one <= target, which is the natural piecewise-constant
    # left interpretation of the forcing axis.
    raw = jdt.searchsorted(times, date.dt, side='right') - 1
    return jnp.clip(raw, 0, times.delta.days.shape[0] - 1).astype(jnp.int32)


def _solar_from_date(date: DateData) -> SolarGeometry:
    """Build Gregorian `SolarGeometry` from exact per-step date metadata.

    Calendar-aware fraction of year (Gregorian honours leap years; see
    `fraction_of_year_elapsed`). The orbital phase tracks the same
    fraction so the solar declination matches the actual day-of-year
    (#410). `synodic_phase` is fraction-of-day × 2π, calendar-independent.
    """
    fraction_of_day = date.dt.delta.seconds / 86400.0
    tyear = date.tyear()
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

def emissions_have_time(ds) -> bool:
    """Whether any emission variable (``emis_*`` / ``aero_emis_*``) is timed.

    A product whose emission fields are all static needs no time alignment,
    so callers resolve ``emissions_align`` only when this is true — a static
    user file loads under the default ``auto`` (#884 applies to time axes
    only).
    """
    return any("time" in ds[v].dims for v in ds.data_vars
               if str(v).startswith(("emis_", "aero_emis_")))


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
    source-grid file first). A time axis is aligned by the explicit
    ``align_mode`` (``wrap_year`` / ``by_date`` / ``by_date_interp``); the
    assembly resolves ``forcing.emissions_align`` to one per product with
    :func:`resolve_align`, and ``auto`` raises here because an in-memory dataset
    carries no manifest identity (#884). Inputs are flux rates, not interval
    totals; bounds-aware conversion of interval totals at ingestion remains
    tracked in #876.
    """
    emis_names = [str(v) for v in ds.data_vars if str(v).startswith("emis_")]
    if not emis_names:
        return None
    has_time = any("time" in ds[n].dims for n in emis_names)
    mode = (align_mode_code(resolve_align(
        align_mode, config_key="forcing.emissions_align"))
        if has_time else BY_DATE)
    times = _times_for_mode(ds, mode) if has_time else None
    out: dict[str, Any] = {}
    for name in emis_names:
        da = ds[name]
        if "time" in da.dims:
            # Lead with time so `_select_time_series` can index axis 0; keep the
            # remaining (horizontal) axes in their file order — the term ravels
            # them to (ncols,).
            others = [d for d in da.dims if d != "time"]
            arr = jnp.asarray(da.transpose("time", *others).values)
            out[name] = make_time_series(arr, times, align_mode=mode)
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
    regridding here — use :mod:`jcm.data.emissions.prepare`). Time alignment and
    the flux-rate contract as in :func:`read_anthropogenic_emissions`.
    """
    prefix = "aero_emis_"
    names = [str(v) for v in ds.data_vars if str(v).startswith(prefix)]
    if not names:
        return None
    has_time = any("time" in ds[n].dims for n in names)
    mode = (align_mode_code(resolve_align(
        align_mode, config_key="forcing.emissions_align"))
        if has_time else BY_DATE)
    times = _times_for_mode(ds, mode) if has_time else None
    out: dict[str, Any] = {}
    for name in names:
        da = ds[name]
        key = name[len(prefix):]
        if "time" in da.dims:
            others = [d for d in da.dims if d != "time"]
            arr = jnp.asarray(da.transpose("time", *others).values)
            out[key] = make_time_series(arr, times, align_mode=mode)
        else:
            out[key] = jnp.asarray(da.values)
    return out


def validate_emissions_grid(mapping, coords, path):
    """Raise if any emission field's horizontal shape != the model grid.

    Lives next to the ``read_*_emissions`` loaders so a user assembling a
    :class:`ForcingData` by hand gets the same guard the CLI does — an
    off-grid file is a size mismatch the emission terms would otherwise fall
    back to zero on, which from either entry point would look like the file
    "did nothing".
    """
    nodal = tuple(coords.horizontal.nodal_shape)
    for name, leaf in mapping.items():
        arr = leaf.values if isinstance(leaf, TimeSeries) else leaf
        spatial = tuple(arr.shape[-2:])
        if spatial != nodal:
            raise ValueError(
                f"forcing.emissions_file {path!r}: field {name!r} has "
                f"horizontal shape {spatial}, but the model grid is {nodal}. "
                "Regrid the file with jcm.data.emissions.prepare first."
            )


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
        arr, *_times_and_mode(ds, align_mode, var_name)
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
    # WRAP_YEAR steps the record by month, never interpolating, as
    # ``bgc_dust_read_monthly`` does — but it bins the year into twelve equal
    # 30.42-day slices, so records 2-11 switch 1-2 days after the calendar
    # month start (#805, shared by every monthly climatology).
    if ds.sizes["time"] != _DUST_MONTHS:
        raise ValueError(
            f"{var_name}: the HAMMOZ potential-source climatology has "
            f"{_DUST_MONTHS} monthly records, found {ds.sizes['time']}. The "
            "Tegen scheme steps it by month start (WRAP_YEAR); a different "
            "record count means a different product.")
    return make_time_series(
        arr, *_times_and_mode(ds, align_mode, var_name)
    )


def _drop_degenerate_time(arr, ds, var_name):
    """Collapse the length-1 ``time`` axis the static HAMMOZ dust files carry."""
    if "time" not in ds[var_name].dims:
        return arr
    if ds.sizes["time"] != 1:
        raise ValueError(
            f"{var_name}: expected a static field (no time axis, or a "
            f"degenerate one), found {ds.sizes['time']} records.")
    return arr[0]


def read_dust_preferential(ds, lat_deg=None, lon_deg=None, var_name="source"):
    """Read the preferential-source area fraction for ``ForcingData.dust_preferential``.

    The HAMMOZ ``dust_preferential_sources.nc`` paleolake / topographic-depression
    field, ``source (time, lat, lon)`` with a degenerate time axis
    (``mo_ham_dust.f90::bgc_read_annual_fields`` reads record 1 only). Returned as
    a static ``(lon, lat)`` array in [0, 1]: this fraction of each cell is given
    soil type 10 (100 % silt), the rest keeps its mapped texture.
    """
    if var_name not in ds.data_vars:
        raise ValueError(
            f"Preferential-source file has no {var_name!r} variable; found "
            f"{sorted(map(str, ds.data_vars))}.")
    arr = _orient_to_model_grid(ds[var_name], lat_deg, lon_deg, name=var_name)
    arr = _drop_degenerate_time(arr, ds, var_name)
    return jnp.asarray(np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0))


#: Soil-type file variables. Type 1 (coarse) is deliberately absent — the scheme
#: takes it as the residual ``1 − Σ``.
_DUST_SOIL_TYPE_VARS = ("type2", "type3", "type4", "type6",
                        "type13", "type14", "type15", "type16", "type17")
#: The globally-complete Zobler partition; the type13-17 group is a separate,
#: OVERLAPPING China-only partition and the two must never be summed together.
_DUST_GLOBAL_SOIL_VARS = ("type2", "type3", "type4", "type6")
_DUST_MONTHS = 12


def read_dust_soil_types(ds, lat_deg=None, lon_deg=None):
    """Read the nine soil-texture area fractions for ``ForcingData.dust_soil_types``.

    The HAMMOZ ``soil_type_all.nc``: ``type2/3/4/6`` (Tegen's global Zobler
    textures) and ``type13..17`` (Cheng's East-Asian textures). Returned as
    ``{var_name: (lon, lat) array}``.

    The two groups are *overlapping* partitions — their nine-way sum reaches 2.18
    over the Gobi — so only the global group is checked to be a partition here;
    reconciling the overlap is the scheme's ``k_dust_easo`` branch.
    """
    missing = [v for v in _DUST_SOIL_TYPE_VARS if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Soil-type file is missing {missing}; the Tegen scheme needs all "
            f"of {list(_DUST_SOIL_TYPE_VARS)} (type1 is the residual).")
    out = {}
    for var in _DUST_SOIL_TYPE_VARS:
        arr = _orient_to_model_grid(ds[var], lat_deg, lon_deg, name=var)
        arr = _drop_degenerate_time(arr, ds, var)
        out[var] = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)
    total = sum(out[v] for v in _DUST_GLOBAL_SOIL_VARS)
    if np.max(total) > 1.0 + 1e-5:
        raise ValueError(
            f"Soil-type file: the global textures {list(_DUST_GLOBAL_SOIL_VARS)} "
            f"sum to {np.max(total):.4f} > 1 — they must be a partition, or the "
            "type-1 residual the scheme derives from them goes negative.")
    return {k: jnp.asarray(v) for k, v in out.items()}


def read_dust_regions(ds, lat_deg=None, lon_deg=None, var_name="regions"):
    """Read the regional-tuning index for ``ForcingData.dust_regions``.

    The HAMMOZ ``dust_regions.nc`` (Huneeus et al. 2011 boxes: 1 everywhere else,
    2 N America, 3 S America, 4 N Africa, 5 S Africa, 6 Middle East, 7 Asia,
    8 Australia). Purely categorical — the Fortran reads it with ``EF_NOINTER``
    — so a non-integer or out-of-range value means the file was interpolated and
    is refused rather than silently rounded.
    """
    if var_name not in ds.data_vars:
        raise ValueError(
            f"Dust-regions file has no {var_name!r} variable; found "
            f"{sorted(map(str, ds.data_vars))}.")
    arr = _orient_to_model_grid(ds[var_name], lat_deg, lon_deg, name=var_name)
    arr = _drop_degenerate_time(arr, ds, var_name)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{var_name}: contains non-finite values.")
    if not np.allclose(arr, np.round(arr), atol=1e-6):
        raise ValueError(
            f"{var_name}: contains non-integer values (e.g. "
            f"{arr[~np.isclose(arr, np.round(arr), atol=1e-6)][0]}). The dust "
            "region mask is categorical and must be regridded "
            "nearest-neighbour, never linearly or conservatively.")
    ints = np.round(arr).astype(np.int32)
    if ints.min() < 1 or ints.max() > _DUST_N_REGIONS:
        raise ValueError(
            f"{var_name}: values span [{ints.min()}, {ints.max()}], outside the "
            f"1-{_DUST_N_REGIONS} tuning-region range.")
    return jnp.asarray(ints, dtype=jnp.int32)


_DUST_N_REGIONS = 8
#: ``surfrough`` values span 0.001-0.08 and the file's ``units`` attribute says
#: ``"1."``; the Fortran compares them against ``r_dust_z0s = 0.001 cm``, so they
#: are centimetres. Read as metres every land cell would clip ``feff`` to zero
#: and the scheme would emit nothing anywhere.
_DUST_ROUGHNESS_CM_UNITS = {"cm", "centimetre", "centimeter", "1.", "1", ""}


def read_dust_roughness(ds, lat_deg=None, lon_deg=None, var_name="surfrough",
                        align_mode: str = "wrap_year"):
    """Read the monthly surface-roughness map for ``ForcingData.dust_roughness``.

    The Prigent et al. (2005) satellite roughness length in **centimetres**
    (``surface_rough_12m.nc``), as a monthly ``WRAP_YEAR`` :class:`TimeSeries`.
    Consumed only on the ``ndurough = 0`` sensitivity path — with the default
    constant roughness ``bgc_dust_read_monthly`` overwrites it immediately.
    """
    if var_name not in ds.data_vars:
        raise ValueError(
            f"Dust-roughness file has no {var_name!r} variable; found "
            f"{sorted(map(str, ds.data_vars))}.")
    units = str(ds[var_name].attrs.get("units", "")).strip().lower()
    if units not in _DUST_ROUGHNESS_CM_UNITS:
        raise ValueError(
            f"{var_name}: units {units!r} are not the centimetres the Tegen "
            "drag partition expects (the HAMMOZ file carries '1.').")
    arr = _orient_to_model_grid(ds[var_name], lat_deg, lon_deg, name=var_name)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if "time" not in ds[var_name].dims:
        return jnp.asarray(arr)
    if ds.sizes["time"] != _DUST_MONTHS:
        raise ValueError(
            f"{var_name}: expected {_DUST_MONTHS} monthly records, found "
            f"{ds.sizes['time']}.")
    return make_time_series(
        arr, *_times_and_mode(ds, align_mode, var_name)
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
    if "time" not in sample.dims:
        raise ValueError(
            "Oxidant variables must carry a time axis (a monthly climatology "
            f"or a dated series); got dims {sample.dims}.")
    mode = align_mode_code(resolve_align(
        align_mode, config_key="forcing.oxidants_align"))
    times = _times_for_mode(ds, mode)
    out: dict[str, Any] = {}
    for var, key in _OXIDANT_VAR_MAP.items():
        arr = _orient_to_model_grid(ds[var], lat_deg, lon_deg, name=var)
        # Defensive: a fill-value cell decoded to NaN means "no data" — treat
        # as zero oxidant rather than let NaN poison the sulfur chemistry.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = make_time_series(arr, times, align_mode=mode)
    return out


def validate_oxidant_levels(ds, coords, path):
    """Cross-check the oxidant file's hybrid coefficients against the model.

    Lives next to :func:`read_oxidant_vmr` — the hyam/hybm boundary-vs-midpoint
    comparison is vertical-coordinate science a user assembling
    :class:`ForcingData` by hand needs as much as the CLI does.

    Only applies when both the file (``hyam``/``hybm``, plus ``p0`` in Pa for
    files storing normalized ``hyam``) and the model
    (:class:`dinosaur.hybrid_coordinates.HybridCoordinates`) define hybrid
    levels; a sigma-coordinate model only gets the level-count assert in
    :func:`read_oxidant_vmr` (documented assumption: the file matches the model
    levels). Full-level model coefficients are boundary midpoints, matching
    the ECHAM ``hyam/hybm`` convention.
    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    vertical = coords.vertical
    if not isinstance(vertical, HybridCoordinates):
        return
    if "hyam" not in ds or "hybm" not in ds:
        return
    hyam = np.asarray(ds["hyam"].values, dtype=float)
    hybm = np.asarray(ds["hybm"].values, dtype=float)
    # HAMMOZ files store hyam normalized by the reference pressure p0 [Pa];
    # dinosaur's a_boundaries are in Pa.
    if "p0" in ds:
        hyam = hyam * float(ds["p0"].values)
    a = np.asarray(vertical.a_boundaries, dtype=float)
    b = np.asarray(vertical.b_boundaries, dtype=float)
    a_full = 0.5 * (a[:-1] + a[1:])
    b_full = 0.5 * (b[:-1] + b[1:])
    if (not np.allclose(a_full, hyam, atol=1.0)          # Pa
            or not np.allclose(b_full, hybm, atol=1e-5)):
        raise ValueError(
            f"forcing.oxidants_file {path!r}: hybrid-level coefficients "
            "(hyam/hybm) don't match the model's vertical grid — the file "
            "must be on the model levels (no vertical interpolation is "
            "done). Use the matching L-grid file (e.g. the T63L47 MACC file "
            "with grid=echam_t63_l47_hybrid) or re-interpolate it."
        )


def packaged_macv2_path() -> str:
    """Filesystem path to the repo-packaged MACv2-SP file (``macv2_file=auto``).

    The MACv2-SP simple-plume file — SPv2.1 (CMIP7; Fiedler & Azoulay, University
    Heidelberg, 2025), the CEDS-scaled successor to Stevens et al. (2017) v1 — is
    resolution-invariant (~19 KB), so it ships in the wheel under ``jcm/data/bc``
    rather than on the HF mirror (SOURCES.md carries provenance + sha256). It is
    the ``macv2_sp`` packaged product in the mirror manifest, resolved through
    the one packaged-product mechanism (:func:`jcm.data.input_resolution.
    resolve_packaged`); ``forcing.macv2_file=auto`` and
    ``from_bundles(aerosol="macv2sp")`` both use this shim, an explicit path
    overrides.
    """
    from jcm.data import input_resolution as ir
    from jcm.data import mirror_manifest as mm
    return ir.resolve_packaged(mm.load_manifest(), "macv2_sp")


def read_macv2_weights(path) -> tuple[TimeSeries, TimeSeries]:
    """Read MACv2-SP time-varying plume weights into two ``TimeSeries`` leaves.

    The MACv2-SP file (the packaged SPv2.1 CMIP7 build, or the older v1) carries
    the static plume geometry (consumed separately by
    :meth:`AerosolParameters.from_dataset`) alongside two time-varying scaling
    arrays:

    * ``year_weight(plume, year)`` over 1850..2100 — the per-year anthropogenic
      amplitude. Returned as ``forcing.aerosol_year_weight``: a ``BY_DATE``
      ``TimeSeries`` of shape ``(year, plume)`` so the model picks the current
      calendar year. Only part of the axis carries real data (SPv2.1: 1850-2023;
      v1: 1850-2016); the trailing years are ``_FillValue`` (delivered as NaN),
      which would inject NaN AOD, so the last valid year is forward-filled (a
      documented jcm convention — the reference STOPs out of range). The
      forward-fill finds the last all-valid year dynamically, so it adapts to
      either file's real span with no version-specific constant.
    * ``ann_cycle(plume, week, feature)`` — the seasonal cycle. Returned as
      ``forcing.aerosol_ann_cycle``: a ``WRAP_YEAR`` ``TimeSeries`` arranged
      ``(week, feature, plume)`` so a ``select(date)`` slice yields the
      ``(feature, plume)`` the term consumes at the current week.

    ``select(date)`` collapses each leaf to its current-step slice, so nothing
    extra is needed at run time. Attach both to a :class:`ForcingData` via
    ``base.copy(aerosol_year_weight=..., aerosol_ann_cycle=...)``.
    """
    import xarray as xr

    ds = path if isinstance(path, xr.Dataset) else xr.open_dataset(path)
    try:
        # year_weight: (plume, year) -> (year, plume). Forward-fill past the
        # last all-valid year so out-of-range years reuse the last real
        # amplitude instead of the file's NaN fill.
        yw_np = np.asarray(ds["year_weight"].values.T, dtype=float)  # (251, 9)
        valid = ~np.isnan(yw_np).any(axis=1)
        last_valid = np.where(valid)[0].max()
        yw_np[last_valid + 1:] = yw_np[last_valid]
        yw = jnp.asarray(yw_np)

        # Exact dated time axis, one sample per year-start. The file labels year Y with the integer Y; treat it as
        # Y-01-01 00:00 UTC.
        years = ds["years"].values.astype(int)
        year_dates = np.asarray(
            [f"{int(y)}-01-01" for y in years], dtype="datetime64[s]")
        year_weight = make_time_series(
            yw, year_dates, align_mode=BY_DATE)

        # ann_cycle: (plume, week, feature) -> (week, feature, plume). WRAP_YEAR
        # repeats every year; the exact labels are informational for this
        # legacy weekly climatology.
        ac = jnp.asarray(np.transpose(ds["ann_cycle"].values, (1, 2, 0)))
        week_dates = np.datetime64("2001-01-01") + np.arange(ac.shape[0]) * np.timedelta64(7, "D")
        ann_cycle = make_time_series(
            ac, week_dates, align_mode=WRAP_YEAR)
    finally:
        if not isinstance(path, xr.Dataset):
            ds.close()
    return year_weight, ann_cycle


# ---------------------------------------------------------------------------
# Prescribed surface fluxes (forced surface mode, jax-gcm#301)
# ---------------------------------------------------------------------------

#: The four flux variables a prescribed-surface-flux source must define — names,
#: units and signs are the surface-exchange contract's
#: (docs/source/design/surface_exchange.md): sensible heat [W/m²] and
#: evaporation [kg/m²/s] positive up, stress [N/m²] positive down. Maps each
#: file variable to the ``ForcingData`` field it fills.
PRESCRIBED_FLUX_FILE_VARS = {
    "sensible_heat_flux": "prescribed_sensible_heat_flux",
    "evaporation": "prescribed_evaporation",
    "stress_u": "prescribed_stress_u",
    "stress_v": "prescribed_stress_v",
}

def _is_datetime_axis(values) -> bool:
    """Return whether ``values`` is a decoded date axis (datetime64 or cftime).

    A plain numeric axis (e.g. an undecodable ``months since`` unit, or bare
    integers) is NOT a date axis: it cannot be placed on the model clock, and
    reading the numbers as an epoch offset would align every sample to 1970.
    """
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return True
    return values.dtype == object and values.size > 0 and hasattr(
        np.ravel(values)[0], "month")


def read_prescribed_surface_fluxes(ds, lat_deg=None, lon_deg=None,
                                   align_mode: str = "auto",
                                   source: str = "prescribed_surface_flux"):
    """Read a forced-mode surface-flux dataset into ``ForcingData`` fields.

    The Python door behind ``forcing.prescribed_surface_flux.file`` (the Hydra
    door calls exactly this): returns a dict keyed by the ``ForcingData``
    field names (``prescribed_sensible_heat_flux``, ``prescribed_evaporation``,
    ``prescribed_stress_u``, ``prescribed_stress_v``), so
    ``forcing.copy(**read_prescribed_surface_fluxes(ds, lat, lon, ...))``
    attaches them. A coupler that already holds the fluxes in memory sets the
    fields directly instead (a bare ``(lon, lat)`` map per coupling interval,
    or a :class:`TimeSeries` built with :func:`make_time_series` and an
    explicit ``align_mode``).

    Each variable is ``(lat, lon)`` (static → bare ``(lon, lat)`` array) or
    ``(time, lat, lon)`` (→ a :class:`TimeSeries`). A length-1 time axis is a
    static field and is collapsed, as the other static readers do
    (:func:`_drop_degenerate_time`). Grid validation/reorientation is
    :func:`_orient_to_model_grid`'s.

    **Time alignment is declared, never inferred from the timestamps** (the
    rule every forcing input shares, :func:`resolve_align`, #884). A one-year
    transient archive (one sample per month, January to December of a specific
    year — e.g. a coupler's history file) is indistinguishable by its
    timestamps from a monthly climatology, and replaying it every year would
    silently recycle that year's fluxes. So ``align_mode`` must be explicit:

    - ``"wrap_year"`` — the file is a climatology: replayed every model year,
      the record of the model clock's calendar month held from the 1st
      (:func:`_select_time_series`).
    - ``"by_date"`` / ``"by_date_interp"`` — aligned on the absolute timestamps
      (piecewise constant / linearly interpolated).
    - ``"auto"`` raises for a time-resolved file: no data-mirror product
      carries prescribed fluxes, so there is no manifest kind to resolve it
      from. (A static file, or a length-1 time axis, needs no alignment.)

    A declared ``wrap_year`` is then VALIDATED: WRAP_YEAR selects record
    ``month - 1`` for the model clock's Gregorian month, so the (ascending)
    samples must be exactly twelve, one per calendar month
    January→December. Anything else raises rather than replaying out of
    phase.

    The time axis is sorted ascending (samples reordered to match) before
    either mode: ``_by_date_index`` uses ``searchsorted`` (requires ascending)
    and WRAP_YEAR's position 0 must be January. Duplicate, non-finite or
    non-date timestamps raise. Coverage of the run window by a BY_DATE axis is
    checked at run start (:func:`by_date_coverage_error`, called from the
    forced-mode terms' ``validate_forcing``), because only the run knows its
    window.

    Args:
        ds: An open ``xarray.Dataset``.
        lat_deg, lon_deg: The model grid (degrees) to validate against.
        align_mode: ``"wrap_year"`` | ``"by_date"`` | ``"by_date_interp"``
            (``"auto"`` raises for a time-resolved file) — the vocabulary of
            ``forcing.align`` / :func:`resolve_align`.
        source: Label (file path) for error messages.

    Returns:
        ``dict`` mapping ``ForcingData`` field name → array or TimeSeries.

    """
    align_mode = str(align_mode)
    if align_mode != "auto" and align_mode not in ALIGN_MODES:
        raise ValueError(
            f"{source}: unknown align {align_mode!r}; expected one of "
            f"{tuple(ALIGN_MODES)}.")
    missing = [v for v in PRESCRIBED_FLUX_FILE_VARS if v not in ds]
    if missing:
        raise ValueError(
            f"{source} is missing the variables {missing}; all of "
            f"{tuple(PRESCRIBED_FLUX_FILE_VARS)} are required (contract "
            "units/signs: docs/source/design/surface_exchange.md).")
    for var in PRESCRIBED_FLUX_FILE_VARS:
        spatial = [d for d in ds[var].dims if d != "time"]
        if sorted(spatial) != ["lat", "lon"]:
            raise ValueError(
                f"{source}: variable {var!r} must be dimensioned (lat, lon) "
                f"with an optional leading time axis; got {ds[var].dims}.")

    timed = [v for v in PRESCRIBED_FLUX_FILE_VARS if "time" in ds[v].dims]
    order = times = None
    mode = None
    if timed and ds.sizes["time"] > 1:
        order, times, mode = _prescribed_flux_time_axis(
            ds, align_mode, source)

    out = {}
    for var, field in PRESCRIBED_FLUX_FILE_VARS.items():
        # (*time, lon, lat), coordinate-validated and lat-oriented.
        values = _orient_to_model_grid(ds[var], lat_deg, lon_deg, name=var)
        if "time" not in ds[var].dims:
            out[field] = jnp.asarray(values)
        elif mode is None:
            # Degenerate (length-1) time axis → a static field.
            out[field] = jnp.asarray(values[0])
        else:
            out[field] = make_time_series(
                np.asarray(values)[order], times, mode)
    if mode in (BY_DATE, BY_DATE_INTERP):
        bounds = _prescribed_flux_time_bounds(ds, order, times, source)
        if bounds is not None:
            out["prescribed_flux_time_bounds"] = bounds
    return out


def _prescribed_flux_time_bounds(ds, order, times, source):
    """Per-sample ``(n, 2)`` ``[start, end]`` dates a file's CF bounds declare.

    Looks for the variable the ``time`` coordinate's ``bounds`` attribute
    names (the CF convention), else a ``time_bnds``/``time_bounds``
    variable; returns ``None`` when there is none, and the run-start coverage
    check then takes coverage from the end samples' cadence
    (:func:`by_date_coverage_error`). The bounds are validated — shape
    ``(time, 2)``, each sample inside its own interval — because a declared
    coverage that disagrees with the stamps is a malformed file, not a hint.
    The intervals are returned as declared (sorted with the samples), NOT
    collapsed to an envelope: disjoint bounds are the file saying it has no
    data between them, and the run-start check honours that gap rather than
    letting a neighbouring sample silently stand in for it. Rejecting
    disjoint bounds outright would refuse legitimate archives (a campaign
    record with a declared outage) that a run avoiding the gap can use.
    """
    import xarray as xr
    name = ds["time"].attrs.get("bounds") or ds["time"].encoding.get("bounds")
    if name is None:
        name = next((n for n in ("time_bnds", "time_bounds") if n in ds), None)
    if name is None or name not in ds:
        return None
    bnds = ds[name]
    if bnds.ndim != 2 or bnds.shape[0] != ds.sizes["time"] or bnds.shape[1] != 2:
        raise ValueError(
            f"{source}: time bounds {name!r} must be shaped (time, 2); got "
            f"{dict(bnds.sizes)}.")
    edges = []
    for k in (0, 1):
        edge = xr.Dataset(coords={"time": ("time", np.asarray(
            bnds.isel({bnds.dims[1]: k}).values)[order])})
        # Exact dates on the model clock, like the samples themselves.
        edges.append(np.asarray(
            _time_axis_from_ds(edge).to_datetime64()).astype("datetime64[s]"))
    lo, hi = edges
    t = np.asarray(times.to_datetime64()).astype("datetime64[s]")
    bad = np.flatnonzero(~((lo <= t) & (t <= hi) & (lo < hi)))
    if bad.size:
        raise ValueError(
            f"{source}: time bounds {name!r} do not bracket their samples "
            f"(e.g. sample {int(bad[0])}); each interval must satisfy "
            "start <= time <= end with start < end.")
    # Kept as an exact (n, 2) Datetime: the bounds ride on ForcingData, and a
    # float32 epoch-seconds array would blur them by ~2 minutes.
    return jdt.Datetime.from_datetime64(np.stack([lo, hi], axis=1))


def _prescribed_flux_time_axis(ds, align_mode, source):
    """Sort, validate and align-resolve a prescribed-flux time axis.

    Returns ``(order, ascending_times, align_int)``; see
    :func:`read_prescribed_surface_fluxes` for the rules.
    """
    if not _is_datetime_axis(ds["time"].values):
        raise ValueError(
            f"{source}: the time coordinate does not decode to dates (dtype "
            f"{ds['time'].dtype}); give it CF 'units' such as 'days since "
            "2000-01-01' (and a 'calendar'). A numeric axis cannot be aligned "
            "to the model clock.")
    # Axis hygiene on the decoded values themselves — datetime64 or cftime
    # objects, both ordered within their calendar — so no Gregorian epoch
    # conversion happens unless the mode is date-aligned: a climatology
    # stamped with idealised calendar dates (CF ``noleap`` year 0, a
    # ``360_day`` calendar) has no Gregorian equivalent, and WRAP_YEAR never
    # reads absolute times anyway.
    import pandas as pd
    raw = np.asarray(ds["time"].values)
    if bool(np.any(pd.isnull(raw))):
        raise ValueError(f"{source}: the time axis has missing (NaT) entries.")
    # Stable sort: BY_DATE's searchsorted needs ascending time, and WRAP_YEAR's
    # position 0 must be January.
    order = np.argsort(raw, kind="stable")
    ordered = raw[order]
    not_increasing = [k for k in range(ordered.size - 1)
                      if not ordered[k + 1] > ordered[k]]
    if not_increasing:
        raise ValueError(
            f"{source}: the time axis has duplicate timestamps (e.g. "
            f"{ordered[not_increasing[0]]}); each sample must have a distinct "
            "time.")

    # No data-mirror product carries prescribed fluxes, so ``auto`` has no
    # manifest kind to resolve from and raises (the shared #884 rule).
    mode = align_mode_code(resolve_align(
        align_mode, config_key="forcing.prescribed_surface_flux.align"))

    if mode == WRAP_YEAR:
        # Calendar months straight off the decoded values (``.dt`` handles
        # cftime calendars) — no epoch conversion.
        months = np.asarray(ds["time"].dt.month, dtype=np.int64)[order]
        if not (months.size == 12
                and np.array_equal(months, np.arange(1, 13))):
            raise ValueError(
                f"{source}: declared a climatology (align=wrap_year), but its "
                f"sorted samples fall in calendar months {months.tolist()}. "
                "Climatology replay (WRAP_YEAR) selects the sample of the "
                "model's calendar month, so it needs exactly twelve "
                "samples, one per month January..December. Supply such a file, "
                "or set forcing.prescribed_surface_flux.align=by_date to align "
                "the samples on their absolute timestamps.")
        return order, _wrap_year_times(ds.isel(time=order)), mode
    # Date-aligned: now (and only now) put the samples on the model's exact
    # Gregorian clock, in the sorted order.
    return order, _time_axis_from_ds(ds.isel(time=order)), mode


def by_date_coverage_error(ts: "TimeSeries", start_seconds: float,
                           end_seconds: float, name: str = "",
                           bounds=None) -> str | None:
    """Message if a date-aligned ``ts`` does not cover ``[start, end]``, else None.

    ``BY_DATE``/``BY_DATE_INTERP`` selection (:func:`_select_time_series`) clamps
    to the first/last sample outside the axis, so running past the end of a
    transient archive would otherwise silently hold its last sample forever.

    The usable window is, in order of preference:

    - ``bounds`` — the archive's own declared coverage (its CF ``time_bnds``,
      see :func:`read_prescribed_surface_fluxes`): an ``(n, 2)``
      :class:`jax_datetime.Datetime` (or seconds since 1970) of per-sample
      ``[start, end]`` intervals (a single ``(2,)`` pair is one interval). Exact for any stamp placement, and DISJOINT intervals are
      honoured as declared gaps: the run must lie inside one contiguous
      stretch of the union, because inside a gap the selection would hold a
      neighbouring sample the file itself says does not apply there. Bounds
      are the only way a gap is known — without them the axis is taken as
      contiguous, since gaps are never inferred from the stamps (#884);
    - otherwise one END interval beyond each end sample — the first interval
      repeated before the first sample, the last after the last
      (:func:`_repeat_cadence`; a calendar-month cadence steps by calendar
      months). Each end sample may stand for the interval before or after its
      stamp (month-start vs mid-month stamps), so the slack at each edge is
      THAT edge's own spacing — e.g. a Jan-1..Dec-1 monthly archive covers its
      calendar year.
      An interior gap never widens an edge: the largest spacing anywhere
      would let a daily archive with one long gap validate a run months past
      its last sample.

    ``WRAP_YEAR`` leaves (a climatology covers every date) return ``None``;
    a single dated sample without bounds covers only its own instant.
    ``start_seconds`` / ``end_seconds`` are seconds since 1970-01-01 of the
    run window (``Model`` computes them from its exact clock). The axis and
    bounds are exact whole-second dates, so the comparison is exact: a run
    ending exactly on the covered edge passes, one second past it fails.
    """
    mode = int(np.asarray(ts.align_mode))
    if mode not in (BY_DATE, BY_DATE_INTERP):
        return None
    t = _host_epoch_seconds(ts.times).astype(float).reshape(-1)
    if isinstance(bounds, jdt.Datetime):
        bounds = _host_epoch_seconds(bounds).astype(float)
    if bounds is not None:
        lo, hi, gap = _declared_coverage(bounds, start_seconds)
        what = "its declared time_bnds"
        if gap is not None:
            what += f", which declare a gap from {_iso(gap[0])} to {_iso(gap[1])}"
    elif t.size < 2:
        # One dated sample and no bounds: there is no cadence to extend it
        # by, so it covers only its own instant (never an assumed span).
        lo = hi = float(t[0])
        what = "a single sample with no declared bounds"
    else:
        lo = _repeat_cadence(t[1], t[0], t[0])
        hi = _repeat_cadence(t[-2], t[-1], t[-1])
        what = "one end-sample interval either side"
    if start_seconds >= lo and end_seconds <= hi:
        return None
    _d = _iso
    return (
        f"{name}: the date-aligned (BY_DATE) time axis spans {_d(t[0])} .. "
        f"{_d(t[-1])} (usable {_d(lo)} .. {_d(hi)}, {what}), but the run "
        f"covers {_d(start_seconds)} .. {_d(end_seconds)}. Outside its "
        "coverage a date-aligned series would silently hold a neighbouring "
        "sample. Supply fluxes covering the whole run; if the file is a "
        "monthly climatology meant to repeat every year, set "
        "forcing.prescribed_surface_flux.align=wrap_year.")


def _iso(seconds: float) -> str:
    """Seconds since 1970-01-01 as an ISO date-time string."""
    import pandas as pd
    return str(pd.Timestamp(float(seconds), unit="s"))[:19]


def _declared_coverage(bounds, start_seconds):
    """Return the contiguous declared stretch a run starting at ``start`` can use.

    ``bounds`` is ``(n, 2)`` (or one ``(2,)`` pair) of ``[start, end]``
    intervals. They are merged where they touch or overlap; the result is
    ``(lo, hi, gap)`` for the merged stretch containing ``start_seconds``
    (else the first stretch after it, else the last), where ``gap`` is the
    ``(end, next_start)`` of a declared gap that bounds it on the right, or
    ``None``.
    """
    iv = np.asarray(bounds, dtype=float).reshape(-1, 2)
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    k = next((i for i, (a, b) in enumerate(merged) if start_seconds <= b),
             len(merged) - 1)
    lo, hi = merged[k]
    gap = (hi, merged[k + 1][0]) if k + 1 < len(merged) else None
    return lo, hi, gap


def _repeat_cadence(a: float, b: float, origin: float) -> float:
    """Step ``origin`` by the interval ``a → b`` (seconds since 1970-01-01).

    Used for the edge slack of a date-aligned archive's coverage
    (:func:`by_date_coverage_error`): ``a → b`` is the archive's end interval
    and ``origin`` the end sample, so the slack is that interval repeated.
    ``b < a`` steps backwards. The stamps are on the model's Gregorian clock
    (a ``cftime`` axis is placed there by its nominal date), and the forms
    recognised as a CALENDAR cadence are, in order:

    1. **Month-end monthly / yearly** — ``a`` and ``b`` both the last day of
       their month (a Feb 28 also counts, being the noleap/365-day month end
       even in a Gregorian leap year) at the same time of day, whole months
       apart. Stepped by calendar months and re-anchored to the target
       month's last day: before Jan 31 is Dec 31, after Feb 28/29 is Mar 31.
       A noleap archive stepped into a leap February ends on Feb 29, the
       month's end on the model clock.
    2. **Same-day monthly / yearly** — ``a`` and ``b`` on the same
       day-of-month and time of day, whole months apart (month-start, fixed
       mid-month day 15, a yearly Jan-1 series): stepped by calendar months,
       so the interval after a Dec-1 sample ends on Jan-1 (December's 31
       days, not November's 30). A day past the target month's end is
       clamped to it.

    Everything else is repeated as its **elapsed length in seconds**, which
    is exact for fixed-length cadences (sub-daily, daily, weekly, any
    constant step) and the documented fallback for irregular calendar ones:
    mid-month stamps whose day varies (CMIP-style Jan 16 12:00 / Feb 15
    00:00), month-end stamps at differing times of day, and a month-end
    sample next to a non-month-end one.
    """
    import pandas as pd

    # The axis is exact whole seconds (``_host_epoch_seconds``), so the
    # calendar test reads the stamps as they are.
    def _snap(x):
        return pd.Timestamp(int(round(float(x))), unit="s")

    def _month_end(t):
        return t.is_month_end or (t.month == 2 and t.day == 28)

    def _seconds(t):
        return float((t - pd.Timestamp(0)).total_seconds())

    ta, tb = _snap(a), _snap(b)
    months = (tb.year - ta.year) * 12 + (tb.month - ta.month)
    if months != 0 and ta.time() == tb.time():
        if _month_end(ta) and _month_end(tb):
            stepped = _snap(origin) + pd.DateOffset(months=months)
            return _seconds(stepped + pd.offsets.MonthEnd(0))
        if ta.day == tb.day:
            return _seconds(_snap(origin) + pd.DateOffset(months=months))
    return float(origin) + (float(b) - float(a))


# ``expand_yearly_files`` is re-exported from the top-of-module import of the
# import-free engine :mod:`jcm.data.input_resolution` (see the imports block);
# its historical home is this module, so the runner and tests still reach it as
# ``jcm.forcing.expand_yearly_files``.


def default_forcing(
    grid: HorizontalGridTypes,
) -> ForcingData:
    """Initialize the default forcing data with prescribed SSTs"""
    sea_surface_temperature = _fixed_ssts(grid)

    return ForcingData.zeros(
        nodal_shape=grid.nodal_shape,sea_surface_temperature=sea_surface_temperature,
    )
