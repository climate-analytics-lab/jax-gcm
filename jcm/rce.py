"""Single-column radiative-convective equilibrium (RCE) on the SCM loop.

This is a thin configuration layer over :class:`jcm.single_column_model.SingleColumnModel`
and the composable ``PhysicsTerm`` stack — there is no bespoke time loop or
hand-rolled radiation/convection call here (cf. issue #523, whose prototype
called ``radiation_scheme_rrtmgp`` directly and integrated in plain Python).
Everything runs through the same ``compute_tendencies`` path the full model uses.

The pieces it wires together:

* **Radiation + convection as terms.** :func:`rce_physics` composes — directly
  with ``ComposablePhysics`` — just the radiation and convection terms plus the
  small diagnostic preamble radiation requires (pressure/density, surface BCs,
  zeroed aerosol + clouds). Betts-Miller is the default convection ("moist-
  adiabat relaxation at fixed RH" = Case 1 of #523); pass any radiation /
  convection term, or hand the SCM a fully custom ``physics`` instead.
* **Fixed SST as the lower boundary.** Set on ``forcing.sea_surface_temperature``;
  ``EchamBoundaryConditions`` (which runs before radiation) copies it into the
  ``surface`` diagnostic the radiation term reads. No surface-flux term needed.
* **Steady insolation through the Forcing interface.** A fixed
  :class:`jcm.forcing.SolarGeometry` on ``forcing.solar`` (the SCM holds forcing
  constant across the scan, so the sun is perpetual). The insolation *magnitude*
  is set by ``RadiationParameters.solar_constant``; tune it to hit a target TOA
  ``toa_sw_down`` (the RCEMIP convention of a fixed sun at a representative
  angle with a scaled solar constant).
* **Free-evolving temperature + a fixed-RH humidity closure.** Temperature is
  handed to the SCM's ``free_evolve`` machinery; specific humidity is re-diagnosed
  each step from the evolving temperature by :func:`fixed_rh_closure` via the
  SCM's ``state_closure`` hook. (A fixed-RH closure cannot be a ``PhysicsTerm``:
  terms communicate only through tendencies and the diagnostics dict, so one
  term cannot overwrite the ``q`` that downstream terms read within a step.)

Units note: ``PhysicsState.specific_humidity`` is **kg/kg** — the canonical
physics convention (confirmed against a full-model run: surface q ≈ 0.02 kg/kg,
RH ~0.2–1), which radiation, ``SundqvistCloudFraction``, ``TiedtkeConvection``
and the Betts-Miller *core* all consume directly. Building this testbed surfaced
a ``/1000`` unit bug in the ``BettsMillerConvection`` *term wrapper* (it treated
the kg/kg state as g/kg), which is fixed alongside this module; the closure here
therefore emits kg/kg.

Example::

    from jcm.rce import rce_column, rce_initial_state, run_rce

    # RRTMGP is the default radiation; convective_rh defaults to
    # relative_humidity − 0.1 so Betts-Miller actually precipitates (see
    # ``rce_column`` on why rhbm must sit below the environmental RH).
    scm = rce_column(sst=300.0, relative_humidity=0.7, co2_ppmv=0.0)
    ic  = rce_initial_state(scm.vertical, sst=300.0, relative_humidity=0.7)
    preds = run_rce(scm, ic, n_days=100)
    t_equilibrium = preds.relaxed_states["temperature"][-1]   # (nlev,) profile
"""

from __future__ import annotations

from typing import Callable, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from dinosaur.hybrid_coordinates import HybridCoordinates

import jcm.constants as c
from jcm.forcing import ForcingData, SolarGeometry
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.aerosol.aerosol_types import AerosolData
from jcm.physics.clouds.cloud_data import CloudData
from jcm.physics.clouds.sundqvist import saturation_specific_humidity
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.convection.betts_miller import (
    BettsMillerConvection,
    BettsMillerParameters,
)
from jcm.physics.diagnostics.moist_air_state import MoistAirColumnState
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.forcing.echam_boundary_conditions import EchamBoundaryConditions
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.radiation.rrtmgp import RRTMGPRadiation, _ensure_rrtmgp
from jcm.single_column_model import SCMPredictions, SingleColumnModel
from jcm.utils import create_initial_tracers


class _ClearSky(PhysicsTerm):
    """Provide zeroed ``aerosol`` and ``clouds`` diagnostics for a clear-sky RCE.

    A radiation term ``requires`` the ``aerosol`` and ``clouds`` diagnostic keys,
    but for a clean clear-sky, aerosol-free RCE both are zero (cf. the zeroed
    ``aerosol_data`` and cloud fields in the issue #523 prototype). This stands
    in for the full ``Macv2SpAerosol`` + ``SundqvistCloudFraction`` terms; the SW
    band count for the aerosol optics is read from the active radiation backend.
    """

    name: ClassVar[str] = "clear_sky"
    category: ClassVar[str] = "clear_sky"
    provides: ClassVar[tuple[str, ...]] = ("aerosol", "clouds")

    def __init__(self) -> None:
        self._n_bnd_sw = 14
        self._n_bnd_lw = 16

    def cache_band_config(self, band_config) -> None:
        """Match the aerosol band count to the radiation backend (RRTMGP / grey)."""
        self._n_bnd_sw = len(band_config.sw_band_centers_nm)
        self._n_bnd_lw = len(band_config.lw_band_centers_nm)

    def __call__(self, state, diagnostics, forcing, terrain):
        """Return zero tendency and zeroed aerosol + cloud diagnostics."""
        nlev, ncols = state.temperature.shape
        return PhysicsTendency.zeros(state.temperature.shape), {
            **diagnostics,
            "aerosol": AerosolData.zeros(
                (ncols,), nlev, n_bnd_sw=self._n_bnd_sw, n_bnd_lw=self._n_bnd_lw,
            ),
            "clouds": CloudData.zeros((ncols,), nlev),
        }


def _pressure_centers(vertical, surface_pressure_pa):
    """Full-level pressure [Pa] for a single column from the vertical coordinate.

    Reuses :meth:`HybridCoordinates.pressure_centers`; for sigma coordinates the
    full-level pressure is just ``sigma_centers · p_s``. Top-first (index 0 =
    model top), matching the column-state ordering.
    """
    if isinstance(vertical, HybridCoordinates):
        return vertical.pressure_centers(surface_pressure_pa)
    return jnp.asarray(vertical.centers) * surface_pressure_pa


# Pressure window [Pa] over which the prescribed RH tapers from its full
# tropospheric value (at and below ``_RH_TAPER_BASE_PA``) to zero (at and above
# ``_RH_TAPER_TOP_PA``). The taper exists *only* to keep the stratosphere dry:
# at the near-vacuum top-of-model pressures ``qsat`` diverges (``es`` exceeds the
# ambient pressure), so an unmodified ``rh·qsat`` injects spurious stratospheric
# moisture (and NaNs RRTMGP). Crucially the taper lives in the *stratosphere*
# (above ~100 hPa), leaving the whole convecting troposphere at the uniform
# environmental RH — the configuration of the validated homebrew RCE. An earlier
# Manabe-Wetherald σ-taper from the surface instead dried the *mid-troposphere*
# (RH ~0.05–0.36 there), which starved Betts-Miller's parcel-referenced trigger
# and silently switched convection off; see :func:`rce_column`.
_RH_TAPER_BASE_PA = 1.0e4   # 100 hPa: full RH at and below this (troposphere)
_RH_TAPER_TOP_PA = 2.0e3    # 20 hPa: RH forced to zero at and above this

# Stratospheric specific-humidity floor [kg/kg] ≈ 1.6 ppmv. The taper drives RH
# (and hence ``q``) to zero aloft, but RRTMGP's gas optics returns NaN on a zero
# water-vapour amount, so we floor ``q`` at a small, physically realistic
# stratospheric value (real lower-stratosphere H₂O is ~3–5 ppmv) rather than zero.
_STRATOSPHERE_Q_FLOOR = 1.0e-6


def _fixed_rh_specific_humidity(pfull, surface_pressure_pa, temperature, rh):
    """Fixed-RH specific humidity in **kg/kg**: uniform troposphere, dry stratosphere.

    RH equals the environmental value ``rh`` everywhere in the troposphere
    (``p ≥`` :data:`_RH_TAPER_BASE_PA`), tapers linearly to zero across the
    stratosphere (:data:`_RH_TAPER_BASE_PA` → :data:`_RH_TAPER_TOP_PA`), and the
    resulting ``q`` is floored at :data:`_STRATOSPHERE_Q_FLOOR` so the dry top
    still carries a trace amount (a hard zero NaNs RRTMGP). Holding the *whole
    troposphere* at the uniform environmental RH (rather than tapering from the
    surface) is what keeps the column moist enough for Betts-Miller to convect —
    the validated homebrew RCE setup. The result is kg/kg, the canonical
    ``PhysicsState.specific_humidity`` unit and the same unit
    :func:`saturation_specific_humidity` returns.

    ``surface_pressure_pa`` is unused now that the taper is in absolute pressure
    (kept in the signature so callers need not special-case it).
    """
    del surface_pressure_pa
    rh_profile = rh * jnp.clip(
        (pfull - _RH_TAPER_TOP_PA) / (_RH_TAPER_BASE_PA - _RH_TAPER_TOP_PA),
        0.0, 1.0,
    )
    qsat = saturation_specific_humidity(pfull, temperature)  # kg/kg
    return jnp.maximum(rh_profile * qsat, _STRATOSPHERE_Q_FLOOR)


def fixed_rh_closure(
    relative_humidity: float, vertical
) -> Callable[[PhysicsState, ForcingData], PhysicsState]:
    """Build a fixed-relative-humidity ``state_closure`` for the SCM.

    Returns ``f(state, forcing) -> state`` that overwrites ``specific_humidity``
    with the fixed-RH profile (uniform environmental RH through the troposphere,
    dry stratosphere; see :func:`_fixed_rh_specific_humidity`) computed from the
    column's *current* temperature. Applied by the SCM each step before physics,
    this slaves humidity to the freely evolving temperature — the fixed-RH
    assumption of Case 1 in issue #523.
    """
    rh = float(relative_humidity)

    def closure(state: PhysicsState, forcing: ForcingData) -> PhysicsState:
        ps = state.normalized_surface_pressure * c.p0
        pfull = _pressure_centers(vertical, ps)
        q = _fixed_rh_specific_humidity(pfull, ps, state.temperature, rh)
        return state.copy(specific_humidity=q)

    return closure


def steady_insolation(
    day_of_year_fraction: float = 0.22,
    time_of_day_fraction: float = 0.5,
) -> SolarGeometry:
    """Build a fixed :class:`SolarGeometry` for perpetual (steady) insolation.

    The SCM passes one static ``forcing`` every step, so a constant
    ``SolarGeometry`` places the sun at a fixed position — the perpetual-sun
    setup RCE uses. The defaults put the sun near an equinox at local noon; the
    geometry fixes the zenith *angle*, and the insolation *magnitude* is then
    controlled by ``RadiationParameters.solar_constant`` (tune it to hit a
    target ``toa_sw_down``). ``time_of_day_fraction`` and ``day_of_year_fraction``
    are fractions in ``[0, 1)``.
    """
    two_pi = 2.0 * np.pi
    return SolarGeometry(
        tyear=jnp.float32(day_of_year_fraction),
        orbital_phase=jnp.float32(two_pi * day_of_year_fraction),
        synodic_phase=jnp.float32(two_pi * time_of_day_fraction),
    )


def rce_physics(
    radiation: PhysicsTerm | None = None,
    convection: PhysicsTerm | None = None,
) -> ComposablePhysics:
    """Compose the minimal radiative-convective stack with ``ComposablePhysics``.

    A radiation term needs a few diagnostics, so the package is just what it
    requires — :class:`MoistAirColumnState` (pressure, density),
    :class:`EchamBoundaryConditions` (surface temperature/albedo and the analytic
    ozone/CO₂ radiation reads), and :class:`_ClearSky` (zeroed aerosol + cloud
    diagnostics) — followed by the ``radiation`` and ``convection`` terms. Six
    terms, composed directly: a clean clear-sky, fixed-SST radiative-convective
    column with no surface fluxes, vertical diffusion, microphysics or aerosol.

    Args:
        radiation: Radiation ``PhysicsTerm``; defaults to RRTMGP. Pass a
            ``GreyTwoStreamRadiation`` for a cheap test, or any custom term.
        convection: Convection ``PhysicsTerm``; defaults to Betts-Miller.

    Returns:
        A ``ComposablePhysics`` (column-vectorised) ready for the SCM.

    """
    radiation = radiation or RRTMGPRadiation(params=RadiationParameters.default())
    convection = convection or BettsMillerConvection()
    band_config = (
        RadiationBandConfig.from_rrtmgp(_ensure_rrtmgp())
        if isinstance(radiation, RRTMGPRadiation)
        else RadiationBandConfig.broadband()
    )
    return ComposablePhysics(
        terms=[
            MoistAirColumnState(),
            EchamBoundaryConditions(),
            _ClearSky(),
            radiation,
            convection,
        ],
        vectorize_columns=True,
        band_config=band_config,
    )


def rce_initial_state(
    vertical,
    *,
    sst: float = 300.0,
    relative_humidity: float = 0.7,
    lapse_rate: float = 6.5e-3,
    stratosphere_temperature: float = 200.0,
    surface_pressure: float | None = None,
) -> PhysicsState:
    """Build a 1-D column ``PhysicsState`` initial condition for an RCE run.

    A ``lapse_rate`` (K/m) profile from ``sst`` capped at
    ``stratosphere_temperature``, with humidity at fixed ``relative_humidity``
    (the same closure the run uses, so step 0 starts consistent). Winds are
    zero; ``qc``/``qi`` tracers are zero. Surface pressure defaults to the
    thermodynamic reference ``c.p0``.
    """
    ps = c.p0 if surface_pressure is None else float(surface_pressure)
    pfull = _pressure_centers(vertical, jnp.asarray(ps))

    # Hydrostatic height with a 7.6 km scale height (matches #523's prototype),
    # purely to seed a plausible lapse-rate profile.
    z = -7.6e3 * jnp.log(pfull / ps)
    temperature = jnp.maximum(sst - lapse_rate * z, stratosphere_temperature)
    q = _fixed_rh_specific_humidity(
        pfull, jnp.asarray(ps), temperature, float(relative_humidity),
    )
    nlev = pfull.shape[0]

    return PhysicsState(
        u_wind=jnp.zeros(nlev),
        v_wind=jnp.zeros(nlev),
        temperature=temperature,
        specific_humidity=q,
        geopotential=c.grav * z,
        normalized_surface_pressure=jnp.asarray(ps / c.p0),
        tracers=create_initial_tracers(nlev),
    )


def rce_column(
    *,
    sst: float = 300.0,
    relative_humidity: float = 0.7,
    convective_rh: float | None = None,
    co2_ppmv: float = 348.0,
    solar_constant: float = 543.2,
    lat_deg: float = 42.55,
    nlev: int = 47,
    vertical=None,
    dt_seconds: float = 1200.0,
    radiation: PhysicsTerm | None = None,
    convection: PhysicsTerm | None = None,
    tau_convection: float = 7200.0,
    interactive_humidity: bool = False,
    day_of_year_fraction: float = 0.22,
    physics: ComposablePhysics | None = None,
) -> SingleColumnModel:
    """Configure a :class:`SingleColumnModel` for radiative-convective equilibrium.

    Builds the minimal RCE physics (unless ``physics`` is supplied), a fixed-SST
    + steady-insolation + fixed-CO₂ ``ForcingData``, and wires temperature into
    ``free_evolve`` with a :func:`fixed_rh_closure` ``state_closure`` (unless
    ``interactive_humidity=True``, in which case humidity evolves freely too and
    no closure is applied).

    Why ``rhbm`` is *not* ``relative_humidity``: these are two physically distinct
    relative humidities and conflating them silently kills convection. The
    environmental RH (``relative_humidity``) is what the fixed-RH closure holds the
    column at; ``rhbm`` is the target RH of Betts-Miller's moist-adiabatic
    *reference* profile. Because the reference humidity is built on the warmer
    lifted *parcel* (Isca's default, ``do_envsat=False``), deep — i.e.
    *precipitating* — convection only switches on when the column is moister than
    that reference, which requires ``rhbm < relative_humidity``. With
    ``rhbm == relative_humidity`` the column lands in Betts-Miller's
    non-precipitating "shallow" branch: under the ``SIMP`` flavor that returns
    *zero* tendency, and even the default ``SHALLOWER`` flavor only
    redistributes moisture without precipitating, so the column relaxes to
    near-*radiative* equilibrium (looks equilibrated, OLR fine) with deep
    convection doing nothing. So
    ``convective_rh`` defaults to ``relative_humidity − 0.1`` (the working homebrew
    used env 0.7 / rhbm 0.6), and supplying ``convective_rh >= relative_humidity``
    for the default Betts-Miller term is rejected as a no-op misconfiguration.

    Args:
        sst: Fixed sea-surface temperature [K] (the radiation lower boundary).
        relative_humidity: Environmental RH the fixed-RH humidity closure holds
            the column at (also seeds ``rce_initial_state``).
        convective_rh: Target RH ``rhbm`` of the *default* Betts-Miller reference
            profile. Must be ``< relative_humidity`` for deep convection (see
            above). Defaults to ``relative_humidity − 0.1`` (floored at 0.05).
            Ignored if ``convection`` is supplied.
        co2_ppmv: CO₂ volume mixing ratio [ppmv] on ``forcing.co2_vmr``.
        solar_constant: Solar constant [W/m²] for the *default* radiation term;
            tune to hit the target TOA SW. Ignored if ``radiation`` is given.
            The default 543.2 delivers the RCEMIP target ``toa_sw_down``
            ≈ 409.6 W/m² at this configuration's fixed sun (µ0 = 0.7427 at
            ``lat_deg=42.55`` / ``day_of_year_fraction=0.22``, orbital
            distance factor 1.0153): 543.2 × 1.0153 × 0.7427 ≈ 409.6. The
            previous default (728.4) was calibrated against the µ0²
            insolation bug in the RRTMGP glue (fixed alongside this change)
            and would now deliver ~549 W/m² — enough to shut Betts-Miller
            deep convection off entirely in the default column.
        lat_deg: Column latitude [deg] (fixes the zenith angle with ``solar``).
        nlev: Number of levels for ``get_echam_levels`` (47 or 40) when
            ``vertical`` is not given.
        vertical: Optional explicit vertical coordinate (e.g. a
            ``SigmaCoordinates`` for a cheap test grid); overrides ``nlev``.
        dt_seconds: Physics timestep [s].
        radiation: Radiation ``PhysicsTerm``; defaults to RRTMGP at
            ``solar_constant``.
        convection: Convection ``PhysicsTerm``; defaults to Betts-Miller at
            ``rhbm=relative_humidity``, ``tau_bm=tau_convection``.
        tau_convection: ``tau_bm`` [s] for the *default* Betts-Miller term.
        interactive_humidity: When ``True``, evolve humidity freely (no closure)
            instead of slaving it to fixed RH. NB: the minimal RCE stack has no
            moisture *source* (no surface evaporation / vertical diffusion term),
            so a freely evolving ``q`` only ever dries out and convection then
            shuts off — for an interactive-``q`` RCE add a surface-flux term to
            ``physics``. The fixed-RH default (this flag ``False``) is Case 1.
        day_of_year_fraction: Sun position for :func:`steady_insolation`.
        physics: Pre-built ``ComposablePhysics`` (skips the default build, so
            you compose any terms you like — the point of ``ComposablePhysics``).

    Returns:
        A ``SingleColumnModel`` ready for :func:`run_rce`.

    """
    if vertical is None:
        vertical = get_echam_levels(nlev)

    if physics is None:
        if radiation is None:
            radiation = RRTMGPRadiation(
                params=RadiationParameters.default(solar_constant=solar_constant),
            )
        if convection is None:
            rhbm = (
                max(float(relative_humidity) - 0.1, 0.05)
                if convective_rh is None
                else float(convective_rh)
            )
            if rhbm >= float(relative_humidity):
                raise ValueError(
                    f"convective_rh ({rhbm}) must be < relative_humidity "
                    f"({float(relative_humidity)}) for the default Betts-Miller "
                    "term: deep (precipitating) convection only fires when the "
                    "column is moister than the convective reference profile. "
                    "With convective_rh >= relative_humidity the scheme stays in "
                    "its non-precipitating branch and produces zero tendency."
                )
            convection = BettsMillerConvection(BettsMillerParameters.default().replace(
                rhbm=jnp.asarray(rhbm),
                tau_bm=jnp.asarray(float(tau_convection)),
            ))
        physics = rce_physics(radiation=radiation, convection=convection)

    forcing = ForcingData.zeros((1, 1)).copy(
        sea_surface_temperature=jnp.full((1, 1), float(sst)),
        co2_vmr=jnp.asarray(float(co2_ppmv)),
        solar=steady_insolation(day_of_year_fraction=day_of_year_fraction),
    )

    if interactive_humidity:
        free_evolve = ("specific_humidity", "temperature")
        state_closure = None
    else:
        free_evolve = ("temperature",)
        state_closure = fixed_rh_closure(relative_humidity, vertical)

    return SingleColumnModel(
        physics=physics,
        vertical=vertical,
        lat_deg=lat_deg,
        lon_deg=0.0,
        forcing=forcing,
        dt_seconds=dt_seconds,
        free_evolve=free_evolve,
        state_closure=state_closure,
    )


def run_rce(scm: SingleColumnModel, initial_state: PhysicsState, *, n_days: float) -> SCMPredictions:
    """Integrate an RCE column for ``n_days`` from ``initial_state``.

    The large-scale boundary state is constant in time for RCE, so the single
    initial condition is broadcast along a length-``n_steps`` time axis and
    handed to ``scm.run``; only the freely evolving prognostics (temperature,
    and humidity when interactive) and the diagnostics actually change step to
    step. The equilibrium profile is ``predictions.relaxed_states["temperature"][-1]``.

    ``scm.run`` returns the full per-step history, so keep ``n_days`` modest (or
    decimate afterwards) for long integrations.
    """
    n_steps = max(1, int(round(n_days * 86400.0 / scm.dt_seconds)))
    prescribed = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (n_steps,) + jnp.shape(x)),
        initial_state,
    )
    return scm.run(prescribed)
