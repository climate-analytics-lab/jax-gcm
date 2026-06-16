"""Single-column radiative-convective equilibrium (RCE) on the SCM loop.

This is a thin configuration layer over :class:`jcm.single_column_model.SingleColumnModel`
and the composable ``PhysicsTerm`` stack — there is no bespoke time loop or
hand-rolled radiation/convection call here (cf. issue #523, whose prototype
called ``radiation_scheme_rrtmgp`` directly and integrated in plain Python).
Everything runs through the same ``compute_tendencies`` path the full model uses.

The pieces it wires together:

* **Radiation + convection as terms.** :func:`rce_physics` starts from
  ``echam_physics(...)`` and, for a clean fixed-SST RCE, trims it to the
  radiatively essential preamble + the radiation term + the chosen convection
  term (Betts-Miller by default — "moist-adiabat relaxation at fixed RH", which
  is exactly Case 1 of #523).
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

    scm = rce_column(sst=300.0, relative_humidity=0.7, solar_constant=728.4,
                     co2_ppmv=0.0, radiation_scheme="rrtmgp")
    ic  = rce_initial_state(scm.vertical, sst=300.0, relative_humidity=0.7)
    preds = run_rce(scm, ic, n_days=100)
    t_equilibrium = preds.relaxed_states["temperature"][-1]   # (nlev,) profile
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from dinosaur.hybrid_coordinates import HybridCoordinates

import jcm.constants as c
from jcm.forcing import ForcingData, SolarGeometry
from jcm.physics_interface import PhysicsState
from jcm.physics.clouds.sundqvist import saturation_specific_humidity
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.convection.betts_miller import (
    BettsMillerConvection,
    BettsMillerParameters,
)
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.single_column_model import SCMPredictions, SingleColumnModel
from jcm.utils import create_initial_tracers

# Categories stripped from ``echam_physics`` for a clean fixed-SST RCE. These
# all run *after* radiation in the ECHAM ordering and nothing upstream of them
# (the radiation preamble + radiation + convection we keep) consumes what they
# provide, so dropping them leaves a package that still passes
# ``ComposablePhysics`` ordering validation. ``"clouds"`` is the cloud
# *microphysics* term (``Echam1MMicrophysics`` / ``Lohmann2MMicrophysics``);
# the diagnostic cloud-fraction term radiation needs is ``"cloud_fraction"``
# (``SundqvistCloudFraction``), which is kept.
_NON_RCE_CATEGORIES = ("clouds", "vertical_diffusion", "surface", "hines", "sso")


def _half_level_coeffs(vertical):
    """Return ``(a_half, b_half)`` so that ``p_half = a_half + b_half · p_s``.

    Mirrors :meth:`BettsMillerConvection.cache_coords`: hybrid grids carry
    ``a_boundaries`` (Pa) and ``b_boundaries`` directly; sigma grids are pure
    ``b`` (``a = 0``).
    """
    if isinstance(vertical, HybridCoordinates):
        a_half = jnp.asarray(vertical.a_boundaries)   # Pa
        b_half = jnp.asarray(vertical.b_boundaries)   # dimensionless
    else:  # SigmaCoordinates
        sigma_boundaries = jnp.asarray(vertical.boundaries)
        a_half = jnp.zeros_like(sigma_boundaries)
        b_half = sigma_boundaries
    return a_half, b_half


def _full_level_pressure(a_half, b_half, surface_pressure_pa):
    """Full-level pressure (Pa) from the half-level coefficients and ``p_s``.

    Broadcasting-native: ``surface_pressure_pa`` may be a scalar (single column)
    or carry trailing horizontal axes, matching the column-physics convention.
    """
    lev = (-1,) + (1,) * jnp.ndim(surface_pressure_pa)
    phalf = (
        a_half.reshape(lev)
        + b_half.reshape(lev) * surface_pressure_pa[None]
    )
    return 0.5 * (phalf[:-1] + phalf[1:])


# Manabe & Wetherald (1967) sigma at which the prescribed RH reaches zero
# (≈ 20 hPa for a 1000 hPa surface). The taper keeps the stratosphere dry:
# without it, ``rh·qsat`` at the near-vacuum top-of-model pressures approaches
# the saturation cap (~1 kg/kg) and injects hundreds of g/kg of spurious
# stratospheric moisture that destabilises the column.
_MW_RH_FLOOR_SIGMA = 0.02

# Stratospheric specific-humidity floor [kg/kg] ≈ 1.6 ppmv. The Manabe-Wetherald
# taper drives RH (and hence ``q``) to exactly zero aloft, but RRTMGP's gas
# optics returns NaN on a zero water-vapour amount, so we floor ``q`` at a small,
# physically realistic stratospheric value (real lower-stratosphere H₂O is
# ~3–5 ppmv) rather than zero.
_STRATOSPHERE_Q_FLOOR = 1.0e-6


def _fixed_rh_specific_humidity(pfull, surface_pressure_pa, temperature, rh):
    """Manabe-Wetherald fixed-RH specific humidity in **kg/kg**.

    RH profile ``rh · max((σ − σ_floor)/(1 − σ_floor), 0)`` with ``σ = p/p_s``:
    surface RH ``rh``, tapering linearly to zero at ``σ = _MW_RH_FLOOR_SIGMA``,
    then floored at :data:`_STRATOSPHERE_Q_FLOOR` so the dry stratosphere still
    carries a trace amount (a hard zero NaNs RRTMGP). The result is kg/kg, the
    canonical ``PhysicsState.specific_humidity`` unit and the same unit
    :func:`saturation_specific_humidity` returns.
    """
    sigma = pfull / surface_pressure_pa
    rh_profile = rh * jnp.clip(
        (sigma - _MW_RH_FLOOR_SIGMA) / (1.0 - _MW_RH_FLOOR_SIGMA), 0.0, 1.0,
    )
    qsat = saturation_specific_humidity(pfull, temperature)  # kg/kg
    return jnp.maximum(rh_profile * qsat, _STRATOSPHERE_Q_FLOOR)


def fixed_rh_closure(
    relative_humidity: float, vertical
) -> Callable[[PhysicsState, ForcingData], PhysicsState]:
    """Build a fixed-relative-humidity ``state_closure`` for the SCM.

    Returns ``f(state, forcing) -> state`` that overwrites ``specific_humidity``
    with the Manabe & Wetherald (1967) fixed-RH profile (see
    :func:`_fixed_rh_specific_humidity`) computed from the column's *current*
    temperature. Applied by the SCM each step before physics, this slaves
    humidity to the freely evolving temperature — the fixed-RH assumption of
    Case 1 in issue #523.
    """
    a_half, b_half = _half_level_coeffs(vertical)
    rh = float(relative_humidity)

    def closure(state: PhysicsState, forcing: ForcingData) -> PhysicsState:
        ps = state.normalized_surface_pressure * c.p0
        pfull = _full_level_pressure(a_half, b_half, ps)
        q_gkg = _fixed_rh_specific_humidity(pfull, ps, state.temperature, rh)
        return state.copy(specific_humidity=q_gkg)

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
    *,
    radiation_scheme: str = "rrtmgp",
    convection: str = "betts_miller",
    solar_constant: float = 728.4,
    relative_humidity: float = 0.7,
    tau_convection: float = 7200.0,
    interactive: bool = False,
    **echam_kwargs,
) -> ComposablePhysics:
    """Build the RCE physics package from the composable ECHAM terms.

    Args:
        radiation_scheme: ``"rrtmgp"`` (default), ``"grey"``, or ``"emulated"``
            — forwarded to ``echam_physics``.
        convection: ``"betts_miller"`` (default; moist-adiabat relaxation toward
            ``relative_humidity``) or ``"tiedtke"`` (keep the ECHAM mass-flux
            scheme). Custom schemes can be swapped in afterwards with
            ``physics.replace("convection", term)``.
        solar_constant: Solar constant [W/m²] on ``RadiationParameters``; sets
            the insolation magnitude.
        relative_humidity: Target RH for the Betts-Miller reference profile
            (``rhbm``); should match the :func:`fixed_rh_closure` value.
        tau_convection: Betts-Miller relaxation timescale ``tau_bm`` [s].
        interactive: When ``False`` (default), trim the stack to the
            radiatively essential preamble + radiation + convection (see
            :data:`_NON_RCE_CATEGORIES`) for a clean fixed-SST, fixed-RH RCE.
            When ``True``, keep the full ECHAM stack (surface fluxes, vertical
            diffusion, microphysics, gravity waves) — the interactive-moisture /
            RCEMIP-style configuration.
        **echam_kwargs: Forwarded to ``echam_physics`` (e.g. ``cloud_scheme``).

    Returns:
        A ready-to-run ``ComposablePhysics``.

    """
    physics = echam_physics(
        radiation_scheme=radiation_scheme,
        radiation=RadiationParameters.default(solar_constant=solar_constant),
        **echam_kwargs,
    )

    if convection == "betts_miller":
        params = BettsMillerParameters.default().replace(
            tau_bm=jnp.asarray(float(tau_convection)),
            rhbm=jnp.asarray(float(relative_humidity)),
        )
        physics = physics.replace("convection", BettsMillerConvection(params))
    elif convection != "tiedtke":
        raise ValueError(
            f"Unknown convection={convection!r}; choose 'betts_miller' or 'tiedtke'."
        )

    if not interactive:
        # Rebuild once with the non-RCE categories filtered out (rather than a
        # chain of ``.remove`` calls) so ordering validation runs a single time
        # on the final term set.
        physics = ComposablePhysics(
            terms=[t for t in physics.terms if t.category not in _NON_RCE_CATEGORIES],
            checkpoint_terms=physics.checkpoint_terms,
            vectorize_columns=physics.vectorize_columns,
            dt_seconds=physics.dt_seconds,
            band_config=physics.band_config,
        )
    return physics


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
    a_half, b_half = _half_level_coeffs(vertical)
    ps = c.p0 if surface_pressure is None else float(surface_pressure)
    pfull = _full_level_pressure(a_half, b_half, jnp.asarray(ps))

    # Hydrostatic height with a 7.6 km scale height (matches #523's prototype),
    # purely to seed a plausible lapse-rate profile.
    z = -7.6e3 * jnp.log(pfull / ps)
    temperature = jnp.maximum(sst - lapse_rate * z, stratosphere_temperature)
    q_gkg = _fixed_rh_specific_humidity(
        pfull, jnp.asarray(ps), temperature, float(relative_humidity),
    )
    nlev = pfull.shape[0]

    return PhysicsState(
        u_wind=jnp.zeros(nlev),
        v_wind=jnp.zeros(nlev),
        temperature=temperature,
        specific_humidity=q_gkg,
        geopotential=c.grav * z,
        normalized_surface_pressure=jnp.asarray(ps / c.p0),
        tracers=create_initial_tracers(nlev),
    )


def rce_column(
    *,
    sst: float = 300.0,
    relative_humidity: float = 0.7,
    co2_ppmv: float = 348.0,
    solar_constant: float = 728.4,
    lat_deg: float = 42.55,
    nlev: int = 47,
    vertical=None,
    dt_seconds: float = 1200.0,
    radiation_scheme: str = "rrtmgp",
    convection: str = "betts_miller",
    tau_convection: float = 7200.0,
    interactive_humidity: bool = False,
    day_of_year_fraction: float = 0.22,
    physics: ComposablePhysics | None = None,
) -> SingleColumnModel:
    """Configure a :class:`SingleColumnModel` for radiative-convective equilibrium.

    Builds the trimmed RCE physics (unless ``physics`` is supplied), a fixed-SST
    + steady-insolation + fixed-CO₂ ``ForcingData``, and wires temperature into
    ``free_evolve`` with a :func:`fixed_rh_closure` ``state_closure`` (unless
    ``interactive_humidity=True``, in which case humidity evolves freely too and
    no closure is applied).

    Args:
        sst: Fixed sea-surface temperature [K] (the radiation lower boundary).
        relative_humidity: Fixed RH for the humidity closure and Betts-Miller.
        co2_ppmv: CO₂ volume mixing ratio [ppmv] on ``forcing.co2_vmr``.
        solar_constant: Solar constant [W/m²]; tune to hit the target TOA SW.
        lat_deg: Column latitude [deg] (fixes the zenith angle with ``solar``).
        nlev: Number of levels for ``get_echam_levels`` (47 or 40) when
            ``vertical`` is not given.
        vertical: Optional explicit vertical coordinate (e.g. a
            ``SigmaCoordinates`` for a cheap test grid); overrides ``nlev``.
        dt_seconds: Physics timestep [s].
        radiation_scheme, convection, tau_convection, interactive_humidity:
            See :func:`rce_physics`.
        day_of_year_fraction: Sun position for :func:`steady_insolation`.
        physics: Optional pre-built physics package (skips :func:`rce_physics`).

    Returns:
        A ``SingleColumnModel`` ready for :func:`run_rce`.

    """
    if vertical is None:
        vertical = get_echam_levels(nlev)

    if physics is None:
        physics = rce_physics(
            radiation_scheme=radiation_scheme,
            convection=convection,
            solar_constant=solar_constant,
            relative_humidity=relative_humidity,
            tau_convection=tau_convection,
            interactive=interactive_humidity,
        )

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
