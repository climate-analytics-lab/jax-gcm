"""Main vertical diffusion scheme for ECHAM physics.

This module provides the main interface for vertical diffusion and boundary layer
physics, integrating turbulence coefficient calculations with the matrix solver.
"""

import functools

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from jcm.physics.clouds.sundqvist import saturation_specific_humidity
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffTendencies, VDiffDiagnostics
)
from .turbulence_coefficients import (
    compute_richardson_number, compute_mixing_length, compute_exchange_coefficients,
    compute_turbulence_diagnostics
)
from .matrix_solver import vertical_diffusion_step
from .tke_budget import (
    compute_tke_exchange_coefficient,
    compute_tke_diagnostics,
    echam_tke_source_update,
    echam_thv_variance_source_update,
)


@jax.jit
def compute_dry_static_energy(
    temperature: jnp.ndarray,
    geopotential: jnp.ndarray
) -> jnp.ndarray:
    """Compute dry static energy.
    
    Args:
        temperature: Temperature [K]
        geopotential: Geopotential [m²/s²]
        
    Returns:
        Dry static energy [J/kg]

    """
    return c.cpd * temperature + geopotential


@jax.jit
def compute_virtual_temperature(
    temperature: jnp.ndarray,
    qv: jnp.ndarray
) -> jnp.ndarray:
    """Compute virtual temperature.
    
    Args:
        temperature: Temperature [K]
        qv: Water vapor mixing ratio [kg/kg]
        
    Returns:
        Virtual temperature [K]

    """
    return temperature * (1.0 + 0.608 * qv)


@jax.jit
def prepare_vertical_diffusion_state(
    u: jnp.ndarray,
    v: jnp.ndarray,
    temperature: jnp.ndarray,
    qv: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    pressure_full: jnp.ndarray,
    pressure_half: jnp.ndarray,
    geopotential: jnp.ndarray,
    height_full: jnp.ndarray,
    height_half: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_fraction: jnp.ndarray,
    roughness_length: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    tke: jnp.ndarray,
    thv_variance: jnp.ndarray,
    roughness_heat: jnp.ndarray = None,
    surface_wetness: jnp.ndarray = None,
) -> VDiffState:
    """Prepare the vertical diffusion state from input variables.

    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        temperature: Temperature [K] (ncol, nlev)
        qv: Water vapor mixing ratio [kg/kg] (ncol, nlev)
        qc: Cloud water mixing ratio [kg/kg] (ncol, nlev)
        qi: Cloud ice mixing ratio [kg/kg] (ncol, nlev)
        pressure_full: Full level pressure [Pa] (ncol, nlev)
        pressure_half: Half level pressure [Pa] (ncol, nlev+1)
        geopotential: Geopotential [m²/s²] (ncol, nlev)
        height_full: Full level height [m] (ncol, nlev)
        height_half: Half level height [m] (ncol, nlev+1)
        surface_temperature: Surface temperature [K] (ncol, nsfc_type)
        surface_fraction: Surface type fraction [-] (ncol, nsfc_type)
        roughness_length: Momentum roughness z0m [m] (ncol, nsfc_type)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        thv_variance: Variance of theta_v [K²] (ncol, nlev)
        roughness_heat: Heat roughness z0h [m] (ncol, nsfc_type). When
            ``None``, defaults to ``0.1·roughness_length`` — a standard
            ratio that's good enough for the original Businger-Dyer
            scheme. The ECHAM-Louis scheme expects per-tile values from
            the boundary forcing; build them at the call site.
        surface_wetness: Effective surface saturation fraction
            (ncol, nsfc_type). When ``None``, defaults to ``1.0`` for
            every tile (open-water / saturated-leaf assumption); the
            ECHAM-Louis scheme uses this to scale land latent flux from
            the JSBACH-equivalent ``cair``.

    Returns:
        Complete vertical diffusion state

    """
    # Compute air masses
    # dp should be positive (higher pressure - lower pressure)
    dp = jnp.diff(pressure_half, axis=1)  # This gives p[k+1] - p[k], which is positive
    air_mass = dp / c.grav

    if roughness_heat is None:
        roughness_heat = 0.1 * roughness_length
    if surface_wetness is None:
        surface_wetness = jnp.ones_like(roughness_length)

    return VDiffState(
        u=u,
        v=v,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        pressure_full=pressure_full,
        pressure_half=pressure_half,
        geopotential=geopotential,
        air_mass=air_mass,
        surface_temperature=surface_temperature,
        surface_fraction=surface_fraction,
        roughness_length=roughness_length,
        roughness_heat=roughness_heat,
        surface_wetness=surface_wetness,
        height_full=height_full,
        height_half=height_half,
        tke=tke,
        thv_variance=thv_variance,
        ocean_u=ocean_u,
        ocean_v=ocean_v
    )


@functools.partial(jax.jit, static_argnames=("couple_surface",))
def vertical_diffusion_column(
    state: VDiffState,
    params: VDiffParameters,
    dt: float,
    couple_surface: bool = True,
) -> Tuple[VDiffTendencies, VDiffDiagnostics]:
    """Compute vertical diffusion for a single column.

    By default the implicit solve carries the ECHAM surface exchange as the
    bottom-row Robin boundary condition for u, v, T and qv: the per-tile
    exchange velocities are computed *before* the matrix step, collapsed to
    grid coefficients / targets (see inline comments), fed into the solve,
    and the delivered surface fluxes are diagnosed from the implicit
    solution (reported == delivered by construction — ECHAM's ``pev_vdiff``
    identity). Set ``couple_surface=False`` to run the interior-only
    operator with the legacy zero-flux (insulating, free-slip) boundaries —
    used by tests that pin the interior diffusion in isolation.

    Args:
        state: Vertical diffusion state
        params: Vertical diffusion parameters
        dt: Time step [s]
        couple_surface: Static flag — include the surface Robin BC (default).

    Returns:
        Tuple of (tendencies, diagnostics)

    """
    # Compute turbulence coefficients
    ri = compute_richardson_number(
        state.u, state.v, state.temperature,
        state.height_full, state.height_half
    )
    
    # Estimate boundary layer height (initial guess)
    pbl_height_guess = jnp.full(state.u.shape[0], 1000.0)
    
    mixing_length = compute_mixing_length(
        state.height_full, state.height_half, ri, pbl_height_guess
    )
    
    exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture = (
        compute_exchange_coefficients(state, params, mixing_length, ri)
    )
    
    # === ECHAM split-update for TKE ============================================
    # Match the ECHAM ``vdiff.f90`` formulation:
    #   1. Apply the source/sink (shear production, buoyancy production,
    #      dissipation) ANALYTICALLY via the implicit ``sqrt(zktest)-1``
    #      formula — see ``echam_tke_source_update``. This step is
    #      unconditionally non-negative and bounded by the production /
    #      dissipation equilibrium, so it cannot blow up regardless of
    #      input.
    #   2. Use that post-source TKE as the matrix-solver input and let
    #      the matrix do ONLY the vertical-transport implicit step.
    #
    # The previous JCM design instead added the source tendency as a
    # forward-Euler increment on top of the matrix tendency. That
    # explicit step has no stability bound — combined with the cross-
    # step ``prev_physics_data`` cache in averaged mode, a single ill-
    # conditioned column ran TKE to ~10¹⁸ in four steps. ECHAM has
    # avoided this for decades by doing the source step analytically.
    # ===========================================================================

    # Step 1: analytic implicit source/sink update on a per-cell basis.
    shear_sq = _column_shear_squared(state.u, state.v, state.height_full)
    buoy_n2 = _column_buoyancy_freq_squared(
        state.temperature, state.height_full,
    )
    post_source_tke = echam_tke_source_update(
        prev_tke=state.tke,
        shear_squared=shear_sq,
        buoy_freq_squared=buoy_n2,
        mixing_length=mixing_length,
        dt=dt,
    )

    # Step 1b: the SAME split for the variance of virtual potential
    # temperature. ECHAM advances ``pthvvar`` in the same loop as TKE
    # (vdiff.f90:857-860) and then hands it to the same implicit transport
    # solve, so the two prognostics stay on the same footing. Without this
    # the variance had no source at all and only ever decayed toward its
    # floor — which is why ``pthvsig`` could not be used and the convective
    # ``zlift`` had to fall back to a constant.
    thv_gradient = _column_thv_gradient(
        state.temperature, state.pressure_full,
        state.qv, state.qc, state.qi, state.height_full,
    )
    # PRE-source TKE, deliberately: ECHAM evaluates BOTH variance terms at
    # ``ztkesq = SQRT(ptkem1)`` — the previous time level — (vdiff.f90:849,
    # 857-858; only the transport coefficients at :855-856 rescale to the
    # post-source ``ztkevn``). ``exchange_coeff_heat`` above already carries
    # √(state.tke), so production and dissipation share one turbulent
    # velocity scale, which is also what makes the documented equilibrium
    # cancellation var* = 2·c_h·l²·G²/c_d exact. Passing the post-source
    # TKE here mixed the two levels (Codex on #690).
    post_source_thv_var = echam_thv_variance_source_update(
        prev_thv_variance=state.thv_variance,
        thv_gradient=thv_gradient,
        exchange_coeff_heat=exchange_coeff_heat,
        tke=state.tke,
        mixing_length=mixing_length,
        dt=dt,
    )

    # Step 2: matrix solver for vertical transport, with the post-source
    # TKE and θ_v variance as input. Build a shallow-copied state so we
    # don't mutate the caller-owned ``state`` and so other variables still
    # see the original ``state.tke`` for their own coupling (if any).
    state_for_solver = state._replace(
        tke=post_source_tke, thv_variance=post_source_thv_var,
    )

    tke_exchange_coeff = compute_tke_exchange_coefficient(
        post_source_tke, mixing_length,
    )

    # Diagnostics still use the old per-source decomposition for now —
    # they're informational, not on the integration path.
    tke_shear_prod, tke_buoyancy_prod, tke_dissipation, _ = (
        compute_tke_diagnostics(
            state_for_solver, params,
            exchange_coeff_momentum, exchange_coeff_heat, mixing_length,
        )
    )

    # Per-tile exchange velocities are computed BEFORE the matrix step so
    # they can serve as the implicit solve's surface boundary condition
    # (previously they were diagnostics-only, decoupled from the column).
    diagnostics = compute_turbulence_diagnostics(
        state_for_solver, params, exchange_coeff_momentum,
        exchange_coeff_heat, exchange_coeff_moisture,
    )

    if couple_surface:
        # === Tile collapse for the surface Robin BC ========================
        # Every tile surface value X_s is prescribed this step (SST forcing,
        # min(SST, ctfreez) ice, stl_am land — the ECHAM "ocean branch"
        # where the Richtmyer–Morton handshake degenerates to a plain Robin
        # BC), so the per-tile rows collapse linearly to one grid
        # coefficient and one flux-weighted target per variable:
        #
        #   moisture:  C_q = Σ_t f_t·w_t·C_E,t
        #              q_s_eff = Σ_t f_t·w_t·C_E,t·q_sat(T_s,t, p_sfc) / C_q
        #   heat:      C_h = Σ_t f_t·C_H,t
        #              T_s_eff = Σ_t f_t·C_H,t·T_s,t / C_h − φ_K/cpd
        #   momentum:  C_m = Σ_t f_t·C_M,t ; target = (ocean_u, ocean_v)
        #
        # The wetness weighting w_t reproduces the port's own tile formula
        # (surface_layer.py: qts = w·qsat + (1−w)·q_air ⇒ flux ρC·w·(qsat −
        # q̂)), ECHAM's cair = csat = w special case of richtmyer_land.
        #
        # The −φ_K/cpd term on the heat target is MANDATORY: the solver
        # diffuses T, not dry static energy s = cp·T + gz (a pre-existing
        # interior infidelity, tracked separately). Coupling T_K directly to
        # T_s would drive a spurious flux equal to the adiabatic lapse
        # across the lowest half-layer; exchanging with T_s − φ_K/cpd (φ_K =
        # g·(height_full[K] − height_half[K+1/2]), height above the surface)
        # makes the bottom exchange exactly the dry-static-energy flux
        # ρ·C_H·(s_s − ŝ_K)/cp.
        # ====================================================================
        frac = state.surface_fraction                       # (ncol, nsfc)
        wet = jnp.clip(state.surface_wetness, 0.0, 1.0)
        ch_t = diagnostics.surface_exchange_heat
        ce_t = diagnostics.surface_exchange_moisture
        cm_t = diagnostics.surface_exchange_momentum

        c_mom = jnp.sum(frac * cm_t, axis=1)
        c_heat = jnp.sum(frac * ch_t, axis=1)
        c_moist = jnp.sum(frac * wet * ce_t, axis=1)

        # Per-tile saturation humidity at the surface pressure — the same
        # thermodynamics the ECHAM-Louis surface layer uses for its qts.
        p_sfc = state.pressure_half[:, -1]
        qsat_tiles = saturation_specific_humidity(
            p_sfc[:, None], state.surface_temperature,
        )
        tiny = 1.0e-12  # C floors at 1e-6 per tile; guard the 0-fraction limit
        q_s_eff = (
            jnp.sum(frac * wet * ce_t * qsat_tiles, axis=1)
            / jnp.maximum(c_moist, tiny)
        )
        phi_k = c.grav * (state.height_full[:, -1] - state.height_half[:, -1])
        t_s_eff = (
            jnp.sum(frac * ch_t * state.surface_temperature, axis=1)
            / jnp.maximum(c_heat, tiny)
        ) - phi_k / c.cpd

        surface_exchange = (c_mom, c_heat, c_moist)
        surface_target = (state.ocean_u, state.ocean_v, t_s_eff, q_s_eff)
    else:
        surface_exchange = None
        surface_target = None

    # The matrix solver returns ``tke_tendency = (matrix_tke_new -
    # state_for_solver.tke) / dt``. Since the caller computes
    # ``new_tke = state.tke + dt * tke_tendency`` against the *original*
    # (raw, pre-source) ``state.tke``, we rewrite ``tke_tendency`` to be
    # in those reference units before returning so the caller's formula
    # recovers ``matrix_tke_new`` directly. Equivalent rewrite:
    #   new_tke_tend = (matrix_tke_new - state.tke) / dt
    #                = ((post_source_tke + dt * transport_tend) - state.tke) / dt
    #                = (post_source_tke - state.tke) / dt + transport_tend
    tendencies, surface_fluxes = vertical_diffusion_step(
        state_for_solver, params,
        exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture,
        dt, tke_exchange_coeff,
        surface_exchange=surface_exchange, surface_target=surface_target,
    )
    tke_tend_rebased = (
        tendencies.tke_tendency + (post_source_tke - state.tke) / dt
    )
    # The θ_v variance goes through the identical split (source step then
    # implicit transport), so it needs the identical rebase — without it the
    # carried variance would silently lose the source increment every step,
    # which is the same way it ended up pinned at its floor before.
    thv_var_tend_rebased = (
        tendencies.thv_var_tendency
        + (post_source_thv_var - state.thv_variance) / dt
    )
    tendencies = tendencies._replace(
        tke_tendency=tke_tend_rebased,
        thv_var_tendency=thv_var_tend_rebased,
    )
    diagnostics = diagnostics._replace(surface_fluxes=surface_fluxes)

    return tendencies, diagnostics


# ----------------------------------------------------------------------
# Helper: column-wise shear² and N², independent of K coefficients so
# they can be fed into the ECHAM analytic TKE update.
# ----------------------------------------------------------------------

@jax.jit
def _column_shear_squared(u: jnp.ndarray, v: jnp.ndarray,
                          height_full: jnp.ndarray) -> jnp.ndarray:
    """(du/dz)² + (dv/dz)² on full levels [1/s²].

    Vertical differences are between adjacent full levels; the top
    level inherits the value just below (matches
    ``compute_shear_production``'s padding convention).
    """
    dz = jnp.diff(height_full, axis=1)
    # ``height_full`` decreases with index (level 0 = top), so dz < 0;
    # squaring makes sign irrelevant.
    du_dz = jnp.diff(u, axis=1) / dz
    dv_dz = jnp.diff(v, axis=1) / dz
    s2 = du_dz * du_dz + dv_dz * dv_dz
    # Pad top: re-use the topmost interior gradient.
    return jnp.concatenate([s2[:, :1], s2], axis=1)


@jax.jit
def _column_buoyancy_freq_squared(temperature: jnp.ndarray,
                                  height_full: jnp.ndarray) -> jnp.ndarray:
    """N² = (g/T) · (dθ/dz) approximated as (g/T) · (dT/dz + g/cp) [1/s²].

    Positive when stably stratified (the warmer-above lapse). Matches
    the sign convention used in ``compute_buoyancy_production``.
    """
    dz = jnp.diff(height_full, axis=1)
    dT_dz = jnp.diff(temperature, axis=1) / dz
    dT_dz_full = jnp.concatenate([dT_dz[:, :1], dT_dz], axis=1)
    lapse = c.grav / c.cpd
    return (c.grav / temperature) * (dT_dz_full + lapse)


def _column_thv_gradient(temperature: jnp.ndarray,
                         pressure_full: jnp.ndarray,
                         qv: jnp.ndarray,
                         qc: jnp.ndarray,
                         qi: jnp.ndarray,
                         height_full: jnp.ndarray) -> jnp.ndarray:
    """∂θ_v/∂z [K/m], the source gradient of the θ_v-variance budget.

    ECHAM ``vdiff.f90``:

        zteta1    = T * (p0/p)**kappa
        ztvir1    = zteta1 * (1 + vtmpc1*q - x)          (x = qc + qi)
        zthvirdif = (ztvir1(jk) - ztvir1(jk+1)) / zhh(jk) * grav

    where ``zhh`` is the geopotential thickness, so the ``* grav`` converts
    it to a per-metre gradient. Condensate loading (``- x``) is part of the
    reference definition and is kept: it is what makes a cloud-topped
    boundary layer's variance differ from a clear one.

    Index 0 is the model top and ``nlev-1`` the surface, so a forward
    difference along the level axis is ``(upper - lower)`` and dz is
    negative-definite going down; taking the difference of both in the same
    direction gives the right sign either way.
    """
    theta = temperature * (c.p0 / pressure_full) ** c.akap
    theta_v = theta * (1.0 + c.vtmpc1 * qv - (qc + qi))
    dz = jnp.diff(height_full, axis=1)
    dthv_dz = jnp.diff(theta_v, axis=1) / dz
    # Repeat the topmost interior value so the result is (ncol, nlev), the
    # same convention ``_column_buoyancy_freq_squared`` uses.
    return jnp.concatenate([dthv_dz[:, :1], dthv_dz], axis=1)


@jax.jit
def vertical_diffusion_scheme(
    u: jnp.ndarray,
    v: jnp.ndarray,
    temperature: jnp.ndarray,
    qv: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    pressure_full: jnp.ndarray,
    pressure_half: jnp.ndarray,
    geopotential: jnp.ndarray,
    height_full: jnp.ndarray,
    height_half: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_fraction: jnp.ndarray,
    roughness_length: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    tke: jnp.ndarray,
    thv_variance: jnp.ndarray,
    dt: float,
    params: VDiffParameters
) -> Tuple[VDiffTendencies, VDiffDiagnostics]:
    """Run vertical diffusion scheme interface.
    
    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        temperature: Temperature [K] (ncol, nlev)
        qv: Water vapor mixing ratio [kg/kg] (ncol, nlev)
        qc: Cloud water mixing ratio [kg/kg] (ncol, nlev)
        qi: Cloud ice mixing ratio [kg/kg] (ncol, nlev)
        pressure_full: Full level pressure [Pa] (ncol, nlev)
        pressure_half: Half level pressure [Pa] (ncol, nlev+1)
        geopotential: Geopotential [m²/s²] (ncol, nlev)
        height_full: Full level height [m] (ncol, nlev)
        height_half: Half level height [m] (ncol, nlev+1)
        surface_temperature: Surface temperature [K] (ncol, nsfc_type)
        surface_fraction: Surface type fraction [-] (ncol, nsfc_type)
        roughness_length: Roughness length [m] (ncol, nsfc_type)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        thv_variance: Variance of theta_v [K²] (ncol, nlev)
        dt: Time step [s]
        params: Vertical diffusion parameters
        
    Returns:
        Tuple of (tendencies, diagnostics)

    """
    # Prepare state
    state = prepare_vertical_diffusion_state(
        u, v, temperature, qv, qc, qi,
        pressure_full, pressure_half, geopotential,
        height_full, height_half,
        surface_temperature, surface_fraction, roughness_length,
        ocean_u, ocean_v, tke, thv_variance
    )
    
    # Compute vertical diffusion
    tendencies, diagnostics = vertical_diffusion_column(state, params, dt)
    
    return tendencies, diagnostics


# Vectorized version for multiple columns
vertical_diffusion_scheme_vectorized = jax.vmap(
    vertical_diffusion_scheme,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
    out_axes=(0, 0)
)


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.diagnostics.moist_air_state import advance_thermo_run  # noqa: E402
from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (  # noqa: E402
    VerticalDiffusionData,
)
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class TteTkeVerticalDiffusion(PhysicsTerm):
    """TKE-based ECHAM vertical-diffusion / boundary-layer term.

    Wraps :func:`vertical_diffusion_column` (already column-batched, no
    per-column vmap needed). Reads pressure / height diagnostics from
    the moist-air dict, surface temperature / roughness from the legacy
    ``"surface"`` key, sea-ice / land-temp / soil-water from forcing,
    ``fmask`` from terrain. Builds the 3-tile (water/ice/land) per-column
    fractions, temperatures, roughness (water uses the Charnock-derived
    heat roughness ``exp(2 - 86 z0^0.375)``), and surface wetness inline.

    Owns the whole turbulent column, ECHAM-style: the per-tile exchange
    velocities are computed before the implicit solve and enter it as the
    bottom-row surface Robin boundary condition for u/v/T/qv, so the term's
    tendencies carry the surface fluxes (drag, sensible/latent heat,
    evaporation) as well as the interior mixing. The delivered fluxes are
    diagnosed from the implicit solution (reported == delivered ==
    column-integrated tendency, the ECHAM ``pev_vdiff`` identity) and
    exported as ``surface_evaporation`` / ``surface_sensible_heat`` /
    ``surface_latent_heat`` / ``surface_stress_u/v``, which the downstream
    ``EchamSurface`` term republishes as the public ``"surface"`` fluxes.

    Reads the previous-step TKE from
    ``diagnostics["vertical_diffusion"].tke`` and writes the updated
    TKE / km / kh / surface exchange coefs / delivered surface fluxes /
    PBL height / friction_velocity back to the public
    ``"vertical_diffusion"`` key. The 0.01 m²/s² TKE clamp matches ECHAM's
    lower bound; without it the coefficient cascade diverges in the upper
    troposphere.
    """

    name: ClassVar[str] = "tte_tke_vertical_diffusion"
    category: ClassVar[str] = "vertical_diffusion"
    # ``vertical_diffusion`` is read for the previous step's TKE — that
    # comes from prev_physics_data, not a same-step upstream term, so it
    # is intentionally not in ``requires``.
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half",
        "height_full", "height_half",
        "surface",
    )
    provides: ClassVar[tuple[str, ...]] = ("vertical_diffusion",)
    # The structural shape comes from the declarative slot; the TKE
    # field gets a non-zero seed in :meth:`initial_carry_state` below.
    carry_slots: ClassVar[dict[str, type]] = {
        "vertical_diffusion": VerticalDiffusionData,
    }

    def __init__(self, params: VDiffParameters | None = None):
        """Hold the scheme-native :class:`VDiffParameters`."""
        self.params = nnx.Param(params or VDiffParameters.default())

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """``qc`` / ``qi`` are mixed by the diffusion solver."""
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
        )

    def initial_carry_state(self, coords) -> dict:
        """Seed the previous-step TKE at the ECHAM floor (0.01 m²/s²).

        ``compute_mixing_length`` and the TKE budget update use the
        carried TKE on every step. Starting from zero would let the
        first step's diffusion coefficients fall to floor everywhere
        and overshoot once turbulence reactivates. Setting the seed at
        the ECHAM lower bound matches the in-loop clamp and gives the
        spin-up step a starting reservoir that the analytic source
        update can build on.
        """
        carry = super().initial_carry_state(coords)
        nlev, ncols = carry["vertical_diffusion"].tke.shape
        carry["vertical_diffusion"] = carry["vertical_diffusion"].copy(
            tke=jnp.full((nlev, ncols), 0.01),
            # θ_v variance seeds at ECHAM's ``ztkemin`` rather than the TKE
            # floor: it is a variance in K², and its budget builds it up
            # from the ambient gradient within the first few steps. Seeding
            # it high would hand the convective trigger a large spurious
            # ``zlift`` on step 0.
            thv_variance=jnp.full((nlev, ncols), 1.0e-10),
        )
        return carry

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute vdiff tendencies and update ``vertical_diffusion``."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        pressure_half = diagnostics["pressure_half"]
        height_full = diagnostics["height_full"]
        height_half = diagnostics["height_half"]

        prev_vdiff = diagnostics.get(
            "vertical_diffusion",
            VerticalDiffusionData.zeros((ncols,), nlev),
        )
        tke = prev_vdiff.tke
        if tke.ndim == 3:
            tke = tke.reshape(nlev, ncols)
        # Carried from the previous step exactly like TKE (ECHAM keeps
        # ``pthvvar`` in the restart file). This used to be re-zeroed every
        # step, which made the variance non-prognostic in practice: its
        # source/dissipation balance never had more than one step to build
        # up, so it sat at its floor and could not be used for anything —
        # the reason the convective ``zlift`` had to read a constant.
        thv_variance = prev_vdiff.thv_variance
        if thv_variance.ndim == 3:
            thv_variance = thv_variance.reshape(nlev, ncols)

        # Surface tile fractions: 0=water, 1=sea-ice, 2=land.
        nsfc_type = 3
        land_fraction = terrain.fmask.reshape(ncols)
        sea_ice_fraction = jnp.clip(
            forcing.sice_am.reshape(ncols), 0.0, 1.0 - land_fraction,
        )
        water_fraction = 1.0 - land_fraction - sea_ice_fraction
        surface_fraction = jnp.zeros((ncols, nsfc_type))
        surface_fraction = surface_fraction.at[:, 0].set(water_fraction)
        surface_fraction = surface_fraction.at[:, 1].set(sea_ice_fraction)
        surface_fraction = surface_fraction.at[:, 2].set(land_fraction)

        # Per-tile surface temperature: SST for water, min(SST, ctfreez)
        # for ice (saline freezing point, ECHAM iniphy.f90:71), stl_am
        # for land.
        surface_in = diagnostics["surface"]
        # Water-tile temperature straight from the SST FORCING, not the
        # blended ``surface.surface_temperature`` (which is snapped to
        # one-or-the-other in mixed coastal cells — with fmask > 0.5 the
        # residual ocean fraction would exchange with the LAND
        # temperature through the new Robin delivery row, corrupting
        # coastal fluxes; Codex review on #555). Same per-tile sources
        # as EchamSurface's rebuild.
        sst_col = forcing.sea_surface_temperature.reshape(ncols)
        land_temp_col = forcing.stl_am.reshape(ncols)
        ctfreez = 271.38
        ice_temp_col = jnp.where(
            sea_ice_fraction > 0.0,
            jnp.minimum(sst_col, ctfreez),
            sst_col,
        )
        surface_temperature = jnp.stack(
            [sst_col, ice_temp_col, land_temp_col], axis=1,
        )

        roughness_length_col = surface_in.roughness_length.reshape(ncols)
        roughness = jnp.stack([
            jnp.full(ncols, 1e-4),
            jnp.full(ncols, 1e-3),
            roughness_length_col,
        ], axis=1)

        # Ocean heat roughness via the ECHAM kB⁻¹ relationship
        # z0h = z0m·exp(2 − 86·z0m^0.375) (mo_surface_ocean). With z0m = 1e-4 m
        # this gives z0h ≈ 4.9e-5 m, just below the momentum roughness. The
        # ``z0m·`` prefactor is essential: the bare ``exp(2 − 86·z0m^0.375)``
        # returns ≈0.49 m — an unphysically large ocean heat roughness (z0h ≫
        # z0m) that corrupts the ECHAM-Louis neutral heat/moisture exchange.
        z0_water = roughness[:, 0] * jnp.exp(2.0 - 86.0 * roughness[:, 0] ** 0.375)
        z0_ice = roughness[:, 1]
        z0_land = roughness[:, 2]
        roughness_heat = jnp.stack([z0_water, z0_ice, z0_land], axis=1)

        soilw_col = jnp.clip(forcing.soilw_am.reshape(ncols), 0.0, 1.0)
        surface_wetness = jnp.stack([
            jnp.ones(ncols),
            jnp.ones(ncols),
            soilw_col,
        ], axis=1)

        ocean_u = jnp.zeros(ncols)
        ocean_v = jnp.zeros(ncols)

        qc = state.tracers.get("qc", jnp.zeros_like(state.temperature))
        qi = state.tracers.get("qi", jnp.zeros_like(state.temperature))

        vdiff_state = prepare_vertical_diffusion_state(
            u=state.u_wind.T,
            v=state.v_wind.T,
            temperature=state.temperature.T,
            qv=state.specific_humidity.T,
            qc=qc.T,
            qi=qi.T,
            pressure_full=pressure_full.T,
            pressure_half=pressure_half.T,
            geopotential=state.geopotential.T,
            height_full=height_full.T,
            height_half=height_half.T,
            surface_temperature=surface_temperature,
            surface_fraction=surface_fraction,
            roughness_length=roughness,
            roughness_heat=roughness_heat,
            surface_wetness=surface_wetness,
            ocean_u=ocean_u,
            ocean_v=ocean_v,
            tke=tke.T,
            thv_variance=thv_variance.T,
        )

        vdiff_tendencies, vdiff_diagnostics = vertical_diffusion_column(
            vdiff_state, params, dt,
        )

        u_tend = vdiff_tendencies.u_tendency.T
        v_tend = vdiff_tendencies.v_tendency.T
        temp_tend = vdiff_tendencies.temperature_tendency.T
        qv_tend = vdiff_tendencies.qv_tendency.T
        qc_tend = vdiff_tendencies.qc_tendency.T
        qi_tend = vdiff_tendencies.qi_tendency.T
        tke_tend = vdiff_tendencies.tke_tendency.T
        thv_var_tend = vdiff_tendencies.thv_var_tendency.T

        km = vdiff_diagnostics.exchange_coeff_momentum.T
        kh = vdiff_diagnostics.exchange_coeff_heat.T
        pbl_height = vdiff_diagnostics.boundary_layer_height
        u_star = vdiff_diagnostics.friction_velocity

        # Per-tile surface exchange velocities (CH·|U|, CE·|U|, CM·|U|, all
        # m/s) from the configured surface-layer scheme. The momentum
        # coefficient is now a real CM·|U| (Louis/Businger drag), not the
        # interior diffusivity Km[lowest] (m²/s) it used to be tiled from —
        # that mismatch made the surface-stress implicit-damping factor in the
        # ``echam_surface`` term dimensionally wrong.
        surface_exchange_heat = vdiff_diagnostics.surface_exchange_heat
        surface_exchange_moisture = vdiff_diagnostics.surface_exchange_moisture
        surface_exchange_momentum = vdiff_diagnostics.surface_exchange_momentum

        # Delivered surface fluxes from the implicit surface-coupled solve
        # (diagnosed from the implicit solution, §1.8 of the ECHAM map:
        # reported == delivered == column-integrated tendency, exactly).
        # ``EchamSurface`` republishes these as the public "surface" fluxes.
        sfc_fluxes = vdiff_diagnostics.surface_fluxes

        # ``tke`` here is the *post-source* TKE (the analytic ECHAM-style
        # implicit update done in ``vertical_diffusion_column``);
        # ``tke_tend`` is purely the matrix-solver transport tendency.
        # The closed-form source step is unconditionally non-negative
        # and bounded by the production/dissipation equilibrium, so the
        # standard 0.01 m²/s² floor is the only safeguard needed here.
        new_tke = jnp.maximum(tke + dt * tke_tend, 0.01)
        # ECHAM floors pthvvar at ztkemin = 1e-10 K² after both the
        # source step and the implicit transport (vdiff.f90:860,1311).
        new_thv_var = jnp.maximum(thv_variance + dt * thv_var_tend, 1.0e-10)
        # pthvsig = SQRT(pthvvar(klev-1)) — the SECOND-lowest full level,
        # not the lowest (vdiff.f90:1338). Levels here run top-first, so
        # klev-1 is index -2.
        new_thv_sigma = jnp.sqrt(new_thv_var[-2])

        tendency = PhysicsTendency(
            u_wind=u_tend,
            v_wind=v_tend,
            temperature=temp_tend,
            specific_humidity=qv_tend,
            tracers={"qc": qc_tend, "qi": qi_tend},
        )

        vdiff_out = prev_vdiff.copy(
            tke=new_tke,
            thv_variance=new_thv_var,
            thv_sigma=new_thv_sigma,
            km=km,
            kh=kh,
            # Same-step moisture-tendency profile for the Tiedtke zdqpbl
            # closure (ECHAM's pqte at cucall time; convection runs after
            # this term in the ECHAM physc ordering).
            qv_tendency=qv_tend,
            surface_exchange_heat=surface_exchange_heat,
            surface_exchange_moisture=surface_exchange_moisture,
            surface_exchange_momentum=surface_exchange_momentum,
            pbl_height=pbl_height,
            surface_friction_velocity=u_star,
            surface_evaporation=sfc_fluxes.evaporation,
            surface_sensible_heat=sfc_fluxes.sensible_heat,
            surface_latent_heat=sfc_fluxes.latent_heat,
            surface_stress_u=sfc_fluxes.stress_u,
            surface_stress_v=sfc_fluxes.stress_v,
        )

        # Advance the running thermodynamic view so downstream terms
        # (Tiedtke convection, then the cloud microphysics) see the
        # post-vdiff (T, q) — ECHAM's ``physc`` runs vdiff before
        # ``cucall``/``cloud`` and each sees the accumulated provisional
        # state (ztp1 = ptm1 + ptte·dt). Same pattern as the convection
        # wrapper; see ``advance_thermo_run`` for the operator-split
        # tendency-ownership rules (nothing is zeroed here).
        diagnostics = advance_thermo_run(
            diagnostics,
            dt,
            d_temperature=temp_tend,
            d_specific_humidity=qv_tend,
            # Condensate view too: downstream cloud terms must see the
            # DIFFUSED qc/qi, not the step-start tracers (Codex review).
            d_qc=qc_tend,
            d_qi=qi_tend,
        )

        return tendency, {**diagnostics, "vertical_diffusion": vdiff_out}