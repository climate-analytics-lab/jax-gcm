"""Main vertical diffusion scheme for ECHAM physics.

This module provides the main interface for vertical diffusion and boundary layer
physics, integrating turbulence coefficient calculations with the matrix solver.
"""

import functools
import logging

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from jcm.forcing import land_snow_cover
from jcm.physics.surface.echam.albedo import CTFREEZ
from jcm.physics.thermodynamics import (
    saturation_specific_humidity,
    saturation_specific_humidity_and_derivative,
)
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffTendencies, VDiffDiagnostics,
    LandBalanceInputs, SurfaceTiles,
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


def _default_sublimation_fraction(like: jnp.ndarray) -> jnp.ndarray:
    """Sublimating share per tile when none is given: the sea-ice tile only.

    Tile index 1 is sea ice in the water/ice/land ordering this package
    uses throughout; a single-tile state has no ice tile.
    """
    out = jnp.zeros_like(like)
    return out.at[:, 1].set(1.0) if like.shape[1] > 1 else out


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
    surface_sublimation_fraction: jnp.ndarray = None,
    surface_cair: jnp.ndarray = None,
    surface_csat: jnp.ndarray = None,
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
        surface_sublimation_fraction: Fraction of each tile's potential
            evaporation that sublimates (ncol, nsfc_type), see
            :class:`VDiffState`. When ``None``: 1 for the sea-ice tile
            (index 1 of the water/ice/land ordering) and 0 elsewhere.
        surface_cair, surface_csat: JSBACH's humidity factors per tile
            (ncol, nsfc_type), see :class:`VDiffState`. ``None``: both
            ``surface_wetness``.

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
    if surface_sublimation_fraction is None:
        surface_sublimation_fraction = _default_sublimation_fraction(
            roughness_length)
    if surface_cair is None:
        surface_cair = surface_wetness
    if surface_csat is None:
        surface_csat = surface_wetness

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
        ocean_v=ocean_v,
        surface_sublimation_fraction=surface_sublimation_fraction,
        surface_cair=surface_cair,
        surface_csat=surface_csat,
    )


@functools.partial(jax.jit, static_argnames=("couple_surface",))
def vertical_diffusion_column(
    state: VDiffState,
    params: VDiffParameters,
    dt: float,
    couple_surface: bool = True,
    land: LandBalanceInputs = None,
) -> Tuple[VDiffTendencies, VDiffDiagnostics]:
    """Compute vertical diffusion for a single column.

    By default the implicit solve carries the ECHAM surface exchange: the
    per-tile exchange velocities are computed *before* the matrix step;
    momentum couples through the fraction-weighted Robin bottom row, heat and
    moisture tile by tile through ECHAM's Richtmyer–Morton relations
    (:func:`~.matrix_solver.couple_surface_tiles`), and the delivered surface
    fluxes are diagnosed from the implicit solution (reported == delivered by
    construction — ECHAM's ``pev_vdiff`` identity). With ``land`` given, the
    land tile's skin temperature is solved with the lowest level instead of
    prescribed. Set ``couple_surface=False`` to run the interior-only
    operator with zero-flux (insulating, free-slip) boundaries — used by the
    forced-flux mode and by tests that pin the interior diffusion.

    Args:
        state: Vertical diffusion state
        params: Vertical diffusion parameters
        dt: Time step [s]
        couple_surface: Static flag — include the surface coupling (default).
        land: Inputs of the land skin energy balance, or ``None`` for a
            prescribed land temperature. Its ``saturation_slope`` is filled
            here, at the surface pressure the tiles' saturation uses.

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
        # === Surface coupling =================================================
        # Momentum: one fraction-weighted drag against the surface current,
        # ECHAM's box-averaged ``cdum`` (mo_surface.f90:1205-1219).
        # Heat and moisture: tile by tile (richtmyer_land/_ocean/_ice and
        # blend_zq_zt, see couple_surface_tiles). Each tile's moisture flux is
        # ρ·C·(csat·q_s − cair·q̂_K): cair = csat = 1 over water and ice, the
        # JSBACH factors over land. The surface saturation is ECHAM's ``ua``
        # table (Sonntag over ice at and below tmelt, over water above).
        # =====================================================================
        frac = state.surface_fraction
        c_mom = jnp.sum(frac * diagnostics.surface_exchange_momentum, axis=1)
        surface_momentum = (c_mom, state.ocean_u, state.ocean_v)
        cair = state.surface_cair if state.surface_cair is not None else state.surface_wetness
        csat = state.surface_csat if state.surface_csat is not None else state.surface_wetness
        sub = state.surface_sublimation_fraction
        if sub is None:
            sub = _default_sublimation_fraction(frac)
        p_sfc = state.pressure_half[:, -1]
        qsat_tiles = saturation_specific_humidity(
            state.surface_temperature, p_sfc[:, None],
        )
        if land is not None:
            _, dqs = saturation_specific_humidity_and_derivative(
                land.temperature, p_sfc)
            land = land._replace(saturation_slope=dqs)
        surface_tiles = SurfaceTiles(
            fraction=frac,
            exchange_heat=diagnostics.surface_exchange_heat,
            exchange_moisture=diagnostics.surface_exchange_moisture,
            cair=jnp.clip(cair, 0.0, 1.0),
            csat=jnp.clip(csat, 0.0, 1.0),
            temperature=state.surface_temperature,
            saturation_humidity=qsat_tiles,
            sublimation_fraction=sub,
            land=land,
        )
    else:
        surface_momentum = None
        surface_tiles = None

    # The matrix solver returns ``tke_tendency = (matrix_tke_new -
    # state_for_solver.tke) / dt``. Since the caller computes
    # ``new_tke = state.tke + dt * tke_tendency`` against the *original*
    # (raw, pre-source) ``state.tke``, we rewrite ``tke_tendency`` to be
    # in those reference units before returning so the caller's formula
    # recovers ``matrix_tke_new`` directly. Equivalent rewrite:
    #   new_tke_tend = (matrix_tke_new - state.tke) / dt
    #                = ((post_source_tke + dt * transport_tend) - state.tke) / dt
    #                = (post_source_tke - state.tke) / dt + transport_tend
    tendencies, surface_fluxes, land_balance = vertical_diffusion_step(
        state_for_solver, params,
        exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture,
        dt, tke_exchange_coeff,
        surface_momentum=surface_momentum, surface_tiles=surface_tiles,
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
    diagnostics = diagnostics._replace(surface_fluxes=surface_fluxes,
                                       land_balance=land_balance)

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
from jcm.physics.radiation import SURFACE_OPTICS_KEY  # noqa: E402
from jcm.physics.surface.echam import jsbach_land  # noqa: E402
from jcm.physics.surface.echam.jsbach_land import JsbachLandParameters  # noqa: E402
from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (  # noqa: E402
    VerticalDiffusionData,
)
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


_LOGGER = logging.getLogger(__name__)
_LOGGED_NO_SOILW_REL = False


def _upper_layer_fill(forcing: ForcingData):
    """Return the bare soil's upper-layer fill: ``soilw_rel``, else ``soilw_am``.

    ``soilw_rel`` (ERA5 swvl1 over the field capacity of the 0-7 cm layer) is
    the quantity ECHAM6.3's 5-layer soil gives ``calc_relative_humidity_upper``;
    a bundle built before it existed carries only ``soilw_am``, which stands in
    (the two agree within a few per cent in most land boxes). Logged once.
    """
    global _LOGGED_NO_SOILW_REL
    if forcing.soilw_rel is not None:
        return forcing.soilw_rel
    if not _LOGGED_NO_SOILW_REL:
        _LOGGER.warning(
            "forcing has no soilw_rel (a bundle built before #787): the bare-soil "
            "humidity of the ECHAM land tile uses soilw_am instead")
        _LOGGED_NO_SOILW_REL = True
    return forcing.soilw_am


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

    The land tile is JSBACH's prescribed-moisture land (#979;
    :mod:`jcm.physics.surface.echam.jsbach_land`): its moisture flux carries
    JSBACH's humidity factors ``cair``/``csat``, and its skin temperature
    (``surface.land_surface_temperature``, carried) is solved with the lowest
    level from the land energy balance against the prescribed soil
    temperature ``stl_am``. The term writes the land tile's balance and
    factors onto the ``"surface"`` diagnostics.

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
    provides: ClassVar[tuple[str, ...]] = ("vertical_diffusion", "surface")
    # The structural shape comes from the declarative slot; the TKE
    # field gets a non-zero seed in :meth:`initial_carry_state` below.
    carry_slots: ClassVar[dict[str, type]] = {
        "vertical_diffusion": VerticalDiffusionData,
    }

    def __init__(self, params: VDiffParameters | None = None,
                 couple_surface: bool = True,
                 land_params: JsbachLandParameters | None = None):
        """Hold the scheme-native :class:`VDiffParameters`.

        Args:
            params: Scheme parameters (defaults to ECHAM values).
            land_params: Constants of the JSBACH land tile
                (:class:`~jcm.physics.surface.echam.jsbach_land.JsbachLandParameters`),
                differentiable leaves.
            couple_surface: Static flag forwarded to
                :func:`vertical_diffusion_column`. ``True`` (default): the
                implicit solve carries the surface exchange as its
                bottom-row Robin BC and the delivered fluxes are diagnosed
                from the solution. ``False``: interior-only diffusion with
                insulating / free-slip bottom boundaries — the forced
                surface mode (jax-gcm#301), where a downstream
                :class:`~jcm.physics.surface.prescribed_flux.
                PrescribedSurfaceFlux` term delivers externally prescribed
                fluxes explicitly instead (the exact replacement seam: the
                solve's surface coupling is OFF, so no double counting).

        """
        self.params = nnx.Param(params or VDiffParameters.default())
        self.couple_surface = couple_surface
        self.land_params = nnx.Param(land_params or JsbachLandParameters.default())

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
        # for ice (saline freezing point, ECHAM iniphy.f90:71), the land
        # skin temperature for land.
        surface_in = diagnostics["surface"]
        # Water-tile temperature straight from the SST FORCING, not the
        # blended ``surface.surface_temperature`` (which is snapped to
        # one-or-the-other in mixed coastal cells — with fmask > 0.5 the
        # residual ocean fraction would exchange with the LAND
        # temperature, corrupting coastal fluxes; Codex review on #555).
        # Same per-tile sources as EchamSurface.
        sst_col = forcing.sea_surface_temperature.reshape(ncols)
        stl_col = forcing.stl_am.reshape(ncols)
        # The land skin temperature is prognostic (#979), carried in
        # ``surface.land_surface_temperature`` and seeded from stl_am when
        # unset (<= 0) by EchamBoundaryConditions; the same rule here keeps a
        # composition without that term well defined.
        carried = getattr(surface_in, "land_surface_temperature", None)
        if carried is None:
            land_temp_col = stl_col
        else:
            carried = carried.reshape(ncols)
            land_temp_col = jnp.where(carried > 0.0, carried, stl_col)
        ice_temp_col = jnp.where(
            sea_ice_fraction > 0.0,
            jnp.minimum(sst_col, CTFREEZ),
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

        # Snow-covered share of the land: the prescribed ``snowc_am`` cover
        # of the non-glacier land plus the glaciers, fully snow covered as
        # JSBACH sets them (#672; convention on ``ForcingData``).
        glac_col = (jnp.zeros(ncols) if forcing.glacier_fraction is None
                    else forcing.glacier_fraction.reshape(ncols))
        snowc_col = jnp.clip(forcing.snowc_am.reshape(ncols), 0.0, 1.0)
        snow_col = land_snow_cover(snowc_col, glac_col)

        # === Land tile: JSBACH's humidity factors (#979) ======================
        # Prescribed-moisture land: soilw_am is the root-zone fill ws/wsmx the
        # water-stress factor reads, soilw_rel (ERA5 swvl1 over its field
        # capacity) the upper-layer fill ECHAM6.3's 5-layer soil gives the
        # bare-soil humidity. Vegetated fraction = the forest fraction.
        land_p = self.land_params.get_value()
        w_root = jnp.clip(forcing.soilw_am.reshape(ncols), 0.0, 1.0)
        w_upper = jnp.clip(_upper_layer_fill(forcing).reshape(ncols), 0.0, 1.0)
        veg_col = (jnp.zeros(ncols) if forcing.forest_fraction is None
                   else jnp.clip(forcing.forest_fraction.reshape(ncols), 0.0, 1.0))
        radiation = diagnostics.get("radiation")
        optics = diagnostics.get(SURFACE_OPTICS_KEY)
        zeros = jnp.zeros(ncols)
        sw_down = zeros if radiation is None else radiation.surface_sw_down.reshape(ncols)
        lw_down = zeros if radiation is None else radiation.surface_lw_down.reshape(ncols)
        if optics is not None and "land_albedo" in optics:
            land_albedo = optics["land_albedo"].reshape(ncols)
            land_emissivity = optics["land_emissivity"].reshape(ncols)
        else:
            from jcm.physics.forcing.echam_boundary_conditions import (
                SurfaceOpticsParameters,
            )
            land_albedo = jnp.clip(jnp.asarray(forcing.alb0).reshape(ncols), 0.0, 1.0)
            land_emissivity = jnp.full(ncols, SurfaceOpticsParameters().land_emissivity)
        land_sw_net = sw_down * (1.0 - land_albedo)
        canopy_conductance = jsbach_land.unstressed_canopy_conductance(
            land_p.leaf_area_index, land_p.par_fraction * land_sw_net, land_p)
        # The canopy factor reads the land tile's exchange velocity C_h·|U|;
        # like ECHAM's zchl it is the previous step's (ECHAM's factors are the
        # ones update_soil formed at the end of the previous step, and this
        # step's coefficients depend on them through the surface-layer
        # humidity). A cold start has none: see below.
        prev_exchange = jnp.asarray(prev_vdiff.surface_exchange_heat).reshape(ncols, nsfc_type)
        p_sfc = pressure_half[-1].reshape(ncols)
        q_sat_land = saturation_specific_humidity(land_temp_col, p_sfc)
        bare_h = jsbach_land.bare_soil_relative_humidity(w_upper)
        cair_land, csat_land, stress = jsbach_land.humidity_factors(
            bare_h, w_root, snowc_col, glac_col, veg_col, canopy_conductance,
            prev_exchange[:, 2], state.specific_humidity[-1].reshape(ncols), q_sat_land,
            land_p)
        # ECHAM starts the land without evaporation: a run that is not a
        # restart sets zcair = zcsat = 0 (mo_surface.f90::init_surface), and
        # update_soil forms the first factors at the end of that step. A carry
        # without a land exchange velocity is that first step.
        cold_start = prev_exchange[:, 2] <= 0.0
        cair_land = jnp.where(cold_start, 0.0, cair_land)
        csat_land = jnp.where(cold_start, 0.0, csat_land)
        # The land constants are float64 leaves under x64; the land tile keeps
        # the state's precision.
        dtype = state.temperature.dtype
        cair_land, csat_land, stress, bare_h, canopy_conductance = (
            x.astype(dtype) for x in (cair_land, csat_land, stress, bare_h, canopy_conductance))
        ones = jnp.ones(ncols, dtype)
        surface_cair = jnp.stack([ones, ones, cair_land], axis=1)
        surface_csat = jnp.stack([ones, ones, csat_land], axis=1)

        # === Land tile: skin energy balance (#979) ============================
        # Solved with the lowest level in the implicit solve whenever the
        # surface is coupled and there is radiation to drive it; otherwise the
        # land keeps the prescribed soil temperature.
        land_inputs = None
        if self.couple_surface and radiation is not None:
            heat_capacity, conductance = (
                x.astype(dtype) for x in jsbach_land.top_layer_thermal_properties(
                    snowc_col, glac_col, land_p))
            # Snow deeper than ECHAM's critical depth (the bundle's own
            # snowc = SWE/60 mm), or any glacier, holds the skin at tmelt.
            melt_capped = (glac_col > 0.0) | (
                (1.0 - glac_col) * snowc_col * land_p.full_cover_snow_water_equivalent
                > land_p.critical_snow_depth)
            land_inputs = LandBalanceInputs(
                temperature=land_temp_col,
                soil_temperature=stl_col,
                saturation_slope=zeros,   # filled by vertical_diffusion_column
                net_shortwave=land_sw_net,
                longwave_down=lw_down,
                emissivity=land_emissivity,
                heat_capacity=heat_capacity,
                conductance=conductance,
                melt_capped=melt_capped,
                params=land_p,
            )

        # Share of each tile's potential evaporation that sublimates (sets
        # the reported latent heat only): all of it over sea ice, the
        # snow-covered part over land.
        surface_sublimation_fraction = jnp.stack([
            jnp.zeros(ncols),
            jnp.ones(ncols),
            snow_col,
        ], axis=1)

        # Zero ocean current: the stress is against a surface at rest.
        # ``ForcingData.ocean_u/ocean_v`` are reserved for a coupled current
        # but are not read here yet (#915, docs/source/design/surface_exchange.md).
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
            surface_wetness=surface_csat,
            surface_sublimation_fraction=surface_sublimation_fraction,
            surface_cair=surface_cair,
            surface_csat=surface_csat,
            ocean_u=ocean_u,
            ocean_v=ocean_v,
            tke=tke.T,
            thv_variance=thv_variance.T,
        )

        vdiff_tendencies, vdiff_diagnostics = vertical_diffusion_column(
            vdiff_state, params, dt, couple_surface=self.couple_surface,
            land=land_inputs,
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
        wind_10m = vdiff_diagnostics.wind_10m

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
            wind_10m=wind_10m,
            wind_10m_u=vdiff_diagnostics.wind_10m_u,
            wind_10m_v=vdiff_diagnostics.wind_10m_v,
            wind_10m_reduction=vdiff_diagnostics.wind_10m_reduction,
            wind_10m_tile=vdiff_diagnostics.wind_10m_tile,
            wind_10m_u_tile=vdiff_diagnostics.wind_10m_u_tile,
            wind_10m_v_tile=vdiff_diagnostics.wind_10m_v_tile,
            surface_fraction=surface_fraction,
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

        # The land tile onto the public surface namespace. A prescribed land
        # (forced fluxes, or no radiation) keeps stl_am and reports no balance.
        lb = vdiff_diagnostics.land_balance
        if lb is None:
            balance = {name: zeros for name in (
                "land_net_radiation", "land_sensible_heat_flux", "land_latent_heat_flux",
                "ground_heat_flux", "snow_melt_heat_flux", "land_heat_storage",
                "land_evaporation")}
            new_land_temperature = stl_col
        else:
            balance = dict(
                land_net_radiation=lb.net_radiation,
                land_sensible_heat_flux=lb.sensible_heat_flux,
                land_latent_heat_flux=lb.latent_heat_flux,
                ground_heat_flux=lb.ground_heat_flux,
                snow_melt_heat_flux=lb.melt_heat_flux,
                land_heat_storage=lb.heat_storage,
                land_evaporation=lb.evaporation,
            )
            new_land_temperature = lb.temperature
        surface_out = surface_in.copy(
            land_surface_temperature=new_land_temperature,
            cair=cair_land,
            csat=csat_land,
            water_stress_factor=stress,
            bare_soil_humidity=bare_h,
            canopy_conductance=canopy_conductance,
            **balance,
        )

        return tendency, {**diagnostics, "vertical_diffusion": vdiff_out,
                          "surface": surface_out}