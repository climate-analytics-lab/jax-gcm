"""Main vertical diffusion scheme for ECHAM physics.

This module provides the main interface for vertical diffusion and boundary layer
physics, integrating turbulence coefficient calculations with the matrix solver.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
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

    # Approximate dry air mass (could be more sophisticated)
    dry_air_mass = air_mass * (1.0 - qv)

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
        dry_air_mass=dry_air_mass,
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


@jax.jit
def vertical_diffusion_column(
    state: VDiffState,
    params: VDiffParameters,
    dt: float
) -> Tuple[VDiffTendencies, VDiffDiagnostics]:
    """Compute vertical diffusion for a single column.
    
    Args:
        state: Vertical diffusion state
        params: Vertical diffusion parameters
        dt: Time step [s]
        
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

    # Step 2: matrix solver for vertical transport, with the post-source
    # TKE as input. Build a shallow-copied state so we don't mutate the
    # caller-owned ``state`` and so other variables still see the original
    # ``state.tke`` for their own coupling (if any).
    state_for_solver = state._replace(tke=post_source_tke)

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

    diagnostics = compute_turbulence_diagnostics(
        state_for_solver, params, exchange_coeff_momentum,
        exchange_coeff_heat, exchange_coeff_moisture,
    )

    # The matrix solver returns ``tke_tendency = (matrix_tke_new -
    # state_for_solver.tke) / dt``. Since the caller computes
    # ``new_tke = state.tke + dt * tke_tendency`` against the *original*
    # (raw, pre-source) ``state.tke``, we rewrite ``tke_tendency`` to be
    # in those reference units before returning so the caller's formula
    # recovers ``matrix_tke_new`` directly. Equivalent rewrite:
    #   new_tke_tend = (matrix_tke_new - state.tke) / dt
    #                = ((post_source_tke + dt * transport_tend) - state.tke) / dt
    #                = (post_source_tke - state.tke) / dt + transport_tend
    tendencies = vertical_diffusion_step(
        state_for_solver, params,
        exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture,
        dt, tke_exchange_coeff,
    )
    tke_tend_rebased = (
        tendencies.tke_tendency + (post_source_tke - state.tke) / dt
    )
    tendencies = tendencies._replace(tke_tendency=tke_tend_rebased)

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

    Reads the previous-step TKE from
    ``diagnostics["vertical_diffusion"].tke`` and writes the updated
    TKE / km / kh / surface exchange coefs / PBL height /
    friction_velocity back to the public ``"vertical_diffusion"`` key.
    The 0.01 m²/s² TKE clamp matches ECHAM's lower bound; without it the
    coefficient cascade diverges in the upper troposphere.
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
        thv_variance = jnp.zeros((nlev, ncols))

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
        sst_col = surface_in.surface_temperature.reshape(ncols)
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

        z0_water = jnp.exp(2.0 - 86.0 * roughness[:, 0] ** 0.375)
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

        km = vdiff_diagnostics.exchange_coeff_momentum.T
        kh = vdiff_diagnostics.exchange_coeff_heat.T
        pbl_height = vdiff_diagnostics.boundary_layer_height
        u_star = vdiff_diagnostics.friction_velocity

        surface_exchange_heat = vdiff_diagnostics.surface_exchange_heat
        surface_exchange_moisture = (
            vdiff_diagnostics.surface_exchange_moisture
        )
        surface_exchange_momentum = jnp.repeat(
            vdiff_diagnostics.exchange_coeff_momentum[:, -1:],
            nsfc_type, axis=1,
        )

        # ``tke`` here is the *post-source* TKE (the analytic ECHAM-style
        # implicit update done in ``vertical_diffusion_column``);
        # ``tke_tend`` is purely the matrix-solver transport tendency.
        # The closed-form source step is unconditionally non-negative
        # and bounded by the production/dissipation equilibrium, so the
        # standard 0.01 m²/s² floor is the only safeguard needed here.
        new_tke = jnp.maximum(tke + dt * tke_tend, 0.01)

        tendency = PhysicsTendency(
            u_wind=u_tend,
            v_wind=v_tend,
            temperature=temp_tend,
            specific_humidity=qv_tend,
            tracers={"qc": qc_tend, "qi": qi_tend},
        )

        vdiff_out = prev_vdiff.copy(
            tke=new_tke,
            km=km,
            kh=kh,
            surface_exchange_heat=surface_exchange_heat,
            surface_exchange_moisture=surface_exchange_moisture,
            surface_exchange_momentum=surface_exchange_momentum,
            pbl_height=pbl_height,
            surface_friction_velocity=u_star,
        )

        return tendency, {**diagnostics, "vertical_diffusion": vdiff_out}