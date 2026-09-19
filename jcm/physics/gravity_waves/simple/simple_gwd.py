"""Gravity wave drag parameterization for ECHAM physics

This module implements the orographic and non-orographic gravity wave drag
parameterizations. The scheme accounts for momentum deposition from breaking
gravity waves that are not resolved by the model grid.

Based on ICON's mo_gwd_wms.f90 and mo_ssodrag.f90

Features:
- Orographic gravity wave drag (mountain waves)
- Non-orographic gravity wave sources
- Wave breaking and momentum deposition
- Critical level filtering

"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
# from functools import partial  # No longer needed
import tree_math

import jcm.constants as c


@tree_math.struct
class SimpleGwdParameters:
    """Parameters for gravity wave drag scheme"""
    
    # Orographic drag parameters
    gkdrag: float           # Surface drag coefficient
    gkwake: float           # Wake drag coefficient
    grcrit: float           # Critical Froude number
    gssec: float            # Security parameter for Richardson number
    gtsec: float            # Security parameter for Brunt-Vaisala frequency
    
    # Non-orographic parameters
    ruwmax: float           # Launch momentum flux for non-orographic waves (N/m²)
    nslope: float           # Slope of wave spectrum
    
    # Wave breaking parameters
    ric: float              # Critical Richardson number
    efmin: float            # Minimum efficiency
    efmax: float            # Maximum efficiency
    
    # Numerical parameters
    zmin: float             # Minimum height for GWD (m)
    zmax: float             # Maximum height for GWD (m)
    
    # Tuning parameters
    gwdrag_cd: float        # Drag coefficient multiplier
    gwdrag_ef: float        # Efficiency factor

    @classmethod
    def default(cls, gkdrag=0.5, gkwake=0.5, grcrit=0.25, gssec=0.0001,
                 gtsec=0.0001, ruwmax=1.0, nslope=1.0, ric=0.25,
                 efmin=0.0, efmax=0.1, zmin=1000.0, zmax=100000.0,
                 gwdrag_cd=1.0, gwdrag_ef=0.05) -> 'SimpleGwdParameters':
        """Return default gravity wave parameters"""
        return cls(
            gkdrag=jnp.array(gkdrag),
            gkwake=jnp.array(gkwake),
            grcrit=jnp.array(grcrit),
            gssec=jnp.array(gssec),
            gtsec=jnp.array(gtsec),
            ruwmax=jnp.array(ruwmax),
            nslope=jnp.array(nslope),
            ric=jnp.array(ric),
            efmin=jnp.array(efmin),
            efmax=jnp.array(efmax),
            zmin=jnp.array(zmin),
            zmax=jnp.array(zmax),
            gwdrag_cd=jnp.array(gwdrag_cd),
            gwdrag_ef=jnp.array(gwdrag_ef)
        )


class SimpleGwdState(NamedTuple):
    """State variables for gravity wave calculations"""
    
    tau_x: jnp.ndarray            # Zonal momentum flux (N/m²)
    tau_y: jnp.ndarray            # Meridional momentum flux (N/m²)
    wave_stress: jnp.ndarray      # Wave stress magnitude (N/m²)
    breaking_level: jnp.ndarray   # Level where waves break
    deposited_momentum: jnp.ndarray  # Momentum deposited (m/s²)


class SimpleGwdTendencies(NamedTuple):
    """Tendencies from gravity wave drag"""
    
    dudt: jnp.ndarray             # Zonal wind tendency (m/s²)
    dvdt: jnp.ndarray             # Meridional wind tendency (m/s²)
    dtedt: jnp.ndarray            # Temperature tendency from dissipation (K/s)


def _safe_hypot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """``sqrt(x**2 + y**2)`` whose derivative is finite at the origin.

    The Euclidean norm is a cone at ``(0, 0)``: ``d|v|/dx = x/|v|`` is ``0/0``
    there, so a bare ``jnp.sqrt`` returns the correct forward value 0 and a NaN
    derivative. Both places this is used sit *exactly* on the origin in ordinary
    operation — a calm surface column has ``u = v = 0``, and the momentum flux
    is identically zero at every level a wave never reaches — so the NaN is the
    common case, not an edge case, and one calm column poisons the gradient of
    the whole batch through the reduction that follows.

    The safe-denominator double-``where`` feeds ``jnp.sqrt`` a 1.0 in exactly
    the cells the outer ``where`` discards, so the differentiated branch is
    evaluated away from the cone tip and the forward result is bit-identical:
    ``where`` selects the literal 0 there, which is what ``sqrt(0)`` returns.
    The derivative reported at the origin is then 0 — no two-sided derivative
    exists at a cone tip, and 0 is the value the surrounding physics wants,
    since a vanishing flux exerts no drag however it is approached.
    """
    square = x**2 + y**2
    positive = square > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


@jax.jit
def brunt_vaisala_frequency(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray
) -> jnp.ndarray:
    """Calculate Brunt-Väisälä frequency
    
    Args:
        temperature: Temperature profile (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        
    Returns:
        N²: Brunt-Väisälä frequency squared (s⁻²) [nlev]

    """
    theta = temperature * (c.p0 / pressure) ** (c.rd / c.cpd)
    
    # Calculate vertical gradient of potential temperature
    # Use one-sided differences at boundaries
    dtheta_dz = jnp.zeros_like(theta)
    
    # Interior points - central differences
    dtheta_dz = dtheta_dz.at[1:-1].set(
        (theta[2:] - theta[:-2]) / (height[2:] - height[:-2])
    )
    
    # Boundaries - one-sided
    dtheta_dz = dtheta_dz.at[0].set(
        (theta[1] - theta[0]) / (height[1] - height[0])
    )
    dtheta_dz = dtheta_dz.at[-1].set(
        (theta[-1] - theta[-2]) / (height[-1] - height[-2])
    )
    
    # Brunt-Väisälä frequency squared
    n2 = c.grav / theta * dtheta_dz
    
    # Apply minimum threshold for stability
    n2 = jnp.maximum(n2, 1e-8)
    
    return n2


@jax.jit
def orographic_source(
    u_sfc: jnp.ndarray,
    v_sfc: jnp.ndarray,
    n_sfc: jnp.ndarray,
    h_std: jnp.ndarray,
    config: SimpleGwdParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate orographic gravity wave source
    
    Args:
        u_sfc: Surface zonal wind (m/s)
        v_sfc: Surface meridional wind (m/s)
        n_sfc: Surface Brunt-Väisälä frequency (s⁻¹)
        h_std: Standard deviation of orography (m)
        config: GW parameters
        
    Returns:
        Tuple of (tau_x, tau_y): Surface momentum fluxes (N/m²)

    """
    # Surface wind speed. The 1 m/s floor below keeps the *forward* division by
    # ``wind_speed`` finite but does nothing for the derivative: ``maximum``
    # passes a zero cotangent back into the norm, and 0 * NaN is still NaN.
    wind_speed = _safe_hypot(u_sfc, v_sfc)
    wind_speed = jnp.maximum(wind_speed, 1.0)  # Minimum wind speed
    
    # Froude number
    froude = wind_speed / (n_sfc * h_std + 1e-10)
    
    # Wave momentum flux (simplified parameterization)
    # Based on linear mountain wave theory
    flux_magnitude = config.gkdrag * n_sfc * wind_speed * h_std**2
    
    # Apply Froude number dependence
    # Flux is reduced for high Froude numbers (flow over mountain)
    froude_factor = jnp.minimum(1.0, config.grcrit / (froude + 0.1))
    flux_magnitude = flux_magnitude * froude_factor
    
    # Project onto wind direction
    tau_x = -flux_magnitude * u_sfc / wind_speed
    tau_y = -flux_magnitude * v_sfc / wind_speed
    
    return tau_x, tau_y


@jax.jit
def wave_breaking_criterion(
    u: jnp.ndarray,
    v: jnp.ndarray,
    n2: jnp.ndarray,
    height: jnp.ndarray,
    tau_x: jnp.ndarray,
    tau_y: jnp.ndarray,
    rho: jnp.ndarray,
    config: SimpleGwdParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Determine wave breaking and momentum deposition
    
    Uses saturation hypothesis - waves break when amplitude exceeds
    critical threshold based on Richardson number criterion.
    
    Args:
        u, v: Wind components (m/s) [nlev]
        n2: Brunt-Väisälä frequency squared (s⁻²) [nlev]
        height: Height (m) [nlev]
        tau_x, tau_y: Momentum fluxes (N/m²) [nlev]
        rho: Air density (kg/m³) [nlev]
        config: Parameters
        
    Returns:
        Tuple of (breaking_mask, deposited_momentum)

    """
    nlev = u.shape[0]
    
    # Calculate wave amplitude from momentum flux
    # tau = rho * u' * w' ~ rho * c * a²
    # where c is phase speed and a is amplitude
    # Identically zero at every level the critical-level filter has stopped the
    # wave at, which is most of the column, so the norm is evaluated at its cone
    # tip as a matter of course.
    tau_mag = _safe_hypot(tau_x, tau_y)
    
    # Intrinsic phase speed (simplified)
    # Use a more realistic value based on typical gravity wave parameters
    c_phase = 20.0  # Typical phase speed ~ 20 m/s
    c_phase = jnp.ones_like(height) * c_phase
    
    # Wave amplitude
    amplitude = jnp.sqrt(jnp.abs(tau_mag) / (rho * c_phase + 1e-10))
    
    # Vertical shear
    du_dz = jnp.zeros(nlev)
    dv_dz = jnp.zeros(nlev)
    
    # Calculate shear (central differences)
    du_dz = du_dz.at[1:-1].set(
        (u[2:] - u[:-2]) / (height[2:] - height[:-2])
    )
    dv_dz = dv_dz.at[1:-1].set(
        (v[2:] - v[:-2]) / (height[2:] - height[:-2])
    )
    
    # Richardson number
    shear2 = du_dz**2 + dv_dz**2 + 1e-10
    richardson = n2 / shear2
    
    # Wave breaking criterion
    # Waves break when Ri < Ri_crit or amplitude exceeds threshold
    breaking_ri = richardson < config.ric
    breaking_amp = amplitude > 0.1 * jnp.sqrt(height)  # Amplitude threshold
    
    breaking_mask = breaking_ri | breaking_amp
    
    # Momentum deposition rate
    # Deposit all momentum flux divergence where breaking occurs
    dtau_x_dz = jnp.zeros(nlev)
    dtau_y_dz = jnp.zeros(nlev)
    
    # Flux divergence (upward decrease in flux = momentum deposition)
    # Use backward differences to ensure proper flux divergence
    dtau_x_dz = dtau_x_dz.at[1:].set(
        (tau_x[1:] - tau_x[:-1]) / (height[1:] - height[:-1])
    )
    dtau_y_dz = dtau_y_dz.at[1:].set(
        (tau_y[1:] - tau_y[:-1]) / (height[1:] - height[:-1])
    )
    
    # Apply breaking mask
    deposited_x = jnp.where(breaking_mask, -dtau_x_dz / rho, 0.0)
    deposited_y = jnp.where(breaking_mask, -dtau_y_dz / rho, 0.0)
    
    deposited_momentum = jnp.stack([deposited_x, deposited_y])
    
    return breaking_mask, deposited_momentum


@jax.jit
def simple_gwd(
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    air_density: jnp.ndarray,
    h_std: jnp.ndarray,
    dt: float,
    config: Optional[SimpleGwdParameters] = None
) -> Tuple[SimpleGwdTendencies, SimpleGwdState]:
    """Calculate gravity wave drag tendencies
    
    Args:
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        h_std: Standard deviation of sub-grid orography (m)
        dt: Time step (s)
        config: GW parameters
        
    Returns:
        Tuple of (tendencies, state)

    """
    if config is None:
        config = SimpleGwdParameters.default()
    
    nlev = u_wind.shape[0]
    
    # Calculate Brunt-Väisälä frequency
    n2 = brunt_vaisala_frequency(temperature, pressure, height)
    n_bv = jnp.sqrt(n2)
    
    # Air density
    rho = air_density
    
    # Initialize momentum fluxes
    tau_x = jnp.zeros(nlev)
    tau_y = jnp.zeros(nlev)
    
    # Orographic source at surface
    tau_x_oro, tau_y_oro = orographic_source(
        u_wind[-1], v_wind[-1], n_bv[-1], h_std, config
    )
    
    # Set surface flux
    tau_x = tau_x.at[-1].set(tau_x_oro * config.gwdrag_cd)
    tau_y = tau_y.at[-1].set(tau_y_oro * config.gwdrag_cd)
    
    # Propagate waves upward and check for breaking
    def propagate_level(carry, level_idx):
        tau_x_curr, tau_y_curr = carry
        
        # Get values at current level
        idx = nlev - 1 - level_idx  # Start from surface
        
        # Skip if above maximum height
        skip = height[idx] > config.zmax
        
        # Check for critical level (wind reversal)
        u_dot_tau = u_wind[idx] * tau_x_curr[idx] + v_wind[idx] * tau_y_curr[idx]
        critical_level = (idx > 0) & (u_dot_tau < 0)
        
        # Apply critical level filtering
        tau_x_new = jnp.where(critical_level | skip, 0.0, tau_x_curr[idx])
        tau_y_new = jnp.where(critical_level | skip, 0.0, tau_y_curr[idx])
        
        # Update flux at level above
        # Use lax.cond to handle the conditional update
        tau_x_curr = lax.cond(
            idx > 0,
            lambda x: x.at[idx-1].set(tau_x_new),
            lambda x: x,
            tau_x_curr
        )
        tau_y_curr = lax.cond(
            idx > 0,
            lambda y: y.at[idx-1].set(tau_y_new),
            lambda y: y,
            tau_y_curr
        )
        
        return (tau_x_curr, tau_y_curr), None
    
    # Propagate from surface upward
    (tau_x, tau_y), _ = lax.scan(
        propagate_level, (tau_x, tau_y), jnp.arange(nlev-1)
    )
    
    # Check for wave breaking and calculate deposition
    breaking_mask, deposited = wave_breaking_criterion(
        u_wind, v_wind, n2, height, tau_x, tau_y, rho, config
    )
    
    # Calculate tendencies
    # Note: deposited is already the acceleration (m/s²)
    dudt = deposited[0]
    dvdt = deposited[1]
    
    # Temperature tendency from dissipation (mechanical heating)
    # KE dissipation: dT/dt = -(u*du/dt + v*dv/dt) / cp
    dtedt = -(u_wind * dudt + v_wind * dvdt) / c.cpd
    
    # Only apply GWD above minimum height
    height_mask = height > config.zmin
    dudt = jnp.where(height_mask, dudt, 0.0)
    dvdt = jnp.where(height_mask, dvdt, 0.0)
    dtedt = jnp.where(height_mask, dtedt, 0.0)
    
    # Create output structures
    tendencies = SimpleGwdTendencies(
        dudt=dudt,
        dvdt=dvdt,
        dtedt=dtedt
    )
    
    state = SimpleGwdState(
        tau_x=tau_x,
        tau_y=tau_y,
        # Both diagnostics are norms of fields that are exactly zero wherever
        # the wave has been filtered out or no breaking was diagnosed, i.e. over
        # most of a typical column; see ``_safe_hypot``.
        wave_stress=_safe_hypot(tau_x, tau_y),
        breaking_level=breaking_mask.astype(jnp.float32),
        deposited_momentum=_safe_hypot(deposited[0], deposited[1])
    )
    
    return tendencies, state

# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class SimpleGwd(PhysicsTerm):
    """Simple monochromatic gravity-wave drag as a composable PhysicsTerm.

    Cheap fallback for the full Hines + Lott-Miller stack; kept available
    for aquaplanet tests but excluded from the default ``echam_physics()``
    factory. Operates on column-vectorized state ``(nlev, ncols)``. Reads
    ``pressure_full``, ``height_full``, ``air_density`` from the moist-air
    diagnostics dict and the model timestep from
    ``diagnostics["_dt_seconds"]`` (injected by ``ComposablePhysics``).
    Writes only u/v/T tendencies; no Data sub-struct.
    """

    name: ClassVar[str] = "simple_gwd"
    category: ClassVar[str] = "simple_gwd"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "height_full", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: SimpleGwdParameters | None = None):
        """Hold the scheme-native :class:`SimpleGwdParameters`."""
        self.params = nnx.Param(params or SimpleGwdParameters.default())

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute u/v/T tendencies from the simple GWD scheme."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        # Placeholder fixed std-dev of sub-grid orography. The real
        # Lott-Miller SSO scheme uses per-column ``terrain.orostd`` instead.
        h_std = jnp.ones(ncols) * 200.0

        tend, _state = jax.vmap(
            simple_gwd,
            in_axes=(1, 1, 1, 1, 1, 1, 0, None, None),
            out_axes=(0, 0),
        )(
            state.u_wind, state.v_wind, state.temperature,
            diagnostics["pressure_full"], diagnostics["height_full"],
            diagnostics["air_density"], h_std, dt, params,
        )

        return PhysicsTendency(
            u_wind=tend.dudt.T,
            v_wind=tend.dvdt.T,
            temperature=tend.dtedt.T,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={},
        ), diagnostics
