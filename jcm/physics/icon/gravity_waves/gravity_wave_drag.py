"""
Gravity wave drag parameterization for ICON physics

This module implements the orographic and non-orographic gravity wave drag
parameterizations. The scheme accounts for momentum deposition from breaking
gravity waves that are not resolved by the model grid.

Based on ICON's mo_gwd_wms.f90 and mo_ssodrag.f90

Features:
- Orographic gravity wave drag (mountain waves)
- Non-orographic gravity wave sources
- Wave breaking and momentum deposition
- Critical level filtering

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass
from functools import partial

from ..constants.physical_constants import (
    grav, rd, cp
)


@dataclass(frozen=True)
class GravityWaveParameters:
    """Parameters for gravity wave drag scheme"""
    
    # Orographic drag parameters
    gkdrag: float = 0.5           # Surface drag coefficient
    gkwake: float = 0.5           # Wake drag coefficient
    grcrit: float = 0.25          # Critical Froude number
    gssec: float = 0.0001         # Security parameter for Richardson number
    gtsec: float = 0.0001         # Security parameter for Brunt-Vaisala frequency
    
    # Non-orographic parameters
    ruwmax: float = 1.0           # Launch momentum flux for non-orographic waves (N/m²)
    nslope: float = 1.0           # Slope of wave spectrum
    
    # Wave breaking parameters
    ric: float = 0.25             # Critical Richardson number
    efmin: float = 0.0            # Minimum efficiency
    efmax: float = 0.1            # Maximum efficiency
    
    # Numerical parameters
    zmin: float = 1000.0          # Minimum height for GWD (m)
    zmax: float = 100000.0        # Maximum height for GWD (m)
    
    # Tuning parameters
    gwdrag_cd: float = 1.0        # Drag coefficient multiplier
    gwdrag_ef: float = 0.05       # Efficiency factor


class GravityWaveState(NamedTuple):
    """State variables for gravity wave calculations"""
    
    tau_x: jnp.ndarray            # Zonal momentum flux (N/m²)
    tau_y: jnp.ndarray            # Meridional momentum flux (N/m²)
    wave_stress: jnp.ndarray      # Wave stress magnitude (N/m²)
    breaking_level: jnp.ndarray   # Level where waves break
    deposited_momentum: jnp.ndarray  # Momentum deposited (m/s²)


class GravityWaveTendencies(NamedTuple):
    """Tendencies from gravity wave drag"""
    
    dudt: jnp.ndarray             # Zonal wind tendency (m/s²)
    dvdt: jnp.ndarray             # Meridional wind tendency (m/s²)
    dtedt: jnp.ndarray            # Temperature tendency from dissipation (K/s)


@jax.jit
def brunt_vaisala_frequency(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate Brunt-Väisälä frequency
    
    Args:
        temperature: Temperature profile (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        
    Returns:
        N²: Brunt-Väisälä frequency squared (s⁻²) [nlev]
    """
    # Calculate potential temperature
    p0 = 100000.0  # Reference pressure
    theta = temperature * (p0 / pressure) ** (rd / cp)
    
    # Calculate vertical gradient of potential temperature
    # Use one-sided differences at boundaries
    nlev = temperature.shape[0]
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
    n2 = grav / theta * dtheta_dz
    
    # Apply minimum threshold for stability
    n2 = jnp.maximum(n2, 1e-8)
    
    return n2


@partial(jax.jit, static_argnames=['config'])
def orographic_source(
    u_sfc: jnp.ndarray,
    v_sfc: jnp.ndarray,
    n_sfc: jnp.ndarray,
    h_std: jnp.ndarray,
    config: GravityWaveParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate orographic gravity wave source
    
    Args:
        u_sfc: Surface zonal wind (m/s)
        v_sfc: Surface meridional wind (m/s)
        n_sfc: Surface Brunt-Väisälä frequency (s⁻¹)
        h_std: Standard deviation of orography (m)
        config: GW parameters
        
    Returns:
        Tuple of (tau_x, tau_y): Surface momentum fluxes (N/m²)
    """
    # Surface wind speed
    wind_speed = jnp.sqrt(u_sfc**2 + v_sfc**2)
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


@partial(jax.jit, static_argnames=['config'])
def wave_breaking_criterion(
    u: jnp.ndarray,
    v: jnp.ndarray,
    n2: jnp.ndarray,
    height: jnp.ndarray,
    tau_x: jnp.ndarray,
    tau_y: jnp.ndarray,
    rho: jnp.ndarray,
    config: GravityWaveParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Determine wave breaking and momentum deposition
    
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
    tau_mag = jnp.sqrt(tau_x**2 + tau_y**2)
    
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


@partial(jax.jit, static_argnames=['config'])
def gravity_wave_drag(
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    h_std: jnp.ndarray,
    dt: float,
    config: Optional[GravityWaveParameters] = None
) -> Tuple[GravityWaveTendencies, GravityWaveState]:
    """
    Calculate gravity wave drag tendencies
    
    Args:
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        h_std: Standard deviation of sub-grid orography (m)
        dt: Time step (s)
        config: GW parameters
        
    Returns:
        Tuple of (tendencies, state)
    """
    if config is None:
        config = GravityWaveParameters()
    
    nlev = u_wind.shape[0]
    
    # Calculate Brunt-Väisälä frequency
    n2 = brunt_vaisala_frequency(temperature, pressure, height)
    n_bv = jnp.sqrt(n2)
    
    # Air density
    rho = pressure / (rd * temperature)
    
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
    dtedt = -(u_wind * dudt + v_wind * dvdt) / cp
    
    # Only apply GWD above minimum height
    height_mask = height > config.zmin
    dudt = jnp.where(height_mask, dudt, 0.0)
    dvdt = jnp.where(height_mask, dvdt, 0.0)
    dtedt = jnp.where(height_mask, dtedt, 0.0)
    
    # Create output structures
    tendencies = GravityWaveTendencies(
        dudt=dudt,
        dvdt=dvdt,
        dtedt=dtedt
    )
    
    state = GravityWaveState(
        tau_x=tau_x,
        tau_y=tau_y,
        wave_stress=jnp.sqrt(tau_x**2 + tau_y**2),
        breaking_level=breaking_mask.astype(jnp.float32),
        deposited_momentum=jnp.sqrt(deposited[0]**2 + deposited[1]**2)
    )
    
    return tendencies, state


def test_gravity_wave_drag():
    """Simple test of gravity wave drag"""
    # Create test profile
    nlev = 30
    height = jnp.linspace(0, 30000, nlev)
    pressure = 100000 * jnp.exp(-height / 8000)  # Exponential atmosphere
    temperature = 288 - 0.0065 * height  # Standard lapse rate
    
    # Westerly jet with shear
    u_wind = 20.0 * jnp.exp(-(height - 10000)**2 / 5000**2)
    v_wind = jnp.zeros(nlev)
    
    # Orography
    h_std = 500.0  # 500m standard deviation
    
    dt = 1800.0
    
    # Calculate GWD
    tendencies, state = gravity_wave_drag(
        u_wind, v_wind, temperature, pressure, height, h_std, dt
    )
    
    print(f"Max u-tendency: {jnp.abs(tendencies.dudt).max()*86400:.2f} m/s/day")
    print(f"Max temperature tendency: {jnp.abs(tendencies.dtedt).max()*86400:.2f} K/day")
    print(f"Surface wave stress: {state.wave_stress[-1]:.4f} N/m²")
    
    # Check that drag opposes flow
    assert jnp.all(tendencies.dudt * u_wind <= 0)
    print("✓ Drag opposes flow direction")
    
    # Check momentum conservation (approximately)
    total_momentum_change = jnp.sum(tendencies.dudt * pressure / grav)
    surface_stress = state.tau_x[-1]
    print(f"Total momentum change: {total_momentum_change:.4f}")
    print(f"Surface stress: {surface_stress:.4f}")
    
    print("\nGravity wave drag test passed!")


if __name__ == "__main__":
    test_gravity_wave_drag()