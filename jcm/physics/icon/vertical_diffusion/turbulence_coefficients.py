"""
Turbulence coefficient calculations for vertical diffusion.

This module computes exchange coefficients for momentum, heat, and moisture
based on atmospheric stability and turbulence closure schemes.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .vertical_diffusion_types import VDiffParameters, VDiffState, VDiffDiagnostics

# Create constants instance
PHYS_CONST = PhysicalConstants()


@jax.jit
def compute_richardson_number(
    u: jnp.ndarray,
    v: jnp.ndarray,
    temperature: jnp.ndarray,
    height_full: jnp.ndarray,
    height_half: jnp.ndarray,
    gravity: float = PHYS_CONST.grav
) -> jnp.ndarray:
    """
    Compute bulk Richardson number for atmospheric stability.
    
    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        temperature: Temperature [K] (ncol, nlev)
        height_full: Full level heights [m] (ncol, nlev)
        height_half: Half level heights [m] (ncol, nlev+1)
        gravity: Gravitational acceleration [m/s²]
        
    Returns:
        Richardson number [-] (ncol, nlev-1)
    """
    # Compute vertical gradients between adjacent full levels
    du_dz = jnp.diff(u, axis=1) / jnp.diff(height_full, axis=1)
    dv_dz = jnp.diff(v, axis=1) / jnp.diff(height_full, axis=1)
    dt_dz = jnp.diff(temperature, axis=1) / jnp.diff(height_full, axis=1)
    
    # Wind shear squared
    shear_squared = du_dz**2 + dv_dz**2
    
    # Average temperature for stability calculation
    temp_avg = 0.5 * (temperature[:, :-1] + temperature[:, 1:])
    
    # Brunt-Väisälä frequency squared (buoyancy frequency)
    # N² = (g/T) * (dT/dz + g/cp)
    lapse_rate = gravity / PHYS_CONST.cp
    buoyancy_freq_squared = (gravity / temp_avg) * (dt_dz + lapse_rate)
    
    # Richardson number: Ri = N² / (du/dz)²
    # Add small value to prevent division by zero
    ri = buoyancy_freq_squared / jnp.maximum(shear_squared, 1e-10)
    
    return ri


@jax.jit
def compute_mixing_length(
    height_full: jnp.ndarray,
    height_half: jnp.ndarray,
    richardson_number: jnp.ndarray,
    boundary_layer_height: jnp.ndarray,
    von_karman: float = 0.4,
    ri_critical: float = 0.25
) -> jnp.ndarray:
    """
    Compute mixing length for turbulence parameterization.
    
    Args:
        height_full: Full level heights [m] (ncol, nlev)
        height_half: Half level heights [m] (ncol, nlev+1)
        richardson_number: Richardson number [-] (ncol, nlev-1)
        boundary_layer_height: PBL height [m] (ncol,)
        von_karman: von Kármán constant [-]
        ri_critical: Critical Richardson number [-]
        
    Returns:
        Mixing length [m] (ncol, nlev)
    """
    ncol, nlev = height_full.shape
    
    # Distance from surface
    surface_height = height_half[:, -1:] # Surface is at bottom
    distance_from_surface = height_full - surface_height
    
    # Distance from top
    top_height = height_half[:, :1]  # Top is at index 0
    distance_from_top = top_height - height_full
    
    # Free atmosphere mixing length (asymptotic value)
    mixing_length_free = jnp.minimum(distance_from_surface, distance_from_top)
    
    # Boundary layer mixing length
    # Near surface: l = κz (von Kármán length scale)
    mixing_length_surface = von_karman * distance_from_surface
    
    # Within PBL: limit by boundary layer height
    pbl_height_expanded = boundary_layer_height[:, None]
    mixing_length_pbl = jnp.minimum(
        mixing_length_surface,
        0.1 * pbl_height_expanded
    )
    
    # Combine surface and free atmosphere contributions
    mixing_length = jnp.minimum(mixing_length_pbl, mixing_length_free)
    
    # Apply stability correction based on Richardson number
    # Extend Richardson number to full levels (pad with boundary values)
    ri_extended = jnp.concatenate([
        richardson_number[:, :1],    # Extend top value
        richardson_number,           # Interior values
        richardson_number[:, -1:]    # Extend bottom value
    ], axis=1)
    
    # Stability function: reduce mixing length for stable conditions
    stability_factor = jnp.where(
        ri_extended < ri_critical,
        jnp.maximum(0.1, (1.0 - ri_extended / ri_critical)**2),
        0.1  # Minimum mixing in stable conditions
    )
    
    mixing_length = mixing_length * stability_factor
    
    return jnp.maximum(mixing_length, 1.0)  # Minimum mixing length


@jax.jit
def compute_exchange_coefficients(
    state: VDiffState,
    params: VDiffParameters,
    mixing_length: jnp.ndarray,
    richardson_number: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute exchange coefficients for momentum, heat, and moisture.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        mixing_length: Mixing length [m] (ncol, nlev)
        richardson_number: Richardson number [-] (ncol, nlev-1)
        
    Returns:
        Tuple of:
        - Momentum exchange coefficient [m²/s] (ncol, nlev)
        - Heat exchange coefficient [m²/s] (ncol, nlev)
        - Moisture exchange coefficient [m²/s] (ncol, nlev)
    """
    ncol, nlev = state.u.shape
    
    # Compute wind shear
    u_diff = jnp.diff(state.u, axis=1)
    v_diff = jnp.diff(state.v, axis=1)
    height_diff = jnp.diff(state.height_full, axis=1)
    
    # Extend to full levels
    u_shear = jnp.concatenate([
        u_diff[:, :1] / height_diff[:, :1],
        u_diff / height_diff,
        u_diff[:, -1:] / height_diff[:, -1:]
    ], axis=1)
    
    v_shear = jnp.concatenate([
        v_diff[:, :1] / height_diff[:, :1],
        v_diff / height_diff,
        v_diff[:, -1:] / height_diff[:, -1:]
    ], axis=1)
    
    shear_magnitude = jnp.sqrt(u_shear**2 + v_shear**2)
    
    # Base exchange coefficient: K = l² * |S|
    # where l is mixing length and S is shear magnitude
    exchange_coeff_base = mixing_length**2 * shear_magnitude
    
    # Apply minimum value
    exchange_coeff_base = jnp.maximum(exchange_coeff_base, 0.1)
    
    # Momentum exchange coefficient
    exchange_coeff_momentum = exchange_coeff_base
    
    # Heat and moisture exchange coefficients
    # Apply Prandtl number correction (typically ~0.7-1.0)
    prandtl_number = 0.8
    exchange_coeff_heat = exchange_coeff_base / prandtl_number
    exchange_coeff_moisture = exchange_coeff_heat  # Assume same as heat
    
    # Apply upper limits to prevent numerical instability
    max_exchange_coeff = 1000.0  # m²/s
    exchange_coeff_momentum = jnp.minimum(exchange_coeff_momentum, max_exchange_coeff)
    exchange_coeff_heat = jnp.minimum(exchange_coeff_heat, max_exchange_coeff)
    exchange_coeff_moisture = jnp.minimum(exchange_coeff_moisture, max_exchange_coeff)
    
    return exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture


@jax.jit
def compute_surface_exchange_coefficients(
    state: VDiffState,
    params: VDiffParameters,
    wind_speed_surface: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    temperature_air: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute surface exchange coefficients for different surface types.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        wind_speed_surface: Wind speed at surface [m/s] (ncol,)
        temperature_surface: Surface temperature [K] (ncol, nsfc_type)
        temperature_air: Air temperature at lowest level [K] (ncol,)
        
    Returns:
        Tuple of:
        - Surface heat exchange coefficient [m²/s] (ncol, nsfc_type)
        - Surface moisture exchange coefficient [m²/s] (ncol, nsfc_type)
    """
    ncol, nsfc_type = temperature_surface.shape
    
    # Roughness lengths for different surface types
    # [water, ice, land]
    z0_heat = jnp.array([1e-4, 1e-4, 1e-2])  # Thermal roughness
    z0_moisture = jnp.array([1e-4, 1e-4, 1e-2])  # Moisture roughness
    
    # Reference height (lowest model level)
    z_ref = state.height_full[:, -1] - state.height_half[:, -1]
    
    # von Kármán constant
    von_karman = 0.4
    
    # Compute exchange coefficients for each surface type
    surface_exchange_heat = jnp.zeros((ncol, nsfc_type))
    surface_exchange_moisture = jnp.zeros((ncol, nsfc_type))
    
    for isfc in range(nsfc_type):
        # Stability correction (simplified)
        theta_surface = temperature_surface[:, isfc]
        theta_air = temperature_air
        
        # Bulk Richardson number for surface layer
        ri_surface = (PHYS_CONST.grav * z_ref[:, None] * 
                     (theta_air - theta_surface) / 
                     (0.5 * (theta_air + theta_surface) * 
                      jnp.maximum(wind_speed_surface**2, 0.01)))
        
        # Stability function (simplified Businger-Dyer)
        stability_heat = jnp.where(
            ri_surface < 0,
            (1.0 - 16.0 * ri_surface)**(-0.5),  # Unstable
            1.0 / (1.0 + 5.0 * ri_surface)     # Stable
        )
        
        # Exchange coefficient: CH = κ² / [ln(z/z0)]²
        log_ratio_heat = jnp.log(z_ref / z0_heat[isfc])
        log_ratio_moisture = jnp.log(z_ref / z0_moisture[isfc])
        
        exchange_heat = (von_karman**2 * wind_speed_surface * 
                        stability_heat / log_ratio_heat**2)
        exchange_moisture = (von_karman**2 * wind_speed_surface * 
                           stability_heat / log_ratio_moisture**2)
        
        surface_exchange_heat = surface_exchange_heat.at[:, isfc].set(
            jnp.maximum(exchange_heat, 1e-6)
        )
        surface_exchange_moisture = surface_exchange_moisture.at[:, isfc].set(
            jnp.maximum(exchange_moisture, 1e-6)
        )
    
    return surface_exchange_heat, surface_exchange_moisture


@jax.jit
def compute_boundary_layer_height(
    state: VDiffState,
    exchange_coeff_heat: jnp.ndarray,
    threshold: float = 1.0
) -> jnp.ndarray:
    """
    Compute boundary layer height based on exchange coefficient profile.
    
    Args:
        state: Atmospheric state
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        threshold: Threshold value for defining PBL top [m²/s]
        
    Returns:
        Boundary layer height [m] (ncol,)
    """
    ncol, nlev = state.height_full.shape
    
    # Find level where exchange coefficient drops below threshold
    # Start from surface (highest index) and work upward
    below_threshold = exchange_coeff_heat < threshold
    
    # Find first level from bottom where condition is met
    # Use cumulative sum to find first occurrence
    pbl_mask = jnp.cumsum(below_threshold[:, ::-1], axis=1)
    first_occurrence = jnp.argmax(pbl_mask > 0, axis=1)
    
    # Convert to height
    # If no level found, use top of atmosphere
    height_from_bottom = state.height_full[:, ::-1]
    surface_height = state.height_half[:, -1]
    
    pbl_height = jnp.where(
        jnp.any(below_threshold, axis=1),
        height_from_bottom[jnp.arange(ncol), first_occurrence] - surface_height,
        height_from_bottom[:, 0] - surface_height  # Use top level
    )
    
    # Ensure minimum PBL height
    pbl_height = jnp.maximum(pbl_height, 50.0)
    
    return pbl_height


@jax.jit
def compute_friction_velocity(
    momentum_flux_u: jnp.ndarray,
    momentum_flux_v: jnp.ndarray,
    air_density: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute friction velocity from surface momentum flux.
    
    Args:
        momentum_flux_u: U-momentum flux [N/m²] (ncol,)
        momentum_flux_v: V-momentum flux [N/m²] (ncol,)
        air_density: Air density [kg/m³] (ncol,)
        
    Returns:
        Friction velocity [m/s] (ncol,)
    """
    momentum_flux_magnitude = jnp.sqrt(momentum_flux_u**2 + momentum_flux_v**2)
    friction_velocity = jnp.sqrt(momentum_flux_magnitude / air_density)
    
    return jnp.maximum(friction_velocity, 0.01)  # Minimum value


@jax.jit
def compute_turbulence_diagnostics(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray
) -> VDiffDiagnostics:
    """
    Compute complete set of turbulence diagnostics.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s]
        exchange_coeff_heat: Heat exchange coefficient [m²/s]
        exchange_coeff_moisture: Moisture exchange coefficient [m²/s]
        
    Returns:
        Complete diagnostics structure
    """
    ncol = state.u.shape[0]
    
    # Compute Richardson number
    ri = compute_richardson_number(
        state.u, state.v, state.temperature,
        state.height_full, state.height_half
    )
    
    # Compute mixing length
    pbl_height = compute_boundary_layer_height(state, exchange_coeff_heat)
    mixing_length = compute_mixing_length(
        state.height_full, state.height_half, ri, pbl_height
    )
    
    # Surface diagnostics
    wind_speed_surface = jnp.sqrt(state.u[:, -1]**2 + state.v[:, -1]**2)
    surface_exchange_heat, surface_exchange_moisture = compute_surface_exchange_coefficients(
        state, params, wind_speed_surface,
        state.surface_temperature, state.temperature[:, -1]
    )
    
    # Placeholder values for fluxes (would be computed in full implementation)
    surface_momentum_flux_u = jnp.zeros(ncol)
    surface_momentum_flux_v = jnp.zeros(ncol)
    surface_heat_flux = jnp.zeros(ncol)
    surface_moisture_flux = jnp.zeros(ncol)
    
    # Air density at surface
    air_density = (state.pressure_full[:, -1] / 
                  (PHYS_CONST.rd * state.temperature[:, -1]))
    
    friction_velocity = compute_friction_velocity(
        surface_momentum_flux_u, surface_momentum_flux_v, air_density
    )
    
    # Convective velocity scale (simplified)
    convective_velocity = jnp.maximum(friction_velocity, 0.1)
    
    return VDiffDiagnostics(
        exchange_coeff_momentum=exchange_coeff_momentum,
        exchange_coeff_heat=exchange_coeff_heat,
        exchange_coeff_moisture=exchange_coeff_moisture,
        surface_exchange_heat=surface_exchange_heat,
        surface_exchange_moisture=surface_exchange_moisture,
        boundary_layer_height=pbl_height,
        friction_velocity=friction_velocity,
        convective_velocity=convective_velocity,
        richardson_number=ri,
        mixing_length=mixing_length,
        surface_momentum_flux_u=surface_momentum_flux_u,
        surface_momentum_flux_v=surface_momentum_flux_v,
        surface_heat_flux=surface_heat_flux,
        surface_moisture_flux=surface_moisture_flux,
        kinetic_energy_dissipation=jnp.zeros(ncol)
    )