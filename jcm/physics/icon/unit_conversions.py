"""
Unit conversion utilities for ICON physics

This module ensures that physics state variables from the dynamics core
are properly converted to the units expected by ICON physics modules.

Physics interface provides:
- u_wind, v_wind: m/s (dimensional)
- temperature: K (dimensional)
- specific_humidity: kg/kg (dimensional)
- geopotential: m²/s² (dimensional)
- surface_pressure: normalized by p0 (dimensionless)

ICON physics expects:
- u_wind, v_wind: m/s
- temperature: K
- specific_humidity: kg/kg
- pressure: Pa
- height: m
- dt: seconds

Date: 2025-01-10
"""

import jax.numpy as jnp
from typing import Tuple
from ..speedy.physical_constants import p0
from .constants.physical_constants import grav, rd


def convert_surface_pressure(surface_pressure_normalized: jnp.ndarray) -> jnp.ndarray:
    """
    Convert normalized surface pressure to Pascal.
    
    Args:
        surface_pressure_normalized: Surface pressure normalized by p0 (dimensionless)
        
    Returns:
        Surface pressure in Pascal
    """
    return surface_pressure_normalized * p0


def calculate_pressure_levels(
    surface_pressure_normalized: jnp.ndarray,
    sigma_levels: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate pressure at each model level.
    
    Args:
        surface_pressure_normalized: Normalized surface pressure [ncols] or [nlat, nlon]
        sigma_levels: Sigma coordinates at level centers [nlev]
        
    Returns:
        Pressure at each level in Pascal [nlev, ncols] or [nlev, nlat, nlon]
    """
    surface_pressure_pa = convert_surface_pressure(surface_pressure_normalized)
    
    # Handle both 1D (ncols) and 2D (nlat, nlon) surface pressure
    if surface_pressure_pa.ndim == 1:
        # [nlev, ncols]
        pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure_pa[jnp.newaxis, :]
    else:
        # [nlev, nlat, nlon]
        pressure_levels = sigma_levels[:, jnp.newaxis, jnp.newaxis] * surface_pressure_pa[jnp.newaxis, :, :]
    
    return pressure_levels


def geopotential_to_height(geopotential: jnp.ndarray) -> jnp.ndarray:
    """
    Convert geopotential to geometric height.
    
    Args:
        geopotential: Geopotential in m²/s²
        
    Returns:
        Geometric height in meters
    """
    return geopotential / grav


def calculate_air_density(
    pressure: jnp.ndarray,
    temperature: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate air density from pressure and temperature.
    
    Args:
        pressure: Pressure in Pascal
        temperature: Temperature in Kelvin
        
    Returns:
        Air density in kg/m³
    """
    return pressure / (rd * temperature)


def calculate_layer_thickness(
    pressure_levels: jnp.ndarray,
    temperature: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate layer thickness using hydrostatic approximation.
    
    Args:
        pressure_levels: Pressure at each level in Pascal [nlev, ...]
        temperature: Temperature at each level in Kelvin [nlev, ...]
        
    Returns:
        Layer thickness in meters [nlev, ...]
    """
    nlev = pressure_levels.shape[0]
    air_density = calculate_air_density(pressure_levels, temperature)
    
    # Initialize with zeros
    dz = jnp.zeros_like(pressure_levels)
    
    # Use hydrostatic approximation: dz = dp/(rho*g)
    # Since pressure increases downward, dp is positive and gives positive dz
    # For levels 1 to nlev-1
    dp = jnp.diff(pressure_levels, axis=0)
    rho_mid = 0.5 * (air_density[1:] + air_density[:-1])
    dz = dz.at[1:].set(dp / (rho_mid * grav))
    
    # For the top layer, use the same thickness as the layer below
    dz = dz.at[0].set(dz[1])
    
    return dz


def prepare_physics_state_2d(state, geometry):
    """
    Prepare physics state with proper unit conversions for 2D (vectorized) format.
    
    This is used within IconPhysics after state has been reshaped to [nlev, ncols].
    
    Args:
        state: PhysicsState in 2D format [nlev, ncols]
        geometry: Geometry object
        
    Returns:
        Tuple of converted quantities needed by physics schemes
    """
    nlev, ncols = state.temperature.shape
    
    # Convert surface pressure to Pascal
    surface_pressure_pa = convert_surface_pressure(state.normalized_surface_pressure)
    
    # Calculate pressure levels
    pressure_levels = calculate_pressure_levels(state.normalized_surface_pressure, geometry.fsg)
    
    # Convert geopotential to height
    height_levels = geopotential_to_height(state.geopotential)
    
    # Calculate air density
    air_density = calculate_air_density(pressure_levels, state.temperature)
    
    # Calculate layer thickness
    layer_thickness = calculate_layer_thickness(pressure_levels, state.temperature)
    
    return {
        'surface_pressure_pa': surface_pressure_pa,
        'pressure_levels': pressure_levels,
        'height_levels': height_levels,
        'air_density': air_density,
        'layer_thickness': layer_thickness
    }


def prepare_physics_state_3d(state, geometry):
    """
    Prepare physics state with proper unit conversions for 3D format.
    
    This could be used for diagnostics or debugging with full 3D arrays.
    
    Args:
        state: PhysicsState in 3D format [nlev, nlat, nlon]
        geometry: Geometry object
        
    Returns:
        Tuple of converted quantities needed by physics schemes
    """
    nlev, nlat, nlon = state.temperature.shape
    
    # Convert surface pressure to Pascal
    surface_pressure_pa = convert_surface_pressure(state.normalized_surface_pressure)
    
    # Calculate pressure levels
    pressure_levels = calculate_pressure_levels(state.normalized_surface_pressure, geometry.fsg)
    
    # Convert geopotential to height
    height_levels = geopotential_to_height(state.geopotential)
    
    # Calculate air density
    air_density = calculate_air_density(pressure_levels, state.temperature)
    
    # Calculate layer thickness
    layer_thickness = calculate_layer_thickness(pressure_levels, state.temperature)
    
    return {
        'surface_pressure_pa': surface_pressure_pa,
        'pressure_levels': pressure_levels,
        'height_levels': height_levels,
        'air_density': air_density,
        'layer_thickness': layer_thickness
    }


def verify_physics_units(state, converted_state):
    """
    Verify that unit conversions produce reasonable values.
    
    Args:
        state: Original PhysicsState
        converted_state: Dictionary of converted quantities
        
    Returns:
        Dictionary of verification results
    """
    checks = {}
    
    # Check surface pressure (should be around 100000 Pa)
    ps_pa = converted_state['surface_pressure_pa']
    checks['surface_pressure_reasonable'] = jnp.all((ps_pa > 50000) & (ps_pa < 110000))
    
    # Check pressure levels (should increase with index since we go top to bottom)
    pressure = converted_state['pressure_levels']
    if pressure.ndim > 1:
        checks['pressure_decreasing'] = jnp.all(jnp.diff(pressure, axis=0) > 0)
    
    # Check heights are positive and decreasing with index (top to bottom)
    height = converted_state['height_levels']
    checks['height_positive'] = jnp.all(height >= 0)
    if height.ndim > 1:
        checks['height_increasing'] = jnp.all(jnp.diff(height, axis=0) < 0)
    
    # Check air density is reasonable (0.1 to 2 kg/m³)
    rho = converted_state['air_density']
    checks['density_reasonable'] = jnp.all((rho > 0.1) & (rho < 2.0))
    
    # Check layer thickness is positive
    dz = converted_state['layer_thickness']
    checks['thickness_positive'] = jnp.all(dz > 0)
    
    return checks