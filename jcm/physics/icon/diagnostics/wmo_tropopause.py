"""
WMO Tropopause Diagnostic for ICON Physics

This module implements the WMO (1957) tropopause definition following the
ICON mo_tropopause.f90 implementation. The tropopause is defined as the
lowest level at which the lapse rate decreases to 2°C per kilometer or less,
provided the average lapse rate between this level and all higher levels
within 2 kilometers does not exceed 2°C per kilometer.

Date: 2025-01-09
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from jcm.physics.icon.constants import physical_constants

# WMO tropopause constants
GWMO = -0.002  # K/m - The -2°C/km threshold
DELTAZ = 2000.0  # m - The 2 km height interval for averaging
P_DEFAULT = 20000.0  # Pa - Default tropopause pressure (~200 hPa)

def compute_geopotential_height(pressure: jnp.ndarray, 
                              temperature: jnp.ndarray,
                              surface_pressure: jnp.ndarray) -> jnp.ndarray:
    """
    Compute geopotential height from pressure and temperature.
    
    Uses the hypsometric equation with proper handling of model levels.
    
    Args:
        pressure: Pressure at model levels [Pa] (shape: [..., nlev])
        temperature: Temperature at model levels [K] (shape: [..., nlev])
        surface_pressure: Surface pressure [Pa] (shape: [...])
        
    Returns:
        Geopotential height [m] (shape: [..., nlev])
    """
    # Constants
    g = physical_constants.grav
    R = physical_constants.rgas
    
    # Ensure surface_pressure has compatible shape for concatenation
    batch_shape = pressure.shape[:-1]
    
    # Handle the case where pressure is 1D (no batch dimensions)
    if len(batch_shape) == 0:
        # For 1D pressure, surface_pressure should be scalar or length-1 array
        surface_pressure_expanded = jnp.squeeze(surface_pressure)
        surface_pressure_with_level = surface_pressure_expanded[None]  # Add level dimension
    else:
        # For batched pressure, broadcast surface_pressure to batch shape
        # Handle case where surface_pressure might already have extra dimensions
        if surface_pressure.ndim > len(batch_shape):
            # Surface pressure has extra dimensions, squeeze them
            surface_pressure_squeezed = jnp.squeeze(surface_pressure, axis=-1)
        else:
            surface_pressure_squeezed = surface_pressure
            
        surface_pressure_expanded = jnp.broadcast_to(surface_pressure_squeezed, batch_shape)
        surface_pressure_with_level = surface_pressure_expanded[..., None]
    
    # For pressure interfaces, use surface pressure as bottom level
    # and each model level as upper boundary
    pressure_lower = jnp.concatenate([surface_pressure_with_level, pressure[..., :-1]], axis=-1)
    pressure_upper = pressure
    
    # Use temperature at the current level for the layer below
    layer_thickness = (R * temperature / g) * jnp.log(pressure_lower / pressure_upper)
    
    # Compute cumulative height from surface
    height = jnp.cumsum(layer_thickness, axis=-1)
    
    return height

def compute_lapse_rate(temperature: jnp.ndarray, 
                      height: jnp.ndarray) -> jnp.ndarray:
    """
    Compute temperature lapse rate dT/dz.
    
    Args:
        temperature: Temperature [K] (shape: [..., nlev])
        height: Geopotential height [m] (shape: [..., nlev])
        
    Returns:
        Lapse rate [K/m] (shape: [..., nlev-1])
    """
    # Compute finite differences
    dT = temperature[..., 1:] - temperature[..., :-1]
    dz = height[..., 1:] - height[..., :-1]
    
    # Avoid division by zero
    dz = jnp.where(dz == 0, 1e-10, dz)
    
    lapse_rate = dT / dz
    
    return lapse_rate

def find_tropopause_level(temperature: jnp.ndarray,
                         pressure: jnp.ndarray,
                         height: jnp.ndarray,
                         ncctop: int = 13,
                         nccbot: int = 35) -> jnp.ndarray:
    """
    Find the tropopause level following WMO definition.
    
    Args:
        temperature: Temperature [K] (shape: [..., nlev])
        pressure: Pressure [Pa] (shape: [..., nlev])
        height: Geopotential height [m] (shape: [..., nlev])
        ncctop: Highest level index for tropopause search
        nccbot: Lowest level index for tropopause search
        
    Returns:
        Tropopause pressure [Pa] (shape: [...])
    """
    # Limit search to specified vertical range
    search_temp = temperature[..., ncctop:nccbot]
    search_pressure = pressure[..., ncctop:nccbot]
    search_height = height[..., ncctop:nccbot]
    
    # Compute lapse rate
    lapse_rate = compute_lapse_rate(search_temp, search_height)
    
    # Find levels where lapse rate is >= GWMO (-2 K/km)
    stable_mask = lapse_rate >= GWMO
    
    nlev_search = search_temp.shape[-1]
    nlev_lapse = lapse_rate.shape[-1]
    batch_shape = temperature.shape[:-1]
    
    def find_tropopause_column(temp_col, pres_col, height_col, lapse_col):
        """Find tropopause for a single column using JAX-compatible operations"""
        
        # Start from bottom (surface) and work up to find the LOWEST level
        # that meets the criteria (this is the key to WMO definition)
        level_indices = jnp.arange(nlev_lapse)
        
        def check_level(k):
            """Check if level k satisfies tropopause criteria"""
            # Check basic lapse rate criterion
            lapse_ok = lapse_col[k] >= GWMO
            
            # Check 2km averaging criterion
            current_height = height_col[k]
            top_height = current_height + DELTAZ
            
            # Find levels within 2km above current level (for height levels)
            above_mask = (height_col >= current_height) & (height_col <= top_height)
            
            # For lapse rate averaging, we only consider the first nlev_lapse heights
            # since lapse rate array has nlev_lapse elements
            above_mask_for_lapse = above_mask[:nlev_lapse]
            
            # Get valid lapse rate values within the 2km window
            valid_lapse_values = jnp.where(above_mask_for_lapse, lapse_col, 0.0)
            num_valid = jnp.sum(above_mask_for_lapse.astype(jnp.float32))
            
            # Compute average, handling division by zero
            avg_lapse = jnp.where(
                num_valid > 0,
                jnp.sum(valid_lapse_values) / jnp.maximum(num_valid, 1e-10),
                GWMO - 1.0  # Value that will fail the test
            )
            
            # Both criteria must be satisfied
            both_ok = lapse_ok & (avg_lapse >= GWMO) & (num_valid > 0)
            
            return both_ok, pres_col[k]
        
        # Check all levels using jax.lax.scan
        def scan_levels(carry, k):
            found, tropopause_p = carry
            level_ok, level_pressure = check_level(k)
            
            # Update result if we haven't found a tropopause yet and this level satisfies criteria
            new_found = found | level_ok
            new_pressure = jnp.where(found, tropopause_p, 
                                   jnp.where(level_ok, level_pressure, tropopause_p))
            
            return (new_found, new_pressure), None
        
        # Initialize: no tropopause found, default pressure
        initial_state = (False, P_DEFAULT)
        
        # Scan through levels from bottom to top
        (found, final_pressure), _ = jax.lax.scan(scan_levels, initial_state, level_indices)
        
        return final_pressure
    
    # Apply to each column in batch
    def process_batch(args):
        temp_batch, pres_batch, height_batch, lapse_batch = args
        return jax.vmap(find_tropopause_column)(temp_batch, pres_batch, height_batch, lapse_batch)
    
    # Process the batch
    result = process_batch((search_temp, search_pressure, search_height, lapse_rate))
    
    return result

def wmo_tropopause(temperature: jnp.ndarray,
                  pressure: jnp.ndarray,
                  surface_pressure: jnp.ndarray,
                  previous_tropopause: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Compute WMO tropopause pressure diagnostic.
    
    Follows the WMO (1957) definition: the tropopause is the lowest level
    at which the lapse rate decreases to 2°C per kilometer or less,
    provided the average lapse rate between this level and all higher
    levels within 2 kilometers does not exceed 2°C per kilometer.
    
    Args:
        temperature: Temperature at model levels [K] (shape: [..., nlev])
        pressure: Pressure at model levels [Pa] (shape: [..., nlev])
        surface_pressure: Surface pressure [Pa] (shape: [...])
        previous_tropopause: Previous tropopause pressure [Pa] (shape: [...])
                           Used as fallback if no tropopause found
        
    Returns:
        Tropopause pressure [Pa] (shape: [...])
    """
    # Compute geopotential height
    height = compute_geopotential_height(pressure, temperature, surface_pressure)
    
    # Find tropopause level
    tropopause_pressure = find_tropopause_level(temperature, pressure, height)
    
    # Use previous value as fallback if available
    if previous_tropopause is not None:
        # If no valid tropopause found (pressure = default), use previous
        use_previous = tropopause_pressure == P_DEFAULT
        tropopause_pressure = jnp.where(use_previous, previous_tropopause, tropopause_pressure)
    
    return tropopause_pressure