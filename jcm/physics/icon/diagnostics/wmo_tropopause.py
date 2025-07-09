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
    
    # Compute layer thickness using hypsometric equation
    # dz = (R * T / g) * ln(p_lower / p_upper)
    
    # For simplicity, assume pressure decreases with height
    # and use midpoint temperatures for layer calculations
    pressure_lower = jnp.concatenate([surface_pressure[..., None], pressure[..., :-1]], axis=-1)
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
    
    # For each column, find the lowest level meeting the criteria
    nlev_search = search_temp.shape[-1]
    
    def find_tropopause_column(temp_col, pres_col, height_col, lapse_col):
        """Find tropopause for a single column"""
        
        # Start from the bottom and work up
        for k in range(nlev_search - 2, -1, -1):  # Skip last level (no lapse rate)
            if lapse_col[k] >= GWMO:
                # Check if this level satisfies the 2km averaging criterion
                current_height = height_col[k]
                top_height = current_height + DELTAZ
                
                # Find levels within 2km above
                above_mask = (height_col >= current_height) & (height_col <= top_height)
                
                if jnp.any(above_mask):
                    # Compute average lapse rate over 2km
                    above_indices = jnp.where(above_mask, jnp.arange(nlev_search), -1)
                    valid_indices = above_indices[above_indices >= 0]
                    
                    if len(valid_indices) > 1:
                        # Take indices that have lapse rate data
                        valid_lapse_indices = valid_indices[valid_indices < len(lapse_col)]
                        
                        if len(valid_lapse_indices) > 0:
                            avg_lapse = jnp.mean(lapse_col[valid_lapse_indices])
                            
                            # Check if average lapse rate also satisfies criterion
                            if avg_lapse >= GWMO:
                                return pres_col[k]
        
        # No tropopause found, return default
        return P_DEFAULT
    
    # Vectorized version using lax.map for each column
    def process_column(args):
        temp_col, pres_col, height_col, lapse_col = args
        return find_tropopause_column(temp_col, pres_col, height_col, lapse_col)
    
    # Prepare arguments for mapping
    batch_shape = temperature.shape[:-1]
    flat_temp = search_temp.reshape(-1, nlev_search)
    flat_pres = search_pressure.reshape(-1, nlev_search)
    flat_height = search_height.reshape(-1, nlev_search)
    flat_lapse = lapse_rate.reshape(-1, nlev_search - 1)
    
    # Apply to each column
    result = jax.vmap(process_column)((flat_temp, flat_pres, flat_height, flat_lapse))
    
    # Reshape back to original batch shape
    tropopause_pressure = result.reshape(batch_shape)
    
    return tropopause_pressure

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