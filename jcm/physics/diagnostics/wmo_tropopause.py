"""WMO Tropopause Diagnostic for ECHAM Physics

This module implements the WMO (1957) tropopause definition following the
ICON mo_tropopause.f90 implementation. The tropopause is defined as the
lowest level at which the lapse rate decreases to 2°C per kilometer or less,
provided the average lapse rate between this level and all higher levels
within 2 kilometers does not exceed 2°C per kilometer.

"""

import jax
import jax.numpy as jnp
from typing import Optional
# The module alias, not ``from jcm.constants import physical_constants``:
# ``set_constants`` REBINDS that module global (PhysicalConstants is a
# NamedTuple, so it cannot be mutated in place), which leaves a captured
# reference pointing at the pre-override object — stale in exactly the same
# way a captured float would be (#772).
import jcm.constants as c

# WMO tropopause constants
GWMO = -0.002  # K/m - The -2°C/km threshold
DELTAZ = 2000.0  # m - The 2 km height interval for averaging
P_DEFAULT = 20000.0  # Pa - Default tropopause pressure (~200 hPa)

def compute_geopotential_height(pressure: jnp.ndarray, 
                              temperature: jnp.ndarray,
                              surface_pressure: jnp.ndarray) -> jnp.ndarray:
    """Compute geopotential height from pressure and temperature.

    Uses the hypsometric equation with proper handling of model levels.

    **Vertical convention — surface-first.** The level axis (last axis) must
    run surface-first: index 0 is the surface (highest pressure) and pressure
    decreases with index towards the model top. This routine prepends the
    surface pressure at index 0 and integrates the hypsometric equation
    *upward* with a ``cumsum`` along the level axis, so height is measured from
    the surface (``height[..., 0]`` is the thickness of the lowest layer, and
    height increases with index). Feeding it a top-first column (index 0 = model
    top) integrates the wrong way and pairs each layer's temperature with the
    wrong interface. The physics-internal and ECHAM input frames are BOTH
    top-first, so callers there must flip to surface-first at the boundary —
    see ``jcm/physics/diagnostics/aerocom.py`` and
    ``docs/source/design/output_vertical_conventions.md``.

    Args:
        pressure: Pressure at model levels [Pa] (shape: [..., nlev]),
            surface-first along the last axis.
        temperature: Temperature at model levels [K] (shape: [..., nlev]),
            surface-first along the last axis.
        surface_pressure: Surface pressure [Pa] (shape: [...])

    Returns:
        Geopotential height [m] (shape: [..., nlev]), surface-first,
        increasing with index.

    """
    # Read per call from the live singleton so set_constants applies (#772).
    g = c.grav
    R = c.rd
    
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
    """Compute temperature lapse rate dT/dz.
    
    Args:
        temperature: Temperature [K] (shape: [..., nlev])
        height: Geopotential height [m] (shape: [..., nlev])
        
    Returns:
        Lapse rate [K/m] (shape: [..., nlev-1])

    """
    # Compute finite differences
    dT = temperature[..., 1:] - temperature[..., :-1]
    dz = height[..., 1:] - height[..., :-1]
    
    # Avoid division by zero and ensure reasonable minimum height difference
    dz = jnp.where(jnp.abs(dz) < 1.0, jnp.sign(dz) * 1.0, dz)
    
    lapse_rate = dT / dz
    
    return lapse_rate

def find_tropopause_level(temperature: jnp.ndarray,
                         pressure: jnp.ndarray,
                         height: jnp.ndarray,
                         ncctop: int = 13,
                         nccbot: int = 35) -> jnp.ndarray:
    """Find the tropopause level following WMO definition.

    **Vertical convention — surface-first.** The level axis (last axis) must
    run surface-first: index 0 is the surface (highest pressure), index
    increasing towards the model top, and ``height`` increasing with index.
    The scan walks the search window from its low-index (near-surface) end
    upward and returns the pressure of the *first* level that satisfies the
    criteria — which is the WMO "lowest level" only when index 0 is the
    surface. On a top-first column it returns the *highest* qualifying level
    instead (a spurious ~30 hPa value on a real sounding, #841). The
    physics-internal frame is top-first, so callers there flip to surface-first
    at the boundary; see ``aerocom.py`` and
    ``docs/source/design/output_vertical_conventions.md``. No data-dependent
    orientation guard is applied here: orientation is a static property of the
    caller, and a traced flip inside ``jit`` would both cost and hide the bug.

    Args:
        temperature: Temperature [K] (shape: [..., nlev]), surface-first.
        pressure: Pressure [Pa] (shape: [..., nlev]), surface-first.
        height: Geopotential height [m] (shape: [..., nlev]), surface-first
            (increasing with index).
        ncctop: Surface-first slice START (inclusive, low index) — the
            near-surface / higher-pressure end of the search window.
        nccbot: Surface-first slice STOP (exclusive, high index) — the
            higher-altitude / lower-pressure end. ``[ncctop:nccbot]`` selects
            the levels the tropopause is searched within. The defaults encode
            the L47 grid; derive them from the actual level pressures on other
            grids.

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
            
            # Find levels within 2km above current level
            # We need to be careful about indexing: lapse_rate[i] represents the lapse rate
            # between height[i] and height[i+1], so we need to map height indices to lapse indices
            above_mask_height = (height_col >= current_height) & (height_col <= top_height)
            
            # For lapse rate averaging, we need to convert height mask to lapse mask
            # Lapse rate [i] corresponds to the layer between height[i] and height[i+1]
            # So if height level i is in the range, then lapse rate i-1 and i might be relevant
            above_mask_for_lapse = above_mask_height[1:]  # Skip first height level since no lapse rate before it
            
            # Get valid lapse rate values within the 2km window
            # Create zeros array with proper shape for broadcasting
            zeros_array = jnp.zeros_like(lapse_col)
            valid_lapse_values = jnp.where(above_mask_for_lapse, lapse_col, zeros_array)
            num_valid = jnp.sum(above_mask_for_lapse.astype(jnp.float32))
            
            # Compute average, handling division by zero
            avg_lapse = jnp.where(
                num_valid > 0,
                jnp.sum(valid_lapse_values) / jnp.maximum(num_valid, 1e-10),
                GWMO - 1.0  # Value that will fail the test
            )
            
            # Additional check: require some temperature variation to avoid 
            # false positives in isothermal atmospheres
            # Check temperature range over 2km span
            temp_range_2km = jnp.max(jnp.where(above_mask_height, temp_col, temp_col[k])) - \
                           jnp.min(jnp.where(above_mask_height, temp_col, temp_col[k]))
            has_temp_variation = temp_range_2km > 1.0  # At least 1K variation over 2km
            
            # Check if we're at a reasonable tropopause height (> 5km typically for flexibility)
            reasonable_height = current_height > 5000.0
            
            # All criteria must be satisfied for a valid tropopause
            # Include temperature variation to avoid false positives in isothermal atmospheres
            both_ok = lapse_ok & (avg_lapse >= GWMO) & (num_valid > 0) & reasonable_height & has_temp_variation
            
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
    if len(batch_shape) == 0:
        # Single column case
        result = find_tropopause_column(search_temp, search_pressure, search_height, lapse_rate)
    else:
        # Multi-column case: flatten batch dimensions for processing
        flat_batch_size = 1
        for dim in batch_shape:
            flat_batch_size *= dim
        
        # Reshape to (flat_batch_size, nlev)
        search_temp_flat = search_temp.reshape(flat_batch_size, nlev_search)
        search_pressure_flat = search_pressure.reshape(flat_batch_size, nlev_search)
        search_height_flat = search_height.reshape(flat_batch_size, nlev_search)
        lapse_rate_flat = lapse_rate.reshape(flat_batch_size, nlev_lapse)
        
        # Process each column
        result_flat = jax.vmap(find_tropopause_column)(
            search_temp_flat, search_pressure_flat, search_height_flat, lapse_rate_flat
        )
        
        # Reshape back to batch shape
        result = result_flat.reshape(batch_shape)
    
    return result

def wmo_tropopause(temperature: jnp.ndarray,
                  pressure: jnp.ndarray,
                  surface_pressure: jnp.ndarray,
                  previous_tropopause: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute WMO tropopause pressure diagnostic.
    
    Follows the WMO (1957) definition: the tropopause is the lowest level
    at which the lapse rate decreases to 2°C per kilometer or less,
    provided the average lapse rate between this level and all higher
    levels within 2 kilometers does not exceed 2°C per kilometer.

    Inputs must be **surface-first** (level axis index 0 = surface); see
    :func:`find_tropopause_level` and :func:`compute_geopotential_height` for
    why, and for how top-first callers convert at the boundary.

    Args:
        temperature: Temperature at model levels [K] (shape: [..., nlev]),
            surface-first along the last axis.
        pressure: Pressure at model levels [Pa] (shape: [..., nlev]),
            surface-first along the last axis.
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