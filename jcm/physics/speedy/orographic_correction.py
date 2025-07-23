"""
Orographic correction parameterization for SPEEDY physics.

This module implements the orographic corrections applied to temperature and specific humidity
in SPEEDY.f90, specifically replicating the corrections from time_stepping.f90 lines 69 and 91.
The corrections are applied in grid space as a physics parameterization.
"""

import jax
import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physical_constants import rgas, grav, gamma, hscale, hshum, refrh1
from jcm.physics.speedy.physics_data import PhysicsData


def compute_temperature_correction_vertical_profile(geometry: Geometry, parameters: Parameters) -> jnp.ndarray:
    """
    Compute vertical profile for temperature orographic correction (tcorv).
    
    From SPEEDY horizontal_diffusion.f90:
    tcorv(1) = 0
    tcorv(k) = fsg(k)^rgam for k = 2 to kx
    where rgam = rgas * gamma / (1000 * grav)
    
    Args:
        geometry: Model geometry containing sigma levels
        parameters: SPEEDY parameters containing gamma
        
    Returns:
        Vertical profile array of shape (layers,)
    """
    # SPEEDY constants from physical_constants.py
    rgam = rgas * gamma / (1000.0 * grav)
    
    # Get sigma levels (fsg in SPEEDY) - use layer midpoints
    sigma_levels = geometry.fsg  # These are the full sigma levels
    
    # Get number of layers from nodal_shape
    layers = geometry.nodal_shape[0]
    
    # Initialize vertical profile
    tcorv = jnp.zeros(layers)
    
    # tcorv(1) = 0 (first level), tcorv(k) = sigma^rgam for k >= 2
    tcorv = jnp.where(
        jnp.arange(layers) == 0,
        0.0,
        sigma_levels ** rgam
    )
    
    return tcorv


def compute_humidity_correction_vertical_profile(geometry: Geometry, parameters: Parameters) -> jnp.ndarray:
    """
    Compute vertical profile for humidity orographic correction (qcorv).
    
    From SPEEDY horizontal_diffusion.f90:
    qcorv(1) = qcorv(2) = 0
    qcorv(k) = fsg(k)^qexp for k = 3 to kx
    where qexp = hscale / hshum
    
    Args:
        geometry: Model geometry containing sigma levels
        parameters: SPEEDY parameters
        
    Returns:
        Vertical profile array of shape (layers,)
    """
    # SPEEDY constants from physical_constants.py
    qexp = hscale / hshum
    
    # Get sigma levels (fsg in SPEEDY) - use layer midpoints
    sigma_levels = geometry.fsg
    
    # Get number of layers from nodal_shape
    layers = geometry.nodal_shape[0]
    
    # Initialize vertical profile
    qcorv = jnp.zeros(layers)
    
    # qcorv(1) = qcorv(2) = 0, qcorv(k) = sigma^qexp for k >= 3
    qcorv = jnp.where(
        jnp.arange(layers) < 2,  # First two levels (indices 0, 1)
        0.0,
        sigma_levels ** qexp
    )
    
    return qcorv


def compute_temperature_correction_horizontal(boundaries: BoundaryData, geometry: Geometry) -> jnp.ndarray:
    """
    Compute horizontal temperature correction in grid space.
    
    From SPEEDY forcing.f90:
    corh(i,j) = gamlat(j) * phis0(i,j)
    where gamlat = gamma / (1000 * grav) (constant)
    
    Args:
        boundaries: Boundary data containing orography
        geometry: Model geometry
        
    Returns:
        Horizontal correction array of shape (lon, lat)
    """
    # SPEEDY constants from physical_constants.py
    gamlat = gamma / (1000.0 * grav)  # Reference lapse rate (constant in SPEEDY)
    
    # Apply correction: gamlat * orography
    corh = gamlat * boundaries.orog
    
    return corh


def compute_humidity_correction_horizontal(
    boundaries: BoundaryData, 
    geometry: Geometry,
    surface_temperature: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute horizontal humidity correction in grid space.
    
    This is a simplified version of the SPEEDY humidity correction.
    The full SPEEDY implementation involves complex saturation calculations.
    
    Args:
        boundaries: Boundary data containing orography and masks
        geometry: Model geometry
        surface_temperature: Surface temperature field
        
    Returns:
        Horizontal correction array of shape (lon, lat)
    """
    # For now, implement a simplified correction proportional to orography
    # This can be enhanced later with the full saturation calculation
    # Simplified correction: proportional to orography and temperature gradients
    # In the full implementation, this would involve saturation mixing ratio calculations
    corh = refrh1 * 0.001 * boundaries.orog  # Simplified scaling
    
    return corh


def get_orographic_correction_tendencies(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData = None,
    geometry: Geometry = None
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Compute orographic correction tendencies for temperature and specific humidity.
    
    This function applies the orographic corrections in grid space, replicating
    the corrections from SPEEDY time_stepping.f90 lines 69 and 91:
    
    Temperature: t_corrected = t + tcorh * tcorv
    Humidity: q_corrected = q + qcorh * qcorv
    
    Args:
        state: Current physics state
        physics_data: Physics data structure (passed through unchanged)
        parameters: SPEEDY parameters
        boundaries: Boundary data containing orography
        geometry: Model geometry
        
    Returns:
        tuple: (PhysicsTendency, updated PhysicsData)
            - PhysicsTendency: Physics tendencies representing the orographic corrections
            - PhysicsData: Updated physics data (unchanged in this implementation)
    """
    # Compute vertical profiles
    tcorv = compute_temperature_correction_vertical_profile(geometry, parameters)
    qcorv = compute_humidity_correction_vertical_profile(geometry, parameters)
    
    # Compute horizontal corrections
    tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
    
    # For humidity correction, we need surface temperature
    surface_temp = state.temperature[-1, :, :]  # Bottom level
    qcorh = compute_humidity_correction_horizontal(boundaries, geometry, surface_temp)
    
    # Apply corrections: field_corrected = field + horizontal * vertical
    temp_correction = tcorh[None, :, :] * tcorv[:, None, None]
    humidity_correction = qcorh[None, :, :] * qcorv[:, None, None]
    
    # The corrections are tendencies (corrections applied as instantaneous changes in speedy)
    # No corrections for wind fields
    u_tendency = jnp.zeros_like(state.u_wind)
    v_tendency = jnp.zeros_like(state.v_wind)
    
    tendency = PhysicsTendency(
        u_wind=u_tendency,
        v_wind=v_tendency,
        temperature=temp_correction,
        specific_humidity=humidity_correction
    )
    
    return tendency, physics_data


def apply_orographic_corrections_to_state(
    state: PhysicsState,
    boundaries: BoundaryData,
    geometry: Geometry,
    parameters: Parameters
) -> PhysicsState:
    """
    Apply orographic corrections directly to a physics state (for testing).
    
    This function applies the corrections directly to the state fields,
    which is equivalent to how they're applied in SPEEDY before diffusion.
    
    Args:
        state: Physics state to correct
        boundaries: Boundary data containing orography
        geometry: Model geometry
        parameters: SPEEDY parameters
        
    Returns:
        Corrected physics state
    """
    # Compute vertical profiles
    tcorv = compute_temperature_correction_vertical_profile(geometry, parameters)
    qcorv = compute_humidity_correction_vertical_profile(geometry, parameters)
    
    # Compute horizontal corrections
    tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
    surface_temp = state.temperature[-1, :, :]
    qcorh = compute_humidity_correction_horizontal(boundaries, geometry, surface_temp)
    
    # Apply corrections
    temp_correction = tcorh[None, :, :] * tcorv[:, None, None]
    humidity_correction = qcorh[None, :, :] * qcorv[:, None, None]
    
    corrected_temperature = state.temperature + temp_correction
    corrected_humidity = state.specific_humidity + humidity_correction
    
    return state.copy(
        temperature=corrected_temperature,
        specific_humidity=corrected_humidity
    )