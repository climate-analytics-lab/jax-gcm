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
    
    # Get number of layers - use shape of sigma_levels to avoid accessing nodal_shape
    layers = sigma_levels.shape[0]
    
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
    qexp = hscale / hshum
    
    # Get sigma levels - use layer midpoints
    sigma_levels = geometry.fsg
    
    # Get number of layers - use shape of sigma_levels to avoid accessing nodal_shape
    layers = sigma_levels.shape[0]
    
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
    gamlat = gamma / (1000.0 * grav)  # Reference lapse rate (constant in SPEEDY)
    
    # Apply correction: gamlat * phis0 (spectrally-filtered surface geopotential)
    corh = gamlat * geometry.phis0
    
    return corh


def compute_humidity_correction_horizontal(
    boundaries: BoundaryData, 
    geometry: Geometry,
    temperature_correction: jnp.ndarray,
    land_temperature: jnp.ndarray,
    day: int = 0
) -> jnp.ndarray:
    """
    Compute horizontal humidity correction in grid space.
    
    This replicates the full SPEEDY humidity correction from forcing.f90:
    1. Calculate surface temperature (land/sea mixture)
    2. Calculate reference temperature with orographic correction
    3. Calculate pressure adjustment
    4. Calculate saturation specific humidity at both conditions
    5. Apply humidity correction: corh = refrh1 * (qref - qsfc)
    
    Args:
        boundaries: Boundary data containing orography and masks
        geometry: Model geometry
        temperature_correction: Horizontal temperature correction (tcorh)
        land_temperature: Land surface temperature from land model
        day: day of year (for SST)
        
    Returns:
        Horizontal correction array of shape (lon, lat)
    """
    from jcm.physics.speedy.humidity import get_qsat
    
    # 1. Calculate surface temperature (land/sea mixture)
    # tsfc = fmask * stl_am + (1 - fmask) * sst_am
    tsfc = boundaries.fmask * land_temperature + (1.0 - boundaries.fmask) * boundaries.tsea[:,:,day]
    
    # 2. Calculate reference temperature with orographic correction
    # tref = tsfc + corh (where corh is the temperature correction)
    tref = tsfc + temperature_correction
    
    # 3. Calculate pressure adjustment due to temperature difference
    # In SPEEDY: pexp = 1./(rgas * gamlat(j)), but gamlat is constant = gamma/(1000*grav)
    # So pexp = 1000*grav/(rgas*gamma)
    pexp = 1000.0 * grav / (rgas * gamma)
    
    # psfc = (tsfc/tref)^pexp
    psfc = (tsfc / tref) ** pexp
    
    # 4. Calculate saturation specific humidity at reference and surface conditions
    # Note: get_qsat expects (temperature, normalized_pressure, sigma_level)
    # For surface calculations, we use sigma=-1.0 and 1.0 as in SPEEDY
    
    # Create dummy normalized pressure (psfc/psfc = 1.0 everywhere)
    normalized_pressure = jnp.ones_like(psfc)
    
    # qref = get_qsat(tref, psfc/psfc, -1.0) - reference conditions
    qref = get_qsat(tref, normalized_pressure, -1.0)
    
    # qsfc = get_qsat(tsfc, psfc, 1.0) - surface conditions  
    qsfc = get_qsat(tsfc, psfc, 1.0)
    
    # 5. Calculate humidity correction
    # corh = refrh1 * (qref - qsfc)
    corh = refrh1 * (qref - qsfc)
    
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
    
    # For humidity correction, we need the temperature correction and land temperature
    # Get land temperature from physics data (land model)
    land_temperature = physics_data.land_model.stl_am
    qcorh = compute_humidity_correction_horizontal(boundaries, geometry, tcorh, land_temperature, physics_data.date.model_day())
    
    # Apply corrections: field_corrected = field + horizontal * vertical
    temp_correction = tcorh * tcorv[:, None, None]
    humidity_correction = qcorh * qcorv[:, None, None]
    
    # In SPEEDY, these corrections are applied instantaneously every timestep before diffusion
    # To replicate this in JAX-GCM's tendency framework, we convert the instantaneous
    # correction to an equivalent tendency by dividing by the model timestep.
    # This ensures the same total correction is applied over one integration step.
    
    # Use the actual model timestep from the physics_data for faithful reproduction of SPEEDY
    model_timestep_seconds = physics_data.date.dt_seconds
    temp_tendency = temp_correction / model_timestep_seconds
    humidity_tendency = humidity_correction / model_timestep_seconds
    
    # No corrections for wind fields
    u_tendency = jnp.zeros_like(state.u_wind)
    v_tendency = jnp.zeros_like(state.v_wind)
    
    tendency = PhysicsTendency(
        u_wind=u_tendency,
        v_wind=v_tendency,
        temperature=temp_tendency,
        specific_humidity=humidity_tendency
    )
    
    return tendency, physics_data


def apply_orographic_corrections_to_state(
    state: PhysicsState,
    boundaries: BoundaryData,
    geometry: Geometry,
    parameters: Parameters,
    land_temperature: jnp.ndarray = None,
    day: int = 0
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
        land_temperature: Land surface temperature (if None, uses a default value)
        day: day of year (for SST)
        
    Returns:
        Corrected physics state
    """
    # Compute vertical profiles
    tcorv = compute_temperature_correction_vertical_profile(geometry, parameters)
    qcorv = compute_humidity_correction_vertical_profile(geometry, parameters)
    
    # Compute horizontal corrections
    tcorh = compute_temperature_correction_horizontal(boundaries, geometry)
    
    # For humidity correction, use provided land temperature or default value
    if land_temperature is None:
        # Use a default land temperature (288K) for testing
        land_temperature = jnp.full(boundaries.orog.shape, 288.0)
    
    qcorh = compute_humidity_correction_horizontal(boundaries, geometry, tcorh, land_temperature, day)
    
    # Apply corrections
    temp_correction = tcorh * tcorv[:, None, None]
    humidity_correction = qcorh * qcorv[:, None, None]
    
    corrected_temperature = state.temperature + temp_correction
    corrected_humidity = state.specific_humidity + humidity_correction
    
    return state.copy(
        temperature=corrected_temperature,
        specific_humidity=corrected_humidity
    )