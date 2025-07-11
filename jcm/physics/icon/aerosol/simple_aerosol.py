from typing import Tuple
import jax.numpy as jnp
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics.icon.icon_physics import PhysicsTendency
from jcm.physics_interface import PhysicsState
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.physics.icon.parameters import Parameters


def get_simple_aerosol(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData = None,
    geometry: Geometry = None
) -> Tuple[PhysicsTendency, PhysicsData]:
    """
    Apply MACv2-SP (Simple Plumes) aerosol scheme
    
    This implements the simplified aerosol parametrization based on
    Kinne et al. climatology with 9 anthropogenic plumes plus natural background.
    
    The scheme computes:
    - Aerosol optical depth (AOD) profiles
    - Single scattering albedo (SSA) profiles  
    - Asymmetry parameter profiles
    - Column-integrated properties
    - Twomey effect on cloud droplet number concentration
    """
        
    nlev, ncols = state.temperature.shape
    
    # Get grid coordinates - for testing, use representative locations
    # In a real implementation, these would come from the grid geometry
    if boundaries is not None and hasattr(boundaries, 'latitude'):
        lat = boundaries.latitude
        lon = boundaries.longitude
    else:
        # Test locations: East Asia, Europe, North America, Africa
        lat = jnp.array([25.0, 50.0, 40.0, 10.0])[:ncols]  # degrees North
        lon = jnp.array([120.0, 10.0, -80.0, 20.0])[:ncols]  # degrees East
    
    # Get height coordinate for vertical distribution
    height_full = physics_data.diagnostics.height_full
    
    # Reference wavelength for optical properties [nm]
    lambda_ref = 550.0
    
    # Initialize output arrays
    aod_profile = jnp.zeros((nlev, ncols))
    ssa_profile = jnp.zeros((nlev, ncols))
    asy_profile = jnp.zeros((nlev, ncols))
    aod_anthropogenic = jnp.zeros(ncols)
    aod_background = jnp.ones(ncols) * parameters.background_aod
    
    # Calculate normalized height coordinate (0 at surface, 1 at 15km)
    eta = jnp.maximum(0.0, jnp.minimum(1.0, height_full / 15000.0))
    
    # Loop over plumes to compute composite aerosol properties
    for iplume in range(parameters.nplumes):
        
        # Calculate vertical distribution using beta function
        beta_a_val = parameters.beta_a[iplume]
        beta_b_val = parameters.beta_b[iplume]
        
        # Beta function vertical profile (normalized)
        prof_unnorm = (eta**(beta_a_val - 1.0) * 
                      (1.0 - eta)**(beta_b_val - 1.0))
        
        # Normalize profile (integrate to 1)
        # Use layer thickness for proper integration
        layer_thickness = physics_data.diagnostics.layer_thickness
        prof_weighted = prof_unnorm * layer_thickness
        prof_sum = jnp.sum(prof_weighted, axis=0, keepdims=True)
        prof_normalized = jnp.where(prof_sum > 0.0, 
                                  prof_weighted / prof_sum, 
                                  0.0)
        
        # Calculate spatial weights for each column
        delta_lat = lat - parameters.plume_lat[iplume]
        delta_lon = lon - parameters.plume_lon[iplume]
        
        # Handle longitude wrapping (simplified)
        delta_lon = jnp.where(jnp.abs(delta_lon) > 180.0,
                             delta_lon - jnp.sign(delta_lon) * 360.0,
                             delta_lon)
        
        # Calculate spatial weights for each feature
        total_weight = jnp.zeros(ncols)
        
        for ifeature in range(parameters.nfeatures):
            # Choose extent parameters based on longitude direction
            sig_lon = jnp.where(delta_lon > 0.0,
                               parameters.sig_lon_E[ifeature, iplume],
                               parameters.sig_lon_W[ifeature, iplume])
            sig_lat = jnp.where(delta_lon > 0.0,
                               parameters.sig_lat_E[ifeature, iplume], 
                               parameters.sig_lat_W[ifeature, iplume])
            
            # Apply rotation
            theta_val = parameters.theta[ifeature, iplume]
            lon_rot = (jnp.cos(theta_val) * delta_lon + 
                      jnp.sin(theta_val) * delta_lat)
            lat_rot = (-jnp.sin(theta_val) * delta_lon + 
                      jnp.cos(theta_val) * delta_lat)
            
            # Calculate Gaussian spatial weight
            a_plume = 0.5 / (sig_lon**2)
            b_plume = 0.5 / (sig_lat**2)
            
            spatial_weight = jnp.exp(-1.0 * (a_plume * lon_rot**2 + 
                                           b_plume * lat_rot**2))
            
            # Apply feature weight
            feature_weight = parameters.ftr_weight[ifeature, iplume]
            total_weight += feature_weight * spatial_weight
        
        # Calculate column AOD contribution from this plume
        column_aod = total_weight * parameters.aod_spmx[iplume]
        
        # Calculate wavelength-dependent optical properties
        ssa = parameters.ssa550[iplume]
        asy = parameters.asy550[iplume]
        
        # Add this plume's contribution to profiles
        aod_contribution = prof_normalized * column_aod[jnp.newaxis, :]
        
        aod_profile += aod_contribution
        ssa_profile += aod_contribution * ssa
        asy_profile += aod_contribution * ssa * asy
        aod_anthropogenic += column_aod
    
    # Normalize SSA and asymmetry parameter profiles
    ssa_profile = jnp.where(aod_profile > 1e-10,
                           ssa_profile / aod_profile,
                           0.95)  # Default SSA
    
    asy_profile = jnp.where(ssa_profile * aod_profile > 1e-10,
                           asy_profile / (ssa_profile * aod_profile),
                           0.65)  # Default asymmetry parameter
    
    # Calculate total column AOD
    aod_total = aod_anthropogenic + aod_background
    
    # Calculate Twomey effect (simplified)
    # CDNC factor based on anthropogenic AOD
    cdnc_factor = 1.0 + 0.3 * jnp.log(1.0 + 10.0 * aod_anthropogenic)
    
    # Update aerosol data
    aerosol_data = physics_data.aerosol.copy(
        aod_profile=aod_profile,
        ssa_profile=ssa_profile,
        asy_profile=asy_profile,
        aod_total=aod_total,
        aod_anthropogenic=aod_anthropogenic,
        aod_background=aod_background,
        cdnc_factor=cdnc_factor
    )
    
    physics_data = physics_data.copy(aerosol=aerosol_data)
    
    # No direct tendencies from aerosol scheme
    # (aerosol effects are applied through radiation)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    
    return physics_tendencies, physics_data
