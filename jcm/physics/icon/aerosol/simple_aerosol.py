from typing import Tuple
import jax.numpy as jnp
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from .aerosol_params import AerosolParameters


def get_simple_aerosol(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: AerosolParameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> Tuple[PhysicsTendency, PhysicsData]:
    """Apply MACv2-SP (Simple Plumes) aerosol scheme
    
    This implements the simplified aerosol parametrization based on
    Kinne et al. climatology with 9 anthropogenic plumes plus natural background.
    
    The scheme computes:
    - Aerosol optical depth (AOD) profiles
    - Single scattering albedo (SSA) profiles  
    - Asymmetry parameter profiles
    - Column-integrated properties
    - Twomey effect on cloud droplet number concentration
    """
    import jax
        
    nlev, ncols = state.temperature.shape
    aerosol_params = parameters.aerosol
    
    # Get grid coordinates from cached coordinates
    lat, lon = jax.numpy.meshgrid(
        physics_data.icon_coords.lat * 180.0 / jnp.pi,  # Convert to degrees
        physics_data.icon_coords.lon * 180.0 / jnp.pi,  # degrees
    )
    # Then reshape to (ncols,) to match column format
    lats = lat.reshape(ncols)
    lons = lon.reshape(ncols)
    
    # Get height coordinate for vertical distribution
    height_full = physics_data.diagnostics.height_full
    
    # Initialize output arrays
    aod_profile = jnp.zeros((nlev, ncols))
    ssa_profile = jnp.zeros((nlev, ncols))
    asy_profile = jnp.zeros((nlev, ncols))
    
    # Get temporal weights from forcing data (allows time-varying emissions)
    year_weight = forcing.aerosol_year_weight
    ann_cycle = forcing.aerosol_ann_cycle

    # Compute the plume spatial distribution once; ``get_anthropogenic_aod`` /
    # ``get_background_aod`` used to recompute it internally so the Gaussian
    # evaluation ran three times per step. Pass it through instead.
    spatial_dist = get_plume_spatial_distribution(lats, lons, aerosol_params)

    # Calculate anthropogenic and background AOD using vectorized operations
    aod_anthropogenic = get_anthropogenic_aod(
        aerosol_params, year_weight, ann_cycle, spatial_dist
    )
    aod_background = get_background_aod(aerosol_params, ann_cycle, spatial_dist)

    # Calculate vertical profiles for each plume using vectorized operations
    plume_profiles = get_vertical_profiles(height_full, aerosol_params)
    
    # Combine plume contributions using vectorized operations
    # plume_profiles: (nplumes, nlev, ncols)
    # spatial_dist: (nplumes, ncols)
    # aod_anthropogenic: (ncols,)
    # Need to broadcast properly for multiplication
    plume_contribution = jnp.sum(
        plume_profiles * 
        (aod_anthropogenic[jnp.newaxis, jnp.newaxis, :] * spatial_dist[:, jnp.newaxis, :]),
        axis=0
    )
    
    # Add background contribution with uniform vertical distribution
    bg_profile = get_background_vertical_profile(height_full)
    bg_contribution = bg_profile[:, jnp.newaxis] * aod_background[jnp.newaxis, :]
    
    # Combine anthropogenic and background contributions
    aod_profile = plume_contribution + bg_contribution
    
    # Calculate optical properties using weighted averages
    ssa_profile, asy_profile, angstrom = get_optical_properties(
        aod_profile, spatial_dist, aerosol_params
    )

    # Calculate total column AOD
    aod_total = jnp.sum(aod_profile, axis=0)

    # Calculate Twomey effect using proper CDNC relationship
    cdnc_factor = get_CDNC(aod_anthropogenic) / get_CDNC(jnp.zeros_like(aod_anthropogenic))

    # Update aerosol data
    aerosol_data = physics_data.aerosol.copy(
        aod_profile=aod_profile,
        ssa_profile=ssa_profile,
        asy_profile=asy_profile,
        aod_total=aod_total,
        aod_anthropogenic=aod_anthropogenic,
        aod_background=aod_background,
        cdnc_factor=cdnc_factor,
        angstrom=angstrom
    )
    
    physics_data = physics_data.copy(aerosol=aerosol_data)
    
    # No direct tendencies from aerosol scheme
    # (aerosol effects are applied through radiation)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    
    return physics_tendencies, physics_data

def get_plume_spatial_distribution(lats, lons, parameters):
    """Calculate spatial distribution of aerosol plumes using Gaussian functions
    
    Args:
        lats: Array of latitudes [degrees]
        lons: Array of longitudes [degrees]
        parameters: AerosolParameters object
        
    Returns:
        Spatial distribution array of shape (nplumes, ncols)

    """
    # Expand dimensions for vectorized operations
    # lats, lons: (ncols,)
    # parameters.*: (nplumes,) or (nfeatures, nplumes)
    
    # get plume-center relative spatial parameters for specifying amplitude of plume at given lat and lon
    delta_lat = lats[jnp.newaxis, :] - parameters.plume_lat[:, jnp.newaxis]  # (nplumes, ncols)
    delta_lon = lons[jnp.newaxis, :] - parameters.plume_lon[:, jnp.newaxis]  # (nplumes, ncols)

    delta_lon_t = jnp.ones_like(parameters.plume_lon) * 180
    delta_lon_t = delta_lon_t.at[0].set(260)  # First plume is different

    # Deal with wrapping
    delta_lon = jnp.where(
        jnp.abs(delta_lon) > delta_lon_t[:, jnp.newaxis], 
        jnp.where(delta_lon >= 0, delta_lon - 360, delta_lon + 360), 
        delta_lon
    )

    # Vectorized calculation for all features and plumes
    # parameters.sig_*: (nfeatures, nplumes)
    # delta_lon: (nplumes, ncols)
    # Need to broadcast: (nfeatures, nplumes, ncols)
    
    sig_lon = jnp.where(
        delta_lon[jnp.newaxis, :, :] > 0.0,  # (1, nplumes, ncols)
        parameters.sig_lon_E[:, :, jnp.newaxis],  # (nfeatures, nplumes, 1)
        parameters.sig_lon_W[:, :, jnp.newaxis]   # (nfeatures, nplumes, 1)
    )
    
    sig_lat = jnp.where(
        delta_lon[jnp.newaxis, :, :] > 0.0,  # (1, nplumes, ncols)
        parameters.sig_lat_E[:, :, jnp.newaxis],  # (nfeatures, nplumes, 1)
        parameters.sig_lat_W[:, :, jnp.newaxis]   # (nfeatures, nplumes, 1)
    )
    
    a_plume = 0.5 / (sig_lon**2)
    b_plume = 0.5 / (sig_lat**2)

    # adjust for a plume specific rotation which helps match plume state to climatology.
    # Rotation per feature and plume
    cos_theta = jnp.cos(parameters.theta)[:, :, jnp.newaxis]  # (nfeatures, nplumes, 1)
    sin_theta = jnp.sin(parameters.theta)[:, :, jnp.newaxis]  # (nfeatures, nplumes, 1)
    
    lon_rot = (cos_theta * delta_lon[jnp.newaxis, :, :] + 
               sin_theta * delta_lat[jnp.newaxis, :, :])  # (nfeatures, nplumes, ncols)
    lat_rot = (-sin_theta * delta_lon[jnp.newaxis, :, :] + 
               cos_theta * delta_lat[jnp.newaxis, :, :])  # (nfeatures, nplumes, ncols)

    # Calculate Gaussian distribution for each feature
    gaussian = jnp.exp(-1.0 * (a_plume * (lon_rot**2) + b_plume * (lat_rot**2)))
    
    # Weight by feature importance and sum over features
    weighted_gaussian = parameters.ftr_weight[:, :, jnp.newaxis] * gaussian
    
    return jnp.sum(weighted_gaussian, axis=0)  # (nplumes, ncols)


def get_background_aod(parameters, ann_cycle, spatial_dist, constant_background=0.02):
    """Calculate background (pre-industrial) aerosol optical depth.

    Args:
        parameters: AerosolParameters object
        ann_cycle: Annual cycle weights (nplumes,) from forcing data
        spatial_dist: Precomputed plume Gaussian distribution (nplumes, ncols)
        constant_background: Constant background AOD value

    Returns:
        Background AOD array of shape (ncols,)

    """
    cw_bg = ann_cycle[:, jnp.newaxis] * parameters.aod_fmbg[:, jnp.newaxis] * spatial_dist

    # calculate contribution to plume from its different features, to get a column weight for the anthropogenic
    #   (cw_an) and the fine-mode background aerosol (cw_bg)
    aod_PI = jnp.sum(cw_bg, axis=0) + constant_background

    return aod_PI


def get_anthropogenic_aod(parameters, year_weight, ann_cycle, spatial_dist):
    """Calculate anthropogenic aerosol optical depth.

    Args:
        parameters: AerosolParameters object
        year_weight: Year-specific emission weights (nplumes,) from forcing data
        ann_cycle: Annual cycle weights (nplumes,) from forcing data
        spatial_dist: Precomputed plume Gaussian distribution (nplumes, ncols)

    Returns:
        Anthropogenic AOD array of shape (ncols,)

    """
    # Use time weights for anthropogenic emissions
    time_weight = year_weight * ann_cycle
    cw_an = time_weight[:, jnp.newaxis] * parameters.aod_spmx[:, jnp.newaxis] * spatial_dist

    aod_anth = jnp.sum(cw_an, axis=0)
    return aod_anth


def get_vertical_profiles(height_full, parameters):
    """Calculate vertical profiles for all plumes using beta function distribution
    
    Args:
        height_full: Height coordinate array of shape (nlev, ncols)
        parameters: AerosolParameters object
        
    Returns:
        Vertical profiles array of shape (nplumes, nlev, ncols)

    """
    # Normalize height to 0-1 range (0 at surface, 1 at 15km)
    height_norm = jnp.clip(height_full / 15000.0, 0.0, 1.0)
    
    # Calculate beta function profiles for each plume
    # height_norm: (nlev, ncols)
    # parameters.beta_a, parameters.beta_b: (nplumes,)
    
    # Expand dimensions for vectorized calculation
    # height_norm: (1, nlev, ncols)
    # beta_a, beta_b: (nplumes, 1, 1)
    height_expanded = height_norm[jnp.newaxis, :, :]
    beta_a_expanded = parameters.beta_a[:, jnp.newaxis, jnp.newaxis]
    beta_b_expanded = parameters.beta_b[:, jnp.newaxis, jnp.newaxis]
    
    # Calculate beta function: x^(a-1) * (1-x)^(b-1)
    # Avoid issues at boundaries by adding small epsilon
    eps = 1e-10
    x = jnp.clip(height_expanded, eps, 1.0 - eps)
    
    beta_profile = (x**(beta_a_expanded - 1)) * ((1 - x)**(beta_b_expanded - 1))
    
    # Normalize profiles to integrate to 1 over height
    profile_sum = jnp.sum(beta_profile, axis=1, keepdims=True)
    profile_sum = jnp.where(profile_sum > 0, profile_sum, 1.0)  # Avoid division by zero
    
    normalized_profiles = beta_profile / profile_sum
    
    return normalized_profiles  # (nplumes, nlev, ncols)


def get_background_vertical_profile(height_full):
    """Calculate vertical profile for background aerosol
    
    Args:
        height_full: Height coordinate array of shape (nlev, ncols)
        
    Returns:
        Background vertical profile array of shape (nlev,)

    """
    # Simple exponential decay for background aerosol
    # Use mean height profile across columns
    height_mean = jnp.mean(height_full, axis=1)
    
    # Exponential decay with 2km scale height
    scale_height = 2000.0  # meters
    profile = jnp.exp(-height_mean / scale_height)
    
    # Normalize to integrate to 1
    profile = profile / jnp.sum(profile)
    
    return profile


def get_optical_properties(aod_profile, spatial_dist, parameters):
    """Calculate single scattering albedo, asymmetry parameter, and Angstrom exponent

    Args:
        aod_profile: AOD profile array of shape (nlev, ncols)
        spatial_dist: Spatial distribution array of shape (nplumes, ncols)
        parameters: AerosolParameters object

    Returns:
        Tuple of (ssa_profile, asy_profile, angstrom) where profiles are
        (nlev, ncols) and angstrom is (ncols,)

    """
    # Weight optical properties by AOD contribution from each plume
    # aod_profile: (nlev, ncols)
    # spatial_dist: (nplumes, ncols)
    # parameters.ssa550, parameters.asy550, parameters.angstrom: (nplumes,)

    # Calculate plume contributions to total AOD
    total_aod = jnp.sum(aod_profile, axis=0, keepdims=True)  # (1, ncols)
    total_aod = jnp.where(total_aod > 0, total_aod, 1.0)  # Avoid division by zero

    # Weight by spatial distribution
    plume_weights = spatial_dist / jnp.sum(spatial_dist, axis=0, keepdims=True)

    # Calculate weighted optical properties
    ssa_weighted = jnp.sum(
        plume_weights * parameters.ssa550[:, jnp.newaxis],
        axis=0
    )
    asy_weighted = jnp.sum(
        plume_weights * parameters.asy550[:, jnp.newaxis],
        axis=0
    )
    angstrom_weighted = jnp.sum(
        plume_weights * parameters.angstrom[:, jnp.newaxis],
        axis=0
    )

    # Expand to full vertical profile
    ssa_profile = jnp.ones_like(aod_profile) * ssa_weighted[jnp.newaxis, :]
    asy_profile = jnp.ones_like(aod_profile) * asy_weighted[jnp.newaxis, :]

    return ssa_profile, asy_profile, angstrom_weighted


def get_CDNC(AOD, A=60, B=20):
    """Derive CDNC from AOD using a relationship of the form: CDNC = A * ln(B*AOD + 1)
    Ross' amazon work: A=410 B=5
    MODIS original: A=16 B=1000
    AEROCOM P1 original: A=60, B=20
    """
    return 1 + A * jnp.log(B * AOD + 1)
