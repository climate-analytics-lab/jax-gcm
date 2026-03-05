from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics.icon.parameters import Parameters
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics_interface import PhysicsState, PhysicsTendency
import jax.numpy as jnp
from jax import jit

@jit
def apply_forcing_data(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData,
    terrain: TerrainData
) -> tuple[PhysicsTendency, PhysicsData]:
    """
    Compute time-varying boundary conditions for ICON physics
    
    This function updates the boundary conditions with time-varying values
    for solar forcing, greenhouse gases, and surface properties.
    
    Args:
        boundaries: Current boundary conditions
        terrain: TerrainData object containing terrain/surface data
        day_of_year: Day of year (1-365)
        time_of_day: Time of day (hours, 0-24)
        year: Year (for solar variability)
        
    Returns:
        Updated boundary conditions
    """
    
    # Get expected output shape from physics_data (column format)
    ncols = physics_data.surface.surface_temperature.shape[0]

    # Compute surface properties based on existing masks
    surface_albedo_vis, surface_albedo_nir, surface_emissivity = _compute_surface_properties(
        terrain.fmask,  # Land fraction
        forcing.sice_am[..., 0] if forcing.sice_am.ndim == 3 else forcing.sice_am,  # Sea ice
    )

    # Surface temperature (use existing SST for ocean, land temperature for land)
    land_temp = forcing.stl_am[..., 0] if forcing.stl_am.ndim == 3 else forcing.stl_am
    sst = forcing.sea_surface_temperature[..., 0] if forcing.sea_surface_temperature.ndim == 3 else forcing.sea_surface_temperature
    surface_temperature = jnp.where(
        terrain.fmask > 0.5,  # Land
        land_temp,  # Land temp
        sst  # SST
    )

    # Roughness length (higher over land)
    roughness_length = jnp.where(
        terrain.fmask > 0.5,  # Land
        0.01,  # 1 cm over land
        0.0001  # 0.1 mm over ocean
    )

    # Reshape surface properties from grid (nlon, nlat) to column (ncols) format
    surface_albedo_vis = surface_albedo_vis.reshape(ncols)
    surface_albedo_nir = surface_albedo_nir.reshape(ncols)
    surface_emissivity = surface_emissivity.reshape(ncols)
    surface_temperature = surface_temperature.reshape(ncols)
    roughness_length = roughness_length.reshape(ncols)
    
    # Greenhouse gas concentrations (uniform for now)
    # Fill arrays with constant values to maintain array shapes
    co2_concentration = 420.0  # ppmv
    ch4_concentration = 1900.0  # ppbv
    o3_concentration = 300.0  # ppbv

    # Sea ice fraction (from existing data)
    #TODO: use these somewhere
    sea_ice_fraction = forcing.sice_am[..., 0] if forcing.sice_am.ndim == 3 else forcing.sice_am
    sea_ice_thickness = jnp.where(sea_ice_fraction > 0.1, 1.0, 0.0)  # 1m where ice exists

    tendencies = PhysicsTendency.zeros(state.temperature.shape)

    radiation_data = physics_data.radiation.copy(
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
    )
    # Fill chemistry arrays with constant values (maintaining original shapes)
    chemistry_data = physics_data.chemistry.copy(
        co2_vmr=jnp.ones_like(physics_data.chemistry.co2_vmr) * co2_concentration,
        methane_vmr=jnp.ones_like(physics_data.chemistry.methane_vmr) * ch4_concentration * 1e-3,  # Convert ppbv to ppmv
        ozone_vmr=jnp.ones_like(physics_data.chemistry.ozone_vmr) * o3_concentration * 1e-3,    # Convert ppbv to ppmv
    )
    surface_data = physics_data.surface.copy(
        surface_temperature=surface_temperature,
        skin_temperature=surface_temperature,
        roughness_length=roughness_length,
    )
    updated_physics_data = physics_data.copy(
        radiation=radiation_data,
        surface=surface_data,
        chemistry=chemistry_data
    )
    return tendencies, updated_physics_data




def _compute_surface_properties(
    land_fraction: jnp.ndarray,
    sea_ice_fraction: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute surface optical properties"""
    # Default values 
    # TODO: Pull these out into parameters - or add them to ForcingData as an input
    land_albedo_vis = 0.15
    land_albedo_nir = 0.25
    land_emissivity = 0.95
    
    ocean_albedo_vis = 0.05
    ocean_albedo_nir = 0.05
    ocean_emissivity = 0.98
    
    seaice_albedo_vis = 0.80
    seaice_albedo_nir = 0.70
    seaice_emissivity = 0.95
    
    # Ocean fraction
    ocean_fraction = 1.0 - land_fraction - sea_ice_fraction
    ocean_fraction = jnp.maximum(ocean_fraction, 0.0)
    
    # Weighted average of surface properties
    albedo_vis = (land_fraction * land_albedo_vis +
                  ocean_fraction * ocean_albedo_vis +
                  sea_ice_fraction * seaice_albedo_vis)
    
    albedo_nir = (land_fraction * land_albedo_nir +
                  ocean_fraction * ocean_albedo_nir +
                  sea_ice_fraction * seaice_albedo_nir)
    
    emissivity = (land_fraction * land_emissivity +
                  ocean_fraction * ocean_emissivity +
                  sea_ice_fraction * seaice_emissivity)
    
    return albedo_vis, albedo_nir, emissivity