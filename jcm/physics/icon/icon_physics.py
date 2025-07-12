"""
Main ICON Physics class for JAX-GCM

This module contains the main IconPhysics class that orchestrates the 
ICON atmospheric physics parameterizations. It follows the same pattern
as SpeedyPhysics but implements the ICON physics suite.

Date: 2025-01-09
"""

import jax
from jax import jit
import jax.numpy as jnp
import tree_math
from collections import abc
from typing import Callable, Tuple, Optional
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
from jcm.physics.icon.constants import physical_constants

# Import physics modules (will be implemented progressively)
from jcm.physics.icon.radiation import radiation_scheme, RadiationDiagnostics
from jcm.physics.icon.convection import tiedtke_nordeng_convection, ConvectionTendencies
from jcm.physics.icon.clouds import shallow_cloud_scheme, cloud_microphysics
from jcm.physics.icon.parameters import Parameters
from jcm.physics.icon.vertical_diffusion import vertical_diffusion_scheme
from jcm.physics.icon.surface import surface_physics_step
from jcm.physics.icon.gravity_waves import gravity_wave_drag
# from jcm.physics.icon.chemistry import chemistry_tendencies
from jcm.physics.icon.aerosol.simple_aerosol import get_simple_aerosol
from jcm.physics.icon.icon_physics_data import PhysicsData

class IconPhysics(Physics):
    """
    ICON atmospheric physics implementation for JAX-GCM
    
    This class implements the ICON physics suite including:
    - Radiation (shortwave and longwave)
    - Convection (Tiedtke-Nordeng scheme)
    - Large-scale cloud microphysics
    - Vertical diffusion and boundary layer
    - Surface fluxes and land model
    - Gravity wave drag
    - Simple chemistry schemes
    """
    
    def __init__(self, 
                 write_output: bool = True,
                 checkpoint_terms: bool = True,
                 parameters: Optional[Parameters] = None):
        """
        Initialize the ICON physics.
        
        Args:
            write_output: Whether to write physics output to predictions
            checkpoint_terms: Whether to checkpoint physics terms
            parameters: Optional physics parameters (uses defaults if None)
        """
        self.write_output = write_output
        self.checkpoint_terms = checkpoint_terms
        
        # Store parameters
        self.parameters = parameters or Parameters.default()
        
        # Build list of physics terms
        self.terms = [
            _prepare_common_physics_state,
            apply_convection, 
            apply_clouds,
            apply_microphysics,
            get_simple_aerosol,
            apply_radiation, 
            apply_vertical_diffusion, 
            apply_surface,
            apply_gravity_waves
        ]
    
    def compute_tendencies(
        self,
        state: PhysicsState,
        boundaries: BoundaryData,
        geometry: Geometry,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
        """
        Compute the physical tendencies given the current state and data structs. Loops through the Speedy physics terms, accumulating the tendencies.

        Args:
            state: Current state variables
            parameters: Parameters object
            boundaries: Boundary data
            geometry: Geometry data
            date: Date data

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data (PhysicsData format)
        """
        physics_data = PhysicsData.zeros(
            geometry.nodal_shape[1:],
            geometry.nodal_shape[0],
            date=date
        )

        # Initialize zero tendencies with tracer tendencies
        tracer_tends = {name: jnp.zeros_like(tracer) for name, tracer in state.tracers.items()}
        tendencies = PhysicsTendency.zeros(state.temperature.shape, tracers=tracer_tends)

        # Get array dimensions for vectorization
        nlev, nlat, nlon = state.temperature.shape
        ncols = nlat * nlon
        
        # Create vectorized physics state for column-wise processing
        # Reshape from [nlev, nlat, nlon] to [nlev, ncols]
        # Also reshape tracers
        vectorized_tracers = {}
        for name, tracer in state.tracers.items():
            vectorized_tracers[name] = tracer.reshape(nlev, ncols)
            
        vectorized_state = PhysicsState(
            u_wind=state.u_wind.reshape(nlev, ncols),
            v_wind=state.v_wind.reshape(nlev, ncols),
            temperature=state.temperature.reshape(nlev, ncols),
            specific_humidity=state.specific_humidity.reshape(nlev, ncols),
            geopotential=state.geopotential.reshape(nlev, ncols),
            surface_pressure=state.surface_pressure.reshape(ncols),
            tracers=vectorized_tracers
        )
        
        # Apply physics terms sequentially with vectorization
        for term in self.terms:
            if self.checkpoint_terms:
                term = jax.checkpoint(term)
            
            # Apply term to vectorized state
            term_tendency, physics_data = term(
                vectorized_state, physics_data, self.parameters, boundaries, geometry
            )
            
            # Reshape tendencies back to 3D and accumulate
            # Also reshape tracer tendencies, ensuring all tracers are present
            reshaped_tracers = {}
            # First, initialize with zeros for all tracers in the state
            for name in state.tracers.keys():
                if name in term_tendency.tracers:
                    reshaped_tracers[name] = term_tendency.tracers[name].reshape(nlev, nlat, nlon)
                else:
                    # If this physics term doesn't have a tendency for this tracer, use zeros
                    reshaped_tracers[name] = jnp.zeros((nlev, nlat, nlon))
                
            reshaped_tendency = PhysicsTendency(
                u_wind=term_tendency.u_wind.reshape(nlev, nlat, nlon),
                v_wind=term_tendency.v_wind.reshape(nlev, nlat, nlon),
                temperature=term_tendency.temperature.reshape(nlev, nlat, nlon),
                specific_humidity=term_tendency.specific_humidity.reshape(nlev, nlat, nlon),
                tracers=reshaped_tracers
            )
            
            tendencies = tendencies + reshaped_tendency
        
        return tendencies, physics_data
    

@jit
def _prepare_common_physics_state(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:

    """
    Prepare common physics variables that are used by multiple physics modules.
    
    This reduces code duplication by computing pressure levels, heights, air density,
    and other commonly needed variables once for all physics modules.
    
    Args:
        state: Physics state variables (already in 2D format [nlev, ncols])
        boundaries: Boundary conditions
        geometry: Model geometry
        
    Returns:
        Dictionary with common physics variables
    """
    from jcm.physics.speedy.physical_constants import p0
    
    # Calculate pressure levels from surface pressure and sigma coordinates
    surface_pressure = state.surface_pressure * p0  # Convert to Pa
    sigma_levels = geometry.fsg  # sigma coordinates at level centers
    pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
    # Calculate pressure at interfaces (half levels)
    sigma_half = geometry.hsg
    pressure_half = sigma_half[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
    
    # Convert geopotential to height
    height_levels = state.geopotential / physical_constants.grav
    
    # Calculate height at interfaces (approximate using hydrostatic)
    dz = jnp.diff(height_levels, axis=0, prepend=height_levels[0:1, :] + 1000.0)
    height_half = jnp.cumsum(dz, axis=0) - dz
    
    # Calculate air density
    rho = pressure_levels / (physical_constants.rd * state.temperature)
    
    # Calculate layer thickness
    dp = jnp.diff(pressure_half, axis=0)
    dz_full = -dp / (rho * physical_constants.grav)
    
    # Calculate relative humidity
    es = 611.2 * jnp.exp(17.67 * (state.temperature - 273.15) / (state.temperature - 29.65))
    e = state.specific_humidity * pressure_levels / (0.622 + 0.378 * state.specific_humidity)
    rel_humidity = e / es

    diagnostic_data = physics_data.diagnostics.copy(
        pressure_full=pressure_levels,
        pressure_half=pressure_half,
        height_full=height_levels,
        height_half=height_half,
        relative_humidity=rel_humidity,
        surface_pressure=surface_pressure,
        air_density=rho,
        layer_thickness=dz_full,
    )

    zero_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return zero_tendencies, physics_data.copy(diagnostics=diagnostic_data)

# Physics term methods
@jit
def apply_radiation(state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:

    """Apply radiation heating rates"""
    from jcm.physics.speedy.physical_constants import p0
    
    # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
    nlev, ncols = state.temperature.shape
    
    # Get date information for solar calculations
    date = physics_data.date
    day_of_year = date.day_of_year if hasattr(date, 'day_of_year') else 172.0  # Default to summer
    seconds_since_midnight = date.seconds_since_midnight if hasattr(date, 'seconds_since_midnight') else 43200.0  # Default to noon
    
    # Get lat/lon from geometry - for now use dummy values
    # In real implementation, these would come from geometry for each column
    latitudes = jnp.zeros(ncols)  # Equator for now
    longitudes = jnp.linspace(-180, 180, ncols)  # Span globe
    
    # Get cloud properties from tracers and previous physics
    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))

    # Define single column radiation function
    def apply_radiation_to_column(temp_col, humid_col, ps_col, geopot_col,
                                qc_col, qi_col, cf_col, lat, lon):
        """Apply radiation to single column"""
        
        tendencies, diagnostics = radiation_scheme(
            temperature=temp_col,
            specific_humidity=humid_col,
            surface_pressure=ps_col,  # Already normalized
            geopotential=geopot_col,
            cloud_water=qc_col,
            cloud_ice=qi_col,
            cloud_fraction=cf_col,
            day_of_year=day_of_year,
            seconds_since_midnight=seconds_since_midnight,
            latitude=lat,
            longitude=lon,
            parameters=parameters.radiation
        )
        
        # Return temperature tendency and key diagnostics
        return (
            tendencies.temperature_tendency,  # K/s
            tendencies.longwave_heating,      # K/s
            tendencies.shortwave_heating,     # K/s
            diagnostics.toa_lw_up,           # W/m²
            diagnostics.toa_sw_down,         # W/m²
            diagnostics.toa_sw_up,           # W/m²
            diagnostics.surface_lw_down,     # W/m²
            diagnostics.surface_sw_down      # W/m²
        )
    
    # Apply radiation using vmap
    rad_temp_tend, rad_out = jax.vmap(
        apply_radiation_to_column,
        in_axes=(1, 1, 0, 1, 1, 1, 1, 0, 0),  # ps, lat, lon are 1D
        out_axes=(1, 1, 1, 0, 0, 0, 0, 0)     # fluxes are scalars per column
    )(state.temperature, state.specific_humidity, state.surface_pressure,
        state.geopotential, cloud_water, cloud_ice, physics_data.clouds.cloud_fraction,
        latitudes, longitudes)
        
    # Create physics tendencies (already in 2D format [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),  # No wind tendencies from radiation
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=rad_temp_tend,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={}
    )
    
    # Update physics data with radiation diagnostics
    updated_physics_data = physics_data.copy(radiation=rad_out)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_convection(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply full Tiedtke-Nordeng convection scheme with updraft/downdraft physics"""
    from jcm.physics.icon.convection.tracer_transport import TracerIndices
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    
    # Extract tracers from physics state and combine with specific humidity
    # Build tracer array with specific humidity as first tracer
    tracer_list = [state.specific_humidity]  # Index 0: water vapor
    
    # Add cloud water (qc), cloud ice (qi), and other tracers if present
    if 'qc' in state.tracers:
        tracer_list.append(state.tracers['qc'])  # Index 1: cloud water
    else:
        tracer_list.append(jnp.zeros_like(state.temperature))  # Default to zero
        
    if 'qi' in state.tracers:
        tracer_list.append(state.tracers['qi'])  # Index 2: cloud ice  
    else:
        tracer_list.append(jnp.zeros_like(state.temperature))  # Default to zero
        
    # Add any additional tracers
    for name, tracer in state.tracers.items():
        if name not in ['qc', 'qi']:
            tracer_list.append(tracer)
            
    # Stack tracers into array [nlev, ncols, ntrac]
    tracers_2d = jnp.stack(tracer_list, axis=-1)
    ntrac = tracers_2d.shape[-1]
    
    # Setup tracer indices
    tracer_indices = TracerIndices(iqv=0, iqc=1, iqi=2, iqt=3)
    
    # Define single column convection function (always includes tracers)
    def apply_tiedtke_to_column(temp_col, humid_col, pressure_col, height_col, u_col, v_col, tracers_col):
        """Apply unified Tiedtke-Nordeng convection with tracer transport to a single column"""
        
        conv_tendencies, conv_state = tiedtke_nordeng_convection(
            temperature=temp_col,
            humidity=humid_col,
            pressure=pressure_col,
            height=height_col,
            u_wind=u_col,
            v_wind=v_col,
            tracers=tracers_col,
            dt=dt,
            config=parameters.convection,
            tracer_indices=tracer_indices
        )
        
        # Return tendencies and tracer tendencies
        return (
            conv_tendencies.dtedt,      # Temperature tendency
            conv_tendencies.dqdt,       # Humidity tendency  
            conv_tendencies.dudt,       # U-wind tendency
            conv_tendencies.dvdt,       # V-wind tendency
            conv_tendencies.qc_conv,    # Convective cloud water
            conv_tendencies.qi_conv,    # Convective cloud ice
            conv_tendencies.precip_conv, # Convective precipitation
            conv_tendencies.dtracer_dt if conv_tendencies.dtracer_dt is not None else jnp.zeros((nlev, ntrac))  # Tracer tendencies
        )
    
    # Apply convection with tracers using vmap
    results = jax.vmap(
        apply_tiedtke_to_column, 
        in_axes=(1, 1, 1, 1, 1, 1, 1), 
        out_axes=(1, 1, 1, 1, 1, 1, 0, 1)  # tracer tendencies: out_axis=1 for [nlev, ncols, ntrac]
    )(state.temperature, state.specific_humidity, pressure_levels, height_levels, 
        state.u_wind, state.v_wind, tracers_2d)
    
    # Unpack results with tracer tendencies
    conv_temp_tend, conv_humid_tend, conv_u_tend, conv_v_tend, qc_conv, qi_conv, precip_conv, tracer_tendencies = results
    
    # Split tracer tendencies back to individual tracers
    tracer_tend_dict = {}
    if ntrac > 1 and 'qc' in state.tracers:
        tracer_tend_dict['qc'] = tracer_tendencies[:, :, 1]
    if ntrac > 2 and 'qi' in state.tracers:
        tracer_tend_dict['qi'] = tracer_tendencies[:, :, 2]
    # Add other tracers
    idx = 3
    for name in state.tracers.keys():
        if name not in ['qc', 'qi'] and idx < ntrac:
            tracer_tend_dict[name] = tracer_tendencies[:, :, idx]
            idx += 1
    
    # Create physics tendencies (already in 2D format [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=conv_u_tend,
        v_wind=conv_v_tend,
        temperature=conv_temp_tend,
        specific_humidity=conv_humid_tend,
        tracers=tracer_tend_dict
    )
    
    # Note: tracer tendencies are already handled in tracer_tend_dict and don't need 3D reshaping
    # They will be reshaped in compute_tendencies when going back to 3D
    
    # Update physics data with convection diagnostics
    convection_data = physics_data.convection.copy(
        qc_conv=qc_conv,
        qi_conv=qi_conv,
        precip_conv=precip_conv,
    )
    updated_physics_data = physics_data.copy(convection=convection_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_clouds(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply shallow cloud scheme with microphysics"""
    nlev, ncols = state.temperature.shape
    
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    surface_pressure = physics_data.diagnostics.surface_pressure
    cloud_water = state.tracers.get('qc')
    cloud_ice = state.tracers.get('qi')
    
    # Get cloud configuration from parameters
    cloud_config = parameters.clouds
    
    # Define single column cloud function
    def apply_clouds_to_column(temp_col, humid_col, pressure_col, qc_col, qi_col, ps_col):
        """Apply shallow cloud scheme to a single column"""
        
        cloud_tendencies, cloud_state = shallow_cloud_scheme(
            temperature=temp_col,
            specific_humidity=humid_col,
            pressure=pressure_col,
            cloud_water=qc_col,
            cloud_ice=qi_col,
            surface_pressure=ps_col,
            dt=dt,
            config=cloud_config
        )
        
        # Return tendencies and diagnostic fields
        return (
            cloud_tendencies.dtedt,      # Temperature tendency
            cloud_tendencies.dqdt,       # Humidity tendency
            cloud_tendencies.dqcdt,      # Cloud water tendency
            cloud_tendencies.dqidt,      # Cloud ice tendency
            cloud_state.cloud_fraction,  # Cloud fraction
            cloud_state.rel_humidity,    # Relative humidity
            cloud_tendencies.rain_flux,  # Rain flux
            cloud_tendencies.snow_flux   # Snow flux
        )
    
    # Apply cloud scheme using vmap
    results = jax.vmap(
        apply_clouds_to_column,
        in_axes=(1, 1, 1, 1, 1, 0),  # surface_pressure is 1D
        out_axes=(1, 1, 1, 1, 1, 1, 0, 0)  # rain/snow fluxes are scalars per column
    )(state.temperature, state.specific_humidity, pressure_levels,
        cloud_water, cloud_ice, surface_pressure)
    
    # Unpack results
    cloud_temp_tend, cloud_humid_tend, cloud_qc_tend, cloud_qi_tend, \
    cloud_fraction, rel_humidity, rain_flux, snow_flux = results
    
    # Build tracer tendencies dictionary
    tracer_tend_dict = {}
    if 'qc' in state.tracers:
        tracer_tend_dict['qc'] = cloud_qc_tend
    if 'qi' in state.tracers:
        tracer_tend_dict['qi'] = cloud_qi_tend
    
    # Create physics tendencies (already in 2D format [nlev, ncols])
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),  # No wind tendencies from clouds
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=cloud_temp_tend,
        specific_humidity=cloud_humid_tend,
        tracers=tracer_tend_dict
    )
    
    # Update physics data with cloud diagnostics
    cloud_data = physics_data.clouds.copy(cloud_fraction=cloud_fraction,
                                           precip_rain=rain_flux,
                                           precip_snow=snow_flux)
    
    diagnostics = physics_data.diagnostics.copy(
        relative_humidity=rel_humidity,
    )

    updated_physics_data = physics_data.copy(clouds=cloud_data, 
                                             diagnostics=diagnostics)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_microphysics(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply cloud microphysics scheme"""
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    cloud_fraction = physics_data.clouds.cloud_fraction
    air_density = physics_data.diagnostics.air_density
    dz = physics_data.diagnostics.layer_thickness

    # Extract cloud water, ice, rain, and snow from state
    cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
    cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
    rain_water = state.tracers.get('qr', jnp.zeros_like(state.temperature))
    snow = state.tracers.get('qs', jnp.zeros_like(state.temperature))
    
    # Droplet number concentration (simple profile)
    droplet_number = jnp.ones_like(state.temperature) * 100e6  # 100 per cm³
    
    # Get microphysics configuration
    micro_config = parameters.microphysics
    
    # Define single column microphysics function
    def apply_microphysics_to_column(temp_col, humid_col, pressure_col, 
                                    qc_col, qi_col, qr_col, qs_col,
                                    cf_col, rho_col, dz_col, nc_col):
        """Apply microphysics to a single column"""
        
        micro_tendencies, micro_state = cloud_microphysics(
            temperature=temp_col,
            specific_humidity=humid_col,
            pressure=pressure_col,
            cloud_water=qc_col,
            cloud_ice=qi_col,
            rain_water=qr_col,
            snow=qs_col,
            cloud_fraction=cf_col,
            air_density=rho_col,
            layer_thickness=dz_col,
            droplet_number=nc_col,
            dt=dt,
            config=micro_config
        )
        
        return (
            micro_tendencies.dtedt,
            micro_tendencies.dqdt,
            micro_tendencies.dqcdt,
            micro_tendencies.dqidt,
            micro_tendencies.dqrdt,
            micro_tendencies.dqsdt,
            micro_state.precip_rain,
            micro_state.precip_snow
        )
    
    # Apply microphysics using vmap
    results = jax.vmap(
        apply_microphysics_to_column,
        in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        out_axes=(1, 1, 1, 1, 1, 1, 0, 0)
    )(state.temperature, state.specific_humidity, pressure_levels,
        cloud_water, cloud_ice, rain_water, snow,
        cloud_fraction, air_density, dz, droplet_number)
    
    # Unpack results
    micro_temp_tend, micro_humid_tend, micro_qc_tend, micro_qi_tend, \
    micro_qr_tend, micro_qs_tend, rain_flux, snow_flux = results
    
    # Build tracer tendencies
    tracer_tend_dict = {}
    if 'qc' in state.tracers:
        tracer_tend_dict['qc'] = micro_qc_tend
    if 'qi' in state.tracers:
        tracer_tend_dict['qi'] = micro_qi_tend
    if 'qr' in state.tracers:
        tracer_tend_dict['qr'] = micro_qr_tend
    if 'qs' in state.tracers:
        tracer_tend_dict['qs'] = micro_qs_tend
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=jnp.zeros_like(state.u_wind),
        v_wind=jnp.zeros_like(state.v_wind),
        temperature=micro_temp_tend,
        specific_humidity=micro_humid_tend,
        tracers=tracer_tend_dict
    )
    
    # Update physics data
    micro_data = physics_data.clouds.copy(
        precip_rain=rain_flux,
        precip_snow=snow_flux,
        droplet_number=droplet_number
    )
    
    updated_physics_data = physics_data.copy(clouds=micro_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_vertical_diffusion(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply vertical diffusion and boundary layer physics"""
    from jcm.physics.icon.vertical_diffusion import prepare_vertical_diffusion_state, vertical_diffusion_column
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt
    pressure_levels = physics_data.diagnostics.pressure_levels
    pressure_half = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    height_half = physics_data.diagnostics.height_half
    cloud_water = state.tracers.get('qc')
    cloud_ice = state.tracers.get('qi')
    
    # Initialize prognostic variables if not present
    tke = physics_data.vertical_diffusion.tke if hasattr(physics_data.vertical_diffusion, 'tke') else jnp.ones((nlev, ncols)) * 0.1
    thv_variance = physics_data.vertical_diffusion.thv_variance if hasattr(physics_data.vertical_diffusion, 'thv_variance') else jnp.zeros((nlev, ncols))
    
    # Surface properties (simplified - should come from boundaries)
    nsfc_type = 3  # water, ice, land
    surface_fraction = jnp.zeros((ncols, nsfc_type))
    surface_fraction = surface_fraction.at[:, 2].set(1.0)  # All land for now
    
    # Get surface properties from boundaries
    surface_temp = boundaries.surface_temperature if hasattr(boundaries, 'surface_temperature') else state.temperature[-1, :]
    roughness_length = boundaries.roughness_length if hasattr(boundaries, 'roughness_length') else jnp.full(ncols, 0.001)

    surface_temperature = jnp.repeat(surface_temp[:, jnp.newaxis], nsfc_type, axis=1)
    roughness = jnp.repeat(roughness_length[:, jnp.newaxis], nsfc_type, axis=1)
    
    # Ocean currents (zero for now)
    ocean_u = jnp.zeros(ncols)
    ocean_v = jnp.zeros(ncols)
    
    # Build tracer array (excluding specific humidity)
    tracer_list = []
    for name, tracer in state.tracers.items():
        if name not in ['qc', 'qi']:  # These are handled separately
            tracer_list.append(tracer)
    
    if tracer_list:
        tracers = jnp.stack(tracer_list, axis=-1)
    else:
        tracers = None
    
    # Define single column vertical diffusion function
    def apply_vdiff_to_column(col_idx):
        """Apply vertical diffusion to a single column"""        
        # Prepare state for this column
        vdiff_state = prepare_vertical_diffusion_state(
            u=state.u_wind[:, col_idx],
            v=state.v_wind[:, col_idx],
            temperature=state.temperature[:, col_idx],
            qv=state.specific_humidity[:, col_idx],
            qc=cloud_water[:, col_idx],
            qi=cloud_ice[:, col_idx],
            pressure_full=pressure_levels[:, col_idx],
            pressure_half=pressure_half[:, col_idx],
            geopotential=state.geopotential[:, col_idx],
            height_full=height_levels[:, col_idx],
            height_half=height_half[:, col_idx],
            surface_temperature=surface_temperature[col_idx, :],
            surface_fraction=surface_fraction[col_idx, :],
            roughness_length=roughness[col_idx, :],
            ocean_u=ocean_u[col_idx],
            ocean_v=ocean_v[col_idx],
            tke=tke[:, col_idx],
            thv_variance=thv_variance[:, col_idx],
            tracers=tracers[:, col_idx, :] if tracers is not None else None
        )
        
        # Compute vertical diffusion
        tendencies, diagnostics = vertical_diffusion_column(
            vdiff_state, parameters.vertical_diffusion, dt
        )
        
        return (
            tendencies.u_tend,
            tendencies.v_tend,
            tendencies.temp_tend,
            tendencies.qv_tend,
            tendencies.qc_tend,
            tendencies.qi_tend,
            tendencies.tke_tend,
            diagnostics.exchange_coeff_momentum,
            diagnostics.exchange_coeff_heat,
            diagnostics.pbl_height,
            diagnostics.surface_friction_velocity,
            diagnostics.surface_buoyancy_flux
        )
    
    # Apply vertical diffusion using vmap
    results = jax.vmap(apply_vdiff_to_column)(jnp.arange(ncols))
    
    # Unpack results
    (u_tend, v_tend, temp_tend, qv_tend, qc_tend, qi_tend, tke_tend,
        km, kh, pbl_height, u_star, b_flux) = results
    
    # Transpose results back to [nlev, ncols]
    u_tend = u_tend.T
    v_tend = v_tend.T
    temp_tend = temp_tend.T
    qv_tend = qv_tend.T
    qc_tend = qc_tend.T
    qi_tend = qi_tend.T
    tke_tend = tke_tend.T
    km = km.T
    kh = kh.T
    
    # Update TKE
    new_tke = tke + dt * tke_tend
    new_tke = jnp.maximum(new_tke, 0.01)  # Minimum TKE
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=u_tend,
        v_wind=v_tend,
        temperature=temp_tend,
        specific_humidity=qv_tend,
        tracers={
            'qc': qc_tend,
            'qi': qi_tend
        }
    )
    
    # Update physics data with vertical diffusion diagnostics
    vdiff_data = physics_data.vertical_diffusion.copy(
        tke=new_tke,
        thv_variance=thv_variance,  # Not updated yet
        exchange_coeff_momentum=km,
        exchange_coeff_heat=kh,
        pbl_height=pbl_height,
        surface_friction_velocity=u_star,
        surface_buoyancy_flux=b_flux,
    )
    
    updated_physics_data = physics_data.copy(vertical_diffusion=vdiff_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_surface(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply surface physics and calculate surface fluxes"""
    from jcm.physics.icon.surface.surface_types import AtmosphericForcing
    from jcm.physics.icon.surface import initialize_surface_state
    
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt
    pressure_levels = physics_data.diagnostics.pressure_full
    # Get surface properties from boundaries
    surface_temp = boundaries.surface_temperature if hasattr(boundaries, 'surface_temperature') else state.temperature[-1, :]

    # Initialize surface state (simplified)
    # In reality, this should come from the model's surface state
    nsfc_type = 3  # Fixed value: water, ice, land
    surface_fractions = jnp.zeros((ncols, nsfc_type))
    surface_fractions = surface_fractions.at[:, 2].set(1.0)  # All land for now (index 2)
    
    ocean_temp = surface_temp
    ice_temp = jnp.repeat(surface_temp[:, jnp.newaxis], 2, axis=1)  # 2 ice layers
    soil_temp = jnp.repeat(surface_temp[:, jnp.newaxis], 4, axis=1)  # 4 soil layers
    
    surface_state = initialize_surface_state(
        ncols, surface_fractions, ocean_temp, ice_temp, soil_temp, parameters.surface
    )
    
    # Prepare atmospheric forcing
    # Use lowest model level for surface conditions
    atm_temp = state.temperature[-1, :]  # Lowest model level
    atm_qv = state.specific_humidity[-1, :]
    atm_u = state.u_wind[-1, :]
    atm_v = state.v_wind[-1, :]
    atm_p = pressure_levels[-1, :]
    
    # Height of lowest model level above surface
    ref_height = physics_data.diagnostics.height_full[-1, :] - physics_data.diagnostics.height_full[-1, :].min()
    ref_height = jnp.maximum(ref_height, 10.0)  # At least 10m
    
    # Define single column surface physics function
    def apply_surface_to_column(col_idx):
        """Apply surface physics to a single column"""
        
        # Create atmospheric forcing for this column
        # Initialize exchange coefficients with dummy values for now
        nsfc_type = 3
        dummy_exchange = jnp.ones(nsfc_type) * 0.001  # Small exchange coefficient
        
        atm_forcing = AtmosphericForcing(
            temperature=atm_temp[col_idx],
            humidity=atm_qv[col_idx],
            u_wind=atm_u[col_idx],
            v_wind=atm_v[col_idx],
            pressure=atm_p[col_idx],
            sw_downward=physics_data.radiation.sw_down[col_idx],
            lw_downward=physics_data.radiation.lw_down[col_idx],
            rain_rate=0.0,  # No rain for now
            snow_rate=0.0,  # No snow for now
            exchange_coeff_heat=dummy_exchange,
            exchange_coeff_moisture=dummy_exchange,
            exchange_coeff_momentum=dummy_exchange
        )
        
        # Extract surface state for this column
        col_surface_state = surface_state.at[col_idx]

        # Apply surface physics
        fluxes, tendencies, diagnostics = surface_physics_step(
            atm_forcing, col_surface_state, dt, parameters.surface
        )
        
        return (
            fluxes.sensible_heat_mean,
            fluxes.latent_heat_mean,
            fluxes.momentum_u_mean,
            fluxes.momentum_v_mean,
            fluxes.evaporation_mean,
            tendencies.surface_temp_tendency.mean(),  # Area-weighted mean
            diagnostics.surface_exchange_coeff_heat,
            diagnostics.surface_exchange_coeff_momentum,
            diagnostics.surface_resistance
        )
    
    # Apply surface physics using vmap
    results = jax.vmap(apply_surface_to_column)(jnp.arange(ncols))
    
    # Unpack results
    (sensible_heat, latent_heat, tau_u, tau_v, evaporation,
        surf_temp_tend, ch, cm, resistance) = results
    
    # Convert fluxes to atmospheric tendencies
    # Only the lowest model level is directly affected by surface fluxes
    
    # Air density at surface
    rho_sfc = pressure_levels[-1, :] / (physical_constants.rd * state.temperature[-1, :])
    
    # Layer thickness at surface (approximate)
    dp_sfc = pressure_levels[-1, :] - pressure_levels[-2, :]
    dz_sfc = dp_sfc / (rho_sfc * physical_constants.grav)
    
    # Surface flux tendencies (applied to lowest level only)
    temp_tend_sfc = sensible_heat / (rho_sfc * physical_constants.cp * dz_sfc)
    qv_tend_sfc = (evaporation / 2.45e6) / (rho_sfc * dz_sfc)  # Latent heat = 2.45 MJ/kg
    u_tend_sfc = -tau_u / (rho_sfc * dz_sfc)
    v_tend_sfc = -tau_v / (rho_sfc * dz_sfc)
    
    # Initialize tendencies (only surface level affected)
    temp_tend = jnp.zeros_like(state.temperature)
    qv_tend = jnp.zeros_like(state.specific_humidity)
    u_tend = jnp.zeros_like(state.u_wind)
    v_tend = jnp.zeros_like(state.v_wind)
    
    # Apply surface tendencies to lowest level
    temp_tend = temp_tend.at[-1, :].set(temp_tend_sfc)
    qv_tend = qv_tend.at[-1, :].set(qv_tend_sfc)
    u_tend = u_tend.at[-1, :].set(u_tend_sfc)
    v_tend = v_tend.at[-1, :].set(v_tend_sfc)
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=u_tend,
        v_wind=v_tend,
        temperature=temp_tend,
        specific_humidity=qv_tend,
        tracers={}
    )
    
    # Update physics data with surface diagnostics
    surface_data = physics_data.surface.copy(
        sensible_heat_flux=sensible_heat,
        latent_heat_flux=latent_heat,
        momentum_flux_u=tau_u,
        momentum_flux_v=tau_v,
        evaporation_flux=evaporation,
        surface_temp_tendency=surf_temp_tend,
        exchange_coeff_heat=ch,
        exchange_coeff_momentum=cm,
        surface_resistance=resistance,
    )
    
    updated_physics_data = physics_data.copy(surface=surface_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_gravity_waves(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply gravity wave drag"""
    nlev, ncols = state.temperature.shape
    dt = parameters.convection.dt_conv
    pressure_levels = physics_data.diagnostics.pressure_full
    height_levels = physics_data.diagnostics.height_full
    
    # Need orography standard deviation - use a placeholder for now
    # In a real implementation, this would come from boundary data
    h_std = jnp.ones(ncols) * 200.0  # 200m standard deviation
    
    # Define single column GWD function
    def apply_gwd_to_column(u_col, v_col, temp_col, pressure_col, height_col, h_std_scalar):
        """Apply gravity wave drag to single column"""
        
        gwd_tendencies, gwd_state = gravity_wave_drag(
            u_wind=u_col,
            v_wind=v_col,
            temperature=temp_col,
            pressure=pressure_col,
            height=height_col,
            h_std=h_std_scalar,
            dt=dt,
            config=parameters.gravity_waves
        )
        
        return (
            gwd_tendencies.dudt,
            gwd_tendencies.dvdt,
            gwd_tendencies.dtedt,
            gwd_state.wave_stress[-1]  # Surface stress
        )
    
    # Apply GWD using vmap
    results = jax.vmap(
        apply_gwd_to_column,
        in_axes=(1, 1, 1, 1, 1, 0),
        out_axes=(1, 1, 1, 0)
    )(state.u_wind, state.v_wind, state.temperature,
        pressure_levels, height_levels, h_std)
    
    # Unpack results
    gwd_u_tend, gwd_v_tend, gwd_temp_tend, surface_stress = results
    
    # Create physics tendencies
    physics_tendencies = PhysicsTendency(
        u_wind=gwd_u_tend,
        v_wind=gwd_v_tend,
        temperature=gwd_temp_tend,
        specific_humidity=jnp.zeros_like(state.specific_humidity),
        tracers={}
    )
    
    # Update physics data
    gravity_wave_data = physics_data.gravity_waves.copy(
        surface_stress=surface_stress,
    )
    
    updated_physics_data = physics_data.copy(gravity_waves=gravity_wave_data)
    
    return physics_tendencies, updated_physics_data

@jit
def apply_chemistry(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
    """Apply chemistry tendencies"""
    # Placeholder - will implement chemistry_tendencies function
    return PhysicsTendency.zeros(state.temperature.shape), physics_data