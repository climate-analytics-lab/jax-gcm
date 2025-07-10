"""
Main ICON Physics class for JAX-GCM

This module contains the main IconPhysics class that orchestrates the 
ICON atmospheric physics parameterizations. It follows the same pattern
as SpeedyPhysics but implements the ICON physics suite.

Date: 2025-01-09
"""

import jax
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
# from jcm.physics.icon.radiation import radiation_heating
from jcm.physics.icon.convection import tiedtke_nordeng_convection, ConvectionTendencies
from jcm.physics.icon.clouds import shallow_cloud_scheme, cloud_microphysics
from jcm.physics.icon.parameters import Parameters
# from jcm.physics.icon.vertical_diffusion import vertical_diffusion
# from jcm.physics.icon.surface import surface_fluxes
from jcm.physics.icon.gravity_waves import gravity_wave_drag
# from jcm.physics.icon.chemistry import chemistry_tendencies

@tree_math.struct
class IconPhysicsData:
    """Data container for ICON physics state and diagnostics"""
    
    date: DateData
    radiation_data: dict
    convection_data: dict
    cloud_data: dict
    surface_data: dict
    gravity_wave_data: dict
    microphysics_data: dict
    
    @classmethod
    def zeros(cls, 
              date: DateData,
              radiation_data: Optional[dict] = None,
              convection_data: Optional[dict] = None,
              cloud_data: Optional[dict] = None,
              surface_data: Optional[dict] = None,
              gravity_wave_data: Optional[dict] = None,
              microphysics_data: Optional[dict] = None):
        return cls(
            date=date,
            radiation_data=radiation_data or {},
            convection_data=convection_data or {},
            cloud_data=cloud_data or {},
            surface_data=surface_data or {},
            gravity_wave_data=gravity_wave_data or {},
            microphysics_data=microphysics_data or {}
        )
    
    def copy(self, **kwargs):
        """Create a copy with updated values"""
        new_data = {
            'date': self.date,
            'radiation_data': self.radiation_data,
            'convection_data': self.convection_data,
            'cloud_data': self.cloud_data,
            'surface_data': self.surface_data,
            'gravity_wave_data': self.gravity_wave_data,
            'microphysics_data': self.microphysics_data,
        }
        new_data.update(kwargs)
        return IconPhysicsData(**new_data)

def set_physics_flags(
    state: PhysicsState,
    physics_data: IconPhysicsData,
    boundaries: Optional[BoundaryData] = None,
    geometry: Optional[Geometry] = None
) -> Tuple[PhysicsTendency, IconPhysicsData]:
    """
    Set flags that indicate whether physics processes should be run.
    
    This function determines which physics processes need to be computed
    based on the current model time step and configuration.
    
    Args:
        state: Current physics state
        physics_data: Current physics data container
        boundaries: Boundary conditions (optional)
        geometry: Model geometry (optional)
        
    Returns:
        Tuple of (initialized physics tendencies, updated physics data)
    """
    # Initialize zero tendencies with tracer tendencies
    tracer_tends = {name: jnp.zeros_like(tracer) for name, tracer in state.tracers.items()}
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape, tracers=tracer_tends)
    
    # For now, return unchanged data
    # In full implementation, this would set flags for:
    # - Radiation time step frequency
    # - Convection triggering
    # - Cloud microphysics activation
    # - Surface flux computation
    
    return physics_tendencies, physics_data

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
        self.terms = self._build_physics_terms()
    
    def _build_physics_terms(self) -> list:
        """Build list of physics terms to be applied"""
        
        # Add physics terms based on configuration
        # These will be implemented progressively

        return [
            self._apply_convection, 
            self._apply_clouds,
            self._apply_microphysics,
            self._apply_gravity_waves
        ]
    
    def compute_tendencies(self, 
                 state: PhysicsState,
                 boundaries: Optional[BoundaryData] = None,
                 geometry: Optional[Geometry] = None,
                 date: Optional[DateData] = None) -> Tuple[PhysicsTendency, IconPhysicsData]:
        """
        Apply ICON physics parameterizations with automatic vectorization.
        
        This method handles the common pattern of reshaping 3D arrays to 2D,
        applying column-wise physics, and reshaping back. This pattern is
        reused across all physics modules.
        
        Args:
            state: Current physics state
            boundaries: Boundary conditions
            geometry: Model geometry
            date: Date data
            
        Returns:
            Tuple of (physics tendencies, updated physics data)
        """
        # Create initial physics data
        physics_data = IconPhysicsData.zeros(
            date=date or DateData.zeros(),
            convection_data={'initialized': True}
        )
        
        # Set physics flags and initialize tendencies with tracers
        tendencies, physics_data = set_physics_flags(
            state, physics_data, boundaries, geometry
        )
        
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
                vectorized_state, physics_data, boundaries, geometry
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
    
    # Physics term methods (to be implemented)
    def _apply_radiation(self, state, physics_data, boundaries, geometry, dt):
        """Apply radiation heating rates"""
        # Placeholder - will implement radiation_heating function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_convection(self, state, physics_data, boundaries, geometry):
        """Apply full Tiedtke-Nordeng convection scheme with updraft/downdraft physics"""
        from jcm.physics.speedy.physical_constants import p0
        from jcm.physics.icon.convection.tiedtke_nordeng import tiedtke_nordeng_convection
        from jcm.physics.icon.convection.tracer_transport import TracerIndices
        
        # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
        nlev, ncols = state.temperature.shape
        dt = 1800.0  # Time step for convection
        
        # Calculate pressure levels from surface pressure and sigma coordinates
        # Surface pressure is normalized, so multiply by p0 to get actual pressure
        surface_pressure = state.surface_pressure * p0  # Pa, shape [ncols]
        
        # Get sigma levels from geometry
        sigma_levels = geometry.fsg  # sigma coordinates at level centers
        
        # Calculate pressure at each level: p = sigma * ps
        # pressure has shape [nlev, ncols]
        pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
        
        # Calculate height from geopotential
        height_levels = state.geopotential / physical_constants.grav  # m
        
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
                config=self.parameters.convection,
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
        
        # Update physics data with convection diagnostics (always includes tracers)
        convection_data = {
            'convection_enabled': True,
            'tiedtke_nordeng_active': True,
            'convective_cloud_water': qc_conv,
            'convective_cloud_ice': qi_conv,
            'convective_precipitation': precip_conv,
            'tracer_tendencies': tracer_tend_dict,
            'last_applied': True
        }
        
        updated_physics_data = physics_data.copy(convection_data=convection_data)
        
        return physics_tendencies, updated_physics_data
    
    def _apply_clouds(self, state, physics_data, boundaries, geometry):
        """Apply shallow cloud scheme with microphysics"""
        from jcm.physics.speedy.physical_constants import p0
        
        # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
        nlev, ncols = state.temperature.shape
        dt = 1800.0  # Time step for cloud physics
        
        # Calculate pressure levels from surface pressure and sigma coordinates
        surface_pressure = state.surface_pressure * p0  # Pa, shape [ncols]
        sigma_levels = geometry.fsg  # sigma coordinates at level centers
        pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
        
        # Get cloud water and ice from tracers, or initialize to zero
        cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
        cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
        
        # Get cloud configuration from parameters
        cloud_config = self.parameters.clouds
        
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
        cloud_data = {
            'cloud_scheme_enabled': True,
            'cloud_fraction': cloud_fraction,
            'relative_humidity': rel_humidity,
            'rain_flux': rain_flux,
            'snow_flux': snow_flux,
            'total_precipitation': rain_flux + snow_flux,
            'last_applied': True
        }
        
        updated_physics_data = physics_data.copy(cloud_data=cloud_data)
        
        return physics_tendencies, updated_physics_data
    
    def _apply_microphysics(self, state, physics_data, boundaries, geometry):
        """Apply cloud microphysics scheme"""
        from jcm.physics.speedy.physical_constants import p0
        
        # Note: state is already in 2D format [nlev, ncols] from compute_tendencies
        nlev, ncols = state.temperature.shape
        dt = 1800.0  # Time step for microphysics
        
        # Calculate pressure levels
        surface_pressure = state.surface_pressure * p0  # Pa
        sigma_levels = geometry.fsg
        pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
        
        # Get hydrometeors from tracers
        cloud_water = state.tracers.get('qc', jnp.zeros_like(state.temperature))
        cloud_ice = state.tracers.get('qi', jnp.zeros_like(state.temperature))
        rain_water = state.tracers.get('qr', jnp.zeros_like(state.temperature))
        snow = state.tracers.get('qs', jnp.zeros_like(state.temperature))
        
        # Get cloud fraction from cloud scheme output
        cloud_fraction = physics_data.cloud_data.get('cloud_fraction', jnp.ones_like(state.temperature) * 0.1)
        
        # Air density using proper gas constant
        air_density = pressure_levels / (physical_constants.rd * state.temperature)
        
        # Layer thickness (approximate from pressure)
        dz = jnp.zeros_like(pressure_levels)
        # Use hydrostatic approximation: dz = dp/(rho*g)
        # Since pressure increases downward, dp is positive
        dz = dz.at[1:].set(
            (pressure_levels[1:] - pressure_levels[:-1]) / (air_density[1:] * physical_constants.grav)
        )
        dz = dz.at[0].set(dz[1])  # Set top layer same as layer below
        
        # Droplet number concentration (simple profile)
        droplet_number = jnp.ones_like(state.temperature) * 100e6  # 100 per cm³
        
        # Get microphysics configuration
        micro_config = self.parameters.microphysics
        
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
        microphysics_data = {
            'microphysics_enabled': True,
            'rain_flux': rain_flux,
            'snow_flux': snow_flux,
            'total_precipitation': rain_flux + snow_flux,
            'last_applied': True
        }
        
        updated_physics_data = physics_data.copy(microphysics_data=microphysics_data)
        
        return physics_tendencies, updated_physics_data
    
    def _apply_vertical_diffusion(self, state, physics_data, boundaries, geometry, dt):
        """Apply vertical diffusion"""
        # Placeholder - will implement vertical_diffusion function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_surface(self, state, physics_data, boundaries, geometry, dt):
        """Apply surface fluxes"""
        # Placeholder - will implement surface_fluxes function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_gravity_waves(self, state, physics_data, boundaries, geometry):
        """Apply gravity wave drag"""
        from jcm.physics.speedy.physical_constants import p0
        
        # Note: state is already in 2D format [nlev, ncols]
        nlev, ncols = state.temperature.shape
        dt = 1800.0  # Time step
        
        # Need orography standard deviation - use a placeholder for now
        # In a real implementation, this would come from boundary data
        h_std = jnp.ones(ncols) * 200.0  # 200m standard deviation
        
        # Calculate pressure and height
        surface_pressure = state.surface_pressure * p0
        sigma_levels = geometry.fsg
        pressure_levels = sigma_levels[:, jnp.newaxis] * surface_pressure[jnp.newaxis, :]
        
        # Convert geopotential to height
        height_levels = state.geopotential / physical_constants.grav
        
        # Get GWD configuration
        gwd_config = self.parameters.gravity_waves if hasattr(self.parameters, 'gravity_waves') else None
        
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
                config=gwd_config
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
        gravity_wave_data = {
            'gravity_wave_drag_enabled': True,
            'surface_stress': surface_stress,
            'mean_stress': jnp.mean(surface_stress),
            'last_applied': True
        }
        
        updated_physics_data = physics_data.copy(gravity_wave_data=gravity_wave_data)
        
        return physics_tendencies, updated_physics_data
    
    def _apply_chemistry(self, state, physics_data, boundaries, geometry, dt):
        """Apply chemistry tendencies"""
        # Placeholder - will implement chemistry_tendencies function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data