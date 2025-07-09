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
from jcm.physics.icon.convection import tiedtke_nordeng_convection, ConvectionConfig, ConvectionTendencies
# from jcm.physics.icon.clouds import cloud_microphysics
# from jcm.physics.icon.vertical_diffusion import vertical_diffusion
# from jcm.physics.icon.surface import surface_fluxes
# from jcm.physics.icon.gravity_waves import gravity_wave_drag
# from jcm.physics.icon.chemistry import chemistry_tendencies

@tree_math.struct
class IconPhysicsData:
    """Data container for ICON physics state and diagnostics"""
    
    date: DateData
    radiation_data: dict
    convection_data: dict
    cloud_data: dict
    surface_data: dict
    
    @classmethod
    def zeros(cls, 
              date: DateData,
              radiation_data: Optional[dict] = None,
              convection_data: Optional[dict] = None,
              cloud_data: Optional[dict] = None,
              surface_data: Optional[dict] = None):
        return cls(
            date=date,
            radiation_data=radiation_data or {},
            convection_data=convection_data or {},
            cloud_data=cloud_data or {},
            surface_data=surface_data or {}
        )
    
    def copy(self, **kwargs):
        """Create a copy with updated values"""
        new_data = {
            'date': self.date,
            'radiation_data': self.radiation_data,
            'convection_data': self.convection_data,
            'cloud_data': self.cloud_data,
            'surface_data': self.surface_data,
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
    # Initialize zero tendencies
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    
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
                 enable_radiation: bool = True,
                 enable_convection: bool = True,
                 enable_clouds: bool = True,
                 enable_vertical_diffusion: bool = True,
                 enable_surface: bool = True,
                 enable_gravity_waves: bool = True,
                 enable_chemistry: bool = False,
                 checkpoint_terms: bool = True,
                 convection_config: Optional[ConvectionConfig] = None):
        """
        Initialize the ICON physics.
        
        Args:
            write_output: Whether to write physics output to predictions
            enable_radiation: Enable radiation calculations
            enable_convection: Enable convection parameterization
            enable_clouds: Enable cloud microphysics
            enable_vertical_diffusion: Enable vertical diffusion
            enable_surface: Enable surface flux calculations
            enable_gravity_waves: Enable gravity wave drag
            enable_chemistry: Enable chemistry schemes
            checkpoint_terms: Whether to checkpoint physics terms
        """
        self.write_output = write_output
        self.enable_radiation = enable_radiation
        self.enable_convection = enable_convection
        self.enable_clouds = enable_clouds
        self.enable_vertical_diffusion = enable_vertical_diffusion
        self.enable_surface = enable_surface
        self.enable_gravity_waves = enable_gravity_waves
        self.enable_chemistry = enable_chemistry
        self.checkpoint_terms = checkpoint_terms
        
        # Store convection configuration
        self.convection_config = convection_config or ConvectionConfig()
        
        # Build list of physics terms
        self.terms = self._build_physics_terms()
    
    def _build_physics_terms(self) -> list:
        """Build list of physics terms to be applied"""
        terms = []
        
        # Add physics terms based on configuration
        # These will be implemented progressively
        
        # if self.enable_radiation:
        #     terms.append(self._apply_radiation)
        
        if self.enable_convection:
            terms.append(self._apply_convection)
        
        # if self.enable_clouds:
        #     terms.append(self._apply_clouds)
        
        # if self.enable_vertical_diffusion:
        #     terms.append(self._apply_vertical_diffusion)
        
        # if self.enable_surface:
        #     terms.append(self._apply_surface)
        
        # if self.enable_gravity_waves:
        #     terms.append(self._apply_gravity_waves)
        
        # if self.enable_chemistry:
        #     terms.append(self._apply_chemistry)
        
        return terms
    
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
        
        # Set physics flags and initialize tendencies
        tendencies, physics_data = set_physics_flags(
            state, physics_data, boundaries, geometry
        )
        
        # Get array dimensions for vectorization
        nlev, nlat, nlon = state.temperature.shape
        ncols = nlat * nlon
        
        # Create vectorized physics state for column-wise processing
        # Reshape from [nlev, nlat, nlon] to [nlev, ncols] 
        vectorized_state = PhysicsState(
            u_wind=state.u_wind.reshape(nlev, ncols),
            v_wind=state.v_wind.reshape(nlev, ncols),
            temperature=state.temperature.reshape(nlev, ncols),
            specific_humidity=state.specific_humidity.reshape(nlev, ncols),
            geopotential=state.geopotential.reshape(nlev, ncols),
            surface_pressure=state.surface_pressure.reshape(ncols)
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
            reshaped_tendency = PhysicsTendency(
                u_wind=term_tendency.u_wind.reshape(nlev, nlat, nlon),
                v_wind=term_tendency.v_wind.reshape(nlev, nlat, nlon),
                temperature=term_tendency.temperature.reshape(nlev, nlat, nlon),
                specific_humidity=term_tendency.specific_humidity.reshape(nlev, nlat, nlon)
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
        
        # Define single column Tiedtke-Nordeng convection function
        def apply_tiedtke_to_column(temp_col, humid_col, pressure_col, height_col, u_col, v_col):
            """Apply full Tiedtke-Nordeng convection to a single column"""
            
            # Apply full Tiedtke-Nordeng convection scheme
            conv_tendencies, conv_state = tiedtke_nordeng_convection(
                temperature=temp_col,
                humidity=humid_col,
                pressure=pressure_col,
                height=height_col,
                u_wind=u_col,
                v_wind=v_col,
                dt=dt,
                config=self.convection_config
            )
            
            # Return tendencies as simple arrays for vmap compatibility
            return (
                conv_tendencies.dtedt,      # Temperature tendency
                conv_tendencies.dqdt,       # Humidity tendency  
                conv_tendencies.dudt,       # U-wind tendency
                conv_tendencies.dvdt,       # V-wind tendency
                conv_tendencies.qc_conv,    # Convective cloud water
                conv_tendencies.qi_conv,    # Convective cloud ice
                conv_tendencies.precip_conv # Convective precipitation
            )
        
        # Apply Tiedtke-Nordeng convection to all columns using vmap
        results = jax.vmap(
            apply_tiedtke_to_column, 
            in_axes=(1, 1, 1, 1, 1, 1), 
            out_axes=(1, 1, 1, 1, 1, 1, 0)  # precip is surface field, so axis 0
        )(state.temperature, state.specific_humidity, pressure_levels, height_levels, 
          state.u_wind, state.v_wind)
        
        # Unpack results
        conv_temp_tend, conv_humid_tend, conv_u_tend, conv_v_tend, qc_conv, qi_conv, precip_conv = results
        
        # Create physics tendencies (already in 2D format [nlev, ncols])
        physics_tendencies = PhysicsTendency(
            u_wind=conv_u_tend,
            v_wind=conv_v_tend,
            temperature=conv_temp_tend,
            specific_humidity=conv_humid_tend
        )
        
        # Update physics data with convection diagnostics
        updated_physics_data = physics_data.copy(
            convection_data={
                'convection_enabled': True,
                'tiedtke_nordeng_active': True,
                'convective_cloud_water': qc_conv,
                'convective_cloud_ice': qi_conv,
                'convective_precipitation': precip_conv,
                'last_applied': True
            }
        )
        
        return physics_tendencies, updated_physics_data
    
    def _apply_clouds(self, state, physics_data, boundaries, geometry, dt):
        """Apply cloud microphysics"""
        # Placeholder - will implement cloud_microphysics function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_vertical_diffusion(self, state, physics_data, boundaries, geometry, dt):
        """Apply vertical diffusion"""
        # Placeholder - will implement vertical_diffusion function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_surface(self, state, physics_data, boundaries, geometry, dt):
        """Apply surface fluxes"""
        # Placeholder - will implement surface_fluxes function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_gravity_waves(self, state, physics_data, boundaries, geometry, dt):
        """Apply gravity wave drag"""
        # Placeholder - will implement gravity_wave_drag function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_chemistry(self, state, physics_data, boundaries, geometry, dt):
        """Apply chemistry tendencies"""
        # Placeholder - will implement chemistry_tendencies function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data