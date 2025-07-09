"""
Main ICON Physics class for JAX-GCM

This module contains the main IconPhysics class that orchestrates the 
ICON atmospheric physics parameterizations. It follows the same pattern
as SpeedyPhysics but implements the ICON physics suite.

Date: 2025-01-09
"""

import jax
import jax.numpy as jnp
from collections import abc
from typing import Callable, Tuple, Optional
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
from jcm.physics.icon.constants import physical_constants

# Import physics modules (will be implemented progressively)
# from jcm.physics.icon.radiation import radiation_heating
# from jcm.physics.icon.convection import convection_tendencies
# from jcm.physics.icon.clouds import cloud_microphysics
# from jcm.physics.icon.vertical_diffusion import vertical_diffusion
# from jcm.physics.icon.surface import surface_fluxes
# from jcm.physics.icon.gravity_waves import gravity_wave_drag
# from jcm.physics.icon.chemistry import chemistry_tendencies

class IconPhysicsData(abc.Mapping):
    """Data container for ICON physics state and diagnostics"""
    
    def __init__(self, 
                 date: DateData,
                 radiation_data: Optional[dict] = None,
                 convection_data: Optional[dict] = None,
                 cloud_data: Optional[dict] = None,
                 surface_data: Optional[dict] = None,
                 **kwargs):
        self.date = date
        self.radiation_data = radiation_data or {}
        self.convection_data = convection_data or {}
        self.cloud_data = cloud_data or {}
        self.surface_data = surface_data or {}
        self._extra_data = kwargs
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra_data[key]
    
    def __iter__(self):
        for key in ['date', 'radiation_data', 'convection_data', 'cloud_data', 'surface_data']:
            yield key
        for key in self._extra_data:
            yield key
    
    def __len__(self):
        return 5 + len(self._extra_data)
    
    def copy(self, **kwargs):
        """Create a copy with updated values"""
        new_data = {
            'date': self.date,
            'radiation_data': self.radiation_data,
            'convection_data': self.convection_data,
            'cloud_data': self.cloud_data,
            'surface_data': self.surface_data,
            **self._extra_data
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
                 checkpoint_terms: bool = True):
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
        
        # Build list of physics terms
        self.terms = self._build_physics_terms()
    
    def _build_physics_terms(self) -> list:
        """Build list of physics terms to be applied"""
        terms = []
        
        # Add physics terms based on configuration
        # These will be implemented progressively
        
        # if self.enable_radiation:
        #     terms.append(self._apply_radiation)
        
        # if self.enable_convection:
        #     terms.append(self._apply_convection)
        
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
    
    def __call__(self, 
                 state: PhysicsState,
                 physics_data: IconPhysicsData,
                 boundaries: Optional[BoundaryData] = None,
                 geometry: Optional[Geometry] = None,
                 dt: float = 1800.0) -> Tuple[PhysicsTendency, IconPhysicsData]:
        """
        Apply ICON physics parameterizations.
        
        Args:
            state: Current physics state
            physics_data: Physics data container
            boundaries: Boundary conditions
            geometry: Model geometry
            dt: Time step (seconds)
            
        Returns:
            Tuple of (physics tendencies, updated physics data)
        """
        # Set physics flags and initialize tendencies
        tendencies, physics_data = set_physics_flags(
            state, physics_data, boundaries, geometry
        )
        
        # Apply physics terms sequentially
        for term in self.terms:
            if self.checkpoint_terms:
                term = jax.checkpoint(term)
            
            term_tendency, physics_data = term(
                state, physics_data, boundaries, geometry, dt
            )
            tendencies = tendencies + term_tendency
        
        return tendencies, physics_data
    
    # Physics term methods (to be implemented)
    def _apply_radiation(self, state, physics_data, boundaries, geometry, dt):
        """Apply radiation heating rates"""
        # Placeholder - will implement radiation_heating function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
    def _apply_convection(self, state, physics_data, boundaries, geometry, dt):
        """Apply convection tendencies"""
        # Placeholder - will implement convection_tendencies function
        return PhysicsTendency.zeros(state.temperature.shape), physics_data
    
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