"""
Overall parameters for ICON physics

This module provides a unified Parameters class that contains all the 
configuration parameters for the various ICON physics parameterizations.

Date: 2025-01-10
"""

from dataclasses import dataclass
from typing import Optional

from .convection import ConvectionParameters
from .clouds import CloudParameters, MicrophysicsParameters
from .gravity_waves import GravityWaveParameters
from .radiation import RadiationParameters


@dataclass(frozen=True)
class Parameters:
    """
    Overall parameters for ICON physics
    
    This class contains all the configuration parameters for the various
    ICON physics parameterizations, following the same pattern as 
    SpeedyPhysics.
    """
    
    # Convection parameters
    convection: ConvectionParameters = ConvectionParameters()
    
    # Cloud parameters (shallow clouds)
    clouds: CloudParameters = CloudParameters()
    
    # Microphysics parameters
    microphysics: MicrophysicsParameters = MicrophysicsParameters()
    
    # Gravity wave parameters
    gravity_waves: GravityWaveParameters = GravityWaveParameters()
    
    # Radiation parameters
    radiation: RadiationParameters = RadiationParameters()
    
    # Vertical diffusion parameters (placeholder for future implementation)
    # vertical_diffusion: VerticalDiffusionParameters = VerticalDiffusionParameters()
    
    # Surface parameters (placeholder for future implementation)
    # surface: SurfaceParameters = SurfaceParameters()
    
    # Gravity wave drag parameters (placeholder for future implementation)
    # gravity_waves: GravityWaveParameters = GravityWaveParameters()
    
    @classmethod
    def default(cls) -> 'Parameters':
        """Create default parameters"""
        return cls()
    
    def with_convection(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated convection parameters"""
        convection_params = self.convection.__class__(**{
            **self.convection.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=convection_params,
            clouds=self.clouds,
            microphysics=self.microphysics,
            gravity_waves=self.gravity_waves,
            radiation=self.radiation
        )
    
    def with_clouds(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated cloud parameters"""
        cloud_params = self.clouds.__class__(**{
            **self.clouds.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=cloud_params,
            microphysics=self.microphysics,
            gravity_waves=self.gravity_waves,
            radiation=self.radiation
        )
    
    def with_microphysics(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated microphysics parameters"""
        micro_params = self.microphysics.__class__(**{
            **self.microphysics.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=micro_params,
            gravity_waves=self.gravity_waves,
            radiation=self.radiation
        )
    
    def with_gravity_waves(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated gravity wave parameters"""
        gwd_params = self.gravity_waves.__class__(**{
            **self.gravity_waves.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=self.microphysics,
            gravity_waves=gwd_params,
            radiation=self.radiation
        )
    
    def with_radiation(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated radiation parameters"""
        rad_params = self.radiation.__class__(**{
            **self.radiation.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=self.microphysics,
            gravity_waves=self.gravity_waves,
            radiation=rad_params
        )