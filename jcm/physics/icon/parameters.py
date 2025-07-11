"""
Overall parameters for ICON physics

This module provides a unified Parameters class that contains all the 
configuration parameters for the various ICON physics parameterizations.

Date: 2025-01-10
"""

import tree_math
from dataclasses import field

from .convection import ConvectionParameters
from .clouds import CloudParameters, MicrophysicsParameters
from .gravity_waves import GravityWaveParameters
from .radiation import RadiationParameters
from .vertical_diffusion.vertical_diffusion_types import VDiffParameters
from .surface import SurfaceParameters
from .aerosol.aerosol_params import AerosolParameters

@tree_math.struct
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
    
    # Vertical diffusion parameters 
    vertical_diffusion: VDiffParameters = VDiffParameters()
    
    # Surface parameters
    surface: SurfaceParameters = SurfaceParameters()
    
    # Gravity wave drag parameters 
    gravity_waves: GravityWaveParameters = GravityWaveParameters()

    # Aerosol parameters
    aerosol: AerosolParameters = field(default_factory=AerosolParameters.default)

    @classmethod
    def default(self):
        return Parameters(
            convection = ConvectionParameters.default(),
            clouds = CloudParameters.default(),
            microphysics = MicrophysicsParameters.default(),
            gravity_waves = GravityWaveParameters.default(),
            radiation = RadiationParameters.default(),
            vertical_diffusion = VDiffParameters.default(),
            surface = SurfaceParameters.default(),
            aerosol = AerosolParameters.default()
        )
