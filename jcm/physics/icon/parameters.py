"""Overall parameters for ICON physics

This module provides a unified Parameters class that contains all the 
configuration parameters for the various ICON physics parameterizations.

Date: 2025-01-10
"""

import tree_math

from .convection import ConvectionParameters
from .clouds import CloudParameters, MicrophysicsParameters
from .gravity_waves import GravityWaveParameters
from .radiation import RadiationParameters
from .vertical_diffusion.vertical_diffusion_types import VDiffParameters
from .surface import SurfaceParameters
from .aerosol.aerosol_params import AerosolParameters

@tree_math.struct
class Parameters:
    """Overall parameters for ICON physics
    
    This class contains all the configuration parameters for the various
    ICON physics parameterizations, following the same pattern as 
    SpeedyPhysics.
    """
    
    # Convection parameters
    convection: ConvectionParameters
    clouds: CloudParameters
    microphysics: MicrophysicsParameters
    gravity_waves: GravityWaveParameters
    radiation: RadiationParameters
    vertical_diffusion: VDiffParameters
    surface: SurfaceParameters
    gravity_waves: GravityWaveParameters
    aerosol: AerosolParameters

    @classmethod
    def default(cls):
        return cls(
            convection = ConvectionParameters.default(),
            clouds = CloudParameters.default(),
            microphysics = MicrophysicsParameters.default(),
            gravity_waves = GravityWaveParameters.default(),
            radiation = RadiationParameters.default(),
            vertical_diffusion = VDiffParameters.default(),
            surface = SurfaceParameters.default(),
            aerosol = AerosolParameters.default()
        )

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
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol
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
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol
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
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol
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
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol
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
            radiation=rad_params,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol
        )
    
    def with_vertical_diffusion(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated vertical diffusion parameters"""
        vdiff_params = self.vertical_diffusion.__class__(**{
            **self.vertical_diffusion.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=self.microphysics,
            gravity_waves=self.gravity_waves,
            radiation=self.radiation,
            vertical_diffusion=vdiff_params,
            surface=self.surface,
            aerosol=self.aerosol
        )
    
    def with_surface(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated surface parameters"""
        surface_params = self.surface.__class__(**{
            **self.surface.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=self.microphysics,
            gravity_waves=self.gravity_waves,
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=surface_params,
            aerosol=self.aerosol
        )
    
    def with_aerosol(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated aerosol parameters"""
        aerosol_params = self.aerosol.__class__(**{
            **self.aerosol.__dict__,
            **kwargs
        })
        return self.__class__(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=self.microphysics,
            gravity_waves=self.gravity_waves,
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=aerosol_params
        )

    def with_timestep(self, dt_seconds: float) -> 'Parameters':
        """Create new Parameters with all physics timesteps set to the model timestep.

        This ensures consistency between the model integration timestep and
        the physics parameterization timesteps. Without sub-timestepping,
        all physics schemes should use the same timestep as the model.

        Args:
            dt_seconds: Model timestep in seconds

        Returns:
            New Parameters with updated timesteps in convection, radiation,
            and microphysics (dt_sedi capped at dt_seconds).

        """
        import jax.numpy as jnp

        # Update convection timestep
        convection_params = self.convection.__class__(**{
            **self.convection.__dict__,
            'dt_conv': jnp.array(dt_seconds)
        })

        # Update radiation timestep
        radiation_params = self.radiation.__class__(**{
            **self.radiation.__dict__,
            'dt_rad': jnp.array(dt_seconds)
        })

        # Update microphysics sedimentation timestep
        # dt_sedi should be <= dt_seconds (it's a sub-timestep)
        dt_sedi = min(float(self.microphysics.dt_sedi), dt_seconds)
        microphysics_params = self.microphysics.__class__(**{
            **self.microphysics.__dict__,
            'dt_sedi': jnp.array(dt_sedi)
        })

        return self.__class__(
            convection=convection_params,
            clouds=self.clouds,
            microphysics=microphysics_params,
            gravity_waves=self.gravity_waves,
            radiation=radiation_params,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol
        )