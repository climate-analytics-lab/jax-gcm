"""Overall parameters for ECHAM physics

This module provides a unified Parameters class that contains all the
configuration parameters for the various ECHAM physics parameterizations.

Date: 2025-01-10
"""

import tree_math

from jcm.physics.convection.tiedtke_nordeng import ConvectionParameters
from jcm.physics.clouds.sundqvist import CloudParameters
from jcm.physics.clouds.echam_1m import MicrophysicsParameters
from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
from jcm.physics.gravity_waves.hines import HinesParameters
from jcm.physics.gravity_waves.simple import SimpleGwdParameters
from jcm.physics.gravity_waves.sso import SSOParameters
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import VDiffParameters
from jcm.physics.surface.echam import SurfaceParameters
from jcm.physics.aerosol.macv2_sp_params import AerosolParameters

@tree_math.struct
class Parameters:
    """Overall parameters for ECHAM physics

    This class contains all the configuration parameters for the various
    ECHAM physics parameterizations, following the same pattern as
    SpeedyPhysics.
    """

    convection: ConvectionParameters
    clouds: CloudParameters
    microphysics: MicrophysicsParameters
    microphysics_2m: CloudParams2M
    hines: HinesParameters
    sso: SSOParameters
    simple_gwd: SimpleGwdParameters
    radiation: RadiationParameters
    vertical_diffusion: VDiffParameters
    surface: SurfaceParameters
    aerosol: AerosolParameters

    @classmethod
    def default(cls):
        return cls(
            convection = ConvectionParameters.default(),
            clouds = CloudParameters.default(),
            microphysics = MicrophysicsParameters.default(),
            microphysics_2m = CloudParams2M.default(),
            hines = HinesParameters.default(),
            sso = SSOParameters.default(),
            simple_gwd = SimpleGwdParameters.default(),
            radiation = RadiationParameters.default(),
            vertical_diffusion = VDiffParameters.default(),
            surface = SurfaceParameters.default(),
            aerosol = AerosolParameters.default()
        )

    def _replace(self, **overrides) -> 'Parameters':
        """Return a copy of self with the named fields replaced."""
        fields = dict(
            convection=self.convection,
            clouds=self.clouds,
            microphysics=self.microphysics,
            microphysics_2m=self.microphysics_2m,
            hines=self.hines,
            sso=self.sso,
            simple_gwd=self.simple_gwd,
            radiation=self.radiation,
            vertical_diffusion=self.vertical_diffusion,
            surface=self.surface,
            aerosol=self.aerosol,
        )
        fields.update(overrides)
        return self.__class__(**fields)

    def with_convection(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated convection parameters"""
        return self._replace(
            convection=self.convection.__class__(**{**self.convection.__dict__, **kwargs})
        )

    def with_clouds(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated cloud parameters"""
        return self._replace(
            clouds=self.clouds.__class__(**{**self.clouds.__dict__, **kwargs})
        )

    def with_microphysics(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated 1-moment microphysics parameters"""
        return self._replace(
            microphysics=self.microphysics.__class__(**{**self.microphysics.__dict__, **kwargs})
        )

    def with_microphysics_2m(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated 2-moment microphysics parameters"""
        return self._replace(
            microphysics_2m=self.microphysics_2m.__class__(**{**self.microphysics_2m.__dict__, **kwargs})
        )

    def with_hines(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated Hines (non-orographic GW) parameters."""
        return self._replace(
            hines=self.hines.__class__(**{**self.hines.__dict__, **kwargs}),
        )

    def with_sso(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated SSO (sub-grid orography) parameters."""
        return self._replace(
            sso=self.sso.__class__(**{**self.sso.__dict__, **kwargs}),
        )

    def with_simple_gwd(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated simple-GWD parameters."""
        return self._replace(
            simple_gwd=self.simple_gwd.__class__(**{**self.simple_gwd.__dict__, **kwargs}),
        )

    def with_radiation(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated radiation parameters"""
        return self._replace(
            radiation=self.radiation.__class__(**{**self.radiation.__dict__, **kwargs})
        )

    def with_vertical_diffusion(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated vertical diffusion parameters"""
        return self._replace(
            vertical_diffusion=self.vertical_diffusion.__class__(**{**self.vertical_diffusion.__dict__, **kwargs})
        )

    def with_surface(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated surface parameters"""
        return self._replace(
            surface=self.surface.__class__(**{**self.surface.__dict__, **kwargs})
        )

    def with_aerosol(self, **kwargs) -> 'Parameters':
        """Create new Parameters with updated aerosol parameters"""
        return self._replace(
            aerosol=self.aerosol.__class__(**{**self.aerosol.__dict__, **kwargs})
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
            and both 1M/2M microphysics (dt_sedi capped at dt_seconds).

        """
        import jax.numpy as jnp

        convection_params = self.convection.__class__(**{
            **self.convection.__dict__,
            'dt_conv': jnp.array(dt_seconds),
        })

        # Radiation has no per-call sub-stepping (all backends do a single
        # forward pass per radiation_interval), so its config doesn't carry
        # a per-step value to update here.
        radiation_params = self.radiation

        # dt_sedi should be <= dt_seconds (it's a sub-timestep)
        dt_sedi_1m = min(float(self.microphysics.dt_sedi), dt_seconds)
        microphysics_params = self.microphysics.__class__(**{
            **self.microphysics.__dict__,
            'dt_sedi': jnp.array(dt_sedi_1m),
        })

        dt_sedi_2m = min(float(self.microphysics_2m.dt_sedi), dt_seconds)
        microphysics_2m_params = self.microphysics_2m.__class__(**{
            **self.microphysics_2m.__dict__,
            'dt_sedi': jnp.array(dt_sedi_2m),
        })

        return self._replace(
            convection=convection_params,
            microphysics=microphysics_params,
            microphysics_2m=microphysics_2m_params,
            radiation=radiation_params,
        )
