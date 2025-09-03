import jax
import jax.numpy as jnp
from collections import abc
from typing import Callable, Tuple
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.boundaries import BoundaryData
from jcm.physics.speedy.params import Parameters
from jcm.geometry import Geometry
from jcm.date import DateData

def set_physics_flags(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:
    from jcm.physics.speedy.physical_constants import nstrad
    '''
    Sets flags that indicate whether a tendency function should be run.
    clouds, get_shortwave_rad_fluxes are the only functions that currently depend on this. 
    This could also apply to forcing and coupling.
    '''
    model_step = physics_data.date.model_step
    compute_shortwave = (jnp.mod(model_step, nstrad) == 0)
    shortwave_data = physics_data.shortwave_rad.copy(compute_shortwave=compute_shortwave)
    physics_data = physics_data.copy(shortwave_rad=shortwave_data)

    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return physics_tendencies, physics_data

class SpeedyPhysics(Physics):
    parameters: Parameters
    write_output: bool
    terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]
    
    def __init__(self, write_output: bool=True,
                 parameters: Parameters=Parameters.default(),
                 sea_coupling_flag=0, checkpoint_terms=True) -> None:
        """
        Initialize the SpeedyPhysics class with the specified parameters.
        
        Args:
            write_output (bool): Flag to indicate whether physics output should be written to predictions.
            parameters (Parameters): Parameters for the physics model.
            sea_coupling_flag (int): Flag to indicate if sea coupling is enabled.
            checkpoint_terms (bool): Flag to indicate if terms should be checkpointed.
        """
        self.write_output = write_output
        self.parameters = parameters

        from jcm.physics.speedy.humidity import spec_hum_to_rel_hum
        from jcm.physics.speedy.convection import get_convection_tendencies
        from jcm.physics.speedy.large_scale_condensation import get_large_scale_condensation_tendencies
        from jcm.physics.speedy.shortwave_radiation import get_shortwave_rad_fluxes, get_clouds
        from jcm.physics.speedy.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
        from jcm.physics.speedy.surface_flux import get_surface_fluxes
        from jcm.physics.speedy.vertical_diffusion import get_vertical_diffusion_tend
        from jcm.physics.speedy.forcing import set_forcing
        from jcm.physics.speedy.orographic_correction import get_orographic_correction_tendencies

        physics_terms = [
            set_physics_flags,
            set_forcing,
            spec_hum_to_rel_hum,
            get_convection_tendencies,
            get_large_scale_condensation_tendencies,
            get_clouds,
            get_shortwave_rad_fluxes,
            get_downward_longwave_rad_fluxes,
            get_surface_fluxes,
            get_upward_longwave_rad_fluxes,
            get_vertical_diffusion_tend,
            # get_orographic_correction_tendencies # orographic corrections applied last
        ]
        if sea_coupling_flag > 0:
            physics_terms.insert(-2, get_surface_fluxes)

        static_argnums = {
            set_forcing: (2,),
        }

        self.terms = physics_terms if not checkpoint_terms else [jax.checkpoint(term, static_argnums=static_argnums.get(term, ()) + (4,)) for term in physics_terms]
    
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
        data = PhysicsData.zeros(
            geometry.nodal_shape[1:],
            geometry.nodal_shape[0],
            date=date
        )
        # the 'physics_terms' return an instance of tendencies and data, data gets overwritten at each step
        # and implicitly passed to the next physics_term. tendencies are summed
        physics_tendency = PhysicsTendency.zeros(shape=state.u_wind.shape)
        
        for term in self.terms:
            tend, data = term(state, data, self.parameters, boundaries, geometry)
            physics_tendency += tend
        
        return physics_tendency, data
