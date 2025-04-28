import jax
from collections import abc
from typing import Callable, Tuple
from jcm.physics_interface import PhysicsState, PhysicsData, PhysicsTendency, Physics
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry
from jcm.date import DateData

class SpeedyPhysics(Physics):
    parameters: Parameters
    terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]
    
    def __init__(self, parameters: Parameters = Parameters.default(), sea_coupling_flag=0, checkpoint_terms=True) -> None:
        """
        Initialize the SpeedyPhysics class with the specified parameters.
        
        Args:
            parameters (Parameters): Parameters for the physics model.
            sea_coupling_flag (int): Flag to indicate if sea coupling is enabled.
            checkpoint_terms (bool): Flag to indicate if terms should be checkpointed.
        """
        self.parameters = parameters

        from jcm.humidity import spec_hum_to_rel_hum
        from jcm.convection import get_convection_tendencies
        from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
        from jcm.shortwave_radiation import get_shortwave_rad_fluxes, clouds
        from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
        from jcm.surface_flux import get_surface_fluxes
        from jcm.vertical_diffusion import get_vertical_diffusion_tend
        from jcm.land_model import couple_land_atm
        from jcm.forcing import set_forcing

        physics_terms = [
            set_forcing,
            spec_hum_to_rel_hum,
            get_convection_tendencies,
            get_large_scale_condensation_tendencies,
            clouds,
            get_shortwave_rad_fluxes,
            get_downward_longwave_rad_fluxes,
            get_surface_fluxes,
            get_upward_longwave_rad_fluxes,
            get_vertical_diffusion_tend,
            couple_land_atm # eventually couple sea model and ice model here
        ]
        if sea_coupling_flag > 0:
            physics_terms.insert(-3, get_surface_fluxes)

        static_argnums = {
            set_forcing: (2,),
            couple_land_atm: (3,),
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
