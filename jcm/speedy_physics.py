import jax
from collections import abc
from typing import Callable, Tuple, Any
from jcm.physics_interface import PhysicsState, PhysicsData, PhysicsTendency, Physics
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry
from jcm.date import DateData

class SpeedyPhysics(Physics):
    terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]
    
    def __init__(self, sea_coupling_flag=0, checkpoint_terms=True):
        """
        Initialize the SpeedyPhysics class with the specified parameters.
        
        Args:
            sea_coupling_flag (int): Flag to indicate if sea coupling is enabled.
            checkpoint_terms (bool): Flag to indicate if terms should be checkpointed.
        """
        self.terms = SpeedyPhysics.get_speedy_physics_terms(sea_coupling_flag, checkpoint_terms)
    
    def compute_tendencies(
        self,
        state: PhysicsState,
        parameters: Parameters,
        boundaries: BoundaryData,
        geometry: Geometry,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
        """
        Compute the physical tendencies given the current state and data structs. Loops through the Speedy physics terms, accumulating the tendencies.

        Args:
            state: Current state variables
            parameters: Model parameters
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
            tend, data = term(state, data, parameters, boundaries, geometry)
            physics_tendency += tend
        
        return physics_tendency, data
    
    @classmethod
    def get_speedy_physics_terms(self, sea_coupling_flag=0, checkpoint_terms=True):
        """
        Returns a list of functions that compute physical tendencies for the model.
        """
        
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

        if not checkpoint_terms:
            return physics_terms

        static_argnums = {
            set_forcing: (2,),
            couple_land_atm: (3,),
        }
        
        # Static argnum 4 is the Geometry object
        return [jax.checkpoint(term, static_argnums=static_argnums.get(term, ()) + (4,)) for term in physics_terms]
