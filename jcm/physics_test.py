import unittest
from dinosaur.primitive_equations import PrimitiveEquations
from dinosaur import primitive_equations_states
from dinosaur.sigma_coordinates import centered_vertical_advection
from jcm.held_suarez_model import HeldSuarezModel
from jcm.held_suarez import HeldSuarezForcing
from jcm.physics import get_physical_tendencies
from jcm.model import SpeedyModel
from jcm.convection import get_convection_tendencies
from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
from jcm.shortwave_radiation import get_shortwave_rad_fluxes, clouds
from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
from jcm.surface_flux import get_surface_fluxes
from jcm.vertical_diffusion import get_vertical_diffusion_tend
from jcm.humidity import spec_hum_to_rel_hum

class TestPhysicsUnit(unittest.TestCase):

    def test_speedy_model_HS94(self):
        speedy_model = HeldSuarezModel()
    
        state = speedy_model.get_initial_state()
        state.tracers = {
            'specific_humidity': primitive_equations_states.gaussian_scalar(
                speedy_model.coords, speedy_model.physics_specs)}
        # Choose a vertical multiplication method
        vertical_matmul_method = 'dense'  # or 'sparse'

        # Define a vertical advection function (optional, using default here)
        vertical_advection = centered_vertical_advection

        # Include vertical advection (optional)
        include_vertical_advection = True

        # Instantiate the PrimitiveEquations object
        dynamics = PrimitiveEquations(
            reference_temperature=speedy_model.ref_temps,
            orography=speedy_model.orography,
            coords=speedy_model.coords,
            physics_specs=speedy_model.physics_specs,
            vertical_matmul_method=vertical_matmul_method,
            vertical_advection=vertical_advection,
            include_vertical_advection=include_vertical_advection)

        hsf = HeldSuarezForcing(speedy_model.coords, speedy_model.physics_specs, speedy_model.ref_temps)

        physics_terms = [ hsf.held_suarez_forcings ] #abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]

        dynamics_tendency = get_physical_tendencies(state,dynamics,physics_terms)

        self.assertIsNotNone(dynamics_tendency)

    def test_speedy_model(self):
        speedy_model = SpeedyModel()
    
        state = speedy_model.get_initial_state()
        state.tracers = {
            'specific_humidity': primitive_equations_states.gaussian_scalar(
                speedy_model.coords, speedy_model.physics_specs)}
        # Choose a vertical multiplication method
        vertical_matmul_method = 'dense'  # or 'sparse'

        # Define a vertical advection function (optional, using default here)
        vertical_advection = centered_vertical_advection

        # Include vertical advection (optional)
        include_vertical_advection = True

        # Instantiate the PrimitiveEquations object
        dynamics = PrimitiveEquations(
            reference_temperature=speedy_model.ref_temps,
            orography=speedy_model.orography,
            coords=speedy_model.coords,
            physics_specs=speedy_model.physics_specs,
            vertical_matmul_method=vertical_matmul_method,
            vertical_advection=vertical_advection,
            include_vertical_advection=include_vertical_advection)

        physics_terms = [
            spec_hum_to_rel_hum, # this could get called in get_physics_tendencies before looping over the physics terms
            get_convection_tendencies,
            get_large_scale_condensation_tendencies,
            clouds,
            get_shortwave_rad_fluxes,
            get_downward_longwave_rad_fluxes,
            get_surface_fluxes, # In speedy this gets called again if air-sea coupling is on
            get_upward_longwave_rad_fluxes,
            get_vertical_diffusion_tend
        ] #abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]

        dynamics_tendency = get_physical_tendencies(state,dynamics,physics_terms)

        self.assertIsNotNone(dynamics_tendency)