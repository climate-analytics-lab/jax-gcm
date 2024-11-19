import unittest
from dinosaur.primitive_equations import PrimitiveEquations
from dinosaur import primitive_equations_states
from dinosaur.sigma_coordinates import centered_vertical_advection
from datetime import datetime

class TestPhysicsUnit(unittest.TestCase):
    def test_speedy_model_HS94(self):
        from jcm.held_suarez_model import HeldSuarezModel
        from jcm.held_suarez import HeldSuarezForcing
        from jcm.physics import get_physical_tendencies

        hs_model = HeldSuarezModel()
    
        state = hs_model.get_initial_state()
        state.tracers = {
            'specific_humidity': primitive_equations_states.gaussian_scalar(
                hs_model.coords, hs_model.physics_specs)}
        # Choose a vertical multiplication method
        vertical_matmul_method = 'dense'  # or 'sparse'

        # Define a vertical advection function (optional, using default here)
        vertical_advection = centered_vertical_advection

        # Include vertical advection (optional)
        include_vertical_advection = True

        # Instantiate the PrimitiveEquations object
        dynamics = PrimitiveEquations(
            reference_temperature=hs_model.ref_temps,
            orography=hs_model.orography,
            coords=hs_model.coords,
            physics_specs=hs_model.physics_specs,
            vertical_matmul_method=vertical_matmul_method,
            vertical_advection=vertical_advection,
            include_vertical_advection=include_vertical_advection)

        hsf = HeldSuarezForcing(hs_model.coords, hs_model.physics_specs, hs_model.ref_temps)

        physics_terms = [ hsf.held_suarez_forcings ] #abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]

        dynamics_tendency = get_physical_tendencies(state, dynamics, hs_model.physics_specs, physics_terms, datetime(2000, 1, 1))

        self.assertIsNotNone(dynamics_tendency)