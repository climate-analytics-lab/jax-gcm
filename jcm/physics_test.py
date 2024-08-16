import unittest
from jcm.physics import get_physical_tendencies
from jcm.held_suarez_forcing import held_suarez_forcings
from jcm.speedy_test_model import SpeedyTestModel
from dinosaur.primitive_equations import PrimitiveEquations
from dinosaur.sigma_coordinates import centered_vertical_advection

class TestPhysicsUnit(unittest.TestCase):

    def test_speedy_model_HS94(self):
        speedy_model = SpeedyTestModel()
    
        state = speedy_model.get_initial_state()

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

        physics_terms = [ held_suarez_forcings ] #abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]

        dynamics_tendency = get_physical_tendencies(state,dynamics,physics_terms)

        self.assertIsNotNone(dynamics_tendency)