import unittest
from dinosaur.primitive_equations import PrimitiveEquations
from dinosaur import primitive_equations_states
from dinosaur.sigma_coordinates import centered_vertical_advection
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry
import jax.numpy as jnp
from jcm.physics import PhysicsState, physics_state_to_dynamics_state, dynamics_state_to_physics_state
class TestPhysicsUnit(unittest.TestCase):
    def test_speedy_model_HS94(self):
        from jcm.held_suarez_model import HeldSuarezModel
        from jcm.held_suarez import HeldSuarezForcing
        from jcm.physics import get_physical_tendencies

        time_step = 10
        hs_model = HeldSuarezModel(time_step=time_step)
    
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
        parameters = Parameters.default()
        boundaries = BoundaryData.zeros((1,1))
        geometry = Geometry.from_grid_shape((1,1), 8)

        physics_terms = [ hsf.held_suarez_forcings ] # abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]

        dynamics_tendency = get_physical_tendencies(state, dynamics, time_step, physics_terms, boundaries, parameters, geometry)

        self.assertIsNotNone(dynamics_tendency)

    def test_initial_state_conversion(self):
        from dinosaur.scales import SI_SCALE
        from dinosaur import primitive_equations
        from dinosaur import xarray_utils
        from jcm.model import get_coords

        PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)
        kx, ix, il = 8, 96, 48
        temp = 288 * jnp.ones((kx, ix, il))
        u = jnp.ones((kx, ix, il)) * 0.5
        v = jnp.ones((kx, ix, il)) * -0.5
        q = jnp.ones((kx, ix, il)) * 0.5
        phi = jnp.ones((kx, ix, il)) * 5000
        sp = jnp.ones((kx, ix, il))

        coords = get_coords(layers=8, horizontal_resolution=31)
        _, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=coords,
            physics_specs=PHYSICS_SPECS,
            p0=1e5,
            p1=5e3,
        )
        ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
        truncated_orography = primitive_equations.truncated_modal_orography(aux_features[xarray_utils.OROGRAPHY], coords)

        primitive = primitive_equations.PrimitiveEquations(
            ref_temps,
            truncated_orography,
            coords,
            PHYSICS_SPECS)

        state = PhysicsState.zeros((kx, ix, il), u, v, temp, q, phi, sp)

        dynamics_state = physics_state_to_dynamics_state(state, primitive)
        physics_state_recovered = dynamics_state_to_physics_state(dynamics_state, primitive)

        self.assertTrue(jnp.allclose(state.temperature, physics_state_recovered.temperature))


    def test_verify_state(self):
        from jcm.physics import verify_state, PhysicsState
        import jax.numpy as jnp

        kx, ix, il = 8, 96, 48
        qa = jnp.ones((kx, il, ix)) * -1

        state = PhysicsState.zeros((kx,ix,il), specific_humidity=qa)

        updated_state = verify_state(state)

        self.assertTrue(jnp.all(updated_state.specific_humidity >= 0))

        qa = jnp.ones((kx, il, ix)) * -1e-5

        state = PhysicsState.zeros((kx,ix,il), specific_humidity=qa)

        updated_state = verify_state(state)
