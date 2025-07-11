import unittest
import jax.numpy as jnp
from dinosaur import primitive_equations_states
from dinosaur.scales import units
from jcm.constants import p0
from jcm.physics_interface import PhysicsState, physics_state_to_dynamics_state, dynamics_state_to_physics_state

class TestPhysicsInterfaceUnit(unittest.TestCase):
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

        coords = get_coords()
        _, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=coords,
            physics_specs=PHYSICS_SPECS,
            p0=p0*units.pascal,
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
        from jcm.physics_interface import verify_state, PhysicsState
        import jax.numpy as jnp

        kx, ix, il = 8, 96, 48
        qa = jnp.ones((kx, il, ix)) * -1

        state = PhysicsState.zeros((kx,ix,il), specific_humidity=qa)

        updated_state = verify_state(state)

        self.assertTrue(jnp.all(updated_state.specific_humidity >= 0))

        qa = jnp.ones((kx, il, ix)) * -1e-5

        state = PhysicsState.zeros((kx,ix,il), specific_humidity=qa)

        updated_state = verify_state(state)
