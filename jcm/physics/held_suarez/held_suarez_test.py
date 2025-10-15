import unittest
import jax.numpy as jnp
class TestHeldSuarezUnit(unittest.TestCase):
    def test_held_suarez_forcing(self):
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarezPhysics
        from jcm.physics_interface import dynamics_state_to_physics_state

        model = Model()
        modal_state = model._prepare_initial_modal_state()
        physics_state = dynamics_state_to_physics_state(modal_state, model.primitive)

        physics_tendencies, _ = HeldSuarezPhysics().compute_tendencies(state=physics_state)

        self.assertIsNotNone(physics_tendencies)

        nodal_zxy = physics_state.u_wind.shape
        self.assertTupleEqual(physics_tendencies.u_wind.shape, nodal_zxy)
        self.assertTupleEqual(physics_tendencies.v_wind.shape, nodal_zxy)
        self.assertTupleEqual(physics_tendencies.temperature.shape, nodal_zxy)
        self.assertTupleEqual(physics_tendencies.specific_humidity.shape, nodal_zxy)
    
        self.assertFalse(jnp.isnan(physics_tendencies.u_wind).any())
        self.assertFalse(jnp.isnan(physics_tendencies.v_wind).any())
        self.assertFalse(jnp.isnan(physics_tendencies.temperature).any())
        self.assertFalse(jnp.isnan(physics_tendencies.specific_humidity).any())


