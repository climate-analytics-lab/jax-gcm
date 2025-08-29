import unittest
import jax.numpy as jnp
class TestHeldSuarezUnit(unittest.TestCase):
    def test_held_suarez_forcing(self):
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarezPhysics
        from jcm.physics_interface import get_physical_tendencies
        from jcm.diffusion import DiffusionFilter

        time_step = 10
        model = Model(time_step=time_step, physics=HeldSuarezPhysics())
    
        dynamics_tendency = get_physical_tendencies(
            state = model._prepare_initial_state(),
            dynamics = model.primitive,
            time_step = time_step * 60,
            physics = HeldSuarezPhysics(model.coords),
            boundaries = None,
            geometry = None,
            diffusion = DiffusionFilter.default(),
            date = None
        )

        self.assertIsNotNone(dynamics_tendency)

    def test_held_suarez_model(self):
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarezPhysics
        
        model = Model(physics=HeldSuarezPhysics())

        _ = model.run(total_time=36)

        final_state = model._final_state_internal

        self.assertFalse(jnp.any(jnp.isnan(final_state.vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.divergence)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.tracers['specific_humidity'])))