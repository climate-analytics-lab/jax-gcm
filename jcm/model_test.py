import unittest
from dinosaur import primitive_equations_states
from jcm.params import Parameters
import jax.tree_util as jtu

class TestModelUnit(unittest.TestCase):

    def test_held_suarez_model(self):
        from jcm.held_suarez_model import HeldSuarezModel
        layers = 8
        model = HeldSuarezModel(
            time_step=180,
            save_interval=1,
            total_time=2,
            layers=layers
        )

        state = model.get_initial_state()
        state.tracers = {'specific_humidity': 1e-4 * primitive_equations_states.gaussian_scalar(model.coords, model.physics_specs)}

        modal_x = 85
        modal_y = 44
        modal_zxy = (layers, modal_x, modal_y)
        output_tzxy = (model.outer_steps, layers, modal_x, modal_y)

        final_state, predictions = model.unroll(state)

        self.assertIsNotNone(final_state)
        self.assertIsNotNone(predictions)

        self.assertIsNotNone(final_state.divergence)
        self.assertIsNotNone(final_state.vorticity)
        self.assertIsNotNone(final_state.temperature_variation)
        self.assertIsNotNone(final_state.log_surface_pressure)
        self.assertIsNotNone(final_state.tracers['specific_humidity'])

        self.assertIsNotNone(predictions.divergence)
        self.assertIsNotNone(predictions.vorticity)
        self.assertIsNotNone(predictions.temperature_variation)
        self.assertIsNotNone(predictions.log_surface_pressure)
        self.assertIsNotNone(predictions.tracers['specific_humidity'])

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1, modal_x, modal_y))
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)

        self.assertTupleEqual(predictions.divergence.shape, output_tzxy)
        self.assertTupleEqual(predictions.vorticity.shape, output_tzxy)
        self.assertTupleEqual(predictions.temperature_variation.shape, output_tzxy)
        self.assertTupleEqual(predictions.log_surface_pressure.shape, (model.outer_steps, 1, modal_x, modal_y))
        self.assertTupleEqual(predictions.tracers['specific_humidity'].shape, output_tzxy)
        
    def test_speedy_model(self):
        from jcm.model import SpeedyModel

        layers = 7
        # optionally add a boundary conditions file
        model = SpeedyModel(
            time_step=720,
            save_interval=1,
            total_time=2,
            layers=layers, 
            parameters=Parameters.default()
        )
    
        state = model.get_initial_state()

        # Specify humidity perturbation in kg/kg
        state.tracers = {'specific_humidity': 1e-2 * primitive_equations_states.gaussian_scalar(model.coords, model.physics_specs)}

        modal_x = model.coords.modal_shape[1]
        modal_y = model.coords.modal_shape[2]
        modal_zxy = (layers, modal_x, modal_y)
        output_tzxy = (model.outer_steps, layers, modal_x, modal_y)
    
        final_state, predictions = model.unroll(state)
        dynamics_predictions = predictions['dynamics']

        self.assertIsNotNone(final_state)
        self.assertIsNotNone(dynamics_predictions)

        self.assertIsNotNone(final_state.divergence)
        self.assertIsNotNone(final_state.vorticity)
        self.assertIsNotNone(final_state.temperature_variation)
        self.assertIsNotNone(final_state.log_surface_pressure)
        self.assertIsNotNone(final_state.tracers['specific_humidity'])

        self.assertIsNotNone(dynamics_predictions.divergence)
        self.assertIsNotNone(dynamics_predictions.vorticity)
        self.assertIsNotNone(dynamics_predictions.temperature_variation)
        self.assertIsNotNone(dynamics_predictions.log_surface_pressure)
        self.assertIsNotNone(dynamics_predictions.tracers['specific_humidity'])

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1, modal_x, modal_y))
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)

        self.assertTupleEqual(dynamics_predictions.divergence.shape, output_tzxy)
        self.assertTupleEqual(dynamics_predictions.vorticity.shape, output_tzxy)
        self.assertTupleEqual(dynamics_predictions.temperature_variation.shape, output_tzxy)
        self.assertTupleEqual(dynamics_predictions.log_surface_pressure.shape, (model.outer_steps, 1, modal_x, modal_y))
        self.assertTupleEqual(dynamics_predictions.tracers['specific_humidity'].shape, output_tzxy)
        
    def test_speedy_model_gradients_isnan(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import SpeedyModel

        def make_ones_dinosaur_State_object(state, choose_sim_time = jnp.float32(1.0)):
            return jtu.tree_map(lambda x: jnp.ones_like(x), state)

        def make_ones_prediction_object(pred):
            return jtu.tree_map(lambda x: jnp.ones_like(x), pred)
        
        #create model that goes through one timestep
        model = SpeedyModel(time_step=30, save_interval=(1/48.), total_time=(1/48.), layers=8, parameters=Parameters.default()) # takes 40 seconds on laptop gpu
        state = model.get_initial_state()

        # Calculate gradients
        primals, f_vjp = jax.vjp(model.unroll, state) 
        
        input = (make_ones_dinosaur_State_object(primals[0]), make_ones_prediction_object(primals[1]))

        df_dstate = f_vjp(input) 
        
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].divergence)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].tracers['specific_humidity'])))
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan

    def test_speedy_model_gradients_multiple_timesteps_isnan(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import SpeedyModel

        def make_ones_dinosaur_State_object(state, choose_sim_time = jnp.float32(1.0)):
            return jtu.tree_map(lambda x: jnp.ones_like(x), state)
        
        def make_ones_prediction_object(pred): 
            return jtu.tree_map(lambda x: jnp.ones_like(x), pred)
        
        model = SpeedyModel(time_step=30, save_interval=(1/48.), total_time=(1/24.), layers=8, parameters=Parameters.default())
        state = model.get_initial_state()

        # Calculate gradients
        primals, f_vjp = jax.vjp(model.unroll, state) 
        input = (make_ones_dinosaur_State_object(primals[0]), make_ones_prediction_object(primals[1]))
        df_dstate = f_vjp(input) 

        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].divergence)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].tracers['specific_humidity'])))
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan
