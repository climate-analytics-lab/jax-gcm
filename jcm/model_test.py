import unittest
from dinosaur import primitive_equations_states
from jcm.params import Parameters

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
        model = SpeedyModel(
            time_step=720,
            save_interval=1,
            total_time=2,
            layers=layers, 
            parameters=Parameters.init()
        )
    
        state = model.get_initial_state()

        # Specify humidity perturbation in kg/kg
        state.tracers = {'specific_humidity': 1e-2 * primitive_equations_states.gaussian_scalar(model.coords, model.physics_specs)}

        modal_x = 85
        modal_y = 44
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