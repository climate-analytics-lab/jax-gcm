import unittest
from dinosaur import primitive_equations_states

import jax.tree_util as jtu
import pandas as pd

from jcm.params import Parameters
from jcm.physics_data import PhysicsData


class TestModelUnit(unittest.TestCase):

    def test_speedy_model(self):
        from jcm.model import SpeedyModel

        print("Building SpeedyModel...")
        layers = 8
        # optionally add a boundary conditions file
        model = SpeedyModel(
            time_step=720,
            save_interval=1,
            total_time=2,
            layers=layers,
            parameters=Parameters.default()
        )
        
        print("Done.")
    
        state = model.get_initial_state()

        # Specify humidity perturbation in kg/kg
        state.tracers = {'specific_humidity': 1e-2 * primitive_equations_states.gaussian_scalar(model.coords, model.physics_specs)}

        modal_zxy, nodal_zxy = model.coords.modal_shape, model.coords.nodal_shape
        nodal_tzxy = (model.outer_steps,) + nodal_zxy
    
        final_state, predictions = model.unroll(state)
        dynamics_predictions = predictions['dynamics']

        self.assertIsNotNone(final_state)
        self.assertIsNotNone(dynamics_predictions)

        self.assertIsNotNone(final_state.divergence)
        self.assertIsNotNone(final_state.vorticity)
        self.assertIsNotNone(final_state.temperature_variation)
        self.assertIsNotNone(final_state.log_surface_pressure)
        self.assertIsNotNone(final_state.tracers['specific_humidity'])

        self.assertIsNotNone(dynamics_predictions.u_wind)
        self.assertIsNotNone(dynamics_predictions.v_wind)
        self.assertIsNotNone(dynamics_predictions.temperature)
        self.assertIsNotNone(dynamics_predictions.specific_humidity)
        self.assertIsNotNone(dynamics_predictions.geopotential)
        self.assertIsNotNone(dynamics_predictions.surface_pressure)

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1,) + modal_zxy[1:])
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)

        self.assertTupleEqual(dynamics_predictions.u_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.v_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.temperature.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.specific_humidity.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.geopotential.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])


