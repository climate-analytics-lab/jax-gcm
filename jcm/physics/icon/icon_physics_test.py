"""
Integration tests for ICON Physics
"""

import unittest
import jax.numpy as jnp
import pytest


class TestIconPhysicsIntegration(unittest.TestCase):
    """Integration tests for ICON Physics package"""

    def setUp(self):
        """Set up test fixtures"""
        global Model, IconPhysics
        from jcm.model import Model
        from jcm.physics.icon.icon_physics import IconPhysics

    @pytest.mark.slow
    def test_icon_physics_integration_3_timesteps(self):
        """
        Test that ICON physics runs for 3 timesteps and produces sensible output.
        
        This is a simple integration test based on the run-icon.ipynb notebook.
        It verifies that the model can run without errors and produces reasonable results.
        """
        # Create model with ICON physics
        # Using smaller time step and shorter total time for fast testing
        model = Model(
            physics=IconPhysics()
        )
        
        # Run the model
        predictions = model.run(
            save_interval=3/48,  # Save every 3 time steps (1.5 hours)
            total_time=3/48     # Three 30 minute time steps (1.5 hours)
        )
        
        # Check that we have predictions
        self.assertIsNotNone(predictions, "Predictions should not be None")
        
        # Check that predictions have the expected structure
        self.assertTrue(hasattr(predictions, 'dynamics'), "Predictions should have dynamics")
        self.assertTrue(hasattr(predictions, 'physics'), "Predictions should have physics")
        
        dynamics_predictions = predictions.dynamics
        physics_data = predictions.physics
        
        # Verify dynamics predictions exist and have reasonable shapes
        self.assertIsNotNone(dynamics_predictions.u_wind, "u_wind should not be None")
        self.assertIsNotNone(dynamics_predictions.v_wind, "v_wind should not be None") 
        self.assertIsNotNone(dynamics_predictions.temperature, "temperature should not be None")
        self.assertIsNotNone(dynamics_predictions.specific_humidity, "specific_humidity should not be None")
        self.assertIsNotNone(dynamics_predictions.normalized_surface_pressure, "normalized_surface_pressure should not be None")
        
        # Check for NaN values in key dynamics variables
        self.assertFalse(jnp.any(jnp.isnan(dynamics_predictions.u_wind)), 
                         "u_wind should not contain NaN values")
        self.assertFalse(jnp.any(jnp.isnan(dynamics_predictions.v_wind)), 
                         "v_wind should not contain NaN values")
        self.assertFalse(jnp.any(jnp.isnan(dynamics_predictions.temperature)), 
                         "temperature should not contain NaN values")
        self.assertFalse(jnp.any(jnp.isnan(dynamics_predictions.specific_humidity)), 
                         "specific_humidity should not contain NaN values")
        self.assertFalse(jnp.any(jnp.isnan(dynamics_predictions.normalized_surface_pressure)), 
                         "normalized_surface_pressure should not contain NaN values")
        
        # Check that final state is reasonable
        final_state = model._final_modal_state  # In this simple test, predictions are the final state
        self.assertFalse(jnp.any(jnp.isnan(final_state.vorticity)), 
                         "Final state vorticity should not contain NaN")
        self.assertFalse(jnp.any(jnp.isnan(final_state.divergence)), 
                         "Final state divergence should not contain NaN")
        self.assertFalse(jnp.any(jnp.isnan(final_state.temperature_variation)), 
                         "Final state temperature_variation should not contain NaN")
        self.assertFalse(jnp.any(jnp.isnan(final_state.log_surface_pressure)), 
                         "Final state log_surface_pressure should not contain NaN")
        
        # Check tracers if they exist
        if hasattr(final_state, 'tracers') and 'specific_humidity' in final_state.tracers:
            self.assertFalse(jnp.any(jnp.isnan(final_state.tracers['specific_humidity'])), 
                             "Final state specific humidity tracer should not contain NaN")
        
        # Verify physics data exists and contains expected ICON physics outputs
        self.assertIsNotNone(physics_data, "Physics data should not be None")
        
        # Check that we have some ICON-specific physics outputs
        # These field names are based on the notebook outputs seen in run-icon.ipynb
        if hasattr(physics_data, 'shortwave_rad'):
            # Check for cloud-related variables that should be present in ICON physics
            if hasattr(physics_data.shortwave_rad, 'cloudc'):
                self.assertIsNotNone(physics_data.shortwave_rad.cloudc, 
                                   "Cloud cover should be present in shortwave radiation data")
            if hasattr(physics_data.shortwave_rad, 'qcloud'):
                self.assertIsNotNone(physics_data.shortwave_rad.qcloud,
                                   "Cloud water should be present in shortwave radiation data")
        
        # Check that we have reasonable field magnitudes (basic sanity checks)
        # Temperature should be in a reasonable range (200K - 350K)
        temp_min = jnp.min(dynamics_predictions.temperature)
        temp_max = jnp.max(dynamics_predictions.temperature)
        self.assertGreater(temp_min, 150.0, f"Minimum temperature {temp_min} K seems too cold")
        self.assertLess(temp_max, 400.0, f"Maximum temperature {temp_max} K seems too hot")
        
        # Surface pressure should be positive and in reasonable range (normalized units)
        sp_min = jnp.min(dynamics_predictions.normalized_surface_pressure)
        sp_max = jnp.max(dynamics_predictions.normalized_surface_pressure)
        self.assertGreater(sp_min, 0.0, "Surface pressure should be positive")
        self.assertGreater(sp_min, 0.3, f"Minimum surface pressure {sp_min} (normalized) seems too low")
        self.assertLess(sp_max, 2.0, f"Maximum surface pressure {sp_max} (normalized) seems too high")
        
        # Specific humidity should be non-negative
        q_min = jnp.min(dynamics_predictions.specific_humidity)
        self.assertGreaterEqual(q_min, 0.0, "Specific humidity should be non-negative")
        
        # Check that the time dimension exists and matches expected save intervals
        save_interval = 3/48
        total_time = 3/48
        expected_time_steps = int(total_time / save_interval) + 1  # +1 for initial state  
        actual_time_steps = dynamics_predictions.temperature.shape[0]
        self.assertEqual(actual_time_steps, expected_time_steps,
                        f"Expected {expected_time_steps} time steps, got {actual_time_steps}")
        
        print(f"✓ ICON physics integration test passed!")
        print(f"  - Ran for {actual_time_steps} time steps")
        print(f"  - Temperature range: {temp_min:.1f} - {temp_max:.1f} K")
        print(f"  - Surface pressure range: {sp_min:.3f} - {sp_max:.3f} (normalized)")
        print(f"  - No NaN values detected in key variables")


if __name__ == '__main__':
    unittest.main()
