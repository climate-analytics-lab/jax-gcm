"""Integration tests for ECHAM Physics."""

import unittest
import jax.numpy as jnp
import pytest


class TestEchamPhysicsIntegration(unittest.TestCase):
    """Integration tests for ECHAM Physics package"""

    def setUp(self):
        """Set up test fixtures"""
        global Model, echam_physics
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

    @pytest.mark.slow
    def test_echam_physics_integration_3_timesteps(self):
        """Test that ECHAM physics runs for 3 timesteps and produces sensible output.

        This test should catch known bugs:
        - Radiation causing excessive cooling (-143 K/day)
        - Convection causing temperature blowup (to 1300 K in 4-6 hours)
        - Vertical diffusion producing T=0K
        """
        import numpy as np
        from jcm.utils import get_coords
        from jcm.terrain import TerrainData

        # Create model with ECHAM physics using sigma coordinates
        sigma_boundaries = np.linspace(0, 1, 41)  # 40 layers
        coords = get_coords(sigma_boundaries, spectral_truncation=31)
        terrain = TerrainData.aquaplanet(coords)

        model = Model(
            coords=coords,
            time_step=30,  # 30 minutes - reasonable for atmospheric physics
            terrain=terrain,
            physics=echam_physics(),
        )
        
        # Run the model for 6 hours (with radiation/convection bugs should crash)
        save_interval = 0.25  # Save every 6 hours
        total_time = 0.25     # Run for 6 hours
        predictions = model.run(
            save_interval=save_interval,
            total_time=total_time
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
        final_state = model._final_dycore_state  # In this simple test, predictions are the final state
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
        
        # Verify physics data exists and contains expected ECHAM physics outputs
        self.assertIsNotNone(physics_data, "Physics data should not be None")

        # Check that we have some ECHAM-specific physics outputs
        # These field names are based on the notebook outputs seen in run-echam.ipynb
        if hasattr(physics_data, 'shortwave_rad'):
            # Check for cloud-related variables that should be present in ECHAM physics
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
        self.assertGreater(temp_min, 150.0, f"Minimum temperature {temp_min} K seems too cold - radiation bug?")
        self.assertLess(temp_max, 350.0, f"Maximum temperature {temp_max} K seems too hot - convection blowup bug?")
        
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
        expected_time_steps = int(total_time / save_interval) 
        actual_time_steps = dynamics_predictions.temperature.shape[0]
        self.assertEqual(actual_time_steps, expected_time_steps,
                        f"Expected {expected_time_steps} time steps, got {actual_time_steps} - model may have crashed")
        
        print("✓ ECHAM physics integration test passed!")
        print(f"  - Ran for {actual_time_steps} time steps")
        print(f"  - Temperature range: {temp_min:.1f} - {temp_max:.1f} K")
        print(f"  - Surface pressure range: {sp_min:.3f} - {sp_max:.3f} (normalized)")
        print("  - No NaN values detected in key variables")


if __name__ == '__main__':
    unittest.main()
