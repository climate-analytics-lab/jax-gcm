"""
Unit tests for leapfrog_filters.py.

Tests verify that the horizontal diffusion filters produce numerically equivalent
results to SPEEDY Fortran implementation and are compatible with JAX gradients.
"""

# Force JAX to use CPU before any imports
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_PLATFORMS'] = 'cpu'

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Union
import sys
sys.path.insert(0, '/Users/ellend/Documents/Development/JAXathon/jax-gcm')

try:
    from dinosaur import spherical_harmonic
    from jcm.leapfrog_filters import multi_timescale_horizontal_diffusion_step_filter
    DINOSAUR_AVAILABLE = True
except ImportError:
    DINOSAUR_AVAILABLE = False
    print("Warning: Dinosaur not available. Tests will be skipped.")


class TestMultiTimescaleHorizontalDiffusion(unittest.TestCase):
    """Test suite for multi-timescale horizontal diffusion filter."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not DINOSAUR_AVAILABLE:
            self.skipTest("Dinosaur not available")
            
        # Create a test grid (T30 resolution to match SPEEDY)
        self.grid = spherical_harmonic.Grid.T30()
        self.dt = 2400.0  # 40 minutes in seconds (standard SPEEDY timestep)
        
        # SPEEDY physical constants (must match exactly)
        self.speedy_constants = {
            'trunc': 30,      # Spectral truncation
            'thd': 2.4,       # Temperature/vorticity diffusion timescale (hours)
            'thdd': 2.4,      # Divergence diffusion timescale (hours)  
            'thds': 12.0,     # Stratospheric diffusion timescale (hours)
            'npowhd': 4,      # Power of Laplacian (del^8 = del^(2*4))
            'npowhd_strat': 1 # Power for stratospheric diffusion (del^2 = del^(2*1))
        }
        
    def test_speedy_style_coefficient_computation(self):
        """Test that diffusion coefficients match SPEEDY Fortran exactly."""
        # SPEEDY parameters
        trunc = self.speedy_constants['trunc']
        tau_hours = self.speedy_constants['thd']
        tau_seconds = tau_hours * 3600.0
        order = self.speedy_constants['npowhd']
        
        # Get total wavenumbers from grid
        _, total_wavenumber = self.grid.modal_axes
        
        # SPEEDY's coefficient computation (from horizontal_diffusion.f90)
        rlap = 1.0 / float(trunc * (trunc + 1))
        hdiff_explicit = 1.0 / tau_seconds
        
        # Compute expected coefficients using SPEEDY's exact method
        expected_coeffs = jnp.ones_like(total_wavenumber, dtype=float)
        
        for j, twn in enumerate(total_wavenumber):
            if twn > 0:  # Skip zero wavenumber mode
                # SPEEDY's Laplacian normalization
                elap = twn * (twn + 1) * rlap
                elapn = elap ** order
                
                # SPEEDY's explicit diffusion coefficient
                dmp = hdiff_explicit * elapn
                
                # SPEEDY's implicit step (assuming dmp1 = 1/(1 + dt*dmp))
                dmp1 = 1.0 / (1.0 + self.dt * dmp)
                
                # Filter coefficient (multiplicative equivalent)
                coeff = 1.0 - self.dt * dmp * dmp1
                expected_coeffs = expected_coeffs.at[j].set(coeff)
        
        # Create filter with single timescale
        timescales = {'temperature_variation': tau_seconds}
        orders = {'temperature_variation': order}
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales, orders
        )
        
        # Extract the computed coefficients from the filter
        # We need to create a test state and examine the filtering behavior
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        
        # Create a mock state for testing
        class MockState:
            def __init__(self):
                self.temperature_variation = test_field
                
            def _asdict(self):
                return {'temperature_variation': self.temperature_variation}
                
            def _replace(self, **kwargs):
                new_state = MockState()
                for key, value in kwargs.items():
                    setattr(new_state, key, value)
                return new_state
        
        state = MockState()
        filtered_state = filter_fn(state)
        
        # The filtered result should be field * coefficients
        actual_coeffs = filtered_state.temperature_variation / test_field
        
        # Compare coefficients (allowing for small numerical differences)
        np.testing.assert_allclose(
            actual_coeffs, expected_coeffs, rtol=1e-10, atol=1e-12,
            err_msg="Diffusion coefficients do not match SPEEDY Fortran implementation"
        )
    
    def test_level_specific_diffusion_speedy_style(self):
        """Test level-specific diffusion matches SPEEDY's stratospheric approach."""
        levels = 8  # SPEEDY has 8 levels
        
        # SPEEDY-style level-specific timescales
        # Strong diffusion in stratosphere (levels 1-2), weaker in troposphere (levels 3-8)
        strat_timescale = self.speedy_constants['thds'] * 3600.0  # 12 hours in seconds
        trop_timescale = self.speedy_constants['thd'] * 3600.0    # 2.4 hours in seconds
        
        # SPEEDY uses del^2 in stratosphere, del^8 in troposphere
        strat_order = self.speedy_constants['npowhd_strat']  # 1 (del^2)
        trop_order = self.speedy_constants['npowhd']         # 4 (del^8)
        
        # Create level-specific arrays
        level_timescales = jnp.array([strat_timescale, strat_timescale] + 
                                   [trop_timescale] * (levels - 2))
        level_orders = jnp.array([strat_order, strat_order] + 
                                [trop_order] * (levels - 2))
        
        timescales = {'temperature_variation': level_timescales}
        orders = {'temperature_variation': level_orders}
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales, orders
        )
        
        # Test with 3D field
        test_field_3d = jnp.ones((levels,) + self.grid.modal_shape, dtype=complex)
        
        class MockState3D:
            def __init__(self):
                self.temperature_variation = test_field_3d
                
            def _asdict(self):
                return {'temperature_variation': self.temperature_variation}
                
            def _replace(self, **kwargs):
                new_state = MockState3D()
                for key, value in kwargs.items():
                    setattr(new_state, key, value)
                return new_state
        
        state = MockState3D()
        filtered_state = filter_fn(state)
        
        # Check that different levels have different filtering
        result = filtered_state.temperature_variation
        
        # Stratospheric levels should have weaker filtering (higher timescale = less diffusion)
        # Tropospheric levels should have stronger filtering (lower timescale = more diffusion)
        strat_coeff_0 = jnp.mean(jnp.abs(result[0] / test_field_3d[0]))
        strat_coeff_1 = jnp.mean(jnp.abs(result[1] / test_field_3d[1]))
        trop_coeff_2 = jnp.mean(jnp.abs(result[2] / test_field_3d[2]))
        
        # Stratospheric coefficients should be closer to 1 (less diffusion)
        self.assertGreater(strat_coeff_0, trop_coeff_2, 
                          "Stratospheric diffusion should be weaker than tropospheric")
        self.assertGreater(strat_coeff_1, trop_coeff_2,
                          "Stratospheric diffusion should be weaker than tropospheric")
        
        # Check that stratospheric levels have similar behavior
        np.testing.assert_allclose(strat_coeff_0, strat_coeff_1, rtol=1e-6,
                                 err_msg="Both stratospheric levels should have similar diffusion")
    
    def test_speedy_fortran_numerical_equivalence(self):
        """Test numerical equivalence with SPEEDY Fortran implementation."""
        # Reference values computed using SPEEDY Fortran horizontal_diffusion.f90
        # with T30 resolution and standard parameters
        
        # SPEEDY parameters used to generate reference
        trunc = 30
        tau_hours = 2.4
        order = 4
        dt_seconds = 2400.0
        
        # Get total wavenumbers (first few for testing)
        _, total_wavenumber = self.grid.modal_axes
        test_wavenumbers = total_wavenumber[:10]  # Test first 10 wavenumbers
        
        # Reference coefficients from SPEEDY Fortran (computed with above parameters)
        # Generated using horizontal_diffusion_reference.f90 with T30, dt=2400s, tau=2.4h, order=4
        fortran_reference_coeffs = jnp.array([
            1.00000000e+00,  # twn=0 (no diffusion)
            1.00000000e+00,  # twn=1 (no diffusion)
            1.00000000e+00,  # twn=2 (no diffusion)
            9.99999999e-01,  # twn=3
            9.99999994e-01,  # twn=4
            9.99999970e-01,  # twn=5
            9.99999884e-01,  # twn=6
            9.99999635e-01,  # twn=7
            9.99999002e-01,  # twn=8
            9.99997564e-01,  # twn=9
        ])
        
        # Create filter with same parameters
        tau_seconds = tau_hours * 3600.0
        timescales = {'temperature_variation': tau_seconds}
        orders = {'temperature_variation': order}
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, dt_seconds, timescales, orders
        )
        
        # Test with unit field
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        
        class MockState:
            def __init__(self):
                self.temperature_variation = test_field
                
            def _asdict(self):
                return {'temperature_variation': self.temperature_variation}
                
            def _replace(self, **kwargs):
                new_state = MockState()
                for key, value in kwargs.items():
                    setattr(new_state, key, value)
                return new_state
        
        state = MockState()
        filtered_state = filter_fn(state)
        
        # Extract coefficients for comparison
        jax_coeffs = jnp.abs(filtered_state.temperature_variation / test_field)
        
        # Compare with Fortran reference (first 10 wavenumbers)
        np.testing.assert_allclose(
            jax_coeffs[:10], fortran_reference_coeffs, rtol=1e-6, atol=1e-8,
            err_msg="JAX-GCM coefficients do not match SPEEDY Fortran reference"
        )
    
    def test_multiple_field_diffusion(self):
        """Test that multiple fields can be filtered with different parameters."""
        # Different timescales for different fields (SPEEDY-style)
        timescales = {
            'vorticity': 2.4 * 3600.0,           # 2.4 hours
            'divergence': 2.4 * 3600.0,          # 2.4 hours (same as vorticity)
            'temperature_variation': 2.4 * 3600.0, # 2.4 hours
            'tracers': 2.4 * 3600.0,             # 2.4 hours for tracers
        }
        
        orders = {
            'vorticity': 4,           # del^8
            'divergence': 4,          # del^8  
            'temperature_variation': 4, # del^8
            'tracers': 4,             # del^8
        }
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales, orders
        )
        
        # Create test state with multiple fields
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        
        class MockMultiFieldState:
            def __init__(self):
                self.vorticity = test_field
                self.divergence = test_field
                self.temperature_variation = test_field
                self.tracers = {'specific_humidity': test_field}
                self.other_field = test_field  # Should not be filtered
                
            def _asdict(self):
                return {
                    'vorticity': self.vorticity,
                    'divergence': self.divergence,
                    'temperature_variation': self.temperature_variation,
                    'tracers': self.tracers,
                    'other_field': self.other_field
                }
                
            def _replace(self, **kwargs):
                new_state = MockMultiFieldState()
                for key, value in kwargs.items():
                    setattr(new_state, key, value)
                return new_state
        
        state = MockMultiFieldState()
        filtered_state = filter_fn(state)
        
        # Check that specified fields are filtered
        self.assertFalse(jnp.allclose(filtered_state.vorticity, state.vorticity))
        self.assertFalse(jnp.allclose(filtered_state.divergence, state.divergence))
        self.assertFalse(jnp.allclose(filtered_state.temperature_variation, state.temperature_variation))
        self.assertFalse(jnp.allclose(filtered_state.tracers['specific_humidity'], 
                                    state.tracers['specific_humidity']))
        
        # Check that unspecified field is unchanged
        np.testing.assert_array_equal(filtered_state.other_field, state.other_field)
        
        # Check that all filtered fields have similar behavior (same parameters)
        vort_coeff = jnp.mean(jnp.abs(filtered_state.vorticity / state.vorticity))
        div_coeff = jnp.mean(jnp.abs(filtered_state.divergence / state.divergence))
        temp_coeff = jnp.mean(jnp.abs(filtered_state.temperature_variation / state.temperature_variation))
        
        np.testing.assert_allclose([vort_coeff, div_coeff, temp_coeff], 
                                 [vort_coeff, vort_coeff, vort_coeff], rtol=1e-10,
                                 err_msg="Fields with same parameters should have same filtering")
    
    def test_jax_gradient_compatibility(self):
        """Test that the filter is compatible with JAX automatic differentiation."""
        # Create filter
        timescales = {'temperature_variation': 2.4 * 3600.0}
        orders = {'temperature_variation': 4}
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales, orders
        )
        
        # Create test field with parameters
        def create_test_field(params):
            # Create a field that depends on the parameters
            return params['amplitude'] * jnp.ones(self.grid.modal_shape, dtype=complex)
        
        def loss_function(params):
            test_field = create_test_field(params)
            
            class MockState:
                def __init__(self):
                    self.temperature_variation = test_field
                    
                def _asdict(self):
                    return {'temperature_variation': self.temperature_variation}
                    
                def _replace(self, **kwargs):
                    new_state = MockState()
                    for key, value in kwargs.items():
                        setattr(new_state, key, value)
                    return new_state
            
            state = MockState()
            filtered_state = filter_fn(state)
            
            # Compute a simple loss
            return jnp.sum(jnp.abs(filtered_state.temperature_variation) ** 2)
        
        # Test gradient computation
        params = {'amplitude': 1.0}
        grad_fn = jax.grad(loss_function)
        gradients = grad_fn(params)
        
        # Check that gradients are computed successfully
        self.assertIsInstance(gradients['amplitude'], (float, jnp.ndarray))
        self.assertFalse(jnp.isnan(gradients['amplitude']))
        self.assertNotEqual(gradients['amplitude'], 0.0)
    
    def test_jax_jit_compatibility(self):
        """Test that the filter can be JIT compiled."""
        # Create filter
        timescales = {'temperature_variation': 2.4 * 3600.0}
        orders = {'temperature_variation': 4}
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales, orders
        )
        
        # JIT compile the filter
        jitted_filter = jax.jit(filter_fn)
        
        # Create test state
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        
        class MockState:
            def __init__(self):
                self.temperature_variation = test_field
                
            def _asdict(self):
                return {'temperature_variation': self.temperature_variation}
                
            def _replace(self, **kwargs):
                new_state = MockState()
                for key, value in kwargs.items():
                    setattr(new_state, key, value)
                return new_state
        
        state = MockState()
        
        # Test both implementations
        regular_result = filter_fn(state)
        jitted_result = jitted_filter(state)
        
        # Results should be identical
        np.testing.assert_allclose(
            regular_result.temperature_variation, jitted_result.temperature_variation,
            rtol=1e-12, atol=1e-15,
            err_msg="JIT compiled filter should produce identical results"
        )
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with zero timescale (should give identity filter)
        timescales = {'temperature_variation': 1e-10}  # Very small timescale
        orders = {'temperature_variation': 4}
        
        filter_fn = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales, orders
        )
        
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        
        class MockState:
            def __init__(self):
                self.temperature_variation = test_field
                
            def _asdict(self):
                return {'temperature_variation': self.temperature_variation}
                
            def _replace(self, **kwargs):
                new_state = MockState()
                for key, value in kwargs.items():
                    setattr(new_state, key, value)
                return new_state
        
        state = MockState()
        filtered_state = filter_fn(state)
        
        # With very small timescale, result should be heavily damped
        max_coeff = jnp.max(jnp.abs(filtered_state.temperature_variation / test_field))
        self.assertLess(max_coeff, 0.1, "Very strong diffusion should heavily damp the field")
        
        # Test with very large timescale (should give nearly identity filter)
        timescales_large = {'temperature_variation': 1e10}  # Very large timescale
        filter_fn_large = multi_timescale_horizontal_diffusion_step_filter(
            self.grid, self.dt, timescales_large, orders
        )
        
        filtered_state_large = filter_fn_large(state)
        
        # With very large timescale, result should be close to original
        min_coeff = jnp.min(jnp.abs(filtered_state_large.temperature_variation / test_field))
        self.assertGreater(min_coeff, 0.99, "Very weak diffusion should barely affect the field")


class TestHorizontalDiffusionLeapfrogStepFilter(unittest.TestCase):
    """Test suite for horizontal_diffusion_leapfrog_step_filter."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not DINOSAUR_AVAILABLE:
            self.skipTest("Dinosaur not available")
        
        self.grid = spherical_harmonic.Grid.T30()
        self.dt = 2400.0
        self.tau = 2.4 * 3600.0  # 2.4 hours
        self.order = 1  # del^2 diffusion
    
    def test_basic_functionality(self):
        """Test basic functionality of horizontal diffusion leapfrog filter."""
        from jcm.leapfrog_filters import horizontal_diffusion_leapfrog_step_filter
        
        filter_fn = horizontal_diffusion_leapfrog_step_filter(
            self.grid, self.dt, self.tau, self.order
        )
        
        # Create test leapfrog state (current, future)
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        current_state = test_field
        future_state = test_field * 1.1  # Slightly different
        
        leapfrog_state = (current_state, future_state)
        
        # Apply filter
        filtered_state = filter_fn(None, leapfrog_state)
        
        # Should return (current, filtered_future)
        self.assertEqual(len(filtered_state), 2)
        np.testing.assert_array_equal(filtered_state[0], current_state)
        self.assertFalse(jnp.allclose(filtered_state[1], future_state))
    
    def test_jax_compatibility(self):
        """Test JAX compatibility for leapfrog filter.""" 
        from jcm.leapfrog_filters import horizontal_diffusion_leapfrog_step_filter
        
        filter_fn = horizontal_diffusion_leapfrog_step_filter(
            self.grid, self.dt, self.tau, self.order
        )
        
        # JIT compile
        jitted_filter = jax.jit(filter_fn)
        
        # Test data
        test_field = jnp.ones(self.grid.modal_shape, dtype=complex)
        leapfrog_state = (test_field, test_field * 1.1)
        
        # Compare results
        regular_result = filter_fn(None, leapfrog_state)
        jitted_result = jitted_filter(None, leapfrog_state)
        
        np.testing.assert_allclose(regular_result[0], jitted_result[0], rtol=1e-12)
        np.testing.assert_allclose(regular_result[1], jitted_result[1], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()