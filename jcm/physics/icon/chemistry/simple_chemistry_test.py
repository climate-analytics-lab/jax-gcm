"""
Tests for simple chemistry schemes

Date: 2025-01-15
"""

import jax.numpy as jnp
import jax
import pytest
from unittest import TestCase

from .simple_chemistry import (
    ChemistryParameters,
    simple_chemistry,
    fixed_ozone_distribution,
    simple_methane_chemistry,
    initialize_chemistry_tracers
)


class TestChemistryParameters(TestCase):
    """Test chemistry parameters"""
    
    def test_default_parameters(self):
        """Test default parameter creation"""
        config = ChemistryParameters.default()
        
        # Check that all parameters are positive
        self.assertGreater(config.ozone_scale_height, 0)
        self.assertGreater(config.ozone_max_vmr, 0)
        self.assertGreater(config.methane_surface_vmr, 0)
        self.assertGreater(config.co2_vmr, 0)
        
        # Check reasonable values
        self.assertGreater(config.ozone_max_vmr, 1000.0)  # > 1 ppmv
        self.assertLess(config.ozone_max_vmr, 20000.0)    # < 20 ppmv
        self.assertGreater(config.co2_vmr, 300.0)         # > 300 ppmv
        self.assertLess(config.co2_vmr, 1000.0)           # < 1000 ppmv


class TestOzoneDistribution(TestCase):
    """Test ozone distribution calculations"""
    
    def test_fixed_ozone_distribution(self):
        """Test fixed ozone distribution"""
        config = ChemistryParameters.default()
        
        # Create test data
        nlev, ncols = 10, 5
        pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
        surface_pressure = jnp.ones(ncols) * 100000.0
        temperature = jnp.ones((nlev, ncols)) * 250.0
        
        ozone_vmr = fixed_ozone_distribution(
            pressure, surface_pressure, temperature, config
        )
        
        # Check output shape
        self.assertEqual(ozone_vmr.shape, (nlev, ncols))
        
        # Check all values are positive
        self.assertTrue(jnp.all(ozone_vmr > 0))
        
        # Check maximum is reasonable
        self.assertLess(jnp.max(ozone_vmr), 20000.0)  # < 20 ppmv
        
        # Check ozone increases with height (up to some level)
        # Lower levels should have less ozone than upper levels
        self.assertLess(ozone_vmr[0, 0], ozone_vmr[-1, 0])


class TestMethaneChemistry(TestCase):
    """Test methane chemistry calculations"""
    
    def test_simple_methane_chemistry(self):
        """Test simple methane chemistry"""
        config = ChemistryParameters.default()
        
        # Create test data
        nlev, ncols = 10, 5
        pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
        temperature = jnp.ones((nlev, ncols)) * 280.0
        methane_vmr = jnp.ones((nlev, ncols)) * 1900.0  # 1.9 ppmv
        dt = 3600.0  # 1 hour
        
        methane_loss = simple_methane_chemistry(
            pressure, temperature, methane_vmr, dt, config
        )
        
        # Check output shape
        self.assertEqual(methane_loss.shape, (nlev, ncols))
        
        # Check all loss rates are positive
        self.assertTrue(jnp.all(methane_loss >= 0))
        
        # Check loss increases with height (lower pressure)
        self.assertGreater(methane_loss[-1, 0], methane_loss[0, 0])


class TestFullChemistry(TestCase):
    """Test full chemistry scheme"""
    
    def test_simple_chemistry_basic(self):
        """Test basic chemistry scheme functionality"""
        config = ChemistryParameters.default()
        
        # Create test data
        nlev, ncols = 10, 5
        pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
        surface_pressure = jnp.ones(ncols) * 100000.0
        temperature = jnp.ones((nlev, ncols)) * 250.0
        
        # Initialize with some values
        current_ozone = jnp.ones((nlev, ncols)) * 5000.0  # 5 ppmv
        current_methane = jnp.ones((nlev, ncols)) * 1800.0  # 1.8 ppmv
        dt = 3600.0
        
        tendencies, state = simple_chemistry(
            pressure, surface_pressure, temperature,
            current_ozone, current_methane, dt, config
        )
        
        # Check output shapes
        self.assertEqual(tendencies.ozone_tend.shape, (nlev, ncols))
        self.assertEqual(tendencies.methane_tend.shape, (nlev, ncols))
        self.assertEqual(state.ozone_vmr.shape, (nlev, ncols))
        self.assertEqual(state.methane_vmr.shape, (nlev, ncols))
        
        # Check that all values are finite
        self.assertTrue(jnp.all(jnp.isfinite(tendencies.ozone_tend)))
        self.assertTrue(jnp.all(jnp.isfinite(tendencies.methane_tend)))
        self.assertTrue(jnp.all(jnp.isfinite(state.ozone_vmr)))
        self.assertTrue(jnp.all(jnp.isfinite(state.methane_vmr)))
        
        # Check that methane tendency is negative (loss)
        self.assertTrue(jnp.all(tendencies.methane_tend <= 0))
        
        # Check that CO2 is constant
        self.assertTrue(jnp.all(tendencies.co2_tend == 0))
        
    def test_chemistry_initialization(self):
        """Test chemistry tracer initialization"""
        config = ChemistryParameters.default()
        
        # Create test data
        nlev, ncols = 10, 5
        pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
        surface_pressure = jnp.ones(ncols) * 100000.0
        temperature = jnp.ones((nlev, ncols)) * 250.0
        
        state = initialize_chemistry_tracers(
            pressure, surface_pressure, temperature, config
        )
        
        # Check output shapes
        self.assertEqual(state.ozone_vmr.shape, (nlev, ncols))
        self.assertEqual(state.methane_vmr.shape, (nlev, ncols))
        self.assertEqual(state.co2_vmr.shape, (nlev, ncols))
        
        # Check all values are positive and finite
        self.assertTrue(jnp.all(state.ozone_vmr > 0))
        self.assertTrue(jnp.all(state.methane_vmr > 0))
        self.assertTrue(jnp.all(state.co2_vmr > 0))
        self.assertTrue(jnp.all(jnp.isfinite(state.ozone_vmr)))
        self.assertTrue(jnp.all(jnp.isfinite(state.methane_vmr)))
        self.assertTrue(jnp.all(jnp.isfinite(state.co2_vmr)))
        
        # Check that methane decreases with height
        self.assertGreater(state.methane_vmr[0, 0], state.methane_vmr[-1, 0])


class TestJAXCompatibility(TestCase):
    """Test JAX compatibility"""
    
    def test_jax_jit_compilation(self):
        """Test JIT compilation of chemistry functions"""
        config = ChemistryParameters.default()
        
        # Test data
        nlev, ncols = 5, 3
        pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
        surface_pressure = jnp.ones(ncols) * 100000.0
        temperature = jnp.ones((nlev, ncols)) * 250.0
        current_ozone = jnp.ones((nlev, ncols)) * 5000.0
        current_methane = jnp.ones((nlev, ncols)) * 1800.0
        dt = 3600.0
        
        # Test JIT compilation
        jitted_chemistry = jax.jit(simple_chemistry)
        
        tendencies, state = jitted_chemistry(
            pressure, surface_pressure, temperature,
            current_ozone, current_methane, dt, config
        )
        
        # Should produce valid output
        self.assertEqual(tendencies.ozone_tend.shape, (nlev, ncols))
        self.assertTrue(jnp.all(jnp.isfinite(tendencies.ozone_tend)))
        
    def test_gradient_computation(self):
        """Test gradient computation"""
        config = ChemistryParameters.default()
        
        def loss_fn(ozone_vmr):
            nlev, ncols = ozone_vmr.shape
            pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
            surface_pressure = jnp.ones(ncols) * 100000.0
            temperature = jnp.ones((nlev, ncols)) * 250.0
            current_methane = jnp.ones((nlev, ncols)) * 1800.0
            dt = 3600.0
            
            tendencies, _ = simple_chemistry(
                pressure, surface_pressure, temperature,
                ozone_vmr, current_methane, dt, config
            )
            return jnp.sum(tendencies.ozone_tend ** 2)
        
        # Test gradient computation
        grad_fn = jax.grad(loss_fn)
        ozone_test = jnp.ones((5, 3)) * 5000.0
        grad = grad_fn(ozone_test)
        
        self.assertEqual(grad.shape, ozone_test.shape)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))


if __name__ == "__main__":
    unittest.main()