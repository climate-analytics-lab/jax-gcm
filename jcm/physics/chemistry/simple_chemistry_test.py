"""Tests for simple chemistry schemes

Date: 2025-01-15
"""

import jax.numpy as jnp
import jax
from unittest import TestCase

from jcm.testing import check_gradients

from .simple_chemistry import (
    ChemistryData,
    ChemistryParameters,
    ppmv_to_mole_fraction,
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
        
        # Check that all parameters are positive (CO2 is a prescribed forcing,
        # not a chemistry parameter — see ForcingData.co2_vmr).
        self.assertGreater(config.ozone_scale_height, 0)
        self.assertGreater(config.ozone_max_vmr, 0)
        self.assertGreater(config.methane_surface_vmr, 0)

        # Check reasonable values
        self.assertEqual(float(config.ozone_max_vmr), 8.0)
        self.assertAlmostEqual(float(config.methane_surface_vmr), 1.9)

    def test_chemistry_data_ppmv_conversion_is_field_specific(self):
        """Public ppmv fields convert exactly once at radiation boundaries."""
        chemistry = ChemistryData.zeros((2,), 3).copy(
            ozone_vmr=jnp.full((3, 2), 8.0),
            methane_vmr=jnp.full((3, 2), 1.9),
        )

        self.assertTrue(jnp.allclose(
            chemistry.ozone_mole_fraction(), 8.0e-6,
        ))
        self.assertTrue(jnp.allclose(
            chemistry.methane_mole_fraction(), 1.9e-6,
        ))
        self.assertTrue(jnp.allclose(
            ppmv_to_mole_fraction(jnp.asarray(0.327)), 0.327e-6,
        ))


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
        self.assertLess(jnp.max(ozone_vmr), 20.0)
        self.assertGreaterEqual(jnp.min(ozone_vmr), 0.01)  # 10 ppbv floor
        
        # Check ozone increases with height (up to some level)
        # Lower levels should have less ozone than upper levels
        self.assertLess(ozone_vmr[0, 0], ozone_vmr[-1, 0])

    def test_every_ozone_parameter_is_live(self):
        """Each ozone field of ``ChemistryParameters`` moves the profile.

        A declared tunable the profile never reads has an exactly-zero
        gradient, which silently misleads calibration (#799). The profile
        spans both branches (below and above the ozone maximum) so each shape
        parameter is exercised where it acts.
        """
        config = ChemistryParameters.default()
        ozone_fields = ["ozone_scale_height", "ozone_max_vmr",
                        "ozone_tropopause_height"]
        declared = [f for f in vars(config) if f.startswith("ozone_")]
        self.assertEqual(sorted(declared), sorted(ozone_fields))

        nlev, ncols = 30, 2
        pressure = jnp.geomspace(100000.0, 50.0, nlev)[:, None] * jnp.ones((1, ncols))
        surface_pressure = jnp.full(ncols, 100000.0)
        temperature = jnp.full((nlev, ncols), 240.0)

        def total_ozone(cfg):
            return jnp.sum(fixed_ozone_distribution(
                pressure, surface_pressure, temperature, cfg))

        grads = jax.grad(total_ozone)(config)
        for field in ozone_fields:
            with self.subTest(field=field):
                self.assertTrue(jnp.isfinite(getattr(grads, field)))
                self.assertNotEqual(float(getattr(grads, field)), 0.0)


class TestMethaneChemistry(TestCase):
    """Test methane chemistry calculations"""
    
    def test_simple_methane_chemistry(self):
        """Test simple methane chemistry"""
        config = ChemistryParameters.default()
        
        # Create test data
        nlev, ncols = 10, 5
        pressure = jnp.linspace(100000, 10000, nlev)[:, None] * jnp.ones((1, ncols))
        temperature = jnp.ones((nlev, ncols)) * 280.0
        methane_vmr = jnp.ones((nlev, ncols)) * 1.9
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
        current_ozone = jnp.ones((nlev, ncols)) * 5.0
        current_methane = jnp.ones((nlev, ncols)) * 1.8
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
        
        # Check output shapes (CO2 is not a chemistry field — it is a
        # prescribed forcing).
        self.assertEqual(state.ozone_vmr.shape, (nlev, ncols))
        self.assertEqual(state.methane_vmr.shape, (nlev, ncols))

        # Check all values are positive and finite
        self.assertTrue(jnp.all(state.ozone_vmr > 0))
        self.assertTrue(jnp.all(state.methane_vmr > 0))
        self.assertTrue(jnp.all(jnp.isfinite(state.ozone_vmr)))
        self.assertTrue(jnp.all(jnp.isfinite(state.methane_vmr)))
        
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
        current_ozone = jnp.ones((nlev, ncols)) * 5.0
        current_methane = jnp.ones((nlev, ncols)) * 1.8
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
            current_methane = jnp.ones((nlev, ncols)) * 1.8
            dt = 3600.0
            
            tendencies, _ = simple_chemistry(
                pressure, surface_pressure, temperature,
                ozone_vmr, current_methane, dt, config
            )
            return jnp.sum(tendencies.ozone_tend ** 2)
        
        # Test gradient computation
        grad_fn = jax.grad(loss_fn)
        ozone_test = jnp.ones((5, 3)) * 5.0
        grad = grad_fn(ozone_test)
        
        self.assertEqual(grad.shape, ozone_test.shape)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))


if __name__ == "__main__":
    import unittest
    unittest.main()


class TestSimpleChemistryGradients(TestCase):
    """AD against a central difference for the chemistry tendencies (#820).

    Green. The scheme hard-unpacks ``nlev, ncols = pressure.shape``, so it is
    not broadcasting-native and there is one shape to check rather than two.
    """

    def test_gradients_match_a_central_difference(self):
        """A 10-level, 4-column block at realistic ozone and methane."""
        nlev, ncols = 10, 4
        pressure = (jnp.linspace(100000.0, 10000.0, nlev)[:, None]
                    * jnp.ones((1, ncols)))
        config = ChemistryParameters.default()
        check_gradients(
            lambda p, ps, t, o3, ch4: simple_chemistry(
                p, ps, t, o3, ch4, 3600.0, config),
            (pressure,
             jnp.ones(ncols) * 1.0e5,
             jnp.ones((nlev, ncols)) * 250.0,
             jnp.ones((nlev, ncols)) * 5.0,
             jnp.ones((nlev, ncols)) * 1.8),
            rtol=1e-3)
