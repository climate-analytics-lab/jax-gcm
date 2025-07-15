"""
Tests for simple boundary conditions

Date: 2025-01-15
"""

import jax.numpy as jnp
import jax
import pytest
from unittest import TestCase

from .simple_boundary_conditions import (
    BoundaryConditionParameters,
    simple_boundary_conditions,
    create_idealized_boundary_conditions,
    compute_solar_zenith_angle,
    compute_solar_irradiance,
    compute_surface_properties
)


class TestBoundaryConditionParameters(TestCase):
    """Test boundary condition parameters"""
    
    def test_default_parameters(self):
        """Test default parameter creation"""
        config = BoundaryConditionParameters.default()
        
        # Check solar parameters
        self.assertGreater(config.solar_constant, 1300.0)
        self.assertLess(config.solar_constant, 1400.0)
        self.assertGreater(config.solar_variability, 0.0)
        self.assertLess(config.solar_variability, 0.01)
        
        # Check greenhouse gas parameters
        self.assertGreater(config.co2_reference, 300.0)
        self.assertLess(config.co2_reference, 1000.0)
        self.assertGreater(config.ch4_reference, 1000.0)
        self.assertLess(config.ch4_reference, 3000.0)
        
        # Check surface parameters
        self.assertGreater(config.land_albedo_vis, 0.0)
        self.assertLess(config.land_albedo_vis, 1.0)
        self.assertGreater(config.ocean_albedo_vis, 0.0)
        self.assertLess(config.ocean_albedo_vis, 0.2)


class TestSolarCalculations(TestCase):
    """Test solar zenith angle and irradiance calculations"""
    
    def test_solar_zenith_angle(self):
        """Test solar zenith angle calculation"""
        # Test data
        latitude = jnp.array([0.0, jnp.pi/4, jnp.pi/2])  # Equator, 45°N, North Pole
        longitude = jnp.zeros(3)
        day_of_year = 180.0  # Summer solstice
        time_of_day = 12.0   # Noon
        
        zenith_angle = compute_solar_zenith_angle(
            latitude, longitude, day_of_year, time_of_day
        )
        
        # Check output shape
        self.assertEqual(zenith_angle.shape, (3,))
        
        # Check all values are between 0 and π
        self.assertTrue(jnp.all(zenith_angle >= 0))
        self.assertTrue(jnp.all(zenith_angle <= jnp.pi))
        
        # At summer solstice, zenith angle should be smaller at higher latitudes
        # (in northern hemisphere)
        self.assertLess(zenith_angle[1], zenith_angle[0])  # 45°N < equator
        
    def test_solar_irradiance(self):
        """Test solar irradiance calculation"""
        config = BoundaryConditionParameters.default()
        
        # Test data
        zenith_angle = jnp.array([0.0, jnp.pi/4, jnp.pi/2])  # Overhead, 45°, horizon
        day_of_year = 180.0
        year = 2020.0
        
        irradiance = compute_solar_irradiance(
            zenith_angle, day_of_year, year, config
        )
        
        # Check output shape
        self.assertEqual(irradiance.shape, (3,))
        
        # Check all values are non-negative
        self.assertTrue(jnp.all(irradiance >= 0))
        
        # Check irradiance decreases with zenith angle
        self.assertGreater(irradiance[0], irradiance[1])  # Overhead > 45°
        self.assertEqual(irradiance[2], 0.0)  # Horizon = 0
        
        # Check maximum irradiance is reasonable
        self.assertGreater(irradiance[0], 1000.0)
        self.assertLess(irradiance[0], 1500.0)


class TestSurfaceProperties(TestCase):
    """Test surface property calculations"""
    
    def test_surface_properties(self):
        """Test surface property computation"""
        config = BoundaryConditionParameters.default()
        
        # Test data
        land_fraction = jnp.array([1.0, 0.0, 0.5])  # Land, ocean, mixed
        sea_ice_fraction = jnp.array([0.0, 0.0, 0.0])  # No sea ice
        
        albedo_vis, albedo_nir, emissivity = compute_surface_properties(
            land_fraction, sea_ice_fraction, config
        )
        
        # Check output shapes
        self.assertEqual(albedo_vis.shape, (3,))
        self.assertEqual(albedo_nir.shape, (3,))
        self.assertEqual(emissivity.shape, (3,))
        
        # Check all values are between 0 and 1
        self.assertTrue(jnp.all(albedo_vis >= 0))
        self.assertTrue(jnp.all(albedo_vis <= 1))
        self.assertTrue(jnp.all(albedo_nir >= 0))
        self.assertTrue(jnp.all(albedo_nir <= 1))
        self.assertTrue(jnp.all(emissivity >= 0))
        self.assertTrue(jnp.all(emissivity <= 1))
        
        # Check that land has higher albedo than ocean
        self.assertGreater(albedo_vis[0], albedo_vis[1])  # Land > ocean
        self.assertGreater(albedo_nir[0], albedo_nir[1])  # Land > ocean
        
        # Check mixed surface has intermediate values
        self.assertGreater(albedo_vis[2], albedo_vis[1])  # Mixed > ocean
        self.assertLess(albedo_vis[2], albedo_vis[0])     # Mixed < land
        
    def test_surface_properties_with_sea_ice(self):
        """Test surface properties with sea ice"""
        config = BoundaryConditionParameters.default()
        
        # Test data with sea ice
        land_fraction = jnp.array([0.0, 0.0])  # Ocean only
        sea_ice_fraction = jnp.array([0.0, 1.0])  # No ice, full ice
        
        albedo_vis, albedo_nir, emissivity = compute_surface_properties(
            land_fraction, sea_ice_fraction, config
        )
        
        # Sea ice should have much higher albedo than ocean
        self.assertGreater(albedo_vis[1], albedo_vis[0])
        self.assertGreater(albedo_nir[1], albedo_nir[0])
        
        # Sea ice albedo should be high
        self.assertGreater(albedo_vis[1], 0.7)
        self.assertGreater(albedo_nir[1], 0.6)


class TestFullBoundaryConditions(TestCase):
    """Test full boundary condition calculations"""
    
    def test_simple_boundary_conditions(self):
        """Test simple boundary condition calculation"""
        config = BoundaryConditionParameters.default()
        
        # Test data
        ncols = 5
        latitude = jnp.linspace(-jnp.pi/2, jnp.pi/2, ncols)
        longitude = jnp.zeros(ncols)
        land_fraction = jnp.ones(ncols) * 0.3
        day_of_year = 180.0
        time_of_day = 12.0
        year = 2020.0
        
        bc_state = simple_boundary_conditions(
            latitude, longitude, land_fraction, day_of_year, time_of_day, year, config=config
        )
        
        # Check output shapes
        self.assertEqual(bc_state.solar_irradiance.shape, (ncols,))
        self.assertEqual(bc_state.solar_zenith_angle.shape, (ncols,))
        self.assertEqual(bc_state.surface_albedo_vis.shape, (ncols,))
        self.assertEqual(bc_state.co2_concentration.shape, (ncols,))
        
        # Check all values are reasonable
        self.assertTrue(jnp.all(bc_state.solar_irradiance >= 0))
        self.assertTrue(jnp.all(bc_state.solar_zenith_angle >= 0))
        self.assertTrue(jnp.all(bc_state.surface_albedo_vis >= 0))
        self.assertTrue(jnp.all(bc_state.surface_albedo_vis <= 1))
        self.assertTrue(jnp.all(bc_state.co2_concentration > 0))
        
        # Check greenhouse gas concentrations are constant
        self.assertTrue(jnp.all(bc_state.co2_concentration == config.co2_reference))
        self.assertTrue(jnp.all(bc_state.ch4_concentration == config.ch4_reference))
        
    def test_idealized_boundary_conditions(self):
        """Test idealized boundary condition creation"""
        ncols = 10
        
        bc_state = create_idealized_boundary_conditions(ncols)
        
        # Check output shapes
        self.assertEqual(bc_state.solar_irradiance.shape, (ncols,))
        self.assertEqual(bc_state.sea_surface_temperature.shape, (ncols,))
        self.assertEqual(bc_state.sea_ice_fraction.shape, (ncols,))
        
        # Check all values are finite
        self.assertTrue(jnp.all(jnp.isfinite(bc_state.solar_irradiance)))
        self.assertTrue(jnp.all(jnp.isfinite(bc_state.sea_surface_temperature)))
        self.assertTrue(jnp.all(jnp.isfinite(bc_state.surface_albedo_vis)))
        
        # Check SST is reasonable
        self.assertTrue(jnp.all(bc_state.sea_surface_temperature > 270.0))
        self.assertTrue(jnp.all(bc_state.sea_surface_temperature < 310.0))
        
        # Check sea ice fraction is between 0 and 1
        self.assertTrue(jnp.all(bc_state.sea_ice_fraction >= 0))
        self.assertTrue(jnp.all(bc_state.sea_ice_fraction <= 1))


class TestJAXCompatibility(TestCase):
    """Test JAX compatibility"""
    
    def test_jax_jit_compilation(self):
        """Test JIT compilation of boundary condition functions"""
        config = BoundaryConditionParameters.default()
        
        # Test data
        ncols = 5
        latitude = jnp.linspace(-jnp.pi/2, jnp.pi/2, ncols)
        longitude = jnp.zeros(ncols)
        land_fraction = jnp.ones(ncols) * 0.3
        day_of_year = 180.0
        time_of_day = 12.0
        year = 2020.0
        
        # Test JIT compilation
        jitted_bc = jax.jit(simple_boundary_conditions, static_argnums=(6,))
        
        bc_state = jitted_bc(
            latitude, longitude, land_fraction, day_of_year, time_of_day, year, config
        )
        
        # Should produce valid output
        self.assertEqual(bc_state.solar_irradiance.shape, (ncols,))
        self.assertTrue(jnp.all(jnp.isfinite(bc_state.solar_irradiance)))
        
    def test_gradient_computation(self):
        """Test gradient computation"""
        config = BoundaryConditionParameters.default()
        
        def loss_fn(latitude):
            longitude = jnp.zeros_like(latitude)
            land_fraction = jnp.ones_like(latitude) * 0.3
            day_of_year = 180.0
            time_of_day = 12.0
            year = 2020.0
            
            bc_state = simple_boundary_conditions(
                latitude, longitude, land_fraction, day_of_year, time_of_day, year, config=config
            )
            return jnp.sum(bc_state.solar_irradiance)
        
        # Test gradient computation
        grad_fn = jax.grad(loss_fn)
        latitude_test = jnp.array([0.0, jnp.pi/4, jnp.pi/2])
        grad = grad_fn(latitude_test)
        
        self.assertEqual(grad.shape, latitude_test.shape)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))


if __name__ == "__main__":
    unittest.main()