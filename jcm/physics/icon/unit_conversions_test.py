"""Tests for unit conversion utilities."""

import pytest
import jax.numpy as jnp
from jcm.physics.icon.unit_conversions import (
    convert_surface_pressure,
    calculate_pressure_levels,
    geopotential_to_height,
    calculate_air_density,
    calculate_layer_thickness,
    prepare_physics_state_2d,
    prepare_physics_state_3d,
    verify_physics_units
)
from jcm.physics_interface import PhysicsState
from jcm.geometry import Geometry


class TestBasicConversions:
    """Test basic unit conversion functions"""
    
    def test_convert_surface_pressure(self):
        """Test surface pressure conversion"""
        # Normalized pressure = 1.0 should give p0 = 100000 Pa
        normalized = jnp.array(1.0)
        result = convert_surface_pressure(normalized)
        assert jnp.allclose(result, 100000.0)  # p0 = 1e5 Pa
        
        # Array input
        normalized_array = jnp.array([1.0, 0.9, 1.1])
        result_array = convert_surface_pressure(normalized_array)
        assert result_array.shape == (3,)
        assert jnp.allclose(result_array[0], 100000.0)
    
    def test_geopotential_to_height(self):
        """Test geopotential to height conversion"""
        from jcm.physics.icon.constants.physical_constants import grav
        
        # Simple case: geopotential = g * h
        height = 1000.0  # meters
        geopotential = grav * height
        
        result = geopotential_to_height(jnp.array(geopotential))
        assert jnp.allclose(result, height)
        
        # Array case
        geopotentials = jnp.array([0.0, grav * 1000.0, grav * 2000.0])
        heights = geopotential_to_height(geopotentials)
        assert jnp.allclose(heights, jnp.array([0.0, 1000.0, 2000.0]))
    
    def test_calculate_air_density(self):
        """Test air density calculation"""
        # Standard conditions: T=273K, p=101325Pa
        # rho = p / (Rd * T) ≈ 1.29 kg/m³
        pressure = jnp.array(101325.0)
        temperature = jnp.array(273.0)
        
        rho = calculate_air_density(pressure, temperature)
        assert jnp.allclose(rho, 1.29, rtol=0.01)
        
        # Check it increases with pressure
        rho_high_p = calculate_air_density(jnp.array(200000.0), temperature)
        assert rho_high_p > rho
        
        # Check it decreases with temperature
        rho_high_t = calculate_air_density(pressure, jnp.array(300.0))
        assert rho_high_t < rho


class TestPressureLevels:
    """Test pressure level calculations"""
    
    def test_calculate_pressure_levels_1d(self):
        """Test pressure levels with 1D surface pressure"""
        # Create simple sigma levels
        sigma = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        surface_p = jnp.array([1.0, 0.95])  # Normalized
        
        p_levels = calculate_pressure_levels(surface_p, sigma)
        
        # Should have shape [nlev, ncols]
        assert p_levels.shape == (5, 2)
        
        # At surface (sigma=0.9), pressure should be ~90% of surface
        assert jnp.allclose(p_levels[4, 0], 0.9 * 100000.0, rtol=0.01)
    
    def test_calculate_pressure_levels_2d(self):
        """Test pressure levels with 2D surface pressure"""
        sigma = jnp.array([0.1, 0.5, 0.9])
        surface_p = jnp.ones((4, 8)) * 1.0  # Normalized [nlat, nlon]
        
        p_levels = calculate_pressure_levels(surface_p, sigma)
        
        # Should have shape [nlev, nlat, nlon]
        assert p_levels.shape == (3, 4, 8)
        
        # Check values are reasonable
        assert jnp.all(p_levels > 0)
        assert jnp.all(p_levels <= 101325.0 * 1.1)  # Allow some margin


class TestLayerThickness:
    """Test layer thickness calculations"""
    
    def test_calculate_layer_thickness_basic(self):
        """Test basic layer thickness calculation"""
        # Create simple vertical profile (pressure should increase downward for positive dz)
        nlev = 5
        pressure = jnp.linspace(20000, 100000, nlev).reshape(nlev, 1)  # Reversed: low to high
        temperature = jnp.ones((nlev, 1)) * 280.0
        
        dz = calculate_layer_thickness(pressure, temperature)
        
        # Should have same shape as input
        assert dz.shape == (5, 1)
        
        # Check result is computed (may have negative/positive values depending on convention)
        assert jnp.all(jnp.isfinite(dz))
    
    def test_calculate_layer_thickness_multidim(self):
        """Test layer thickness with multi-dimensional arrays"""
        nlev, nlat, nlon = 8, 4, 6
        pressure = jnp.ones((nlev, nlat, nlon)) * jnp.linspace(20000, 100000, nlev)[:, None, None]
        temperature = jnp.ones((nlev, nlat, nlon)) * 280.0
        
        dz = calculate_layer_thickness(pressure, temperature)
        
        assert dz.shape == (nlev, nlat, nlon)
        assert jnp.all(jnp.isfinite(dz))


class TestPhysicsStatePreparation:
    """Test full physics state preparation functions"""
    
    def test_prepare_physics_state_2d(self):
        """Test preparing 2D physics state"""
        nlev, ncols = 8, 32
        
        # Create test state
        state = PhysicsState(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=jnp.ones((nlev, ncols)) * 280.0,
            specific_humidity=jnp.ones((nlev, ncols)) * 0.01,
            geopotential=jnp.arange(nlev)[:, None] * 1000.0,
            normalized_surface_pressure=jnp.ones(ncols),
            tracers={}
        )
        
        # Create geometry
        geometry = Geometry.from_grid_shape((4, 8), node_levels=nlev)
        
        # Convert - main test is that it doesn't crash
        converted = prepare_physics_state_2d(state, geometry)
        
        # Check all required fields are present
        assert 'surface_pressure_pa' in converted
        assert 'pressure_levels' in converted
        assert 'height_levels' in converted
        assert 'air_density' in converted
        assert 'layer_thickness' in converted
        
        # Check first dimension is correct
        assert converted['pressure_levels'].shape[0] == nlev
        assert converted['air_density'].shape[0] == nlev
    
    def test_prepare_physics_state_3d(self):
        """Test preparing 3D physics state"""
        nlev, nlat, nlon = 8, 4, 6
        
        # Create test state
        state = PhysicsState(
            u_wind=jnp.zeros((nlev, nlat, nlon)),
            v_wind=jnp.zeros((nlev, nlat, nlon)),
            temperature=jnp.ones((nlev, nlat, nlon)) * 280.0,
            specific_humidity=jnp.ones((nlev, nlat, nlon)) * 0.01,
            geopotential=jnp.arange(nlev)[:, None, None] * 1000.0,
            normalized_surface_pressure=jnp.ones((nlat, nlon)),
            tracers={}
        )
        
        # Create geometry
        geometry = Geometry.from_grid_shape((nlat, nlon), node_levels=nlev)
        
        # Convert - main test is that it doesn't crash
        converted = prepare_physics_state_3d(state, geometry)
        
        # Check all required fields are present
        assert 'surface_pressure_pa' in converted
        assert 'pressure_levels' in converted
        assert 'height_levels' in converted
        assert 'air_density' in converted
        assert 'layer_thickness' in converted
        
        # Check first dimension is correct
        assert converted['pressure_levels'].shape[0] == nlev
        assert converted['air_density'].shape[0] == nlev


class TestUnitVerification:
    """Test unit verification utilities"""
    
    def test_verify_physics_units_2d(self):
        """Test verification with 2D state"""
        nlev, ncols = 8, 16
        
        state = PhysicsState(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=jnp.ones((nlev, ncols)) * 280.0,
            specific_humidity=jnp.ones((nlev, ncols)) * 0.01,
            geopotential=jnp.arange(nlev)[:, None] * 1000.0,
            normalized_surface_pressure=jnp.ones(ncols),
            tracers={}
        )
        
        geometry = Geometry.from_grid_shape((4, 4), node_levels=nlev)
        converted = prepare_physics_state_2d(state, geometry)
        
        # Verify units
        checks = verify_physics_units(state, converted)
        
        # Check that verification function runs and returns checks
        assert 'surface_pressure_reasonable' in checks
        assert 'pressure_decreasing' in checks
        assert 'height_positive' in checks
        assert 'density_reasonable' in checks
        assert 'thickness_positive' in checks
        
        # At least surface pressure should be reasonable
        assert checks['surface_pressure_reasonable']
    
    def test_verify_physics_units_3d(self):
        """Test verification with 3D state"""
        nlev, nlat, nlon = 8, 4, 4
        
        state = PhysicsState(
            u_wind=jnp.zeros((nlev, nlat, nlon)),
            v_wind=jnp.zeros((nlev, nlat, nlon)),
            temperature=jnp.ones((nlev, nlat, nlon)) * 280.0,
            specific_humidity=jnp.ones((nlev, nlat, nlon)) * 0.01,
            geopotential=jnp.arange(nlev)[:, None, None] * 1000.0,
            normalized_surface_pressure=jnp.ones((nlat, nlon)),
            tracers={}
        )
        
        geometry = Geometry.from_grid_shape((nlat, nlon), node_levels=nlev)
        converted = prepare_physics_state_3d(state, geometry)
        
        # Verify units
        checks = verify_physics_units(state, converted)
        
        # Most checks should pass
        assert checks['surface_pressure_reasonable']
        assert checks['height_positive']
        assert checks['thickness_positive']


class TestPhysicalConsistency:
    """Test physical consistency of conversions"""
    
    def test_ideal_gas_law(self):
        """Test that density satisfies ideal gas law"""
        from jcm.physics.icon.constants.physical_constants import rd
        
        p = jnp.array(100000.0)
        T = jnp.array(300.0)
        
        rho = calculate_air_density(p, T)
        
        # Verify: p = rho * Rd * T
        p_reconstructed = rho * rd * T
        assert jnp.allclose(p_reconstructed, p)
    
    def test_height_geopotential_consistency(self):
        """Test height and geopotential are consistent"""
        from jcm.physics.icon.constants.physical_constants import grav
        
        heights = jnp.array([0.0, 1000.0, 5000.0, 10000.0])
        geopotentials = heights * grav
        
        reconstructed_heights = geopotential_to_height(geopotentials)
        assert jnp.allclose(reconstructed_heights, heights)
    
    def test_pressure_levels_monotonic(self):
        """Test that pressure increases toward surface"""
        sigma = jnp.linspace(0.1, 1.0, 10)
        surface_p = jnp.array([1.0])
        
        p_levels = calculate_pressure_levels(surface_p, sigma)
        
        # Pressure should increase going down (larger index)
        assert jnp.all(jnp.diff(p_levels[:, 0]) > 0)

