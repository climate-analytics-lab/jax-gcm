"""
Tests for unit conversions in ICON physics

Date: 2025-01-10
"""

import jax.numpy as jnp
import pytest
from .unit_conversions import (
    convert_surface_pressure,
    calculate_pressure_levels,
    geopotential_to_height,
    calculate_air_density,
    calculate_layer_thickness,
    prepare_physics_state_2d,
    verify_physics_units
)
from jcm.physics_interface import PhysicsState
from jcm.geometry import Geometry
from ..speedy.physical_constants import p0


class TestUnitConversions:
    """Test unit conversion functions"""
    
    def test_surface_pressure_conversion(self):
        """Test surface pressure conversion from normalized to Pa"""
        # Normalized surface pressure around 1.0
        ps_norm = jnp.array([0.98, 1.0, 1.02])
        
        ps_pa = convert_surface_pressure(ps_norm)
        
        # Should be around 100000 Pa
        assert jnp.allclose(ps_pa, ps_norm * p0)
        assert jnp.all((ps_pa > 90000) & (ps_pa < 110000))
    
    def test_pressure_levels_calculation(self):
        """Test pressure level calculation"""
        # Simple test with 5 levels
        nlev = 5
        sigma = jnp.linspace(0.1, 1.0, nlev)  # Top to surface (standard ordering)
        ps_norm = jnp.array([1.0, 1.01])  # 2 columns
        
        pressure = calculate_pressure_levels(ps_norm, sigma)
        
        # Check shape
        assert pressure.shape == (nlev, 2)
        
        # Check values increase with index (top to bottom)
        assert jnp.all(jnp.diff(pressure, axis=0) > 0)
        
        # Check surface pressure matches (last level)
        assert jnp.allclose(pressure[-1, :], ps_norm * p0)
        
        # Check top pressure (first level)
        assert jnp.allclose(pressure[0, :], 0.1 * ps_norm * p0)
    
    def test_geopotential_to_height(self):
        """Test geopotential to height conversion"""
        # Geopotential at different levels
        geopotential = jnp.array([0, 9810, 19620, 29430])  # 0, 1000, 2000, 3000 m
        
        height = geopotential_to_height(geopotential)
        
        expected = jnp.array([0, 1000, 2000, 3000])
        assert jnp.allclose(height, expected, rtol=1e-3)
    
    def test_air_density_calculation(self):
        """Test air density calculation"""
        # Standard atmosphere at sea level
        pressure = jnp.array(101325.0)  # Pa
        temperature = jnp.array(288.15)  # K
        
        rho = calculate_air_density(pressure, temperature)
        
        # Standard density is about 1.225 kg/m³
        assert jnp.abs(rho - 1.225) < 0.01
        
        # Test at altitude
        pressure_alt = jnp.array(50000.0)  # Pa (about 5.5 km)
        temperature_alt = jnp.array(255.0)  # K
        
        rho_alt = calculate_air_density(pressure_alt, temperature_alt)
        
        # Should be less dense
        assert rho_alt < rho
        assert 0.5 < rho_alt < 0.8
    
    def test_layer_thickness_calculation(self):
        """Test layer thickness calculation"""
        # Simple 3-level atmosphere (top to bottom)
        pressure = jnp.array([20000, 50000, 100000])  # Pa
        temperature = jnp.array([220, 260, 288])  # K
        
        dz = calculate_layer_thickness(pressure, temperature)
        
        # All thicknesses should be positive
        assert jnp.all(dz > 0)
        
        # Lower layers should be thinner (higher density) 
        # Since index 2 is surface, it should have smaller dz than index 0 (top)
        assert dz[2] < dz[0]
        
        # Reasonable values (hundreds to thousands of meters)
        assert jnp.all(dz > 100)
        assert jnp.all(dz < 10000)
    
    def test_prepare_physics_state_2d(self):
        """Test full physics state preparation"""
        # Create a simple test state
        nlev = 10
        ncols = 20
        
        # Create state
        height = jnp.linspace(0, 10000, nlev)
        temp_profile = 288 - 0.0065 * height
        temperature = jnp.broadcast_to(temp_profile[:, None], (nlev, ncols))
        
        state = PhysicsState(
            u_wind=jnp.ones((nlev, ncols)) * 10,
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=temperature,
            specific_humidity=jnp.ones((nlev, ncols)) * 0.01,
            geopotential=jnp.broadcast_to((height * 9.81)[:, None], (nlev, ncols)),
            surface_pressure=jnp.ones(ncols),  # Normalized
            tracers={}
        )
        
        # Create simple geometry
        class SimpleGeometry:
            fsg = jnp.linspace(0.1, 1.0, nlev)  # Top to surface
        
        geometry = SimpleGeometry()
        
        # Prepare state
        converted = prepare_physics_state_2d(state, geometry)
        
        # Verify all fields are present
        assert 'surface_pressure_pa' in converted
        assert 'pressure_levels' in converted
        assert 'height_levels' in converted
        assert 'air_density' in converted
        assert 'layer_thickness' in converted
        
        # Verify shapes
        assert converted['surface_pressure_pa'].shape == (ncols,)
        assert converted['pressure_levels'].shape == (nlev, ncols)
        assert converted['height_levels'].shape == (nlev, ncols)
        assert converted['air_density'].shape == (nlev, ncols)
        assert converted['layer_thickness'].shape == (nlev, ncols)
    
    def test_verify_physics_units(self):
        """Test unit verification"""
        # Create reasonable converted state
        converted = {
            'surface_pressure_pa': jnp.array([98000, 101000, 103000]),
            'pressure_levels': jnp.array([[20000, 21000, 22000],
                                         [50000, 51000, 52000],
                                         [98000, 101000, 103000]]),
            'height_levels': jnp.array([[8000, 7900, 7800],
                                       [5000, 4900, 4800],
                                       [0, 0, 0]]),
            'air_density': jnp.array([[0.3, 0.31, 0.32],
                                     [0.7, 0.71, 0.72],
                                     [1.2, 1.21, 1.22]]),
            'layer_thickness': jnp.array([[3000, 2950, 2900],
                                         [3000, 3000, 3000],
                                         [2000, 1950, 1900]])
        }
        
        state = None  # Not used in verification
        
        checks = verify_physics_units(state, converted)
        
        # All checks should pass
        assert checks['surface_pressure_reasonable']
        assert checks['pressure_decreasing']
        assert checks['height_positive']
        assert checks['height_increasing']
        assert checks['density_reasonable']
        assert checks['thickness_positive']


def test_unit_conversions():
    """Run all unit conversion tests"""
    test = TestUnitConversions()
    
    print("Testing surface pressure conversion...")
    test.test_surface_pressure_conversion()
    print("✓ Surface pressure conversion passed")
    
    print("\nTesting pressure level calculation...")
    test.test_pressure_levels_calculation()
    print("✓ Pressure levels passed")
    
    print("\nTesting geopotential to height...")
    test.test_geopotential_to_height()
    print("✓ Height conversion passed")
    
    print("\nTesting air density...")
    test.test_air_density_calculation()
    print("✓ Air density passed")
    
    print("\nTesting layer thickness...")
    test.test_layer_thickness_calculation()
    print("✓ Layer thickness passed")
    
    print("\nTesting full state preparation...")
    test.test_prepare_physics_state_2d()
    print("✓ State preparation passed")
    
    print("\nTesting unit verification...")
    test.test_verify_physics_units()
    print("✓ Unit verification passed")
    
    print("\nAll unit conversion tests passed!")


if __name__ == "__main__":
    test_unit_conversions()