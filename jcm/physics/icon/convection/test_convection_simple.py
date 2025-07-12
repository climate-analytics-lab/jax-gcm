"""
Simplified tests for convection scheme focusing on basic functionality

Date: 2025-01-09
"""

import jax.numpy as jnp
import jax
import numpy as np

# Import basic convection functions
from jcm.physics.icon.convection.tiedtke_nordeng import (
    ConvectionParameters,
    saturation_mixing_ratio,
    find_cloud_base,
    calculate_cape_cin
)


def create_simple_atmosphere(nlev=20):
    """Create a simple test atmospheric profile"""
    # Pressure levels (Pa) - from surface (1000 hPa) to top (~200 hPa)
    pressure = jnp.linspace(1e5, 2e4, nlev)
    
    # Height (m) - increases with decreasing pressure
    height = jnp.linspace(0, 12000, nlev)
    
    # Temperature - linear lapse rate, warm at surface
    temperature = 300.0 - 6.5e-3 * height
    
    # Humidity - high at surface, decreasing with height, but limited by saturation
    surface_humidity = 0.012  # 12 g/kg at surface
    humidity_profile = surface_humidity * jnp.exp(-height / 2000.0)
    
    # Limit humidity to 90% of saturation to avoid super-saturation
    qs_profile = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
    humidity = jnp.minimum(humidity_profile, 0.9 * qs_profile)
    
    # Simple wind profile
    u_wind = jnp.full(nlev, 10.0)
    v_wind = jnp.zeros(nlev)
    
    return {
        'temperature': temperature,
        'humidity': humidity, 
        'pressure': pressure,
        'height': height,
        'u_wind': u_wind,
        'v_wind': v_wind
    }


def test_basic_functions():
    """Test basic convection functions"""
    print("=== Testing Basic Functions ===")
    
    # Test saturation mixing ratio
    temp = 300.0
    press = 1e5
    qs = saturation_mixing_ratio(press, temp)
    print(f"Saturation mixing ratio: {qs*1000:.1f} g/kg")
    assert 0.01 < qs < 0.05, "Unrealistic saturation mixing ratio"
    
    # Test with array inputs
    temps = jnp.array([280.0, 290.0, 300.0, 310.0])
    qs_array = jax.vmap(saturation_mixing_ratio, in_axes=(None, 0))(press, temps)
    print(f"Temperature dependence: {qs_array*1000}")
    assert jnp.all(jnp.diff(qs_array) > 0), "Should increase with temperature"
    
    print("✓ Basic functions working")


def test_cloud_base():
    """Test cloud base detection"""
    print("\n=== Testing Cloud Base Detection ===")
    
    atm = create_simple_atmosphere()
    config = ConvectionParameters.default()
    
    cloud_base, has_cloud_base = find_cloud_base(
        atm['temperature'], atm['humidity'], atm['pressure'], config
    )
    
    print(f"Cloud base found: {has_cloud_base}")
    if has_cloud_base:
        cb_height = atm['height'][cloud_base]
        print(f"Cloud base: level {cloud_base}, height {cb_height/1000:.1f} km")
        assert 200 < cb_height < 3000, f"Unrealistic cloud base height: {cb_height}"
    
    print("✓ Cloud base detection working")


def test_cape():
    """Test CAPE calculation"""
    print("\n=== Testing CAPE Calculation ===")
    
    atm = create_simple_atmosphere()
    config = ConvectionParameters.default()
    
    # Find cloud base first
    cloud_base, has_cloud_base = find_cloud_base(
        atm['temperature'], atm['humidity'], atm['pressure'], config
    )
    
    if has_cloud_base:
        cape, cin = calculate_cape_cin(
            atm['temperature'], atm['humidity'], atm['pressure'],
            atm['height'], cloud_base, config
        )
        
        print(f"CAPE: {cape:.0f} J/kg")
        print(f"CIN: {cin:.0f} J/kg")
        
        assert cape >= 0, "CAPE should be non-negative"
        assert cin >= 0, "CIN should be non-negative"
        assert cape < 10000, "CAPE too high"
        
    print("✓ CAPE calculation working")


def test_jax_transformations():
    """Test JAX transformations on basic functions"""
    print("\n=== Testing JAX Transformations ===")
    
    # Test JIT compilation
    jitted_saturation = jax.jit(saturation_mixing_ratio)
    qs = jitted_saturation(1e5, 300.0)
    print(f"JIT compilation successful: {qs*1000:.1f} g/kg")
    
    # Test vectorization
    pressures = jnp.array([1e5, 8e4, 6e4, 4e4])
    temperatures = jnp.array([300.0, 290.0, 280.0, 270.0])
    
    vmap_saturation = jax.vmap(saturation_mixing_ratio)
    qs_vec = vmap_saturation(pressures, temperatures)
    print(f"Vectorized calculation: {qs_vec*1000}")
    
    # Test gradient
    def loss_fn(temp):
        return saturation_mixing_ratio(1e5, temp)
    
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(300.0)
    print(f"Gradient: {gradient:.2e}")
    assert gradient > 0, "Gradient should be positive (qs increases with T)"
    
    print("✓ JAX transformations working")


def test_physical_consistency():
    """Test physical consistency of calculations"""
    print("\n=== Testing Physical Consistency ===")
    
    atm = create_simple_atmosphere()
    
    # Check that humidity is reasonable
    max_humidity = jnp.max(atm['humidity'])
    min_humidity = jnp.min(atm['humidity'])
    print(f"Humidity range: {min_humidity*1000:.3f} to {max_humidity*1000:.1f} g/kg")
    assert 0 < max_humidity < 0.05, "Humidity out of realistic range"
    
    # Check temperature profile
    temp_gradient = jnp.mean(jnp.diff(atm['temperature']) / jnp.diff(atm['height']))
    print(f"Temperature lapse rate: {-temp_gradient*1000:.1f} K/km")
    assert 5 < -temp_gradient*1000 < 10, "Unrealistic lapse rate"
    
    # Check saturation at each level
    qs_profile = jax.vmap(saturation_mixing_ratio)(atm['pressure'], atm['temperature'])
    rel_humidity = atm['humidity'] / qs_profile
    max_rh = jnp.max(rel_humidity)
    print(f"Maximum relative humidity: {max_rh:.1%}")
    assert max_rh < 1.2, "Super-saturation too high"
    
    print("✓ Physical consistency checks passed")


def run_simple_tests():
    """Run all simple tests"""
    print("="*50)
    print("SIMPLE CONVECTION TESTS")
    print("="*50)
    
    try:
        test_basic_functions()
        test_cloud_base()
        test_cape()
        test_jax_transformations()
        test_physical_consistency()
        
        print("\n" + "="*50)
        print("ALL SIMPLE TESTS PASSED! ✓")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)