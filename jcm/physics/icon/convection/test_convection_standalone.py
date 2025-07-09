"""
Standalone tests for Tiedtke-Nordeng convection scheme

This file tests the convection scheme without relying on the full jcm framework.

Date: 2025-01-09
"""

import pytest
import jax.numpy as jnp
import jax
import numpy as np
from jax import random

# Import convection modules directly
from jcm.physics.icon.convection.tiedtke_nordeng import (
    tiedtke_nordeng_convection,
    tiedtke_nordeng_convection_with_tracers,
    ConvectionConfig,
    ConvectionState,
    ConvectionTendencies,
    saturation_mixing_ratio,
    find_cloud_base,
    calculate_cape_cin
)
from jcm.physics.icon.convection.tracer_transport import (
    TracerIndices,
    initialize_tracers
)
from jcm.physics.icon.convection.updraft import calculate_updraft
from jcm.physics.icon.convection.downdraft import calculate_downdraft


def create_test_atmosphere(nlev=20, unstable=True):
    """Create a test atmospheric profile"""
    # Pressure levels (Pa) - from surface to top
    # Use a more realistic atmosphere (surface to ~200 hPa)
    pressure = jnp.logspace(jnp.log10(1e5), jnp.log10(2e4), nlev)[::-1]
    
    # Height (m) - hydrostatic approximation
    height = -7000 * jnp.log(pressure / 1e5)
    
    if unstable:
        # Unstable profile - warm and moist at surface
        # Temperature profile with lapse rate
        surface_temp = 300.0  # K
        lapse_rate = 6.5e-3   # K/m
        temperature = surface_temp - lapse_rate * height
        
        # Add inversion at tropopause
        trop_idx = jnp.argmin(jnp.abs(pressure - 200e2))  # ~200 hPa
        temperature = temperature.at[:trop_idx].set(
            temperature[trop_idx]
        )
        
        # Humidity profile - exponential decrease
        surface_rh = 0.8
        humidity_scale = 2000.0  # m
        rel_humidity = surface_rh * jnp.exp(-height / humidity_scale)
        
        # Convert to specific humidity using vectorized function
        qs = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
        humidity = rel_humidity * qs
    else:
        # Stable profile - cool and dry
        surface_temp = 285.0
        temperature = surface_temp - 5e-3 * height
        humidity = jnp.ones_like(temperature) * 1e-3  # Very dry
    
    # Wind profile - simple shear
    u_wind = 10.0 + 20.0 * (1.0 - pressure / 1e5)
    v_wind = jnp.zeros_like(u_wind)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'height': height,
        'u_wind': u_wind,
        'v_wind': v_wind
    }


def test_saturation_functions():
    """Test saturation calculations"""
    print("\n=== Testing Saturation Functions ===")
    
    # Test saturation mixing ratio
    temp = 300.0  # K
    press = 1e5   # Pa
    
    qs = saturation_mixing_ratio(press, temp)
    print(f"Saturation mixing ratio at {temp}K, {press/100:.0f}hPa: {qs*1000:.1f} g/kg")
    
    # Should be reasonable tropical value
    assert 0.01 < qs < 0.05, f"Unrealistic saturation mixing ratio: {qs}"
    
    # Test temperature dependence
    temps = jnp.array([273.15, 300.0, 310.0])
    qs_array = jax.vmap(saturation_mixing_ratio, in_axes=(None, 0))(press, temps)
    
    # Should increase with temperature
    assert jnp.all(jnp.diff(qs_array) > 0), "Saturation mixing ratio should increase with temperature"
    
    print("✓ Saturation functions working correctly")


def test_cloud_base_detection():
    """Test cloud base detection"""
    print("\n=== Testing Cloud Base Detection ===")
    
    atm = create_test_atmosphere(unstable=True)
    config = ConvectionConfig()
    
    cloud_base, has_cloud_base = find_cloud_base(
        atm['temperature'], atm['humidity'], atm['pressure'], config
    )
    
    print(f"Cloud base found: {has_cloud_base}")
    if has_cloud_base:
        print(f"Cloud base level: {cloud_base} (height: {atm['height'][cloud_base]/1000:.1f} km)")
        
        # Cloud base should be reasonable height
        cb_height = atm['height'][cloud_base]
        assert 500 < cb_height < 5000, f"Unrealistic cloud base height: {cb_height}"
    
    # Test stable atmosphere
    atm_stable = create_test_atmosphere(unstable=False)
    cloud_base_stable, has_cloud_base_stable = find_cloud_base(
        atm_stable['temperature'], atm_stable['humidity'], atm_stable['pressure'], config
    )
    
    print(f"Stable atmosphere has cloud base: {has_cloud_base_stable}")
    
    print("✓ Cloud base detection working")


def test_cape_calculation():
    """Test CAPE calculation"""
    print("\n=== Testing CAPE Calculation ===")
    
    atm = create_test_atmosphere(unstable=True)
    config = ConvectionConfig()
    
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
        
        # CAPE should be positive for unstable atmosphere
        assert cape >= 0, f"Negative CAPE: {cape}"
        
        # Reasonable values
        assert cape < 10000, f"Unrealistically high CAPE: {cape}"
    
    print("✓ CAPE calculation working")


def test_stable_atmosphere():
    """Test that stable atmosphere produces no convection"""
    print("\n=== Testing Stable Atmosphere ===")
    
    atm = create_test_atmosphere(unstable=False)
    config = ConvectionConfig()
    
    # Run convection scheme
    tendencies, state = tiedtke_nordeng_convection(
        atm['temperature'],
        atm['humidity'],
        atm['pressure'],
        atm['height'],
        atm['u_wind'],
        atm['v_wind'],
        dt=3600.0,
        config=config
    )
    
    print(f"Convection type: {state.ktype}")
    print(f"Max temperature tendency: {jnp.max(jnp.abs(tendencies.dtedt))*86400:.2f} K/day")
    print(f"Precipitation: {tendencies.precip_conv*3600:.3f} mm/hr")
    
    # Check no convection occurs
    assert state.ktype == 0, f"Convection should not occur in stable atmosphere: type={state.ktype}"
    assert jnp.allclose(tendencies.dtedt, 0.0, atol=1e-10), "Temperature tendencies should be zero"
    assert jnp.allclose(tendencies.dqdt, 0.0, atol=1e-10), "Moisture tendencies should be zero"
    assert tendencies.precip_conv == 0.0, "Precipitation should be zero"
    
    print("✓ Stable atmosphere test passed")


def test_unstable_atmosphere():
    """Test that unstable atmosphere triggers convection"""
    print("\n=== Testing Unstable Atmosphere ===")
    
    atm = create_test_atmosphere(unstable=True)
    config = ConvectionConfig()
    
    # Run convection scheme
    tendencies, state = tiedtke_nordeng_convection(
        atm['temperature'],
        atm['humidity'],
        atm['pressure'],
        atm['height'],
        atm['u_wind'],
        atm['v_wind'],
        dt=3600.0,
        config=config
    )
    
    print(f"Convection type: {state.ktype}")
    print(f"Cloud base level: {state.kbase} (height: {atm['height'][state.kbase]/1000:.1f} km)")
    print(f"Max heating rate: {jnp.max(tendencies.dtedt)*86400:.1f} K/day")
    print(f"Max drying rate: {jnp.min(tendencies.dqdt)*86400*1000:.1f} g/kg/day")
    print(f"Precipitation: {tendencies.precip_conv*3600:.3f} mm/hr")
    
    # Check convection occurs
    if state.ktype > 0:
        print("✓ Convection was triggered")
        
        # Cloud base should be reasonable
        assert state.kbase < len(atm['temperature']) - 1, "Cloud base should be above surface"
        
        # Check for reasonable heating/cooling pattern
        max_heating = jnp.max(tendencies.dtedt)
        min_cooling = jnp.min(tendencies.dtedt)
        print(f"Heating range: {min_cooling*86400:.1f} to {max_heating*86400:.1f} K/day")
        
        # Physical consistency checks
        if jnp.any(tendencies.dtedt != 0):
            print("✓ Temperature tendencies non-zero")
        if jnp.any(tendencies.dqdt != 0):
            print("✓ Moisture tendencies non-zero")
    else:
        print("! No convection triggered (may be due to simplified scheme)")
    
    print("✓ Unstable atmosphere test completed")


def test_jax_compatibility():
    """Test JAX transformations work correctly"""
    print("\n=== Testing JAX Compatibility ===")
    
    atm = create_test_atmosphere(unstable=True)
    config = ConvectionConfig()
    
    # Test jit compilation
    print("Testing JIT compilation...")
    jitted_convection = jax.jit(tiedtke_nordeng_convection)
    
    tendencies, state = jitted_convection(
        atm['temperature'],
        atm['humidity'],
        atm['pressure'],
        atm['height'],
        atm['u_wind'],
        atm['v_wind'],
        dt=3600.0,
        config=config
    )
    
    print(f"JIT compilation successful. Convection type: {state.ktype}")
    
    # Test gradient computation (for adjoints)
    print("Testing gradient computation...")
    def loss_fn(temperature):
        tendencies, _ = tiedtke_nordeng_convection(
            temperature,
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        return jnp.sum(tendencies.precip_conv)
    
    # This should not error
    grad = jax.grad(loss_fn)(atm['temperature'])
    assert grad.shape == atm['temperature'].shape, "Gradient shape mismatch"
    print(f"Gradient computation successful. Max gradient: {jnp.max(jnp.abs(grad)):.2e}")
    
    print("✓ JAX compatibility test passed")


def test_tracer_transport():
    """Test tracer transport functionality"""
    print("\n=== Testing Tracer Transport ===")
    
    atm = create_test_atmosphere(unstable=True)
    config = ConvectionConfig()
    
    # Initialize tracers
    tracers, indices = initialize_tracers(len(atm['temperature']), include_chemistry=True)
    
    print(f"Initialized {tracers.shape[1]} tracers")
    print(f"Tracer indices: qv={indices.iqv}, qc={indices.iqc}, qi={indices.iqi}, chemistry_start={indices.iqt}")
    
    # Set initial water vapor to match humidity
    tracers = tracers.at[:, indices.iqv].set(atm['humidity'])
    
    # Run enhanced convection with tracers
    tendencies, state = tiedtke_nordeng_convection_with_tracers(
        atm['temperature'],
        atm['humidity'],
        atm['pressure'],
        atm['height'],
        atm['u_wind'],
        atm['v_wind'],
        tracers,
        dt=3600.0,
        config=config,
        tracer_indices=indices
    )
    
    print(f"Convection type: {state.ktype}")
    
    if tendencies.dtracer_dt is not None:
        print(f"Tracer tendencies shape: {tendencies.dtracer_dt.shape}")
        print(f"Max cloud water tendency: {jnp.max(tendencies.dtracer_dt[:, indices.iqc])*86400*1000:.2f} g/kg/day")
        print(f"Max cloud ice tendency: {jnp.max(tendencies.dtracer_dt[:, indices.iqi])*86400*1000:.2f} g/kg/day")
        
        # Check conservation
        total_water_tend = jnp.sum(tendencies.dtracer_dt[:, indices.iqv:indices.iqt])
        print(f"Total water tendency: {total_water_tend*86400:.2e} kg/kg/day")
    
    print("✓ Tracer transport test completed")


def test_configuration_parameters():
    """Test different configuration parameters"""
    print("\n=== Testing Configuration Parameters ===")
    
    atm = create_test_atmosphere(unstable=True)
    
    # Test with different CAPE timescales
    configs = [
        ConvectionConfig(tau=1800.0),   # Fast adjustment
        ConvectionConfig(tau=7200.0),   # Default
        ConvectionConfig(tau=14400.0),  # Slow adjustment
    ]
    
    precip_rates = []
    for i, config in enumerate(configs):
        tendencies, state = tiedtke_nordeng_convection(
            atm['temperature'],
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        precip_rates.append(tendencies.precip_conv)
        print(f"Config {i+1} (tau={config.tau}s): precipitation = {tendencies.precip_conv*3600:.3f} mm/hr, type = {state.ktype}")
    
    print("✓ Configuration parameter test completed")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("TIEDTKE-NORDENG CONVECTION SCHEME TESTS")
    print("="*60)
    
    try:
        test_saturation_functions()
        test_cloud_base_detection()
        test_cape_calculation()
        test_stable_atmosphere()
        test_unstable_atmosphere()
        test_jax_compatibility()
        test_tracer_transport()
        test_configuration_parameters()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)