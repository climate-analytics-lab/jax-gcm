"""
Tests for WMO tropopause diagnostic

Date: 2025-01-09
"""

import jax.numpy as jnp
import pytest
from jcm.physics.icon.diagnostics.wmo_tropopause import (
    wmo_tropopause, 
    compute_geopotential_height,
    compute_lapse_rate,
    find_tropopause_level,
    GWMO,
    P_DEFAULT
)

def create_test_atmosphere():
    """Create a realistic test atmosphere profile"""
    # Create a typical atmospheric profile
    nlev = 40
    
    # Pressure levels from surface to ~10 hPa
    pressure_levels = jnp.logspace(jnp.log10(100000), jnp.log10(1000), nlev)
    
    # Temperature profile with tropospheric and stratospheric regions
    # Troposphere: decreasing with height
    # Stratosphere: increasing with height
    temperature = jnp.zeros(nlev)
    
    # Surface temperature
    T_surface = 288.0  # K
    
    # Tropospheric lapse rate (6.5 K/km)
    lapse_trop = 0.0065  # K/m
    
    # Tropopause at ~200 hPa
    p_tropopause = 20000.0  # Pa
    T_tropopause = 220.0  # K
    
    # Stratospheric warming rate
    lapse_strat = -0.001  # K/m (warming with height)
    
    # Build temperature profile
    for k in range(nlev):
        p = pressure_levels[k]
        if p > p_tropopause:
            # Troposphere
            # Use simple relationship: T = T_surface - lapse * height
            # Approximate height from pressure using scale height
            height = -7000 * jnp.log(p / 100000)  # Simple approximation
            temperature = temperature.at[k].set(T_surface - lapse_trop * height)
        else:
            # Stratosphere
            height = -7000 * jnp.log(p / 100000)
            height_trop = -7000 * jnp.log(p_tropopause / 100000)
            temperature = temperature.at[k].set(T_tropopause + lapse_strat * (height - height_trop))
    
    # Ensure monotonic decreasing with height in troposphere
    temperature = jnp.maximum(temperature, T_tropopause)
    
    surface_pressure = jnp.array([100000.0])  # Pa
    
    return temperature, pressure_levels, surface_pressure

def test_compute_geopotential_height():
    """Test geopotential height computation"""
    # Simple test case
    nlev = 5
    temperature = jnp.array([288.0, 285.0, 280.0, 275.0, 270.0])
    pressure = jnp.array([100000.0, 85000.0, 70000.0, 50000.0, 30000.0])
    surface_pressure = jnp.array([100000.0])
    
    height = compute_geopotential_height(pressure, temperature, surface_pressure)
    
    # Check that height increases with decreasing pressure
    assert jnp.all(height[1:] > height[:-1])
    
    # Check that surface height is zero
    assert jnp.abs(height[0]) < 1e-10
    
    # Check reasonable magnitudes (should be in km range)
    assert height[-1] > 5000  # Top level should be > 5km
    assert height[-1] < 50000  # But not unreasonably high

def test_compute_lapse_rate():
    """Test lapse rate computation"""
    # Create a profile with known lapse rate
    nlev = 5
    height = jnp.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    
    # Constant lapse rate of -6.5 K/km
    temperature = jnp.array([288.0, 281.5, 275.0, 268.5, 262.0])
    
    lapse_rate = compute_lapse_rate(temperature, height)
    
    # Should have nlev-1 values
    assert lapse_rate.shape == (nlev - 1,)
    
    # Should be approximately -6.5e-3 K/m
    expected_lapse = -6.5e-3
    assert jnp.allclose(lapse_rate, expected_lapse, atol=1e-6)

def test_find_tropopause_level():
    """Test tropopause level finding"""
    # Create test atmosphere
    temperature, pressure, surface_pressure = create_test_atmosphere()
    
    # Add batch dimension
    temperature = temperature[None, :]
    pressure = pressure[None, :]
    
    # Compute height
    height = compute_geopotential_height(pressure, temperature, surface_pressure)
    
    # Find tropopause with appropriate search range for 40-level atmosphere
    # Search from level 5 to 35 to avoid surface and very high levels
    tropopause_pressure = find_tropopause_level(temperature, pressure, height, 
                                               ncctop=5, nccbot=35)
    
    # Should find a reasonable tropopause pressure
    assert tropopause_pressure.shape == (1,)
    assert tropopause_pressure[0] > 10000  # > 100 hPa
    assert tropopause_pressure[0] < 40000  # < 400 hPa

def test_wmo_tropopause():
    """Test complete WMO tropopause function"""
    # Create test atmosphere
    temperature, pressure, surface_pressure = create_test_atmosphere()
    
    # Add batch dimensions to test vectorization
    batch_shape = (2, 3)
    temperature = jnp.broadcast_to(temperature, batch_shape + temperature.shape)
    pressure = jnp.broadcast_to(pressure, batch_shape + pressure.shape)
    surface_pressure = jnp.broadcast_to(surface_pressure, batch_shape + surface_pressure.shape)
    
    # Compute tropopause
    tropopause_pressure = wmo_tropopause(temperature, pressure, surface_pressure)
    
    # Check output shape
    assert tropopause_pressure.shape == batch_shape
    
    # Check reasonable values
    assert jnp.all(tropopause_pressure > 10000)  # > 100 hPa
    assert jnp.all(tropopause_pressure < 40000)  # < 400 hPa

def test_wmo_tropopause_with_previous():
    """Test WMO tropopause with previous values"""
    # Create test atmosphere
    temperature, pressure, surface_pressure = create_test_atmosphere()
    
    # Create a case where no tropopause is found (isothermal atmosphere)
    temperature = jnp.full_like(temperature, 250.0)
    
    # Previous tropopause value
    previous_tropopause = jnp.array([25000.0])
    
    # Compute tropopause
    tropopause_pressure = wmo_tropopause(
        temperature[None, :], 
        pressure[None, :], 
        surface_pressure,
        previous_tropopause
    )
    
    # Should use previous value when no tropopause found
    # (isothermal atmosphere doesn't meet WMO criteria)
    assert jnp.allclose(tropopause_pressure, previous_tropopause)

def test_wmo_tropopause_fallback():
    """Test fallback to default value"""
    # Create isothermal atmosphere (no tropopause)
    nlev = 20
    temperature = jnp.full((1, nlev), 250.0)
    pressure = jnp.logspace(jnp.log10(100000), jnp.log10(1000), nlev)[None, :]
    surface_pressure = jnp.array([100000.0])
    
    # Compute tropopause
    tropopause_pressure = wmo_tropopause(temperature, pressure, surface_pressure)
    
    # Should return default value
    assert jnp.allclose(tropopause_pressure, P_DEFAULT)

def test_wmo_constants():
    """Test that constants are set correctly"""
    assert GWMO == -0.002  # -2 K/km
    assert P_DEFAULT == 20000.0  # 200 hPa

if __name__ == "__main__":
    test_compute_geopotential_height()
    test_compute_lapse_rate()
    test_find_tropopause_level()
    test_wmo_tropopause()
    test_wmo_tropopause_with_previous()
    test_wmo_tropopause_fallback()
    test_wmo_constants()
    print("All WMO tropopause tests passed!")