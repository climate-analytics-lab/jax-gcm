#!/usr/bin/env python3
"""
Test script to validate the enhanced radiation scheme with improved gas optics.

This script tests:
1. Enhanced gas optics with temperature dependence
2. Expanded spectral resolution (6 SW + 8 LW bands)
3. Realistic atmospheric profiles
4. Integration with the full radiation scheme
"""

import jax.numpy as jnp
import jax
from jcm.physics.icon.radiation.gas_optics import (
    water_vapor_continuum,
    co2_absorption,
    ozone_absorption_sw,
    ozone_absorption_lw,
    gas_optical_depth_lw,
    gas_optical_depth_sw
)
from jcm.physics.icon.radiation.constants import N_SW_BANDS, N_LW_BANDS
from jcm.physics.icon.radiation.radiation_scheme import radiation_scheme
from jcm.physics.icon.radiation.radiation_types import RadiationParameters

def test_enhanced_gas_optics():
    """Test enhanced gas optics with expanded spectral resolution"""
    print("Testing enhanced gas optics...")
    
    # Create realistic atmospheric profile
    nlev = 30
    pressure = jnp.logspace(jnp.log10(100000), jnp.log10(100), nlev)  # Pa
    temperature = 288.0 - 6.5e-3 * jnp.logspace(jnp.log10(1), jnp.log10(20000), nlev)  # K
    h2o_vmr = 0.015 * jnp.exp(-jnp.linspace(0, 12, nlev))  # Exponential decrease
    o3_vmr = jnp.where(
        pressure < 10000,  # Stratosphere
        8e-6 * jnp.exp(-((jnp.log(pressure) - jnp.log(5000)) ** 2) / 8),  # Peak around 5000 Pa
        1e-6  # Troposphere
    )
    
    # Layer properties
    air_density = pressure / (287.0 * temperature)
    height = -287.0 * temperature * jnp.log(pressure / 100000.0) / 9.81
    layer_thickness = jnp.concatenate([
        jnp.abs(jnp.diff(height)),
        jnp.array([jnp.abs(height[-1] - height[-2])])
    ])
    
    # Test longwave gas optical depths
    print(f"Testing LW gas optics with {N_LW_BANDS} bands...")
    tau_lw = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, 420e-6,
        layer_thickness, air_density
    )
    
    assert tau_lw.shape == (nlev, N_LW_BANDS)
    assert jnp.all(tau_lw >= 0)
    assert jnp.all(jnp.isfinite(tau_lw))
    
    # Test shortwave gas optical depths
    print(f"Testing SW gas optics with {N_SW_BANDS} bands...")
    tau_sw = gas_optical_depth_sw(
        pressure, temperature, h2o_vmr, o3_vmr,
        layer_thickness, air_density, 0.6
    )
    
    assert tau_sw.shape == (nlev, N_SW_BANDS)
    assert jnp.all(tau_sw >= 0)
    assert jnp.all(jnp.isfinite(tau_sw))
    
    # Validate spectral distribution
    print("Validating spectral distribution...")
    
    # H2O should dominate in specific LW bands
    h2o_bands = [1, 4, 7]  # H2O rotation, continuum, and main bands
    for band in h2o_bands:
        assert jnp.sum(tau_lw[:, band]) > jnp.sum(tau_lw[:, 0])  # More than window
    
    # CO2 should be strongest in bands 2 and 3
    co2_bands = [2, 3]
    for band in co2_bands:
        assert jnp.sum(tau_lw[:, band]) > jnp.sum(tau_lw[:, 0])  # More than window
    
    # O3 should be strongest in UV bands (0, 1)
    uv_bands = [0, 1]
    for band in uv_bands:
        assert jnp.sum(tau_sw[:, band]) > jnp.sum(tau_sw[:, 4])  # More than NIR
    
    print("✓ Enhanced gas optics tests passed!")
    
    # Display realistic absorption values
    print("\nRealistic optical depth values:")
    print(f"  LW total: {jnp.sum(tau_lw):.2f}")
    print(f"  SW total: {jnp.sum(tau_sw):.2f}")
    print(f"  LW per band: {jnp.sum(tau_lw, axis=0)}")
    print(f"  SW per band: {jnp.sum(tau_sw, axis=0)}")
    
    return tau_lw, tau_sw

def test_temperature_dependence():
    """Test temperature dependence of enhanced gas optics"""
    print("\nTesting temperature dependence...")
    
    nlev = 10
    pressure = jnp.ones(nlev) * 50000.0  # Pa
    h2o_vmr = jnp.ones(nlev) * 0.005
    o3_vmr = jnp.ones(nlev) * 2e-6
    
    # Test at different temperatures
    T_cold = jnp.ones(nlev) * 220.0  # K
    T_warm = jnp.ones(nlev) * 300.0  # K
    
    # CO2 absorption should change with temperature
    co2_cold = co2_absorption(T_cold, pressure, 400e-6, 2)
    co2_warm = co2_absorption(T_warm, pressure, 400e-6, 2)
    
    # Should have different values
    assert not jnp.allclose(co2_cold, co2_warm)
    
    # H2O continuum should have strong temperature dependence
    h2o_cold = water_vapor_continuum(T_cold, pressure, h2o_vmr, 4)
    h2o_warm = water_vapor_continuum(T_warm, pressure, h2o_vmr, 4)
    
    # Should have different values (direction depends on exponential factor)
    assert not jnp.allclose(h2o_cold, h2o_warm, rtol=1e-3)
    
    # O3 should have temperature dependence
    o3_cold = ozone_absorption_sw(o3_vmr, T_cold, 0)
    o3_warm = ozone_absorption_sw(o3_vmr, T_warm, 0)
    
    assert not jnp.allclose(o3_cold, o3_warm)
    
    print("✓ Temperature dependence tests passed!")

def test_full_radiation_scheme():
    """Test full radiation scheme with enhanced gas optics"""
    print("\nTesting full radiation scheme integration...")
    
    # Create test atmosphere
    nlev = 25
    temperature = jnp.linspace(288.0, 200.0, nlev)
    specific_humidity = jnp.linspace(0.01, 0.001, nlev)
    surface_pressure = 101325.0  # Pa
    geopotential = jnp.linspace(0, 200000, nlev)  # m²/s²
    
    # Simple cloud profile
    cloud_water = jnp.zeros(nlev)
    cloud_ice = jnp.zeros(nlev)
    cloud_fraction = jnp.zeros(nlev)
    
    # Time and location
    day_of_year = 180.0  # Summer solstice
    seconds_since_midnight = 43200.0  # Noon
    latitude = 45.0  # Mid-latitudes
    longitude = 0.0
    
    # Enhanced parameters with more bands
    params = RadiationParameters.default(
        n_sw_bands=N_SW_BANDS,
        n_lw_bands=N_LW_BANDS,
        co2_vmr=420e-6  # Current CO2 level
    )
    
    # Run radiation scheme
    tendencies, diagnostics = radiation_scheme(
        temperature=temperature,
        specific_humidity=specific_humidity,
        surface_pressure=surface_pressure,
        geopotential=geopotential,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        day_of_year=day_of_year,
        seconds_since_midnight=seconds_since_midnight,
        latitude=latitude,
        longitude=longitude,
        parameters=params
    )
    
    # Validate outputs
    assert tendencies.temperature_tendency.shape == (nlev,)
    assert jnp.all(jnp.isfinite(tendencies.temperature_tendency))
    
    # Check that heating rates are reasonable
    lw_heating = tendencies.longwave_heating
    sw_heating = tendencies.shortwave_heating
    
    assert jnp.all(jnp.isfinite(lw_heating))
    assert jnp.all(jnp.isfinite(sw_heating))
    
    # LW and SW should have finite heating rates
    # (sign depends on specific atmospheric profile and solar conditions)
    assert jnp.all(jnp.abs(lw_heating) < 1e-2)  # Reasonable magnitude
    assert jnp.all(jnp.abs(sw_heating) < 1e-2)  # Reasonable magnitude
    
    # Check diagnostic fluxes
    assert jnp.all(jnp.isfinite(diagnostics.sw_flux_up))
    assert jnp.all(jnp.isfinite(diagnostics.sw_flux_down))
    assert jnp.all(jnp.isfinite(diagnostics.lw_flux_up))
    assert jnp.all(jnp.isfinite(diagnostics.lw_flux_down))
    
    print("✓ Full radiation scheme integration tests passed!")
    
    # Display results
    print(f"\nRadiation scheme results:")
    print(f"  Total heating rate: {jnp.mean(tendencies.temperature_tendency):.2e} K/s")
    print(f"  LW cooling rate: {jnp.mean(lw_heating):.2e} K/s")
    print(f"  SW warming rate: {jnp.mean(sw_heating):.2e} K/s")
    print(f"  TOA SW down: {diagnostics.toa_sw_down:.1f} W/m²")
    print(f"  TOA SW up: {diagnostics.toa_sw_up:.1f} W/m²")
    print(f"  TOA LW up: {diagnostics.toa_lw_up:.1f} W/m²")
    print(f"  Surface SW down: {diagnostics.surface_sw_down:.1f} W/m²")
    print(f"  Surface LW down: {diagnostics.surface_lw_down:.1f} W/m²")
    
    return tendencies, diagnostics

if __name__ == "__main__":
    print("=== Testing Enhanced ICON Radiation Scheme ===")
    print(f"Spectral resolution: {N_SW_BANDS} SW bands, {N_LW_BANDS} LW bands")
    print()
    
    # Run all tests
    try:
        tau_lw, tau_sw = test_enhanced_gas_optics()
        test_temperature_dependence()
        tendencies, diagnostics = test_full_radiation_scheme()
        
        print("\n=== All Enhanced Radiation Tests Passed! ===")
        print("✓ Enhanced gas optics with temperature dependence")
        print("✓ Expanded spectral resolution (6 SW + 8 LW bands)")
        print("✓ Realistic atmospheric profiles")
        print("✓ Full radiation scheme integration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise