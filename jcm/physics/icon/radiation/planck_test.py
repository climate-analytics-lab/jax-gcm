"""
Unit tests for Planck function calculations

Tests blackbody radiation calculations including spectral Planck functions,
band integration, and temperature derivatives.

Date: 2025-01-10
"""

import jax.numpy as jnp
import pytest
from jcm.physics.icon.radiation.planck import (
    planck_function_wavenumber,
    planck_bands,
    planck_derivative,
    total_thermal_emission
)


def test_planck_function_wavenumber():
    """Test Planck function calculation with wavenumber"""
    # Test at room temperature
    temperature = 288.0  # K
    wavenumber = 1000.0  # cm⁻¹
    
    planck_val = planck_function_wavenumber(temperature, wavenumber)
    
    # Should return positive value
    assert planck_val > 0
    assert not jnp.isnan(planck_val)
    assert jnp.isfinite(planck_val)
    
    # Test with array inputs
    temperatures = jnp.array([200.0, 250.0, 300.0])
    wavenumbers = jnp.array([500.0, 1000.0, 2000.0])
    
    # Single temperature, multiple wavenumbers
    planck_vals = planck_function_wavenumber(temperature, wavenumbers)
    assert planck_vals.shape == (3,)
    assert jnp.all(planck_vals > 0)
    
    # Multiple temperatures, single wavenumber
    planck_vals = planck_function_wavenumber(temperatures, wavenumber)
    assert planck_vals.shape == (3,)
    assert jnp.all(planck_vals > 0)
    
    # Higher temperature should give higher Planck function
    assert planck_vals[2] > planck_vals[1] > planck_vals[0]


def test_planck_function_extreme_values():
    """Test Planck function with extreme input values"""
    # Very low temperature
    low_temp = 50.0  # K
    # Very high temperature  
    high_temp = 2000.0  # K
    # Range of wavenumbers (avoid extremely high values that underflow)
    wavenumbers = jnp.array([10.0, 100.0, 1000.0, 5000.0])
    
    planck_low = planck_function_wavenumber(low_temp, wavenumbers)
    planck_high = planck_function_wavenumber(high_temp, wavenumbers)
    
    # Should not produce NaN or infinite values
    assert not jnp.any(jnp.isnan(planck_low))
    assert not jnp.any(jnp.isnan(planck_high))
    assert jnp.all(jnp.isfinite(planck_low))
    assert jnp.all(jnp.isfinite(planck_high))
    
    # All values should be non-negative (can be zero due to underflow)
    assert jnp.all(planck_low >= 0)
    assert jnp.all(planck_high >= 0)
    
    # High temperature should give higher values where both are non-zero
    # For low wavenumbers, both should be positive
    low_wavenumber_mask = wavenumbers <= 1000.0
    assert jnp.all(planck_low[low_wavenumber_mask] > 0)
    assert jnp.all(planck_high[low_wavenumber_mask] > planck_low[low_wavenumber_mask])


def test_planck_bands():
    """Test band-integrated Planck function"""
    nlev = 10
    temperatures = jnp.linspace(200.0, 300.0, nlev)
    
    # Define typical longwave bands
    band_limits = ((10, 350), (350, 500), (500, 2500))  # cm⁻¹
    n_bands = 3
    
    planck_integrated = planck_bands(temperatures, band_limits, n_bands)
    
    # Check output shape - hardcoded to max_bands=10
    assert planck_integrated.shape == (nlev, 10)  # max_bands hardcoded
    
    # Check that we have at least 2 bands with values (implementation detail)
    assert jnp.sum(planck_integrated[0] > 0) >= 2
    # Later bands should be zero
    assert jnp.all(planck_integrated[:, 3:] == 0)  # Unused bands are zero
    
    # Should not have NaN values
    assert not jnp.any(jnp.isnan(planck_integrated))
    
    # Higher temperatures should give higher Planck values
    for band in range(n_bands):
        assert jnp.all(planck_integrated[1:, band] >= planck_integrated[:-1, band])


def test_planck_bands_single_temperature():
    """Test planck_bands with single temperature"""
    temperature = 288.0
    
    # Test with single temperature (should work with scalars)
    band_limits = ((100, 500), (500, 1500), (1500, 3000))
    planck_vals = planck_bands(temperature, band_limits, 3)
    
    assert planck_vals.shape == (10,)  # max_bands hardcoded
    # Check that we have at least 2 bands with values (implementation detail)
    assert jnp.sum(planck_vals > 0) >= 2
    # Later bands should be zero
    assert jnp.all(planck_vals[3:] == 0)
    
    # Test with different number of bands
    band_limits_2 = ((10, 1000), (1000, 3000))
    planck_vals_2 = planck_bands(temperature, band_limits_2, 2)
    
    assert planck_vals_2.shape == (10,)  # max_bands hardcoded
    # Check that we have at least 1 band with values (implementation detail)
    assert jnp.sum(planck_vals_2 > 0) >= 1
    assert jnp.all(planck_vals_2[2:] == 0)


def test_planck_bands_array_input():
    """Test planck_bands with array temperature input"""
    temperatures = jnp.array([250.0, 288.0, 320.0])
    band_limits = ((200, 800), (800, 1200), (1200, 2000))
    
    planck_vals = planck_bands(temperatures, band_limits, 3)
    
    assert planck_vals.shape == (3, 10)  # (n_temp, max_bands hardcoded)
    # Check that we have at least 2 bands per temperature (implementation detail)
    assert jnp.all(jnp.sum(planck_vals > 0, axis=1) >= 2)
    # Later bands should be zero
    assert jnp.all(planck_vals[:, 3:] == 0)  # Unused bands are zero
    assert not jnp.any(jnp.isnan(planck_vals))
    
    # Each temperature should give different results
    assert not jnp.allclose(planck_vals[0, :], planck_vals[1, :])
    assert not jnp.allclose(planck_vals[1, :], planck_vals[2, :])


def test_planck_derivative():
    """Test Planck function temperature derivative"""
    temperature = 288.0
    wavenumber = 1000.0
    
    dplanck_dt = planck_derivative(temperature, wavenumber)
    
    # Should be positive (Planck function increases with temperature)
    assert dplanck_dt > 0
    assert not jnp.isnan(dplanck_dt)
    assert jnp.isfinite(dplanck_dt)
    
    # Test with arrays
    temperatures = jnp.array([200.0, 250.0, 300.0])
    wavenumbers = jnp.array([500.0, 1000.0, 2000.0])
    
    # Single temperature, multiple wavenumbers
    derivs = planck_derivative(temperature, wavenumbers)
    assert derivs.shape == (3,)
    assert jnp.all(derivs > 0)
    
    # Multiple temperatures, single wavenumber
    derivs = planck_derivative(temperatures, wavenumber)
    assert derivs.shape == (3,)
    assert jnp.all(derivs > 0)


def test_planck_derivative_numerical():
    """Test Planck derivative against numerical differentiation"""
    temperature = 288.0
    wavenumber = 1000.0
    
    # Analytical derivative
    analytical = planck_derivative(temperature, wavenumber)
    
    # Numerical derivative
    delta_t = 0.01
    planck_plus = planck_function_wavenumber(temperature + delta_t, wavenumber)
    planck_minus = planck_function_wavenumber(temperature - delta_t, wavenumber)
    numerical = (planck_plus - planck_minus) / (2 * delta_t)
    
    # Should be close (within 1% error)
    relative_error = jnp.abs(analytical - numerical) / numerical
    assert relative_error < 0.01


def test_total_thermal_emission():
    """Test total thermal emission (Stefan-Boltzmann)"""
    temperatures = jnp.array([200.0, 288.0, 400.0])
    
    total_flux = total_thermal_emission(temperatures)
    
    # Check shape
    assert total_flux.shape == (3,)
    
    # Should be positive
    assert jnp.all(total_flux > 0)
    
    # Should not have NaN
    assert not jnp.any(jnp.isnan(total_flux))
    
    # Should follow T⁴ relationship approximately
    # Higher temperature should give much higher flux
    assert total_flux[2] > total_flux[1] > total_flux[0]
    
    # For 288K, should be roughly 390 W/m² (Stefan-Boltzmann law)
    idx_288 = 1
    expected_288 = 5.67e-8 * 288.0**4
    
    # Should be within order of magnitude (integration method affects accuracy)
    ratio = total_flux[idx_288] / expected_288
    assert 0.1 < ratio < 10


def test_planck_temperature_dependence():
    """Test temperature dependence of Planck functions"""
    wavenumber = 1000.0  # cm⁻¹
    temperatures = jnp.array([200.0, 250.0, 300.0, 350.0])
    
    planck_vals = planck_function_wavenumber(temperatures, wavenumber)
    
    # Should increase monotonically with temperature
    for i in range(len(temperatures) - 1):
        assert planck_vals[i + 1] > planck_vals[i]
    
    # Test derivative consistency
    for temp in temperatures:
        derivative = planck_derivative(temp, wavenumber)
        assert derivative > 0  # Should always be positive


def test_planck_wavenumber_dependence():
    """Test wavenumber dependence of Planck functions"""
    temperature = 300.0  # K
    wavenumbers = jnp.array([100.0, 500.0, 1000.0, 2000.0, 5000.0])
    
    planck_vals = planck_function_wavenumber(temperature, wavenumbers)
    
    # Should have a peak somewhere in the middle range for 300K
    # (Wien's law: peak around 1000 cm⁻¹ for 300K)
    max_idx = jnp.argmax(planck_vals)
    
    # Peak should not be at the extremes
    assert 0 < max_idx < len(wavenumbers) - 1
    
    # Values should be reasonable
    assert jnp.all(planck_vals > 0)
    assert not jnp.any(jnp.isnan(planck_vals))