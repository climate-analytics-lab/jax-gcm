"""
Planck function calculations for longwave radiation

This module computes Planck functions and related quantities
for thermal radiation calculations.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple
from functools import partial


# Physical constants
H_PLANCK = 6.62607015e-34    # Planck constant (J·s)
C_LIGHT = 2.99792458e8       # Speed of light (m/s)
K_BOLTZMANN = 1.380649e-23   # Boltzmann constant (J/K)
STEFAN_BOLTZMANN = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)


@jax.jit
def planck_function_wavenumber(
    temperature: jnp.ndarray,
    wavenumber: float
) -> jnp.ndarray:
    """
    Calculate Planck function for given temperature and wavenumber.
    
    B(ν,T) = 2hc²ν³ / (exp(hcν/kT) - 1)
    
    Args:
        temperature: Temperature (K)
        wavenumber: Wavenumber (cm⁻¹)
        
    Returns:
        Planck radiance (W/m²/sr/cm⁻¹)
    """
    # Convert wavenumber from cm⁻¹ to m⁻¹
    nu = wavenumber * 100.0
    
    # Calculate hc/kT
    hc_kt = (H_PLANCK * C_LIGHT) / (K_BOLTZMANN * temperature)
    
    # Planck function
    # Factor of 1e-2 converts from W/m²/sr/m⁻¹ to W/m²/sr/cm⁻¹
    b_nu = 2.0 * H_PLANCK * C_LIGHT**2 * nu**3 / (jnp.exp(hc_kt * nu) - 1.0) * 1e-2
    
    return b_nu


@jax.jit
def integrated_planck_function(
    temperature: jnp.ndarray,
    band_limits: Tuple[float, float]
) -> jnp.ndarray:
    """
    Calculate band-integrated Planck function.
    
    Integrates Planck function over a spectral band.
    Uses simplified integration with representative wavenumbers.
    
    Args:
        temperature: Temperature (K)
        band_limits: (lower, upper) wavenumber limits (cm⁻¹)
        
    Returns:
        Integrated Planck radiance (W/m²/sr)
    """
    # Use several points for integration
    n_points = 5
    wavenumbers = jnp.linspace(band_limits[0], band_limits[1], n_points)
    
    # Calculate Planck function at each wavenumber
    b_values = jax.vmap(lambda nu: planck_function_wavenumber(temperature, nu))(wavenumbers)
    
    # Trapezoidal integration
    delta_nu = (band_limits[1] - band_limits[0]) / (n_points - 1)
    integrated = jnp.trapz(b_values, dx=delta_nu, axis=0)
    
    return integrated


@partial(jax.jit, static_argnames=['n_bands'])
def planck_bands(
    temperature: jnp.ndarray,
    band_limits: Tuple[Tuple[float, float], ...],
    n_bands: int = 3
) -> jnp.ndarray:
    """
    Calculate Planck function for multiple spectral bands.
    
    Args:
        temperature: Temperature (K) [nlev]
        band_limits: Tuple of band limit tuples
        n_bands: Number of bands
        
    Returns:
        Band-integrated Planck function (W/m²/sr) [nlev, n_bands]
    """
    nlev = temperature.shape[0]
    planck = jnp.zeros((nlev, n_bands))
    
    for band in range(n_bands):
        b_band = integrated_planck_function(temperature, band_limits[band])
        planck = planck.at[:, band].set(b_band)
    
    return planck


@jax.jit
def planck_derivative(
    temperature: jnp.ndarray,
    wavenumber: float
) -> jnp.ndarray:
    """
    Calculate derivative of Planck function with respect to temperature.
    
    dB/dT = B(T) * (hcν/kT²) * exp(hcν/kT) / (exp(hcν/kT) - 1)
    
    Args:
        temperature: Temperature (K)
        wavenumber: Wavenumber (cm⁻¹)
        
    Returns:
        dB/dT (W/m²/sr/cm⁻¹/K)
    """
    # Get Planck function
    b = planck_function_wavenumber(temperature, wavenumber)
    
    # Convert wavenumber
    nu = wavenumber * 100.0
    
    # Calculate exponential term
    hc_kt = (H_PLANCK * C_LIGHT * nu) / (K_BOLTZMANN * temperature)
    exp_term = jnp.exp(hc_kt)
    
    # Derivative
    db_dt = b * hc_kt * exp_term / (temperature * (exp_term - 1.0))
    
    return db_dt


@jax.jit
def total_thermal_emission(temperature: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate total thermal emission using Stefan-Boltzmann law.
    
    E = σT⁴
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Total emission (W/m²)
    """
    return STEFAN_BOLTZMANN * temperature**4


@jax.jit
def effective_temperature(flux: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate effective temperature from thermal flux.
    
    T = (F/σ)^(1/4)
    
    Args:
        flux: Thermal flux (W/m²)
        
    Returns:
        Effective temperature (K)
    """
    return (flux / STEFAN_BOLTZMANN) ** 0.25


@partial(jax.jit, static_argnames=['n_bands'])
def band_fraction(
    temperature: jnp.ndarray,
    band_limits: Tuple[Tuple[float, float], ...],
    n_bands: int = 3
) -> jnp.ndarray:
    """
    Calculate fraction of total thermal emission in each band.
    
    Args:
        temperature: Temperature (K)
        band_limits: Tuple of band limit tuples  
        n_bands: Number of bands
        
    Returns:
        Fraction of total emission in each band [n_bands]
    """
    # Total emission
    total = total_thermal_emission(temperature) / jnp.pi  # Convert to radiance
    
    # Band emissions
    fractions = jnp.zeros(n_bands)
    
    for band in range(n_bands):
        b_band = integrated_planck_function(temperature, band_limits[band])
        fractions = fractions.at[band].set(b_band / total)
    
    return fractions


# Utility functions for layer calculations
@jax.jit
def layer_planck_function(
    t_level_below: jnp.ndarray,
    t_level_above: jnp.ndarray,
    band_limits: Tuple[float, float]
) -> jnp.ndarray:
    """
    Calculate effective Planck function for a layer.
    
    Uses linear-in-tau approximation.
    
    Args:
        t_level_below: Temperature at lower level (K)
        t_level_above: Temperature at upper level (K)
        band_limits: Spectral band limits
        
    Returns:
        Layer-averaged Planck function (W/m²/sr)
    """
    # Simple average (can be improved with linear-in-tau)
    t_layer = 0.5 * (t_level_below + t_level_above)
    return integrated_planck_function(t_layer, band_limits)


@jax.jit
def interface_planck_function(
    t_below: jnp.ndarray,
    t_above: jnp.ndarray,
    band_limits: Tuple[float, float]
) -> jnp.ndarray:
    """
    Calculate Planck function at interface.
    
    Linear interpolation in temperature.
    
    Args:
        t_below: Temperature below interface (K)
        t_above: Temperature above interface (K)
        band_limits: Spectral band limits
        
    Returns:
        Interface Planck function (W/m²/sr)
    """
    t_interface = 0.5 * (t_below + t_above)
    return integrated_planck_function(t_interface, band_limits)


# Test functions
def test_planck_functions():
    """Test Planck function calculations"""
    
    # Test single Planck function
    T = 300.0
    nu = 1000.0  # cm⁻¹
    
    B = planck_function_wavenumber(T, nu)
    assert B > 0
    assert jnp.isfinite(B)
    
    # Test Stefan-Boltzmann
    emission = total_thermal_emission(T)
    expected = STEFAN_BOLTZMANN * T**4
    assert jnp.abs(emission - expected) < 1e-6
    
    # Test band integration
    band = (500.0, 1500.0)  # cm⁻¹
    B_band = integrated_planck_function(T, band)
    assert B_band > 0
    assert jnp.isfinite(B_band)
    
    # Test multiple bands
    bands = ((10, 350), (350, 500), (500, 2500))
    B_bands = planck_bands(jnp.array([250.0, 300.0]), bands, 3)
    assert B_bands.shape == (2, 3)
    assert jnp.all(B_bands > 0)
    
    # Test that warmer temperature gives more emission
    assert B_bands[1, :].sum() > B_bands[0, :].sum()
    
    print("Planck function tests passed!")


if __name__ == "__main__":
    test_planck_functions()