"""Planck function calculations for longwave radiation

This module computes Planck functions and related quantities
for thermal radiation calculations.

"""

import jax.numpy as jnp
import jax
from typing import Tuple
# from functools import partial  # Not needed anymore


# Physical constants
H_PLANCK = 6.62607015e-34    # Planck constant (J·s)
C_LIGHT = 2.99792458e8       # Speed of light (m/s)
K_BOLTZMANN = 1.380649e-23   # Boltzmann constant (J/K)
STEFAN_BOLTZMANN = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)

# The second radiation constant hc/k, in m·K. Folded here, in Python's
# float64, rather than left as ``(H_PLANCK * C_LIGHT) / (K_BOLTZMANN * T)``
# for the reason in ``planck_function_wavenumber``: the derivative of that
# form underflows in float32.
HC_OVER_K = (H_PLANCK * C_LIGHT) / K_BOLTZMANN  # 1.4387769e-2 m·K


@jax.jit
def planck_function_wavenumber(
    temperature: jnp.ndarray,
    wavenumber: float
) -> jnp.ndarray:
    """Calculate Planck function for given temperature and wavenumber.

    B(ν,T) = 2hc²ν³ / (exp(hcν/kT) - 1)

    Args:
        temperature: Temperature (K)
        wavenumber: Wavenumber (cm⁻¹)

    Returns:
        Planck radiance (W/m²/sr/cm⁻¹)

    """
    # Convert wavenumber from cm⁻¹ to m⁻¹
    nu = wavenumber * 100.0

    # Calculate hc/kT.
    #
    # Divide by the temperature itself, never by ``K_BOLTZMANN *
    # temperature``. The two are the same number, but the *derivative* of the
    # second form is not representable in float32: the quotient's VJP is
    # ``-numerator / denominator**2``, and ``(1.38e-23 * 250)**2 ~ 1.2e-41``
    # is below float32's smallest normal (1.18e-38), so it flushes toward
    # zero and the reported ``d(hc_kt)/dT`` overflows to ``+inf`` — at every
    # temperature and every wavenumber, not at some corner. The forward value
    # is perfectly finite, which is why this survived: it only shows up when
    # something differentiates the longwave, and then it takes the whole
    # longwave flux gradient with it.
    #
    # With ``hc/k`` folded into one O(1e-2) constant the denominator is the
    # temperature, whose square is O(1e5), and the derivative is ordinary.
    hc_kt = HC_OVER_K / temperature

    # Planck function
    # CRITICAL FIX: Was * 1e-2, should be * 100.0 (or * 1e2)
    # Formula gives W/(m² sr m⁻¹), multiply by 100 to get W/(m² sr cm⁻¹)
    # Previous bug: used 1e-2 instead of 1e2, giving values 10,000x too small!
    b_nu = 2.0 * H_PLANCK * C_LIGHT**2 * nu**3 / (jnp.exp(hc_kt * nu) - 1.0) * 100.0

    return b_nu


@jax.jit
def integrated_planck_function(
    temperature: jnp.ndarray,
    band_limits: Tuple[float, float]
) -> jnp.ndarray:
    """Calculate band-integrated Planck function.

    Integrates Planck function over a spectral band and converts to flux.

    Args:
        temperature: Temperature (K)
        band_limits: (lower, upper) wavenumber limits (cm⁻¹)

    Returns:
        Integrated Planck flux (W/m²)

    """
    # Use several points for integration
    n_points = 5
    wavenumbers = jnp.linspace(band_limits[0], band_limits[1], n_points)

    # Calculate Planck function at each wavenumber
    b_values = jax.vmap(lambda nu: planck_function_wavenumber(temperature, nu))(wavenumbers)

    # Trapezoidal integration (manual since JAX doesn't have trapz)
    delta_nu = (band_limits[1] - band_limits[0]) / (n_points - 1)
    # Trapezoidal rule: sum of (f[i] + f[i+1])/2 * dx
    # CRITICAL FIX: Must sum only over wavenumber axis (axis=0), not all dimensions
    # When temperature is multi-valued, b_values is (n_wavenumbers, n_temps)
    integrated_radiance = delta_nu * (0.5 * b_values[0] + jnp.sum(b_values[1:-1], axis=0) + 0.5 * b_values[-1])

    # Return integrated radiance (W/m²/sr)
    # The two-stream equations will handle the geometric factors
    # IMPORTANT: Despite the variable name, this is radiance not flux!
    return integrated_radiance


def planck_bands_lw(
    temperature: jnp.ndarray,
    band_limits: Tuple[Tuple[float, float], ...]
) -> jnp.ndarray:
    """Calculate LW planck bands"""
    from ..constants import N_LW_BANDS

    # Ensure temperature is at least 1D
    temp_array = jnp.atleast_1d(temperature)
    is_scalar = temperature.ndim == 0

    nlev = temp_array.shape[0]
    planck = jnp.zeros((nlev, N_LW_BANDS))

    # CRITICAL FIX: Removed @jax.jit decorator
    # The JIT was causing the Python for loop to not properly handle temperature dependence
    # Each call to integrated_planck_function was returning the same value!
    # Calculate for each band
    for band in range(min(len(band_limits), N_LW_BANDS)):
        b_band = integrated_planck_function(temp_array, band_limits[band])
        planck = planck.at[:, band].set(b_band)
    
    # Return scalar result if input was scalar
    if is_scalar:
        return planck[0, :]
    else:
        return planck


@jax.jit
def total_thermal_emission(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate total thermal emission using Stefan-Boltzmann law.
    
    E = σT⁴
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Total emission (W/m²)

    """
    return STEFAN_BOLTZMANN * temperature**4
