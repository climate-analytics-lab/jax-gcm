"""Tests for WMO tropopause diagnostic

Date: 2025-01-09
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jcm.physics.diagnostics.wmo_tropopause import (
    wmo_tropopause, 
    compute_geopotential_height,
    compute_lapse_rate,
    find_tropopause_level,
    GWMO,
    P_DEFAULT
)
from jcm.testing import check_gradients

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

def _sounding(nlev=47, p_top=1000.0, ncols=None):
    """Build a mid-latitude sounding on a surface-first pressure column.

    Surface-first — index 0 is the surface — because that is the ordering
    ``compute_geopotential_height`` is written for: it prepends the surface
    pressure to the level axis and integrates the hypsometric equation
    upward with ``cumsum``, and ``find_tropopause_level``'s scan then walks
    from the bottom of its search window, which is what makes "the lowest
    level satisfying the criterion" the WMO definition it claims to be.

    The temperature is a 6.5 K/km troposphere up to 11.5 km and a 1.8 K/km
    warming stratosphere above, so the 2 K/km threshold is crossed once and
    cleanly. The module's own ``create_test_atmosphere`` cannot serve here:
    its ``jnp.maximum(temperature, T_tropopause)`` clamp leaves the whole
    stratosphere isothermal at 220 K, which fails the finder's "at least
    1 K of variation over 2 km" criterion at every level, so it returns
    ``P_DEFAULT`` — a constant, whose gradient is trivially zero and tests
    nothing.
    """
    pressure = np.logspace(np.log10(101325.0), np.log10(p_top), nlev)
    height = -7500.0 * np.log(pressure / 101325.0)
    temperature = np.where(height <= 11500.0,
                           288.0 - 0.0065 * height,
                           213.25 + 0.0018 * (height - 11500.0))
    temperature = jnp.asarray(temperature, jnp.float32)
    pressure = jnp.asarray(pressure, jnp.float32)
    surface_pressure = jnp.asarray(101325.0, jnp.float32)
    if ncols is not None:
        temperature = jnp.stack([temperature + 2.0 * k
                                 for k in range(ncols)])
        pressure = jnp.broadcast_to(pressure, (ncols, nlev))
        surface_pressure = jnp.full((ncols,), 101325.0)
    return temperature, pressure, surface_pressure


class TestWmoTropopauseGradients:
    """Gradients of the WMO tropopause diagnostic (#820).

    The diagnostic returns *a pressure level selected by a scan*, so its
    output is piecewise constant in everything except the pressure it picks
    out. That is what ``reference="adjoint"`` in ``jcm.testing`` exists for:
    a central difference across the staircase measures the step, not the
    derivative. The continuous half — the hypsometric height integral and
    the lapse rate built on it — is compared against a difference
    separately. Both are green.

    The scan's temperature dependence is entirely inside comparisons
    (``lapse >= GWMO``, the 2 km window mask, ``height > 5000``, the 1 K
    variation test), so ``d(p_tropopause)/dT`` is identically zero. That is
    a property of the definition, not a defect to guard away: a
    differentiable tropopause needs a smooth diagnostic (a soft selection
    over the criterion), which is a different quantity. The zero is pinned
    below so such a change cannot land unnoticed.
    """

    @pytest.mark.parametrize("seed", [0, 4])
    @pytest.mark.parametrize("ncols", [None, 3], ids=["column", "batch"])
    def test_height_and_lapse_rate_match_a_central_difference(self, ncols,
                                                              seed):
        """The continuous half, composed as ``wmo_tropopause`` composes it.

        The lapse rate is checked on the height the integral produces rather
        than on an independently-perturbed height array. With the two free
        of each other, the projection of a difference operator over a
        20-level column is dominated by cancellation between adjacent
        layers and a float32 secant of it is noise at every rung — the
        ladder reports no usable step, which is a statement about that
        contrived direction rather than about the derivative. Composed, it
        is the quantity the diagnostic actually differentiates.
        """
        temperature, pressure, surface_pressure = _sounding(
            nlev=20, p_top=5000.0, ncols=ncols)
        check_gradients(compute_geopotential_height,
                        (pressure, temperature, surface_pressure),
                        rtol=1e-3, seed=seed)
        check_gradients(
            lambda t, p, ps: compute_lapse_rate(
                t, compute_geopotential_height(p, t, ps)),
            (temperature, pressure, surface_pressure), rtol=1e-3, seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    @pytest.mark.parametrize("ncols", [None, 3], ids=["column", "batch"])
    def test_tropopause_is_adjoint_and_pressure_is_live(self, ncols, seed):
        """Both AD modes agree, and the selected pressure carries gradient.

        ``live_inputs`` names the pressure because that is the only input
        the scan's output is a function of: the answer is an element of it.
        Temperature and surface pressure reach the result only through
        comparisons.
        """
        temperature, pressure, surface_pressure = _sounding(ncols=ncols)
        # A found tropopause, not the not-found default — otherwise the
        # output would be a constant and every gradient trivially zero.
        result = wmo_tropopause(temperature, pressure, surface_pressure)
        assert jnp.all(result != P_DEFAULT)
        assert jnp.all(result > 5000.0) and jnp.all(result < 40000.0)
        check_gradients(wmo_tropopause,
                        (temperature, pressure, surface_pressure),
                        reference="adjoint", live_inputs=["[1]"], seed=seed)

    def test_previous_value_fallback_is_adjoint_and_live(self):
        """The ``previous_tropopause`` branch, on an isothermal column.

        Nothing satisfies the criteria there, so the result is the previous
        value and the gradient has to flow through that input instead.
        """
        temperature, pressure, surface_pressure = _sounding(ncols=3)
        isothermal = jnp.full_like(temperature, 250.0)
        previous = jnp.full((3,), 22000.0)
        assert jnp.allclose(
            wmo_tropopause(isothermal, pressure, surface_pressure, previous),
            previous)
        check_gradients(wmo_tropopause,
                        (isothermal, pressure, surface_pressure, previous),
                        reference="adjoint", live_inputs=["[3]"])

    def test_temperature_carries_no_gradient(self):
        """``d(p_tropopause)/dT`` is identically zero, and finite.

        Every temperature dependence in the scan passes through a
        comparison. Pinned here so that a later smoothing of the criterion
        — a change to the diagnostic, not a guard — shows up as a failure
        rather than as a silently different gradient.
        """
        temperature, pressure, surface_pressure = _sounding(ncols=3)
        gradients = jax.grad(
            lambda t, p, ps: jnp.sum(wmo_tropopause(t, p, ps) ** 2),
            argnums=(0, 1, 2),
        )(temperature, pressure, surface_pressure)
        names = ("temperature", "pressure", "surface_pressure")
        for name, gradient in zip(names, gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite: {gradient}")
        assert jnp.all(gradients[0] == 0.0), (
            "d/dtemperature is no longer identically zero; the tropopause "
            "criterion has been smoothed, which changes the diagnostic")
        assert jnp.any(np.asarray(gradients[1]) != 0.0), (
            "the selected pressure no longer carries a gradient")
        assert jnp.all(gradients[2] == 0.0)


if __name__ == "__main__":
    test_compute_geopotential_height()
    test_compute_lapse_rate()
    test_find_tropopause_level()
    test_wmo_tropopause()
    test_wmo_tropopause_with_previous()
    test_wmo_tropopause_fallback()
    test_wmo_constants()
    print("All WMO tropopause tests passed!")