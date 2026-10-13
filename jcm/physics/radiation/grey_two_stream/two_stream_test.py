"""Unit tests for two-stream radiative transfer solver

Tests the two-stream approximation implementation including
coefficients, layer properties, flux calculations, and heating rates.

Date: 2025-01-10
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jcm.physics.radiation.grey_two_stream.two_stream import (
    two_stream_coefficients,
    layer_reflectance_transmittance,
    delta_eddington_scaling,
    longwave_fluxes,
    shortwave_fluxes,
    flux_to_heating_rate
)
from jcm.physics.radiation.radiation_types import OpticalProperties
from jcm.physics.radiation.grey_two_stream.planck import planck_bands_lw
from jcm.testing import check_gradients


def test_two_stream_coefficients():
    """Test two-stream coefficient calculations"""
    tau = jnp.array([0.1, 0.5, 1.0])
    ssa = jnp.array([0.9, 0.8, 0.7])
    g = jnp.array([0.85, 0.85, 0.85])
    
    # Test LW (no solar angle)
    gamma1, gamma2, gamma3, gamma4 = two_stream_coefficients(ssa, g, mu0=None)
    assert gamma1.shape == tau.shape
    assert jnp.all(gamma3 == 0)  # No direct beam
    assert jnp.all(gamma4 == 1)
    
    # Test SW
    mu0 = 0.5
    gamma1, gamma2, gamma3, gamma4 = two_stream_coefficients(ssa, g, mu0)
    assert jnp.all(gamma3 > 0)
    assert jnp.all(jnp.abs(gamma3 + gamma4 - 1.0) < 1e-10)


def test_layer_properties():
    """Test layer reflectance and transmittance"""
    tau = jnp.array([0.1, 1.0, 10.0])
    ssa = jnp.array([0.9, 0.9, 0.9])
    g = jnp.array([0.85, 0.85, 0.85])
    
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(tau, ssa, g, mu0=0.5)
    
    # Physical constraints
    assert jnp.all(R_dif >= 0) and jnp.all(R_dif <= 1)
    assert jnp.all(T_dif >= 0) and jnp.all(T_dif <= 1)
    assert jnp.all(R_dif + T_dif <= 1)  # Energy conservation
    
    # Larger optical depth = less transmission
    assert T_dif[0] > T_dif[1] > T_dif[2]


def test_layer_properties_large_tau():
    """Test layer properties with large optical depths (regression test for NaN fix)"""
    tau_values = jnp.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ssa = jnp.zeros_like(tau_values)  # Pure absorption (LW case)
    g = jnp.zeros_like(tau_values)
    
    # Should not produce NaN for any optical depth
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(tau_values, ssa, g, mu0=None)
    
    assert not jnp.any(jnp.isnan(R_dif))
    assert not jnp.any(jnp.isnan(T_dif))
    
    # Physical constraints
    assert jnp.all(R_dif >= 0)
    assert jnp.all(T_dif >= 0)
    
    # Large optical depths should have near-zero transmission
    assert T_dif[-1] < 1e-10  # tau=1000 should have virtually no transmission


def test_heating_rate():
    """Test flux to heating rate conversion"""
    nlev = 10
    flux_up = jnp.linspace(100, 300, nlev + 1)
    flux_down = jnp.linspace(400, 200, nlev + 1)
    pressure = jnp.linspace(100000, 10000, nlev + 1)
    
    heating = flux_to_heating_rate(flux_up, flux_down, pressure)
    
    assert heating.shape == (nlev,)
    # Net flux divergence should give heating/cooling
    assert jnp.any(heating != 0)
    
    # Should not have NaN values
    assert not jnp.any(jnp.isnan(heating))


def test_heating_rate_zero_pressure_gradient():
    """Test heating rate with zero pressure gradient"""
    nlev = 5
    flux_up = jnp.ones(nlev + 1) * 100.0
    flux_down = jnp.ones(nlev + 1) * 200.0
    
    # Constant pressure (zero gradient)
    pressure = jnp.ones(nlev + 1) * 50000.0
    
    # Should handle zero pressure gradient gracefully
    heating = flux_to_heating_rate(flux_up, flux_down, pressure)
    
    # With constant fluxes and zero pressure gradient, heating should be infinite or NaN
    # But the function should not crash
    assert heating.shape == (nlev,)


def test_longwave_fluxes():
    """Test longwave flux calculation"""
    nlev = 10
    n_lw_bands = 3
    
    # Create test optical properties
    tau_lw = jnp.ones((nlev, n_lw_bands)) * 0.5
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.zeros((nlev, n_lw_bands)),  # Pure absorption
        asymmetry_factor=jnp.zeros((nlev, n_lw_bands))
    )
    
    # Temperature profile
    temperature = jnp.linspace(250, 290, nlev)
    
    # Planck functions
    lw_bands = ((10, 350), (350, 500), (500, 2500))
    planck_layer = planck_bands_lw(temperature, lw_bands)
    planck_interface = planck_bands_lw(
        jnp.linspace(250, 290, nlev + 1), lw_bands
    )
    
    # Surface properties
    surface_emissivity = 0.98
    surface_temp = 290.0
    surface_planck = planck_bands_lw(jnp.array([surface_temp]), lw_bands)[0]
    
    # Calculate fluxes
    flux_up_lw, flux_down_lw = longwave_fluxes(
        lw_optics, planck_layer, planck_interface,
        surface_emissivity, surface_planck, n_lw_bands
    )
    
    # Output has one column per active band (``n_bands`` is now a static
    # argument, so the flux shape matches the requested band count).
    assert flux_up_lw.shape == (nlev + 1, n_lw_bands)
    assert flux_down_lw.shape == (nlev + 1, n_lw_bands)
    
    # Check physical constraints
    assert jnp.all(flux_up_lw >= 0)
    assert jnp.all(flux_down_lw >= 0)
    
    # Should not have NaN values
    assert not jnp.any(jnp.isnan(flux_up_lw))
    assert not jnp.any(jnp.isnan(flux_down_lw))


def test_shortwave_fluxes():
    """Test shortwave flux calculation"""
    nlev = 10
    n_sw_bands = 2
    
    # Create test optical properties
    tau_sw = jnp.ones((nlev, n_sw_bands)) * 0.3
    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=jnp.ones((nlev, n_sw_bands)) * 0.9,
        asymmetry_factor=jnp.ones((nlev, n_sw_bands)) * 0.85
    )
    
    # Solar parameters
    cos_zenith = 0.5
    toa_flux = jnp.array([500.0, 500.0])  # W/m²
    surface_albedo = jnp.array([0.15, 0.15])
    
    # Calculate fluxes
    flux_up_sw, flux_down_sw, flux_dir, flux_dif = shortwave_fluxes(
        sw_optics, cos_zenith, toa_flux, surface_albedo, n_sw_bands
    )
    
    # Output has one column per active band (``n_bands`` is now a static
    # argument, so the flux shape matches the requested band count).
    assert flux_up_sw.shape == (nlev + 1, n_sw_bands)
    assert flux_down_sw.shape == (nlev + 1, n_sw_bands)
    assert flux_dir.shape == (nlev + 1, n_sw_bands)
    assert flux_dif.shape == (nlev + 1, n_sw_bands)
    
    # Check physical constraints
    assert jnp.all(flux_up_sw >= 0)
    assert jnp.all(flux_down_sw >= 0)
    assert jnp.all(flux_dir >= 0)
    assert jnp.all(flux_dif >= 0)
    
    # Net downward in SW (more down than up)
    assert jnp.all(flux_down_sw >= flux_up_sw)
    
    # Should not have NaN values
    assert not jnp.any(jnp.isnan(flux_up_sw))
    assert not jnp.any(jnp.isnan(flux_down_sw))


def test_two_stream_integration():
    """Integration test combining longwave and shortwave"""
    nlev = 20
    n_lw_bands = 3
    n_sw_bands = 2
    
    # Create test optical properties
    tau_lw = jnp.ones((nlev, n_lw_bands)) * 0.5
    tau_sw = jnp.ones((nlev, n_sw_bands)) * 0.3
    
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.zeros((nlev, n_lw_bands)),
        asymmetry_factor=jnp.zeros((nlev, n_lw_bands))
    )
    
    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=jnp.ones((nlev, n_sw_bands)) * 0.9,
        asymmetry_factor=jnp.ones((nlev, n_sw_bands)) * 0.85
    )
    
    # Temperature profile
    temperature = jnp.linspace(250, 290, nlev)
    
    # Planck functions
    lw_bands = ((10, 350), (350, 500), (500, 2500))
    planck_layer = planck_bands_lw(temperature, lw_bands)
    planck_interface = planck_bands_lw(
        jnp.linspace(250, 290, nlev + 1), lw_bands
    )
    
    # Surface properties
    surface_emissivity = 0.98
    surface_temp = 290.0
    surface_planck = planck_bands_lw(jnp.array([surface_temp]), lw_bands)[0]
    
    # Test LW
    flux_up_lw, flux_down_lw = longwave_fluxes(
        lw_optics, planck_layer, planck_interface,
        surface_emissivity, surface_planck, n_lw_bands
    )
    
    # Test SW
    cos_zenith = 0.5
    toa_flux = jnp.array([500.0, 500.0])
    surface_albedo = jnp.array([0.15, 0.15])
    
    flux_up_sw, flux_down_sw, flux_dir, flux_dif = shortwave_fluxes(
        sw_optics, cos_zenith, toa_flux, surface_albedo, n_sw_bands
    )
    
    # Test heating rate calculations
    pressure_interfaces = jnp.linspace(100000, 0, nlev + 1)
    
    lw_heating = flux_to_heating_rate(
        jnp.sum(flux_up_lw, axis=1), 
        jnp.sum(flux_down_lw, axis=1), 
        pressure_interfaces
    )
    
    sw_heating = flux_to_heating_rate(
        jnp.sum(flux_up_sw, axis=1), 
        jnp.sum(flux_down_sw, axis=1), 
        pressure_interfaces
    )
    
    total_heating = lw_heating + sw_heating
    
    # Verify no NaN values in final result
    assert not jnp.any(jnp.isnan(total_heating))
    
    # Check shapes
    assert lw_heating.shape == (nlev,)
    assert sw_heating.shape == (nlev,)
    assert total_heating.shape == (nlev,)


def test_extreme_optical_depths():
    """Test with extreme optical depth values"""
    tau_values = jnp.array([1e-10, 1e-5, 1e-2, 1.0, 100.0, 10000.0])
    ssa = jnp.ones_like(tau_values) * 0.9  # High scattering
    g = jnp.ones_like(tau_values) * 0.85
    
    # Should handle extreme values without NaN
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(tau_values, ssa, g, mu0=0.6)
    
    assert not jnp.any(jnp.isnan(R_dif))
    assert not jnp.any(jnp.isnan(T_dif))
    assert not jnp.any(jnp.isnan(R_dir))
    assert not jnp.any(jnp.isnan(T_dir))
    
    # Physical constraints
    assert jnp.all(R_dif >= 0) and jnp.all(R_dif <= 1)
    assert jnp.all(T_dif >= 0) and jnp.all(T_dif <= 1)
    
    # Very small optical depths should have high transmission
    assert T_dif[0] > 0.99
    assert T_dif[1] > 0.99
    
    # Very large optical depths should have low transmission
    assert T_dif[-1] < 1e-10
    assert T_dif[-2] < 1e-10


def test_longwave_realistic_olr():
    """BUG TEST: Ensure longwave radiation produces realistic OLR.

    Bug found: OLR is 0.17 W/m² instead of expected ~240 W/m²
    """
    nlev = 15
    n_lw_bands = 3

    # Realistic atmospheric profile
    tau_lw = jnp.array([
        [0.01, 0.01, 0.01],  # TOA - very thin
        [0.02, 0.02, 0.02],
        [0.03, 0.03, 0.03],
        [0.05, 0.05, 0.05],
        [0.08, 0.08, 0.08],
        [0.12, 0.12, 0.12],
        [0.18, 0.18, 0.18],
        [0.27, 0.27, 0.27],
        [0.40, 0.40, 0.40],
        [0.60, 0.60, 0.60],
        [0.90, 0.90, 0.90],
        [1.35, 1.35, 1.35],
        [2.00, 2.00, 2.00],
        [3.00, 3.00, 3.00],
        [4.50, 4.50, 4.50],  # Surface - thickest
    ])

    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.zeros((nlev, n_lw_bands)),  # Pure absorption for LW
        asymmetry_factor=jnp.zeros((nlev, n_lw_bands))
    )

    # Realistic temperature profile
    temperature = jnp.array([200, 200, 200, 200, 204, 214, 223, 232,
                            242, 251, 260, 269, 279, 288, 288])

    # Planck functions
    lw_bands = ((10, 350), (350, 500), (500, 2500))
    planck_layer = planck_bands_lw(temperature, lw_bands)

    # Interface temperatures (linearly interpolated)
    temp_interface = jnp.concatenate([
        jnp.array([temperature[0]]),
        0.5 * (temperature[:-1] + temperature[1:]),
        jnp.array([temperature[-1]])
    ])
    planck_interface = planck_bands_lw(temp_interface, lw_bands)

    # Surface properties
    surface_emissivity = 0.98
    surface_temp = 288.0
    surface_planck = planck_bands_lw(jnp.array([surface_temp]), lw_bands)[0]

    # Calculate fluxes
    flux_up_lw, flux_down_lw = longwave_fluxes(
        lw_optics, planck_layer, planck_interface,
        surface_emissivity, surface_planck, n_lw_bands
    )

    # TOA upward flux (OLR) is sum of all bands at top level
    olr = jnp.sum(flux_up_lw[0, :n_lw_bands])

    # BUG CHECK: OLR should be realistic (100-500 W/m² range)
    # Earth's global mean OLR is ~240 W/m²
    # Realistic range depends on atmospheric opacity and temperature profile
    assert olr > 50.0, f"OLR {olr:.1f} W/m² too small - likely LW flux bug (expected ~150-350 W/m²)"
    assert olr < 500.0, f"OLR {olr:.1f} W/m² too large - check for flux amplification bugs"

    # Surface upward LW should be close to surface emission
    # Note: surface_planck is in W/m²/sr (radiance), need to multiply by π for flux
    surface_up = jnp.sum(flux_up_lw[-1, :n_lw_bands])
    expected_surface = surface_emissivity * jnp.pi * jnp.sum(surface_planck[:n_lw_bands])
    assert jnp.abs(surface_up - expected_surface) / expected_surface < 0.1, \
        f"Surface LW up {surface_up:.1f} W/m² differs from expected {expected_surface:.1f} W/m²"


def test_shortwave_toa_net_flux():
    """BUG TEST: Ensure shortwave doesn't have 100% reflection at TOA.

    Bug found: SW up at TOA = 1306.59 W/m², SW down = 1306.60 W/m²
    (99.99% reflection, essentially nothing entering atmosphere!)
    """
    nlev = 15
    n_sw_bands = 2

    # Realistic atmospheric optical properties
    # Stratosphere (low tau), troposphere (higher tau)
    tau_sw = jnp.array([
        [0.02, 0.01],  # TOA
        [0.03, 0.02],
        [0.04, 0.03],
        [0.06, 0.04],
        [0.09, 0.06],
        [0.13, 0.09],
        [0.19, 0.13],
        [0.28, 0.19],
        [0.41, 0.28],
        [0.61, 0.41],
        [0.91, 0.61],
        [1.36, 0.91],
        [2.03, 1.36],
        [3.04, 2.03],
        [4.55, 3.04],  # Surface
    ])

    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=jnp.ones((nlev, n_sw_bands)) * 0.85,  # Moderate scattering
        asymmetry_factor=jnp.ones((nlev, n_sw_bands)) * 0.85
    )

    # Solar parameters (noon at mid-latitude)
    cos_zenith = 0.5  # 60° solar zenith angle
    toa_flux = jnp.array([600.0, 600.0])  # W/m² per band
    surface_albedo = jnp.array([0.2, 0.2])  # Typical land surface

    # Calculate fluxes
    flux_up_sw, flux_down_sw, flux_dir, flux_dif = shortwave_fluxes(
        sw_optics, cos_zenith, toa_flux, surface_albedo, n_sw_bands
    )

    # TOA fluxes
    toa_down = jnp.sum(flux_down_sw[0, :n_sw_bands])
    toa_up = jnp.sum(flux_up_sw[0, :n_sw_bands])
    toa_net = toa_down - toa_up

    # BUG CHECK: TOA should not have 100% reflection
    # Planetary albedo is typically 0.3-0.4, so we expect 60-70% to enter atmosphere
    reflection_fraction = toa_up / toa_down if toa_down > 0 else 1.0

    assert reflection_fraction < 0.6, \
        f"TOA reflection {reflection_fraction*100:.1f}% too high - likely SW flux bug (expected ~30-40%)"

    # Net flux entering atmosphere should be positive and significant
    total_toa_flux = jnp.sum(toa_flux[:n_sw_bands])
    net_fraction = toa_net / total_toa_flux if total_toa_flux > 0 else 0.0

    assert net_fraction > 0.4, \
        f"Net SW flux entering atmosphere is only {net_fraction*100:.1f}% of TOA - too low!"

    # Surface should receive some SW radiation
    # Note: For optically thick atmospheres (τ>10), transmission can be <1%
    # This test mainly checks that surface flux is non-zero and properly reflects surface albedo
    surface_down = jnp.sum(flux_down_sw[-1, :n_sw_bands])

    assert surface_down > 0.0, "Surface SW down is zero - radiation not reaching surface!"

    # Check that surface albedo is being applied correctly (per band)
    for band in range(n_sw_bands):
        surf_down_band = flux_down_sw[-1, band]
        surf_up_band = flux_up_sw[-1, band]
        expected_up = surf_down_band * surface_albedo[band]

        if surf_down_band > 0.01:  # Only check if there's significant downward flux
            rel_error = jnp.abs(surf_up_band - expected_up) / (expected_up + 1e-10)
            assert rel_error < 0.01, \
                f"Band {band}: Surface up {surf_up_band:.2f} W/m² doesn't match expected {expected_up:.2f} W/m²"

class TestTwoStreamGradients:
    """The conservative-scattering limit, and the eigenvalue's conditioning.

    ``layer_reflectance_transmittance`` builds the Eddington eigenvalue from
    ``gamma1`` and ``gamma2``, and both degenerate as the single-scattering
    albedo approaches 1 — which is not an exotic corner but exactly where
    shortwave liquid-cloud optics sits (``ssa ~ 0.9999``). These tests fence
    the two conditioning hazards that survive there:

    * ``gamma1**2 - gamma2**2`` is a catastrophic cancellation as ``ssa -> 1``
      (the two squares agree to four digits at ``ssa = 0.9999``), and feeding
      the round-off that survives into ``sqrt`` — whose derivative is
      ``1/(2*sqrt(x))`` — amplifies it. The scheme now forms the same
      quantity factored, ``3*(1 - ssa)*(1 - ssa*g)``, which has no
      cancellation and is exactly 0 at the limit.
    * ``lambda`` itself has a square-root cusp at ``ssa = 1``, but R and T are
      *even* in ``lambda`` and hence smooth functions of the smooth
      ``lambda**2 = 3(1-ssa)(1-ssa*g)`` — a genuine two-sided derivative
      exists at the endpoint. The scheme evaluates an even Taylor series in
      ``(lambda*tau)**2`` for ``lambda*tau < 0.1`` so autodiff receives that
      derivative rather than a guard's zero. The reflectance/transmittance
      are also written so ``gamma1`` and ``gamma1 + lambda`` only ever
      multiply in — never divide — so the ``gamma2/gamma1`` 0/0 at
      ``ssa = g = 1`` that earlier forms needed no longer arises.
    """

    NLEV = 6

    def _layer(self, ssa_value, g_value=0.85, tau_value=0.3):
        """Build (tau, ssa, g) profiles at one operating point."""
        return (jnp.full((self.NLEV,), tau_value),
                jnp.full((self.NLEV,), ssa_value),
                jnp.full((self.NLEV,), g_value))

    @staticmethod
    def _sum_of_squares(tau, ssa, g, mu0=0.5):
        """Reduce all four layer coefficients to one differentiable scalar."""
        R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(
            tau, ssa, g, mu0)
        return jnp.sum(R_dif ** 2 + T_dif ** 2 + R_dir ** 2 + T_dir ** 2)

    @pytest.mark.parametrize(
        "ssa, g",
        [(1.0, 0.85),      # conservative scattering, realistic asymmetry
         (1.0, 1.0),       # gamma1 == gamma2 == 0: the 0/0 in the layer albedo
         (0.999999, 0.85),
         (0.0, 0.0)],      # pure absorption, the longwave case
    )
    def test_gradients_are_finite_at_the_scattering_limits(self, ssa, g):
        """No limit of (ssa, g) may return a non-finite derivative.

        ``ssa = g = 1`` is the sharpest corner: ``gamma1``, ``gamma2`` and
        ``lambda`` all vanish. Both branches divide only by denominators
        bounded below by 1, and near the limit the series branch is a
        polynomial in the smooth ``lambda**2`` — nothing there is singular.
        """
        tau, ssa_p, g_p = self._layer(ssa, g)
        grads = jax.grad(self._sum_of_squares, argnums=(0, 1, 2))(
            tau, ssa_p, g_p)
        for name, grad in zip(("tau", "ssa", "g"), grads):
            assert jnp.all(jnp.isfinite(grad)), (
                f"d/d{name} is not finite at ssa={ssa}, g={g}: {grad}")

    def test_conservative_limit_derivative_is_the_true_endpoint_derivative(self):
        """At ``ssa = 1`` autodiff must return the real derivative, not a
        guard's substitute and not cancellation noise.

        R and T are even functions of ``lambda``, hence smooth functions of
        ``lambda**2 = 3(1-ssa)(1-ssa*g)``: the endpoint derivative exists
        two-sided and has a closed form. At ``x2 = (lambda*tau)**2 = 0``,
        with ``D = 1 + gamma1*tau``, ``dgamma1/dssa = -(4+3g)/4``,
        ``dgamma2/dssa = (4-3g)/4`` and ``dx2/dssa = -3(1-g)*tau**2``:

            dT/dssa = -[dgamma1*tau + (1/2 + gamma1*tau/6)*dx2] / D**2
            dR/dssa = [dN*D - gamma1*tau*dD] / D**2,
              dN = dgamma2*tau + gamma1*tau*dx2/6,  dD = -the bracket above.

        Two historical failure modes bracketed this value: the subtracted
        eigenvalue form reported 5e5 of pure ``sqrt``-of-round-off noise, and
        a double-``where`` that pinned ``lambda``/``S`` at their limit values
        dropped the ``dx2`` channel and reported 0.4597 for a true 0.4789
        (dT/dssa at tau=0.3, g=0.85 — PR #856 review). The even-series branch
        must reproduce the closed form.
        """
        tau_v, g_v = 0.3, 0.85
        tau, ssa, g = self._layer(1.0, g_v, tau_value=tau_v)

        def r_and_t(s):
            r, t, _, _ = layer_reflectance_transmittance(tau, s, g, None)
            return r[0], t[0]

        d_r = jax.grad(lambda s: r_and_t(s)[0])(ssa)[0]
        d_t = jax.grad(lambda s: r_and_t(s)[1])(ssa)[0]

        g1 = 3.0 * (1.0 - g_v) / 4.0
        dg1 = -(4.0 + 3.0 * g_v) / 4.0
        dg2 = (4.0 - 3.0 * g_v) / 4.0
        dx2 = -3.0 * (1.0 - g_v) * tau_v**2
        big_d = 1.0 + g1 * tau_v
        d_denom = dg1 * tau_v + (0.5 + g1 * tau_v / 6.0) * dx2
        dt_exact = -d_denom / big_d**2
        dn = dg2 * tau_v + g1 * tau_v * dx2 / 6.0
        dr_exact = (dn * big_d - g1 * tau_v * d_denom) / big_d**2

        assert float(d_t) == pytest.approx(dt_exact, rel=1e-4)
        assert float(d_r) == pytest.approx(dr_exact, rel=1e-4)

    @pytest.mark.parametrize("seed", [0, 4])
    def test_layer_coefficients_match_a_central_difference(self, seed):
        """AD against the secant at a realistic scattering cloud layer.

        ``ssa = 0.93`` and ``g = 0.85``: a water cloud in the shortwave, well
        clear of the ``ssa -> 1`` limit above and of the ``ssa > 0.001``
        pure-absorption switch, and at an optical depth far below the
        ``lambda_tau >= 88`` asymptotic branch.
        """
        tau, ssa, g = self._layer(0.93, 0.85, tau_value=2.4)
        check_gradients(
            lambda t, s, a: layer_reflectance_transmittance(t, s, a, 0.5),
            (tau, ssa, g), rtol=1e-3, seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    def test_longwave_fluxes_match_a_central_difference(self, seed):
        """AD against the secant through the two flux recurrences."""
        n_bands = 3
        optics = OpticalProperties(
            optical_depth=jnp.linspace(0.12, 0.9, self.NLEV)[:, None]
            * jnp.ones((1, n_bands)),
            single_scatter_albedo=jnp.zeros((self.NLEV, n_bands)),
            asymmetry_factor=jnp.zeros((self.NLEV, n_bands)),
        )
        lw_bands = ((10, 350), (350, 500), (500, 2500))
        planck_layer = planck_bands_lw(
            jnp.linspace(245.0, 291.0, self.NLEV), lw_bands)
        planck_interface = planck_bands_lw(
            jnp.linspace(243.0, 293.0, self.NLEV + 1), lw_bands)
        surface_planck = planck_bands_lw(jnp.array([292.0]), lw_bands)[0]

        def f(tau, planck_l, planck_i, sfc_planck, emissivity):
            return longwave_fluxes(
                optics._replace(optical_depth=tau), planck_l, planck_i,
                emissivity, sfc_planck, n_bands)

        check_gradients(
            f, (optics.optical_depth, planck_layer, planck_interface,
                surface_planck, jnp.array(0.98)),
            rtol=1e-3, seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    def test_shortwave_fluxes_match_a_central_difference(self, seed):
        """AD against the secant through the shortwave two-stream solve."""
        n_bands = 2
        tau = jnp.linspace(0.08, 0.6, self.NLEV)[:, None] * jnp.ones((1, n_bands))
        ssa = jnp.full((self.NLEV, n_bands), 0.93)
        asym = jnp.full((self.NLEV, n_bands), 0.85)

        def f(tau, ssa, asym, toa_flux, albedo):
            return shortwave_fluxes(
                OpticalProperties(optical_depth=tau,
                                  single_scatter_albedo=ssa,
                                  asymmetry_factor=asym),
                0.62, toa_flux, albedo, n_bands)

        check_gradients(
            f, (tau, ssa, asym, jnp.array([620.0, 480.0]),
                jnp.array([0.13, 0.19])),
            rtol=1e-3, seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    def test_heating_rate_matches_a_central_difference(self, seed):
        """``flux_to_heating_rate`` is broadcasting-native over trailing axes.

        Checked as a single ``(nlev+1,)`` column and as a ``(nlev+1, 4)``
        block, because the routine reduces only over axis 0 and so is one of
        the few pieces here that genuinely broadcasts.
        """
        up = jnp.linspace(100.0, 300.0, self.NLEV + 1)
        down = jnp.linspace(400.0, 200.0, self.NLEV + 1)
        p_half = jnp.linspace(101000.0, 9000.0, self.NLEV + 1)
        check_gradients(flux_to_heating_rate, (up, down, p_half),
                        rtol=1e-3, seed=seed)

        spread = jnp.array([0.85, 1.0, 1.15, 1.3])
        check_gradients(
            flux_to_heating_rate,
            (up[:, None] * spread, down[:, None] * spread,
             p_half[:, None] * jnp.ones(4)),
            rtol=1e-3, seed=seed)

    @pytest.mark.parametrize(
        "tau", [0.1, 1.0, 10.0, 30.0, 50.0, 60.0, 200.0, 1.0e4, 1.0e6])
    @pytest.mark.parametrize("ssa, g", [(0.0, 0.0), (0.93, 0.85)])
    def test_both_ad_modes_survive_an_optically_thick_layer(self, tau, ssa, g):
        """Thick layers must not return NaN in *either* AD mode.

        The reflectance/transmittance uses one expression at every optical
        depth — there is no thick-layer branch to disagree with — and forms it
        from the decaying exponentials ``exp(-lambda*tau)`` and
        ``exp(-2*lambda*tau)`` only. No growing ``exp(+lambda*tau)`` appears,
        so neither an overflow on a discarded ``where`` branch (which reverse
        mode turned into ``0 * inf``) nor a ``denominator**2`` reaching 1e38
        (which forward mode turned into NaN across a thick longwave column) can
        occur. This sweeps ``tau`` well past where both used to strike.

        ``tau = 1e6`` additionally fences the *series* branch's discarded-side
        overflow: ``where`` evaluates both branches, and without the ``x2``
        clamp the unused Taylor polynomial went inf/inf = NaN in float32 there
        (``tau * x2**3`` passes 3.4e38 near ``lambda*tau ~ 1e6``), so every
        reverse-mode gradient — w.r.t. tau, ssa and g alike — was NaN while
        the forward value stayed (0, 0) (PR #856 review). At these depths all
        true derivatives are exp-suppressed to ~0, so finiteness in both
        modes, not a central-difference match against a function that is flat
        to round-off, is the meaningful assertion.
        """
        tau_p, ssa_p, g_p = self._layer(ssa, g, tau_value=tau)
        _, tangents = jax.jvp(
            lambda t, s, a: layer_reflectance_transmittance(t, s, a, None),
            (tau_p, ssa_p, g_p),
            (jnp.ones_like(tau_p), jnp.ones_like(ssa_p), jnp.ones_like(g_p)))
        for name, t in zip(("R_dif", "T_dif", "R_dir", "T_dir"), tangents):
            assert jnp.all(jnp.isfinite(t)), f"jvp of {name} is not finite"

        grads = jax.grad(
            lambda t, s, a: jnp.sum(
                layer_reflectance_transmittance(t, s, a, None)[0]
                + layer_reflectance_transmittance(t, s, a, None)[1]),
            argnums=(0, 1, 2))(tau_p, ssa_p, g_p)
        for name, grad in zip(("tau", "ssa", "g"), grads):
            assert jnp.all(jnp.isfinite(grad)), f"vjp d/d{name} is not finite"


class TestLayerForwardValue:
    """The diffuse layer albedo/transmittance is the exact homogeneous-layer
    two-stream solution (Meador & Weaver 1980 eq. 14-15; Toon et al. 1989).

    ``layer_reflectance_transmittance`` evaluates a rearrangement of that
    solution; these tests pin the *forward* value it must reproduce, which the
    earlier ``gamma2`` / ``gamma2/gamma1`` substitutions got wrong (#848). They
    are the reference the gradient tests above cannot see: an AD check confirms
    the derivative of whatever is computed, not that the right thing is.
    """

    @staticmethod
    def _reference(tau, ssa, g):
        """Meador-Weaver diffuse R, T for one Eddington layer, in float64.

        ``Gamma = gamma2/(gamma1 + lambda)``; at ``lambda = 0`` (conservative
        scattering) the ratio form is 0/0 and the closed limit
        ``R = gamma1*tau/(1 + gamma1*tau)``, ``T = 1 - R`` applies.
        """
        g1 = (7.0 - ssa * (4.0 + 3.0 * g)) / 4.0
        g2 = -(1.0 - ssa * (4.0 - 3.0 * g)) / 4.0
        lam = np.sqrt(max(g1 * g1 - g2 * g2, 0.0))
        if lam == 0.0:
            r = g1 * tau / (1.0 + g1 * tau)
            return r, 1.0 - r
        gamma = g2 / (g1 + lam)
        e2 = np.exp(-2.0 * lam * tau)
        e = np.exp(-lam * tau)
        denom = 1.0 - gamma * gamma * e2
        r = gamma * (1.0 - e2) / denom
        t = (1.0 - gamma * gamma) * e / denom
        return max(r, 0.0), t  # the code clips the small-ssa negative-R corner

    @pytest.mark.parametrize(
        "tau, ssa, g",
        [(10.0, 0.99, 0.85),   # ordinary shortwave water cloud
         (2.0, 0.99, 0.85),
         (0.3, 0.9, 0.7),      # thin scattering layer
         (10.0, 0.5, 0.6),     # weakly scattering
         (1.0, 0.999, 0.85),   # near-conservative
         (5.0, 0.85, 0.85)],
    )
    def test_matches_meador_weaver(self, tau, ssa, g):
        """R and T equal the exact solution, not the old ``gamma2`` proxy.

        At ``ssa = 0.99, g = 0.85, tau = 10`` the pre-fix code returned
        ``R = 0.099`` against the exact 0.446 — the defect this pins.
        """
        r_code, t_code, _, _ = layer_reflectance_transmittance(
            jnp.array([tau]), jnp.array([ssa]), jnp.array([g]), None)
        r_ref, t_ref = self._reference(tau, ssa, g)
        assert float(r_code[0]) == pytest.approx(r_ref, abs=1e-5, rel=1e-4)
        assert float(t_code[0]) == pytest.approx(t_ref, abs=1e-5, rel=1e-4)

    @pytest.mark.parametrize("tau", [1.0, 10.0, 50.0])
    def test_conservative_scattering_reflects_and_conserves(self, tau):
        """``ssa = 1``: a non-absorbing layer reflects, and ``R + T = 1``.

        The pre-fix code returned ``R = 0, T = 1`` for any ``tau`` — a
        conservative cloud that reflected nothing. The closed limit is
        ``R = gamma1*tau/(1 + gamma1*tau)`` (0.529 at tau = 10, g = 0.85).
        """
        g = 0.85
        r, t, _, _ = layer_reflectance_transmittance(
            jnp.array([tau]), jnp.array([1.0]), jnp.array([g]), None)
        g1 = (7.0 - (4.0 + 3.0 * g)) / 4.0
        assert float(r[0]) == pytest.approx(g1 * tau / (1.0 + g1 * tau),
                                            abs=1e-5, rel=1e-4)
        assert float(r[0] + t[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(r[0]) > 0.05  # it reflects something

    @pytest.mark.parametrize(
        "ssa, g", [(0.9, 0.7), (0.99, 0.85), (0.999, 0.85)])
    def test_semi_infinite_albedo_is_gamma(self, ssa, g):
        """As ``tau -> inf`` the layer albedo tends to ``Gamma`` and ``T -> 0``.

        This is the limit the old ``lambda_tau >= 88`` branch approximated as
        ``clip(gamma2/gamma1)``; it is now the natural tau -> inf limit of the
        single expression, so it needs no separate branch to reach.
        """
        r, t, _, _ = layer_reflectance_transmittance(
            jnp.array([1.0e6]), jnp.array([ssa]), jnp.array([g]), None)
        g1 = (7.0 - ssa * (4.0 + 3.0 * g)) / 4.0
        g2 = -(1.0 - ssa * (4.0 - 3.0 * g)) / 4.0
        gamma = g2 / (g1 + np.sqrt(g1 * g1 - g2 * g2))
        assert float(r[0]) == pytest.approx(gamma, abs=1e-5, rel=1e-4)
        assert float(t[0]) < 1e-6

    @pytest.mark.parametrize("ssa, g", [(0.9, 0.7), (0.99, 0.85), (0.5, 0.6)])
    @pytest.mark.parametrize("lambda_tau", [80.0, 86.0, 87.9, 88.0, 88.1, 90.0])
    def test_continuous_across_the_old_cutoff(self, ssa, g, lambda_tau):
        """No step in R at the retired ``lambda_tau = 88`` switch.

        A ~7x jump in the layer albedo across this value was the visible symptom
        of #848 (the two branches used different wrong forms). With one
        expression everywhere, R at any ``lambda_tau`` near 88 equals the
        semi-infinite limit ``Gamma`` to round-off — there is nothing to step.
        """
        g1 = (7.0 - ssa * (4.0 + 3.0 * g)) / 4.0
        g2 = -(1.0 - ssa * (4.0 - 3.0 * g)) / 4.0
        lam = np.sqrt(g1 * g1 - g2 * g2)
        tau = lambda_tau / lam
        r, _, _, _ = layer_reflectance_transmittance(
            jnp.array([tau]), jnp.array([ssa]), jnp.array([g]), None)
        gamma = max(g2 / (g1 + lam), 0.0)
        assert float(r[0]) == pytest.approx(gamma, abs=1e-5, rel=1e-4)

    @pytest.mark.parametrize(
        "ssa, g, tau", [(0.99, 0.85, 2.0), (0.9, 0.7, 1.0), (0.999, 0.85, 3.0)])
    def test_adding_identity(self, ssa, g, tau):
        """A homogeneous ``2*tau`` layer equals two ``tau`` layers combined.

        ``R(2 tau) = R + T^2 R/(1 - R^2)`` for a homogeneous layer, a
        self-contained consistency check the exact solution satisfies and the
        old ``gamma2`` form violated by ~23%.
        """
        r1, t1, _, _ = layer_reflectance_transmittance(
            jnp.array([tau]), jnp.array([ssa]), jnp.array([g]), None)
        r2, _, _, _ = layer_reflectance_transmittance(
            jnp.array([2.0 * tau]), jnp.array([ssa]), jnp.array([g]), None)
        r1, t1 = float(r1[0]), float(t1[0])
        combined = r1 + t1 * t1 * r1 / (1.0 - r1 * r1)
        assert float(r2[0]) == pytest.approx(combined, abs=1e-5, rel=1e-4)

    def test_energy_bound_over_a_full_sweep(self):
        """``R + T <= 1`` for every (ssa, g, tau), with no NaN.

        Energy can only be conserved or absorbed, never created. The clip
        upper-bounds R and T individually; this checks the physically stronger
        joint bound holds *before* any per-value clip could mask a violation.
        The tolerance is a float32 allowance: ``R + T <= 1`` is exact
        analytically (``R + T - 1 = (gamma2 - gamma1) S/denom <= 0``), but at
        the conservative limit where it equals 1 exactly the float32 division
        rounds a few ulp over.
        """
        ssa = jnp.linspace(0.0, 1.0, 21)[:, None, None]
        g = jnp.linspace(0.0, 0.99, 11)[None, :, None]
        tau = jnp.array([1e-3, 0.1, 1.0, 5.0, 50.0, 500.0])[None, None, :]
        ssa, g, tau = jnp.broadcast_arrays(ssa, g, tau)
        r, t, _, _ = layer_reflectance_transmittance(tau, ssa, g, None)
        assert not jnp.any(jnp.isnan(r)) and not jnp.any(jnp.isnan(t))
        assert float(jnp.max(r + t)) <= 1.0 + 1e-4


def _eddington_gammas(ssa, g, mu0):
    """Eddington ``gamma1..4`` (Toon et al. 1989, Table 1), float64."""
    g1 = (7.0 - ssa * (4.0 + 3.0 * g)) / 4.0
    g2 = -(1.0 - ssa * (4.0 - 3.0 * g)) / 4.0
    g3 = (2.0 - 3.0 * g * mu0) / 4.0
    return g1, g2, g3, 1.0 - g3


def _delta_scaled(tau, ssa, g):
    """Joseph et al. (1976) delta-Eddington scaling, float64."""
    gp = max(g, 0.0)
    f = gp * gp
    return ((1.0 - ssa * f) * tau, ssa * (1.0 - f) / (1.0 - ssa * f),
            g - f / (1.0 + gp))


def _direct_reference(tau, ssa, g, mu0):
    """Textbook direct-beam layer solution in float64 (Toon et al. 1989).

    Particular solution ``C+/C-`` over ``lambda**2 - 1/mu0**2`` plus the
    homogeneous solution, written independently of the code's rearrangement:
    the four boundary constants are solved as a 2x2 linear system. Valid away
    from the resonance and from ``lambda = 0``.
    """
    g1, g2, g3, g4 = _eddington_gammas(ssa, g, mu0)
    m = 1.0 / mu0
    lam = np.sqrt(g1 * g1 - g2 * g2)
    a = ssa * m * (g3 * (g1 - m) + g2 * g4) / (lam * lam - m * m)
    b = ssa * m * (g4 * (g1 + m) + g2 * g3) / (lam * lam - m * m)
    # Homogeneous modes exp(+/- lam t): F+ = c1 u1 e^{lam t} + c2 u2 e^{-lam t},
    # F- = c1 e^{lam t} + c2 e^{-lam t}, with u = F+/F- of each mode.
    u_plus = g2 / (g1 - lam)      # mode e^{+lam t}: (g1 - lam) F+ = g2 F-
    u_minus = g2 / (g1 + lam)     # mode e^{-lam t}: (g1 + lam) F+ = g2 F-
    e_p, e_m = np.exp(lam * tau), np.exp(-lam * tau)
    E = np.exp(-m * tau)
    # F-(0) = 0 and F+(tau) = 0.
    mat = np.array([[1.0, 1.0], [u_plus * e_p, u_minus * e_m]])
    rhs = np.array([-b, -a * E])
    c1, c2 = np.linalg.solve(mat, rhs)
    r_dir = a + c1 * u_plus + c2 * u_minus
    t_dir = b * E + c1 * e_p + c2 * e_m
    return r_dir, t_dir


class TestShortwaveEnergyConservation:
    """The shortwave two-stream conserves energy and reflects from clouds.

    The direct beam is solved with the Toon et al. (1989) / Meador & Weaver
    (1980) source functions on delta-Eddington layers and the column is joined
    by the adding method, so with no absorption everything incident at the
    top leaves through the top or is absorbed at the surface, and a thick
    conservative cloud has the two-stream cloud albedo (#855: the scattered
    beam used to be dropped, and such a cloud reflected ~0).
    """

    SSA = (0.5, 0.9, 0.999, 1.0)
    TAU = (0.1, 1.0, 10.0, 100.0)
    MU0 = (0.1, 0.5, 1.0)
    NLEV = 10

    @staticmethod
    def _fluxes(tau, ssa, g, mu0, albedo, toa=1.0):
        optics = OpticalProperties(
            optical_depth=jnp.asarray(tau, jnp.float32)[:, None],
            single_scatter_albedo=jnp.asarray(ssa, jnp.float32)[:, None],
            asymmetry_factor=jnp.asarray(g, jnp.float32)[:, None])
        up, down, direct, diffuse = shortwave_fluxes(
            optics, mu0, jnp.array([toa]), jnp.array([albedo]), 1)
        return up[:, 0], down[:, 0], direct[:, 0], diffuse[:, 0]

    def _column(self, tau_total, ssa, mu0, albedo, g=0.85):
        n = self.NLEV
        return self._fluxes(np.full(n, tau_total / n), np.full(n, ssa),
                            np.full(n, g), mu0, albedo)

    @pytest.mark.parametrize("g", [-1.0, -0.9, 0.0, 0.85, 1.0])
    @pytest.mark.parametrize("albedo", [0.0, 0.3])
    def test_reflection_transmission_absorption_close(self, albedo, g):
        """``R + T + A = 1`` over the sweep, each term from its own fluxes.

        ``R`` is the TOA upward flux, ``T`` the net flux into the surface and
        ``A`` the sum of the layers' net-flux convergences, each of which must
        be non-negative (a layer cannot emit shortwave). With ``ssa = 1`` no
        layer absorbs, so ``R + T = 1`` on its own. The TOA downward flux is
        exactly the incident beam (no diffuse light enters from space).
        Measured float32 closure: 5e-7.

        The asymmetry sweep covers the whole physical range: backward
        scattering (``g < 0``, passed through the delta scaling unscaled),
        isotropic, a cloud droplet, and the pure forward peak ``g = 1``,
        which the scaling makes transparent. For ``g mu0 < -2/3`` the
        Eddington downward direct-scattering fraction
        ``gamma4 = (2 + 3 g mu0)/4`` is itself negative — a property of the
        closure, not of the scaling or the solver — and the diffuse
        *component* of the downward flux dips below zero by up to 1.5 % of
        the incident flux (at ``g = -1``, overhead sun, a thin layer). Energy
        is still conserved there and the total downward flux, the upward
        flux and every layer's absorption stay non-negative; only the diffuse
        component's sign is relaxed, and only in that regime.
        """
        for ssa in self.SSA:
            for tau in self.TAU:
                for mu0 in self.MU0:
                    up, down, direct, diffuse = self._column(
                        tau, ssa, mu0, albedo, g=g)
                    case = (f"g={g} ssa={ssa} tau={tau} mu0={mu0} "
                            f"albedo={albedo}")
                    for flux in (up, down, direct, diffuse):
                        assert np.all(np.isfinite(np.asarray(flux))), case
                    assert float(down[0]) == pytest.approx(1.0, abs=1e-7), case
                    assert float(diffuse[0]) == 0.0, case
                    net = np.asarray(down - up, np.float64)
                    layer_absorption = net[:-1] - net[1:]
                    r, t = float(up[0]), float(net[-1])
                    a = float(layer_absorption.sum())
                    assert abs(r + t + a - 1.0) < 1e-6, case
                    assert np.all(layer_absorption > -1e-6), case
                    assert np.all(np.asarray(up) >= 0.0), case
                    assert np.all(np.asarray(down) >= 0.0), case
                    if g * mu0 >= -2.0 / 3.0:
                        assert np.all(np.asarray(diffuse) >= 0.0), case
                    else:
                        assert np.all(np.asarray(diffuse) > -0.02), case
                    if ssa == 1.0:
                        assert abs(r + t - 1.0) < 1e-6, case
                    else:
                        assert a > 0.0, case
                    # The surface reflects its albedo of what reaches it.
                    assert float(up[-1]) == pytest.approx(
                        albedo * float(down[-1]), abs=1e-7), case

    def test_delta_scaling_domain(self):
        """The forward-peak truncation acts only on ``g > 0``.

        ``g <= 0`` passes through unchanged (``f = max(g, 0)**2``): applied
        to backward scattering the formula would leave the physical range
        (``g = -0.9 -> g' = -9``) or divide by zero (``g = -1``). The scaled
        asymmetry is continuous with slope 1 through ``g = 0``, so its
        gradient is finite (and one) there from either side.
        """
        g = jnp.array([-1.0, -0.9, -0.3, 0.0, 0.3, 0.85, 1.0])
        tau, ssa, gs = delta_eddington_scaling(
            jnp.ones_like(g), jnp.full_like(g, 0.9), g)
        for x in (tau, ssa, gs):
            assert np.all(np.isfinite(np.asarray(x)))
        np.testing.assert_array_equal(np.asarray(gs[:4]), np.asarray(g[:4]))
        np.testing.assert_array_equal(np.asarray(tau[:4]), 1.0)
        np.testing.assert_allclose(np.asarray(ssa[:4]), 0.9, rtol=1e-7)
        assert np.all((np.asarray(gs) >= -1.0) & (np.asarray(gs) <= 0.5))
        for side in (-1e-4, 0.0, 1e-4):
            grads = jax.grad(
                lambda a: jnp.sum(jnp.stack(delta_eddington_scaling(
                    jnp.float32(2.0), jnp.float32(0.9), a))),
            )(jnp.float32(side))
            assert jnp.isfinite(grads)
            dg = jax.grad(lambda a: delta_eddington_scaling(
                jnp.float32(2.0), jnp.float32(0.9), a)[2])(jnp.float32(side))
            assert float(dg) == pytest.approx(1.0, abs=1e-3)

    def test_pure_backscatter_layer_is_finite_and_conserves(self):
        """``g = -1``: no NaN in the fluxes or their gradients, energy closes.

        This divided by zero when the truncation was applied to ``g < 0``.
        """
        n = self.NLEV
        up, down, direct, diffuse = self._fluxes(
            np.full(n, 0.5), np.full(n, 0.99), np.full(n, -1.0), 1.0, 0.2)
        for flux in (up, down, direct, diffuse):
            assert np.all(np.isfinite(np.asarray(flux)))
        assert np.all(np.asarray(up) >= 0.0)
        assert np.all(np.asarray(down) >= 0.0)
        net = np.asarray(down - up, np.float64)
        absorption = net[:-1] - net[1:]
        assert abs(float(up[0]) + net[-1] + absorption.sum() - 1.0) < 1e-6
        assert np.all(absorption > -1e-6)

        def total(g):
            optics = OpticalProperties(
                optical_depth=jnp.full((n, 1), 0.5),
                single_scatter_albedo=jnp.full((n, 1), 0.99),
                asymmetry_factor=g)
            u, d, _, _ = shortwave_fluxes(
                optics, 1.0, jnp.array([1.0]), jnp.array([0.2]), 1)
            return jnp.sum(u) + jnp.sum(d)

        grad = jax.grad(total)(jnp.full((n, 1), -1.0))
        assert jnp.all(jnp.isfinite(grad))

    @pytest.mark.parametrize("mu0", [0.1, 0.5, 1.0])
    def test_layer_scatters_all_of_a_conservative_beam(self, mu0):
        """Per layer: ``R_dir + T_dir + exp(-tau/mu0) = 1`` at ``ssa = 1``.

        The layer-level identity behind the column closure, on the
        delta-scaled properties the solver uses.
        """
        tau = jnp.array(self.TAU, jnp.float32)
        t, s, a = _delta_scaled(np.asarray(self.TAU), 1.0, 0.85)
        _, _, r_dir, t_dir = layer_reflectance_transmittance(
            jnp.asarray(t, jnp.float32), jnp.full_like(tau, s),
            jnp.full_like(tau, a), mu0)
        total = r_dir + t_dir + jnp.exp(-jnp.asarray(t, jnp.float32) / mu0)
        np.testing.assert_allclose(np.asarray(total), 1.0, atol=1e-6)

    @pytest.mark.parametrize("tau", [10.0, 82.0, 300.0])
    @pytest.mark.parametrize("mu0", [0.2, 0.5, 1.0])
    def test_thick_conservative_cloud_has_the_two_stream_albedo(self, tau, mu0):
        """A non-absorbing cloud reflects the closed-form two-stream albedo.

        For ``ssa = 1`` Meador & Weaver (1980) give, over a black surface,
        ``R = [gamma1 tau + (gamma3 - gamma1 mu0)(1 - exp(-tau/mu0))]
        / (1 + gamma1 tau)`` with the delta-Eddington ``gamma``. The #855
        reproduction (tau = 82, g = 0.85, overhead sun) reflected 4e-36 of
        1370 W/m2; the exact value is 0.878.
        """
        g = 0.85
        ts, _, gs = _delta_scaled(tau, 1.0, g)
        g1, _, g3, _ = _eddington_gammas(1.0, gs, mu0)
        expected = (g1 * ts + (g3 - g1 * mu0) * (1.0 - np.exp(-ts / mu0))) / (
            1.0 + g1 * ts)
        for n in (1, self.NLEV):
            up, _, _, _ = self._fluxes(np.full(n, tau / n), np.ones(n),
                                       np.full(n, g), mu0, 0.0)
            assert float(up[0]) == pytest.approx(expected, rel=1e-5)
        if tau >= 82.0:
            assert expected > 0.6

    def test_issue_855_reproduction(self):
        """The #855 case: tau = 82, ssa = 1, g = 0.85, mu0 = 1, 1370 W/m2."""
        up, down, _, _ = self._fluxes(np.array([82.0]), np.array([1.0]),
                                      np.array([0.85]), 1.0, 0.0, toa=1370.0)
        assert float(up[0]) / 1370.0 == pytest.approx(0.8778, abs=1e-3)
        assert float(up[0] + down[-1]) == pytest.approx(1370.0, rel=1e-6)

    @pytest.mark.parametrize("ssa", [0.9, 0.999, 1.0])
    @pytest.mark.parametrize("tau", [0.3, 5.0, 60.0])
    def test_adding_matches_one_homogeneous_layer(self, ssa, tau):
        """Ten sublayers joined by adding equal the single-layer solution.

        The two-stream solution of a homogeneous slab is exact, so the adding
        method must reproduce it from any subdivision — a check on the adding
        recurrences and on the direct-beam sources together. (Below
        ``ssa ~ 0.4`` the Eddington diffuse reflectance is clipped at 0, which
        breaks this identity at the 1e-3 level by design; see
        ``layer_reflectance_transmittance``.)
        """
        for albedo in (0.0, 0.3):
            one = self._fluxes(np.array([tau]), np.array([ssa]),
                               np.array([0.85]), 0.6, albedo)
            ten = self._column(tau, ssa, 0.6, albedo)
            for a, b in zip(one, ten):
                np.testing.assert_allclose(
                    [float(a[0]), float(a[-1])], [float(b[0]), float(b[-1])],
                    atol=2e-6)

    @pytest.mark.parametrize("ssa", SSA)
    @pytest.mark.parametrize("mu0", MU0)
    def test_reflectance_increases_with_optical_depth(self, ssa, mu0):
        """Over a black surface, a thicker cloud never reflects less.

        Exactly monotone wherever the Eddington diffuse reflectance is
        non-negative. At ``ssa = 0.5, g = 0.85`` the delta-scaled layer has
        ``ssa' = 0.22 < 1/(4 - 3 g')``, where the Eddington closure's
        ``gamma2`` — and with it the exact solution's diffuse reflectance — is
        negative (the artefact ``layer_reflectance_transmittance`` clips for
        the diffuse field). The direct-beam solution carries it, so R peaks
        and then settles 0.3 % lower onto its semi-infinite value; that
        closure artefact is bounded here rather than hidden.
        """
        taus = np.logspace(-3, 3, 61)
        r = np.array([
            float(self._fluxes(np.array([t]), np.array([ssa]),
                               np.array([0.85]), mu0, 0.0)[0][0])
            for t in taus])
        if ssa >= 0.9:
            assert np.all(np.diff(r) >= -1e-7), np.diff(r).min()
        else:
            assert np.all(np.diff(r) >= -5e-3 * r.max()), np.diff(r).min()
        assert r[-1] > r[0]


class TestDirectBeamLayerSolution:
    """``R_dir``/``T_dir`` against an independent float64 reference."""

    @pytest.mark.parametrize(
        "tau, ssa, g, mu0",
        [(0.3, 0.9, 0.5, 0.6),     # lambda^2 below the form switch
         (2.0, 0.99, 0.46, 1.0),
         (5.0, 0.5, 0.3, 0.4),     # above the switch
         (3.0, 0.2, 0.4, 0.8),
         (1.2, 0.3, 0.2, 0.62),
         (40.0, 0.95, 0.45, 0.3)],
    )
    def test_matches_textbook_solution(self, tau, ssa, g, mu0):
        _, _, r_dir, t_dir = layer_reflectance_transmittance(
            jnp.array([tau]), jnp.array([ssa]), jnp.array([g]), mu0)
        r_ref, t_ref = _direct_reference(tau, ssa, g, mu0)
        assert float(r_dir[0]) == pytest.approx(r_ref, rel=1e-5, abs=1e-7)
        assert float(t_dir[0]) == pytest.approx(t_ref, rel=1e-5, abs=1e-7)

    def test_continuous_through_the_resonance(self):
        """At ``lambda = 1/mu0`` the ratio form is 0/0; the code is not.

        The value at the resonance must lie on the smooth curve through the
        reference on either side of it, and its derivative must be finite.
        """
        ssa, g, tau = 0.2, 0.4, 1.0
        g1, g2, _, _ = _eddington_gammas(ssa, g, 1.0)
        mu_res = 1.0 / np.sqrt(g1 * g1 - g2 * g2)
        lo = np.array(_direct_reference(tau, ssa, g, mu_res * (1 - 1e-3)))
        hi = np.array(_direct_reference(tau, ssa, g, mu_res * (1 + 1e-3)))

        def layer(mu0):
            _, _, r, t = layer_reflectance_transmittance(
                jnp.array([tau]), jnp.array([ssa]), jnp.array([g]), mu0)
            return jnp.stack([r[0], t[0]])

        at = np.asarray(layer(jnp.float32(mu_res)))
        np.testing.assert_allclose(at, 0.5 * (lo + hi), rtol=1e-5)
        d = jax.jacfwd(layer)(jnp.float32(mu_res))
        assert jnp.all(jnp.isfinite(d))
        np.testing.assert_allclose(
            np.asarray(d), (hi - lo) / (2e-3 * mu_res), rtol=1e-2)

    def test_continuous_across_the_form_switch(self):
        """The two exact forms agree where ``lambda**2`` crosses 1/4."""
        g, tau, mu0 = 0.45, 3.0, 0.7
        # Solve 3 (1 - ssa)(1 - ssa g) = 0.25 for ssa.
        ssa_switch = ((1 + g) - np.sqrt((1 + g) ** 2
                                        - 4 * g * (1 - 0.25 / 3))) / (2 * g)
        vals = []
        for ssa in (ssa_switch * (1 - 1e-6), ssa_switch * (1 + 1e-6)):
            _, _, r, t = layer_reflectance_transmittance(
                jnp.array([tau]), jnp.array([ssa]), jnp.array([g]), mu0)
            vals.append((float(r[0]), float(t[0])))
        np.testing.assert_allclose(vals[0], vals[1], rtol=1e-5)


class TestShortwaveGradients:
    """Gradients of the shortwave solve across the conservation sweep."""

    NLEV = 4

    def _f(self, mu0):
        def f(tau, ssa, asym, toa, albedo):
            return shortwave_fluxes(
                OpticalProperties(optical_depth=tau, single_scatter_albedo=ssa,
                                  asymmetry_factor=asym),
                mu0, toa, albedo, 1)
        return f

    def _args(self, tau, ssa, g=0.85):
        n = self.NLEV
        return (jnp.full((n, 1), tau / n, jnp.float32),
                jnp.full((n, 1), ssa, jnp.float32),
                jnp.full((n, 1), g, jnp.float32),
                jnp.array([430.0]), jnp.array([0.2]))

    def test_gradients_finite_over_the_sweep(self):
        """Reverse mode is finite everywhere.

        Including ``ssa = 1`` exactly, ``ssa = 1 - 1e-7`` (``lambda -> 0``),
        the thickest layers and the lowest sun.
        """
        ssas = (0.5, 0.9, 0.999, 1.0 - 1e-7, 1.0)
        for mu0 in TestShortwaveEnergyConservation.MU0:
            f = self._f(mu0)

            def scalar(*args):
                up, down, _, _ = f(*args)
                return jnp.sum(up) + jnp.sum(down)

            grad = jax.jit(jax.grad(scalar, argnums=(0, 1, 2, 3, 4)))
            for ssa in ssas:
                for tau in TestShortwaveEnergyConservation.TAU:
                    for name, gr in zip(("tau", "ssa", "g", "toa", "albedo"),
                                        grad(*self._args(tau, ssa))):
                        assert jnp.all(jnp.isfinite(gr)), (
                            f"d/d{name} not finite at ssa={ssa} tau={tau} "
                            f"mu0={mu0}")

    @pytest.mark.parametrize("ssa", [1.0, 1.0 - 1e-7])
    def test_ssa_derivative_at_the_pure_forward_corner(self, ssa):
        """``g = 1``: d(surface SW down)/d(ssa) equals the one-sided difference.

        At ``ssa = g = 1`` the delta scaling makes the layer transparent and
        its scaled ssa has no unique limit (0 along ``g = 1``, 1 along
        ``ssa = 1``). The corner takes the ``g = 1`` limit so the ssa
        derivative, taken from below as ssa cannot exceed 1, is the true one
        (it had followed the ``ssa' = 1`` branch: 0.84 against 3.33). The
        ``tau = 0`` subgradient of the old reflectance clip is also gone.
        The g-derivative at this exact corner is direction-dependent and is
        not asserted.
        """
        n = self.NLEV

        def surface_down(s):
            _, down, _, _ = shortwave_fluxes(
                OpticalProperties(
                    optical_depth=jnp.full((n, 1), 0.5, jnp.float32),
                    single_scatter_albedo=jnp.full((n, 1), 1.0, jnp.float32)
                    * s,
                    asymmetry_factor=jnp.ones((n, 1), jnp.float32)),
                0.6, jnp.array([1.0]), jnp.array([0.2]), 1)
            return down[-1, 0]

        ad = float(jax.grad(surface_down)(jnp.float32(ssa)))
        h = 1e-3
        fd = (float(surface_down(jnp.float32(ssa)))
              - float(surface_down(jnp.float32(ssa - h)))) / h
        assert np.isfinite(ad)
        assert ad == pytest.approx(fd, rel=2e-3)

    @pytest.mark.parametrize(
        "tau, ssa, mu0",
        [(10.0, 1.0, 0.5),        # conservative limit, lambda = 0
         (1.0, 0.999, 1.0),
         (100.0, 0.9, 0.1),
         (0.1, 0.5, 0.5)])
    def test_matches_a_central_difference(self, tau, ssa, mu0):
        check_gradients(self._f(mu0), self._args(tau, ssa), rtol=2e-3)
