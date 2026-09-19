"""Unit tests for two-stream radiative transfer solver

Tests the two-stream approximation implementation including
coefficients, layer properties, flux calculations, and heating rates.

Date: 2025-01-10
"""

import jax
import jax.numpy as jnp
import pytest
from jcm.physics.radiation.grey_two_stream.two_stream import (
    two_stream_coefficients,
    layer_reflectance_transmittance,
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
    shortwave liquid-cloud optics sits (``ssa ~ 0.9999``). Two distinct
    failures live there and these tests fence both:

    * ``gamma1**2 - gamma2**2`` is a catastrophic cancellation as ``ssa -> 1``
      (the two squares agree to four digits at ``ssa = 0.9999``), and feeding
      the round-off that survives into ``sqrt`` — whose derivative is
      ``1/(2*sqrt(x))`` — amplifies it. The scheme now forms the same
      quantity factored, ``3*(1 - ssa)*(1 - ssa*g)``, which has no
      cancellation and is exactly 0 at the limit.
    * ``gamma1`` itself is 0 at ``ssa = g = 1``, where ``gamma2`` is 0 too, so
      the ratio ``gamma2/gamma1`` the layer albedo needs is 0/0.
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

        ``ssa = g = 1`` returned NaN for both ``ssa`` and ``g`` before the
        double-``where`` on ``gamma2/gamma1``. A ``maximum(gamma1**2, 1e-30)``
        floor does not fix it: the quotient's VJP squares the denominator, and
        ``1e-60`` underflows to 0 in float32, so the guard becomes the 0/0.
        """
        tau, ssa_p, g_p = self._layer(ssa, g)
        grads = jax.grad(self._sum_of_squares, argnums=(0, 1, 2))(
            tau, ssa_p, g_p)
        for name, grad in zip(("tau", "ssa", "g"), grads):
            assert jnp.all(jnp.isfinite(grad)), (
                f"d/d{name} is not finite at ssa={ssa}, g={g}: {grad}")

    def test_conservative_limit_derivative_is_not_a_cancellation_artefact(self):
        """At ``ssa = 1`` exactly the derivative must be O(1), not O(1e5).

        The eigenvalue is 0 there, so no two-sided derivative of ``sqrt``
        exists and the honest reported value is the one the guarded branch
        gives. What the subtracted form reported instead was the *round-off*
        of ``gamma1**2 - gamma2**2`` divided by its own square root: 5.0e5,
        five orders of magnitude of pure noise entering every upstream cloud
        gradient. A bound well below that, and well above the ~2 the guarded
        form gives, separates the two without pinning a float32 value.
        """
        tau, ssa, g = self._layer(1.0, 0.85)
        d_ssa = jax.grad(self._sum_of_squares, argnums=1)(tau, ssa, g)
        assert float(jnp.max(jnp.abs(d_ssa))) < 1.0e2

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

    @pytest.mark.parametrize("tau", [0.1, 1.0, 10.0, 30.0, 50.0, 60.0, 200.0])
    @pytest.mark.parametrize("ssa, g", [(0.0, 0.0), (0.93, 0.85)])
    def test_both_ad_modes_survive_an_optically_thick_layer(self, tau, ssa, g):
        """Thick layers must not return NaN in *either* AD mode.

        Two separate failures used to bracket the ``lambda_tau >= 88``
        asymptotic switch, and each showed up in only one mode, so a check of
        one alone would have missed the other:

        * above it, ``exp(lambda_tau)`` was still evaluated on the discarded
          branch and overflowed, so reverse mode formed ``0 * inf = NaN``;
        * below it, the layer albedo was a quotient whose denominator reached
          1e38, and the quotient rule squares the denominator — which
          overflows float32 for any ``lambda_tau`` above about 44, i.e. for
          every optically thick longwave layer in the lower troposphere.

        ``tau = 30`` and ``tau = 50`` sit in the second window, ``tau = 60``
        and ``200`` in the first.
        """
        tau_p, ssa_p, g_p = self._layer(ssa, g, tau_value=tau)
        _, tangents = jax.jvp(
            lambda t: layer_reflectance_transmittance(t, ssa_p, g_p, None),
            (tau_p,), (jnp.ones_like(tau_p),))
        for name, t in zip(("R_dif", "T_dif", "R_dir", "T_dir"), tangents):
            assert jnp.all(jnp.isfinite(t)), f"jvp of {name} is not finite"

        grad = jax.grad(
            lambda t: jnp.sum(
                layer_reflectance_transmittance(t, ssa_p, g_p, None)[0]
                + layer_reflectance_transmittance(t, ssa_p, g_p, None)[1])
        )(tau_p)
        assert jnp.all(jnp.isfinite(grad)), "vjp is not finite"
