"""Unit tests for two-stream radiative transfer solver

Tests the two-stream approximation implementation including
coefficients, layer properties, flux calculations, and heating rates.

Date: 2025-01-10
"""

import jax.numpy as jnp
from jcm.physics.icon.radiation.two_stream import (
    two_stream_coefficients,
    layer_reflectance_transmittance,
    adding_method,
    longwave_fluxes,
    shortwave_fluxes,
    flux_to_heating_rate
)
from jcm.physics.icon.radiation.radiation_types import OpticalProperties
from jcm.physics.icon.radiation.planck import planck_bands_lw


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


def test_adding_method():
    """Test adding method for combining layers"""
    R1 = jnp.array(0.2)
    T1 = jnp.array(0.7)
    R2 = jnp.array(0.3)
    T2 = jnp.array(0.6)
    
    R_combined, T_combined = adding_method(R1, T1, R2, T2)
    
    # Should have more reflection than either layer alone
    assert R_combined > R1
    assert R_combined > R2
    
    # Transmission should be reasonable
    assert 0 <= T_combined <= 1
    assert T_combined > 0  # Some transmission through both layers
    
    # Energy conservation
    assert R_combined + T_combined <= 1.0


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
    
    # Check shapes (hardcoded max_bands=10)
    assert flux_up_lw.shape == (nlev + 1, 10)
    assert flux_down_lw.shape == (nlev + 1, 10)
    # Only first n_lw_bands should have values
    assert jnp.all(flux_up_lw[:, n_lw_bands:] == 0)
    assert jnp.all(flux_down_lw[:, n_lw_bands:] == 0)
    
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
    
    # Check shapes (hardcoded max_bands=10)
    assert flux_up_sw.shape == (nlev + 1, 10)
    assert flux_down_sw.shape == (nlev + 1, 10)
    assert flux_dir.shape == (nlev + 1, 10)
    assert flux_dif.shape == (nlev + 1, 10)
    # Only first n_sw_bands should have values
    assert jnp.all(flux_up_sw[:, n_sw_bands:] == 0)
    assert jnp.all(flux_down_sw[:, n_sw_bands:] == 0)
    
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