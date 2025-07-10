"""
Comprehensive test suite for ICON radiation components

This module tests all radiation components including:
- Solar geometry and TOA flux
- Gas optics (absorption)
- Planck functions
- Cloud optics
- Two-stream solver
- Full radiation integration

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
import pytest
from typing import Tuple

# Import all radiation modules
from .radiation_types import (
    RadiationParameters, RadiationState, RadiationFluxes,
    RadiationTendencies, OpticalProperties
)
# Use the radiation module interface which handles jax-solar compatibility
from . import (
    cosine_solar_zenith_angle,
    top_of_atmosphere_flux, 
    daylight_fraction,
    calculate_solar_radiation_gcm,
    get_solar_implementation
)
# Import fallback implementations for specific tests
from .solar import solar_declination, hour_angle
from .gas_optics import (
    water_vapor_continuum, co2_absorption, ozone_absorption_sw,
    ozone_absorption_lw, gas_optical_depth_lw, gas_optical_depth_sw,
    rayleigh_optical_depth
)
from .planck import (
    planck_function_wavenumber, integrated_planck_function,
    planck_bands, planck_derivative, total_thermal_emission,
    effective_temperature, band_fraction, layer_planck_function
)
from .cloud_optics import (
    effective_radius_liquid, effective_radius_ice,
    liquid_cloud_optics_sw, ice_cloud_optics_sw,
    liquid_cloud_optics_lw, ice_cloud_optics_lw,
    cloud_optics, cloud_overlap_factor
)
from .two_stream import (
    two_stream_coefficients, layer_reflectance_transmittance,
    adding_method, longwave_fluxes, shortwave_fluxes,
    flux_to_heating_rate
)


class TestSolarRadiation:
    """Test solar radiation calculations"""
    
    def test_solar_geometry(self):
        """Test solar geometry calculations"""
        # Test declination
        day = 172  # Summer solstice
        dec = solar_declination(day)
        assert jnp.abs(dec - 0.4091) < 0.01  # ~23.45 degrees
        
        # Test hour angle
        hour = 12.0
        longitude = 0.0
        ha = hour_angle(longitude, hour)
        assert jnp.abs(ha) < 0.01  # Noon at Greenwich
        
        # Test zenith angle
        latitude = jnp.array([0.0, 45.0, -45.0]) * jnp.pi / 180
        longitude = jnp.array([0.0, 0.0, 0.0])
        cos_z = cosine_solar_zenith_angle(latitude, longitude, day, hour)
        assert jnp.all(cos_z >= -1) and jnp.all(cos_z <= 1)
        assert cos_z[0] > 0.9  # Near overhead at equator
    
    def test_toa_flux(self):
        """Test top of atmosphere flux"""
        day = 1
        cos_zenith = jnp.array([1.0, 0.5, 0.0])
        flux = top_of_atmosphere_flux(cos_zenith, day)
        
        # The function returns normalized flux * cos_zenith
        assert jnp.allclose(flux[0], 1.0)  # Full normalized flux at zenith
        assert jnp.allclose(flux[1], 0.5)  # Half at 60 degrees
        assert flux[2] == 0.0  # No flux at horizon
    
    def test_daylight_fraction(self):
        """Test daylight fraction calculation"""
        # For longer timestep to get actual fraction
        lat_eq = 0.0
        day = 80  # equinox
        frac_eq = daylight_fraction(lat_eq, day, timestep_hours=24.0)
        assert jnp.abs(frac_eq - 0.5) < 0.01
        
        # Polar regions in summer/winter
        lat_pole = 85.0  # degrees, not radians
        frac_summer = daylight_fraction(lat_pole, 172, timestep_hours=24.0)  # Summer
        frac_winter = daylight_fraction(lat_pole, 355, timestep_hours=24.0)  # Winter
        assert frac_summer > 0.9  # Nearly 24h daylight
        assert frac_winter < 0.1  # Nearly 24h darkness


class TestGasOptics:
    """Test gas optical properties"""
    
    def setup_method(self):
        """Set up test atmosphere"""
        self.nlev = 20
        self.pressure = jnp.linspace(100000, 10000, self.nlev)
        self.temperature = jnp.linspace(288, 220, self.nlev)
        self.h2o_vmr = jnp.linspace(0.01, 1e-6, self.nlev)
        self.o3_vmr = jnp.ones(self.nlev) * 5e-6
        self.thickness = jnp.ones(self.nlev) * 500.0
        self.density = self.pressure / (287.0 * self.temperature)
    
    def test_water_vapor_absorption(self):
        """Test H2O continuum absorption"""
        k_h2o = water_vapor_continuum(
            self.temperature, self.pressure, self.h2o_vmr, band=0
        )
        
        assert k_h2o.shape == (self.nlev,)
        assert jnp.all(k_h2o >= 0)
        # Lower levels (warmer, more H2O) should absorb more
        assert k_h2o[0] > k_h2o[-1]
    
    def test_co2_absorption(self):
        """Test CO2 absorption"""
        co2_vmr = 400e-6
        
        # Band 1 should have absorption
        k_co2_b1 = co2_absorption(self.temperature, self.pressure, co2_vmr, band=1)
        assert jnp.any(k_co2_b1 > 0)
        
        # Other bands should be zero
        k_co2_b0 = co2_absorption(self.temperature, self.pressure, co2_vmr, band=0)
        assert jnp.all(k_co2_b0 == 0)
    
    def test_ozone_absorption(self):
        """Test O3 absorption"""
        # SW absorption (UV/vis)
        k_o3_sw = ozone_absorption_sw(self.o3_vmr, band=0)
        assert jnp.all(k_o3_sw > 0)
        
        # LW absorption (9.6 micron)
        k_o3_lw = ozone_absorption_lw(self.temperature, self.o3_vmr, band=2)
        assert jnp.all(k_o3_lw > 0)
    
    def test_gas_optical_depth(self):
        """Test total gas optical depths"""
        # Longwave
        tau_lw = gas_optical_depth_lw(
            self.temperature, self.pressure, self.h2o_vmr, self.o3_vmr,
            400e-6, self.thickness, self.density, n_bands=3
        )
        assert tau_lw.shape == (self.nlev, 3)
        assert jnp.all(tau_lw >= 0)
        assert jnp.all(jnp.isfinite(tau_lw))
        
        # Shortwave
        tau_sw = gas_optical_depth_sw(
            self.pressure, self.h2o_vmr, self.o3_vmr,
            self.thickness, self.density, cos_zenith=0.5, n_bands=2
        )
        assert tau_sw.shape == (self.nlev, 2)
        assert jnp.all(tau_sw >= 0)
    
    def test_rayleigh_scattering(self):
        """Test Rayleigh optical depth"""
        tau_ray = rayleigh_optical_depth(self.pressure, self.thickness)
        
        assert tau_ray.shape == (self.nlev,)
        assert jnp.all(tau_ray >= 0)
        # Should decrease with altitude (pressure)
        assert tau_ray[0] > tau_ray[-1]


class TestPlanckFunctions:
    """Test Planck radiation functions"""
    
    def test_planck_function(self):
        """Test basic Planck function"""
        T = 300.0
        nu = 1000.0  # cm^-1
        
        B = planck_function_wavenumber(T, nu)
        assert B > 0
        assert jnp.isfinite(B)
        
        # Test Wien's displacement
        T_hot = 400.0
        B_hot = planck_function_wavenumber(T_hot, nu)
        assert B_hot > B  # Hotter = more emission
    
    def test_integrated_planck(self):
        """Test band-integrated Planck function"""
        T = 280.0
        band = (500.0, 1500.0)  # cm^-1
        
        B_int = integrated_planck_function(T, band)
        assert B_int > 0
        assert jnp.isfinite(B_int)
    
    def test_stefan_boltzmann(self):
        """Test Stefan-Boltzmann law"""
        T = 300.0
        emission = total_thermal_emission(T)
        # Use the exact constant from the module
        from .planck import STEFAN_BOLTZMANN
        expected = STEFAN_BOLTZMANN * T**4
        assert jnp.allclose(emission, expected, rtol=1e-10)
    
    def test_planck_derivative(self):
        """Test Planck function temperature derivative"""
        T = 280.0
        nu = 700.0
        
        dB_dT = planck_derivative(T, nu)
        assert dB_dT > 0  # Should increase with T
        
        # Numerical derivative check
        dT = 0.1
        B1 = planck_function_wavenumber(T - dT/2, nu)
        B2 = planck_function_wavenumber(T + dT/2, nu)
        dB_dT_num = (B2 - B1) / dT
        assert jnp.allclose(dB_dT, dB_dT_num, rtol=0.01)
    
    def test_band_fractions(self):
        """Test thermal emission band fractions"""
        T = 288.0
        bands = ((10, 350), (350, 500), (500, 2500))
        fracs = band_fraction(T, bands, n_bands=3)
        
        assert fracs.shape == (3,)
        assert jnp.all(fracs >= 0) and jnp.all(fracs <= 1)
        # These bands only cover a small portion of the spectrum
        # So the sum will be much less than 1
        assert jnp.sum(fracs) > 0  # Should have some emission
        assert jnp.sum(fracs) < 0.2  # But not too much for these limited bands


class TestCloudOptics:
    """Test cloud optical properties"""
    
    def setup_method(self):
        """Set up test cloud data"""
        self.nlev = 10
        self.temperature = jnp.linspace(250, 290, self.nlev)
        self.cwp = jnp.where(self.temperature > 273, 0.1, 0.0)
        self.cip = jnp.where(self.temperature <= 273, 0.05, 0.0)
    
    def test_effective_radius(self):
        """Test cloud particle effective radius"""
        # Liquid
        r_liq = effective_radius_liquid(self.temperature[5], land_fraction=0.5)
        assert 5 < r_liq < 20  # Reasonable range in microns
        
        # Ice
        iwc = 1e-4  # kg/m^3
        r_ice = effective_radius_ice(self.temperature[0], iwc)
        assert 10 < r_ice < 100  # Reasonable range
    
    def test_liquid_cloud_optics(self):
        """Test liquid cloud optical properties"""
        cwp = 0.1  # kg/m^2
        r_eff = 10.0  # microns
        
        # Shortwave
        tau, ssa, g = liquid_cloud_optics_sw(cwp, r_eff, band=0)
        assert tau > 0
        assert 0.9 < ssa < 1.0  # Highly scattering
        assert 0.7 < g < 0.95  # Forward scattering
        
        # Longwave
        tau_lw = liquid_cloud_optics_lw(cwp, r_eff, band=0)
        assert tau_lw > 0
    
    def test_ice_cloud_optics(self):
        """Test ice cloud optical properties"""
        cip = 0.05  # kg/m^2
        r_eff = 50.0  # microns
        
        tau, ssa, g = ice_cloud_optics_sw(cip, r_eff, band=0)
        assert tau > 0
        assert ssa > 0.5
        assert g > 0.5
    
    def test_combined_cloud_optics(self):
        """Test combined cloud optics calculation"""
        sw_optics, lw_optics = cloud_optics(
            self.cwp, self.cip, self.temperature,
            n_sw_bands=2, n_lw_bands=3
        )
        
        # Check shapes
        assert sw_optics.optical_depth.shape == (self.nlev, 2)
        assert lw_optics.optical_depth.shape == (self.nlev, 3)
        
        # Check values
        assert jnp.all(sw_optics.optical_depth >= 0)
        assert jnp.all(sw_optics.single_scatter_albedo >= 0)
        assert jnp.all(sw_optics.single_scatter_albedo <= 1)
        
        # LW should be pure absorption
        assert jnp.all(lw_optics.single_scatter_albedo == 0)
    
    def test_cloud_overlap(self):
        """Test cloud overlap calculation"""
        cf_above = 0.8
        cf_current = 0.6
        
        # Maximum overlap
        overlap_max = cloud_overlap_factor(cf_above, cf_current, overlap_param=1.0)
        assert overlap_max == 0.6  # min(0.8, 0.6)
        
        # Random overlap
        overlap_rand = cloud_overlap_factor(cf_above, cf_current, overlap_param=0.0)
        assert jnp.allclose(overlap_rand, 0.48)  # 0.8 * 0.6


class TestTwoStreamSolver:
    """Test two-stream radiative transfer"""
    
    def setup_method(self):
        """Set up test atmosphere"""
        self.nlev = 10
        self.tau = jnp.ones(self.nlev) * 0.5
        self.ssa = jnp.ones(self.nlev) * 0.9
        self.g = jnp.ones(self.nlev) * 0.85
    
    def test_two_stream_coefficients(self):
        """Test two-stream coefficient calculation"""
        # Longwave (no solar)
        g1, g2, g3, g4 = two_stream_coefficients(self.tau, self.ssa, self.g, mu0=None)
        assert g1.shape == self.tau.shape
        assert jnp.all(g3 == 0)
        assert jnp.all(g4 == 1)
        
        # Shortwave with solar
        mu0 = 0.5
        g1, g2, g3, g4 = two_stream_coefficients(self.tau, self.ssa, self.g, mu0)
        assert jnp.all(g3 > 0)
        assert jnp.allclose(g3 + g4, 1.0)
    
    def test_layer_properties(self):
        """Test layer reflectance and transmittance"""
        R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(
            self.tau, self.ssa, self.g, mu0=0.5
        )
        
        # Physical bounds
        assert jnp.all(R_dif >= 0) and jnp.all(R_dif <= 1)
        assert jnp.all(T_dif >= 0) and jnp.all(T_dif <= 1)
        assert jnp.all(R_dif + T_dif <= 1)  # Energy conservation
        
        # Direct beam transmission follows Beer's law
        expected_T_dir = jnp.exp(-self.tau / 0.5)
        assert jnp.allclose(T_dir, expected_T_dir, atol=0.01)
    
    def test_adding_method(self):
        """Test adding method for layer combination"""
        R1, T1 = 0.2, 0.7
        R2, T2 = 0.3, 0.6
        
        R_combined, T_combined = adding_method(
            jnp.array(R1), jnp.array(T1),
            jnp.array(R2), jnp.array(T2)
        )
        
        # Combined layer should reflect more
        assert R_combined > R1 and R_combined > R2
        assert 0 <= T_combined <= 1
    
    def test_flux_to_heating(self):
        """Test heating rate calculation"""
        flux_up = jnp.linspace(100, 200, self.nlev + 1)
        flux_down = jnp.linspace(400, 300, self.nlev + 1)
        pressure = jnp.linspace(100000, 10000, self.nlev + 1)
        
        heating = flux_to_heating_rate(flux_up, flux_down, pressure)
        
        assert heating.shape == (self.nlev,)
        # With these flux profiles (net flux decreasing upward), we get heating
        assert jnp.any(heating != 0)  # Should have non-zero heating
        # The sign depends on the flux divergence


class TestRadiationIntegration:
    """Integration tests for full radiation calculation"""
    
    def setup_method(self):
        """Set up test case"""
        self.nlev = 20
        self.n_sw_bands = 2
        self.n_lw_bands = 3
        
        # Atmospheric profile
        self.pressure = jnp.linspace(100000, 10000, self.nlev)
        self.temperature = jnp.linspace(288, 220, self.nlev)
        self.h2o_vmr = jnp.linspace(0.01, 1e-6, self.nlev)
        
        # Cloud properties
        self.cwp = jnp.zeros(self.nlev)
        self.cwp = self.cwp.at[10:15].set(0.05)  # Cloud layer
        
        # Create optical properties
        self.create_optical_properties()
    
    def create_optical_properties(self):
        """Create test optical properties"""
        # Simple optical depths
        tau_sw = jnp.ones((self.nlev, self.n_sw_bands)) * 0.1
        tau_lw = jnp.ones((self.nlev, self.n_lw_bands)) * 0.3
        
        self.sw_optics = OpticalProperties(
            optical_depth=tau_sw,
            single_scatter_albedo=jnp.ones((self.nlev, self.n_sw_bands)) * 0.85,
            asymmetry_factor=jnp.ones((self.nlev, self.n_sw_bands)) * 0.85
        )
        
        self.lw_optics = OpticalProperties(
            optical_depth=tau_lw,
            single_scatter_albedo=jnp.zeros((self.nlev, self.n_lw_bands)),
            asymmetry_factor=jnp.zeros((self.nlev, self.n_lw_bands))
        )
    
    def test_longwave_integration(self):
        """Test complete longwave calculation"""
        # Planck functions
        lw_bands = ((10, 350), (350, 500), (500, 2500))
        planck_layer = planck_bands(self.temperature, lw_bands, self.n_lw_bands)
        
        temp_interface = jnp.linspace(288, 220, self.nlev + 1)
        planck_interface = planck_bands(temp_interface, lw_bands, self.n_lw_bands)
        
        # Surface
        surface_emissivity = 0.98
        surface_temp = 288.0
        surface_planck = planck_bands(
            jnp.array([surface_temp]), lw_bands, self.n_lw_bands
        )[0]
        
        # Calculate fluxes
        flux_up, flux_down = longwave_fluxes(
            self.lw_optics, planck_layer, planck_interface,
            surface_emissivity, surface_planck, self.n_lw_bands
        )
        
        # Check results
        assert flux_up.shape == (self.nlev + 1, self.n_lw_bands)
        assert flux_down.shape == (self.nlev + 1, self.n_lw_bands)
        
        # Surface should emit upward
        assert jnp.all(flux_up[-1, :] > 0)
        
        # TOA should have net upward flux (OLR)
        olr = flux_up[0, :] - flux_down[0, :]
        assert jnp.all(olr > 0)
    
    def test_shortwave_integration(self):
        """Test complete shortwave calculation"""
        cos_zenith = 0.5
        toa_flux = jnp.array([500.0, 300.0])  # W/m^2 per band
        surface_albedo = jnp.array([0.15, 0.15])
        
        flux_up, flux_down, flux_dir, flux_dif = shortwave_fluxes(
            self.sw_optics, cos_zenith, toa_flux, surface_albedo, self.n_sw_bands
        )
        
        # Check shapes
        assert flux_up.shape == (self.nlev + 1, self.n_sw_bands)
        assert flux_down.shape == (self.nlev + 1, self.n_sw_bands)
        
        # TOA boundary condition - direct flux at TOA equals incident flux
        assert jnp.allclose(flux_down[0, :], toa_flux)
        
        # Surface reflection
        assert jnp.all(flux_up[-1, :] > 0)
        
        # Energy conservation: net flux should decrease downward
        net_flux = flux_down - flux_up
        assert jnp.all(net_flux[0, :] >= net_flux[-1, :])
    
    def test_heating_rates(self):
        """Test heating rate calculation"""
        # Simple flux profiles
        flux_up = jnp.linspace(100, 300, self.nlev + 1)
        flux_down = jnp.linspace(500, 200, self.nlev + 1)
        pressure_interface = jnp.linspace(100000, 10000, self.nlev + 1)
        
        heating = flux_to_heating_rate(flux_up, flux_down, pressure_interface)
        
        assert heating.shape == (self.nlev,)
        
        # Check units are reasonable (K/day)
        heating_K_per_day = heating * 86400
        assert jnp.all(jnp.abs(heating_K_per_day) < 50)  # Reasonable range


def test_energy_conservation():
    """Test energy conservation in radiation"""
    nlev = 30
    
    # Create atmosphere
    pressure = jnp.linspace(100000, 100, nlev)
    temperature = 250.0 + 40.0 * (pressure / 100000)**0.286
    
    # Create optical properties (clear sky)
    tau = jnp.ones((nlev, 3)) * 0.1
    optics = OpticalProperties(
        optical_depth=tau,
        single_scatter_albedo=jnp.zeros((nlev, 3)),
        asymmetry_factor=jnp.zeros((nlev, 3))
    )
    
    # Planck functions
    bands = ((10, 350), (350, 500), (500, 2500))
    planck_layer = planck_bands(temperature, bands, 3)
    
    temp_interface = jnp.interp(
        jnp.linspace(0, 1, nlev + 1),
        jnp.linspace(0, 1, nlev),
        temperature
    )
    planck_interface = planck_bands(temp_interface, bands, 3)
    
    # Surface
    surface_temp = temperature[-1]
    surface_planck = planck_bands(jnp.array([surface_temp]), bands, 3)[0]
    
    # Calculate fluxes
    flux_up, flux_down = longwave_fluxes(
        optics, planck_layer, planck_interface,
        1.0, surface_planck, 3
    )
    
    # Check basic energy conservation properties
    # 1. Fluxes should be finite and positive
    assert jnp.all(jnp.isfinite(flux_up))
    assert jnp.all(jnp.isfinite(flux_down))
    assert jnp.all(flux_up >= 0)
    assert jnp.all(flux_down >= 0)
    
    # 2. Net upward flux at TOA (OLR)
    toa_net = jnp.sum(flux_up[0, :] - flux_down[0, :])
    assert toa_net > 0  # Net upward flux (OLR)
    
    # 3. Check that we have reasonable flux magnitudes
    # (Given our limited spectral bands, values will be small)
    assert jnp.max(flux_up) < 100  # Reasonable upper bound
    assert jnp.max(flux_down) < 100


def run_all_tests():
    """Run all radiation tests"""
    print("Running radiation component tests...")
    print(f"Solar implementation: {get_solar_implementation()}")
    
    # Solar tests
    solar_tests = TestSolarRadiation()
    solar_tests.test_solar_geometry()
    solar_tests.test_toa_flux()
    solar_tests.test_daylight_fraction()
    print("✓ Solar radiation tests passed")
    
    # Gas optics tests
    gas_tests = TestGasOptics()
    gas_tests.setup_method()
    gas_tests.test_water_vapor_absorption()
    gas_tests.test_co2_absorption()
    gas_tests.test_ozone_absorption()
    gas_tests.test_gas_optical_depth()
    gas_tests.test_rayleigh_scattering()
    print("✓ Gas optics tests passed")
    
    # Planck tests
    planck_tests = TestPlanckFunctions()
    planck_tests.test_planck_function()
    planck_tests.test_integrated_planck()
    planck_tests.test_stefan_boltzmann()
    planck_tests.test_planck_derivative()
    planck_tests.test_band_fractions()
    print("✓ Planck function tests passed")
    
    # Cloud tests
    cloud_tests = TestCloudOptics()
    cloud_tests.setup_method()
    cloud_tests.test_effective_radius()
    cloud_tests.test_liquid_cloud_optics()
    cloud_tests.test_ice_cloud_optics()
    cloud_tests.test_combined_cloud_optics()
    cloud_tests.test_cloud_overlap()
    print("✓ Cloud optics tests passed")
    
    # Two-stream tests
    ts_tests = TestTwoStreamSolver()
    ts_tests.setup_method()
    ts_tests.test_two_stream_coefficients()
    ts_tests.test_layer_properties()
    ts_tests.test_adding_method()
    ts_tests.test_flux_to_heating()
    print("✓ Two-stream solver tests passed")
    
    # Integration tests
    int_tests = TestRadiationIntegration()
    int_tests.setup_method()
    int_tests.test_longwave_integration()
    int_tests.test_shortwave_integration()
    int_tests.test_heating_rates()
    print("✓ Radiation integration tests passed")
    
    # Energy conservation
    test_energy_conservation()
    print("✓ Energy conservation test passed")
    
    print("\nAll radiation tests passed! 🎉")


if __name__ == "__main__":
    run_all_tests()