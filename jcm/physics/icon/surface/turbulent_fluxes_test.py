"""
Unit tests for turbulent flux calculations.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from jcm.physics.icon.surface.turbulent_fluxes import (
    compute_bulk_richardson_number, compute_stability_functions,
    compute_exchange_coefficients, compute_surface_humidity,
    compute_turbulent_fluxes, compute_surface_resistances,
    compute_surface_diagnostics
)
from jcm.physics.icon.surface.surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing,
    SurfaceFluxes, SurfaceResistances
)


class TestBulkRichardsonNumber:
    """Test bulk Richardson number calculation."""
    
    def test_stable_conditions(self):
        """Test Richardson number for stable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Cold surface, warm air (stable)
        temp_air = jnp.array([290.0, 295.0, 300.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.008
        wind_speed = jnp.ones(ncol) * 5.0
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be positive for stable conditions
        assert jnp.all(ri_bulk > 0)
    
    def test_unstable_conditions(self):
        """Test Richardson number for unstable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Warm surface, cold air (unstable)
        temp_air = jnp.array([280.0, 285.0, 290.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 300.0
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.012
        wind_speed = jnp.ones(ncol) * 5.0
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be negative for unstable conditions
        assert jnp.all(ri_bulk < 0)
    
    def test_neutral_conditions(self):
        """Test Richardson number for neutral conditions."""
        ncol, nsfc_type = 2, 3
        
        # Same temperature (neutral)
        temp_air = jnp.array([290.0, 295.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 290.0
        temp_surface = temp_surface.at[0, :].set(290.0)
        temp_surface = temp_surface.at[1, :].set(295.0)
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.01
        wind_speed = jnp.ones(ncol) * 5.0
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be near zero for neutral conditions
        assert jnp.all(jnp.abs(ri_bulk) < 0.1)
    
    def test_low_wind_conditions(self):
        """Test Richardson number with low wind speed."""
        ncol, nsfc_type = 2, 3
        
        temp_air = jnp.array([290.0, 295.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.008
        wind_speed = jnp.array([0.1, 0.05])  # Very low wind
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be finite (no division by zero)
        assert jnp.all(jnp.isfinite(ri_bulk))
        # Should be large for low wind speeds
        assert jnp.all(ri_bulk > 1.0)


class TestStabilityFunctions:
    """Test stability function calculations."""
    
    def test_stable_stability_functions(self):
        """Test stability functions for stable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Positive Richardson numbers (stable)
        ri_bulk = jnp.ones((ncol, nsfc_type)) * 0.1
        
        phi_h, phi_m = compute_stability_functions(ri_bulk)
        
        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)
        
        # Stability functions should be > 1 for stable conditions
        assert jnp.all(phi_h >= 1.0)
        assert jnp.all(phi_m >= 1.0)
        
        # Check specific values
        expected_phi = 1.0 + 5.0 * 0.1
        assert jnp.allclose(phi_h, expected_phi)
        assert jnp.allclose(phi_m, expected_phi)
    
    def test_unstable_stability_functions(self):
        """Test stability functions for unstable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Negative Richardson numbers (unstable)
        ri_bulk = jnp.ones((ncol, nsfc_type)) * (-0.1)
        
        phi_h, phi_m = compute_stability_functions(ri_bulk)
        
        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)
        
        # Stability functions should be > 1 for unstable conditions (enhance mixing)
        assert jnp.all(phi_h > 1.0)
        assert jnp.all(phi_m > 1.0)
        
        # Should be positive and finite
        assert jnp.all(phi_h > 0.0)
        assert jnp.all(phi_m > 0.0)
        assert jnp.all(jnp.isfinite(phi_h))
        assert jnp.all(jnp.isfinite(phi_m))
    
    def test_neutral_stability_functions(self):
        """Test stability functions for neutral conditions."""
        ncol, nsfc_type = 2, 3
        
        # Zero Richardson numbers (neutral)
        ri_bulk = jnp.zeros((ncol, nsfc_type))
        
        phi_h, phi_m = compute_stability_functions(ri_bulk)
        
        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)
        
        # Should be unity for neutral conditions
        assert jnp.allclose(phi_h, 1.0)
        assert jnp.allclose(phi_m, 1.0)
    
    def test_stability_function_limits(self):
        """Test stability function limits."""
        ncol, nsfc_type = 3, 3
        
        # Very stable conditions
        ri_bulk_stable = jnp.ones((ncol, nsfc_type)) * 1.0
        phi_h_stable, phi_m_stable = compute_stability_functions(ri_bulk_stable)
        
        # Should be limited
        stable_limit = 0.2
        expected_phi_stable = 1.0 + 5.0 * stable_limit
        assert jnp.allclose(phi_h_stable, expected_phi_stable)
        
        # Very unstable conditions
        ri_bulk_unstable = jnp.ones((ncol, nsfc_type)) * (-1.0)
        phi_h_unstable, phi_m_unstable = compute_stability_functions(ri_bulk_unstable)
        
        # Should be finite and positive
        assert jnp.all(jnp.isfinite(phi_h_unstable))
        assert jnp.all(jnp.isfinite(phi_m_unstable))
        assert jnp.all(phi_h_unstable > 0.0)
        assert jnp.all(phi_m_unstable > 0.0)


class TestExchangeCoefficients:
    """Test exchange coefficient calculations."""
    
    def test_exchange_coefficient_calculation(self):
        """Test basic exchange coefficient calculation."""
        ncol, nsfc_type = 3, 3
        
        wind_speed = jnp.array([2.0, 5.0, 10.0])
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd, ch, cq = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat, stability_momentum
        )
        
        assert cd.shape == (ncol, nsfc_type)
        assert ch.shape == (ncol, nsfc_type)
        assert cq.shape == (ncol, nsfc_type)
        
        # Should be positive
        assert jnp.all(cd > 0.0)
        assert jnp.all(ch > 0.0)
        assert jnp.all(cq > 0.0)
        
        # Should increase with wind speed
        assert jnp.all(cd[1, :] > cd[0, :])
        assert jnp.all(cd[2, :] > cd[1, :])
    
    def test_exchange_coefficient_roughness_dependence(self):
        """Test dependence on roughness length."""
        ncol, nsfc_type = 2, 3
        
        wind_speed = jnp.ones(ncol) * 5.0
        roughness_momentum_smooth = jnp.ones((ncol, nsfc_type)) * 1e-4
        roughness_momentum_rough = jnp.ones((ncol, nsfc_type)) * 1e-2
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 1e-4
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd_smooth, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum_smooth, roughness_heat,
            stability_heat, stability_momentum
        )
        
        cd_rough, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum_rough, roughness_heat,
            stability_heat, stability_momentum
        )
        
        # Rougher surface should have higher exchange coefficients
        assert jnp.all(cd_rough > cd_smooth)
    
    def test_exchange_coefficient_stability_dependence(self):
        """Test dependence on stability."""
        ncol, nsfc_type = 2, 3
        
        wind_speed = jnp.ones(ncol) * 5.0
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat_stable = jnp.ones((ncol, nsfc_type)) * 1.5
        stability_momentum_stable = jnp.ones((ncol, nsfc_type)) * 1.5
        stability_heat_unstable = jnp.ones((ncol, nsfc_type)) * 0.8
        stability_momentum_unstable = jnp.ones((ncol, nsfc_type)) * 0.8
        
        cd_stable, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat_stable, stability_momentum_stable
        )
        
        cd_unstable, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat_unstable, stability_momentum_unstable
        )
        
        # Unstable conditions should have higher exchange coefficients
        assert jnp.all(cd_unstable > cd_stable)
    
    def test_minimum_wind_speed(self):
        """Test minimum wind speed handling."""
        ncol, nsfc_type = 2, 3
        params = SurfaceParameters(min_wind_speed=1.0)
        
        wind_speed = jnp.array([0.1, 0.5])  # Below minimum
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd, ch, cq = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, params=params
        )
        
        # Should be finite and positive
        assert jnp.all(jnp.isfinite(cd))
        assert jnp.all(cd > 0.0)
        
        # Should be based on minimum wind speed
        cd_min, _, _ = compute_exchange_coefficients(
            jnp.ones(ncol) * params.min_wind_speed, 
            roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, params=params
        )
        
        assert jnp.allclose(cd, cd_min)


class TestSurfaceHumidity:
    """Test surface humidity calculations."""
    
    def test_surface_humidity_calculation(self):
        """Test surface humidity calculation."""
        ncol, nsfc_type = 3, 3
        
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        pressure = jnp.ones(ncol) * 101325.0
        
        q_surface = compute_surface_humidity(temp_surface, pressure)
        
        assert q_surface.shape == (ncol, nsfc_type)
        assert jnp.all(q_surface > 0.0)
        assert jnp.all(q_surface < 0.1)  # Should be reasonable
    
    def test_surface_humidity_temperature_dependence(self):
        """Test temperature dependence of surface humidity."""
        ncol, nsfc_type = 3, 3
        
        temp_cold = jnp.ones((ncol, nsfc_type)) * 260.0
        temp_warm = jnp.ones((ncol, nsfc_type)) * 300.0
        pressure = jnp.ones(ncol) * 101325.0
        
        q_cold = compute_surface_humidity(temp_cold, pressure)
        q_warm = compute_surface_humidity(temp_warm, pressure)
        
        # Warmer surface should have higher humidity
        assert jnp.all(q_warm > q_cold)
    
    def test_surface_humidity_pressure_dependence(self):
        """Test pressure dependence of surface humidity."""
        ncol, nsfc_type = 2, 3
        
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        pressure_low = jnp.ones(ncol) * 85000.0
        pressure_high = jnp.ones(ncol) * 101325.0
        
        q_low_p = compute_surface_humidity(temp_surface, pressure_low)
        q_high_p = compute_surface_humidity(temp_surface, pressure_high)
        
        # Lower pressure should have higher specific humidity
        assert jnp.all(q_low_p > q_high_p)
    
    def test_surface_humidity_bounds(self):
        """Test surface humidity bounds."""
        ncol, nsfc_type = 3, 3
        
        # Test extreme conditions
        temp_surface = jnp.ones((ncol, nsfc_type)) * 350.0  # Very hot
        pressure = jnp.ones(ncol) * 101325.0
        
        q_surface = compute_surface_humidity(temp_surface, pressure)
        
        # Should be clipped to reasonable bounds
        assert jnp.all(q_surface <= 0.1)  # Max 100 g/kg
        assert jnp.all(q_surface >= 0.0)


class TestTurbulentFluxes:
    """Test turbulent flux calculations."""
    
    def setup_method(self):
        """Set up test data."""
        self.ncol, self.nsfc_type = 3, 3
        
        # Atmospheric state
        self.atmospheric_state = AtmosphericForcing(
            temperature=jnp.array([290.0, 295.0, 300.0]),
            humidity=jnp.array([0.01, 0.012, 0.015]),
            u_wind=jnp.array([5.0, 3.0, 8.0]),
            v_wind=jnp.array([2.0, 4.0, 1.0]),
            pressure=jnp.array([101325.0, 95000.0, 85000.0]),
            sw_downward=jnp.array([300.0, 250.0, 400.0]),
            lw_downward=jnp.array([350.0, 320.0, 380.0]),
            rain_rate=jnp.array([1e-6, 2e-6, 0.0]),
            snow_rate=jnp.array([0.0, 0.0, 1e-7]),
            exchange_coeff_heat=jnp.ones((self.ncol, self.nsfc_type)) * 0.01,
            exchange_coeff_moisture=jnp.ones((self.ncol, self.nsfc_type)) * 0.01,
            exchange_coeff_momentum=jnp.ones((self.ncol, self.nsfc_type)) * 0.01
        )
        
        # Surface state
        self.surface_state = SurfaceState(
            temperature=jnp.array([[280.0, 275.0, 285.0], 
                                  [285.0, 270.0, 290.0], 
                                  [290.0, 265.0, 295.0]]),
            temperature_rad=jnp.array([280.0, 282.0, 287.0]),
            fraction=jnp.array([[0.6, 0.2, 0.2], 
                               [0.4, 0.3, 0.3], 
                               [0.8, 0.1, 0.1]]),
            ocean_temp=jnp.array([280.0, 285.0, 290.0]),
            ocean_u=jnp.zeros(self.ncol),
            ocean_v=jnp.zeros(self.ncol),
            ice_thickness=jnp.ones((self.ncol, 2)) * 2.0,
            ice_temp=jnp.ones((self.ncol, 2)) * 270.0,
            snow_depth=jnp.array([0.0, 0.1, 0.05]),
            soil_temp=jnp.ones((self.ncol, 4)) * 280.0,
            soil_moisture=jnp.ones((self.ncol, 4)) * 0.3,
            vegetation_temp=jnp.array([285.0, 290.0, 295.0]),
            roughness_momentum=jnp.ones((self.ncol, self.nsfc_type)) * 0.01,
            roughness_heat=jnp.ones((self.ncol, self.nsfc_type)) * 0.001,
            albedo_visible_direct=jnp.ones((self.ncol, self.nsfc_type)) * 0.1,
            albedo_visible_diffuse=jnp.ones((self.ncol, self.nsfc_type)) * 0.1,
            albedo_nir_direct=jnp.ones((self.ncol, self.nsfc_type)) * 0.2,
            albedo_nir_diffuse=jnp.ones((self.ncol, self.nsfc_type)) * 0.2
        )
    
    def test_turbulent_flux_calculation(self):
        """Test basic turbulent flux calculation."""
        exchange_coeffs = jnp.ones((self.ncol, self.nsfc_type)) * 0.01
        
        fluxes = compute_turbulent_fluxes(
            self.atmospheric_state, self.surface_state,
            exchange_coeffs, exchange_coeffs, exchange_coeffs
        )
        
        assert isinstance(fluxes, SurfaceFluxes)
        assert fluxes.sensible_heat.shape == (self.ncol, self.nsfc_type)
        assert fluxes.latent_heat.shape == (self.ncol, self.nsfc_type)
        assert fluxes.momentum_u.shape == (self.ncol, self.nsfc_type)
        assert fluxes.momentum_v.shape == (self.ncol, self.nsfc_type)
        assert fluxes.evaporation.shape == (self.ncol, self.nsfc_type)
        
        # Check mean fluxes
        assert fluxes.sensible_heat_mean.shape == (self.ncol,)
        assert fluxes.latent_heat_mean.shape == (self.ncol,)
        assert fluxes.momentum_u_mean.shape == (self.ncol,)
        assert fluxes.momentum_v_mean.shape == (self.ncol,)
        assert fluxes.evaporation_mean.shape == (self.ncol,)
    
    def test_flux_directions(self):
        """Test flux directions make physical sense."""
        exchange_coeffs = jnp.ones((self.ncol, self.nsfc_type)) * 0.01
        
        fluxes = compute_turbulent_fluxes(
            self.atmospheric_state, self.surface_state,
            exchange_coeffs, exchange_coeffs, exchange_coeffs
        )
        
        # Where air is warmer than surface, sensible heat should be negative (downward)
        temp_diff = (self.atmospheric_state.temperature[:, None] - 
                    self.surface_state.temperature)
        
        # Sensible heat flux should have opposite sign to temperature difference
        sensible_sign = jnp.sign(fluxes.sensible_heat)
        temp_diff_sign = jnp.sign(temp_diff)
        
        # They should be the same sign (both use same sign convention)
        assert jnp.allclose(sensible_sign, temp_diff_sign)
    
    def test_flux_magnitude_scaling(self):
        """Test flux magnitude scaling with exchange coefficients."""
        exchange_coeffs_low = jnp.ones((self.ncol, self.nsfc_type)) * 0.005
        exchange_coeffs_high = jnp.ones((self.ncol, self.nsfc_type)) * 0.02
        
        fluxes_low = compute_turbulent_fluxes(
            self.atmospheric_state, self.surface_state,
            exchange_coeffs_low, exchange_coeffs_low, exchange_coeffs_low
        )
        
        fluxes_high = compute_turbulent_fluxes(
            self.atmospheric_state, self.surface_state,
            exchange_coeffs_high, exchange_coeffs_high, exchange_coeffs_high
        )
        
        # Higher exchange coefficients should give higher flux magnitudes
        assert jnp.all(jnp.abs(fluxes_high.sensible_heat) >= jnp.abs(fluxes_low.sensible_heat))
        assert jnp.all(jnp.abs(fluxes_high.latent_heat) >= jnp.abs(fluxes_low.latent_heat))
        assert jnp.all(jnp.abs(fluxes_high.momentum_u) >= jnp.abs(fluxes_low.momentum_u))


if __name__ == "__main__":
    pytest.main([__file__])