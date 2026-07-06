"""Unit tests for turbulent flux calculations."""

import pytest
import jax.numpy as jnp

from jcm.physics.surface.echam.turbulent_fluxes import (
    compute_bulk_richardson_number, compute_stability_functions,
    compute_exchange_coefficients, compute_surface_humidity
)
from jcm.physics.surface.echam.surface_types import SurfaceParameters


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
        """Test stability functions for unstable conditions.

        Businger-Dyer Φ_m, Φ_h appear in the denominator of the bulk
        exchange coefficients (``CH = κ²/(ln·Φ_m·Φ_h)·…``). To enhance
        turbulent mixing under unstable buoyancy (the standard textbook
        result for free convection), they must be **< 1** for ζ < 0.
        """
        ncol, nsfc_type = 3, 3

        # Negative Richardson numbers (unstable)
        ri_bulk = jnp.ones((ncol, nsfc_type)) * (-0.1)

        phi_h, phi_m = compute_stability_functions(ri_bulk)

        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)

        # Stability functions must be < 1 under unstable conditions so
        # that CH = κ²/(ln·Φ_m·Φ_h) is *larger* than neutral — the
        # boundary-layer enhancement of bulk exchange.
        assert jnp.all(phi_h < 1.0)
        assert jnp.all(phi_m < 1.0)
        # Finite & positive guards.
        assert jnp.all(phi_h > 0.0)
        assert jnp.all(phi_m > 0.0)
        assert jnp.all(jnp.isfinite(phi_h))
        assert jnp.all(jnp.isfinite(phi_m))
        # Specific values: ζ ≈ Ri = -0.1 → Φ_h = (1 + 16·0.1)^(-1/2)
        # ≈ 0.620, Φ_m = (1 + 16·0.1)^(-1/4) ≈ 0.787.
        expected_phi_h = (1.0 + 16.0 * 0.1) ** (-0.5)
        expected_phi_m = (1.0 + 16.0 * 0.1) ** (-0.25)
        assert jnp.allclose(phi_h, expected_phi_h, rtol=1e-4)
        assert jnp.allclose(phi_m, expected_phi_m, rtol=1e-4)
    
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
            stability_heat, stability_momentum, min_wind_speed=1.0, von_karman=0.4
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
            stability_heat, stability_momentum, min_wind_speed=1.0, von_karman=0.4
        )
        
        cd_rough, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum_rough, roughness_heat,
            stability_heat, stability_momentum, min_wind_speed=1.0, von_karman=0.4
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
            stability_heat_stable, stability_momentum_stable,
            min_wind_speed=1.0, von_karman=0.4
        )
        
        cd_unstable, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat_unstable, stability_momentum_unstable,
            min_wind_speed=1.0, von_karman=0.4
        )
        
        # Unstable conditions should have higher exchange coefficients
        assert jnp.all(cd_unstable > cd_stable)
    
    def test_minimum_wind_speed(self):
        """Test minimum wind speed handling."""
        ncol, nsfc_type = 2, 3
        params = SurfaceParameters.default(min_wind_speed=1.0)
        
        wind_speed = jnp.array([0.1, 0.5])  # Below minimum
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd, ch, cq = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, params.min_wind_speed, params.von_karman
        )
        
        # Should be finite and positive
        assert jnp.all(jnp.isfinite(cd))
        assert jnp.all(cd > 0.0)
        
        # Should be based on minimum wind speed
        cd_min, _, _ = compute_exchange_coefficients(
            jnp.ones(ncol) * params.min_wind_speed, 
            roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, params.min_wind_speed, params.von_karman
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


if __name__ == "__main__":
    pytest.main([__file__])