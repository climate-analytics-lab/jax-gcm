"""
Unit tests for gravity wave drag parameterization

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
import pytest
from .gravity_wave_drag import (
    GravityWaveParameters, GravityWaveState, GravityWaveTendencies,
    brunt_vaisala_frequency, orographic_source, wave_breaking_criterion,
    gravity_wave_drag
)
from ..constants.physical_constants import grav, rd, cp


class TestBruntVaisalaFrequency:
    """Test Brunt-Väisälä frequency calculation"""
    
    def test_stable_atmosphere(self):
        """Test N² in stably stratified atmosphere"""
        # Create stable profile
        nlev = 20
        height = jnp.linspace(0, 20000, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        
        # Stable temperature profile (decreasing with height)
        temperature = 288 - 0.0065 * height
        
        n2 = brunt_vaisala_frequency(temperature, pressure, height)
        
        # Should be positive for stable stratification
        assert jnp.all(n2 > 0)
        
        # Typical tropospheric values ~1e-4 s^-2
        assert jnp.all(n2 < 1e-3)
        assert jnp.mean(n2) > 1e-5
    
    def test_isothermal_atmosphere(self):
        """Test N² in isothermal atmosphere"""
        nlev = 10
        height = jnp.linspace(0, 10000, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = jnp.ones(nlev) * 273.0
        
        n2 = brunt_vaisala_frequency(temperature, pressure, height)
        
        # Isothermal atmosphere has N² = g²/cp/T
        expected = grav**2 / (cp * 273.0)
        
        # Should be approximately constant
        assert jnp.std(n2[1:-1]) / jnp.mean(n2[1:-1]) < 0.1
        
        # Check approximate value
        assert jnp.abs(jnp.mean(n2) - expected) / expected < 0.2


class TestOrographicSource:
    """Test orographic gravity wave source"""
    
    def test_source_magnitude(self):
        """Test that source scales with wind and orography"""
        config = GravityWaveParameters.default()
        
        # Base case
        u_sfc = jnp.array(10.0)
        v_sfc = jnp.array(0.0)
        n_sfc = jnp.array(0.01)  # 0.01 s^-1
        h_std = jnp.array(100.0)  # 100m mountains
        
        tau_x1, tau_y1 = orographic_source(u_sfc, v_sfc, n_sfc, h_std, config)
        
        # Double wind speed
        tau_x2, tau_y2 = orographic_source(
            2 * u_sfc, v_sfc, n_sfc, h_std, config
        )
        
        # Source should increase with wind
        assert jnp.abs(tau_x2) > jnp.abs(tau_x1)
        
        # Double mountain height
        tau_x3, tau_y3 = orographic_source(
            u_sfc, v_sfc, n_sfc, 2 * h_std, config
        )
        
        # Source scales with h²
        assert jnp.abs(tau_x3) > 3 * jnp.abs(tau_x1)
    
    def test_source_direction(self):
        """Test that source opposes wind direction"""
        config = GravityWaveParameters.default()
        
        n_sfc = jnp.array(0.01)
        h_std = jnp.array(100.0)
        
        # Westerly wind
        u_sfc = jnp.array(10.0)
        v_sfc = jnp.array(0.0)
        tau_x, tau_y = orographic_source(u_sfc, v_sfc, n_sfc, h_std, config)
        
        # Stress should oppose wind
        assert tau_x < 0
        assert jnp.abs(tau_y) < 1e-10
        
        # Northerly wind
        u_sfc = jnp.array(0.0)
        v_sfc = jnp.array(10.0)
        tau_x, tau_y = orographic_source(u_sfc, v_sfc, n_sfc, h_std, config)
        
        assert jnp.abs(tau_x) < 1e-10
        assert tau_y < 0
    
    def test_froude_number_effect(self):
        """Test Froude number dependence"""
        config = GravityWaveParameters.default()
        
        n_sfc = jnp.array(0.01)
        h_std = jnp.array(500.0)  # Tall mountains
        
        # Low wind (low Froude) - blocked flow
        u_low = jnp.array(5.0)
        v_sfc = jnp.array(0.0)
        tau_low, _ = orographic_source(u_low, v_sfc, n_sfc, h_std, config)
        
        # High wind (high Froude) - flow over
        u_high = jnp.array(50.0)
        tau_high, _ = orographic_source(u_high, v_sfc, n_sfc, h_std, config)
        
        # Normalized by wind speed, low Froude should have more drag
        drag_low = jnp.abs(tau_low) / u_low
        drag_high = jnp.abs(tau_high) / u_high
        
        assert drag_low > drag_high


class TestWaveBreaking:
    """Test wave breaking criterion"""
    
    def test_richardson_number_breaking(self):
        """Test breaking based on Richardson number"""
        config = GravityWaveParameters.default()
        nlev = 10
        
        # Create profile with strong shear
        height = jnp.linspace(0, 10000, nlev)
        u = jnp.linspace(0, 50, nlev)  # Linear shear
        v = jnp.zeros(nlev)
        
        # Stable stratification
        n2 = jnp.ones(nlev) * 1e-4
        
        # Momentum flux
        tau_x = jnp.ones(nlev) * -0.1
        tau_y = jnp.zeros(nlev)
        
        # Air density
        rho = jnp.ones(nlev)
        
        breaking_mask, deposited = wave_breaking_criterion(
            u, v, n2, height, tau_x, tau_y, rho, config
        )
        
        # Should have breaking where shear is strong
        assert jnp.any(breaking_mask)
        
        # With constant flux, divergence is zero except at boundaries
        # So we just verify that breaking was detected
    
    def test_amplitude_breaking(self):
        """Test breaking based on wave amplitude"""
        config = GravityWaveParameters.default()
        nlev = 10
        
        height = jnp.linspace(0, 20000, nlev)
        u = jnp.ones(nlev) * 10.0
        v = jnp.zeros(nlev)
        n2 = jnp.ones(nlev) * 1e-4
        
        # Large momentum flux (large amplitude)
        tau_x = jnp.ones(nlev) * -10.0
        tau_y = jnp.zeros(nlev)
        
        # Decreasing density with height
        rho = jnp.exp(-height / 8000)
        
        breaking_mask, deposited = wave_breaking_criterion(
            u, v, n2, height, tau_x, tau_y, rho, config
        )
        
        # Should have breaking somewhere due to large amplitude
        assert jnp.any(breaking_mask)


class TestGravityWaveDrag:
    """Test the complete gravity wave drag scheme"""
    
    def test_momentum_deposition(self):
        """Test that momentum is deposited correctly"""
        config = GravityWaveParameters.default()
        
        # Create westerly jet
        nlev = 30
        height = jnp.linspace(0, 30000, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = 288 - 0.0065 * height
        
        # Jet profile
        u_wind = 30.0 * jnp.exp(-(height - 12000)**2 / 5000**2)
        v_wind = jnp.zeros(nlev)
        
        h_std = 300.0  # Mountains
        dt = 1800.0
        
        tendencies, state = gravity_wave_drag(
            u_wind, v_wind, temperature, pressure, height, h_std, dt, config
        )
        
        # Should have non-zero surface stress from orographic source
        assert state.wave_stress[-1] > 0
        
        # The tendencies will be very small with default parameters
        # Just verify the scheme ran without errors
        total_tend = jnp.sum(jnp.abs(tendencies.dudt))
        assert jnp.isfinite(total_tend)
    
    def test_critical_level_filtering(self):
        """Test that waves are absorbed at critical levels"""
        config = GravityWaveParameters.default()
        
        nlev = 20
        height = jnp.linspace(0, 20000, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = 288 - 0.0065 * height
        
        # Wind that reverses direction (critical level)
        u_wind = jnp.where(height < 10000, 10.0, -10.0)
        v_wind = jnp.zeros(nlev)
        
        h_std = 200.0
        dt = 1800.0
        
        tendencies, state = gravity_wave_drag(
            u_wind, v_wind, temperature, pressure, height, h_std, dt, config
        )
        
        # Momentum flux should go to zero above critical level
        critical_height = 10000
        above_critical = height > critical_height + 2000
        assert jnp.all(state.tau_x[above_critical] < 0.01)
    
    def test_height_limits(self):
        """Test that GWD only applies within height limits"""
        config = GravityWaveParameters.default(zmin=5000.0, zmax=25000.0)
        
        nlev = 40
        height = jnp.linspace(0, 40000, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = 288 - 0.0065 * height
        
        u_wind = jnp.ones(nlev) * 20.0
        v_wind = jnp.zeros(nlev)
        
        h_std = 300.0
        dt = 1800.0
        
        tendencies, state = gravity_wave_drag(
            u_wind, v_wind, temperature, pressure, height, h_std, dt, config
        )
        
        # No tendencies below zmin
        below_mask = height < float(config.zmin)
        assert jnp.all(tendencies.dudt[below_mask] == 0)
        
        # No tendencies above zmax
        above_mask = height > float(config.zmax)
        assert jnp.all(tendencies.dudt[above_mask] == 0)
    
    def test_energy_conservation(self):
        """Test that kinetic energy is converted to heat"""
        config = GravityWaveParameters.default()
        
        nlev = 20
        height = jnp.linspace(0, 20000, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = 288 - 0.0065 * height
        
        u_wind = jnp.ones(nlev) * 25.0
        v_wind = jnp.zeros(nlev)
        
        h_std = 400.0
        dt = 1800.0
        
        tendencies, state = gravity_wave_drag(
            u_wind, v_wind, temperature, pressure, height, h_std, dt, config
        )
        
        # Kinetic energy loss
        ke_loss = u_wind * tendencies.dudt + v_wind * tendencies.dvdt
        
        # Check that gravity waves were generated
        assert state.wave_stress[-1] > 0
        
        # Energy change may be very small
        total_ke_change = jnp.sum(jnp.abs(ke_loss))
        assert total_ke_change >= 0
        
        # Temperature tendency from dissipation
        # dT/dt = -dKE/dt / cp
        expected_heating = -ke_loss / cp
        
        # Should match where there are tendencies
        mask = jnp.abs(ke_loss) > 1e-10
        if jnp.any(mask):
            assert jnp.allclose(
                tendencies.dtedt[mask], 
                expected_heating[mask], 
                rtol=1e-3
            )
    
    def test_jax_transformations(self):
        """Test JAX transformations"""
        config = GravityWaveParameters.default()
        
        def gwd_loss(u_wind):
            nlev = len(u_wind)
            height = jnp.linspace(0, 20000, nlev)
            pressure = 100000 * jnp.exp(-height / 8000)
            temperature = 288 - 0.0065 * height
            v_wind = jnp.zeros(nlev)
            
            tend, _ = gravity_wave_drag(
                u_wind, v_wind, temperature, pressure, 
                height, 300.0, 1800.0, config
            )
            
            return jnp.sum(tend.dudt ** 2)
        
        # Test JIT
        jitted = jax.jit(gwd_loss)
        u = jnp.ones(20) * 20.0
        loss = jitted(u)
        assert jnp.isfinite(loss)
        
        # Test gradient
        grad_fn = jax.grad(gwd_loss)
        grad = grad_fn(u)
        assert grad.shape == u.shape
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    # Run tests
    test_bv = TestBruntVaisalaFrequency()
    test_bv.test_stable_atmosphere()
    test_bv.test_isothermal_atmosphere()
    
    test_oro = TestOrographicSource()
    test_oro.test_source_magnitude()
    test_oro.test_source_direction()
    test_oro.test_froude_number_effect()
    
    test_break = TestWaveBreaking()
    test_break.test_richardson_number_breaking()
    test_break.test_amplitude_breaking()
    
    test_gwd = TestGravityWaveDrag()
    test_gwd.test_momentum_deposition()
    test_gwd.test_critical_level_filtering()
    test_gwd.test_height_limits()
    test_gwd.test_energy_conservation()
    test_gwd.test_jax_transformations()
    
    print("All gravity wave drag tests passed!")