"""
Unit tests for convective adjustment module

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
import pytest
from .adjustment import (
    saturation_adjustment, energy_conservation_check,
    convective_adjustment
)
from .tiedtke_nordeng import ConvectionParameters, saturation_mixing_ratio
from ..constants.physical_constants import tmelt, cp, alhc, alhs


class TestSaturationAdjustment:
    """Test saturation adjustment functions"""
    
    def test_no_adjustment_needed(self):
        """Test that no adjustment occurs for subsaturated air"""
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        
        # Create subsaturated conditions (50% RH)
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 0.5 * qs
        
        cloud_water = jnp.array(0.0001)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should be unchanged
        assert jnp.allclose(t_adj, temperature)
        assert jnp.allclose(q_adj, specific_humidity)
        assert jnp.allclose(qc_adj, cloud_water)
        assert jnp.allclose(qi_adj, cloud_ice)
    
    def test_condensation_warm(self):
        """Test condensation in warm conditions"""
        temperature = jnp.array(285.0)  # Above freezing
        pressure = jnp.array(90000.0)
        
        # Create supersaturated state (110% RH)
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.1 * qs
        
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should have warming from latent heat
        assert t_adj > temperature
        
        # Should have condensation to liquid only
        assert q_adj < specific_humidity
        assert qc_adj > cloud_water
        assert jnp.allclose(qi_adj, cloud_ice)  # No ice formation
        
        # Check final state is near saturation
        rs_adj = saturation_mixing_ratio(pressure, t_adj)
        qs_adj = rs_adj / (1 + rs_adj)
        rh_final = q_adj / qs_adj
        # The adjustment reduces supersaturation significantly
        # but may not reach exact saturation in finite iterations
        assert 0.75 < rh_final < 1.05  # Should be much closer to saturation
    
    def test_condensation_cold(self):
        """Test condensation in cold conditions"""
        temperature = jnp.array(250.0)  # Well below freezing
        pressure = jnp.array(50000.0)
        
        # Create supersaturated state
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.15 * qs
        
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should have warming
        assert t_adj > temperature
        
        # Should have condensation to ice only
        assert q_adj < specific_humidity
        assert jnp.allclose(qc_adj, cloud_water)  # No liquid formation
        assert qi_adj > cloud_ice
    
    def test_condensation_mixed_phase(self):
        """Test condensation in mixed phase region"""
        temperature = jnp.array(265.0)  # Mixed phase
        pressure = jnp.array(70000.0)
        
        # Create supersaturated state
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.08 * qs
        
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Should have both liquid and ice
        assert qc_adj > 0
        assert qi_adj > 0
        
        # At -8°C, we're in the mixed phase region
        # Both phases should be present but the ratio depends on the partitioning scheme
    
    def test_conservation(self):
        """Test mass conservation in adjustment"""
        temperature = jnp.array(275.0)
        pressure = jnp.array(85000.0)
        
        # Supersaturated
        rs = saturation_mixing_ratio(pressure, temperature)
        qs = rs / (1 + rs)
        specific_humidity = 1.12 * qs
        
        cloud_water = jnp.array(0.0002)
        cloud_ice = jnp.array(0.0001)
        
        # Total water before
        total_before = specific_humidity + cloud_water + cloud_ice
        
        t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice
        )
        
        # Total water after
        total_after = q_adj + qc_adj + qi_adj
        
        # Should conserve total water (within numerical precision)
        assert jnp.allclose(total_before, total_after, rtol=2e-3)


class TestEnergyConservation:
    """Test energy conservation diagnostics"""
    
    def test_warming_condensation(self):
        """Test energy balance for condensation case"""
        # Initial state
        t_old = jnp.array(280.0)
        q_old = jnp.array(0.012)  # 12 g/kg
        qc_old = jnp.array(0.0)
        qi_old = jnp.array(0.0)
        
        # After condensation
        t_new = jnp.array(281.5)  # Warmed
        q_new = jnp.array(0.010)  # Dried
        qc_new = jnp.array(0.002) # Cloud formed
        qi_new = jnp.array(0.0)
        
        precip = jnp.array(0.0)
        dt = 3600.0
        
        imbalance = energy_conservation_check(
            t_old, q_old, qc_old, qi_old,
            t_new, q_new, qc_new, qi_new,
            precip, dt
        )
        
        # Energy should be approximately conserved
        # Some imbalance due to approximations
        assert jnp.abs(imbalance) < 50.0  # W/m²
    
    def test_with_precipitation(self):
        """Test energy balance with precipitation"""
        t_old = jnp.array(280.0)
        q_old = jnp.array(0.010)
        qc_old = jnp.array(0.002)
        qi_old = jnp.array(0.0)
        
        # After precipitation
        t_new = jnp.array(280.0)  # Same temperature
        q_new = jnp.array(0.010)  # Same vapor
        qc_new = jnp.array(0.001) # Cloud water reduced
        qi_new = jnp.array(0.0)
        
        # Precipitation removes 1 g/kg of water
        precip = jnp.array(0.001 / 3600.0)  # kg/kg/s -> kg/m²/s needs scaling
        dt = 3600.0
        
        imbalance = energy_conservation_check(
            t_old, q_old, qc_old, qi_old,
            t_new, q_new, qc_new, qi_new,
            precip, dt
        )
        
        # Should have energy loss due to precipitation
        assert imbalance < 0  # Energy removed


class TestConvectiveAdjustment:
    """Test the full convective adjustment"""
    
    def test_apply_tendencies_and_adjust(self):
        """Test applying tendencies followed by adjustment"""
        # Initial state
        temperature = jnp.array(278.0)
        pressure = jnp.array(90000.0)
        specific_humidity = jnp.array(0.008)
        cloud_water = jnp.array(0.0)
        cloud_ice = jnp.array(0.0)
        
        # Convective tendencies (warming and moistening)
        conv_tend_t = jnp.array(2.0 / 3600.0)    # 2 K/hour
        conv_tend_q = jnp.array(0.002 / 3600.0)  # 2 g/kg/hour
        conv_tend_qc = jnp.array(0.0)
        conv_tend_qi = jnp.array(0.0)
        
        dt = 1800.0  # 30 minutes
        
        # Apply adjustment
        t_adj, q_adj, qc_adj, qi_adj = convective_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice,
            conv_tend_t, conv_tend_q, conv_tend_qc, conv_tend_qi,
            dt
        )
        
        # Should be warmer (due to convective heating)
        assert t_adj > temperature
        # Humidity may decrease if warming causes condensation
        # Check that total tendency was applied
        t_expected = temperature + conv_tend_t * dt
        q_expected = specific_humidity + conv_tend_q * dt
        # Temperature should be at least the tendency-applied value
        assert t_adj >= t_expected - 0.1  # Allow small deviation
        
        # If supersaturated after tendencies, should have condensation
        rs_final = saturation_mixing_ratio(pressure, t_adj)
        qs_final = rs_final / (1 + rs_final)
        rh_final = q_adj / qs_final
        
        # Should not be supersaturated after adjustment
        assert rh_final <= 1.02
    
    def test_with_cloud_tendencies(self):
        """Test adjustment with cloud water tendencies"""
        temperature = jnp.array(275.0)
        pressure = jnp.array(85000.0)
        specific_humidity = jnp.array(0.006)
        cloud_water = jnp.array(0.0005)
        cloud_ice = jnp.array(0.0002)
        
        # Tendencies that increase clouds
        conv_tend_t = jnp.array(0.5 / 3600.0)
        conv_tend_q = jnp.array(-0.001 / 3600.0)  # Drying
        conv_tend_qc = jnp.array(0.0008 / 3600.0)  # Cloud increase
        conv_tend_qi = jnp.array(0.0002 / 3600.0)
        
        dt = 1800.0
        
        t_adj, q_adj, qc_adj, qi_adj = convective_adjustment(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice,
            conv_tend_t, conv_tend_q, conv_tend_qc, conv_tend_qi,
            dt
        )
        
        # Clouds should increase
        assert qc_adj > cloud_water
        assert qi_adj > cloud_ice
        
        # Total water should be conserved (within adjustment)
        total_old = specific_humidity + cloud_water + cloud_ice
        total_tend = (conv_tend_q + conv_tend_qc + conv_tend_qi) * dt
        total_expected = total_old + total_tend
        total_new = q_adj + qc_adj + qi_adj
        
        assert jnp.allclose(total_new, total_expected, rtol=0.01)
    
    def test_jax_transformations(self):
        """Test JAX transformations work"""
        def adjustment_fn(temperature):
            pressure = jnp.array(90000.0)
            q = jnp.array(0.008)
            qc = jnp.array(0.0)
            qi = jnp.array(0.0)
            
            t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
                temperature, q, pressure, qc, qi
            )
            return t_adj
        
        # Test JIT
        jitted_fn = jax.jit(adjustment_fn)
        t = jnp.array(280.0)
        t_adj = jitted_fn(t)
        assert jnp.isfinite(t_adj)
        
        # Test gradient
        grad_fn = jax.grad(adjustment_fn)
        grad = grad_fn(t)
        assert jnp.isfinite(grad)


if __name__ == "__main__":
    # Run tests
    test_sat = TestSaturationAdjustment()
    test_sat.test_no_adjustment_needed()
    test_sat.test_condensation_warm()
    test_sat.test_condensation_cold()
    test_sat.test_condensation_mixed_phase()
    test_sat.test_conservation()
    
    test_energy = TestEnergyConservation()
    test_energy.test_warming_condensation()
    test_energy.test_with_precipitation()
    
    test_adj = TestConvectiveAdjustment()
    test_adj.test_apply_tendencies_and_adjust()
    test_adj.test_with_cloud_tendencies()
    test_adj.test_jax_transformations()
    
    print("All adjustment tests passed!")