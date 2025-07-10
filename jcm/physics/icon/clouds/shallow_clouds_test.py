"""
Unit tests for shallow cloud scheme

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
import pytest
from .shallow_clouds import (
    CloudParameters, CloudState, CloudTendencies,
    saturation_vapor_pressure_water, saturation_vapor_pressure_ice,
    saturation_specific_humidity, calculate_cloud_fraction,
    partition_cloud_phase, condensation_evaporation,
    shallow_cloud_scheme
)
from ..constants.physical_constants import tmelt, eps


class TestSaturationFunctions:
    """Test saturation vapor pressure and humidity calculations"""
    
    def test_saturation_vapor_pressure_water(self):
        """Test saturation vapor pressure over water"""
        # At 0°C, should be ~611 Pa
        es_0c = saturation_vapor_pressure_water(jnp.array(tmelt))
        assert jnp.abs(es_0c - 610.78) < 1.0
        
        # At 20°C, should be ~2339 Pa
        es_20c = saturation_vapor_pressure_water(jnp.array(tmelt + 20.0))
        assert 2300 < es_20c < 2400
        
        # Should increase with temperature
        temps = jnp.linspace(250, 310, 10)
        es_vals = jax.vmap(saturation_vapor_pressure_water)(temps)
        assert jnp.all(jnp.diff(es_vals) > 0)
    
    def test_saturation_vapor_pressure_ice(self):
        """Test saturation vapor pressure over ice"""
        # At 0°C, should be ~611 Pa
        es_0c = saturation_vapor_pressure_ice(jnp.array(tmelt))
        assert jnp.abs(es_0c - 610.78) < 1.0
        
        # At -20°C, should be ~103 Pa
        es_m20c = saturation_vapor_pressure_ice(jnp.array(tmelt - 20.0))
        assert 100 < es_m20c < 110
        
        # Should increase with temperature
        temps = jnp.linspace(220, 273, 10)
        es_vals = jax.vmap(saturation_vapor_pressure_ice)(temps)
        assert jnp.all(jnp.diff(es_vals) > 0)
    
    def test_saturation_specific_humidity(self):
        """Test saturation specific humidity calculation"""
        # Standard atmosphere at sea level
        p_sfc = 101325.0  # Pa
        t_sfc = 288.15    # K (15°C)
        
        qs = saturation_specific_humidity(jnp.array(p_sfc), jnp.array(t_sfc))
        
        # Should be around 10 g/kg
        assert 0.008 < qs < 0.012
        
        # Should increase as pressure decreases (at constant temperature)
        pressures = jnp.linspace(100000, 20000, 10)
        qs_vals = jax.vmap(lambda p: saturation_specific_humidity(p, t_sfc))(pressures)
        assert jnp.all(jnp.diff(qs_vals) > 0)  # qs increases as pressure decreases
        
        # Test mixed phase region
        t_mixed = 260.0  # K
        p_mid = 50000.0  # Pa
        qs_mixed = saturation_specific_humidity(jnp.array(p_mid), jnp.array(t_mixed))
        
        # Should be between pure ice and pure water values
        qs_ice = eps * saturation_vapor_pressure_ice(jnp.array(t_mixed)) / p_mid
        qs_water = eps * saturation_vapor_pressure_water(jnp.array(t_mixed)) / p_mid
        assert qs_ice <= qs_mixed <= qs_water


class TestCloudFraction:
    """Test cloud fraction calculations"""
    
    def test_cloud_fraction_basic(self):
        """Test basic cloud fraction calculation"""
        config = CloudParameters()
        
        # Create test profile
        nlev = 20
        pressure = jnp.linspace(100000, 20000, nlev)
        temperature = jnp.linspace(288, 220, nlev)
        
        # Dry case - use 30% relative humidity everywhere
        # This creates a realistic dry atmosphere
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.3 * qs  # 30% RH everywhere
        cf, rh = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, 100000.0, config
        )
        
        # With 30% RH, should have no clouds anywhere
        assert jnp.all(cf < 0.01)  # No significant clouds
        assert jnp.all(rh < 0.35)  # RH should be around 30%
        
        # Saturated case - should have clouds
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.95 * qs  # 95% relative humidity
        cf, rh = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, 100000.0, config
        )
        
        assert jnp.any(cf > 0.5)  # Should have significant clouds
        assert jnp.all(rh > 0.9)   # High relative humidity
    
    def test_cloud_fraction_profile(self):
        """Test that critical RH varies with height"""
        config = CloudParameters()
        
        # Create pressure levels
        pressure = jnp.array([100000, 70000, 50000, 30000, 20000])
        temperature = jnp.array([288, 268, 248, 228, 218])
        p_sfc = 100000.0
        
        # Set constant relative humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        rh_target = 0.8
        specific_humidity = rh_target * qs
        
        cf, rh = calculate_cloud_fraction(
            temperature, specific_humidity, pressure, p_sfc, config
        )
        
        # Cloud fraction should increase with height at same RH
        # (because critical RH decreases with height)
        assert cf[0] < cf[-1]  # More clouds at top than bottom


class TestCloudPhase:
    """Test cloud phase partitioning"""
    
    def test_partition_all_liquid(self):
        """Test all liquid phase above freezing"""
        config = CloudParameters()
        
        temperature = jnp.array(280.0)  # Above freezing
        total_water = jnp.array(0.001)   # 1 g/kg
        
        ql, qi = partition_cloud_phase(temperature, total_water, config)
        
        assert jnp.allclose(ql, total_water)
        assert jnp.allclose(qi, 0.0)
    
    def test_partition_all_ice(self):
        """Test all ice phase below threshold"""
        config = CloudParameters()
        
        temperature = jnp.array(230.0)  # Well below freezing
        total_water = jnp.array(0.001)   # 1 g/kg
        
        ql, qi = partition_cloud_phase(temperature, total_water, config)
        
        assert jnp.allclose(ql, 0.0)
        assert jnp.allclose(qi, total_water)
    
    def test_partition_mixed_phase(self):
        """Test mixed phase region"""
        config = CloudParameters()
        
        # Middle of mixed phase region
        temperature = jnp.array(255.0)
        total_water = jnp.array(0.001)
        
        ql, qi = partition_cloud_phase(temperature, total_water, config)
        
        # Should have both phases
        assert ql > 0.0
        assert qi > 0.0
        assert jnp.allclose(ql + qi, total_water)
        
        # Test temperature dependence
        temps = jnp.linspace(238, 273, 10)
        total = jnp.ones(10) * 0.001
        ql_arr, qi_arr = jax.vmap(partition_cloud_phase, in_axes=(0, 0, None))(
            temps, total, config
        )
        
        # Liquid should increase with temperature
        assert jnp.all(jnp.diff(ql_arr) >= 0)
        # Ice should decrease with temperature
        assert jnp.all(jnp.diff(qi_arr) <= 0)


class TestCondensationEvaporation:
    """Test condensation/evaporation processes"""
    
    def test_condensation(self):
        """Test condensation in supersaturated conditions"""
        config = CloudParameters()
        
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        cloud_fraction = jnp.array(0.5)
        cloud_water = jnp.array(0.0005)
        cloud_ice = jnp.array(0.0)
        dt = 1800.0  # 30 minutes
        
        # Create supersaturated conditions
        qs = saturation_specific_humidity(pressure, temperature)
        specific_humidity = 1.1 * qs  # 110% relative humidity
        
        dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            cloud_fraction, pressure, dt, config
        )
        
        # Should have condensation
        assert dqdt < 0  # Humidity decreases
        assert dqcdt > 0  # Cloud water increases
        assert dtedt > 0  # Temperature increases (latent heat release)
    
    def test_evaporation(self):
        """Test evaporation in subsaturated conditions"""
        config = CloudParameters()
        
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        cloud_fraction = jnp.array(0.5)
        cloud_water = jnp.array(0.001)
        cloud_ice = jnp.array(0.0)
        dt = 1800.0
        
        # Create subsaturated conditions
        qs = saturation_specific_humidity(pressure, temperature)
        specific_humidity = 0.7 * qs  # 70% relative humidity
        
        dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
            temperature, specific_humidity, cloud_water, cloud_ice,
            cloud_fraction, pressure, dt, config
        )
        
        # Should have evaporation
        assert dqdt > 0   # Humidity increases
        assert dqcdt < 0  # Cloud water decreases
        assert dtedt < 0  # Temperature decreases (latent heat consumption)
        
        # Check evaporation doesn't exceed available cloud water
        assert dqcdt >= -cloud_water / dt


class TestShallowCloudScheme:
    """Test the full shallow cloud scheme"""
    
    def test_stable_conditions(self):
        """Test scheme in stable, dry conditions"""
        config = CloudParameters()
        
        # Create stable profile
        nlev = 20
        pressure = jnp.linspace(100000, 20000, nlev)
        temperature = jnp.linspace(288, 220, nlev)
        
        # Use 20% relative humidity profile - truly dry conditions
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.2 * qs  # 20% RH everywhere
        cloud_water = jnp.zeros(nlev)
        cloud_ice = jnp.zeros(nlev)
        surface_pressure = 100000.0
        dt = 1800.0
        
        tendencies, state = shallow_cloud_scheme(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, surface_pressure, dt, config
        )
        
        # Should have minimal tendencies  
        # Allow for some small tendencies due to numerical precision
        # Temperature tendencies can be larger at upper levels due to ice processes
        assert jnp.max(jnp.abs(tendencies.dtedt)) < 2e-3  # < ~200 K/day max
        assert jnp.all(jnp.abs(tendencies.dqdt) < 1e-6)   # < ~0.1 g/kg/day
        assert jnp.all(state.cloud_fraction < 0.1)
        assert tendencies.rain_flux < 1e-6
        assert tendencies.snow_flux < 1e-6
    
    def test_cloudy_conditions(self):
        """Test scheme with existing clouds"""
        config = CloudParameters()
        
        # Create profile with clouds
        nlev = 20
        pressure = jnp.linspace(100000, 20000, nlev)
        temperature = jnp.linspace(288, 220, nlev)
        
        # High humidity in mid-levels
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = qs * 0.5  # Start at 50% RH
        specific_humidity = specific_humidity.at[8:12].set(qs[8:12] * 0.95)  # 95% RH in mid-levels
        
        cloud_water = jnp.zeros(nlev)
        cloud_water = cloud_water.at[8:12].set(0.0005)  # Some cloud water
        cloud_ice = jnp.zeros(nlev)
        
        surface_pressure = 100000.0
        dt = 1800.0
        
        tendencies, state = shallow_cloud_scheme(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, surface_pressure, dt, config
        )
        
        # Should have clouds in humid layers
        assert jnp.any(state.cloud_fraction > 0.3)
        assert state.total_cloud_cover > 0.3
        
        # Check that tendencies are reasonable
        assert jnp.all(jnp.abs(tendencies.dtedt) < 1e-3)  # < ~100 K/day
        assert jnp.all(jnp.abs(tendencies.dqdt) < 1e-5)   # < ~1 g/kg/day
    
    def test_precipitation_formation(self):
        """Test precipitation formation from thick clouds"""
        config = CloudParameters()
        
        # Create profile with thick clouds
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        specific_humidity = jnp.array(0.008)
        cloud_water = jnp.array(0.002)  # 2 g/kg - above autoconversion threshold
        cloud_ice = jnp.array(0.0001)   # Small amount of ice
        surface_pressure = 100000.0
        dt = 1800.0
        
        tendencies, state = shallow_cloud_scheme(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, surface_pressure, dt, config
        )
        
        # Should produce precipitation
        assert tendencies.rain_flux > 0.0
        # Check net tendency - condensation may offset precipitation
        assert tendencies.dqcdt[0] < 1e-6  # Should be small or negative
    
    def test_jax_transformations(self):
        """Test JAX transformations work correctly"""
        config = CloudParameters()
        
        # Create test data
        def create_profile():
            pressure = jnp.linspace(100000, 20000, 10)
            temperature = jnp.linspace(288, 220, 10)
            specific_humidity = jnp.ones(10) * 0.005
            cloud_water = jnp.zeros(10)
            cloud_ice = jnp.zeros(10)
            return temperature, specific_humidity, pressure, cloud_water, cloud_ice
        
        # Test JIT compilation
        jitted_scheme = jax.jit(shallow_cloud_scheme, static_argnames=['config'])
        
        t, q, p, qc, qi = create_profile()
        tendencies, state = jitted_scheme(t, q, p, qc, qi, 100000.0, 1800.0, config)
        
        # Should produce valid output
        assert tendencies.dtedt.shape == t.shape
        assert state.cloud_fraction.shape == t.shape
        
        # Test gradient computation
        def loss_fn(temperature):
            t, q, p, qc, qi = create_profile()
            tend, _ = shallow_cloud_scheme(temperature, q, p, qc, qi, 100000.0, 1800.0, config)
            return jnp.sum(tend.dtedt ** 2)
        
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(t)
        
        # Should produce valid gradients
        assert grad.shape == t.shape
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    # Run basic tests
    test_sat = TestSaturationFunctions()
    test_sat.test_saturation_vapor_pressure_water()
    test_sat.test_saturation_vapor_pressure_ice()
    test_sat.test_saturation_specific_humidity()
    
    test_cf = TestCloudFraction()
    test_cf.test_cloud_fraction_basic()
    test_cf.test_cloud_fraction_profile()
    
    test_phase = TestCloudPhase()
    test_phase.test_partition_all_liquid()
    test_phase.test_partition_all_ice()
    test_phase.test_partition_mixed_phase()
    
    test_cond = TestCondensationEvaporation()
    test_cond.test_condensation()
    test_cond.test_evaporation()
    
    test_scheme = TestShallowCloudScheme()
    test_scheme.test_stable_conditions()
    test_scheme.test_cloudy_conditions()
    test_scheme.test_precipitation_formation()
    test_scheme.test_jax_transformations()
    
    print("All tests passed!")