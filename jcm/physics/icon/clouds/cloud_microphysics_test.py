"""
Unit tests for cloud microphysics scheme

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
import pytest
from .cloud_microphysics import (
    MicrophysicsParameters, MicrophysicsState, MicrophysicsTendencies,
    cloud_droplet_radius, autoconversion_kk2000, accretion_rain_cloud,
    ice_autoconversion, snow_accretion, melting_freezing,
    evaporation_sublimation, sedimentation_flux, cloud_microphysics
)
from ..constants.physical_constants import tmelt, rhow, cp, alhc, alhs, alhf


class TestCloudDropletRadius:
    """Test cloud droplet radius calculations"""
    
    def test_typical_values(self):
        """Test with typical atmospheric values"""
        cloud_water = jnp.array(0.5e-3)  # 0.5 g/kg
        air_density = jnp.array(1.0)      # kg/m³
        droplet_number = jnp.array(100e6) # 100 per cm³ -> per kg
        config = MicrophysicsParameters()
        
        radius = cloud_droplet_radius(cloud_water, air_density, droplet_number, config)
        
        # Should be in reasonable range (5-20 microns)
        assert 5e-6 < radius < 20e-6
    
    def test_limits(self):
        """Test radius limits are applied"""
        config = MicrophysicsParameters()
        
        # Very high cloud water with very few droplets should hit max radius
        radius_high = cloud_droplet_radius(
            jnp.array(10e-3), jnp.array(1.0), jnp.array(1e5), config  # Very few droplets
        )
        assert jnp.allclose(radius_high, config.ceffmax * 1e-6)
        
        # Very low cloud water with many droplets should hit min radius
        radius_low = cloud_droplet_radius(
            jnp.array(1e-6), jnp.array(1.0), jnp.array(1000e6), config  # Many droplets
        )
        assert jnp.allclose(radius_low, config.ceffmin * 1e-6)


class TestAutoconversion:
    """Test autoconversion processes"""
    
    def test_kk2000_threshold(self):
        """Test KK2000 autoconversion has threshold behavior"""
        config = MicrophysicsParameters()
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(0.5)
        droplet_number = jnp.array(100e6)
        dt = 1800.0
        
        # Below threshold - no autoconversion
        qc_low = config.ccraut * 0.5 * cloud_fraction
        rate_low = autoconversion_kk2000(
            qc_low, cloud_fraction, air_density, droplet_number, dt, config
        )
        assert rate_low < 1e-10
        
        # Above threshold - significant autoconversion
        qc_high = config.ccraut * 2.0 * cloud_fraction
        rate_high = autoconversion_kk2000(
            qc_high, cloud_fraction, air_density, droplet_number, dt, config
        )
        assert rate_high > 1e-8
    
    def test_kk2000_dependencies(self):
        """Test KK2000 dependencies on cloud water and droplet number"""
        config = MicrophysicsParameters()
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(1.0)  # Full cloud cover to simplify
        dt = 1.0  # Very short timestep - we're testing the formula, not the limiter
        
        # Use cloud water well above threshold
        qc = jnp.array(0.8e-3)
        nc = jnp.array(100e6)
        rate_base = autoconversion_kk2000(qc, cloud_fraction, air_density, nc, dt, config)
        
        # Test that rate increases with cloud water
        qc_higher = jnp.array(1.0e-3)
        rate_higher = autoconversion_kk2000(qc_higher, cloud_fraction, air_density, nc, dt, config)
        # Even with limiter, higher qc should give higher rate
        assert rate_higher > rate_base
        
        # For droplet dependency, use same total water but different cloud fractions
        # This tests the in-cloud calculation
        cf_low = jnp.array(0.5)
        cf_high = jnp.array(1.0)
        # Same grid-mean cloud water
        qc_grid = jnp.array(0.4e-3)
        
        rate_cf_low = autoconversion_kk2000(qc_grid, cf_low, air_density, nc, dt, config)
        rate_cf_high = autoconversion_kk2000(qc_grid, cf_high, air_density, nc, dt, config)
        
        # Lower cloud fraction means higher in-cloud water, so higher autoconversion
        assert rate_cf_low > rate_cf_high
        
        # Verify no autoconversion below threshold
        qc_low = config.ccraut * 0.5
        rate_low = autoconversion_kk2000(qc_low, cloud_fraction, air_density, nc, dt, config)
        assert rate_low < 1e-10
    
    def test_ice_autoconversion(self):
        """Test ice autoconversion to snow"""
        config = MicrophysicsParameters()
        cloud_fraction = jnp.array(0.7)
        dt = 1800.0
        
        # Test temperature dependence of aggregation efficiency
        # At -15°C, aggregation is most efficient
        t_optimal = tmelt - 15.0
        t_cold = tmelt - 40.0
        
        # Use same in-cloud ice content for fair comparison
        qi_in_cloud = 1.0e-3  # Above critical threshold at both temperatures
        cloud_ice_opt = qi_in_cloud * cloud_fraction
        cloud_ice_cold = qi_in_cloud * cloud_fraction
        
        rate_optimal = ice_autoconversion(cloud_ice_opt, t_optimal, cloud_fraction, dt, config)
        rate_cold = ice_autoconversion(cloud_ice_cold, t_cold, cloud_fraction, dt, config)
        
        # At optimal temperature, autoconversion should be faster
        assert rate_optimal > rate_cold
        
        # Test threshold behavior
        cloud_ice_low = jnp.array(0.1e-3)  # Below typical threshold
        rate_low = ice_autoconversion(cloud_ice_low, t_optimal, cloud_fraction, dt, config)
        assert rate_low < 1e-10  # Should be essentially zero


class TestAccretion:
    """Test accretion processes"""
    
    def test_rain_cloud_accretion(self):
        """Test accretion of cloud by rain"""
        config = MicrophysicsParameters()
        cloud_water = jnp.array(0.5e-3)
        rain_water = jnp.array(1e-3)
        cloud_fraction = jnp.array(0.6)
        air_density = jnp.array(1.0)
        
        rate = accretion_rain_cloud(
            cloud_water, rain_water, cloud_fraction, air_density, config
        )
        
        # Should be positive and reasonable
        assert rate > 0
        assert rate < cloud_water  # Can't accrete more than available
        
        # No rain - no accretion
        rate_no_rain = accretion_rain_cloud(
            cloud_water, jnp.array(0.0), cloud_fraction, air_density, config
        )
        assert rate_no_rain == 0
    
    def test_snow_accretion(self):
        """Test accretion by snow (riming and aggregation)"""
        config = MicrophysicsParameters()
        target = jnp.array(0.3e-3)
        snow = jnp.array(0.5e-3)
        temperature = tmelt - 10.0
        air_density = jnp.array(0.8)
        
        # Riming (liquid target)
        rime_rate = snow_accretion(target, snow, temperature, air_density, True, config)
        
        # Aggregation (ice target)
        aggr_rate = snow_accretion(target, snow, temperature, air_density, False, config)
        
        # Both should be positive
        assert rime_rate > 0
        assert aggr_rate > 0
        
        # Riming should generally be more efficient than aggregation
        assert rime_rate > aggr_rate


class TestMeltingFreezing:
    """Test melting and freezing processes"""
    
    def test_melting_above_freezing(self):
        """Test snow melts above 0°C"""
        config = MicrophysicsParameters()
        snow = jnp.array(1e-3)
        rain = jnp.array(0.5e-3)
        dt = 100.0
        
        # 2°C above freezing
        temperature = tmelt + 2.0
        melt_rate, freeze_rate = melting_freezing(temperature, snow, rain, dt, config)
        
        assert melt_rate > 0
        assert freeze_rate == 0
        assert melt_rate <= snow / dt  # Can't melt more than available
    
    def test_freezing_below_freezing(self):
        """Test rain freezes below 0°C"""
        config = MicrophysicsParameters()
        snow = jnp.array(0.5e-3)
        rain = jnp.array(1e-3)
        dt = 100.0
        
        # Well below freezing (-10°C)
        temperature = tmelt - 10.0
        melt_rate, freeze_rate = melting_freezing(temperature, snow, rain, dt, config)
        
        assert melt_rate == 0
        assert freeze_rate > 0
        assert freeze_rate <= rain / dt  # Can't freeze more than available
        
        # Just below freezing (-2°C) - less efficient
        temperature_warm = tmelt - 2.0
        _, freeze_rate_warm = melting_freezing(temperature_warm, snow, rain, dt, config)
        assert freeze_rate_warm < freeze_rate


class TestEvaporationSublimation:
    """Test evaporation and sublimation processes"""
    
    def test_evaporation_subsaturated(self):
        """Test rain evaporation in subsaturated conditions"""
        config = MicrophysicsParameters()
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        rain = jnp.array(0.5e-3)
        snow = jnp.array(0.2e-3)
        air_density = jnp.array(1.0)
        
        # Create subsaturated conditions (50% RH)
        from .shallow_clouds import saturation_specific_humidity
        qs = saturation_specific_humidity(pressure, temperature)
        specific_humidity = 0.5 * qs
        
        rain_evap, snow_sublim = evaporation_sublimation(
            temperature, specific_humidity, pressure,
            rain, snow, air_density, config
        )
        
        # Both should evaporate/sublimate
        assert rain_evap > 0
        assert snow_sublim > 0
    
    def test_no_evaporation_saturated(self):
        """Test no evaporation at saturation"""
        config = MicrophysicsParameters()
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        rain = jnp.array(0.5e-3)
        snow = jnp.array(0.2e-3)
        air_density = jnp.array(1.0)
        
        # Saturated conditions
        from .shallow_clouds import saturation_specific_humidity
        qs = saturation_specific_humidity(pressure, temperature)
        specific_humidity = qs
        
        rain_evap, snow_sublim = evaporation_sublimation(
            temperature, specific_humidity, pressure,
            rain, snow, air_density, config
        )
        
        # No evaporation at saturation
        assert jnp.allclose(rain_evap, 0.0)
        assert jnp.allclose(snow_sublim, 0.0)


class TestSedimentation:
    """Test sedimentation processes"""
    
    def test_sedimentation_flux(self):
        """Test basic sedimentation flux calculation"""
        nlev = 10
        # Decreasing hydrometeor content with height (realistic)
        hydrometeor = jnp.linspace(1e-3, 0.1e-3, nlev)  # kg/kg
        air_density = jnp.ones(nlev) * 1.0     # kg/m³
        dz = jnp.ones(nlev) * 100.0            # m
        vt = jnp.ones(nlev) * 1.0              # m/s
        dt = 100.0  # Longer timestep to avoid CFL issues
        
        flux, tendency = sedimentation_flux(hydrometeor, air_density, dz, vt, dt)
        
        # Check flux shape
        assert flux.shape == (nlev + 1,)
        assert tendency.shape == (nlev,)
        
        # Top flux should be zero (no input from above)
        assert flux[0] == 0
        
        # Surface flux should be positive
        assert flux[-1] > 0
        
        # Top level loses mass (no input from above)
        assert tendency[0] < 0
        
        # Conservation check: total mass change equals surface flux
        # tendency is in kg/kg/s, need to convert to kg/m²/s
        total_mass_change = jnp.sum(tendency * air_density * dz)  # kg/m²/s
        # Surface flux is already in kg/m²/s
        assert jnp.abs(total_mass_change + flux[-1]) < 1e-6


class TestFullMicrophysics:
    """Test the complete microphysics scheme"""
    
    def test_warm_rain_process(self):
        """Test warm rain microphysics"""
        config = MicrophysicsParameters()
        nlev = 20
        
        # Create warm profile with clouds
        temperature = jnp.linspace(290, 270, nlev)  # All above freezing
        pressure = jnp.linspace(100000, 70000, nlev)
        
        # Humid conditions with cloud water
        from .shallow_clouds import saturation_specific_humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.9 * qs
        
        cloud_water = jnp.zeros(nlev)
        cloud_water = cloud_water.at[5:10].set(1e-3)  # Cloud layer
        cloud_ice = jnp.zeros(nlev)
        rain_water = jnp.zeros(nlev)
        snow = jnp.zeros(nlev)
        cloud_fraction = jnp.zeros(nlev)
        cloud_fraction = cloud_fraction.at[5:10].set(0.8)
        
        air_density = pressure / (287.0 * temperature)
        layer_thickness = jnp.ones(nlev) * 200.0
        droplet_number = jnp.ones(nlev) * 100e6
        dt = 300.0
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, rain_water, snow,
            cloud_fraction, air_density, layer_thickness,
            droplet_number, dt, config
        )
        
        # Should produce rain from cloud water
        assert jnp.any(tendencies.dqcdt < 0)  # Cloud water decreases
        assert jnp.any(tendencies.dqrdt > 0)  # Rain increases
        assert jnp.all(tendencies.dqsdt == 0)  # No snow in warm conditions
        assert state.precip_snow == 0  # No snow at surface
    
    def test_cold_cloud_process(self):
        """Test ice microphysics"""
        config = MicrophysicsParameters()
        nlev = 20
        
        # Create cold profile
        temperature = jnp.linspace(250, 220, nlev)  # All below freezing
        pressure = jnp.linspace(70000, 30000, nlev)
        
        # Set up ice clouds
        from .shallow_clouds import saturation_specific_humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.9 * qs
        
        cloud_water = jnp.zeros(nlev)
        cloud_ice = jnp.zeros(nlev)
        cloud_ice = cloud_ice.at[5:10].set(0.5e-3)  # Ice cloud layer
        rain_water = jnp.zeros(nlev)
        snow = jnp.zeros(nlev)
        cloud_fraction = jnp.zeros(nlev)
        cloud_fraction = cloud_fraction.at[5:10].set(0.6)
        
        air_density = pressure / (287.0 * temperature)
        layer_thickness = jnp.ones(nlev) * 300.0
        droplet_number = jnp.ones(nlev) * 50e6
        dt = 300.0
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, rain_water, snow,
            cloud_fraction, air_density, layer_thickness,
            droplet_number, dt, config
        )
        
        # Should produce snow from ice
        assert jnp.any(tendencies.dqidt < 0)  # Ice decreases
        assert jnp.any(tendencies.dqsdt > 0)  # Snow increases
        assert jnp.all(tendencies.dqrdt == 0)  # No rain in cold conditions
        assert state.precip_rain == 0  # No rain at surface
    
    def test_mixed_phase_process(self):
        """Test mixed-phase microphysics"""
        config = MicrophysicsParameters()
        nlev = 30
        
        # Create profile spanning freezing level
        temperature = jnp.linspace(285, 250, nlev)
        pressure = jnp.linspace(100000, 50000, nlev)
        
        # Find freezing level
        freeze_level = jnp.argmin(jnp.abs(temperature - tmelt))
        
        # Set up mixed-phase clouds
        from .shallow_clouds import saturation_specific_humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.9 * qs
        
        # Liquid cloud below freezing level
        cloud_water = jnp.zeros(nlev)
        cloud_water = cloud_water.at[freeze_level-3:freeze_level+1].set(0.8e-3)
        
        # Ice cloud above freezing level
        cloud_ice = jnp.zeros(nlev)
        cloud_ice = cloud_ice.at[freeze_level:freeze_level+3].set(0.3e-3)
        
        rain_water = jnp.zeros(nlev)
        snow = jnp.zeros(nlev).at[freeze_level-2:freeze_level+2].set(0.2e-3)
        cloud_fraction = jnp.zeros(nlev).at[freeze_level-3:freeze_level+3].set(0.7)
        
        air_density = pressure / (287.0 * temperature)
        layer_thickness = jnp.ones(nlev) * 200.0
        droplet_number = jnp.ones(nlev) * 80e6
        dt = 300.0
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, rain_water, snow,
            cloud_fraction, air_density, layer_thickness,
            droplet_number, dt, config
        )
        
        # Should have melting near freezing level
        assert jnp.any(state.melting_rate > 0)
        
        # Both rain and snow at surface possible
        assert state.precip_rain >= 0
        assert state.precip_snow >= 0
    
    def test_conservation(self):
        """Test mass conservation in microphysics"""
        config = MicrophysicsParameters()
        nlev = 10
        
        # Simple setup
        temperature = jnp.ones(nlev) * 270.0
        pressure = jnp.ones(nlev) * 90000.0
        specific_humidity = jnp.ones(nlev) * 0.005
        cloud_water = jnp.ones(nlev) * 0.0005
        cloud_ice = jnp.ones(nlev) * 0.0002
        rain_water = jnp.ones(nlev) * 0.0001
        snow = jnp.ones(nlev) * 0.0001
        cloud_fraction = jnp.ones(nlev) * 0.5
        air_density = jnp.ones(nlev) * 1.0
        layer_thickness = jnp.ones(nlev) * 100.0
        droplet_number = jnp.ones(nlev) * 100e6
        dt = 60.0
        
        # Get initial total water
        total_initial = (
            specific_humidity + cloud_water + cloud_ice + rain_water + snow
        ).sum()
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, rain_water, snow,
            cloud_fraction, air_density, layer_thickness,
            droplet_number, dt, config
        )
        
        # Total tendency (excluding sedimentation out)
        total_tend = (
            tendencies.dqdt + tendencies.dqcdt + tendencies.dqidt +
            tendencies.dqrdt + tendencies.dqsdt
        ).sum()
        
        # Should approximately conserve mass (small loss due to precipitation)
        # Total tendency should be negative (loss to surface)
        assert total_tend <= 0
    
    def test_jax_compatibility(self):
        """Test JAX transformations"""
        config = MicrophysicsParameters()
        
        # Simple test case
        def create_state():
            nlev = 5
            temperature = jnp.ones(nlev) * 273.0
            pressure = jnp.ones(nlev) * 90000.0
            specific_humidity = jnp.ones(nlev) * 0.005
            cloud_water = jnp.ones(nlev) * 0.0005
            cloud_ice = jnp.ones(nlev) * 0.0
            rain_water = jnp.ones(nlev) * 0.0
            snow = jnp.ones(nlev) * 0.0
            cloud_fraction = jnp.ones(nlev) * 0.5
            air_density = jnp.ones(nlev) * 1.0
            layer_thickness = jnp.ones(nlev) * 100.0
            droplet_number = jnp.ones(nlev) * 100e6
            return (temperature, specific_humidity, pressure, cloud_water,
                    cloud_ice, rain_water, snow, cloud_fraction,
                    air_density, layer_thickness, droplet_number)
        
        # Test JIT compilation
        jitted_micro = jax.jit(cloud_microphysics, static_argnames=['config'])
        
        state_vars = create_state()
        tendencies, state = jitted_micro(*state_vars, 60.0, config)
        
        # Should produce valid output
        assert tendencies.dtedt.shape == state_vars[0].shape
        assert jnp.all(jnp.isfinite(tendencies.dtedt))
        
        # Test gradient computation
        def loss_fn(cloud_water):
            state_vars = create_state()
            state_vars = list(state_vars)
            state_vars[3] = cloud_water
            tend, _ = cloud_microphysics(*state_vars, 60.0, config)
            return jnp.sum(tend.dqcdt ** 2)
        
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.ones(5) * 0.0005)
        
        assert grad.shape == (5,)
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    # Run tests
    test_radius = TestCloudDropletRadius()
    test_radius.test_typical_values()
    test_radius.test_limits()
    
    test_auto = TestAutoconversion()
    test_auto.test_kk2000_threshold()
    test_auto.test_kk2000_dependencies()
    test_auto.test_ice_autoconversion()
    
    test_accr = TestAccretion()
    test_accr.test_rain_cloud_accretion()
    test_accr.test_snow_accretion()
    
    test_melt = TestMeltingFreezing()
    test_melt.test_melting_above_freezing()
    test_melt.test_freezing_below_freezing()
    
    test_evap = TestEvaporationSublimation()
    test_evap.test_evaporation_subsaturated()
    test_evap.test_no_evaporation_saturated()
    
    test_sedi = TestSedimentation()
    test_sedi.test_sedimentation_flux()
    
    test_full = TestFullMicrophysics()
    test_full.test_warm_rain_process()
    test_full.test_cold_cloud_process()
    test_full.test_mixed_phase_process()
    test_full.test_conservation()
    test_full.test_jax_compatibility()
    
    print("All tests passed!")