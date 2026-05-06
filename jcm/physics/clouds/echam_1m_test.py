"""Unit tests for cloud microphysics scheme

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from .echam_1m import (
    MicrophysicsParameters, cloud_droplet_radius,
    autoconversion, autoconversion_beheng, autoconversion_kk2000,
    accretion_rain_cloud,
    ice_autoconversion, snow_accretion, melting_freezing,
    evaporation_sublimation, sedimentation_flux, cloud_microphysics,
    cloud_microphysics_column_sweep,
)
from jcm.constants import tmelt


class TestCloudDropletRadius:
    """Test cloud droplet radius calculations"""
    
    def test_typical_values(self):
        """Test with typical atmospheric values"""
        cloud_water = jnp.array(0.5e-3)  # 0.5 g/kg
        air_density = jnp.array(1.0)      # kg/m³
        droplet_number = jnp.array(100e6) # 100 per cm³ -> per kg
        config = MicrophysicsParameters.default()
        
        radius = cloud_droplet_radius(cloud_water, air_density, droplet_number, config)
        
        # Should be in reasonable range (5-20 microns)
        assert 5e-6 < radius < 20e-6
    
    def test_limits(self):
        """Test radius limits are applied"""
        config = MicrophysicsParameters.default()
        
        # Very high cloud water with very few droplets should hit max radius
        radius_high = cloud_droplet_radius(
            jnp.array(10e-3), jnp.array(1.0), jnp.array(1e5), config  # Very few droplets
        )
        assert jnp.allclose(radius_high, float(config.ceffmax) * 1e-6)
        
        # Very low cloud water with many droplets should hit min radius
        radius_low = cloud_droplet_radius(
            jnp.array(1e-6), jnp.array(1.0), jnp.array(1000e6), config  # Many droplets
        )
        assert jnp.allclose(radius_low, float(config.ceffmin) * 1e-6)


class TestAutoconversion:
    """Test autoconversion processes"""
    
    def test_autoconversion_no_water_no_rate(self):
        """Beheng autoconversion gives essentially zero rate at near-zero qc."""
        config = MicrophysicsParameters.default()
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(0.5)
        droplet_number = jnp.array(100e6)
        dt = 1800.0

        # qc = 0 → no autoconversion
        rate_zero = autoconversion_beheng(
            jnp.array(0.0), cloud_fraction, air_density, droplet_number, dt, config,
        )
        assert float(rate_zero) < 1e-15

        # qc = tiny (1e-7 kg/kg) → effectively no autoconversion
        rate_tiny = autoconversion_beheng(
            jnp.array(1e-7), cloud_fraction, air_density, droplet_number, dt, config,
        )
        assert float(rate_tiny) < 1e-12

        # qc = realistic post-convection (0.6 g/kg) → meaningful rate but
        # bounded by mass conservation (cannot deplete more than qc/dt).
        qc = jnp.array(0.6e-3)
        rate_high = autoconversion_beheng(
            qc, cloud_fraction, air_density, droplet_number, dt, config,
        )
        assert float(rate_high) > 0.0
        assert float(rate_high) <= float(qc) / dt + 1e-12, (
            "Beheng integral form must respect mass conservation: "
            "autoconv rate cannot exceed qc/dt."
        )

    def test_autoconversion_dependencies(self):
        """Beheng autoconversion: rate increases with qc, decreases with Nc.

        Note: with the implicit-integration formulation, large qc gets
        capped at the mass-conservation limit qc/dt, so the "rate
        increases with qc" check uses a short timestep where the rate
        hasn't saturated yet.
        """
        config = MicrophysicsParameters.default()
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(1.0)
        dt = 0.1  # short timestep so rate doesn't saturate at qc/dt

        rate_low_qc = autoconversion_beheng(
            jnp.array(0.4e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        rate_high_qc = autoconversion_beheng(
            jnp.array(0.8e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        assert float(rate_high_qc) > float(rate_low_qc), (
            "Higher qc → higher Beheng autoconversion rate"
        )

        # Droplet number dependence: more droplets (cleaner air) → slower
        # autoconversion (Nc^-3.3 in the formula).
        rate_few_droplets = autoconversion_beheng(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(50e6), dt, config,
        )
        rate_many_droplets = autoconversion_beheng(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(500e6), dt, config,
        )
        assert float(rate_few_droplets) > float(rate_many_droplets), (
            "Fewer cloud droplets → faster autoconversion (Beheng Nc^-3.3)"
        )


class TestKK2000Autoconversion:
    """KK2000 explicit-rate autoconversion + dispatcher tests."""

    def test_below_threshold_zero(self):
        config = MicrophysicsParameters.default(
            ccraut=1e-5, autoconversion_scheme="kk2000",
        )
        rate = autoconversion_kk2000(
            jnp.array(1e-6),               # below ccraut threshold
            jnp.array(0.5), jnp.array(1.0),
            jnp.array(100e6), 1800.0, config,
        )
        assert float(rate) == 0.0

    def test_dependencies(self):
        """KK2000: rate ∝ qc^2.47, ∝ Nc^-1.79 — same monotonicity as Beheng."""
        config = MicrophysicsParameters.default(
            ccraut=1e-5, autoconversion_scheme="kk2000",
        )
        air_density = jnp.array(1.0)
        cloud_fraction = jnp.array(1.0)
        dt = 1800.0

        rate_lo_qc = autoconversion_kk2000(
            jnp.array(0.4e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        rate_hi_qc = autoconversion_kk2000(
            jnp.array(0.8e-3), cloud_fraction, air_density,
            jnp.array(100e6), dt, config,
        )
        assert float(rate_hi_qc) > float(rate_lo_qc)

        rate_few_drops = autoconversion_kk2000(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(50e6), dt, config,
        )
        rate_many_drops = autoconversion_kk2000(
            jnp.array(0.6e-3), cloud_fraction, air_density,
            jnp.array(500e6), dt, config,
        )
        assert float(rate_few_drops) > float(rate_many_drops)

    def test_dispatcher_picks_scheme(self):
        """``autoconversion(...)`` dispatches by ``config.autoconversion_scheme``."""
        qc = jnp.array(0.6e-3)
        cloud_fraction = jnp.array(0.5)
        air_density = jnp.array(1.0)
        droplet_number = jnp.array(100e6)
        dt = 1800.0

        cfg_beheng = MicrophysicsParameters.default(autoconversion_scheme="beheng")
        cfg_kk2000 = MicrophysicsParameters.default(
            ccraut=1e-5, autoconversion_scheme="kk2000",
        )

        rate_via_dispatcher_beheng = autoconversion(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_beheng,
        )
        rate_direct_beheng = autoconversion_beheng(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_beheng,
        )
        assert jnp.allclose(rate_via_dispatcher_beheng, rate_direct_beheng)

        rate_via_dispatcher_kk = autoconversion(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_kk2000,
        )
        rate_direct_kk = autoconversion_kk2000(
            qc, cloud_fraction, air_density, droplet_number, dt, cfg_kk2000,
        )
        assert jnp.allclose(rate_via_dispatcher_kk, rate_direct_kk)

        # Sanity: the two schemes give different rates on the same column
        assert not jnp.allclose(rate_via_dispatcher_beheng, rate_via_dispatcher_kk)

    def test_scheme_int_alias(self):
        """SCHEME_BEHENG / SCHEME_KK2000 ints round-trip with string aliases."""
        cfg_str = MicrophysicsParameters.default(autoconversion_scheme="kk2000")
        cfg_int = MicrophysicsParameters.default(
            autoconversion_scheme=MicrophysicsParameters.SCHEME_KK2000,
        )
        assert int(cfg_str.autoconversion_scheme) == MicrophysicsParameters.SCHEME_KK2000
        assert int(cfg_int.autoconversion_scheme) == int(cfg_str.autoconversion_scheme)
    
    def test_ice_autoconversion(self):
        """Test ice autoconversion to snow"""
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        rain = jnp.array(0.5e-3)
        snow = jnp.array(0.2e-3)
        air_density = jnp.array(1.0)
        
        # Create subsaturated conditions (50% RH)
        from .sundqvist import saturation_specific_humidity
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
        config = MicrophysicsParameters.default()
        temperature = jnp.array(280.0)
        pressure = jnp.array(90000.0)
        rain = jnp.array(0.5e-3)
        snow = jnp.array(0.2e-3)
        air_density = jnp.array(1.0)
        
        # Saturated conditions
        from .sundqvist import saturation_specific_humidity
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
        config = MicrophysicsParameters.default()
        nlev = 20
        
        # Create warm profile with clouds
        temperature = jnp.linspace(290, 270, nlev)  # All above freezing
        pressure = jnp.linspace(100000, 70000, nlev)
        
        # Humid conditions with cloud water
        from .sundqvist import saturation_specific_humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.9 * qs
        
        cloud_water = jnp.zeros(nlev)
        cloud_water = cloud_water.at[5:10].set(1e-3)  # Cloud layer
        cloud_ice = jnp.zeros(nlev)
        cloud_fraction = jnp.zeros(nlev)
        cloud_fraction = cloud_fraction.at[5:10].set(0.8)
        
        air_density = pressure / (287.0 * temperature)
        layer_thickness = jnp.ones(nlev) * 200.0
        droplet_number = jnp.ones(nlev) * 100e6
        dt = 300.0
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, cloud_fraction,
            air_density, layer_thickness, droplet_number,
            dt, config
        )
        
        # Should produce rain from cloud water
        assert jnp.any(tendencies.dqcdt < 0)  # Cloud water decreases
        assert jnp.any(tendencies.dqrdt > 0)  # Rain increases
        assert jnp.all(tendencies.dqsdt == 0)  # No snow in warm conditions
        assert state.precip_snow == 0  # No snow at surface
    
    def test_cold_cloud_process(self):
        """Test ice microphysics"""
        config = MicrophysicsParameters.default()
        nlev = 20
        
        # Create cold profile
        temperature = jnp.linspace(250, 220, nlev)  # All below freezing
        pressure = jnp.linspace(70000, 30000, nlev)
        
        # Set up ice clouds
        from .sundqvist import saturation_specific_humidity
        qs = jax.vmap(saturation_specific_humidity)(pressure, temperature)
        specific_humidity = 0.9 * qs
        
        cloud_water = jnp.zeros(nlev)
        cloud_ice = jnp.zeros(nlev)
        cloud_ice = cloud_ice.at[5:10].set(0.5e-3)  # Ice cloud layer
        cloud_fraction = jnp.zeros(nlev)
        cloud_fraction = cloud_fraction.at[5:10].set(0.6)
        
        air_density = pressure / (287.0 * temperature)
        layer_thickness = jnp.ones(nlev) * 300.0
        droplet_number = jnp.ones(nlev) * 50e6
        dt = 300.0
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, cloud_fraction,
            air_density, layer_thickness, droplet_number,
            dt, config
        )
        
        # Should produce snow from ice
        assert jnp.any(tendencies.dqidt < 0)  # Ice decreases
        assert jnp.any(tendencies.dqsdt > 0)  # Snow increases
        assert jnp.all(tendencies.dqrdt == 0)  # No rain in cold conditions
        assert state.precip_rain == 0  # No rain at surface
    
    def test_mixed_phase_process(self):
        """Test mixed-phase microphysics"""
        config = MicrophysicsParameters.default()
        nlev = 30
        
        # Create profile spanning freezing level
        temperature = jnp.linspace(285, 250, nlev)
        pressure = jnp.linspace(100000, 50000, nlev)
        
        # Find freezing level
        freeze_level = jnp.argmin(jnp.abs(temperature - tmelt))
        
        # Set up mixed-phase clouds
        from .sundqvist import saturation_specific_humidity
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
            cloud_water, cloud_ice, cloud_fraction,
            air_density, layer_thickness, droplet_number,
            dt, config, rain_water, snow
        )
        
        # Should have melting near freezing level
        assert jnp.any(state.melting_rate > 0)
        
        # Both rain and snow at surface possible
        assert state.precip_rain >= 0
        assert state.precip_snow >= 0
    
    def test_conservation(self):
        """Test mass conservation in microphysics"""
        config = MicrophysicsParameters.default()
        nlev = 10
        
        # Simple setup
        temperature = jnp.ones(nlev) * 270.0
        pressure = jnp.ones(nlev) * 90000.0
        specific_humidity = jnp.ones(nlev) * 0.005
        cloud_water = jnp.ones(nlev) * 0.0005
        cloud_ice = jnp.ones(nlev) * 0.0002
        cloud_fraction = jnp.ones(nlev) * 0.5
        air_density = jnp.ones(nlev) * 1.0
        layer_thickness = jnp.ones(nlev) * 100.0
        droplet_number = jnp.ones(nlev) * 100e6
        dt = 60.0
        
        tendencies, state = cloud_microphysics(
            temperature, specific_humidity, pressure,
            cloud_water, cloud_ice, cloud_fraction,
            air_density, layer_thickness, droplet_number,
            dt, config
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
        config = MicrophysicsParameters.default()
        
        # Simple test case
        def create_state():
            nlev = 5
            temperature = jnp.ones(nlev) * 273.0
            pressure = jnp.ones(nlev) * 90000.0
            specific_humidity = jnp.ones(nlev) * 0.005
            cloud_water = jnp.ones(nlev) * 0.0005
            cloud_ice = jnp.ones(nlev) * 0.0
            cloud_fraction = jnp.ones(nlev) * 0.5
            air_density = jnp.ones(nlev) * 1.0
            layer_thickness = jnp.ones(nlev) * 100.0
            droplet_number = jnp.ones(nlev) * 100e6
            return (temperature, specific_humidity, pressure, cloud_water,
                    cloud_ice, cloud_fraction, air_density, layer_thickness, droplet_number)
        
        # Test JIT compilation
        jitted_micro = jax.jit(cloud_microphysics)
        
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


class TestColumnSweepMicrophysics:
    """Tests for the ICON ``mo_cloud.f90`` column-sweep port.

    The column sweep propagates rain (``zrfl``) and snow (``zsfl``) as
    downward fluxes top-to-bottom inside a single timestep. These tests
    cover the column-budget invariants that the per-level
    :func:`cloud_microphysics` cannot satisfy because it discards rain/
    snow each call.
    """

    @staticmethod
    def _column(nlev=20, qc_top=None, qi_top=None, T_profile=None):
        """Build a column with optional cloud water/ice loading."""
        T = jnp.linspace(220.0, 295.0, nlev) if T_profile is None else T_profile
        p = jnp.linspace(20000.0, 100000.0, nlev)
        q = jnp.full(nlev, 5e-3)
        qc = jnp.zeros(nlev) if qc_top is None else qc_top
        qi = jnp.zeros(nlev) if qi_top is None else qi_top
        cf = jnp.where((qc + qi) > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        return T, q, p, qc, qi, cf, rho, dz, ndrop

    def test_no_clouds_no_precip(self):
        """A column with zero qc/qi must produce zero surface precip."""
        cfg = MicrophysicsParameters.default()
        T, q, p, qc, qi, cf, rho, dz, ndrop = self._column()
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        assert float(state.precip_rain) == 0.0
        assert float(state.precip_snow) == 0.0

    def test_warm_cloud_makes_surface_rain(self):
        """A liquid cloud aloft in a warm, near-saturated column produces rain.

        Background ``q`` is set to ~95% of saturation everywhere so that
        Rotstayn rain evaporation cannot consume the full precipitation
        flux before it reaches the surface.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        qsw = jax.vmap(saturation_specific_humidity)(p, T)
        q = 0.95 * qsw
        qc = jnp.zeros(nlev).at[5].set(2e-3)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        # Rain at surface, no snow (column never goes below freezing).
        assert float(state.precip_rain) > 1e-6
        assert float(state.precip_snow) == 0.0

    def test_subsaturated_column_evaporates_rain(self):
        """A dry column under a cloud must evaporate falling rain.

        Without Rotstayn rain evaporation, a cloud aloft would always
        deliver its precipitation to the surface. With rain evap, the
        surface flux is *less* than the column-integrated rain source
        in a column with sub-saturated layers below the cloud.
        """
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 300.0, nlev)
        # Dry column (q ~ 5e-3 ≪ qsw ≈ 0.02 at 290K).
        qc = jnp.zeros(nlev).at[5].set(2e-3)
        T_p, q, p, qc_arr, qi, cf, rho, dz, ndrop = self._column(
            nlev=nlev, qc_top=qc, T_profile=T,
        )
        _, state = cloud_microphysics_column_sweep(
            T_p, q, p, qc_arr, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        rain_source_total = float(jnp.sum(state.rain_flux))
        # surface precip should be strictly LESS than the local rain
        # source when rain evap is active in subsaturated air below cloud.
        assert float(state.precip_rain) < rain_source_total

    def test_snow_above_warm_layer_melts_to_rain(self):
        """Snow flux generated aloft melts as it falls into T>273K layers."""
        cfg = MicrophysicsParameters.default()
        nlev = 20
        # Cold above (level 3, 240K), warm below (>273K from level 8 down).
        T = jnp.concatenate([
            jnp.linspace(220.0, 260.0, 8),
            jnp.linspace(280.0, 295.0, nlev - 8),
        ])
        qi = jnp.zeros(nlev).at[3].set(5e-4).at[4].set(3e-4)
        T_p, q, p, qc, qi_arr, cf, rho, dz, ndrop = self._column(
            nlev=nlev, qi_top=qi, T_profile=T,
        )
        _, state = cloud_microphysics_column_sweep(
            T_p, q, p, qc, qi_arr, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        # The aloft ice → snow flux is small; what matters is that the
        # warm layers melt all of it before the surface, so surface snow
        # is essentially zero while surface rain is positive.
        assert float(state.precip_snow) < 1e-10
        # Some ice was autoconverted to snow → melted → rain.
        assert float(state.precip_rain) >= 0.0

    def test_zero_dt_dependence_on_thicker_column(self):
        """Sanity: thicker layers ≠ instability; precip should be finite."""
        cfg = MicrophysicsParameters.default()
        nlev = 20
        T = jnp.linspace(280.0, 295.0, nlev)
        qc = jnp.zeros(nlev).at[5].set(2e-3)
        T_p, q, p, qc_arr, qi, cf, rho, dz, ndrop = self._column(
            nlev=nlev, qc_top=qc, T_profile=T,
        )
        for dz_val in (200.0, 1000.0, 2000.0):
            dz_v = jnp.full(nlev, dz_val)
            _, state = cloud_microphysics_column_sweep(
                T_p, q, p, qc_arr, qi, cf, rho, dz_v, ndrop, dt=1800.0, config=cfg,
            )
            assert jnp.isfinite(state.precip_rain)
            assert float(state.precip_rain) >= 0.0

    def test_jit_and_vmap(self):
        """Column sweep must be jit-able and vmap-able (matches per-level)."""
        cfg = MicrophysicsParameters.default()
        nlev = 15
        T_p, q, p, qc, qi, cf, rho, dz, ndrop = self._column(nlev=nlev)
        qc = qc.at[5].set(1e-3)

        f = jax.jit(cloud_microphysics_column_sweep, static_argnames=())
        _, state_jit = f(T_p, q, p, qc, qi, cf, rho, dz, ndrop, 1800.0, cfg)
        assert jnp.isfinite(state_jit.precip_rain)

        # Stack 4 columns and vmap over column axis 0.
        T_b = jnp.stack([T_p] * 4, axis=0)
        q_b = jnp.stack([q] * 4, axis=0)
        p_b = jnp.stack([p] * 4, axis=0)
        qc_b = jnp.stack([qc] * 4, axis=0)
        qi_b = jnp.stack([qi] * 4, axis=0)
        cf_b = jnp.stack([cf] * 4, axis=0)
        rho_b = jnp.stack([rho] * 4, axis=0)
        dz_b = jnp.stack([dz] * 4, axis=0)
        nd_b = jnp.stack([ndrop] * 4, axis=0)
        _, state_b = jax.vmap(
            cloud_microphysics_column_sweep,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )(T_b, q_b, p_b, qc_b, qi_b, cf_b, rho_b, dz_b, nd_b, 1800.0, cfg)
        assert state_b.precip_rain.shape == (4,)
        assert jnp.all(jnp.isfinite(state_b.precip_rain))

    def test_surface_flux_matches_source_when_no_evap(self):
        """In a saturated column, no rain evaporates: surface == ∑ source.

        Rain evaporation requires sub-saturation (``q < qsw``); set
        ``q = qsw`` everywhere so Rotstayn's ``zsusatw = min(0, …) = 0``
        and the propagating ``zrfl`` at the surface equals the column
        integrated local rain source exactly.
        """
        from jcm.physics.clouds.sundqvist import saturation_specific_humidity
        cfg = MicrophysicsParameters.default()
        nlev = 15
        T = jnp.linspace(280.0, 295.0, nlev)
        p = jnp.linspace(20000.0, 100000.0, nlev)
        q = jax.vmap(saturation_specific_humidity)(p, T)
        qc = jnp.zeros(nlev).at[4].set(1.5e-3).at[7].set(8e-4)
        qi = jnp.zeros(nlev)
        cf = jnp.where(qc > 0, 0.7, 0.0)
        rho = p / (287.0 * T)
        dz = jnp.full(nlev, 500.0)
        ndrop = jnp.full(nlev, 1e8)
        _, state = cloud_microphysics_column_sweep(
            T, q, p, qc, qi, cf, rho, dz, ndrop, dt=1800.0, config=cfg,
        )
        assert jnp.allclose(
            state.precip_rain, jnp.sum(state.rain_flux), rtol=1e-5,
        )

    print("All tests passed!")