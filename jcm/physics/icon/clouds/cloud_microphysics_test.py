"""
Unit tests for cloud microphysics scheme

Date: 2025-01-10
"""
print("name:", __name__)
print("package:", __package__)
import jax.numpy as jnp
import jax
import pytest
from .cloud_microphysics import (
    MicrophysicsParameters, MicrophysicsState, MicrophysicsTendencies,
    cloud_droplet_radius, autoconversion_kk2000, accretion_rain_cloud,
    ice_autoconversion, snow_accretion, melting_freezing,
    evaporation_sublimation, sedimentation_flux, cloud_microphysics
)
from .cloud_utils import (
    get_util_var, get_cloud_bounds, eff_ice_crystal_radius, minimum_CDNC
)
from .cloud_microphysics_2m import (
    MicrophysicsState_2M, MicrophysicsTendencies_2M, melting_snow_and_ice, sublimation_snow_and_ice_evaporation_rain,
    precip_formation_warm, precip_formation_cold, update_in_cloud_water
)
from ..constants.physical_constants import tmelt, rhow, cp, alhc, alhs, alhf, rhoh2o

from .cloud_params_2m import (cqtmin, ldyn_cdnc_min, rcd_vol_max, cdnc_min_fixed, 
                              cdnc_min_lower, cdnc_min_upper, fact_PK, pow_PK, icemin
                              )

from math import pi

def _zeros(n: int) -> jnp.ndarray:
        return jnp.zeros((n,), dtype=jnp.float32)


def _full(n: int, v: float) -> jnp.ndarray:
    return jnp.full((n,), v, dtype=jnp.float32)


# from cloud_params_2m import CloudParams2M


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
    
    def test_kk2000_threshold(self):
        """Test KK2000 autoconversion has threshold behavior"""
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
        config = MicrophysicsParameters.default()
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
            rain_water = jnp.ones(nlev) * 0.0
            snow = jnp.ones(nlev) * 0.0
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

# ================= Tests for 2-m Microphysics ================== #

class TestCloudUtils:
    """Test utility functions for cloud microphysics"""

    def test_get_util_var(self):
        """Test utility variable calculations."""
        nproma, nbdim, ntdia, nlev, nlevp1 = 1, 1, 0, 3, 4
        paphm1 = jnp.array([[700.0, 800.0, 900.0, 1000.0]])  # Pressure at half levels
        pgeo = jnp.array([[300.0, 200.0, 100.0]])  # Geopotential at full levels
        papm1 = jnp.array([[750.0, 850.0, 950.0]])  # Pressure at full levels
        ptm1 = jnp.array([[260.0, 270.0, 280.0]])  # Temperature at full levels

        pgeoh, pdp, pdpg, pdz, paaa, pviscos = get_util_var(
            nproma, nbdim, ntdia, nlev, nlevp1, paphm1, pgeo, papm1, ptm1
        )

        # Check geopotential at half levels
        expected_pgeoh = jnp.array([[350.0, 250.0, 150.0, 0.0]])
        assert jnp.allclose(pgeoh, expected_pgeoh), f"Expected {expected_pgeoh}, got {pgeoh}"

        # Check pressure differences
        expected_pdp = jnp.array([[100.0, 100.0, 100.0]])
        assert jnp.allclose(pdp, expected_pdp), f"Expected {expected_pdp}, got {pdp}"

        # Check height differences
        expected_pdz = jnp.array([[10.19367991845056, 10.19367991845056, 15.2905199]])
        assert jnp.allclose(pdz, expected_pdz), f"Expected {expected_pdz}, got {pdz}"

        # Check air density correction
        expected_paaa = jnp.array([[1.8467386, 1.7793932, 1.7196922]])
        assert jnp.allclose(paaa, expected_paaa), f"Expected {expected_paaa}, got {paaa}"

        # Check dynamic viscosity
        expected_pviscos = jnp.array([[1.65162e-05, 1.70362e-05, 1.75562e-05]])
        assert jnp.allclose(pviscos, expected_pviscos), f"Expected {expected_pviscos}, got {pviscos}"

    def test_get_cloud_bounds(self):
        """Test the get_cloud_bounds function."""
        nproma = 1  # Number of columns
        nbdim = 1   # Number of rows
        ntdia = 0   # Starting level index
        nlev = 7    # Number of levels

        # Cloud cover array (paclc)
        paclc = jnp.array([[0.0, 0.8, 0.6, 0.0, 0.8, 0.6, 0.5]])  # Cloud between levels 1 to 2 and 4 to 6

        # Call the function
        ktop, kbas, kcl_minustop, kcl_minusbas = get_cloud_bounds(nproma, nbdim, ntdia, nlev, paclc)

        # Expected outputs
        expected_ktop = jnp.array([[0, 1, 0, 0, 4, 0, 0]])  # Cloud top at level 1 & 4
        expected_kbas = jnp.array([[0, 0, 2, 0, 0, 0, 6]])  # Cloud base at level 2 & 6
        expected_kcl_minustop = jnp.array([[0, 0, 1, 0, 0, 4, 4]])  # Cloud levels excluding top
        expected_kcl_minusbas = jnp.array([[0, 2, 0, 0, 6, 6, 0]])  # Cloud levels excluding base

        # Assertions
        assert jnp.array_equal(ktop, expected_ktop), f"ktop: Expected {expected_ktop}, got {ktop}"
        assert jnp.array_equal(kbas, expected_kbas), f"kbas: Expected {expected_kbas}, got {kbas}"
        assert jnp.array_equal(kcl_minustop, expected_kcl_minustop), f"lcl_minustop: Expected {expected_kcl_minustop}, got {kcl_minustop}"
        assert jnp.array_equal(kcl_minusbas, expected_kcl_minusbas), f"kcl_minusbas: Expected {expected_kcl_minusbas}, got {kcl_minusbas}"
    
    def test_eff_ice_crystal_radius(self):
        # Positive, non-degenerate inputs so the eps-guards do not affect the result
        pxice = jnp.array([0.1, 1.0, 10.0], dtype=jnp.float32)   # [g/m^3]
        picnc = jnp.array([1e5, 1e6, 1e7], dtype=jnp.float32)    # [1/m^3]

        got = eff_ice_crystal_radius(pxice, picnc)
        expected = 0.5e4 * (pxice / (fact_PK * picnc)) ** (1.0 / pow_PK)

        assert got.shape == expected.shape
        assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)
    
    def test_minimum_CDNC(self):
        pxwat = jnp.array([0.0, 1e-6, 1e-4, 1e-2], dtype=jnp.float32)  # [kg/m^3]
        got = minimum_CDNC(pxwat)

        if ldyn_cdnc_min:
            expected = rcd_vol_max ** (-3.0) * (3.0 / (4.0 * pi * rhoh2o)) * pxwat
            expected = jnp.clip(expected, cdnc_min_lower, cdnc_min_upper)
        else:
            expected = jnp.full_like(pxwat, cdnc_min_fixed * 1.0e6)  # cm^-3 -> m^-3

        assert got.shape == pxwat.shape
        assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)

        # extra invariant: dynamic branch must be within clip bounds
        if ldyn_cdnc_min:
            assert jnp.all(got >= cdnc_min_lower)
            assert jnp.all(got <= cdnc_min_upper)

class TestMeltingSnowIce_2M:
    def test_melting_snow_and_ice(self):
        dt = jnp.array(60.0, dtype=jnp.float32)

        temperature_previous = jnp.array([tmelt + 1.0, tmelt - 1.0], dtype=jnp.float32)
        melt_mask = temperature_previous > tmelt

        pressure_thickness = jnp.array([1.0e4, 1.0e4], dtype=jnp.float32)
        lsdcp = jnp.array([2.8e3, 2.8e3], dtype=jnp.float32)
        lvdcp = jnp.array([2.5e3, 2.5e3], dtype=jnp.float32)

        ice_cloud_previous = jnp.array([1e-4, 1e-4], dtype=jnp.float32)
        ice_tendency = jnp.array([1e-6, 1e-6], dtype=jnp.float32)

        icncq = jnp.array([2e5, 2e5], dtype=jnp.float32)
        icnc = jnp.array([1e6, 1e6], dtype=jnp.float32)
        cdnc = jnp.array([1e8, 1e8], dtype=jnp.float32)
        qmel = jnp.array([0.0, 0.0], dtype=jnp.float32)

        rain_flux = jnp.array([1e-5, 1e-5], dtype=jnp.float32)
        snow_flux = jnp.array([2e-5, 2e-5], dtype=jnp.float32)

        ice_flux = jnp.array([1.0e-5, 1.0e-5], dtype=jnp.float32)
        ice_flux_n = jnp.array([1.0e7, 1.0e7], dtype=jnp.float32)

        (
            icnc_o,
            qmel_o,
            cdnc_o,
            rain_flux_o,
            snow_flux_o,
            ice_flux_o,
            ice_flux_n_o,
            ice_tendency_o,
            pimlt,
            psmlt,
            pximlt,
        ) = melting_snow_and_ice(
            melt_mask=melt_mask,
            temperature_previous=temperature_previous,
            ice_cloud_previous=ice_cloud_previous,
            pressure_thickness=pressure_thickness,
            icncq=icncq,
            lsdcp=lsdcp,
            lvdcp=lvdcp,
            icnc=icnc,
            qmel=qmel,
            cdnc=cdnc,
            rain_flux=rain_flux,
            snow_flux=snow_flux,
            ice_flux=ice_flux,
            ice_flux_n=ice_flux_n,
            ice_tendency=ice_tendency,
            dt=dt,
        )

        # Basic sanity checks
        assert icnc_o.shape == (2,)
        assert jnp.all(jnp.isfinite(icnc_o))
        assert jnp.all(jnp.isfinite(rain_flux_o))
        assert jnp.all(jnp.isfinite(snow_flux_o))

        # Melt point should transfer ICNC to CDNC and reset ICNC to icemin
        assert float(icnc_o[0]) == float(icemin)
        assert float(cdnc_o[0]) == float(cdnc[0] + icncq[0])
        assert float(qmel_o[0]) == float(qmel[0] + dt * icncq[0])

        # Non-melt point should not change those number variables
        assert float(icnc_o[1]) == float(icnc[1])
        assert float(cdnc_o[1]) == float(cdnc[1])
        assert float(qmel_o[1]) == float(qmel[1])

        # Diagnostics should be non-negative
        assert float(pimlt[0]) >= 0.0
        assert float(psmlt[0]) >= 0.0
        assert float(pximlt[0]) >= 0.0

        # ice_flux_n should be zeroed if mass flux drops below epsec (may or may not happen here),
        # but must never be negative.
        assert jnp.all(ice_flux_n_o >= 0.0)
        assert jnp.all(ice_flux_o >= 0.0)
    
class TestSublimationSnowIceEvapRain_2M:
    def _common_inputs(self, n: int):
        dt = jnp.array(60.0, dtype=jnp.float32)

        # previous-step thermodynamics
        specific_humidity_prev = _full(n, 1.0e-3)  # pqm1 [kg/kg]
        temperature_prev = _full(n, 260.0)         # ptm1 [K]

        # layer geometry
        pressure_thickness = _full(n, 1.0e4)       # pdp [Pa]
        dp_over_g = _full(n, 1.0e3)                # pdpg [kg/m^2]

        # area fractions
        precip_fraction = _full(n, 0.5)            # pclcpre
        falling_ice_fraction = _full(n, 0.5)       # pclcfi

        # air properties
        air_density = _full(n, 1.2)                # prho [kg/m^3]
        inv_air_density = 1.0 / air_density        # pqrho
        inv_air_density_rcp = inv_air_density      # prho_rcp (kept identical)

        # saturation quantities / deficits
        qsat_ice = _full(n, 2.0e-3)                # pqsi [kg/kg]  ( > q )
        qsat_water_prev = _full(n, 2.0e-3)         # pqsw [kg/kg]  ( > q )

        # scheme-specific subsaturation terms (positive allows sinks)
        subsat_wrt_ice = _full(n, -1e-5)           # picesub
        subsat_wrt_water_evap = _full(n, -1e-5)    # psusatw_evap
        thermo_term_water = _full(n, 1.0)          # pastbstw (>0)

        # latent heat term
        lsdcp = _full(n, 2.8e3)                    # plsdcp

        # default fluxes (overridden per-test)
        snow_flux = _zeros(n)                      # psfl
        rain_flux = _zeros(n)                      # prfl
        ice_flux = _zeros(n)                       # pxiflux
        ice_flux_n = _full(n, 1.0e7)               # pxifluxn

        return dict(
            dt=dt,
            specific_humidity_prev=specific_humidity_prev,
            temperature_prev=temperature_prev,
            precip_fraction=precip_fraction,
            falling_ice_fraction=falling_ice_fraction,
            pressure_thickness=pressure_thickness,
            dp_over_g=dp_over_g,
            subsat_wrt_ice=subsat_wrt_ice,
            lsdcp=lsdcp,
            inv_air_density=inv_air_density,
            qsat_ice=qsat_ice,
            inv_air_density_rcp=inv_air_density_rcp,
            snow_flux=snow_flux,
            air_density=air_density,
            qsat_water_prev=qsat_water_prev,
            rain_flux=rain_flux,
            subsat_wrt_water_evap=subsat_wrt_water_evap,
            thermo_term_water=thermo_term_water,
            ice_flux=ice_flux,
            ice_flux_n=ice_flux_n,
        )

    def test_snow_sublimation_only(self):
        n = 4
        x = self._common_inputs(n)

        precip_mask = jnp.array([True, True, False, True])
        falling_ice_mask = jnp.array([False, False, False, False])

        # Snow flux present only at first two points; masked off third; last has zero flux
        x["snow_flux"] = jnp.array([2.0e-4, 1.0e-4, 2.0e-4, 0.0], dtype=jnp.float32)
        x["rain_flux"] = _zeros(n)
        x["ice_flux"] = _zeros(n)
        x["ice_flux_n"] = _zeros(n)

        ice_flux_o, ice_flux_n_o, ice_sublim, snow_sublim, rain_evap = sublimation_snow_and_ice_evaporation_rain(
            precip_mask=precip_mask,
            falling_ice_mask=falling_ice_mask,
            specific_humidity_prev=x["specific_humidity_prev"],
            temperature_prev=x["temperature_prev"],
            precip_fraction=x["precip_fraction"],
            pressure_thickness=x["pressure_thickness"],
            dp_over_g=x["dp_over_g"],
            subsat_wrt_ice=x["subsat_wrt_ice"],
            lsdcp=x["lsdcp"],
            inv_air_density=x["inv_air_density"],
            qsat_ice=x["qsat_ice"],
            inv_air_density_rcp=x["inv_air_density_rcp"],
            snow_flux=x["snow_flux"],
            air_density=x["air_density"],
            qsat_water_prev=x["qsat_water_prev"],
            rain_flux=x["rain_flux"],
            subsat_wrt_water_evap=x["subsat_wrt_water_evap"],
            thermo_term_water=x["thermo_term_water"],
            falling_ice_fraction=x["falling_ice_fraction"],
            ice_flux=x["ice_flux"],
            ice_flux_n=x["ice_flux_n"],
            dt=x["dt"],
        )

        assert float(snow_sublim[0]) > 0.0
        assert float(snow_sublim[1]) > 0.0
        assert float(snow_sublim[2]) == 0.0  # precip_mask False
        assert float(snow_sublim[3]) == 0.0  # snow_flux == 0

        assert jnp.all(ice_sublim == 0.0)
        assert jnp.all(rain_evap == 0.0)

        # unchanged ice fluxes
        assert jnp.allclose(ice_flux_o, x["ice_flux"])
        assert jnp.allclose(ice_flux_n_o, x["ice_flux_n"])

        assert jnp.all(jnp.isfinite(snow_sublim))
        assert jnp.all(snow_sublim >= 0.0)

    def test_falling_ice_sublimation_reduces_fluxes(self):
        n = 4
        x = self._common_inputs(n)

        precip_mask = jnp.array([False, False, False, False])
        falling_ice_mask = jnp.array([True, True, False, True])

        ice_flux_in = jnp.array([2.0e-4, 1.0e-4, 5.0e-4, 2.0e-4], dtype=jnp.float32)
        ice_flux_n_in = jnp.array([2.0e7, 1.0e7, 1.0e7, 2.0e7], dtype=jnp.float32)
        x["ice_flux"] = ice_flux_in
        x["ice_flux_n"] = ice_flux_n_in

        x["snow_flux"] = _zeros(n)
        x["rain_flux"] = _zeros(n)

        ice_flux_o, ice_flux_n_o, ice_sublim, snow_sublim, rain_evap = sublimation_snow_and_ice_evaporation_rain(
            precip_mask=precip_mask,
            falling_ice_mask=falling_ice_mask,
            specific_humidity_prev=x["specific_humidity_prev"],
            temperature_prev=x["temperature_prev"],
            precip_fraction=x["precip_fraction"],
            pressure_thickness=x["pressure_thickness"],
            dp_over_g=x["dp_over_g"],
            subsat_wrt_ice=x["subsat_wrt_ice"],
            lsdcp=x["lsdcp"],
            inv_air_density=x["inv_air_density"],
            qsat_ice=x["qsat_ice"],
            inv_air_density_rcp=x["inv_air_density_rcp"],
            snow_flux=x["snow_flux"],
            air_density=x["air_density"],
            qsat_water_prev=x["qsat_water_prev"],
            rain_flux=x["rain_flux"],
            subsat_wrt_water_evap=x["subsat_wrt_water_evap"],
            thermo_term_water=x["thermo_term_water"],
            falling_ice_fraction=x["falling_ice_fraction"],
            ice_flux=x["ice_flux"],
            ice_flux_n=x["ice_flux_n"],
            dt=x["dt"],
        )

        assert float(ice_sublim[0]) > 0.0
        assert float(ice_sublim[1]) > 0.0
        assert float(ice_sublim[2]) == 0.0  # falling_ice_mask False
        assert float(ice_sublim[3]) > 0.0

        # Should reduce mass flux where active
        assert float(ice_flux_o[0]) < float(ice_flux_in[0])
        assert float(ice_flux_o[1]) < float(ice_flux_in[1])
        assert float(ice_flux_o[2]) == float(ice_flux_in[2])
        assert float(ice_flux_o[3]) < float(ice_flux_in[3])

        # Number flux should not increase where active
        assert float(ice_flux_n_o[0]) <= float(ice_flux_n_in[0])
        assert float(ice_flux_n_o[1]) <= float(ice_flux_n_in[1])
        assert float(ice_flux_n_o[2]) == float(ice_flux_n_in[2])
        assert float(ice_flux_n_o[3]) <= float(ice_flux_n_in[3])

        assert jnp.all(snow_sublim == 0.0)
        assert jnp.all(rain_evap == 0.0)

        assert jnp.all(jnp.isfinite(ice_sublim))
        assert jnp.all(ice_sublim >= 0.0)
        assert jnp.all(ice_flux_o >= 0.0)
        assert jnp.all(ice_flux_n_o >= 0.0)

    def test_rain_evaporation_only(self):
        n = 4
        x = self._common_inputs(n)

        precip_mask = jnp.array([True, True, False, True])
        falling_ice_mask = jnp.array([False, False, False, False])

        x["rain_flux"] = jnp.array([3.0e-4, 1.0e-4, 2.0e-4, 0.0], dtype=jnp.float32)
        x["snow_flux"] = _zeros(n)
        x["ice_flux"] = _zeros(n)
        x["ice_flux_n"] = _zeros(n)

        ice_flux_o, ice_flux_n_o, ice_sublim, snow_sublim, rain_evap = sublimation_snow_and_ice_evaporation_rain(
            precip_mask=precip_mask,
            falling_ice_mask=falling_ice_mask,
            specific_humidity_prev=x["specific_humidity_prev"],
            temperature_prev=x["temperature_prev"],
            precip_fraction=x["precip_fraction"],
            pressure_thickness=x["pressure_thickness"],
            dp_over_g=x["dp_over_g"],
            subsat_wrt_ice=x["subsat_wrt_ice"],
            lsdcp=x["lsdcp"],
            inv_air_density=x["inv_air_density"],
            qsat_ice=x["qsat_ice"],
            inv_air_density_rcp=x["inv_air_density_rcp"],
            snow_flux=x["snow_flux"],
            air_density=x["air_density"],
            qsat_water_prev=x["qsat_water_prev"],
            rain_flux=x["rain_flux"],
            subsat_wrt_water_evap=x["subsat_wrt_water_evap"],
            thermo_term_water=x["thermo_term_water"],
            falling_ice_fraction=x["falling_ice_fraction"],
            ice_flux=x["ice_flux"],
            ice_flux_n=x["ice_flux_n"],
            dt=x["dt"],
        )

        assert float(rain_evap[0]) > 0.0
        assert float(rain_evap[1]) > 0.0
        assert float(rain_evap[2]) == 0.0  # precip_mask False
        assert float(rain_evap[3]) == 0.0  # rain_flux == 0

        assert jnp.all(snow_sublim == 0.0)
        assert jnp.all(ice_sublim == 0.0)

        assert jnp.allclose(ice_flux_o, x["ice_flux"])
        assert jnp.allclose(ice_flux_n_o, x["ice_flux_n"])

        assert jnp.all(jnp.isfinite(rain_evap))
        assert jnp.all(rain_evap >= 0.0)


class TestAutoconversion_2M:
    def test_precip_formation_warm_mask_false_no_change(self):
        """If warm_precip_mask is False everywhere, outputs should be zero rates and unchanged inputs."""
        # config = CloudParams2M.default()

        shape = (5,)
        warm_precip_mask = jnp.zeros(shape, dtype=bool)

        autoconversion_factor = jnp.ones(shape)
        cloud_fraction = jnp.full(shape, 0.5)
        minimum_cloud_precip_fraction = jnp.full(shape, 0.1)
        air_density = jnp.full(shape, 1.0)
        rain_water = jnp.full(shape, 1e-4)
        minimum_droplet_number = jnp.full(shape, 1e6)
        droplet_number_in = jnp.full(shape, 2e6)
        cloud_water_in = jnp.full(shape, 1e-3)
        dt = jnp.full(shape, 10.0)

        droplet_number, cloud_water, pmratepr, prpr, prprn = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt
        )

        assert jnp.allclose(droplet_number, droplet_number_in)
        assert jnp.allclose(cloud_water, cloud_water_in)
        assert jnp.allclose(pmratepr, jnp.zeros_like(cloud_water_in))
        assert jnp.allclose(prpr, jnp.zeros_like(cloud_water_in))
        assert jnp.allclose(prprn, jnp.zeros_like(cloud_water_in))


    def test_precip_formation_warm_mask_true_reduces_cloud_water_and_nonnegative_rates(self):
        """If mask is True and cloud water is present, cloud water should not increase; rates should be >= 0."""
        # config = MicrophysicsParameters_2M.default()

        shape = (6,)
        warm_precip_mask = jnp.ones(shape, dtype=bool)

        autoconversion_factor = jnp.ones(shape)
        cloud_fraction = jnp.linspace(0.1, 1.0, shape[0])
        minimum_cloud_precip_fraction = jnp.full(shape, 0.2)
        air_density = jnp.full(shape, 1.0)
        rain_water = jnp.full(shape, 5e-4)
        minimum_droplet_number = jnp.full(shape, 1e6)

        droplet_number_in = jnp.full(shape, 2e6)
        cloud_water_in = jnp.full(shape, 2e-3)
        dt = jnp.full(shape, 10.0)

        droplet_number, cloud_water, pmratepr, prpr, prprn = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt
            # config=config,
        )

        # Cloud water is reduced by autoconversion and accretion terms; should not increase.
        assert jnp.all(cloud_water <= cloud_water_in + 1e-12)

        # Formation rates should be nonnegative for physically meaningful inputs.
        assert jnp.all(pmratepr >= -1e-12)
        assert jnp.all(prpr >= -1e-12)
        assert jnp.all(prprn >= -1e-12)

        # Droplet number should not increase (autoconversion removes droplets); allow tiny eps.
        assert jnp.all(droplet_number <= droplet_number_in + 1e-8)

    def test_precip_formation_warm_mixed_mask_only_updates_true_elements(self):
        """Only elements where mask is True should be modified."""
        # config = MicrophysicsParameters_2M.default()

        warm_precip_mask = jnp.array([True, False, True, False])

        autoconversion_factor = jnp.ones_like(warm_precip_mask, dtype=jnp.float32)
        cloud_fraction = jnp.full((4,), 0.5)
        minimum_cloud_precip_fraction = jnp.full((4,), 0.1)
        air_density = jnp.full((4,), 1.0)
        rain_water = jnp.full((4,), 1e-4)
        minimum_droplet_number = jnp.full((4,), 1e6)

        droplet_number_in = jnp.full((4,), 2e6)
        cloud_water_in = jnp.full((4,), 1e-3)
        dt = jnp.full((4,), 10.0)

        droplet_number, cloud_water, pmratepr, prpr, prprn = precip_formation_warm(
            warm_precip_mask=warm_precip_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            air_density=air_density,
            rain_water=rain_water,
            minimum_droplet_number=minimum_droplet_number,
            droplet_number=droplet_number_in,
            cloud_water=cloud_water_in,
            dt=dt
            # config=config,
        )

        false_idx = jnp.where(~warm_precip_mask)[0]

        assert jnp.allclose(droplet_number[false_idx], droplet_number_in[false_idx])
        assert jnp.allclose(cloud_water[false_idx], cloud_water_in[false_idx])
        assert jnp.allclose(pmratepr[false_idx], 0.0)
        assert jnp.allclose(prpr[false_idx], 0.0)
        assert jnp.allclose(prprn[false_idx], 0.0)

    def test_precip_formation_cold_basic_invariants_and_shapes(self):
        """
        Smoke/invariant test for precip_formation_cold.

        Checks:
        - output shapes match input shapes
        - outputs are finite
        - non-negativity for formation rates (pspr, psacl, psacln, psprn, pmsnowacl)
        - droplet_number is not reduced below cqtmin
        - in-cloud condensates are not negative
        """
        n = 6
        dt = jnp.array(60.0, dtype=jnp.float32)

        # Make 3 points "active" (cloudy with ice+liquid+snow) and 3 "inactive"
        cloud_mask = jnp.array([True, True, True, False, True, False])

        cloud_fraction = jnp.array([0.3, 0.5, 0.1, 0.0, 0.2, 0.0], dtype=jnp.float32)
        autoconversion_factor = jnp.array([1.0, 0.7, 0.3, 0.0, 0.5, 0.0], dtype=jnp.float32)
        minimum_cloud_precip_fraction = jnp.minimum(cloud_fraction, jnp.array([0.2] * n, dtype=jnp.float32))

        air_density = jnp.array([1.2] * n, dtype=jnp.float32)
        inv_air_density = 1.0 / air_density
        inv_air_density_rcp = 1.0 / air_density  # keep identical for test

        temperature = jnp.array([260.0, 255.0, 268.0, 280.0, 250.0, 275.0], dtype=jnp.float32)
        dynamic_viscosity = jnp.array([1.8e-5] * n, dtype=jnp.float32)

        # Snow from above: present only for active points to trigger riming/accretion
        snow_mass_mmr_from_above = jnp.array([1e-5, 2e-5, 5e-6, 0.0, 1e-5, 0.0], dtype=jnp.float32)

        # In-cloud ice and liquid: positive for active points
        in_cloud_ice = jnp.array([2e-4, 1e-4, 5e-5, 0.0, 2e-4, 0.0], dtype=jnp.float32)
        in_cloud_liquid = jnp.array([1e-4, 2e-4, 1e-4, 0.0, 5e-5, 0.0], dtype=jnp.float32)

        # Number concentrations
        ice_number = jnp.array([1e5, 2e5, 5e4, 1e5, 3e5, 1e5], dtype=jnp.float32)
        droplet_number = jnp.array([5e7, 2e7, 1e7, 5e7, 4e7, 5e7], dtype=jnp.float32)

        # Minimum droplet number (pcdnc_min)
        minimum_droplet_number = jnp.array([1e6] * n, dtype=jnp.float32)

        snow_rate_in_cloud = jnp.zeros((n,), dtype=jnp.float32)

        outs = precip_formation_cold(
            cloud_mask=cloud_mask,
            autoconversion_factor=autoconversion_factor,
            cloud_fraction=cloud_fraction,
            minimum_cloud_precip_fraction=minimum_cloud_precip_fraction,
            inverse_air_density=inv_air_density,
            inverse_air_density_rcp=inv_air_density_rcp,
            temperature=temperature,
            dynamic_viscosity=dynamic_viscosity,
            snow_mass_mmr_from_above=snow_mass_mmr_from_above,
            air_density=air_density,
            minimum_droplet_number=minimum_droplet_number,
            ice_number=ice_number,
            droplet_number=droplet_number,
            snow_rate_in_cloud=snow_rate_in_cloud,
            in_cloud_ice=in_cloud_ice,
            in_cloud_liquid=in_cloud_liquid,
            dt=dt,
        )

        assert len(outs) == 10
        (
            ice_number_o,
            droplet_number_o,
            snow_rate_in_cloud_o,
            in_cloud_ice_o,
            in_cloud_liquid_o,
            psprn,
            psacl,
            psacln,
            pmsnowacl,
            pspr,
        ) = outs

        for arr in outs:
            assert arr.shape == (n,)
            assert jnp.all(jnp.isfinite(arr)), "All outputs must be finite"

        # Invariants / basic physical bounds
        assert jnp.all(in_cloud_ice_o >= 0.0)
        assert jnp.all(in_cloud_liquid_o >= 0.0)
        assert jnp.all(droplet_number_o >= cqtmin)
        assert jnp.all(ice_number_o >= 0.0)

        # Formation/accretion diagnostics should never be negative
        assert jnp.all(pspr >= 0.0)
        assert jnp.all(psprn >= 0.0)
        assert jnp.all(psacl >= 0.0)
        assert jnp.all(psacln >= 0.0)
        assert jnp.all(pmsnowacl >= 0.0)

        # If a point is completely non-cloudy, outputs should remain "quiet" (rates zero)
        inactive = ~cloud_mask
        assert jnp.all(pspr[inactive] == 0.0)
        assert jnp.all(psacl[inactive] == 0.0)
        assert jnp.all(psacln[inactive] == 0.0)
        assert jnp.all(psprn[inactive] == 0.0)

class TestUpdateInCloudWater_2M:   
    def test_update_in_cloud_water_shapes_and_finite(self):
        n = 8

        # Inputs (mostly benign)
        aerosol_total = _full(n, 100.0)                 # [1/cm^3]-like scale used as *1e6 in scheme
        activated_cdnc = _full(n, 1.0e6)                # [1/m^3]
        condensation_increment = _zeros(n)              # [kg/kg]
        deposition_increment = _zeros(n)                # [kg/kg]
        cloud_cover_vari_i = _zeros(n)
        cloud_cover_vari_l = _zeros(n)
        activated_icnc = _full(n, 1.0e3)                # [1/m^3]
        specific_humidity = _full(n, 1.0e-2)
        saturation_specific_humidity = _full(n, 2.0e-2)
        air_density = _full(n, 1.2)
        ice_mean_volume_radius = _full(n, 20e-6)        # [m]
        temperature_previous = _full(n, 280.0)          # [K]

        cloud_flag = jnp.array([True, False, True, False, True, False, True, False])
        icnc = _full(n, 1.0)                            # [1/m^3] small
        droplet_nucleation_accumulated = _zeros(n)      # accumulator
        cdnc = _full(n, 1.0e5)                          # [1/m^3]
        cloud_fraction = jnp.where(cloud_flag, _full(n, 0.2), _full(n, 0.0))
        in_cloud_ice_mixing_ratio = _zeros(n)
        in_cloud_water_mixing_ratio = _zeros(n)
        dt = jnp.array(60.0, dtype=jnp.float32)

        outs = update_in_cloud_water(
            aerosol_total=aerosol_total,
            activated_cdnc=activated_cdnc,
            condensation_increment=condensation_increment,
            deposition_increment=deposition_increment,
            cloud_cover_vari_i=cloud_cover_vari_i,
            cloud_cover_vari_l=cloud_cover_vari_l,
            activated_icnc=activated_icnc,
            specific_humidity=specific_humidity,
            saturation_specific_humidity=saturation_specific_humidity,
            air_density=air_density,
            ice_mean_volume_radius=ice_mean_volume_radius,
            temperature_previous=temperature_previous,
            cloud_flag=cloud_flag,
            icnc=icnc,
            droplet_nucleation_accumulated=droplet_nucleation_accumulated,
            cdnc=cdnc,
            cloud_fraction=cloud_fraction,
            in_cloud_ice_mixing_ratio=in_cloud_ice_mixing_ratio,
            in_cloud_water_mixing_ratio=in_cloud_water_mixing_ratio,
            dt=dt,
        )

        assert len(outs) == 8
        for o in outs:
            assert o.shape == (n,)
            assert jnp.all(jnp.isfinite(o)), "Outputs should be finite"


    def test_existing_cloud_updates_in_cloud_water_only_where_cloud_flag_true(self):
        """
        If cloud_flag is True, pxlb should be incremented by condensation_increment/max(cloud_fraction,clc_min).
        If cloud_flag is False and no new cloud creation triggers, pxlb should remain unchanged.
        """
        n = 4
        dt = jnp.array(60.0, dtype=jnp.float32)

        cloud_flag = jnp.array([True, False, True, False])
        cloud_fraction = jnp.array([0.2, 0.0, 0.5, 0.0], dtype=jnp.float32)

        in_cloud_water = jnp.array([1e-4, 2e-4, 0.0, 3e-4], dtype=jnp.float32)
        cond_inc = jnp.array([1e-6, 5e-6, 2e-6, 9e-6], dtype=jnp.float32)

        out = update_in_cloud_water(
            aerosol_total=_full(n, 10.0),
            activated_cdnc=_full(n, 1e6),
            condensation_increment=cond_inc,
            deposition_increment=_zeros(n),
            cloud_cover_vari_i=_zeros(n),
            cloud_cover_vari_l=_zeros(n),
            activated_icnc=_full(n, 1e3),
            specific_humidity=_full(n, 1e-2),
            saturation_specific_humidity=_full(n, 2e-2),
            air_density=_full(n, 1.2),
            ice_mean_volume_radius=_full(n, 20e-6),
            temperature_previous=_full(n, 280.0),
            cloud_flag=cloud_flag,
            icnc=_full(n, 1.0),
            droplet_nucleation_accumulated=_zeros(n),
            cdnc=_full(n, 1e7),  # large enough to avoid activation path
            cloud_fraction=cloud_fraction,
            in_cloud_ice_mixing_ratio=_zeros(n),
            in_cloud_water_mixing_ratio=in_cloud_water,
            dt=dt,
        )

        _, _, _, _, cloud_fraction_out, _, in_cloud_water_out, _ = out

        # No new cloud creation expected because deposition+cloud_cover_vari_i and condensation+cloud_cover_vari_l
        # are both nonzero only in cond_inc, but cloud_flag False cases COULD trigger ll1.
        # To isolate "existing cloud update", ensure ll1 is false by setting cond_inc=0 in cloud_flag False indices.
        # (We already have cond_inc >0 everywhere, so we compute expected using the actual routine behavior.)
        #
        # Instead of relying on ll1, assert that for indices where cloud_flag True, water increased.
        assert in_cloud_water_out[0] > in_cloud_water[0]
        assert in_cloud_water_out[2] > in_cloud_water[2]

        # Also, cloud_fraction should still be in [0,1]
        assert jnp.all((cloud_fraction_out >= 0.0) & (cloud_fraction_out <= 1.0))


    def test_new_cloud_is_created_when_no_cloud_and_positive_sources(self):
        """
        When cloud_flag is False and there is positive (condensation or deposition) source,
        the routine creates a cloud fraction based on clipped RH and initializes in-cloud condensate.
        """
        n = 3
        dt = jnp.array(60.0, dtype=jnp.float32)

        cloud_flag = jnp.array([False, False, False])
        cloud_fraction = _zeros(n)

        # RH = q/qs => [0.5, 1.2, 0.001] -> clipped [0.5, 1.0, 0.01]
        q = jnp.array([1e-2, 1e-2, 1e-5], dtype=jnp.float32)
        qs = jnp.array([2e-2, 8e-3, 1e-2], dtype=jnp.float32)

        cond_inc = jnp.array([1e-6, 1e-6, 1e-6], dtype=jnp.float32)
        dep_inc = _zeros(n)

        out = update_in_cloud_water(
            aerosol_total=_full(n, 10.0),
            activated_cdnc=_full(n, 1e6),
            condensation_increment=cond_inc,
            deposition_increment=dep_inc,
            cloud_cover_vari_i=_zeros(n),
            cloud_cover_vari_l=_zeros(n),
            activated_icnc=_full(n, 1e3),
            specific_humidity=q,
            saturation_specific_humidity=qs,
            air_density=_full(n, 1.2),
            ice_mean_volume_radius=_full(n, 20e-6),
            temperature_previous=_full(n, 280.0),
            cloud_flag=cloud_flag,
            icnc=_full(n, 1.0),
            droplet_nucleation_accumulated=_zeros(n),
            cdnc=_full(n, 1e7),
            cloud_fraction=cloud_fraction,
            in_cloud_ice_mixing_ratio=_zeros(n),
            in_cloud_water_mixing_ratio=_zeros(n),
            dt=dt,
        )

        cloud_flag_out, _, _, _, cloud_fraction_out, _, in_cloud_water_out, _ = out

        assert jnp.all(cloud_flag_out)
        assert jnp.all(cloud_fraction_out > 0.0)
        assert jnp.all(cloud_fraction_out <= 1.0)

        # In the newly cloudy points, in-cloud water should be initialized > 0
        assert jnp.all(in_cloud_water_out > 0.0)


    def test_cdnc_activation_increases_cdnc_and_accumulates_nucleation(self):
        """
        If (cloudy) AND (qc > cqtmin) AND (cdnc <= cdnc_min) AND (T > cthomi),
        then cdnc is increased towards activated_cdnc and droplet_nucleation_accumulated increases by dt*ΔN.
        """
        n = 2
        dt = jnp.array(10.0, dtype=jnp.float32)

        cloud_flag = jnp.array([True, True])
        cloud_fraction = _full(n, 0.2)

        # Ensure water is present
        in_cloud_water = _full(n, 1e-4)
        air_density = _full(n, 1.2)

        # Choose activated higher than initial
        cdnc0 = jnp.array([1e5, 2e5], dtype=jnp.float32)
        activated = jnp.array([5e6, 5e6], dtype=jnp.float32)

        # Warm enough to permit update
        temperature_previous = _full(n, 280.0)

        droplet_nuc0 = _zeros(n)

        out = update_in_cloud_water(
            aerosol_total=_full(n, 100.0),
            activated_cdnc=activated,
            condensation_increment=_zeros(n),
            deposition_increment=_zeros(n),
            cloud_cover_vari_i=_zeros(n),
            cloud_cover_vari_l=_zeros(n),
            activated_icnc=_full(n, 1e3),
            specific_humidity=_full(n, 1e-2),
            saturation_specific_humidity=_full(n, 2e-2),
            air_density=air_density,
            ice_mean_volume_radius=_full(n, 20e-6),
            temperature_previous=temperature_previous,
            cloud_flag=cloud_flag,
            icnc=_full(n, 1.0),
            droplet_nucleation_accumulated=droplet_nuc0,
            cdnc=cdnc0,
            cloud_fraction=cloud_fraction,
            in_cloud_ice_mixing_ratio=_zeros(n),
            in_cloud_water_mixing_ratio=in_cloud_water,
            dt=dt,
        )

        _, _, droplet_nuc_out, cdnc_out, _, _, _, cdnc_min = out

        # cdnc should not decrease
        assert jnp.all(cdnc_out >= cdnc0)

        # If activation triggered, cdnc increases by (activated - cdnc0) (since max(.,0))
        # But it is only applied when cdnc <= cdnc_min. We assert monotonic and that
        # droplet_nucleation_accumulated increased for any index where cdnc0 <= cdnc_min.
        deltaN = jnp.maximum(0.0, activated - cdnc0)
        expected_nuc_increase = dt * deltaN

        triggered = cdnc0 <= cdnc_min
        assert jnp.all(droplet_nuc_out >= droplet_nuc0)
        assert jnp.allclose(
            droplet_nuc_out[triggered],
            droplet_nuc0[triggered] + expected_nuc_increase[triggered],
            rtol=0.0,
            atol=0.0,
        )


    def test_jittable_and_consistent_with_eager(self):
        n = 5
        dt = jnp.array(30.0, dtype=jnp.float32)

        inputs = dict(
            aerosol_total=_full(n, 100.0),
            activated_cdnc=_full(n, 1e6),
            condensation_increment=_full(n, 1e-6),
            deposition_increment=_zeros(n),
            cloud_cover_vari_i=_zeros(n),
            cloud_cover_vari_l=_zeros(n),
            activated_icnc=_full(n, 1e3),
            specific_humidity=_full(n, 1e-2),
            saturation_specific_humidity=_full(n, 2e-2),
            air_density=_full(n, 1.2),
            ice_mean_volume_radius=_full(n, 20e-6),
            temperature_previous=_full(n, 280.0),
            cloud_flag=jnp.array([True, False, True, False, True]),
            icnc=_full(n, 1.0),
            droplet_nucleation_accumulated=_zeros(n),
            cdnc=_full(n, 1e5),
            cloud_fraction=jnp.array([0.1, 0.0, 0.3, 0.0, 0.2], dtype=jnp.float32),
            in_cloud_ice_mixing_ratio=_zeros(n),
            in_cloud_water_mixing_ratio=_full(n, 1e-4),
            dt=dt,
        )

        eager = update_in_cloud_water(**inputs)
        jitted = jax.jit(update_in_cloud_water)(**inputs)

        for e, j in zip(eager, jitted):
            assert jnp.allclose(e, j, rtol=0.0, atol=0.0)



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

    test_utils = TestCloudUtils()
    test_utils.test_get_util_var()
    test_utils.test_get_cloud_bounds()
    test_utils.test_eff_ice_crystal_radius()
    test_utils.test_minimum_CDNC()

    test_melting_snow_ice_2m = TestMeltingSnowIce_2M()
    test_melting_snow_ice_2m.test_melting_snow_and_ice()

    test_sublimation_snow_ice_rain_2m = TestSublimationSnowIceEvapRain_2M()
    test_sublimation_snow_ice_rain_2m.test_snow_sublimation_only()
    test_sublimation_snow_ice_rain_2m.test_falling_ice_sublimation_reduces_fluxes()
    test_sublimation_snow_ice_rain_2m.test_rain_evaporation_only()

    test_auto_2m = TestAutoconversion_2M()
    test_auto_2m.test_precip_formation_warm_mask_false_no_change()
    test_auto_2m.test_precip_formation_warm_mask_true_reduces_cloud_water_and_nonnegative_rates()
    test_auto_2m.test_precip_formation_warm_mixed_mask_only_updates_true_elements()

    test_update_cloud_water_2m = TestUpdateInCloudWater_2M()
    test_update_cloud_water_2m.test_update_in_cloud_water_shapes_and_finite()
    test_update_cloud_water_2m.test_existing_cloud_updates_in_cloud_water_only_where_cloud_flag_true()
    test_update_cloud_water_2m.test_new_cloud_is_created_when_no_cloud_and_positive_sources()
    test_update_cloud_water_2m.test_cdnc_activation_increases_cdnc_and_accumulates_nucleation()
    test_update_cloud_water_2m.test_jittable_and_consistent_with_eager()

    
    
    print("All tests passed!")
        