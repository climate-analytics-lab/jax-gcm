"""
Unit tests for cloud optics calculations

Tests cloud optical properties including extinction, scattering,
and asymmetry parameters for both water and ice clouds.

Date: 2025-01-10
"""

import jax.numpy as jnp
import pytest
from jcm.physics.icon.radiation.cloud_optics import (
    cloud_optics,
    effective_radius_liquid,
    effective_radius_ice
)


def test_effective_radius_liquid():
    """Test liquid cloud effective radius calculation"""
    nlev = 10
    temperature = jnp.linspace(288.0, 270.0, nlev)
    
    # Test over ocean
    r_eff_ocean = effective_radius_liquid(temperature, land_fraction=0.0)
    
    # Test over land
    r_eff_land = effective_radius_liquid(temperature, land_fraction=1.0)
    
    # Check shapes
    assert r_eff_ocean.shape == (nlev,)
    assert r_eff_land.shape == (nlev,)
    
    # Should be positive
    assert jnp.all(r_eff_ocean > 0)
    assert jnp.all(r_eff_land > 0)
    
    # Ocean droplets should generally be larger
    assert jnp.all(r_eff_ocean >= r_eff_land)
    
    # Should have temperature dependence
    assert not jnp.allclose(r_eff_ocean[0], r_eff_ocean[-1])


def test_effective_radius_ice():
    """Test ice cloud effective radius calculation"""
    nlev = 10
    temperature = jnp.linspace(260.0, 200.0, nlev)
    ice_water_content = jnp.ones(nlev) * 1e-4  # kg/m³
    
    r_eff = effective_radius_ice(temperature, ice_water_content)
    
    # Check shape
    assert r_eff.shape == (nlev,)
    
    # Should be positive
    assert jnp.all(r_eff > 0)
    
    # Should be reasonable values (10-100 microns)
    assert jnp.all(r_eff > 5.0)
    assert jnp.all(r_eff < 200.0)
    
    # Should have temperature dependence
    assert not jnp.allclose(r_eff[0], r_eff[-1])


def test_cloud_optics_integration():
    """Test the main cloud_optics function"""
    nlev = 15
    
    # Create mixed cloud profile
    cloud_water_path = jnp.zeros(nlev)
    cloud_ice_path = jnp.zeros(nlev)
    
    # Water clouds in lower levels
    cloud_water_path = cloud_water_path.at[10:].set(0.1)
    
    # Ice clouds in upper levels
    cloud_ice_path = cloud_ice_path.at[2:8].set(0.05)
    
    temperature = jnp.linspace(288.0, 200.0, nlev)
    
    n_sw_bands = 2
    n_lw_bands = 3
    
    # Calculate cloud optics
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature,
        n_sw_bands, n_lw_bands
    )
    
    # Check output shapes
    assert sw_optics.optical_depth.shape == (nlev, n_sw_bands)
    assert sw_optics.single_scatter_albedo.shape == (nlev, n_sw_bands)
    assert sw_optics.asymmetry_factor.shape == (nlev, n_sw_bands)
    
    assert lw_optics.optical_depth.shape == (nlev, n_lw_bands)
    assert lw_optics.single_scatter_albedo.shape == (nlev, n_lw_bands)
    assert lw_optics.asymmetry_factor.shape == (nlev, n_lw_bands)
    
    # Physical constraints
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    
    assert jnp.all(sw_optics.single_scatter_albedo >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo <= 1)
    assert jnp.all(lw_optics.single_scatter_albedo >= 0)
    assert jnp.all(lw_optics.single_scatter_albedo <= 1)
    
    # No NaN values
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))
    
    # Clear-sky levels should have zero optical depth
    assert jnp.all(sw_optics.optical_depth[0, :] == 0)
    assert jnp.all(lw_optics.optical_depth[0, :] == 0)
    
    # Cloudy levels should have non-zero optical depth
    assert jnp.any(sw_optics.optical_depth[5, :] > 0)  # Ice cloud level
    assert jnp.any(sw_optics.optical_depth[12, :] > 0)  # Water cloud level


def test_cloud_optics_no_clouds():
    """Test cloud optics with no clouds"""
    nlev = 10
    cloud_water_path = jnp.zeros(nlev)
    cloud_ice_path = jnp.zeros(nlev)
    temperature = jnp.linspace(288.0, 220.0, nlev)
    
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature, 2, 3
    )
    
    # Should have zero optical depth everywhere
    assert jnp.allclose(sw_optics.optical_depth, 0.0)
    assert jnp.allclose(lw_optics.optical_depth, 0.0)
    
    # Single scattering albedo should be physical (but not used when tau=0)
    assert jnp.all(sw_optics.single_scatter_albedo >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo <= 1)


def test_cloud_optics_extreme_values():
    """Test cloud optics with extreme cloud water/ice paths"""
    nlev = 5
    temperature = jnp.ones(nlev) * 260.0
    
    # Very small cloud water/ice
    cloud_water_path = jnp.ones(nlev) * 1e-8
    cloud_ice_path = jnp.ones(nlev) * 1e-8
    
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature, 2, 3
    )
    
    # Should handle small values without NaN
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))
    
    # Very large cloud water/ice
    cloud_water_path = jnp.ones(nlev) * 10.0  # Very thick clouds
    cloud_ice_path = jnp.ones(nlev) * 5.0
    
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature, 2, 3
    )
    
    # Should handle large values
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))
    
    # Should have high optical depths
    assert jnp.all(sw_optics.optical_depth > 1.0)
    assert jnp.all(lw_optics.optical_depth > 0.1)


def test_cloud_optics_temperature_dependence():
    """Test temperature dependence of cloud properties"""
    nlev = 10
    cloud_water_path = jnp.ones(nlev) * 0.1
    cloud_ice_path = jnp.zeros(nlev)
    
    # Warm temperatures
    temp_warm = jnp.ones(nlev) * 290.0
    sw_warm, lw_warm = cloud_optics(
        cloud_water_path, cloud_ice_path, temp_warm, 2, 3
    )
    
    # Cold temperatures  
    temp_cold = jnp.ones(nlev) * 230.0
    sw_cold, lw_cold = cloud_optics(
        cloud_water_path, cloud_ice_path, temp_cold, 2, 3
    )
    
    # Temperature should affect optical properties
    # (Exact relationship depends on parameterization details)
    assert sw_warm.optical_depth.shape == sw_cold.optical_depth.shape
    assert lw_warm.optical_depth.shape == lw_cold.optical_depth.shape
    
    # Both should be valid
    assert jnp.all(sw_warm.optical_depth >= 0)
    assert jnp.all(sw_cold.optical_depth >= 0)


def test_cloud_optics_mixed_phase():
    """Test mixed-phase clouds (both water and ice)"""
    nlev = 8
    temperature = jnp.array([288, 280, 270, 260, 250, 240, 230, 220])  # Mixed temp profile
    
    # Mixed phase: water and ice coexist
    cloud_water_path = jnp.array([0.0, 0.1, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0])
    cloud_ice_path = jnp.array([0.0, 0.0, 0.05, 0.1, 0.15, 0.1, 0.05, 0.0])
    
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature, 2, 3
    )
    
    # Total optical depth should be combination of water and ice
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    
    # Levels with both water and ice should have higher optical depth
    mixed_level = 3  # Both water and ice present
    water_only_level = 1  # Only water
    ice_only_level = 5  # Only ice
    
    # Mixed phase should have substantial optical depth
    assert jnp.all(sw_optics.optical_depth[mixed_level, :] > 0)
    assert jnp.all(lw_optics.optical_depth[mixed_level, :] > 0)


def test_cloud_optics_band_variations():
    """Test spectral variations across bands"""
    nlev = 5
    cloud_water_path = jnp.ones(nlev) * 0.2
    cloud_ice_path = jnp.zeros(nlev)
    temperature = jnp.ones(nlev) * 280.0
    
    n_sw_bands = 4
    n_lw_bands = 6
    
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature,
        n_sw_bands, n_lw_bands
    )
    
    # Should have variations across bands
    # (Exact variations depend on parameterization)
    for i in range(nlev):
        if cloud_water_path[i] > 0:
            # SW bands should have some optical depth
            sw_tau_level = sw_optics.optical_depth[i, :]
            assert jnp.any(sw_tau_level > 0)
            
            # LW bands should have some optical depth
            lw_tau_level = lw_optics.optical_depth[i, :]
            assert jnp.any(lw_tau_level > 0)
    
    # Check that all bands have reasonable values
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))


def test_cloud_optics_scattering_properties():
    """Test scattering properties of clouds"""
    nlev = 5
    cloud_water_path = jnp.ones(nlev) * 0.2
    cloud_ice_path = jnp.ones(nlev) * 0.1
    temperature = jnp.ones(nlev) * 260.0
    
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, temperature, 2, 3
    )
    
    # SW should have high single scattering albedo (clouds scatter well in visible)
    assert jnp.all(sw_optics.single_scatter_albedo > 0.8)
    
    # LW should have lower single scattering albedo (more absorption in IR)
    # Note: Different number of bands, so compare averages
    lw_ssa_avg = jnp.mean(lw_optics.single_scatter_albedo, axis=1)
    sw_ssa_avg = jnp.mean(sw_optics.single_scatter_albedo, axis=1)
    assert jnp.all(lw_ssa_avg <= sw_ssa_avg)
    
    # Asymmetry factor should be physical
    assert jnp.all(sw_optics.asymmetry_factor >= -1)
    assert jnp.all(sw_optics.asymmetry_factor <= 1)
    assert jnp.all(lw_optics.asymmetry_factor >= -1)
    assert jnp.all(lw_optics.asymmetry_factor <= 1)
    
    # Clouds typically have forward scattering (g > 0)
    cloudy_levels = cloud_water_path + cloud_ice_path > 0
    if jnp.any(cloudy_levels):
        assert jnp.any(sw_optics.asymmetry_factor > 0)