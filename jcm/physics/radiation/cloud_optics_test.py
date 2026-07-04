"""Unit tests for cloud optics calculations

Tests cloud optical properties including extinction, scattering,
and asymmetry parameters for both water and ice clouds.

Date: 2025-01-10
"""

import jax.numpy as jnp
from jcm.physics.radiation.cloud_optics import (
    cloud_optics,
    effective_radius_liquid,
    effective_radius_ice
)


def test_effective_radius_liquid():
    """Test liquid cloud effective radius calculation"""
    cdnc_factor = jnp.array(1.0)  # No aerosol influence
    # Test over ocean
    r_eff_ocean = effective_radius_liquid(cdnc_factor, land_fraction=0.0)

    # Test over land
    r_eff_land = effective_radius_liquid(cdnc_factor, land_fraction=1.0)
    
    # Check that we get scalar outputs
    assert r_eff_ocean.shape == ()
    assert r_eff_land.shape == ()
    
    # Should be positive
    assert r_eff_ocean > 0
    assert r_eff_land > 0
    
    # Ocean droplets should generally be larger
    assert r_eff_ocean >= r_eff_land
    

def test_effective_radius_ice():
    """Moss/Foot power law: r_eff = 83.8 * IWC^0.216 (in-cloud IWC, g/m3).

    ECHAM mo_cloud_optics.f90:358 reference values.
    """
    # 0.01 g/m3 -> 83.8 * 0.01**0.216 ~ 31 um
    r_mid = effective_radius_ice(jnp.array(0.01))
    assert jnp.allclose(r_mid, 83.8 * 0.01**0.216, rtol=1e-6)
    assert 30.0 < r_mid < 32.0

    # Thin cirrus, 1e-4 g/m3 -> ~11.4 um (the fabricated T-ramp formula
    # produced 40-160 um here, saturating the RRTMGP LUT edge).
    r_thin = effective_radius_ice(jnp.array(1e-4))
    assert jnp.allclose(r_thin, 83.8 * 1e-4**0.216, rtol=1e-6)
    assert r_thin < 15.0

    # Monotonically increasing with IWC, profile shape preserved
    iwc = jnp.logspace(-5, 0, 10)
    r_eff = effective_radius_ice(iwc)
    assert r_eff.shape == (10,)
    assert jnp.all(jnp.diff(r_eff) > 0)

    # Zero-IWC guard: finite, positive value and a finite gradient
    # (double-where around the x**0.216 singularity at x = 0).
    import jax
    r_zero, grad_zero = jax.value_and_grad(
        lambda x: effective_radius_ice(x)
    )(jnp.array(0.0))
    assert jnp.isfinite(r_zero) and r_zero > 0
    assert jnp.isfinite(grad_zero)


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
    
    layer_thickness = jnp.full(nlev, 500.0)  # m

    # Calculate cloud optics
    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Check output shapes - now using fixed bands
    from jcm.physics.radiation.constants import N_SW_BANDS, N_LW_BANDS
    assert sw_optics.optical_depth.shape == (nlev, N_SW_BANDS)
    assert sw_optics.single_scatter_albedo.shape == (nlev, N_SW_BANDS)
    assert sw_optics.asymmetry_factor.shape == (nlev, N_SW_BANDS)
    
    assert lw_optics.optical_depth.shape == (nlev, N_LW_BANDS)
    assert lw_optics.single_scatter_albedo.shape == (nlev, N_LW_BANDS)
    assert lw_optics.asymmetry_factor.shape == (nlev, N_LW_BANDS)
    
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
    layer_thickness = jnp.full(nlev, 500.0)

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
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
    layer_thickness = jnp.full(nlev, 500.0)

    # Very small cloud water/ice
    cloud_water_path = jnp.ones(nlev) * 1e-8
    cloud_ice_path = jnp.ones(nlev) * 1e-8

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )

    # Should handle small values without NaN
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))

    # Very large cloud water/ice
    cloud_water_path = jnp.ones(nlev) * 10.0  # Very thick clouds
    cloud_ice_path = jnp.ones(nlev) * 5.0

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Should handle large values
    assert not jnp.any(jnp.isnan(sw_optics.optical_depth))
    assert not jnp.any(jnp.isnan(lw_optics.optical_depth))
    
    # Should have high optical depths (only first 2 bands for SW, 3 for LW)
    assert jnp.all(sw_optics.optical_depth[:, :2] > 1.0)
    assert jnp.all(lw_optics.optical_depth[:, :3] > 0.1)


def test_cloud_optics_iwc_dependence():
    """Ice optics respond to in-cloud IWC via the Moss/Foot r_eff.

    Same ice path spread over a thinner layer means higher in-cloud IWC,
    hence larger crystals — the optical properties must differ.
    """
    nlev = 10
    cloud_water_path = jnp.zeros(nlev)
    cloud_ice_path = jnp.ones(nlev) * 0.05

    # Thick layers: low IWC, small crystals
    sw_thick, lw_thick = cloud_optics(
        cloud_water_path, cloud_ice_path, jnp.full(nlev, 2000.0),
        jnp.array(1.0),
    )
    # Thin layers: high IWC, large crystals
    sw_thin, lw_thin = cloud_optics(
        cloud_water_path, cloud_ice_path, jnp.full(nlev, 100.0),
        jnp.array(1.0),
    )

    assert sw_thick.optical_depth.shape == sw_thin.optical_depth.shape
    assert jnp.all(sw_thick.optical_depth >= 0)
    assert jnp.all(sw_thin.optical_depth >= 0)
    # The r_eff difference must actually reach the optics
    assert not jnp.allclose(
        sw_thick.optical_depth, sw_thin.optical_depth
    )


def test_cloud_optics_mixed_phase():
    """Test mixed-phase clouds (both water and ice)"""
    layer_thickness = jnp.full(8, 500.0)

    # Mixed phase: water and ice coexist
    cloud_water_path = jnp.array([0.0, 0.1, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0])
    cloud_ice_path = jnp.array([0.0, 0.0, 0.05, 0.1, 0.15, 0.1, 0.05, 0.0])

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # Total optical depth should be combination of water and ice
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(lw_optics.optical_depth >= 0)
    
    # Levels with both water and ice should have higher optical depth
    mixed_level = 3  # Both water and ice present
    
    # Mixed phase should have substantial optical depth (only first n_bands)
    assert jnp.all(sw_optics.optical_depth[mixed_level, :2] > 0)
    assert jnp.all(lw_optics.optical_depth[mixed_level, :3] > 0)


def test_cloud_optics_band_variations():
    """Test spectral variations across bands"""
    nlev = 5
    cloud_water_path = jnp.ones(nlev) * 0.2
    cloud_ice_path = jnp.zeros(nlev)
    layer_thickness = jnp.full(nlev, 500.0)

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
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
    layer_thickness = jnp.full(nlev, 500.0)

    sw_optics, lw_optics = cloud_optics(
        cloud_water_path, cloud_ice_path, layer_thickness, jnp.array(1.0)
    )
    
    # SW should have high single scattering albedo (clouds scatter well in visible) - only first 2 bands
    assert jnp.all(sw_optics.single_scatter_albedo[:, :2] > 0.8)
    
    # LW should have lower single scattering albedo (more absorption in IR)
    # Note: Different number of bands, so compare averages
    lw_ssa_avg = jnp.mean(lw_optics.single_scatter_albedo, axis=1)
    sw_ssa_avg = jnp.mean(sw_optics.single_scatter_albedo, axis=1)
    assert jnp.all(lw_ssa_avg <= sw_ssa_avg)
    
    # Asymmetry factor should be physical (only first bands have values)
    assert jnp.all(sw_optics.asymmetry_factor[:, :2] >= -1)
    assert jnp.all(sw_optics.asymmetry_factor[:, :2] <= 1)
    assert jnp.all(lw_optics.asymmetry_factor[:, :3] >= -1)
    assert jnp.all(lw_optics.asymmetry_factor[:, :3] <= 1)
    
    # Clouds typically have forward scattering (g > 0)
    cloudy_levels = cloud_water_path + cloud_ice_path > 0
    if jnp.any(cloudy_levels):
        assert jnp.any(sw_optics.asymmetry_factor > 0)