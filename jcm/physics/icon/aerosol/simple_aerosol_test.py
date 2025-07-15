"""
Unit tests for simple_aerosol module

Tests the MACv2-SP (Simple Plumes) aerosol scheme implementation
with proper JAX compatibility and vectorization.
"""

import jax.numpy as jnp
import jax
import pytest
from jax import random
from jcm.physics.icon.aerosol.simple_aerosol import (
    get_simple_aerosol,
    get_plume_spatial_distribution,
    get_anthropogenic_aod,
    get_background_aod,
    get_vertical_profiles,
    get_background_vertical_profile,
    get_optical_properties,
    get_CDNC,
)
from jcm.physics.icon.aerosol.aerosol_params import AerosolParameters
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics.icon.icon_physics import PhysicsTendency
from jcm.physics_interface import PhysicsState
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
import tree_math


class TestAerosolParameters:
    """Test AerosolParameters class"""
    
    def test_default_parameters(self):
        """Test default parameter creation"""
        params = AerosolParameters.default()
        
        assert params.nplumes == 9
        assert params.nfeatures == 2
        assert params.plume_lat.shape == (9,)
        assert params.plume_lon.shape == (9,)
        assert params.beta_a.shape == (9,)
        assert params.beta_b.shape == (9,)
        assert params.aod_spmx.shape == (9,)
        assert params.aod_fmbg.shape == (9,)
        assert params.asy550.shape == (9,)
        assert params.ssa550.shape == (9,)
        assert params.angstrom.shape == (9,)
        assert params.sig_lon_E.shape == (2, 9)
        assert params.sig_lon_W.shape == (2, 9)
        assert params.sig_lat_E.shape == (2, 9)
        assert params.sig_lat_W.shape == (2, 9)
        assert params.theta.shape == (2, 9)
        assert params.ftr_weight.shape == (2, 9)
        assert params.ann_cycle.shape == (9,)
        assert params.year_weight.shape == (9,)
        assert jnp.isscalar(params.background_aod)
    
    def test_parameter_ranges(self):
        """Test parameter values are in reasonable ranges"""
        params = AerosolParameters.default()
        
        # Latitude should be in [-90, 90]
        assert jnp.all(params.plume_lat >= -90)
        assert jnp.all(params.plume_lat <= 90)
        
        # Longitude should be in [-180, 180]
        assert jnp.all(params.plume_lon >= -180)
        assert jnp.all(params.plume_lon <= 180)
        
        # Beta parameters should be positive
        assert jnp.all(params.beta_a > 0)
        assert jnp.all(params.beta_b > 0)
        
        # AOD values should be positive
        assert jnp.all(params.aod_spmx > 0)
        assert jnp.all(params.aod_fmbg > 0)
        assert params.background_aod > 0
        
        # SSA should be in [0, 1]
        assert jnp.all(params.ssa550 >= 0)
        assert jnp.all(params.ssa550 <= 1)
        
        # Asymmetry parameter should be in [-1, 1]
        assert jnp.all(params.asy550 >= -1)
        assert jnp.all(params.asy550 <= 1)
        
        # Spatial extents should be positive
        assert jnp.all(params.sig_lon_E > 0)
        assert jnp.all(params.sig_lon_W > 0)
        assert jnp.all(params.sig_lat_E > 0)
        assert jnp.all(params.sig_lat_W > 0)
        
        # Feature weights should be positive and sum to 1 for each plume
        assert jnp.all(params.ftr_weight > 0)
        weight_sums = jnp.sum(params.ftr_weight, axis=0)
        assert jnp.allclose(weight_sums, 1.0, rtol=1e-10)


class TestVerticalProfiles:
    """Test vertical profile calculations"""
    
    def test_vertical_profiles_shape(self):
        """Test vertical profile output shapes"""
        params = AerosolParameters.default()
        nlev, ncols = 20, 100
        
        # Create test height array
        height_full = jnp.linspace(0, 15000, nlev)[:, jnp.newaxis]
        height_full = jnp.repeat(height_full, ncols, axis=1)
        
        profiles = get_vertical_profiles(height_full, params)
        
        assert profiles.shape == (params.nplumes, nlev, ncols)
        
        # Check normalization - each profile should integrate to 1
        profile_integrals = jnp.sum(profiles, axis=1)
        assert jnp.allclose(profile_integrals, 1.0, rtol=1e-6)
    
    def test_background_vertical_profile(self):
        """Test background vertical profile"""
        nlev, ncols = 20, 100
        
        # Create test height array
        height_full = jnp.linspace(0, 15000, nlev)[:, jnp.newaxis]
        height_full = jnp.repeat(height_full, ncols, axis=1)
        
        profile = get_background_vertical_profile(height_full)
        
        assert profile.shape == (nlev,)
        assert jnp.allclose(jnp.sum(profile), 1.0, rtol=1e-6)
        
        # Should be monotonically decreasing with height
        assert jnp.all(jnp.diff(profile) <= 0)
    
    def test_beta_function_properties(self):
        """Test beta function vertical profiles have correct properties"""
        params = AerosolParameters.default()
        nlev = 50
        ncols = 10
        
        # Create test height array
        height_full = jnp.linspace(0, 15000, nlev)[:, jnp.newaxis]
        height_full = jnp.repeat(height_full, ncols, axis=1)
        
        profiles = get_vertical_profiles(height_full, params)
        
        # Check that profiles are non-negative
        assert jnp.all(profiles >= 0)
        
        # Check that profiles have maximum somewhere in the middle
        # (not at boundaries for beta > 1)
        for i in range(params.nplumes):
            if params.beta_a[i] > 1 and params.beta_b[i] > 1:
                profile = profiles[i, :, 0]
                max_idx = jnp.argmax(profile)
                assert max_idx > 0 and max_idx < nlev - 1


class TestSpatialDistribution:
    """Test spatial distribution calculations"""
    
    def test_spatial_distribution_shape(self):
        """Test spatial distribution output shape"""
        params = AerosolParameters.default()
        ncols = 1000
        
        # Create test coordinates
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        spatial_dist = get_plume_spatial_distribution(lats, lons, params)
        
        assert spatial_dist.shape == (params.nplumes, ncols)
        assert jnp.all(spatial_dist >= 0)
        assert jnp.all(spatial_dist <= 1)
    
    def test_plume_centers_maximum(self):
        """Test that plumes have maximum at their centers"""
        params = AerosolParameters.default()
        
        # Test each plume center
        for i in range(params.nplumes):
            center_lat = params.plume_lat[i]
            center_lon = params.plume_lon[i]
            
            # Create small grid around center
            dlat = jnp.linspace(-5, 5, 21)
            dlon = jnp.linspace(-5, 5, 21)
            
            lats = center_lat + dlat
            lons = center_lon + dlon
            
            spatial_dist = get_plume_spatial_distribution(lats, lons, params)
            
            # Maximum should be at or near center
            max_idx = jnp.argmax(spatial_dist[i, :])
            assert max_idx >= 8 and max_idx <= 12  # Around center index (10)
    
    def test_longitude_wrapping(self):
        """Test longitude wrapping is handled correctly"""
        params = AerosolParameters.default()
        ncols = 100
        
        # Test with wrapped longitudes
        lats = jnp.zeros(ncols)
        lons = jnp.linspace(170, 190, ncols)  # Crosses 180° meridian
        
        spatial_dist = get_plume_spatial_distribution(lats, lons, params)
        
        # Should not have NaN or infinite values
        assert jnp.all(jnp.isfinite(spatial_dist))
        assert jnp.all(spatial_dist >= 0)


class TestAODCalculations:
    """Test AOD calculation functions"""
    
    def test_anthropogenic_aod_shape(self):
        """Test anthropogenic AOD calculation shape"""
        params = AerosolParameters.default()
        ncols = 500
        
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        aod_anth = get_anthropogenic_aod(lats, lons, params)
        
        assert aod_anth.shape == (ncols,)
        assert jnp.all(aod_anth >= 0)
    
    def test_background_aod_shape(self):
        """Test background AOD calculation shape"""
        params = AerosolParameters.default()
        ncols = 500
        
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        aod_bg = get_background_aod(lats, lons, params)
        
        assert aod_bg.shape == (ncols,)
        assert jnp.all(aod_bg >= 0)
    
    def test_aod_plume_regions(self):
        """Test that AOD is higher in plume regions"""
        params = AerosolParameters.default()
        
        # Test coordinates at plume centers vs remote regions
        plume_lats = params.plume_lat[:3]  # First 3 plumes
        plume_lons = params.plume_lon[:3]
        
        # Remote regions (ocean)
        remote_lats = jnp.array([0.0, 0.0, 0.0])
        remote_lons = jnp.array([0.0, 90.0, 180.0])
        
        aod_plume = get_anthropogenic_aod(plume_lats, plume_lons, params)
        aod_remote = get_anthropogenic_aod(remote_lats, remote_lons, params)
        
        # AOD should be higher at plume centers
        assert jnp.all(aod_plume > aod_remote)


class TestOpticalProperties:
    """Test optical property calculations"""
    
    def test_optical_properties_shape(self):
        """Test optical properties output shapes"""
        params = AerosolParameters.default()
        nlev, ncols = 30, 200
        
        # Create test data
        aod_profile = jnp.ones((nlev, ncols)) * 0.1
        spatial_dist = jnp.ones((params.nplumes, ncols)) / params.nplumes
        
        ssa_profile, asy_profile = get_optical_properties(aod_profile, spatial_dist, params)
        
        assert ssa_profile.shape == (nlev, ncols)
        assert asy_profile.shape == (nlev, ncols)
        
        # Check value ranges
        assert jnp.all(ssa_profile >= 0)
        assert jnp.all(ssa_profile <= 1)
        assert jnp.all(asy_profile >= -1)
        assert jnp.all(asy_profile <= 1)
    
    def test_optical_properties_weighted_average(self):
        """Test that optical properties are proper weighted averages"""
        params = AerosolParameters.default()
        nlev, ncols = 10, 50
        
        # Create test data with single plume dominating
        aod_profile = jnp.ones((nlev, ncols)) * 0.1
        spatial_dist = jnp.zeros((params.nplumes, ncols))
        spatial_dist = spatial_dist.at[0, :].set(1.0)  # Only first plume
        
        ssa_profile, asy_profile = get_optical_properties(aod_profile, spatial_dist, params)
        
        # Should match first plume properties
        expected_ssa = params.ssa550[0]
        expected_asy = params.asy550[0]
        
        assert jnp.allclose(ssa_profile, expected_ssa, rtol=1e-6)
        assert jnp.allclose(asy_profile, expected_asy, rtol=1e-6)


class TestCDNC:
    """Test CDNC calculation"""
    
    def test_cdnc_function(self):
        """Test CDNC calculation function"""
        aod_values = jnp.array([0.0, 0.1, 0.2, 0.5, 1.0])
        
        cdnc = get_CDNC(aod_values)
        
        assert cdnc.shape == aod_values.shape
        assert jnp.all(cdnc >= 0)
        
        # Should be monotonically increasing
        assert jnp.all(jnp.diff(cdnc) >= 0)
        
        # Should be zero when AOD is zero
        assert cdnc[0] == 0.0
    
    def test_cdnc_parameters(self):
        """Test CDNC with different parameters"""
        aod = jnp.array([0.1, 0.2, 0.3])
        
        # Test different parameter sets
        cdnc1 = get_CDNC(aod, A=60, B=20)
        cdnc2 = get_CDNC(aod, A=410, B=5)
        cdnc3 = get_CDNC(aod, A=16, B=1000)
        
        assert cdnc1.shape == aod.shape
        assert cdnc2.shape == aod.shape
        assert cdnc3.shape == aod.shape
        
        # All should be positive and finite
        assert jnp.all(cdnc1 > 0)
        assert jnp.all(cdnc2 > 0)
        assert jnp.all(cdnc3 > 0)
        assert jnp.all(jnp.isfinite(cdnc1))
        assert jnp.all(jnp.isfinite(cdnc2))
        assert jnp.all(jnp.isfinite(cdnc3))


class TestJAXCompatibility:
    """Test JAX compatibility and transformations"""
    
    def test_jit_compilation(self):
        """Test that functions can be JIT compiled"""
        params = AerosolParameters.default()
        
        # JIT compile the spatial distribution function
        jit_spatial = jax.jit(get_plume_spatial_distribution)
        
        ncols = 100
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        result = jit_spatial(lats, lons, params)
        
        assert result.shape == (params.nplumes, ncols)
        assert jnp.all(jnp.isfinite(result))
    
    def test_vmap_compatibility(self):
        """Test vectorization with vmap"""
        params = AerosolParameters.default()
        
        # Create batch of coordinates
        batch_size = 10
        ncols = 50
        
        lats_batch = jnp.stack([jnp.linspace(-90, 90, ncols) for _ in range(batch_size)])
        lons_batch = jnp.stack([jnp.linspace(-180, 180, ncols) for _ in range(batch_size)])
        
        # Vectorize over batch dimension
        vmap_spatial = jax.vmap(get_plume_spatial_distribution, in_axes=(0, 0, None))
        
        result = vmap_spatial(lats_batch, lons_batch, params)
        
        assert result.shape == (batch_size, params.nplumes, ncols)
        assert jnp.all(jnp.isfinite(result))
    
    def test_gradient_computation(self):
        """Test gradient computation through functions"""
        params = AerosolParameters.default()
        
        def aod_sum(lats, lons):
            return jnp.sum(get_anthropogenic_aod(lats, lons, params))
        
        ncols = 20
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        # Compute gradients
        grad_fn = jax.grad(aod_sum, argnums=(0, 1))
        grads = grad_fn(lats, lons)
        
        assert len(grads) == 2
        assert grads[0].shape == (ncols,)
        assert grads[1].shape == (ncols,)
        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))


class TestIntegration:
    """Integration tests for the full aerosol scheme"""
    
    def test_simple_integration(self):
        """Test basic integration of aerosol functions"""
        params = AerosolParameters.default()
        nlev, ncols = 8, 100
        
        # Create test coordinates
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        # Create test height array
        height_full = jnp.linspace(0, 15000, nlev)[:, jnp.newaxis]
        height_full = jnp.repeat(height_full, ncols, axis=1)
        
        # Test individual functions work together
        aod_anth = get_anthropogenic_aod(lats, lons, params)
        aod_bg = get_background_aod(lats, lons, params)
        
        assert aod_anth.shape == (ncols,)
        assert aod_bg.shape == (ncols,)
        assert jnp.all(aod_anth >= 0)
        assert jnp.all(aod_bg >= 0)
        
        # Test vertical profiles
        profiles = get_vertical_profiles(height_full, params)
        assert profiles.shape == (params.nplumes, nlev, ncols)
        
        # Test spatial distribution
        spatial_dist = get_plume_spatial_distribution(lats, lons, params)
        assert spatial_dist.shape == (params.nplumes, ncols)
        
        # Test optical properties
        aod_profile = jnp.ones((nlev, ncols)) * 0.1
        ssa_profile, asy_profile = get_optical_properties(aod_profile, spatial_dist, params)
        
        assert ssa_profile.shape == (nlev, ncols)
        assert asy_profile.shape == (nlev, ncols)
        assert jnp.all(ssa_profile >= 0)
        assert jnp.all(ssa_profile <= 1)
        assert jnp.all(asy_profile >= -1)
        assert jnp.all(asy_profile <= 1)
    
    def test_jit_compilation(self):
        """Test that individual functions can be JIT compiled"""
        params = AerosolParameters.default()
        ncols = 50
        
        # Test JIT compilation of each function
        jit_spatial = jax.jit(get_plume_spatial_distribution)
        jit_aod_anth = jax.jit(get_anthropogenic_aod)
        jit_aod_bg = jax.jit(get_background_aod)
        
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        # All should compile and run without errors
        spatial_dist = jit_spatial(lats, lons, params)
        aod_anth = jit_aod_anth(lats, lons, params)
        aod_bg = jit_aod_bg(lats, lons, params)
        
        assert spatial_dist.shape == (params.nplumes, ncols)
        assert aod_anth.shape == (ncols,)
        assert aod_bg.shape == (ncols,)
        assert jnp.all(jnp.isfinite(spatial_dist))
        assert jnp.all(jnp.isfinite(aod_anth))
        assert jnp.all(jnp.isfinite(aod_bg))
    
    def test_gradient_computation(self):
        """Test gradient computation for differentiability"""
        params = AerosolParameters.default()
        
        def total_aod(lats, lons):
            return jnp.sum(get_anthropogenic_aod(lats, lons, params))
        
        ncols = 20
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        
        # Should be able to compute gradients
        grad_fn = jax.grad(total_aod, argnums=(0, 1))
        grads = grad_fn(lats, lons)
        
        assert len(grads) == 2
        assert grads[0].shape == (ncols,)
        assert grads[1].shape == (ncols,)
        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))
    
    def test_conservation_properties(self):
        """Test conservation properties of individual functions"""
        params = AerosolParameters.default()
        nlev, ncols = 8, 100
        
        # Create test height array
        height_full = jnp.linspace(0, 15000, nlev)[:, jnp.newaxis]
        height_full = jnp.repeat(height_full, ncols, axis=1)
        
        # Test that vertical profiles integrate to 1
        profiles = get_vertical_profiles(height_full, params)
        profile_integrals = jnp.sum(profiles, axis=1)
        
        assert jnp.allclose(profile_integrals, 1.0, rtol=1e-6)
        
        # Test background profile integrates to 1
        bg_profile = get_background_vertical_profile(height_full)
        bg_integral = jnp.sum(bg_profile)
        
        assert jnp.allclose(bg_integral, 1.0, rtol=1e-6)
        
        # Test that optical properties are reasonable weighted averages
        lats = jnp.linspace(-90, 90, ncols)
        lons = jnp.linspace(-180, 180, ncols)
        spatial_dist = get_plume_spatial_distribution(lats, lons, params)
        
        aod_profile = jnp.ones((nlev, ncols)) * 0.1
        ssa_profile, asy_profile = get_optical_properties(aod_profile, spatial_dist, params)
        
        # Should be within range of parameter values
        assert jnp.all(ssa_profile >= jnp.min(params.ssa550))
        assert jnp.all(ssa_profile <= jnp.max(params.ssa550))
        assert jnp.all(asy_profile >= jnp.min(params.asy550))
        assert jnp.all(asy_profile <= jnp.max(params.asy550))


if __name__ == "__main__":
    pytest.main([__file__])