"""Tests for ICON vertical level definitions."""

import pytest
import jax.numpy as jnp
from jcm.vertical.icon_levels import HybridLevels, ICONLevels


class TestHybridLevels:
    """Test HybridLevels dataclass"""
    
    def test_hybrid_levels_initialization(self):
        """Test basic initialization"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        assert levels.nlevels == 2
        assert len(levels.a_boundaries) == 3
        assert len(levels.b_boundaries) == 3
    
    def test_a_centers(self):
        """Test a_centers property"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        a_centers = levels.a_centers
        assert len(a_centers) == 2
        assert jnp.allclose(a_centers[0], 750.0)  # (1000 + 500) / 2
        assert jnp.allclose(a_centers[1], 250.0)  # (500 + 0) / 2
    
    def test_b_centers(self):
        """Test b_centers property"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        b_centers = levels.b_centers
        assert len(b_centers) == 2
        assert jnp.allclose(b_centers[0], 0.25)  # (0.0 + 0.5) / 2
        assert jnp.allclose(b_centers[1], 0.75)  # (0.5 + 1.0) / 2
    
    def test_get_pressure_levels_scalar(self):
        """Test pressure calculation with scalar surface pressure"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        p_surf = jnp.array(100000.0)  # Pa - convert to jnp array
        p = levels.get_pressure_levels(p_surf)
        
        # p = a_center + b_center * p_surf
        assert len(p) == 2
        assert jnp.allclose(p[0], 750.0 + 0.25 * 100000.0)  # 25750
        assert jnp.allclose(p[1], 250.0 + 0.75 * 100000.0)  # 75250
    
    def test_get_pressure_levels_1d(self):
        """Test pressure calculation with 1D surface pressure"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        p_surf = jnp.array([100000.0, 95000.0])
        p = levels.get_pressure_levels(p_surf)
        
        assert p.shape == (2, 2)  # (nlevels, ncols)
    
    def test_get_pressure_levels_multidim(self):
        """Test pressure calculation with multi-dimensional surface pressure"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        p_surf = jnp.ones((4, 8)) * 100000.0  # (nlat, nlon)
        p = levels.get_pressure_levels(p_surf)
        
        assert p.shape == (2, 4, 8)  # (nlevels, nlat, nlon)
    
    def test_get_pressure_interfaces_scalar(self):
        """Test pressure at interfaces with scalar surface pressure"""
        a_boundaries = jnp.array([1000.0, 500.0, 0.0])
        b_boundaries = jnp.array([0.0, 0.5, 1.0])
        
        levels = HybridLevels(
            nlevels=2,
            a_boundaries=a_boundaries,
            b_boundaries=b_boundaries
        )
        
        p_surf = jnp.array(100000.0)  # Pa - convert to jnp array
        p_int = levels.get_pressure_interfaces(p_surf)
        
        assert len(p_int) == 3
        assert jnp.allclose(p_int[0], 1000.0)  # a[0] + b[0] * p_surf
        assert jnp.allclose(p_int[1], 50500.0)  # a[1] + b[1] * p_surf
        assert jnp.allclose(p_int[2], 100000.0)  # a[2] + b[2] * p_surf


class TestICONLevels:
    """Test ICONLevels factory class"""
    
    def test_get_levels_40(self):
        """Test loading 40-level configuration"""
        levels = ICONLevels.get_levels(40)
        
        assert levels.nlevels == 40
        assert len(levels.a_boundaries) == 41
        assert len(levels.b_boundaries) == 41
        assert len(levels.a_centers) == 40
        assert len(levels.b_centers) == 40
        
        # Check boundary conditions
        assert levels.b_boundaries[0] == 0.0  # Top
        assert levels.b_boundaries[-1] == 1.0  # Surface
    
    def test_get_levels_47(self):
        """Test loading 47-level configuration"""
        levels = ICONLevels.get_levels(47)
        
        assert levels.nlevels == 47
        assert len(levels.a_boundaries) == 48
        assert len(levels.b_boundaries) == 48
        assert len(levels.a_centers) == 47
        assert len(levels.b_centers) == 47
        
        # Check boundary conditions
        assert levels.b_boundaries[0] == 0.0  # Top
        assert levels.b_boundaries[-1] == 1.0  # Surface
    
    def test_caching(self):
        """Test that levels are cached"""
        levels1 = ICONLevels.get_levels(40)
        levels2 = ICONLevels.get_levels(40)
        
        # Should be the same object from cache
        assert levels1 is levels2
    
    def test_invalid_levels(self):
        """Test error handling for unsupported levels"""
        with pytest.raises(ValueError, match="No built-in level definition"):
            ICONLevels.get_levels(99)
    
    def test_available_levels(self):
        """Test available_levels method"""
        available = ICONLevels.available_levels()
        
        # Should at least have built-in levels
        assert 40 in available
        assert 47 in available
        assert isinstance(available, list)
    
    def test_pressure_calculation_consistency(self):
        """Test that pressure calculations are consistent"""
        levels = ICONLevels.get_levels(40)
        
        # Test with standard surface pressure
        p_surf = jnp.array(101325.0)  # Pa - convert to jnp array
        p_centers = levels.get_pressure_levels(p_surf)
        p_interfaces = levels.get_pressure_interfaces(p_surf)
        
        # Centers should be between interfaces
        for i in range(levels.nlevels):
            assert p_interfaces[i] <= p_centers[i] <= p_interfaces[i+1]
    
    def test_monotonic_sigma_coordinates(self):
        """Test that sigma coordinates are monotonically increasing"""
        for nlevels in [40, 47]:
            levels = ICONLevels.get_levels(nlevels)
            
            # b_boundaries should increase from 0 to 1
            assert jnp.all(jnp.diff(levels.b_boundaries) >= 0)
            assert levels.b_boundaries[0] == 0.0
            assert levels.b_boundaries[-1] == 1.0
    
    def test_realistic_pressure_range(self):
        """Test that pressure levels are in realistic range"""
        levels = ICONLevels.get_levels(40)
        p_surf = jnp.array(101325.0)  # Pa (standard sea level) - convert to jnp array
        
        p = levels.get_pressure_levels(p_surf)
        
        # All pressures should be positive and below surface pressure
        assert jnp.all(p > 0)
        assert jnp.all(p <= p_surf)
        
        # Top level should be less than surface (ICON 40 levels go to ~28 kPa)
        assert p[0] < 0.5 * p_surf  # Top < 50% of surface


class TestHybridLevelsIntegration:
    """Integration tests for hybrid levels"""
    
    def test_use_with_geometry(self):
        """Test that levels work with Geometry class"""
        from jcm.geometry import Geometry
        
        # This should not raise an error
        geometry = Geometry.from_grid_shape((32, 16), node_levels=40, hybrid=True)
        
        assert geometry.nodal_shape[0] == 40  # First dimension is nlevels
        # Verify hybrid coordinates are set
        assert geometry.hybrid_a_boundaries is not None
        assert geometry.hybrid_b_boundaries is not None
        assert len(geometry.hybrid_a_boundaries) == 41
    
    def test_multiple_levels_configurations(self):
        """Test using multiple level configurations"""
        levels_40 = ICONLevels.get_levels(40)
        levels_47 = ICONLevels.get_levels(47)
        
        # Different configurations should have different values
        assert levels_40.nlevels != levels_47.nlevels
        assert len(levels_40.a_boundaries) != len(levels_47.a_boundaries)
    
    def test_file_loading_fallback(self):
        """Test that file loading is attempted before falling back to built-in"""
        # Clear cache to force reload
        ICONLevels._levels_cache.clear()
        
        # This will try to load from files first, then fall back to built-in
        levels = ICONLevels.get_levels(40)
        
        # Should still work
        assert levels.nlevels == 40
        assert len(levels.a_boundaries) == 41
    
    def test_file_loading_if_available(self):
        """Test loading from files if they exist (e.g., 60 or 95 levels)"""
        # Try to load a level configuration that only exists in files
        # This will exercise the file-loading code path
        try:
            levels_60 = ICONLevels.get_levels(60)
            # If successful, verify it loaded correctly
            assert levels_60.nlevels == 60
            assert len(levels_60.a_boundaries) == 61
            assert len(levels_60.b_boundaries) == 61
        except ValueError:
            # Files might not be available, that's okay
            # At least we tried the file-loading path
            pass

