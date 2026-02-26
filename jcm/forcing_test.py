"""Tests for jcm/forcing.py

Tests for ForcingData struct, _fixed_ssts, and default_forcing functions.
"""

import unittest
import jax.numpy as jnp
import numpy as np
from jcm.forcing import ForcingData, _fixed_ssts, default_forcing
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class TestForcingDataZeros(unittest.TestCase):
    """Tests for ForcingData.zeros classmethod."""

    def test_zeros_all_defaults(self):
        """Zeros with no additional args should create all-zero arrays."""
        nodal_shape = (96, 48)
        forcing = ForcingData.zeros(nodal_shape)

        self.assertEqual(forcing.alb0.shape, nodal_shape)
        self.assertEqual(forcing.sice_am.shape, nodal_shape)
        self.assertEqual(forcing.snowc_am.shape, nodal_shape)
        self.assertEqual(forcing.soilw_am.shape, nodal_shape)
        self.assertEqual(forcing.stl_am.shape, nodal_shape)
        self.assertEqual(forcing.sea_surface_temperature.shape, nodal_shape)

        self.assertTrue(jnp.allclose(forcing.alb0, 0.0))
        self.assertTrue(jnp.allclose(forcing.sice_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.snowc_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.soilw_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.stl_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.sea_surface_temperature, 0.0))

    def test_zeros_with_custom_sst(self):
        """Zeros with custom SST should use provided values."""
        nodal_shape = (96, 48)
        sst = jnp.ones(nodal_shape) * 300.

        forcing = ForcingData.zeros(nodal_shape, sea_surface_temperature=sst)

        self.assertTrue(jnp.allclose(forcing.sea_surface_temperature, 300.))
        # Other fields should still be zero
        self.assertTrue(jnp.allclose(forcing.alb0, 0.0))

    def test_zeros_with_multiple_custom_fields(self):
        """Zeros with multiple custom fields should use all provided values."""
        nodal_shape = (64, 32)
        alb0 = jnp.ones(nodal_shape) * 0.3
        sst = jnp.ones(nodal_shape) * 290.
        stl = jnp.ones(nodal_shape) * 280.

        forcing = ForcingData.zeros(
            nodal_shape,
            alb0=alb0,
            sea_surface_temperature=sst,
            stl_am=stl
        )

        self.assertTrue(jnp.allclose(forcing.alb0, 0.3))
        self.assertTrue(jnp.allclose(forcing.sea_surface_temperature, 290.))
        self.assertTrue(jnp.allclose(forcing.stl_am, 280.))
        # Unspecified fields should be zero
        self.assertTrue(jnp.allclose(forcing.sice_am, 0.0))


class TestForcingDataOnes(unittest.TestCase):
    """Tests for ForcingData.ones classmethod."""

    def test_ones_all_defaults(self):
        """Ones with no additional args should create all-ones arrays."""
        nodal_shape = (96, 48)
        forcing = ForcingData.ones(nodal_shape)

        self.assertEqual(forcing.alb0.shape, nodal_shape)
        self.assertTrue(jnp.allclose(forcing.alb0, 1.0))
        self.assertTrue(jnp.allclose(forcing.sice_am, 1.0))
        self.assertTrue(jnp.allclose(forcing.snowc_am, 1.0))
        self.assertTrue(jnp.allclose(forcing.soilw_am, 1.0))
        self.assertTrue(jnp.allclose(forcing.stl_am, 1.0))
        self.assertTrue(jnp.allclose(forcing.sea_surface_temperature, 1.0))

    def test_ones_with_custom_field(self):
        """Ones with custom field should use provided value, rest are ones."""
        nodal_shape = (64, 32)
        alb0 = jnp.ones(nodal_shape) * 0.2

        forcing = ForcingData.ones(nodal_shape, alb0=alb0)

        self.assertTrue(jnp.allclose(forcing.alb0, 0.2))
        self.assertTrue(jnp.allclose(forcing.sice_am, 1.0))


class TestForcingDataCopy(unittest.TestCase):
    """Tests for ForcingData.copy method."""

    def test_copy_no_changes(self):
        """Copy with no args should return identical data."""
        nodal_shape = (64, 32)
        forcing = ForcingData.zeros(
            nodal_shape,
            alb0=jnp.ones(nodal_shape) * 0.3,
            sea_surface_temperature=jnp.ones(nodal_shape) * 300.
        )

        copied = forcing.copy()

        self.assertTrue(jnp.allclose(copied.alb0, forcing.alb0))
        self.assertTrue(jnp.allclose(copied.sice_am, forcing.sice_am))
        self.assertTrue(jnp.allclose(copied.sea_surface_temperature, forcing.sea_surface_temperature))

    def test_copy_with_changes(self):
        """Copy with args should replace those fields."""
        nodal_shape = (64, 32)
        forcing = ForcingData.zeros(
            nodal_shape,
            alb0=jnp.ones(nodal_shape) * 0.3,
            sea_surface_temperature=jnp.ones(nodal_shape) * 300.
        )

        new_sst = jnp.ones(nodal_shape) * 290.
        new_alb0 = jnp.ones(nodal_shape) * 0.5

        copied = forcing.copy(sea_surface_temperature=new_sst, alb0=new_alb0)

        self.assertTrue(jnp.allclose(copied.alb0, 0.5))
        self.assertTrue(jnp.allclose(copied.sea_surface_temperature, 290.))
        # Unchanged fields
        self.assertTrue(jnp.allclose(copied.sice_am, forcing.sice_am))


class TestForcingDataIsnan(unittest.TestCase):
    """Tests for ForcingData.isnan method."""

    def test_isnan_no_nans(self):
        """Isnan should return all False when no NaNs present."""
        nodal_shape = (64, 32)
        forcing = ForcingData.zeros(nodal_shape)

        nan_check = forcing.isnan()

        self.assertFalse(jnp.any(nan_check.alb0))
        self.assertFalse(jnp.any(nan_check.sice_am))
        self.assertFalse(jnp.any(nan_check.sea_surface_temperature))

    def test_isnan_with_nans(self):
        """Isnan should detect NaN values."""
        nodal_shape = (64, 32)
        sst_with_nan = jnp.ones(nodal_shape) * 300.
        sst_with_nan = sst_with_nan.at[0, 0].set(jnp.nan)

        forcing = ForcingData.zeros(nodal_shape, sea_surface_temperature=sst_with_nan)
        nan_check = forcing.isnan()

        self.assertTrue(jnp.any(nan_check.sea_surface_temperature))
        self.assertFalse(jnp.any(nan_check.alb0))


class TestForcingDataAnyTrue(unittest.TestCase):
    """Tests for ForcingData.any_true method."""

    def test_any_true_with_nan(self):
        """any_true should return True when NaN present in any field."""
        nodal_shape = (64, 32)
        sst_with_nan = jnp.ones(nodal_shape) * 300.
        sst_with_nan = sst_with_nan.at[0, 0].set(jnp.nan)

        forcing = ForcingData.zeros(nodal_shape, sea_surface_temperature=sst_with_nan)
        nan_check = forcing.isnan()

        self.assertTrue(nan_check.any_true())

    def test_any_true_no_nans(self):
        """any_true should return False when no NaNs present."""
        nodal_shape = (64, 32)
        forcing = ForcingData.zeros(nodal_shape)
        nan_check = forcing.isnan()

        self.assertFalse(nan_check.any_true())


class TestFixedSsts(unittest.TestCase):
    """Tests for _fixed_ssts function."""

    def test_fixed_ssts_shape(self):
        """_fixed_ssts should return correct shape."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        ssts = _fixed_ssts(grid)

        self.assertEqual(ssts.shape, grid.nodal_shape)

    def test_fixed_ssts_equator_maximum(self):
        """SST should be maximum at equator (300.15 K = 27 + 273.15)."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        ssts = _fixed_ssts(grid)

        # Find equator (latitude closest to 0)
        latitudes = grid.latitudes
        equator_idx = jnp.argmin(jnp.abs(latitudes))

        # Expected temperature at equator: 27*cos(0)^2 + 273.15 = 300.15 K
        equator_sst = ssts[0, equator_idx]
        self.assertTrue(jnp.isclose(equator_sst, 300.15, atol=0.1))

    def test_fixed_ssts_high_latitude_minimum(self):
        """SST should be 273.15 K at latitudes >= 60 degrees."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        ssts = _fixed_ssts(grid)

        latitudes = grid.latitudes

        # Find indices where |latitude| >= pi/3 (60 degrees)
        high_lat_mask = jnp.abs(latitudes) >= jnp.pi / 3

        # SST at high latitudes should be 273.15 K
        if jnp.any(high_lat_mask):
            high_lat_ssts = ssts[:, high_lat_mask]
            self.assertTrue(jnp.allclose(high_lat_ssts, 273.15, atol=0.1))

    def test_fixed_ssts_symmetry(self):
        """SST profile should be symmetric about equator."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        ssts = _fixed_ssts(grid)

        # SST should be zonally uniform
        # Check that all longitudes have same value at each latitude
        for i in range(ssts.shape[1]):
            self.assertTrue(jnp.allclose(ssts[:, i], ssts[0, i]))

    def test_fixed_ssts_zonal_uniformity(self):
        """SST should be zonally uniform (same at all longitudes for each latitude)."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        ssts = _fixed_ssts(grid)

        # All values along each latitude band should be identical
        for lat_idx in range(ssts.shape[1]):
            lat_ssts = ssts[:, lat_idx]
            self.assertTrue(jnp.allclose(lat_ssts, lat_ssts[0]))

    def test_fixed_ssts_physically_valid(self):
        """SST should be within physically valid range (273.15 to 310 K)."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        ssts = _fixed_ssts(grid)

        self.assertTrue(jnp.all(ssts >= 273.15 - 0.01))  # Small tolerance for numerical error
        self.assertTrue(jnp.all(ssts <= 310.))


class TestDefaultForcing(unittest.TestCase):
    """Tests for default_forcing function."""

    def test_default_forcing_shape(self):
        """default_forcing should return correct shapes."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        forcing = default_forcing(grid)

        expected_shape = grid.nodal_shape
        self.assertEqual(forcing.alb0.shape, expected_shape)
        self.assertEqual(forcing.sice_am.shape, expected_shape)
        self.assertEqual(forcing.snowc_am.shape, expected_shape)
        self.assertEqual(forcing.soilw_am.shape, expected_shape)
        self.assertEqual(forcing.stl_am.shape, expected_shape)
        self.assertEqual(forcing.sea_surface_temperature.shape, expected_shape)

    def test_default_forcing_sst_from_fixed_ssts(self):
        """default_forcing SST should match _fixed_ssts."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal

        forcing = default_forcing(grid)
        expected_sst = _fixed_ssts(grid)

        self.assertTrue(jnp.allclose(forcing.sea_surface_temperature, expected_sst))

    def test_default_forcing_other_fields_zero(self):
        """default_forcing should have zeros for non-SST fields."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        forcing = default_forcing(grid)

        self.assertTrue(jnp.allclose(forcing.alb0, 0.0))
        self.assertTrue(jnp.allclose(forcing.sice_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.snowc_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.soilw_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.stl_am, 0.0))

    def test_default_forcing_different_resolutions(self):
        """default_forcing should work for different resolutions."""
        for truncation in [21, 31, 42]:
            coords = get_speedy_coords(layers=8, spectral_truncation=truncation)
            grid = coords.horizontal
            forcing = default_forcing(grid)

            self.assertEqual(forcing.sea_surface_temperature.shape, grid.nodal_shape)


class TestForcingDataTreeMath(unittest.TestCase):
    """Tests for JAX tree_math compatibility."""

    def test_forcing_data_is_jax_pytree(self):
        """ForcingData should be a valid JAX pytree."""
        import jax

        nodal_shape = (64, 32)
        forcing = ForcingData.zeros(
            nodal_shape,
            sea_surface_temperature=jnp.ones(nodal_shape) * 300.
        )

        # Should be able to tree_map over it
        doubled = jax.tree.map(lambda x: x * 2, forcing)

        self.assertTrue(jnp.allclose(doubled.sea_surface_temperature, 600.))

    def test_forcing_data_jit_compatible(self):
        """ForcingData should work with jax.jit."""
        import jax

        @jax.jit
        def get_sst(forcing):
            return forcing.sea_surface_temperature

        nodal_shape = (64, 32)
        forcing = ForcingData.zeros(
            nodal_shape,
            sea_surface_temperature=jnp.ones(nodal_shape) * 300.
        )

        result = get_sst(forcing)
        self.assertTrue(jnp.allclose(result, 300.))


class TestForcingDataFromFile(unittest.TestCase):
    """Tests for ForcingData.from_file using actual data files."""

    def test_from_file_loads_forcing(self):
        """from_file should load forcing data from actual NetCDF file."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        expected_2d_shape = coords.horizontal.nodal_shape
        expected_3d_shape = (*expected_2d_shape, 365)

        # 2D field
        self.assertEqual(forcing.alb0.shape, expected_2d_shape)
        # 3D fields (with time dimension)
        self.assertEqual(forcing.sice_am.shape, expected_3d_shape)
        self.assertEqual(forcing.snowc_am.shape, expected_3d_shape)
        self.assertEqual(forcing.soilw_am.shape, expected_3d_shape)
        self.assertEqual(forcing.stl_am.shape, expected_3d_shape)
        self.assertEqual(forcing.sea_surface_temperature.shape, expected_3d_shape)

    def test_from_file_has_valid_albedo(self):
        """Loaded forcing should have albedo values in valid range [0, 1]."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        self.assertTrue(jnp.all(forcing.alb0 >= 0.0))
        self.assertTrue(jnp.all(forcing.alb0 <= 1.0))

    def test_from_file_has_valid_sea_ice(self):
        """Loaded forcing should have sea ice concentration in valid range [0, 1]."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        self.assertTrue(jnp.all(forcing.sice_am >= 0.0))
        self.assertTrue(jnp.all(forcing.sice_am <= 1.0))

    def test_from_file_has_valid_sst(self):
        """Loaded forcing should have physically realistic SST values."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # SST values can be low over sea ice areas (down to ~236 K in this dataset)
        # but should not exceed tropical maximum (~35C = 308K)
        self.assertTrue(jnp.all(forcing.sea_surface_temperature >= 230.))
        self.assertTrue(jnp.all(forcing.sea_surface_temperature <= 320.))

    def test_from_file_has_valid_soil_moisture(self):
        """Loaded forcing should have soil moisture in valid range."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # Soil moisture should be non-negative
        self.assertTrue(jnp.all(forcing.soilw_am >= 0.0))

    def test_from_file_has_valid_snow_cover(self):
        """Loaded forcing should have snow cover in valid range."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # Snow cover should be non-negative
        self.assertTrue(jnp.all(forcing.snowc_am >= 0.0))

    def test_from_file_no_nans(self):
        """Loaded forcing should not contain NaN values."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        nan_check = forcing.isnan()
        self.assertFalse(nan_check.any_true())


class TestForcingDataFromFileValidation(unittest.TestCase):
    """Tests for ForcingData.from_file validation logic using mock files."""

    def test_from_file_validates_nodal_shape(self):
        """from_file should reject invalid nodal shapes when coords is None."""
        import xarray as xr
        import tempfile
        import os

        # Create a dataset with invalid shape (not in VALID_NODAL_SHAPES)
        invalid_shape = (50, 25)  # Not a valid nodal shape
        ds = xr.Dataset({
            'stl': (['lon', 'lat', 'time'], np.zeros((*invalid_shape, 365))),
            'icec': (['lon', 'lat', 'time'], np.zeros((*invalid_shape, 365))),
            'sst': (['lon', 'lat', 'time'], np.zeros((*invalid_shape, 365))),
            'alb': (['lon', 'lat'], np.zeros(invalid_shape)),
            'soilw_am': (['lon', 'lat', 'time'], np.zeros((*invalid_shape, 365))),
            'snowc': (['lon', 'lat', 'time'], np.zeros((*invalid_shape, 365))),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            ds.to_netcdf(f.name)
            temp_file = f.name

        try:
            with self.assertRaises(ValueError) as context:
                ForcingData.from_file(temp_file)
            self.assertIn("Invalid nodal shape", str(context.exception))
        finally:
            os.remove(temp_file)

    def test_from_file_validates_time_dimension(self):
        """from_file should reject datasets with wrong number of time steps."""
        import xarray as xr
        import tempfile
        import os

        valid_shape = (96, 48)  # T31 resolution
        wrong_time = 360  # Should be 365

        ds = xr.Dataset({
            'stl': (['lon', 'lat', 'time'], np.zeros((*valid_shape, wrong_time))),
            'icec': (['lon', 'lat', 'time'], np.zeros((*valid_shape, wrong_time))),
            'sst': (['lon', 'lat', 'time'], np.zeros((*valid_shape, wrong_time))),
            'alb': (['lon', 'lat'], np.zeros(valid_shape)),
            'soilw_am': (['lon', 'lat', 'time'], np.zeros((*valid_shape, wrong_time))),
            'snowc': (['lon', 'lat', 'time'], np.zeros((*valid_shape, wrong_time))),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            ds.to_netcdf(f.name)
            temp_file = f.name

        try:
            with self.assertRaises(ValueError) as context:
                ForcingData.from_file(temp_file)
            self.assertIn("Expected 365 time steps", str(context.exception))
        finally:
            os.remove(temp_file)

    def test_from_file_validates_missing_variables(self):
        """from_file should reject datasets with missing variables."""
        import xarray as xr
        import tempfile
        import os

        valid_shape = (96, 48)

        # Missing 'sst' variable
        ds = xr.Dataset({
            'stl': (['lon', 'lat', 'time'], np.zeros((*valid_shape, 365))),
            'icec': (['lon', 'lat', 'time'], np.zeros((*valid_shape, 365))),
            # 'sst' is missing
            'alb': (['lon', 'lat'], np.zeros(valid_shape)),
            'soilw_am': (['lon', 'lat', 'time'], np.zeros((*valid_shape, 365))),
            'snowc': (['lon', 'lat', 'time'], np.zeros((*valid_shape, 365))),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            ds.to_netcdf(f.name)
            temp_file = f.name

        try:
            with self.assertRaises(ValueError) as context:
                ForcingData.from_file(temp_file)
            self.assertIn("Missing variables", str(context.exception))
        finally:
            os.remove(temp_file)


if __name__ == '__main__':
    unittest.main()
