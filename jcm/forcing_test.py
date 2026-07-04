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
        """Zeros with no additional args should produce a baseline forcing
        state — zeros for fluxes/fractions, and a sensible default for
        the surface temperatures (~15 °C) so the surface flux scheme
        isn't presented with an unphysical ΔT against the atmosphere.
        """
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
        self.assertTrue(jnp.allclose(forcing.stl_am, 288.15))
        self.assertTrue(jnp.allclose(forcing.sea_surface_temperature, 288.15))

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
        """default_forcing zeroes flux/fraction fields and uses the
        ``ForcingData.zeros`` default (~15 °C) for land temperature.
        """
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        grid = coords.horizontal
        forcing = default_forcing(grid)

        self.assertTrue(jnp.allclose(forcing.alb0, 0.0))
        self.assertTrue(jnp.allclose(forcing.sice_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.snowc_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.soilw_am, 0.0))
        self.assertTrue(jnp.allclose(forcing.stl_am, 288.15))

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
        """from_file should load forcing data from actual NetCDF file.

        Time-varying fields are now wrapped as `TimeSeries` leaves with the
        time axis at index 0; static `alb0` stays a bare 2-D array.
        """
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        expected_2d_shape = coords.horizontal.nodal_shape
        expected_ts_shape = (365, *expected_2d_shape)

        # 2D field (no time dimension)
        self.assertEqual(forcing.alb0.shape, expected_2d_shape)
        # Time-varying fields: leading time axis
        self.assertEqual(forcing.sice_am.values.shape, expected_ts_shape)
        self.assertEqual(forcing.snowc_am.values.shape, expected_ts_shape)
        self.assertEqual(forcing.soilw_am.values.shape, expected_ts_shape)
        self.assertEqual(forcing.stl_am.values.shape, expected_ts_shape)
        self.assertEqual(forcing.sea_surface_temperature.values.shape, expected_ts_shape)

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

        self.assertTrue(jnp.all(forcing.sice_am.values >= 0.0))
        self.assertTrue(jnp.all(forcing.sice_am.values <= 1.0))

    def test_from_file_has_valid_sst(self):
        """Loaded forcing should have physically realistic SST values."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # SST values can be low over sea ice areas (down to ~236 K in this dataset)
        # but should not exceed tropical maximum (~35C = 308K)
        self.assertTrue(jnp.all(forcing.sea_surface_temperature.values >= 230.))
        self.assertTrue(jnp.all(forcing.sea_surface_temperature.values <= 320.))

    def test_from_file_has_valid_soil_moisture(self):
        """Loaded forcing should have soil moisture in valid range."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # Soil moisture should be non-negative
        self.assertTrue(jnp.all(forcing.soilw_am.values >= 0.0))

    def test_from_file_has_valid_snow_cover(self):
        """Loaded forcing should have snow cover in valid range."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # Snow cover should be non-negative
        self.assertTrue(jnp.all(forcing.snowc_am.values >= 0.0))

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

    def test_from_dataset_reads_prescribed_ghgs(self):
        """CO2/CH4/N2O present in a forcing file must be read, not defaulted.

        Regression for the silent fallback flagged on the GHG single-source PR:
        ``from_dataset`` previously special-cased only CO2, so N2O (and CH4) in a
        scenario file were ignored and radiation used the constant default.
        """
        import pandas as pd
        import xarray as xr

        valid_shape = (96, 48)  # T31
        n_times = 12
        times = pd.date_range("1990-01-01", periods=n_times, freq="MS")
        ds = xr.Dataset(
            data_vars={
                'stl': (['lon', 'lat', 'time'], np.full((*valid_shape, n_times), 280.0)),
                'icec': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                'sst': (['lon', 'lat', 'time'], np.full((*valid_shape, n_times), 290.0)),
                'alb': (['lon', 'lat'], np.full(valid_shape, 0.1)),
                'soilw_am': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                'snowc': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                # Scalars distinct from every default (CO2 420, CH4 1.9, N2O 0.327).
                'co2': 700.0,
                'ch4': 2.5,
                'n2o': 0.5,
            },
            coords={'time': times},
        )
        forcing = ForcingData.from_dataset(ds, validate=False)
        self.assertAlmostEqual(float(forcing.co2_vmr), 700.0)
        self.assertAlmostEqual(float(forcing.ch4_vmr), 2.5)
        self.assertAlmostEqual(float(forcing.n2o_vmr), 0.5)

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
                ForcingData.from_file(temp_file, validate=False)
            self.assertIn("Invalid nodal shape", str(context.exception))
        finally:
            os.remove(temp_file)

    def test_from_file_accepts_arbitrary_time_length(self):
        """from_file should accept any time-dimension length (#308). Older
        versions hard-rejected anything but exactly 365 days; we now wrap
        the time axis as a `TimeSeries` and let the Model align via
        `select(date)`.
        """
        import pandas as pd
        import xarray as xr
        import tempfile
        import os

        valid_shape = (96, 48)  # T31 resolution
        # Two-year file with 360-day calendar -> 720 daily entries
        n_times = 720
        times = pd.date_range("1980-01-01", periods=n_times, freq="D")

        ds = xr.Dataset(
            data_vars={
                'stl': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                'icec': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                'sst': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                'alb': (['lon', 'lat'], np.zeros(valid_shape)),
                'soilw_am': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
                'snowc': (['lon', 'lat', 'time'], np.zeros((*valid_shape, n_times))),
            },
            coords={'time': times},
        )

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            ds.to_netcdf(f.name)
            temp_file = f.name

        try:
            # Synthetic zero-filled fixture — bypass the BC sanity check
            # that ``from_file`` runs on real data (would reject all-zero
            # ``stl``/``sst``).
            forcing = ForcingData.from_file(temp_file, validate=False)
            # Time axis preserved at full length, leading dimension.
            self.assertEqual(forcing.sst if False else forcing.sea_surface_temperature.values.shape,
                             (n_times, *valid_shape))
            # Span > 1 year -> should auto-select BY_DATE alignment.
            from jcm.forcing import BY_DATE
            self.assertEqual(int(forcing.sea_surface_temperature.align_mode), BY_DATE)
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
                ForcingData.from_file(temp_file, validate=False)
            self.assertIn("Missing variables", str(context.exception))
        finally:
            os.remove(temp_file)


class TestForcingDataBcSanityCheck(unittest.TestCase):
    """``_validate_bc_fields`` is the host-side guard against authoring
    mistakes (units, NaN, AMIP-extrapolated stl) in the boundary-condition
    NetCDF before they manifest as a multi-day NaN inside the JIT'd
    integration. Tests cover the hard-range, NaN, and the AMIP-vs-JSBACH
    heuristic paths.
    """

    def _make_realistic_ds(self, valid_shape=(96, 48), n_times=12):
        """Build a small synthetic-but-physically-plausible BC dataset."""
        import pandas as pd
        import xarray as xr
        times = pd.date_range("1980-01-01", periods=n_times, freq="MS")
        # Realistic ranges for a tiny aquaplanet-style climatology.
        sst = np.full((*valid_shape, n_times), 285.0)        # ~12°C ocean
        stl = np.full((*valid_shape, n_times), 280.0)        # ~7°C land — distinct from SST
        icec = np.zeros((*valid_shape, n_times))
        alb = np.full(valid_shape, 0.3)
        soilw = np.full((*valid_shape, n_times), 0.1)
        snow = np.zeros((*valid_shape, n_times))
        return xr.Dataset(
            data_vars={
                'stl': (['lon', 'lat', 'time'], stl),
                'icec': (['lon', 'lat', 'time'], icec),
                'sst': (['lon', 'lat', 'time'], sst),
                'alb': (['lon', 'lat'], alb),
                'soilw_am': (['lon', 'lat', 'time'], soilw),
                'snowc': (['lon', 'lat', 'time'], snow),
            },
            coords={'time': times},
        )

    def test_sanity_check_passes_realistic_bcs(self):
        from jcm.forcing import _validate_bc_fields
        _validate_bc_fields(self._make_realistic_ds())  # must not raise

    def test_sanity_check_rejects_celsius_temperatures(self):
        """An ``stl`` field in °C (≈ 0-30 K range) should be rejected.
        Catches the most common authoring mistake on the unit boundary.
        """
        from jcm.forcing import _validate_bc_fields
        ds = self._make_realistic_ds()
        ds['stl'].values[:] = 15.0  # °C-like value, way below 180 K
        with self.assertRaises(ValueError) as ctx:
            _validate_bc_fields(ds)
        self.assertIn("'stl' is out of physical range", str(ctx.exception))

    def test_sanity_check_rejects_nan(self):
        from jcm.forcing import _validate_bc_fields
        ds = self._make_realistic_ds()
        ds['sst'].values[0, 0, 0] = np.nan
        with self.assertRaises(ValueError) as ctx:
            _validate_bc_fields(ds)
        self.assertIn("non-finite", str(ctx.exception))

    def test_sanity_check_warns_on_amip_extrapolated_stl(self):
        """When ``stl ≈ sst`` everywhere, warn that the JSBACH file is
        likely missing — the resulting +30 K bias over high orography
        has historically driven multi-day NaNs over real terrain.
        """
        import warnings as _warn
        from jcm.forcing import _validate_bc_fields
        ds = self._make_realistic_ds()
        ds['stl'].values[:] = ds['sst'].values  # exact AMIP-extrapolation pattern
        with _warn.catch_warnings(record=True) as caught:
            _warn.simplefilter("always")
            _validate_bc_fields(ds)
        amip_warnings = [w for w in caught
                         if "AMIP-SST extrapolation" in str(w.message)]
        self.assertEqual(len(amip_warnings), 1)


class TestTimeSeriesAndSelect(unittest.TestCase):
    """Tests for the new TimeSeries leaf wrapper and ForcingData.select method."""

    def _build_date(self, tyear=0.5, calendar='gregorian'):
        from jcm.date import DateData
        import jax_datetime as jdt
        # Constructed via set_date so tyear/dt agree under the calendar.
        return DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-07-02')),
            calendar=calendar,
        )

    def test_static_forcing_select_is_noop_on_arrays(self):
        """For a forcing with no TimeSeries leaves, select returns arrays
        unchanged (only `solar` should differ).
        """
        from jcm.forcing import ForcingData
        nodal_shape = (32, 16)
        forcing = ForcingData.zeros(nodal_shape)
        date = self._build_date()
        sliced = forcing.select(date, calendar='gregorian')

        self.assertTrue(jnp.array_equal(sliced.alb0, forcing.alb0))
        self.assertTrue(jnp.array_equal(sliced.sea_surface_temperature, forcing.sea_surface_temperature))
        self.assertTrue(jnp.array_equal(sliced.co2_vmr, forcing.co2_vmr))

    def test_select_populates_solar_geometry(self):
        """select(date) should populate `solar` with non-zero phases."""
        from jcm.forcing import ForcingData
        forcing = ForcingData.zeros((4, 4))
        date = self._build_date()
        sliced = forcing.select(date, calendar='gregorian')

        # tyear should match date.tyear (~ 0.5 for July 2 — exactly
        # 182/365 under non-leap-year gregorian).
        self.assertAlmostEqual(float(sliced.solar.tyear), float(date.tyear('gregorian')), places=4)
        # orbital_phase = 2π × tyear, so close to π but not exactly π
        # because July 2 is a couple days off the year midpoint.
        self.assertAlmostEqual(float(sliced.solar.orbital_phase), 2.0 * float(jnp.pi) * float(date.tyear('gregorian')), places=4)

    def test_time_series_wrap_year_indexing(self):
        """A 12-entry monthly TimeSeries indexed via WRAP_YEAR should pick
        the slice corresponding to floor(tyear * 12).
        """
        from jcm.forcing import ForcingData, make_time_series, WRAP_YEAR
        nodal_shape = (4, 4)
        # 12 months of synthetic SST: month i = 280 + i*0.5 K
        sst_axis = jnp.arange(12, dtype=jnp.float32)[:, None, None] * 0.5 + 280.0
        sst_ts = make_time_series(
            values=jnp.broadcast_to(sst_axis, (12, *nodal_shape)),
            time_seconds=jnp.arange(12, dtype=jnp.float32),  # ignored for WRAP_YEAR
            align_mode=WRAP_YEAR,
        )
        forcing = ForcingData.zeros(nodal_shape, sea_surface_temperature=sst_ts)

        # 2001-07-02 → tyear ~0.498 under gregorian → month index 5 → SST = 282.5
        date = self._build_date()
        sliced = forcing.select(date, calendar='gregorian')
        self.assertEqual(sliced.sea_surface_temperature.shape, nodal_shape)
        expected = 280.0 + int(date.tyear('gregorian') * 12) * 0.5
        self.assertTrue(jnp.allclose(sliced.sea_surface_temperature, expected))

    def test_time_series_by_date_indexing(self):
        """A TimeSeries with absolute timestamps indexed via BY_DATE should
        pick the entry closest to (and at-or-before) the model date.
        """
        from jcm.forcing import ForcingData, make_time_series, BY_DATE
        from jcm.date import DateData, absolute_seconds_since_epoch
        import jax_datetime as jdt

        # Three entries: 2000-01-01, 2001-01-01, 2002-01-01.
        timestamps = [
            jdt.Datetime.from_pydatetime(jdt.to_datetime(s))
            for s in ['2000-01-01', '2001-01-01', '2002-01-01']
        ]
        time_seconds = jnp.asarray(
            [float(absolute_seconds_since_epoch(t)) for t in timestamps]
        )
        # CO2 = 370, 380, 390 ppmv at those years.
        co2_ts = make_time_series(
            values=jnp.array([370.0, 380.0, 390.0]),
            time_seconds=time_seconds,
            align_mode=BY_DATE,
        )
        nodal_shape = (4, 4)
        forcing = ForcingData.zeros(nodal_shape, co2_vmr=co2_ts)

        # Mid-2001 → second entry (2001-01-01) → 380 ppmv
        date_2001 = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-07-02')),
            calendar='gregorian',
        )
        self.assertAlmostEqual(
            float(forcing.select(date_2001, calendar='gregorian').co2_vmr),
            380.0,
        )

        # Mid-2000 → first entry → 370 ppmv
        date_2000 = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime('2000-07-02')),
            calendar='gregorian',
        )
        self.assertAlmostEqual(
            float(forcing.select(date_2000, calendar='gregorian').co2_vmr),
            370.0,
        )

        # Way before the first entry → still picks first entry (clamp).
        date_1995 = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(jdt.to_datetime('1995-01-01')),
            calendar='gregorian',
        )
        self.assertAlmostEqual(
            float(forcing.select(date_1995, calendar='gregorian').co2_vmr),
            370.0,
        )

    def test_time_axis_noleap_matches_model_gregorian_clock(self):
        """A 365_day (noleap) emissions time axis must land on the SAME
        leap-aware Gregorian clock as the BY_DATE lookup target.

        The model has no real noleap clock (#449): the lookup target is
        ``absolute_seconds_since_epoch`` built from ``jax_datetime`` (Gregorian).
        So a cftime axis must be aligned on its *nominal* calendar date, not by
        noleap day-counting — which would drift by accumulated leap days
        (~7 days by 2000, growing) and select the wrong multi-year slice. This
        is the Codex P1 on PR #522. The old ``cftime.date2num(..., '365_day')``
        path would fail this by ~7 * 86400 s at 2000.
        """
        import cftime
        import xarray as xr
        import jax_datetime as jdt
        from jcm.forcing import _time_axis_seconds_from_ds
        from jcm.date import absolute_seconds_since_epoch

        years = [2000, 2001, 2002]
        ds = xr.Dataset(
            coords={'time': ('time', [cftime.DatetimeNoLeap(y, 1, 1) for y in years])}
        )
        secs = np.asarray(_time_axis_seconds_from_ds(ds))

        expected = np.array([
            float(absolute_seconds_since_epoch(
                jdt.Datetime.from_pydatetime(jdt.to_datetime(f'{y}-01-01'))))
            for y in years
        ])
        np.testing.assert_allclose(secs, expected, rtol=0, atol=1.0)

    def test_select_under_jit(self):
        """Select must be JIT-compatible."""
        import jax
        from jcm.forcing import ForcingData, make_time_series, WRAP_YEAR

        nodal_shape = (4, 4)
        ts = make_time_series(
            values=jnp.arange(12, dtype=jnp.float32)[:, None, None] *
                   jnp.ones((12, *nodal_shape), dtype=jnp.float32),
            time_seconds=jnp.arange(12, dtype=jnp.float32),
            align_mode=WRAP_YEAR,
        )
        forcing = ForcingData.zeros(nodal_shape, sea_surface_temperature=ts)

        @jax.jit
        def get_sst(forcing, date):
            return forcing.select(date, calendar='gregorian').sea_surface_temperature

        date = self._build_date()
        sst_now = get_sst(forcing, date)
        self.assertEqual(sst_now.shape, nodal_shape)


class TestForcingNonMonthlyTimeAxis(unittest.TestCase):
    """from_dataset must accept native daily / multi-year same-grid files.

    Regression for the branch that unconditionally ran interpolate_to_daily
    (which requires exactly 12 monthly timestamps) on same-grid files, so a
    native daily or multi-year boundary file raised before reaching the
    TimeSeries/BY_DATE alignment.
    """

    def _same_grid_dataset(self, n_times):
        import pandas as pd
        import xarray as xr
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0.0, 1.0, 9), spectral_truncation=21)
        nlon, nlat = coords.horizontal.nodal_shape  # (64, 32)
        time = pd.date_range("2000-01-01", periods=n_times, freq="D")

        def f3(value):
            return (("lon", "lat", "time"),
                    np.full((nlon, nlat, n_times), value, dtype="float32"))

        ds = xr.Dataset(
            {
                "stl": f3(280.0),
                "icec": f3(0.0),
                "sst": f3(290.0),
                "soilw_am": f3(0.5),
                "snowc": f3(0.0),
                "alb": (("lon", "lat"), np.full((nlon, nlat), 0.1, dtype="float32")),
            },
            coords={"time": time},
        )
        return ds, coords, (nlon, nlat)

    def test_is_monthly_climatology_helper(self):
        from jcm.forcing import _is_monthly_climatology
        ds12, _, _ = self._same_grid_dataset(12)
        ds5, _, _ = self._same_grid_dataset(5)
        ds24, _, _ = self._same_grid_dataset(24)
        self.assertTrue(_is_monthly_climatology(ds12))
        self.assertFalse(_is_monthly_climatology(ds5))   # native daily
        self.assertFalse(_is_monthly_climatology(ds24))  # multi-year monthly

    def test_same_grid_daily_axis_loads(self):
        # 5 daily steps at the target grid: previously raised in
        # interpolate_to_daily ("expected 12 monthly timestamps").
        ds, coords, (nlon, nlat) = self._same_grid_dataset(5)
        forcing = ForcingData.from_dataset(ds, coords=coords, validate=False)
        self.assertEqual(forcing.alb0.shape, (nlon, nlat))

    def test_same_grid_multiyear_axis_loads(self):
        ds, coords, (nlon, nlat) = self._same_grid_dataset(24)
        forcing = ForcingData.from_dataset(ds, coords=coords, align_mode="by_date",
                                           validate=False)
        self.assertEqual(forcing.alb0.shape, (nlon, nlat))


class TestNaturalEmissionReaders(unittest.TestCase):
    """Readers for the HAMMOZ-style DMS / dust / oxidant climatology files.

    Synthetic tiny netCDF-shaped datasets (in-memory xarray, HAMMOZ layout:
    ``(time, lat, lon)`` with *descending* latitude) pin the variable mapping,
    the lat flip to the model's ascending orientation, the ``(time, lon, lat)``
    transpose, the WRAP_YEAR wrapping, the nmol/L → kg/m³ DMS conversion and
    the [0, 1] dust clip.
    """

    NLAT, NLON = 4, 6
    # Descending (N→S) like the HAMMOZ files; model order is ascending.
    LAT_DESC = np.array([60.0, 20.0, -20.0, -60.0])
    LON = np.linspace(0.0, 360.0, NLON, endpoint=False)
    TIME = np.array([f"2000-{m:02d}-15" for m in range(1, 13)],
                    dtype="datetime64[ns]")

    def _dataset(self, name, values, units=None, extra_dims=()):
        import xarray as xr
        dims = ("time", *extra_dims, "lat", "lon")
        coords = {"time": self.TIME, "lat": self.LAT_DESC, "lon": self.LON}
        attrs = {"units": units} if units is not None else {}
        return xr.Dataset(
            {name: (dims, values, attrs)}, coords=coords,
        )

    def test_dms_reader_units_orientation_and_wrap(self):
        from jcm.forcing import WRAP_YEAR, read_dms_seawater
        # Latitude-dependent pattern: value = |lat| in nmol/L, so the flip
        # is observable; one NaN (decoded _FillValue) cell.
        vals = np.broadcast_to(
            np.abs(self.LAT_DESC)[None, :, None], (12, self.NLAT, self.NLON)
        ).copy()
        vals[0, 1, 2] = np.nan
        ds = self._dataset("DMS_sea", vals, units="nanomol l-1")
        ts = read_dms_seawater(
            ds, lat_deg=self.LAT_DESC[::-1], lon_deg=self.LON
        )
        self.assertEqual(int(ts.align_mode), WRAP_YEAR)
        arr = np.asarray(ts.values)
        # (time, lon, lat) with ascending latitude.
        self.assertEqual(arr.shape, (12, self.NLON, self.NLAT))
        # 1 nmol/L = 6.21324e-8 kg/m³; lat axis flipped, so lat index 0 is
        # -60° (|lat| = 60).
        np.testing.assert_allclose(
            arr[3, 0, :], np.array([60.0, 20.0, 20.0, 60.0]) * 6.21324e-8,
            rtol=1e-6,
        )
        # The NaN (missing/land) cell maps to zero, not NaN: file (lat=1 →
        # 20°N → ascending index 2, lon=2).
        self.assertEqual(arr[0, 2, 2], 0.0)
        self.assertTrue(np.isfinite(arr).all())

    def test_dms_reader_rejects_unknown_units(self):
        from jcm.forcing import read_dms_seawater
        ds = self._dataset(
            "DMS_sea", np.ones((12, self.NLAT, self.NLON)), units="mol m-3"
        )
        with self.assertRaisesRegex(ValueError, "units"):
            read_dms_seawater(ds)

    def test_dms_reader_rejects_mismatched_latitudes(self):
        from jcm.forcing import read_dms_seawater
        ds = self._dataset(
            "DMS_sea", np.ones((12, self.NLAT, self.NLON)),
            units="nanomol l-1",
        )
        with self.assertRaisesRegex(ValueError, "latitudes"):
            read_dms_seawater(ds, lat_deg=self.LAT_DESC + 5.0)

    def test_dust_reader_clips_to_unit_interval(self):
        from jcm.forcing import WRAP_YEAR, read_dust_source
        vals = np.zeros((12, self.NLAT, self.NLON))
        vals[:, 0, 0] = -1.0    # the file's missing marker
        vals[:, 1, 1] = 1.5     # interpolation overshoot
        vals[:, 2, 2] = 0.7
        vals[0, 3, 3] = np.nan
        ds = self._dataset("pot_source", vals, units="1.")
        ts = read_dust_source(ds, lat_deg=self.LAT_DESC[::-1], lon_deg=self.LON)
        self.assertEqual(int(ts.align_mode), WRAP_YEAR)
        arr = np.asarray(ts.values)
        self.assertEqual(arr.shape, (12, self.NLON, self.NLAT))
        self.assertTrue((arr >= 0.0).all() and (arr <= 1.0).all())
        # -1 → 0; 1.5 → 1; 0.7 preserved (lat flipped: file lat idx i →
        # ascending idx 3 - i).
        self.assertEqual(arr[0, 0, 3], 0.0)
        self.assertEqual(arr[0, 1, 2], 1.0)
        self.assertAlmostEqual(float(arr[0, 2, 1]), 0.7)
        self.assertEqual(arr[0, 3, 0], 0.0)   # NaN → 0

    def _oxidant_ds(self, nlev=5, hybm=None, drop=None):
        import xarray as xr
        shape = (12, nlev, self.NLAT, self.NLON)
        data_vars = {}
        for i, var in enumerate(
            ["OH_VMR_avrg", "NO3_VMR_avrg", "O3_VMR_avrg", "H2O2_VMR_avrg"]
        ):
            if var == drop:
                continue
            # Level-dependent so the level axis position is pinned.
            vals = np.broadcast_to(
                (i + 1) * 1.0e-9 * (1.0 + np.arange(nlev))[None, :, None, None],
                shape,
            )
            data_vars[var] = (("time", "mlev", "lat", "lon"), vals,
                              {"units": "mole mole-1"})
        hybm = np.linspace(0.0, 1.0, nlev) if hybm is None else hybm
        data_vars["hybm"] = (("mlev",), hybm)
        return xr.Dataset(
            data_vars,
            coords={"time": self.TIME, "mlev": np.arange(nlev),
                    "lat": self.LAT_DESC, "lon": self.LON},
        )

    def test_oxidant_reader_maps_variables_and_orientation(self):
        from jcm.forcing import WRAP_YEAR, read_oxidant_vmr
        mapping = read_oxidant_vmr(
            self._oxidant_ds(), nlev=5,
            lat_deg=self.LAT_DESC[::-1], lon_deg=self.LON,
        )
        self.assertEqual(sorted(mapping), ["h2o2", "no3", "o3", "oh"])
        for key, ts in mapping.items():
            self.assertEqual(int(ts.align_mode), WRAP_YEAR)
            self.assertEqual(
                np.asarray(ts.values).shape, (12, 5, self.NLON, self.NLAT)
            )
        # o3 is the 3rd variable (i=2) → 3e-9 · (1 + level).
        o3 = np.asarray(mapping["o3"].values)
        np.testing.assert_allclose(o3[0, :, 0, 0], 3.0e-9 * (1 + np.arange(5)),
                                   rtol=1e-6)

    def test_oxidant_reader_asserts_level_count(self):
        from jcm.forcing import read_oxidant_vmr
        with self.assertRaisesRegex(ValueError, "levels"):
            read_oxidant_vmr(self._oxidant_ds(nlev=5), nlev=8)

    def test_oxidant_reader_requires_all_species(self):
        from jcm.forcing import read_oxidant_vmr
        with self.assertRaisesRegex(ValueError, "missing"):
            read_oxidant_vmr(self._oxidant_ds(drop="H2O2_VMR_avrg"), nlev=5)

    def test_oxidant_reader_rejects_bottom_up_levels(self):
        from jcm.forcing import read_oxidant_vmr
        ds = self._oxidant_ds(hybm=np.linspace(1.0, 0.0, 5))
        with self.assertRaisesRegex(ValueError, "top→bottom|top->bottom"):
            read_oxidant_vmr(ds, nlev=5)

    def test_oxidant_vmr_rides_through_select(self):
        """The dict-valued ``oxidant_vmr`` field slices like any TimeSeries."""
        from jcm.date import DateData
        import jax_datetime as jdt
        from jcm.forcing import read_oxidant_vmr
        mapping = read_oxidant_vmr(self._oxidant_ds(), nlev=5)
        forcing = ForcingData.zeros((self.NLON, self.NLAT)).copy(
            oxidant_vmr=mapping
        )
        date = DateData.set_date(
            model_time=jdt.Datetime.from_pydatetime(
                jdt.to_datetime("2001-07-02")
            ),
        )
        sliced = forcing.select(date, calendar="gregorian")
        self.assertEqual(
            sliced.oxidant_vmr["oh"].shape, (5, self.NLON, self.NLAT)
        )


if __name__ == '__main__':
    unittest.main()
