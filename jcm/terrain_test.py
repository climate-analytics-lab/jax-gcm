"""Tests for jcm/terrain.py

Tests for TerrainData struct and get_terrain function.
"""

import unittest
import jax.numpy as jnp
from jcm.terrain import TerrainData, get_terrain
from jcm.constants import grav
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class TestGetTerrain(unittest.TestCase):
    """Tests for the get_terrain function."""

    def test_get_terrain_with_nodal_shape_only(self):
        """When only nodal_shape is provided, should return zeros for both orography and fmask."""
        nodal_shape = (96, 48)
        orography, fmask = get_terrain(nodal_shape=nodal_shape)

        self.assertEqual(orography.shape, nodal_shape)
        self.assertEqual(fmask.shape, nodal_shape)
        self.assertTrue(jnp.allclose(orography, 0.0))
        self.assertTrue(jnp.allclose(fmask, 0.0))

    def test_get_terrain_no_inputs_raises_error(self):
        """Should raise ValueError when no inputs are provided."""
        with self.assertRaises(ValueError) as context:
            get_terrain()
        self.assertIn("Must provide at least one of", str(context.exception))

    def test_get_terrain_orography_only(self):
        """When only orography is provided, fmask should be derived from orography > 0."""
        nodal_shape = (64, 32)
        # Create orography with some areas elevated and some at sea level
        orography = jnp.zeros(nodal_shape)
        orography = orography.at[:32, :16].set(500.)  # Mountains in one quadrant

        result_orog, result_fmask = get_terrain(orography=orography)

        # fmask should be 1 where orography > 0, 0 otherwise
        expected_fmask = (orography > 0.0).astype(jnp.float32)
        self.assertTrue(jnp.allclose(result_orog, orography))
        self.assertTrue(jnp.allclose(result_fmask, expected_fmask))

    def test_get_terrain_fmask_only(self):
        """When only fmask is provided, orography should default to zeros."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.5

        result_orog, result_fmask = get_terrain(fmask=fmask)

        self.assertTrue(jnp.allclose(result_orog, 0.0))
        self.assertEqual(result_fmask.shape, nodal_shape)

    def test_get_terrain_both_provided(self):
        """When both orography and fmask are provided, use them directly."""
        nodal_shape = (64, 32)
        orography = jnp.ones(nodal_shape) * 500.
        fmask = jnp.ones(nodal_shape) * 0.7

        result_orog, result_fmask = get_terrain(orography=orography, fmask=fmask)

        self.assertTrue(jnp.allclose(result_orog, orography))
        self.assertTrue(jnp.allclose(result_fmask, 0.7))

    def test_get_terrain_fmask_threshold_low(self):
        """Values below fmask_threshold should be set to exactly 0."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.05  # Below default threshold of 0.1

        result_orog, result_fmask = get_terrain(fmask=fmask)

        self.assertTrue(jnp.allclose(result_fmask, 0.0))

    def test_get_terrain_fmask_threshold_high(self):
        """Values above 1.0 - fmask_threshold should be set to exactly 1."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.95  # Above 1.0 - 0.1 = 0.9

        result_orog, result_fmask = get_terrain(fmask=fmask)

        self.assertTrue(jnp.allclose(result_fmask, 1.0))

    def test_get_terrain_fmask_threshold_custom(self):
        """Custom fmask_threshold should be respected."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.25

        # With threshold 0.1, 0.25 is in the middle range
        result_orog, result_fmask = get_terrain(fmask=fmask, fmask_threshold=0.1)
        self.assertTrue(jnp.allclose(result_fmask, 0.25))

        # With threshold 0.3, 0.25 should be rounded to 0
        result_orog, result_fmask = get_terrain(fmask=fmask, fmask_threshold=0.3)
        self.assertTrue(jnp.allclose(result_fmask, 0.0))


class TestTerrainDataCopy(unittest.TestCase):
    """Tests for TerrainData.copy method."""

    def test_copy_no_changes(self):
        """Copy with no arguments should return identical data."""
        nodal_shape = (64, 32)
        terrain = TerrainData(
            orog=jnp.ones(nodal_shape) * 100.,
            phis0=jnp.ones(nodal_shape) * grav * 100.,
            fmask=jnp.ones(nodal_shape) * 0.5,
            lfluxland=jnp.bool_(True)
        )

        copied = terrain.copy()

        self.assertTrue(jnp.allclose(copied.orog, terrain.orog))
        self.assertTrue(jnp.allclose(copied.phis0, terrain.phis0))
        self.assertTrue(jnp.allclose(copied.fmask, terrain.fmask))
        self.assertEqual(bool(copied.lfluxland), bool(terrain.lfluxland))

    def test_copy_with_changes(self):
        """Copy with arguments should replace those fields."""
        nodal_shape = (64, 32)
        terrain = TerrainData(
            orog=jnp.ones(nodal_shape) * 100.,
            phis0=jnp.ones(nodal_shape) * grav * 100.,
            fmask=jnp.ones(nodal_shape) * 0.5,
            lfluxland=jnp.bool_(True)
        )

        new_orog = jnp.ones(nodal_shape) * 200.
        new_fmask = jnp.ones(nodal_shape) * 0.8
        copied = terrain.copy(orog=new_orog, fmask=new_fmask)

        self.assertTrue(jnp.allclose(copied.orog, 200.))
        self.assertTrue(jnp.allclose(copied.phis0, terrain.phis0))  # Unchanged
        self.assertTrue(jnp.allclose(copied.fmask, 0.8))
        self.assertEqual(bool(copied.lfluxland), True)  # Unchanged


class TestTerrainDataFromCoords(unittest.TestCase):
    """Tests for TerrainData.from_coords classmethod."""

    def test_from_coords_defaults(self):
        """Default from_coords should create aquaplanet-like terrain."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_coords(coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertEqual(terrain.phis0.shape, expected_shape)
        self.assertEqual(terrain.fmask.shape, expected_shape)
        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))
        self.assertTrue(jnp.allclose(terrain.fmask, 0.0))
        self.assertEqual(bool(terrain.lfluxland), False)

    def test_from_coords_with_orography(self):
        """from_coords with orography should compute fmask from orography."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nodal_shape = coords.horizontal.nodal_shape

        # Create orography with mountains in half the domain
        orography = jnp.zeros(nodal_shape)
        orography = orography.at[:nodal_shape[0]//2, :].set(1000.)

        terrain = TerrainData.from_coords(coords, orography=orography)

        # Check shapes
        self.assertEqual(terrain.orog.shape, nodal_shape)
        self.assertEqual(terrain.phis0.shape, nodal_shape)

        # fmask should be 1 where orography > 0
        expected_fmask = (orography > 0.0).astype(jnp.float32)
        self.assertTrue(jnp.allclose(terrain.fmask, expected_fmask))

    def test_from_coords_with_fmask(self):
        """from_coords with fmask but no orography should have flat terrain."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nodal_shape = coords.horizontal.nodal_shape

        fmask = jnp.ones(nodal_shape) * 0.5

        terrain = TerrainData.from_coords(coords, fmask=fmask)

        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))

    def test_from_coords_with_lfluxland(self):
        """from_coords lfluxland parameter should be respected."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)

        terrain_land = TerrainData.from_coords(coords, lfluxland=True)
        terrain_no_land = TerrainData.from_coords(coords, lfluxland=False)

        self.assertEqual(bool(terrain_land.lfluxland), True)
        self.assertEqual(bool(terrain_no_land.lfluxland), False)

    def test_from_coords_spectral_truncation_phis0(self):
        """phis0 should be spectrally truncated version of grav * orog."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nodal_shape = coords.horizontal.nodal_shape

        # Create varying orography
        orography = jnp.ones(nodal_shape) * 1000.

        terrain = TerrainData.from_coords(coords, orography=orography)

        # For uniform orography, spectral truncation should preserve the field
        expected_phis0 = grav * orography
        self.assertTrue(jnp.allclose(terrain.phis0, expected_phis0, rtol=1e-5))


class TestTerrainDataAquaplanet(unittest.TestCase):
    """Tests for TerrainData.aquaplanet classmethod."""

    def test_aquaplanet_all_zeros(self):
        """Aquaplanet terrain should have all zeros."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.aquaplanet(coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))
        self.assertTrue(jnp.allclose(terrain.fmask, 0.0))
        self.assertEqual(bool(terrain.lfluxland), False)

    def test_aquaplanet_different_resolutions(self):
        """Aquaplanet should work for different resolutions."""
        for truncation in [21, 31, 42]:
            coords = get_speedy_coords(layers=8, spectral_truncation=truncation)
            terrain = TerrainData.aquaplanet(coords)

            expected_shape = coords.horizontal.nodal_shape
            self.assertEqual(terrain.orog.shape, expected_shape)


class TestTerrainDataSingleColumn(unittest.TestCase):
    """Tests for TerrainData.single_column classmethod."""

    def test_single_column_defaults(self):
        """Default single column should be flat ocean."""
        terrain = TerrainData.single_column()

        self.assertEqual(terrain.orog.shape, (1, 1))
        self.assertEqual(terrain.phis0.shape, (1, 1))
        self.assertEqual(terrain.fmask.shape, (1, 1))
        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))
        self.assertTrue(jnp.allclose(terrain.fmask, 0.0))
        self.assertEqual(bool(terrain.lfluxland), False)

    def test_single_column_with_orography(self):
        """Single column with orography should compute phis0 correctly."""
        orog_height = 500.
        terrain = TerrainData.single_column(orog=orog_height)

        self.assertTrue(jnp.allclose(terrain.orog, orog_height))
        self.assertTrue(jnp.allclose(terrain.phis0, grav * orog_height))

    def test_single_column_with_fmask(self):
        """Single column fmask should be set correctly."""
        terrain = TerrainData.single_column(fmask=1.0)

        self.assertTrue(jnp.allclose(terrain.fmask, 1.0))

    def test_single_column_with_lfluxland(self):
        """Single column lfluxland should be set correctly."""
        terrain_land = TerrainData.single_column(lfluxland=True)
        terrain_ocean = TerrainData.single_column(lfluxland=False)

        self.assertEqual(bool(terrain_land.lfluxland), True)
        self.assertEqual(bool(terrain_ocean.lfluxland), False)

    def test_single_column_custom_phis0(self):
        """Single column with custom phis0 should override default calculation."""
        orog_height = 500.
        custom_phis0 = 1234.5  # Different from grav * orog

        terrain = TerrainData.single_column(orog=orog_height, phis0=custom_phis0)

        self.assertTrue(jnp.allclose(terrain.orog, orog_height))
        self.assertTrue(jnp.allclose(terrain.phis0, custom_phis0))


class TestTerrainDataFromFile(unittest.TestCase):
    """Tests for TerrainData.from_file classmethod using actual data files."""

    def test_from_file_loads_terrain(self):
        """from_file should load terrain data from actual NetCDF file."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertEqual(terrain.phis0.shape, expected_shape)
        self.assertEqual(terrain.fmask.shape, expected_shape)

    def test_from_file_has_realistic_orography(self):
        """Loaded terrain should have realistic orography values."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # Orography should be non-negative (no below sea level in this dataset)
        # and have a reasonable max (Mt Everest ~8849m)
        self.assertTrue(jnp.all(terrain.orog >= -500.))  # Allow some below sea level
        self.assertTrue(jnp.all(terrain.orog <= 9000.))

    def test_from_file_has_valid_fmask(self):
        """Loaded terrain should have fmask values in [0, 1]."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # fmask should be between 0 and 1
        self.assertTrue(jnp.all(terrain.fmask >= 0.0))
        self.assertTrue(jnp.all(terrain.fmask <= 1.0))

    def test_from_file_phis0_matches_grav_times_orog(self):
        """phis0 should be spectrally truncated grav * orog."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # phis0 should be approximately grav * orog (with some spectral truncation difference)
        phi0_direct = grav * terrain.orog
        # The spectral truncation can cause differences, but the mean should be close
        self.assertTrue(jnp.isclose(jnp.mean(terrain.phis0), jnp.mean(phi0_direct), rtol=0.01))

    def test_from_file_lfluxland_default_true(self):
        """from_file should default lfluxland to True."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        self.assertEqual(bool(terrain.lfluxland), True)

    def test_from_file_lfluxland_can_be_set_false(self):
        """from_file lfluxland parameter should be respected."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords, lfluxland=False)

        self.assertEqual(bool(terrain.lfluxland), False)

    def test_from_file_interpolates_to_different_resolution(self):
        """from_file should interpolate terrain when coords resolution differs."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        # Use T21 coords with T31 terrain file - should interpolate
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # Should match the target coords resolution
        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertEqual(terrain.fmask.shape, expected_shape)

        # Should still have physically valid data after interpolation
        self.assertTrue(jnp.all(terrain.orog >= -500.))
        self.assertTrue(jnp.all(terrain.fmask >= 0.0))
        self.assertTrue(jnp.all(terrain.fmask <= 1.0))


class TestTerrainDataTreeMath(unittest.TestCase):
    """Tests for JAX tree_math compatibility."""

    def test_terrain_data_is_jax_pytree(self):
        """TerrainData should be a valid JAX pytree."""
        import jax

        terrain = TerrainData.single_column(orog=100., fmask=0.5)

        # Should be able to tree_map over it
        doubled = jax.tree.map(lambda x: x * 2, terrain)

        self.assertTrue(jnp.allclose(doubled.orog, 200.))
        self.assertTrue(jnp.allclose(doubled.fmask, 1.0))

    def test_terrain_data_jit_compatible(self):
        """TerrainData should work with jax.jit."""
        import jax

        @jax.jit
        def get_phis0(terrain):
            return terrain.phis0

        terrain = TerrainData.single_column(orog=100., fmask=0.5)
        result = get_phis0(terrain)

        self.assertTrue(jnp.allclose(result, grav * 100.))


if __name__ == '__main__':
    unittest.main()
