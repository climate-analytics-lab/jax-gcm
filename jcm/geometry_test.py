import unittest
import jax.tree_util as jtu
import jax.numpy as jnp
from jcm.geometry import get_terrain

class TestGeometryUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx, Geometry
        from jcm.geometry import Geometry
        ix, il, kx = 96, 48, 8

    def test_from_coords(self):
        from jcm.utils import get_coords
        coords = get_coords(layers=kx, spectral_truncation=31)
        geo = Geometry.from_coords(coords)
        has_nans = any(jnp.isnan(x).any() for x in jtu.tree_leaves(geo) if isinstance(x, jnp.ndarray))
        self.assertFalse(has_nans)

    def test_from_grid_shape(self):
        geo = Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx)
        has_nans = any(jnp.isnan(x).any() for x in jtu.tree_leaves(geo) if isinstance(x, jnp.ndarray))
        self.assertFalse(has_nans)

    def test_single_column(self):
        geo = Geometry.single_column_geometry(num_levels=kx)
        has_nans = any(jnp.isnan(x).any() for x in jtu.tree_leaves(geo) if isinstance(x, jnp.ndarray))
        self.assertFalse(has_nans)

class TestGetTerrain(unittest.TestCase):
    """Tests for the get_terrain function to ensure proper handling of edge cases."""

    def setUp(self):
        """Set up test data."""
        self.nodal_shape = (96, 48)
        self.test_orography = jnp.ones(self.nodal_shape) * 100.0  # 100m elevation
        self.test_fmask = jnp.ones(self.nodal_shape) * 0.5  # 50% land

    def test_both_provided(self):
        """Test that when both orography and fmask are provided, both are returned as-is."""
        orog, fmask = get_terrain(fmask=self.test_fmask, orography=self.test_orography)

        self.assertTrue(jnp.allclose(orog, self.test_orography))
        self.assertTrue(jnp.allclose(fmask, self.test_fmask))
        self.assertEqual(orog.shape, self.nodal_shape)
        self.assertEqual(fmask.shape, self.nodal_shape)

    def test_only_orography_provided(self):
        """Test that when only orography is provided, fmask defaults to land where orography >0 (all land in test)."""
        orog, fmask = get_terrain(orography=self.test_orography)

        self.assertTrue(jnp.allclose(orog, self.test_orography))
        self.assertTrue(jnp.allclose(fmask, jnp.ones(self.nodal_shape)))
        self.assertEqual(orog.shape, self.nodal_shape)
        self.assertEqual(fmask.shape, self.nodal_shape)

    def test_only_fmask_provided(self):
        """Test that when only fmask is provided, orography defaults to zeros (flat)."""
        orog, fmask = get_terrain(fmask=self.test_fmask)

        self.assertTrue(jnp.allclose(orog, jnp.zeros(self.nodal_shape)))
        self.assertTrue(jnp.allclose(fmask, self.test_fmask))
        self.assertEqual(orog.shape, self.nodal_shape)
        self.assertEqual(fmask.shape, self.nodal_shape)

    def test_nodal_shape_provided(self):
        """Test that when only nodal_shape is provided, both default to zeros."""
        orog, fmask = get_terrain(nodal_shape=self.nodal_shape)

        self.assertTrue(jnp.allclose(orog, jnp.zeros(self.nodal_shape)))
        self.assertTrue(jnp.allclose(fmask, jnp.zeros(self.nodal_shape)))
        self.assertEqual(orog.shape, self.nodal_shape)
        self.assertEqual(fmask.shape, self.nodal_shape)

    def test_nothing_provided_raises_error(self):
        """Test that when nothing is provided, a ValueError is raised."""
        with self.assertRaises(ValueError) as context:
            get_terrain()

        self.assertIn("Must provide at least one of", str(context.exception))

    def test_orography_takes_precedence_over_nodal_shape(self):
        """Test that when both orography and nodal_shape are provided, orography is used."""
        orog, fmask = get_terrain(orography=self.test_orography, nodal_shape=(64, 32))

        # Should use orography shape, not nodal_shape
        self.assertEqual(orog.shape, self.nodal_shape)
        self.assertEqual(fmask.shape, self.nodal_shape)
        self.assertTrue(jnp.allclose(orog, self.test_orography))

    def test_fmask_takes_precedence_over_nodal_shape(self):
        """Test that when both fmask and nodal_shape are provided, fmask is used."""
        orog, fmask = get_terrain(fmask=self.test_fmask, nodal_shape=(64, 32))

        # Should use fmask shape, not nodal_shape
        self.assertEqual(orog.shape, self.nodal_shape)
        self.assertEqual(fmask.shape, self.nodal_shape)
        self.assertTrue(jnp.allclose(fmask, self.test_fmask))

    def test_zeros_like_preserves_dtype(self):
        """Test that zeros_like preserves the dtype of the provided array."""
        # Test with float32
        orog_f32 = jnp.ones(self.nodal_shape, dtype=jnp.float32) * 100.0
        _, fmask = get_terrain(orography=orog_f32)
        self.assertEqual(fmask.dtype, jnp.float32)

        # Test with same dtype for fmask
        fmask_f32 = jnp.ones(self.nodal_shape, dtype=jnp.float32) * 0.5
        orog, _ = get_terrain(fmask=fmask_f32)
        self.assertEqual(orog.dtype, jnp.float32)

    def test_non_square_shapes(self):
        """Test that the function works with non-square nodal shapes."""
        non_square_shape = (8, 128, 64)  # (kx, ix, il) - note this is 3D
        orog_3d = jnp.ones(non_square_shape) * 50.0

        orog, fmask = get_terrain(orography=orog_3d)

        self.assertEqual(orog.shape, non_square_shape)
        self.assertEqual(fmask.shape, non_square_shape)
        self.assertTrue(jnp.allclose(orog, orog_3d))
        self.assertTrue(jnp.allclose(fmask, jnp.ones(non_square_shape)))


if __name__ == '__main__':
    unittest.main()
