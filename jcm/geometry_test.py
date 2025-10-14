import unittest
import jax.tree_util as jtu
import jax.numpy as jnp
class TestGeometryUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx, Geometry
        from jcm.geometry import Geometry
        ix, il, kx = 96, 48, 8

    def test_from_coords(self):
        from jcm.geometry import get_coords
        coords = get_coords(layers=kx, nodal_shape=(ix, il))
        geo = Geometry.from_coords(coords)
        has_nans = any(jnp.isnan(x).any() for x in jtu.tree_leaves(geo))
        self.assertFalse(has_nans)

    def test_from_grid_shape(self):
        geo = Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx)
        has_nans = any(jnp.isnan(x).any() for x in jtu.tree_leaves(geo))
        self.assertFalse(has_nans)

    def test_single_column(self):
        geo = Geometry.single_column_geometry(num_levels=kx)
        has_nans = any(jnp.isnan(x).any() for x in jtu.tree_leaves(geo))
        self.assertFalse(has_nans)