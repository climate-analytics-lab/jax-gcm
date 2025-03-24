import unittest

class TestGeometryUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx, Geometry
        from jcm.geometry import Geometry
        ix, il, kx = 96, 48, 8

    def test_initialize_geometry(self):
        geo = Geometry.from_grid_shape((ix, il), kx)
        # Check that hsg is not null.
        self.assertIsNotNone(geo.hsg)
