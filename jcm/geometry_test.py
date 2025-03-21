import unittest

class TestGeometryUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx, Geometry
        from jcm.geometry import Geometry
        ix, il, kx = 96, 48, 8

    def test_initialize_geometry(self):
        geo = Geometry.initialize_geometry((ix, il), kx)
        # Check that hsg is not null.
        self.assertIsNotNone(geo.hsg)
