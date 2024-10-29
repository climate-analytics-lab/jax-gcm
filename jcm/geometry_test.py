import unittest

class TestGeometryUnit(unittest.TestCase):

    def setUp():
        global geo
        from jcm import geometry as geo

    def test_initialize_geometry(self):
        geo.initialize_geometry()

        # Check that hsg is not null.
        self.assertIsNotNone(geo.hsg)
