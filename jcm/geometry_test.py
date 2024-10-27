import unittest
from jcm import geometry

class TestGeometryUnit(unittest.TestCase):

    def test_initialize_geometry(self):
        geometry.initialize_geometry()

        # Check that hsg is not null.
        self.assertIsNotNone(geometry.hsg)
