import unittest
import geometry

class TestGeometryUnit(unittest.TestCase):

    def test_initialize_geometry(self):
        vals = geometry.initialize_geometry()

        # Check that hsg is not null.
        self.assertIsNotNone(vals[0])

unittest.main()