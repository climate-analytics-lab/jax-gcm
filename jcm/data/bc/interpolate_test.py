import unittest
from jcm.data.bc.interpolate import main

class TestInterpolateUnit(unittest.TestCase):

    def test_interpolate(self):
        # Just test that the interpolate main function runs without error (even if run multiple times)
        self.assertEqual(main(['31']), 0)
        self.assertEqual(main(['31']), 0)