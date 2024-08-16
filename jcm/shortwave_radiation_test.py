import unittest
import jax.numpy as jnp
from jax import random
from jcm.shortwave_radiation import get_zonal_average_fields,solar
from jcm.physical_constants import solc,epssw
from jcm.params import il, ix
from unittest.mock import patch
# Assuming the functions are imported from the module where they're defined
# from your_module import get_zonal_average_fields, solar
from jcm.geometry import sia

class TestZonalAverageFields(unittest.TestCase):

    def setUp(self):
        # Set up test case with known inputs
        self.tyear = 0.25  # Example time of the year (spring equinox)
        self.solc = solc
        self.il = il
        self.ix = ix
        self.epssw = epssw

    def test_output_shapes(self):
        # Ensure that the output shapes are correct
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear
        )
        
        self.assertEqual(fsol.shape, (self.ix, self.il))
        self.assertEqual(ozupp.shape, (self.ix, self.il))
        self.assertEqual(ozone.shape, (self.ix, self.il))
        self.assertEqual(stratz.shape, (self.ix, self.il))
        self.assertEqual(zenit.shape, (self.ix, self.il))

    def test_solar_radiation_values(self):
        # Test that the solar radiation values are computed correctly
        fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(
            self.tyear
        )
        
        topsr = solar(self.tyear)
        self.assertTrue(jnp.allclose(fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        fsol, ozupp, ozone, zenit, stratz, = get_zonal_average_fields(
            self.tyear
        )
        
        fs0 = 6.0
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - fsol, 0) == stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(
            self.tyear
        )
        
        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * sia**2 - 0.5
        expected_ozone = 0.4 * self.epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (self.tyear + 10.0 / 365.0)))  + 1.8 * flat2)
        print
        self.assertTrue(jnp.allclose(ozone[:, 0], fsol[:, 0] * expected_ozone[0]))

    def test_random_input_consistency(self):
        # Check that random inputs produce consistent outputs
        key = random.PRNGKey(0)
        
        fsol, ozupp, ozone, zenit, stratz= get_zonal_average_fields(
            self.tyear
        )
        
        # Ensure outputs are consistent and within expected ranges
        self.assertTrue(jnp.all(fsol >= 0))
        self.assertTrue(jnp.all(ozupp >= 0))
        self.assertTrue(jnp.all(ozone >= 0))
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(zenit >= 0))