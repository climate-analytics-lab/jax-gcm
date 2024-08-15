import unittest
import jax.numpy as jnp
from jax import random
from jcm.shortwave_radiation import get_zonal_average_fields,solar
from jcm.physical_constants import solc,epssw
from jcm.params import il, ix
from unittest.mock import patch
# Assuming the functions are imported from the module where they're defined
# from your_module import get_zonal_average_fields, solar

class TestZonalAverageFields(unittest.TestCase):

    def setUp(self):
        # Set up test case with known inputs
        self.tyear = 0.25  # Example time of the year (spring equinox)
        self.sia = jnp.linspace(-1, 1, il)  # Example sine of latitude array
        self.coa = jnp.linspace(0, 1, il)  # Example cosine of latitude array
        self.solc = solc
        self.il = il
        self.ix = ix
        self.epssw = epssw

    def test_output_shapes(self):
        # Ensure that the output shapes are correct
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear, self.sia, self.coa
        )
        
        self.assertEqual(fsol.shape, (self.ix, self.il))
        self.assertEqual(ozupp.shape, (self.ix, self.il))
        self.assertEqual(ozone.shape, (self.ix, self.il))
        self.assertEqual(stratz.shape, (self.ix, self.il))
        self.assertEqual(zenit.shape, (self.ix, self.il))

    def test_solar_radiation_values(self):
        # Test that the solar radiation values are computed correctly
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear, self.sia, self.coa
        )
        
        topsr = solar(self.tyear)
        self.assertTrue(jnp.allclose(fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear, self.sia, self.coa
        )
        
        fs0 = 6.0
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - fsol, 0) == stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear, self.sia, self.coa
        )
        
        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * self.sia ** 2 - 0.5
        expected_ozone = 0.4 * self.epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (self.tyear + 10.0 / 365.0))) * self.sia + 1.8 * flat2)
        self.assertTrue(jnp.allclose(ozone[:, 0], fsol[:, 0] * expected_ozone[0]))

    def test_random_input_consistency(self):
        # Check that random inputs produce consistent outputs
        key = random.PRNGKey(0)
        sia = random.uniform(key, (self.il,), minval=-1, maxval=1)
        coa = jnp.sqrt(1 - sia**2)
        
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear, sia, coa
        )
        
        # Ensure outputs are consistent and within expected ranges
        self.assertTrue(jnp.all(fsol >= 0))
        self.assertTrue(jnp.all(ozupp >= 0))
        self.assertTrue(jnp.all(ozone >= 0))
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(zenit >= 0))


# class TestZonalAverageFields(unittest.TestCase):

#     def setUp(self):
#         # Set up test case with known inputs
#         self.tyear = 0.25  # Example time of the year (spring equinox)
#         self.sia = jnp.linspace(-1, 1, il)  # Example sine of latitude array
#         self.coa = jnp.linspace(0, 1, il)  # Example cosine of latitude array
#         self.solc = solc
#         self.il = il
#         self.ix = ix
#         self.epssw = epssw

#     @patch('jcm.shortwave_radiation.solar')
#     def test_solar_radiation_values(self, mock_solar):
#         # Mocking the solar function to return a predefined value
#         mock_solar.return_value = jnp.array([500.0] * self.il)
        
#         fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
#             self.tyear, self.sia, self.coa
#         )
#         print(jnp.all(fsol[:,0]))
#         # Check that fsol uses the mocked solar values
#         self.assertTrue(jnp.all(fsol[:, 0] == 500.0))

#     @patch('jcm.shortwave_radiation.solar')
#     def test_polar_night_cooling(self, mock_solar):
#         # Mock the solar function to control the input values for testing polar night cooling
#         mock_solar.return_value = jnp.array([100.0] * self.il)
        
#         fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
#             self.tyear, self.sia, self.coa
#         )
        
#         fs0 = 6.0
#         self.assertTrue(jnp.all(stratz >= 0))
#         self.assertTrue(jnp.all(jnp.maximum(fs0 - fsol, 0) == stratz))

#     @patch('jcm.shortwave_radiation.solar')
#     def test_ozone_absorption(self, mock_solar):
#         # Mock the solar function for testing ozone absorption
#         mock_solar.return_value = jnp.array([300.0] * self.il)
        
#         fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
#             self.tyear, self.sia, self.coa
#         )
        
#         # Expected form for ozone based on the provided formula
#         flat2 = 1.5 * self.sia ** 2 - 0.5
#         expected_ozone = 0.4 * self.epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (self.tyear + 10.0 / 365.0))) * self.sia + 1.8 * flat2)
#         self.assertTrue(jnp.allclose(ozone[:, 0], 300.0 * expected_ozone[0]))

#     @patch('jcm.shortwave_radiation.solar')
#     def test_random_input_consistency(self, mock_solar):
#         # Mock the solar function for consistent outputs with random inputs
#         mock_solar.return_value = jnp.array([250.0] * self.il)
        
#         key = random.PRNGKey(0)
#         sia = random.uniform(key, (self.il,), minval=-1, maxval=1)
#         coa = jnp.sqrt(1 - sia**2)
        
#         fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
#             self.tyear, sia, coa
#         )
        
#         # Ensure outputs are consistent and within expected ranges
#         self.assertTrue(jnp.all(fsol >= 0))
#         self.assertTrue(jnp.all(ozupp >= 0))
#         self.assertTrue(jnp.all(ozone >= 0))
#         self.assertTrue(jnp.all(stratz >= 0))
#         self.assertTrue(jnp.all(zenit >= 0))