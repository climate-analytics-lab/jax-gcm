import unittest
import jax.numpy as jnp
from jax import random
from jax import jit, vmap

# Assume the function is defined in a module named `my_module`
from jcm.shortwave_radiation import get_zonal_average_fields

class TestGetZonalAverageFields(unittest.TestCase):
    
    def setUp(self):
        # Set up any required test data here
        self.tyear = 0.5
        self.il = 10
        self.ix = 5
        self.solc = 1361  # Example solar constant in W/m^2
        self.epssw = 0.2  # Example ozone absorption constant
        self.sia = jnp.linspace(-1.0, 1.0, self.il)
        self.coa = jnp.sqrt(1 - self.sia ** 2)  # Assuming cosine of latitude
        self.rng_key = random.PRNGKey(0)  # Random key for reproducibility

    def test_output_shape(self):
        fsol, ozupp, ozone, stratz = get_zonal_average_fields(
            self.tyear, self.sia, self.coa, self.solc, self.il, self.ix, self.epssw)
        
        self.assertEqual(fsol.shape, (self.ix, self.il))
        self.assertEqual(ozupp.shape, (self.ix, self.il))
        self.assertEqual(ozone.shape, (self.ix, self.il))
        self.assertEqual(stratz.shape, (self.ix, self.il))

    def test_values(self):
        fsol, ozupp, ozone, stratz = get_zonal_average_fields(
            self.tyear, self.sia, self.coa, self.solc, self.il, self.ix, self.epssw)

        # Perform assertions on the values
        self.assertTrue(jnp.all(fsol >= 0))  # Solar radiation should be non-negative
        self.assertTrue(jnp.all(ozupp >= 0))  # Ozone depth should be non-negative
        self.assertTrue(jnp.all(ozone >= 0))  # Ozone concentration should be non-negative
        self.assertTrue(jnp.all(stratz >= 0))  # Polar night cooling should be non-negative

    def test_edge_cases(self):
        # Test with edge case values
        edge_sia = jnp.array([0.0] * self.il)
        edge_coa = jnp.array([1.0] * self.il)
        
        fsol, ozupp, ozone, stratz = get_zonal_average_fields(
            self.tyear, edge_sia, edge_coa, self.solc, self.il, self.ix, self.epssw)
        
        self.assertTrue(jnp.all(fsol >= 0))
        self.assertTrue(jnp.all(ozupp >= 0))
        self.assertTrue(jnp.all(ozone >= 0))
        self.assertTrue(jnp.all(stratz >= 0))
    
    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            get_zonal_average_fields(
                self.tyear, self.sia, self.coa, "invalid_solc", self.il, self.ix, self.epssw)
        
        with self.assertRaises(ValueError):
            get_zonal_average_fields(
                self.tyear, self.sia, self.coa, self.solc, -1, self.ix, self.epssw)
        
        with self.assertRaises(ValueError):
            get_zonal_average_fields(
                self.tyear, self.sia, self.coa, self.solc, self.il, self.ix, -0.1)

# if __name__ == '__main__':
#     unittest.main()