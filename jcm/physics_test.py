import unittest
from jcm.physics import get_large_scale_condensation_tendencies
import jax.numpy as jnp

class TestPhysicsUnit(unittest.TestCase):

    def test_get_physical_tendencies(self):
        return 