import unittest
from jcm.convection import diagnose_convection, get_convection_tendencies
from jax import random
import jax.numpy as jnp

class TestConvectionUnit(unittest.TestCase):

    def test_diagnose_convection(self):
        key = random.PRNGKey(0)
        ix, il, kx = 4, 4, 10
        psa = random.uniform(key, (ix, il))
        se = random.uniform(key, (ix, il, kx))
        qa = random.uniform(key, (ix, il, kx))
        qsat = random.uniform(key, (ix, il, kx))

        itop, qdif = diagnose_convection(psa, se, qa, qsat)

        # Check that itop and qdif is not null.
        self.assertIsNotNone(itop)
        self.assertIsNotNone(qdif)
    
    def test_get_convective_tendencies(self):
        key = random.PRNGKey(0)
        ix, il, kx = 4, 4, 10
        psa = random.uniform(key, (ix, il))
        se = random.uniform(key, (ix, il, kx))
        qa = random.uniform(key, (ix, il, kx))
        qsat = random.uniform(key, (ix, il, kx))

        itop, dfse, dfqa, cbmf, precnv = get_convection_tendencies(psa, se, qa, qsat)

        # Check that  dfse, dfqa, cbmf and precnv is not null.
        self.assertIsNotNone(itop)
        self.assertIsNotNone(dfse)
        self.assertIsNotNone(dfqa)
        self.assertIsNotNone(cbmf)
        self.assertIsNotNone(precnv)
