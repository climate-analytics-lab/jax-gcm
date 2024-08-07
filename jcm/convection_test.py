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
        itop = jnp.ones((ix, il), dtype=int) * (kx - 2)
        cbmf = jnp.ones((ix, il))
        precnv = jnp.ones((ix, il))
        dfse = jnp.ones((ix, il, kx))
        dfqa = jnp.ones((ix, il, kx))

        itop, dfse, dfqa, cbmf, precnv = get_convection_tendencies(psa, se, qa, qsat, itop, cbmf, precnv, dfse, dfqa)

        # Check that  dfse, dfqa, cbmf and precnv is not null.
        self.assertIsNotNone(itop)
        self.assertIsNotNone(dfse)
        self.assertIsNotNone(dfqa)
        self.assertIsNotNone(cbmf)
        self.assertIsNotNone(precnv)
