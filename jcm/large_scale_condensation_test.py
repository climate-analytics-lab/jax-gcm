import unittest
import large_scale_condensation
import jax.numpy as jnp

class TestLargeScaleCondensationUnit(unittest.TestCase):

    def test_get_large_scale_condensation_tendencies(self):
        ix, il, kx = 1, 1, 8
        psa = jnp.ones((ix, il))
        qa = jnp.ones((ix, il, kx))
        qsat = jnp.ones((ix, il, kx))
        itop = jnp.full((ix, il), kx - 1)

        itop, precls, dtlsc, dqlsc = large_scale_condensation.get_large_scale_condensation_tendencies(psa, qa, qsat, itop, fsg, dhs, p0, cp, alhc, grav)
        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertIsNotNone(itop)
        self.assertIsNotNone(precls)
        self.assertIsNotNone(dtlsc)
        self.assertIsNotNone(dqlsc)

unittest.main()