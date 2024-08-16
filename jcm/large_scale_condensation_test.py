import unittest
from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
import jax.numpy as jnp
import numpy as np

class TestLargeScaleCondensationUnit(unittest.TestCase):

    def test_get_large_scale_condensation_tendencies(self):
        ix, il, kx = 1, 1, 8
        psa = jnp.ones((ix, il))
        qa = jnp.ones((ix, il, kx))
        qsat = jnp.ones((ix, il, kx))
        itop = jnp.full((ix, il), kx - 1)

        itop, precls, dtlsc, dqlsc = get_large_scale_condensation_tendencies(psa, qa, qsat, itop)
        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertIsNotNone(itop)
        self.assertIsNotNone(precls)
        self.assertIsNotNone(dtlsc)
        self.assertIsNotNone(dqlsc)

    def test_get_large_scale_condensation_tendencies_realistic(self):
        ix, il, kx = 1, 1, 8
        psa = jnp.ones((ix, il)) * 1.0110737
        qa = jnp.asarray([[[16.148024  , 10.943978  ,  5.851813  ,  2.4522789 ,  0.02198645,
        0.16069981,  0.        ,  0.        ]]])
        qsat = jnp.asarray([[[1.64229703e-01, 1.69719307e-02, 1.45193088e-01, 1.98833509e+00,
       4.58917155e+00, 9.24226425e+00, 1.48490220e+01, 2.02474803e+01]]])
        itop = jnp.ones((ix, il)) * 4

        itop, precls, dtlsc, dqlsc = get_large_scale_condensation_tendencies(psa, qa, qsat, itop)
        
        np.testing.assert_allclose(dtlsc, jnp.asarray([[[0.00000000e+00, 1.59599063e-05, 7.07364228e-05, 1.45072684e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]), atol=1e-4, rtol=0)
        np.testing.assert_allclose(dqlsc, jnp.asarray([[[ 0.00000000e+00, -7.59054545e-04, -3.98269278e-04, -5.82378946e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]), atol=1e-4, rtol=0)
        self.assertAlmostEqual(precls, jnp.asarray([1.293]), delta=0.05)
        self.assertEqual(itop, jnp.asarray([[1]])) # Note this is 2 in the Fortran code, but indexing from 1, so should be 1 in the python

