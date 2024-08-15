import unittest
from jcm.shortwave_radiation_clouds import clouds
import jax.numpy as jnp

class TestClouds(unittest.TestCase):

    def test_clouds(self):

        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))
        rh = jnp.ones((ix,il,kx))
        precnv = jnp.ones((ix,il))
        precls = jnp.ones((ix,il))
        iptop = jnp.ones((ix,il))
        gse = jnp.ones((ix,il))
        fmask = jnp.ones((ix,il))

        icltop, cloudc, clstr = clouds(qa,rh,precnv,precls,iptop,gse,fmask,icltop,cloudc,clstr)
        
        # Check that icltop, cloudc, and clstr are not null.
        self.assertIsNotNone(icltop)
        self.assertIsNotNone(cloudc)
        self.assertIsNotNone(clstr)

        # Check that our outputs are the right shape
        self.assertEqual(icltop.shape,precnv.shape)
        self.assertEqual(cloudc.shape,precnv.shape)
        self.assertEqual(clstr.shape,precnv.shape)
