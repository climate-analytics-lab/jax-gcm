import unittest
from jcm.shortwave_radiation_clouds import clouds
import jax.numpy as jnp

class TestClouds(unittest.TestCase):

    def test_clouds_general(self):
        
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

    def test_clouds_case1(self):

        a = 5
        b = 0.25
        c = -1
        d = 4
        e = 3
        f = 0.01
        g = 3400
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 3
        cloudc_true = 0.6324555414579978
        clstr_true = 127.5000050663948

        icltop, cloudc, clstr = clouds(qa,rh,precnv,precls,iptop,gse,fmask,icltop,cloudc,clstr)
        
        # Check that icltop, cloudc, and clstr are not null.
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

def test_clouds_case2(self):

        a = 512983
        b = 0.25234
        c = -3298429847
        d = 2.134
        e = 3
        f = 0.010134985739
        g = -10000000000
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 3
        cloudc_true = 1.0
        clstr_true = -378510015.04063606

        icltop, cloudc, clstr = clouds(qa,rh,precnv,precls,iptop,gse,fmask,icltop,cloudc,clstr)
        
        # Check that icltop, cloudc, and clstr are not null.
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

def test_clouds_case3(self):

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 0
        cloudc_true = 0.0
        clstr_true = 0.0

        icltop, cloudc, clstr = clouds(qa,rh,precnv,precls,iptop,gse,fmask,icltop,cloudc,clstr)
        
        # Check that icltop, cloudc, and clstr are not null.
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

def test_clouds_case4(self):

        a = -1
        b = -1
        c = -1
        d = -1
        e = -1
        f = -1
        g = -1
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = -1
        cloudc_true = 1.0
        clstr_true = 0.15000000596046448

        icltop, cloudc, clstr = clouds(qa,rh,precnv,precls,iptop,gse,fmask,icltop,cloudc,clstr)
        
        # Check that icltop, cloudc, and clstr are not null.
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)