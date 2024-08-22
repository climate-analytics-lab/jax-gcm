import unittest
from jcm import vertical_diffusion
import jax.numpy as jnp
import numpy as np

class Test_VerticalDiffusion_Unit(unittest.TestCase):

    def test_get_vertical_diffusion_tend(self):
        se = jnp.ones((96,48))[:,:,jnp.newaxis] * jnp.linspace(400,300,8)[jnp.newaxis, jnp.newaxis, :]
        rh = jnp.ones((96,48))[:,:,jnp.newaxis] * jnp.linspace(0.1,0.9,8)[jnp.newaxis, jnp.newaxis, :]
        qa = jnp.ones((96,48))[:,:,jnp.newaxis] * jnp.array([1, 4, 7.3, 8.8, 12, 18, 24, 26])[jnp.newaxis, jnp.newaxis, :]
        qsat = jnp.ones((96,48))[:,:,jnp.newaxis] * jnp.array([5, 8, 10, 13, 16, 21, 28, 31])[jnp.newaxis, jnp.newaxis, :]
        phi = jnp.ones((96,48))[:,:,jnp.newaxis] * jnp.linspace(150000,0,8)[jnp.newaxis, jnp.newaxis, :]
        icnv = jnp.ones((96,48))*6
        
        utenvd, vtenvd, ttenvd, qtenvd = vertical_diffusion.get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv)
        np.testing.assert_array_almost_equal(utenvd[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(vtenvd[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(ttenvd[20, 20], np.asarray([-2.4538091e-04, -3.8170350e-05,  4.4986453e-05,  1.0326448e-04, 1.3632278e-04,  1.5194372e-04,  1.4609606e-04,  2.5449507e-04]))
        np.testing.assert_array_almost_equal(qtenvd[20, 20], np.asarray([0.0000000e+00,  0.0000000e+00, -8.6116625e-06,  6.4587462e-06, 0.0000000e+00,  0.0000000e+00, -4.4515664e-06,  5.7870352e-06]))
        np.testing.assert_array_almost_equal(ttenvd[40, 40], np.asarray([-2.4538091e-04, -3.8170350e-05,  4.4986453e-05,  1.0326448e-04, 1.3632278e-04,  1.5194372e-04,  1.4609606e-04,  2.5449507e-04]))
        np.testing.assert_array_almost_equal(qtenvd[40, 40], np.asarray( [ 0.0000000e+00,  0.0000000e+00, -8.6116625e-06,  6.4587462e-06, 0.0000000e+00,  0.0000000e+00, -4.4515664e-06,  5.7870352e-06]))
