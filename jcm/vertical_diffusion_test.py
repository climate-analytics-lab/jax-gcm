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
        sigh = jnp.linspace(1.0, 0.0, 9) # Half-level sigma
        fsg = jnp.array([0.025, 0.095, 0.19999999, 0.33999997, 0.51, 0.685, 0.835, 0.95])
        dhs = jnp.array([0.05, 0.09, 0.11999999, 0.16, 0.18000004, 0.16999996, 0.13, 0.10000002])

        import dataclasses
        @dataclasses.dataclass
        class DiffusionConstants:
            shallow_convection_relax_time: float = 6
            moisture_diffusion_relax_time: float = 24
            super_adiabatic_relax_time: float = 6
            shallow_reduction_factor: float = 0.5
            relative_humidity_max_gradient: float = 0.5
            dry_static_energy_min_gradient: float = 0.1 

        utenvd, vtenvd, ttenvd, qtenvd = vertical_diffusion.get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv, DiffusionConstants(), fsg, dhs, sigh)
        np.testing.assert_array_almost_equal(utenvd[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(vtenvd[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(ttenvd[20, 20], np.asarray([-2.4538091e-04, -3.8170350e-05,  4.4986453e-05,  1.0326448e-04, 1.3632278e-04,  1.5194372e-04,  1.4609606e-04,  2.5449507e-04]))
        np.testing.assert_array_almost_equal(qtenvd[20, 20], np.asarray([0.0000000e+00,  0.0000000e+00, -8.6116625e-06,  6.4587462e-06, 0.0000000e+00,  0.0000000e+00, -4.4515664e-06,  5.7870352e-06]))
        np.testing.assert_array_almost_equal(ttenvd[40, 40], np.asarray([-2.4538091e-04, -3.8170350e-05,  4.4986453e-05,  1.0326448e-04, 1.3632278e-04,  1.5194372e-04,  1.4609606e-04,  2.5449507e-04]))
        np.testing.assert_array_almost_equal(qtenvd[40, 40], np.asarray( [ 0.0000000e+00,  0.0000000e+00, -8.6116625e-06,  6.4587462e-06, 0.0000000e+00,  0.0000000e+00, -4.4515664e-06,  5.7870352e-06]))
