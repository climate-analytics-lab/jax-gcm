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

        utenvd, vtenvd, ttenvd, qtenvd = vertical_diffusion.get_tend_comparison(se, rh, qa, qsat, phi, icnv, DiffusionConstants(), fsg, dhs, sigh)
        np.testing.assert_array_almost_equal(utenvd[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(vtenvd[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(ttenvd[20, 20], np.asarray([2.78098343e-04,  1.39862327e-04,  8.50690984e-05,  3.73100365e-05, 3.67983118e-06, -2.65383233e-05, -6.18272700e-05, -3.07837272e-04]))
        ##  ttenvd at the the second sigma level gives  inf, which might be due to the Step 3: Damping of super-adiabatic lapse rate
        np.testing.assert_array_almost_equal(qtenvd[20, 20], np.asarray([0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.99411857e-06,  7.24205452e-06,  1.30163929e-05, -4.72222055e-05]))

