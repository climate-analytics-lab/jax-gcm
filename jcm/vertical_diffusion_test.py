import unittest
from jcm.vertical_diffusion import get_vertical_diffusion_tend
import jax.numpy as jnp
import numpy as np
from jcm.physics_data import PhysicsData, HumidityData, ConvectionData
from jcm.physics import PhysicsState

class Test_VerticalDiffusion_Unit(unittest.TestCase):

    def test_get_vertical_diffusion_tend(self):
        ix, il, kx = 96, 48, 8
        se = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(400,300,kx)[jnp.newaxis, jnp.newaxis, :]
        rh = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(0.1,0.9,kx)[jnp.newaxis, jnp.newaxis, :]
        qa = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.array([1, 4, 7.3, 8.8, 12, 18, 24, 26])[jnp.newaxis, jnp.newaxis, :]
        qsat = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.array([5, 8, 10, 13, 16, 21, 28, 31])[jnp.newaxis, jnp.newaxis, :]
        phi = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(150000,0,kx)[jnp.newaxis, jnp.newaxis, :]
        iptop = jnp.ones((ix,il))*1
        
        humidity_data = HumidityData((ix,il), kx, rh=rh, qsat=qsat)
        convection_data = ConvectionData((ix,il), kx, iptop=iptop, se=se)
        physics_data = PhysicsData((ix,il), kx, humidity=humidity_data, convection=convection_data)
        state = PhysicsState(u_wind=jnp.zeros_like(qa),
                             v_wind=jnp.zeros_like(qa),
                             temperature=jnp.zeros_like(qa),
                             specific_humidity=qa,
                             geopotential=phi,
                             surface_pressure=jnp.zeros((ix, il)))
        
        # utenvd, vtenvd, ttenvd, qtenvd = get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv)
        physics_tendencies, _ = get_vertical_diffusion_tend(state, physics_data)

        np.testing.assert_array_almost_equal(physics_tendencies.u_wind[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(physics_tendencies.v_wind[20, 20], np.asarray([0., 0., 0., 0., 0., 0., 0., 0.]))
        np.testing.assert_array_almost_equal(physics_tendencies.temperature[20, 20], np.asarray([-2.4538091e-04, -3.8170350e-05,  4.4986453e-05,  1.0326448e-04, 1.3632278e-04,  1.5194372e-04,  1.4609606e-04,  2.5449507e-04]))
        np.testing.assert_array_almost_equal(physics_tendencies.specific_humidity[20, 20], np.asarray([0.0000000e+00,  0.0000000e+00, -8.6116625e-06,  6.4587462e-06, 0.0000000e+00,  0.0000000e+00, -4.4515664e-06,  5.7870352e-06]))
        np.testing.assert_array_almost_equal(physics_tendencies.temperature[40, 40], np.asarray([-2.4538091e-04, -3.8170350e-05,  4.4986453e-05,  1.0326448e-04, 1.3632278e-04,  1.5194372e-04,  1.4609606e-04,  2.5449507e-04]))
        np.testing.assert_array_almost_equal(physics_tendencies.specific_humidity[40, 40], np.asarray( [ 0.0000000e+00,  0.0000000e+00, -8.6116625e-06,  6.4587462e-06, 0.0000000e+00,  0.0000000e+00, -4.4515664e-06,  5.7870352e-06]))
