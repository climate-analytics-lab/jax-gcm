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

        utenvd, vtenvd, ttenvd, qtenvd = physics_tendencies.u_wind, physics_tendencies.v_wind, physics_tendencies.temperature, physics_tendencies.specific_humidity

        self.assertTrue(np.allclose(utenvd, np.zeros_like(utenvd), atol=1e-4))
        self.assertTrue(np.allclose(vtenvd, np.zeros_like(vtenvd), atol=1e-4))
        self.assertTrue(np.allclose(ttenvd[0,0,:], np.array([ 2.78098357e-04,  1.39862334e-04,  8.50690617e-05,  3.73100450e-05,
        3.67983799e-06, -2.65383318e-05, -6.18272365e-05, -3.07837296e-04]), atol=1e-4))
        self.assertTrue(np.allclose(qtenvd[0,0,:], np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.99411916e-06,  7.24206425e-06,  1.30163815e-05, -4.72222083e-05]), atol=1e-4))