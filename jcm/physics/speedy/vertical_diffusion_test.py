import jax
import unittest
import jax.numpy as jnp
import numpy as np
import functools
from jax.test_util import check_vjp, check_jvp

class Test_VerticalDiffusion_Unit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 1, 1, 8

        global HumidityData, ConvectionData, PhysicsData, PhysicsState, PhysicsTendency, get_vertical_diffusion_tend, \
            parameters, speedy_coords, terrain, ForcingData
        from jcm.physics.speedy.physics_data import HumidityData, ConvectionData, PhysicsData
        from jcm.physics.speedy.params import Parameters
        from jcm.forcing import ForcingData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.vertical_diffusion import get_vertical_diffusion_tend
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords

        speedy_coords = SpeedyCoords.single_column_coords(num_levels=kx)
        parameters = Parameters.default()
        terrain = TerrainData.single_column()

    def test_get_vertical_diffusion_tend(self):
        se = jnp.ones((ix,il)) * jnp.linspace(400,300,kx)[:, jnp.newaxis, jnp.newaxis]
        rh = jnp.ones((ix,il)) * jnp.linspace(0.1,0.9,kx)[:, jnp.newaxis, jnp.newaxis]
        qa = jnp.ones((ix,il)) * jnp.array([1, 4, 7.3, 8.8, 12, 18, 24, 26])[:, jnp.newaxis, jnp.newaxis]
        qsat = jnp.ones((ix,il)) * jnp.array([5, 8, 10, 13, 16, 21, 28, 31])[:, jnp.newaxis, jnp.newaxis]
        phi = jnp.ones((ix,il)) * jnp.linspace(150000,0,kx)[:, jnp.newaxis, jnp.newaxis]
        iptop = jnp.ones((ix,il), dtype=int)*1
        
        zxy = (kx, ix, il)
        xy = (ix, il)
        humidity_data = HumidityData.zeros((ix,il), kx, rh=rh, qsat=qsat)
        convection_data = ConvectionData.zeros((ix,il), kx, iptop=iptop, se=se)
        physics_data = PhysicsData.zeros((ix,il), kx, humidity=humidity_data, convection=convection_data, speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=phi)
        forcing = ForcingData.ones(xy)
        
        # utenvd, vtenvd, ttenvd, qtenvd = get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv)
        physics_tendencies, _ = get_vertical_diffusion_tend(state, physics_data, parameters, forcing, terrain)

        utenvd, vtenvd, ttenvd, qtenvd = physics_tendencies.u_wind, physics_tendencies.v_wind, physics_tendencies.temperature, physics_tendencies.specific_humidity

        self.assertTrue(np.allclose(utenvd, np.zeros_like(utenvd), atol=1e-9))
        self.assertTrue(np.allclose(vtenvd, np.zeros_like(vtenvd), atol=1e-9))
        self.assertTrue(np.allclose(ttenvd[:,0,0], np.array([ 2.78098357e-04,  1.39862334e-04,  8.50690617e-05,  3.73100450e-05,
        3.67983799e-06, -2.65383318e-05, -6.18272365e-05, -3.07837296e-04]), atol=1e-9))
        self.assertTrue(np.allclose(qtenvd[:,0,0], np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.99411916e-06,  7.24206425e-06,  1.30163815e-05, -4.72222083e-05]), atol=1e-9))

    def test_get_vertical_diffusion_gradients_isnan_ones(self):
        """Test that we can calculate gradients of vertical diffusion without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)

        # Calculate gradient
        primals, f_vjp = jax.vjp(get_vertical_diffusion_tend, state, physics_data, parameters, forcing, terrain)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
        input = (tends, datas)
        df_dstate, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_get_vertical_diffusion_gradient_check(self):
        """Test that we get correct gradient values"""
        from jcm.utils import convert_back, convert_to_float
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx, speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_vertical_diffusion_tend(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(tend_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.000001)

        
