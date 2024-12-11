import jax
import unittest
import jax.numpy as jnp
import numpy as np
class Test_VerticalDiffusion_Unit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)

        global HumidityData, ConvectionData, PhysicsData, PhysicsState, PhysicsTendency, get_vertical_diffusion_tend
        from jcm.physics_data import HumidityData, ConvectionData, PhysicsData
        from jcm.physics import PhysicsState, PhysicsTendency
        from jcm.vertical_diffusion import get_vertical_diffusion_tend

    def test_get_vertical_diffusion_tend(self):
        se = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(400,300,kx)[jnp.newaxis, jnp.newaxis, :]
        rh = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(0.1,0.9,kx)[jnp.newaxis, jnp.newaxis, :]
        qa = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.array([1, 4, 7.3, 8.8, 12, 18, 24, 26])[jnp.newaxis, jnp.newaxis, :]
        qsat = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.array([5, 8, 10, 13, 16, 21, 28, 31])[jnp.newaxis, jnp.newaxis, :]
        phi = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(150000,0,kx)[jnp.newaxis, jnp.newaxis, :]
        iptop = jnp.ones((ix,il))*1
        
        xyz = (ix, il, kx)
        humidity_data = HumidityData.zeros((ix,il), kx, rh=rh, qsat=qsat)
        convection_data = ConvectionData.zeros((ix,il), kx, iptop=iptop, se=se)
        physics_data = PhysicsData.zeros((ix,il), kx, humidity=humidity_data, convection=convection_data)
        state = PhysicsState.zeros(xyz, specific_humidity=qa, geopotential=phi)
        
        # utenvd, vtenvd, ttenvd, qtenvd = get_vertical_diffusion_tend(se, rh, qa, qsat, phi, icnv)
        physics_tendencies, _ = get_vertical_diffusion_tend(state, physics_data)

        utenvd, vtenvd, ttenvd, qtenvd = physics_tendencies.u_wind, physics_tendencies.v_wind, physics_tendencies.temperature, physics_tendencies.specific_humidity

        self.assertTrue(np.allclose(utenvd, np.zeros_like(utenvd), atol=1e-4))
        self.assertTrue(np.allclose(vtenvd, np.zeros_like(vtenvd), atol=1e-4))
        self.assertTrue(np.allclose(ttenvd[0,0,:], np.array([ 2.78098357e-04,  1.39862334e-04,  8.50690617e-05,  3.73100450e-05,
        3.67983799e-06, -2.65383318e-05, -6.18272365e-05, -3.07837296e-04]), atol=1e-4))
        self.assertTrue(np.allclose(qtenvd[0,0,:], np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.99411916e-06,  7.24206425e-06,  1.30163815e-05, -4.72222083e-05]), atol=1e-4))

    def test_get_vertical_diffusion_gradients(self): 
        """Test that we can calculate gradients of vertical diffusion without getting NaN values"""
        xy = (ix, il)
        xyz = (ix, il, kx)
        physics_data = PhysicsData.zeros(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.zeros(xyz)

        # Calculate gradient
        primals, f_vjp = jax.vjp(get_vertical_diffusion_tend, state, physics_data) 
        tends = PhysicsTendency.ones(xyz)
        datas = PhysicsData.ones(xy,kx) 
        input = (tends, datas)
        df_dstates, df_ddatas = f_vjp(input)

        # Checking if the function with respect to the input states is nan
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.surface_pressure)))

        # Checking if the function with respect to the input physics data is nan
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.rlds)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.dfabs)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.ftop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.slr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.qcloud)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.fsol)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.rsds)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ssr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ozone)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ozupp)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.zenit)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.stratz)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.gse)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.icltop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.cloudc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.cloudstr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ftop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.dfabs)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.psa)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.se)))
        # self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.iptop))) doesn't work bc current type is int
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.cbmf)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.precnv)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.alb_l)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.alb_s)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.albsfc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.snowc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.tau2)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.st4a)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.stratc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.flux)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.humidity.rh)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.humidity.qsat)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.precls)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.dtlsc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.dqlsc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.stl_am)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.soilw_am)))
        # Not testing df_ddatas.surface_flux.lfluxland because it is a bool type
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.ustr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.vstr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.shf)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.evap)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.slru)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.hfluxn)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.tsfc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.tskin)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.u0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.v0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.t0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.fmask)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.phi0)))
        # No testing df_ddatas.date
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.sea_model.tsea)))

        # def test_vertical_diffusion_gradients(self):
        #     """Test that we can calculate gradients of vertical diffusion without getting NaN values"""
        #     # Use exactly the same test inputs as test_get_vertical_diffusion_tend
        #     se = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(400,300,kx)[jnp.newaxis, jnp.newaxis, :]
        #     rh = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(0.1,0.9,kx)[jnp.newaxis, jnp.newaxis, :]
        #     qa = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.array([1, 4, 7.3, 8.8, 12, 18, 24, 26])[jnp.newaxis, jnp.newaxis, :]
        #     qsat = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.array([5, 8, 10, 13, 16, 21, 28, 31])[jnp.newaxis, jnp.newaxis, :]
        #     phi = jnp.ones((ix,il))[:,:,jnp.newaxis] * jnp.linspace(150000,0,kx)[jnp.newaxis, jnp.newaxis, :]
        #     iptop = jnp.ones((ix,il))*1
            
        #     xyz = (ix, il, kx)
        #     humidity_data = HumidityData.zeros((ix,il), kx, rh=rh, qsat=qsat)
        #     convection_data = ConvectionData.zeros((ix,il), kx, iptop=iptop, se=se)
        #     physics_data = PhysicsData.zeros((ix,il), kx, humidity=humidity_data, convection=convection_data)
        #     state = PhysicsState.zeros(xyz, specific_humidity=qa, geopotential=phi)

        #     # Function to get vertical diffusion outputs for gradient testing
        #     def get_diffusion_outputs(state, physics_data):
        #         tendencies, _ = get_vertical_diffusion_tend(state, physics_data)
        #         # Return concatenated tendencies
        #         return jnp.concatenate([
        #             tendencies.temperature.ravel(),
        #             tendencies.specific_humidity.ravel()
        #         ])

        #     # Calculate gradients
        #     primals, grad_fn = jax.vjp(get_diffusion_outputs, state, physics_data)
            
        #     # Input for gradient calculation
        #     grad_input = jnp.ones_like(primals)
            
        #     # Get gradients
        #     (state_grads, physics_grads) = grad_fn(grad_input)

        #     # Verify no NaN gradients in state variables
        #     self.assertFalse(jnp.any(jnp.isnan(state_grads.specific_humidity)))
        #     self.assertFalse(jnp.any(jnp.isnan(state_grads.geopotential)))
            
        #     # Verify no NaN gradients in physics data
        #     self.assertFalse(jnp.any(jnp.isnan(physics_grads.convection.se)))
        #     self.assertFalse(jnp.any(jnp.isnan(physics_grads.humidity.rh)))
        #     self.assertFalse(jnp.any(jnp.isnan(physics_grads.humidity.qsat)))