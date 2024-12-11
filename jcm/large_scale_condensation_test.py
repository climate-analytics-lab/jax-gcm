import unittest
import jax.numpy as jnp
import numpy as np
import jax

class TestLargeScaleCondensationUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 1, 1, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)

        global ConvectionData, HumidityData, PhysicsData, PhysicsState, PhysicsTendency, get_large_scale_condensation_tendencies
        from jcm.physics_data import ConvectionData, HumidityData, PhysicsData
        from jcm.physics import PhysicsState, PhysicsTendency
        from jcm.large_scale_condensation import get_large_scale_condensation_tendencies

    def test_get_large_scale_condensation_tendencies(self):
        xy = (ix,il)
        xyz = (ix,il,kx)
        psa = jnp.ones((ix, il))
        qa = jnp.ones((ix, il, kx))
        qsat = jnp.ones((ix, il, kx))
        itop = jnp.full((ix, il), kx - 1)

        convection = ConvectionData.zeros(xy, kx, psa=psa,iptop=itop)
        humidity = HumidityData.zeros(xy, kx, qsat=qsat)
        state = state = PhysicsState.zeros(xyz, specific_humidity=qa)
        physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)

        physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data)
        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertIsNotNone(physics_data.convection.iptop)
        self.assertIsNotNone(physics_data.condensation.precls)
        self.assertIsNotNone(physics_tendencies.temperature)
        self.assertIsNotNone(physics_tendencies.specific_humidity)

    def test_get_large_scale_condensation_tendencies_realistic(self):
        xy = (ix,il)
        xyz = (ix,il,kx)
        psa = jnp.ones((ix, il)) * 1.0110737
        qa = jnp.asarray([[[16.148024  , 10.943978  ,  5.851813  ,  2.4522789 ,  0.02198645,
        0.16069981,  0.        ,  0.        ]]])
        qsat = jnp.asarray([[[1.64229703e-01, 1.69719307e-02, 1.45193088e-01, 1.98833509e+00,
       4.58917155e+00, 9.24226425e+00, 1.48490220e+01, 2.02474803e+01]]])
        itop = jnp.ones((ix, il)) * 4

        convection = ConvectionData.zeros(xy, kx, psa=psa,iptop=itop)
        humidity = HumidityData.zeros(xy, kx, qsat=qsat)
        state = PhysicsState.zeros(xyz, specific_humidity=qa)
        physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)

        physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data)
        
        np.testing.assert_allclose(physics_tendencies.temperature, jnp.asarray([[[0.00000000e+00, 1.59599063e-05, 7.07364228e-05, 1.45072684e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]), atol=1e-4, rtol=0)
        np.testing.assert_allclose(physics_tendencies.specific_humidity, jnp.asarray([[[ 0.00000000e+00, -7.59054545e-04, -3.98269278e-04, -5.82378946e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]), atol=1e-4, rtol=0)
        self.assertAlmostEqual(physics_data.condensation.precls, jnp.asarray([1.293]), delta=0.05)
        self.assertEqual(physics_data.convection.iptop, jnp.asarray([[1]])) # Note this is 2 in the Fortran code, but indexing from 1, so should be 1 in the python

    def test_get_large_scale_condensation_tendencies_gradients(self):    
        """Test that we can calculate gradients of large-scale condensation without getting NaN values"""
        xy = (ix, il)
        xyz = (ix, il, kx)
        physics_data = PhysicsData.zeros(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.zeros(xyz)

        # Calculate gradient
        primals, f_vjp = jax.vjp(get_large_scale_condensation_tendencies, state, physics_data) 
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

