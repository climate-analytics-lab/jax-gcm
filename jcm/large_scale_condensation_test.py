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

        global ConvectionData, HumidityData, PhysicsData, PhysicsState, PhysicsTendency, ConvectionParameters, Parameters, BoundaryData, get_large_scale_condensation_tendencies
        from jcm.physics_data import ConvectionData, HumidityData, PhysicsData
        from jcm.physics import PhysicsState, PhysicsTendency
        from jcm.params import Parameters, ConvectionParameters
        from jcm.boundaries import BoundaryData
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
        params = Parameters(
            convection=ConvectionParameters()
        )
        boundaries = BoundaryData.ones(xy)

        physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data, params, boundaries)
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
        itop = jnp.ones((ix, il), dtype=int) * 4

        convection = ConvectionData.zeros(xy, kx, psa=psa,iptop=itop)
        humidity = HumidityData.zeros(xy, kx, qsat=qsat)
        state = PhysicsState.zeros(xyz, specific_humidity=qa)
        physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)
        params = Parameters(
            convection=ConvectionParameters()
        )
        boundaries = BoundaryData.ones(xy)

        physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data, params, boundaries)
        
        np.testing.assert_allclose(physics_tendencies.temperature, jnp.asarray([[[0.00000000e+00, 1.59599063e-05, 7.07364228e-05, 1.45072684e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]), atol=1e-4, rtol=0)
        np.testing.assert_allclose(physics_tendencies.specific_humidity, jnp.asarray([[[ 0.00000000e+00, -7.59054545e-04, -3.98269278e-04, -5.82378946e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]), atol=1e-4, rtol=0)
        self.assertAlmostEqual(physics_data.condensation.precls, jnp.asarray([1.293]), delta=0.05)
        self.assertEqual(physics_data.convection.iptop, jnp.asarray([[1]])) # Note this is 2 in the Fortran code, but indexing from 1, so should be 1 in the python

    def test_get_large_scale_condensation_tendencies_gradients_isnan_ones(self):    
        """Test that we can calculate gradients of large-scale condensation without getting NaN values"""
        xy = (ix, il)
        xyz = (ix, il, kx)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(xyz)
        params = Parameters(
            convection=ConvectionParameters()
        )
        boundaries = BoundaryData.ones(xy)

        # Calculate gradient
        _, f_vjp = jax.vjp(get_large_scale_condensation_tendencies, state, physics_data, params, boundaries) 
        tends = PhysicsTendency.ones(xyz)
        datas = PhysicsData.ones(xy,kx) 
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dboundaries = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dboundaries.isnan())


