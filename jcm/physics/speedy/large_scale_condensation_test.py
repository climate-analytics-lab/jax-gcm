import unittest
import jax.numpy as jnp
import numpy as np
import jax

class TestLargeScaleCondensationUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 1, 1, 8

        global ConvectionData, HumidityData, PhysicsData, PhysicsState, PhysicsTendency, parameters, geometry, BoundaryData, get_large_scale_condensation_tendencies, p0, grav
        from jcm.physics.speedy.physics_data import ConvectionData, HumidityData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        parameters = Parameters.default()
        geometry = Geometry.from_grid_shape((ix, il), kx)
        from jcm.boundaries import BoundaryData
        from jcm.physics.speedy.large_scale_condensation import get_large_scale_condensation_tendencies
        from jcm.constants import p0, grav

    def test_get_large_scale_condensation_tendencies(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones(xy)
        qa = jnp.ones(zxy)
        qsat = jnp.ones(zxy)
        itop = jnp.full((ix, il), kx - 1)

        convection = ConvectionData.zeros(xy, kx, iptop=itop)
        humidity = HumidityData.zeros(xy, kx, qsat=qsat)
        state = state = PhysicsState.zeros(zxy, specific_humidity=qa, normalized_surface_pressure=psa)
        physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)
        boundaries = BoundaryData.ones(xy)

        physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data, parameters, boundaries, geometry)
        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertIsNotNone(physics_data.convection.iptop)
        self.assertIsNotNone(physics_data.condensation.precls)
        self.assertIsNotNone(physics_tendencies.temperature)
        self.assertIsNotNone(physics_tendencies.specific_humidity)

    def test_get_large_scale_condensation_tendencies_realistic(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) * 1.0110737
        qa = jnp.asarray([16.148024  , 10.943978  ,  5.851813  ,  2.4522789 ,  0.02198645,
        0.16069981,  0.        ,  0.        ])
        qsat = jnp.asarray([1.64229703e-01, 1.69719307e-02, 1.45193088e-01, 1.98833509e+00,
       4.58917155e+00, 9.24226425e+00, 1.48490220e+01, 2.02474803e+01])
        itop = jnp.ones((ix, il), dtype=int) * 4

        convection = ConvectionData.zeros(xy, kx, iptop=itop)
        humidity = HumidityData.zeros(xy, kx, qsat=qsat[:, jnp.newaxis, jnp.newaxis])
        state = PhysicsState.zeros(zxy, specific_humidity=qa[:, jnp.newaxis, jnp.newaxis],normalized_surface_pressure=psa)
        physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)
        boundaries = BoundaryData.ones(xy)

        physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data, parameters, boundaries, geometry)
        
        np.testing.assert_allclose(physics_tendencies.temperature[1:], jnp.asarray([1.59599063e-05, 7.07364228e-05, 1.45072684e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])[:,jnp.newaxis,jnp.newaxis], atol=1e-4, rtol=0)
        np.testing.assert_allclose(physics_tendencies.specific_humidity[1:], jnp.asarray([-7.59054545e-04, -3.98269278e-04, -5.82378946e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])[:,jnp.newaxis,jnp.newaxis], atol=1e-4, rtol=0)
        self.assertAlmostEqual(
            physics_data.condensation.precls,
            -jnp.sum(physics_tendencies.specific_humidity * geometry.dhs[:, jnp.newaxis, jnp.newaxis] * p0 / grav, axis=0),
            delta=0.05
        )
        self.assertEqual(physics_data.convection.iptop, jnp.asarray([[1]])) # Note this is 2 in the Fortran code, but indexing from 1, so should be 1 in the python

    def test_get_large_scale_condensation_tendencies_gradients_isnan_ones(self):
        """Test that we can calculate gradients of large-scale condensation without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state = PhysicsState.ones(zxy)
        boundaries = BoundaryData.ones(xy)

        # Calculate gradient
        _, f_vjp = jax.vjp(get_large_scale_condensation_tendencies, state, physics_data, parameters, boundaries, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dboundaries, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dboundaries.isnan().any_true())


