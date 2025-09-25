import unittest
import jax.numpy as jnp
import numpy as np
import jax
import functools

class TestLargeScaleCondensationUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 1, 1, 8

        global ConvectionData, HumidityData, PhysicsData, PhysicsState, PhysicsTendency, parameters, geometry, BoundaryData, get_large_scale_condensation_tendencies
        from jcm.physics.speedy.physics_data import ConvectionData, HumidityData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        parameters = Parameters.default()
        geometry = Geometry.from_grid_shape((ix, il), kx)
        from jcm.boundaries import BoundaryData
        from jcm.physics.speedy.large_scale_condensation import get_large_scale_condensation_tendencies

    # def test_get_large_scale_condensation_tendencies(self):
    #     xy = (ix,il)
    #     zxy = (kx,ix,il)
    #     psa = jnp.ones(xy)
    #     qa = jnp.ones(zxy)
    #     qsat = jnp.ones(zxy)
    #     itop = jnp.full((ix, il), kx - 1)

    #     convection = ConvectionData.zeros(xy, kx, iptop=itop)
    #     humidity = HumidityData.zeros(xy, kx, qsat=qsat)
    #     state = state = PhysicsState.zeros(zxy, specific_humidity=qa, normalized_surface_pressure=psa)
    #     physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)
    #     boundaries = BoundaryData.ones(xy)

    #     physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data, parameters, boundaries, geometry)
    #     # Check that itop, precls, dtlsc, and dqlsc are not null.
    #     self.assertIsNotNone(physics_data.convection.iptop)
    #     self.assertIsNotNone(physics_data.condensation.precls)
    #     self.assertIsNotNone(physics_tendencies.temperature)
    #     self.assertIsNotNone(physics_tendencies.specific_humidity)

    # def test_get_large_scale_condensation_tendencies_realistic(self):
    #     xy = (ix,il)
    #     zxy = (kx,ix,il)
    #     psa = jnp.ones((ix, il)) * 1.0110737
    #     qa = jnp.asarray([16.148024  , 10.943978  ,  5.851813  ,  2.4522789 ,  0.02198645,
    #     0.16069981,  0.        ,  0.        ])
    #     qsat = jnp.asarray([1.64229703e-01, 1.69719307e-02, 1.45193088e-01, 1.98833509e+00,
    #    4.58917155e+00, 9.24226425e+00, 1.48490220e+01, 2.02474803e+01])
    #     itop = jnp.ones((ix, il), dtype=int) * 4

    #     convection = ConvectionData.zeros(xy, kx, iptop=itop)
    #     humidity = HumidityData.zeros(xy, kx, qsat=qsat[:, jnp.newaxis, jnp.newaxis])
    #     state = PhysicsState.zeros(zxy, specific_humidity=qa[:, jnp.newaxis, jnp.newaxis],normalized_surface_pressure=psa)
    #     physics_data = PhysicsData.zeros(xy, kx, humidity=humidity, convection=convection)
    #     boundaries = BoundaryData.ones(xy)

    #     physics_tendencies, physics_data = get_large_scale_condensation_tendencies(state, physics_data, parameters, boundaries, geometry)
        
    #     np.testing.assert_allclose(physics_tendencies.temperature, jnp.asarray([0.00000000e+00, 1.59599063e-05, 7.07364228e-05, 1.45072684e-04,
    #    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])[:,jnp.newaxis,jnp.newaxis], atol=1e-4, rtol=0)
    #     np.testing.assert_allclose(physics_tendencies.specific_humidity, jnp.asarray([ 0.00000000e+00, -7.59054545e-04, -3.98269278e-04, -5.82378946e-05,
    #     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])[:,jnp.newaxis,jnp.newaxis], atol=1e-4, rtol=0)
    #     self.assertAlmostEqual(physics_data.condensation.precls, jnp.asarray([1.293]), delta=0.05)
    #     self.assertEqual(physics_data.convection.iptop, jnp.asarray([[1]])) # Note this is 2 in the Fortran code, but indexing from 1, so should be 1 in the python

    # def test_get_large_scale_condensation_tendencies_gradients_isnan_ones(self):
    #     """Test that we can calculate gradients of large-scale condensation without getting NaN values"""
    #     xy = (ix, il)
    #     zxy = (kx, ix, il)
    #     physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
    #     state = PhysicsState.ones(zxy)
    #     boundaries = BoundaryData.ones(xy)

    #     # Calculate gradient
    #     _, f_vjp = jax.vjp(get_large_scale_condensation_tendencies, state, physics_data, parameters, boundaries, geometry)
    #     tends = PhysicsTendency.ones(zxy)
    #     datas = PhysicsData.ones(xy,kx)
    #     input = (tends, datas)
    #     df_dstates, df_ddatas, df_dparams, df_dboundaries, df_dgeometry = f_vjp(input)

    #     self.assertFalse(df_ddatas.isnan().any_true())
    #     self.assertFalse(df_dstates.isnan().any_true())
    #     self.assertFalse(df_dparams.isnan().any_true())
    #     self.assertFalse(df_dboundaries.isnan().any_true())

    # Gradient test
    def test_get_large_scale_condensation_tendencies_realistic_gradient_check(self):
        from jax.test_util import check_vjp, check_jvp
        from jax.tree_util import tree_map
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

        # Converting functions
        def check_type_convert_to_float(x): # Do error catch block
            try:
                return x.astype(jnp.float32)
            except AttributeError:
                return jnp.float32(x)
        def convert_to_float(x): 
            return tree_map(check_type_convert_to_float, x)
        def check_type_convert_back(x, x0):
            try: 
                if x0.dtype == jnp.float32:
                    return x
                else:
                    return x0
            except AttributeError:
                if type(x0) == jnp.float32:
                    return x
                else:
                    return x0
        def convert_back(x, x0):
            return tree_map(check_type_convert_back, x, x0)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, boundaries_f,geometry_f):
            tend_out, data_out = get_large_scale_condensation_tendencies(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(tend_out), convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        # check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, boundaries_floats, geometry_floats), 
        #                         atol=None, rtol=1, eps=0.000001)





