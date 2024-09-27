import unittest
from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
from jcm.physics_data import PhysicsData, CondensationData, HumidityData, ConvectionData
from jcm.physics import PhysicsState, PhysicsTendency
import jax.numpy as jnp
import numpy as np

class TestLargeScaleCondensationUnit(unittest.TestCase):

    def test_get_large_scale_condensation_tendencies(self):
        ix, il, kx = 1, 1, 8
        xy = (ix,il)
        psa = jnp.ones((ix, il))
        qa = jnp.ones((ix, il, kx))
        qsat = jnp.ones((ix, il, kx))
        itop = jnp.full((ix, il), kx - 1)

        convection = ConvectionData(xy, kx, psa=psa,iptop=itop)
        humidity = HumidityData(xy, kx, qsat=qsat)
        state = PhysicsState(u_wind=jnp.zeros_like(qa),
                             v_wind=jnp.zeros_like(qa),
                             temperature=jnp.zeros_like(qa),
                             specific_humidity=qa,
                             geopotential=jnp.zeros_like(qa),
                             surface_pressure=jnp.zeros((ix, il)))
        physics_data = PhysicsData(xy, kx, humidity=humidity, convection=convection)

        _, physics_data = get_large_scale_condensation_tendencies(physics_data, state)
        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertIsNotNone(physics_data.convection.iptop)
        self.assertIsNotNone(physics_data.condensation.precls)
        self.assertIsNotNone(physics_data.condensation.dtlsc)
        self.assertIsNotNone(physics_data.condensation.dqlsc)

    def test_get_large_scale_condensation_tendencies_realistic(self):
        ix, il, kx = 1, 1, 8
        xy = (ix,il)
        psa = jnp.ones((ix, il)) * 1.0110737
        qa = jnp.asarray([[[16.148024  , 10.943978  ,  5.851813  ,  2.4522789 ,  0.02198645,
        0.16069981,  0.        ,  0.        ]]])
        qsat = jnp.asarray([[[1.64229703e-01, 1.69719307e-02, 1.45193088e-01, 1.98833509e+00,
       4.58917155e+00, 9.24226425e+00, 1.48490220e+01, 2.02474803e+01]]])
        itop = jnp.ones((ix, il)) * 4

        convection = ConvectionData(xy, kx, psa=psa,iptop=itop)
        humidity = HumidityData(xy, kx, qsat=qsat)
        state = PhysicsState(u_wind=jnp.zeros_like(qa),
                             v_wind=jnp.zeros_like(qa),
                             temperature=jnp.zeros_like(qa),
                             specific_humidity=qa,
                             geopotential=jnp.zeros_like(qa),
                             surface_pressure=jnp.zeros((ix, il)))
        physics_data = PhysicsData(xy, kx, humidity=humidity, convection=convection)

        _, physics_data = get_large_scale_condensation_tendencies(physics_data, state)
        
        np.testing.assert_allclose(physics_data.condensation.dtlsc, jnp.asarray([[[0.00000000e+00, 1.59599063e-05, 7.07364228e-05, 1.45072684e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]), atol=1e-4, rtol=0)
        np.testing.assert_allclose(physics_data.condensation.dqlsc, jnp.asarray([[[ 0.00000000e+00, -7.59054545e-04, -3.98269278e-04, -5.82378946e-05,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]), atol=1e-4, rtol=0)
        self.assertAlmostEqual(physics_data.condensation.precls, jnp.asarray([1.293]), delta=0.05)
        self.assertEqual(physics_data.convection.iptop, jnp.asarray([[1]])) # Note this is 2 in the Fortran code, but indexing from 1, so should be 1 in the python

