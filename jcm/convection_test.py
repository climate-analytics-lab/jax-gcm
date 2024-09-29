import unittest
from jcm.convection import diagnose_convection, get_convection_tendencies
import jax.numpy as jnp
from jcm.physics import PhysicsState
from jcm.physics_data import PhysicsData, ConvectionData, HumidityData

class TestConvectionUnit(unittest.TestCase):
    def test_diagnose_convection_moist_adiabat(self):
        il, ix, kx = 96, 48, 8

        psa = jnp.ones((il, ix)) #normalized surface pressure

        #test using moist adiabatic temperature profile with mid-troposphere dry anomaly
        se = jnp.array([482562.19904568, 404459.50322158, 364997.46113127, 343674.54474717, 328636.42287272, 316973.69544231, 301500., 301500.])
        qa = jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])

        se_broadcast = jnp.tile(se[jnp.newaxis, jnp.newaxis, :], (il, ix, kx))
        qa_broadcast = jnp.tile(qa[jnp.newaxis, jnp.newaxis, :], (il, ix, kx))
        qsat_broadcast = jnp.tile(qsat[jnp.newaxis, jnp.newaxis, :], (il, ix, kx))

        itop, qdif = diagnose_convection(psa, se_broadcast, qa_broadcast * 1000., qsat_broadcast * 1000.)

        test_itop = 4
        test_qdif = 1.1395
        # Check that itop and qdif is not null.
        self.assertEqual(itop[0,0], test_itop)
        self.assertAlmostEqual(qdif[0,0],test_qdif,places=4)
     
    def test_get_convective_tendencies_moist_adiabat(self):
        il, ix, kx = 96, 48, 8

        psa = jnp.ones((il, ix)) #normalized surface pressure

        #test using moist adiabatic temperature profile with mid-troposphere dry anomaly
        se = jnp.array([482562.19904568, 404459.50322158, 364997.46113127, 343674.54474717, 328636.42287272, 316973.69544231, 301500., 301500.])
        qa = jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])

        se_broadcast = jnp.tile(se[jnp.newaxis, jnp.newaxis, :], (il, ix, 1))
        qa_broadcast = jnp.tile(qa[jnp.newaxis, jnp.newaxis, :], (il, ix, 1))
        qsat_broadcast = jnp.tile(qsat[jnp.newaxis, jnp.newaxis, :], (il, ix, 1))

        convection = ConvectionData((ix, il), kx, psa=psa, se=se_broadcast)
        humidity = HumidityData((ix, il), kx, qsat=qsat_broadcast*1000.)
        state = PhysicsState(u_wind=jnp.zeros_like(qa),
                             v_wind=jnp.zeros_like(qa),
                             temperature=jnp.zeros_like(qa),
                             specific_humidity=qa_broadcast*1000.,
                             geopotential=jnp.zeros_like(qa),
                             surface_pressure=jnp.zeros((ix, il)))
        physics_data = PhysicsData((ix, il), kx, humidity=humidity, convection=convection)

        _, physics_data = get_convection_tendencies(physics_data,state)

        test_cbmf = jnp.array(0.019614903)
        test_precnv = jnp.array(0.21752352)
        test_dfse = jnp.array([  0., 0., 0., 0. ,-29.774475, 402.0166, 171.78418, 0.])
        test_dfqa = jnp.array([ 0., 0., 0., 0.01235308,  0.07379276, -0.15330768, -0.08423203, -0.05377656])

        # Check that itop and qdif is not null.
        self.assertAlmostEqual(physics_data.convection.cbmf[0,0], test_cbmf, places=4)
        self.assertAlmostEqual(physics_data.convection.precnv[0,0], test_precnv, places=4)
        #check a few values of the fluxes
        self.assertAlmostEqual(physics_data.convection.dfse[0,0,4], test_dfse[4], places=2)
        self.assertAlmostEqual(physics_data.convection.dfqa[0,0,4], test_dfqa[4], places=2) 
        self.assertAlmostEqual(physics_data.convection.dfse[0,0,5], test_dfse[5], places=2)
        self.assertAlmostEqual(physics_data.convection.dfqa[0,0,5], test_dfqa[5], places=2) 
        self.assertAlmostEqual(physics_data.convection.dfse[0,0,6], test_dfse[6], places=2)
        self.assertAlmostEqual(physics_data.convection.dfqa[0,0,6], test_dfqa[6], places=2) 
      
