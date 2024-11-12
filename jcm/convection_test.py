import unittest
import jax.numpy as jnp

class TestConvectionUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)
        
        global ConvectionData, HumidityData, PhysicsData, PhysicsState, diagnose_convection, get_convection_tendencies, grdscp, grdsig
        from jcm.physics_data import ConvectionData, HumidityData, PhysicsData
        from jcm.physics import PhysicsState
        from jcm.convection import diagnose_convection, get_convection_tendencies
        from jcm.physical_constants import grdscp, grdsig

    def test_diagnose_convection_moist_adiabat(self):
        psa = jnp.ones((ix, il)) #normalized surface pressure

        #test using moist adiabatic temperature profile with mid-troposphere dry anomaly
        se = jnp.array([482562.19904568, 404459.50322158, 364997.46113127, 343674.54474717, 328636.42287272, 316973.69544231, 301500., 301500.])
        qa = jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])

        se_broadcast = jnp.tile(se[jnp.newaxis, jnp.newaxis, :], (ix, il, kx))
        qa_broadcast = jnp.tile(qa[jnp.newaxis, jnp.newaxis, :], (ix, il, kx))
        qsat_broadcast = jnp.tile(qsat[jnp.newaxis, jnp.newaxis, :], (ix, il, kx))

        itop, qdif = diagnose_convection(psa, se_broadcast, qa_broadcast * 1000., qsat_broadcast * 1000.)

        test_itop = 5
        test_qdif = 1.1395
        # Check that itop and qdif is not null.
        self.assertEqual(itop[0,0], test_itop)
        self.assertAlmostEqual(qdif[0,0],test_qdif,places=4)
     
    def test_get_convective_tendencies_moist_adiabat(self):
        psa = jnp.ones((ix, il)) #normalized surface pressure
        xyz = (ix, il, kx)
        #test using moist adiabatic temperature profile with mid-troposphere dry anomaly
        se = jnp.array([482562.19904568, 404459.50322158, 364997.46113127, 343674.54474717, 328636.42287272, 316973.69544231, 301500., 301500.])
        qa = jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])

        se_broadcast = jnp.tile(se[jnp.newaxis, jnp.newaxis, :], (ix, il, 1))
        qa_broadcast = jnp.tile(qa[jnp.newaxis, jnp.newaxis, :], (ix, il, 1))
        qsat_broadcast = jnp.tile(qsat[jnp.newaxis, jnp.newaxis, :], (ix, il, 1))

        convection = ConvectionData.zeros((ix, il), kx, psa=psa, se=se_broadcast)
        humidity = HumidityData.zeros((ix, il), kx, qsat=qsat_broadcast*1000.)
        state = PhysicsState.zeros(xyz,specific_humidity=qa_broadcast*1000.)
        physics_data = PhysicsData.zeros((ix, il), kx, humidity=humidity, convection=convection)

        physics_tendencies, physics_data = get_convection_tendencies(state, physics_data)

        test_cbmf = jnp.array(0.019614903)
        test_precnv = jnp.array(0.21752352)
        test_dfse = jnp.array([  0., 0., 0., 0. ,-29.774475, 402.0166, 171.78418, 0.])
        test_dfqa = jnp.array([ 0., 0., 0., 0.01235308,  0.07379276, -0.15330768, -0.08423203, -0.05377656])

        rhs = 1/physics_data.convection.psa
        test_ttend = test_dfse
        test_ttend = test_ttend.at[1:].set(test_dfse[1:] * rhs[0,0] * grdscp[1:])

        test_qtend = test_dfqa
        test_qtend = test_qtend.at[1:].set(test_dfqa[1:] * rhs[0,0] * grdsig[1:])

        # Check that itop and qdif is not null.
        self.assertAlmostEqual(physics_data.convection.cbmf[0,0], test_cbmf, places=4)
        self.assertAlmostEqual(physics_data.convection.precnv[0,0], test_precnv, places=4)

        #check a few values of the fluxes
        self.assertAlmostEqual(physics_tendencies.temperature[0,0,4], test_ttend[4], places=2)
        self.assertAlmostEqual(physics_tendencies.specific_humidity[0,0,4], test_qtend[4], places=2) 
        self.assertAlmostEqual(physics_tendencies.temperature[0,0,5], test_ttend[5], places=2)
        self.assertAlmostEqual(physics_tendencies.specific_humidity[0,0,5], test_qtend[5], places=2) 
        self.assertAlmostEqual(physics_tendencies.temperature[0,0,6], test_ttend[6], places=2)
        self.assertAlmostEqual(physics_tendencies.specific_humidity[0,0,6], test_qtend[6], places=2) 
      
