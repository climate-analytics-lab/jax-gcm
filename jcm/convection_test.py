import unittest
import jax.numpy as jnp
import jax
from jax import tree_util

class TestConvectionUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)
        
        global ConvectionData, HumidityData, BoundaryData, PhysicsData, PhysicsState, parameters, diagnose_convection, get_convection_tendencies, grdscp, grdsig, PhysicsTendency, get_qsat, fsg, rgas, cp
        from jcm.boundaries import BoundaryData
        from jcm.params import Parameters
        parameters = Parameters.default()
        from jcm.physics_data import ConvectionData, HumidityData, PhysicsData
        from jcm.physics import PhysicsState, PhysicsTendency
        from jcm.convection import diagnose_convection, get_convection_tendencies
        from jcm.physical_constants import grdscp, grdsig, rgas, cp
        from jcm.humidity import get_qsat
        from jcm.geometry import fsg

    def test_diagnose_convection_varying(self):
        ps = jnp.ones((ix, il))
        ta = 300 * jnp.ones((kx, ix, il)) * (fsg[:, jnp.newaxis, jnp.newaxis]**(.05 * jnp.cos(3*jnp.arange(il)[jnp.newaxis, jnp.newaxis, :] / il)**3))
        qsat = get_qsat(ta, ps, fsg[:, jnp.newaxis, jnp.newaxis])
        qa = jnp.sin(2*jnp.arange(ix)[jnp.newaxis, :, jnp.newaxis]/ix)**2 * qsat * 3.5
        phi = rgas * ta * jnp.log(fsg[:, jnp.newaxis, jnp.newaxis])
        se = cp * ta + phi
        
        iptop, qdif = diagnose_convection(ps, se, qa, qsat, parameters)

        from os import path
        test_data_dir = path.dirname(path.realpath(__file__)) + '/data/test'
        iptop_f90 = jnp.load(test_data_dir + '/iptop.npy')
        qdif_f90 = jnp.load(test_data_dir + '/qdif.npy')

        assert jnp.allclose(iptop, iptop_f90, atol=1e-4)
        assert jnp.allclose(qdif, qdif_f90, atol=1e-4)

    def test_diagnose_convection_isothermal(self):
        psa = jnp.ones((ix, il))
        
        se = jnp.array([594060.  , 483714.2 , 422181.7 , 378322.1 , 344807.97, 320423.78,
       304056.8 , 293391.7 ])
        qa = jnp.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])
        qsat = get_qsat(jnp.ones((1,1,1)) * 288., jnp.ones((1,1,1)), fsg[:, jnp.newaxis, jnp.newaxis])
        
        se_broadcast = jnp.tile(se[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qa_broadcast = jnp.tile(qa[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qsat_broadcast = jnp.tile(qsat, (1, ix, il))
        
        itop, qdif = diagnose_convection(psa, se_broadcast, qa_broadcast * 1000., qsat_broadcast * 1000., parameters)
        
        self.assertTrue(jnp.allclose(itop, jnp.ones((ix, il))*9))
        self.assertTrue(jnp.allclose(qdif, jnp.zeros((ix, il))))

    def test_get_convection_tendencies_isnan_ones(self): 
        xy = (ix, il)
        zxy = (kx, ix, il)
        
        physics_data = PhysicsData.ones(xy, kx)  
        
        state = PhysicsState.ones(zxy)

        boundaries = BoundaryData.ones(xy)
        
        primals, f_vjp = jax.vjp(get_convection_tendencies, state, physics_data, parameters, boundaries) 
        
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx)
        input = (tends, datas)
        
        df_dstate, df_ddatas, df_dparams, df_dboundaries = f_vjp(input)
        
        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dboundaries.isnan().any_true())

    def test_diagnose_convection_moist_adiabat(self):
        psa = jnp.ones((ix, il)) #normalized surface pressure

        #test using moist adiabatic temperature profile with mid-troposphere dry anomaly
        se = jnp.array([482562.19904568, 404459.50322158, 364997.46113127, 343674.54474717, 328636.42287272, 316973.69544231, 301500., 301500.])
        qa = jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])

        se_broadcast = jnp.tile(se[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qa_broadcast = jnp.tile(qa[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qsat_broadcast = jnp.tile(qsat[:, jnp.newaxis, jnp.newaxis], (1, ix, il))

        itop, qdif = diagnose_convection(psa, se_broadcast, qa_broadcast * 1000., qsat_broadcast * 1000., parameters)

        test_itop = 5
        test_qdif = 1.1395
        # Check that itop and qdif is not null.
        self.assertEqual(itop[0,0], test_itop)
        self.assertAlmostEqual(qdif[0,0],test_qdif,places=4)

    def test_get_convection_tendencies_varying(self):
        ps = jnp.ones((ix, il))
        ta = 300 * jnp.ones((kx, ix, il)) * (fsg[:, jnp.newaxis, jnp.newaxis]**(.05 * jnp.cos(3*jnp.arange(il)[jnp.newaxis, jnp.newaxis, :] / il)**3))
        qsat = get_qsat(ta, ps, fsg[:, jnp.newaxis, jnp.newaxis])
        qa = jnp.sin(2*jnp.arange(ix)[jnp.newaxis, :, jnp.newaxis]/ix)**2 * qsat * 3.5
        phi = rgas * ta * jnp.log(fsg[:, jnp.newaxis, jnp.newaxis])
        se = cp * ta + phi

        convection = ConvectionData.zeros((ix, il), kx, psa=ps, se=se)
        humidity = HumidityData.zeros((ix, il), kx, qsat=qsat)
        state = PhysicsState.zeros((ix, il, kx), specific_humidity=qa)
        physics_data = PhysicsData.zeros((ix, il), kx, humidity=humidity, convection=convection)
        boundaries = BoundaryData.zeros((ix,il))

        physics_tendencies, physics_data = get_convection_tendencies(state, physics_data, parameters, boundaries)

        from os import path
        test_data_dir = path.dirname(path.realpath(__file__)) + '/data/test'
        iptop_f90 = jnp.load(test_data_dir + '/iptop.npy')
        cmbf_f90 = jnp.load(test_data_dir + '/cbmf.npy')
        precnv_f90 = jnp.load(test_data_dir + '/precnv.npy')
        dfse_f90 = jnp.load(test_data_dir + '/dfse.npy')
        dfqa_f90 = jnp.load(test_data_dir + '/dfqa.npy')

        assert jnp.allclose(physics_data.convection.iptop, iptop_f90, atol=1e-4)
        assert jnp.allclose(physics_data.convection.cbmf, cmbf_f90, atol=1e-4)
        assert jnp.allclose(physics_data.convection.precnv, precnv_f90, atol=1e-4)

        rps = 1/ps
        ttend_f90 = dfse_f90.at[1:].set(dfse_f90[1:] * rps[jnp.newaxis] * grdscp[1:, jnp.newaxis, jnp.newaxis])
        qtend_f90 = dfqa_f90.at[1:].set(dfqa_f90[1:] * rps[jnp.newaxis] * grdsig[1:, jnp.newaxis, jnp.newaxis])

        # FIXME: These tests are failing due to a bug in calculating dfse and dfqa
        # assert jnp.allclose(physics_tendencies.temperature, ttend_f90, atol=1e-4)
        # assert jnp.allclose(physics_tendencies.specific_humidity, qtend_f90, atol=1e-4)

    def test_get_convection_tendencies_isothermal(self):
        psa = jnp.ones((ix, il))
        
        se = jnp.array([594060.  , 483714.2 , 422181.7 , 378322.1 , 344807.97, 320423.78,
       304056.8 , 293391.7 ])
        qa = jnp.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])
        qsat = qsat = get_qsat(jnp.ones((1,1,1)) * 288., jnp.ones((1,1,1)), fsg[:, jnp.newaxis, jnp.newaxis])
        
        se_broadcast = jnp.tile(se[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qa_broadcast = jnp.tile(qa[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qsat_broadcast = jnp.tile(qsat, (1, ix, il))
        
        convection = ConvectionData.zeros((ix, il), kx, psa=psa, se=se_broadcast)
        humidity = HumidityData.zeros((ix, il), kx, qsat=qsat_broadcast)
        state = PhysicsState.zeros((ix, il, kx), specific_humidity=qa_broadcast)
        physics_data = PhysicsData.zeros((ix, il), kx, humidity=humidity, convection=convection)

        boundaries = BoundaryData.zeros((ix,il))

        physics_tendencies, physics_data = get_convection_tendencies(state, physics_data, parameters, boundaries)

        self.assertTrue(jnp.allclose(physics_data.convection.iptop, jnp.ones((ix, il))*9))
        self.assertTrue(jnp.allclose(physics_data.convection.cbmf, jnp.zeros((ix, il))))
        self.assertTrue(jnp.allclose(physics_data.convection.precnv, jnp.zeros((ix, il))))
        self.assertTrue(jnp.allclose(physics_tendencies.temperature, jnp.zeros((kx, ix, il))))
        self.assertTrue(jnp.allclose(physics_tendencies.specific_humidity, jnp.zeros((kx, ix, il))))    

    def test_get_convection_tendencies_moist_adiabat(self):
        psa = jnp.ones((ix, il)) #normalized surface pressure
        zxy = (kx, ix, il)
        #test using moist adiabatic temperature profile with mid-troposphere dry anomaly
        se = jnp.array([482562.19904568, 404459.50322158, 364997.46113127, 343674.54474717, 328636.42287272, 316973.69544231, 301500., 301500.])
        qa = jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])

        se_broadcast = jnp.tile(se[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qa_broadcast = jnp.tile(qa[:, jnp.newaxis, jnp.newaxis], (1, ix, il))
        qsat_broadcast = jnp.tile(qsat[:, jnp.newaxis, jnp.newaxis], (1, ix, il))

        convection = ConvectionData.zeros((ix, il), kx, psa=psa, se=se_broadcast)
        humidity = HumidityData.zeros((ix, il), kx, qsat=qsat_broadcast*1000.)
        state = PhysicsState.zeros(zxy, specific_humidity=qa_broadcast*1000.)
        physics_data = PhysicsData.zeros((ix, il), kx, humidity=humidity, convection=convection)

        boundaries = BoundaryData.zeros((ix,il))

        physics_tendencies, physics_data = get_convection_tendencies(state, physics_data, parameters, boundaries)

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
        self.assertAlmostEqual(physics_tendencies.temperature[4,0,0], test_ttend[4], places=2)
        self.assertAlmostEqual(physics_tendencies.specific_humidity[4,0,0], test_qtend[4], places=2) 
        self.assertAlmostEqual(physics_tendencies.temperature[5,0,0], test_ttend[5], places=2)
        self.assertAlmostEqual(physics_tendencies.specific_humidity[5,0,0], test_qtend[5], places=2) 
        self.assertAlmostEqual(physics_tendencies.temperature[6,0,0], test_ttend[6], places=2)
        self.assertAlmostEqual(physics_tendencies.specific_humidity[6,0,0], test_qtend[6], places=2) 