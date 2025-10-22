import unittest
import jax.numpy as jnp
import numpy as np
import jax
import functools
from jax.test_util import check_vjp, check_jvp
import pytest
# truth for test cases are generated from https://github.com/duncanwp/speedy_test

class TestSolar(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global solar, geometry
        from jcm.physics.speedy.shortwave_radiation import solar
        from jcm.geometry import Geometry
        geometry = Geometry.from_grid_shape((ix, il), kx)

    def test_solar(self):
        self.assertTrue(np.allclose(solar(0.2, geometry=geometry), np.array([
            59.64891891,  82.51370562, 109.0996075 , 135.94454033,
            162.48195582, 188.46471746, 213.72891835, 238.14170523,
            261.58627434, 283.95547202, 305.15011948, 325.07762082,
            343.65189868, 360.79323687, 376.42841812, 390.49090207,
            402.92092072, 413.66583083, 422.68006932, 429.9254984 ,
            435.37150003, 438.9950085 , 440.78070068, 440.7209988 ,
            438.81611994, 435.07404132, 429.51050427, 422.14893274,
            413.02032164, 402.16320111, 389.62332055, 375.45360549,
            359.71400001, 342.47101119, 323.7977572 , 303.77351671,
            282.48360014, 260.01911561, 236.4767785 , 211.95903738,
            186.57407167, 160.43718712, 133.67240691, 106.41888862,
            78.84586166,  51.20481384,  24.06562443,   0.89269878]), atol=1e-4))
        self.assertTrue(np.allclose(solar(0.4, geometry=geometry), np.array([
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 1.17528392e-01, 1.13271540e+01, 2.91320240e+01,
            5.00775958e+01, 7.28770444e+01, 9.68131455e+01, 1.21415906e+02,
            1.46344316e+02, 1.71332241e+02, 1.96160737e+02, 2.20642698e+02,
            2.44613680e+02, 2.67926725e+02, 2.90448515e+02, 3.12057588e+02,
            3.32642980e+02, 3.52103122e+02, 3.70345744e+02, 3.87287495e+02,
            4.02853935e+02, 4.16979874e+02, 4.29609344e+02, 4.40696200e+02,
            4.50204647e+02, 4.58109880e+02, 4.64399211e+02, 4.69073258e+02,
            4.72147819e+02, 4.73656558e+02, 4.73654825e+02, 4.72225630e+02,
            4.69489091e+02, 4.65618250e+02, 4.60867185e+02, 4.55625373e+02,
            4.50536488e+02, 4.46820735e+02, 4.47873663e+02, 4.58140604e+02,
            4.66603495e+02, 4.73109251e+02, 4.77630650e+02, 4.80148724e+02]), atol=1e-4))
        self.assertTrue(np.allclose(solar(0.6, geometry=geometry), np.array([
            0., 0., 0., 0., 2.42301138, 17.44981519, 37.44706963, 59.86771264,
            83.6333103, 108.1344301, 132.97031768, 157.84825598, 182.53801702,
            206.84837586, 230.61437093, 253.6899679, 275.94351445, 297.25534724,
            317.5157371, 336.62422101, 354.48898098, 371.02626785, 386.16057506,
            399.82446689, 411.95866549, 422.51235541, 431.44315853, 438.71756928,
            444.31126415, 448.20948277, 450.40765545, 450.9120464, 449.74077685,
            446.92519666, 442.51191674, 436.56582757, 429.17485652, 420.45766136,
            410.57670499, 399.7619425, 388.35679371, 376.91876172, 366.48029222,
            359.54828853, 363.72218759, 368.79349031, 372.31796687, 374.28083132]), atol=1e-4))
        self.assertTrue(np.allclose(solar(0.8, geometry=geometry), np.array([
            2.40672590e+02, 2.39410416e+02, 2.37278513e+02, 2.48984331e+02,
            2.66799442e+02, 2.86134104e+02, 3.05646230e+02, 3.24707974e+02,
            3.42958056e+02, 3.60158149e+02, 3.76136095e+02, 3.90759256e+02,
            4.03921448e+02, 4.15535691e+02, 4.25530154e+02, 4.33845751e+02,
            4.40434599e+02, 4.45259173e+02, 4.48291587e+02, 4.49513271e+02,
            4.48914644e+02, 4.46494901e+02, 4.42261840e+02, 4.36231709e+02,
            4.28429095e+02, 4.18886672e+02, 4.07645224e+02, 3.94753408e+02,
            3.80267620e+02, 3.64252011e+02, 3.46778141e+02, 3.27925150e+02,
            3.07779834e+02, 2.86436505e+02, 2.63997727e+02, 2.40574768e+02,
            2.16288991e+02, 1.91274040e+02, 1.65679673e+02, 1.39678886e+02,
            1.13480705e+02, 8.73568473e+01, 6.16981674e+01, 3.71583316e+01,
            1.51012308e+01, 1.34429313e-01, 0.00000000e+00, 0.00000000e+00]), atol=1e-4))
        self.assertTrue(np.allclose(solar(1.0, geometry=geometry), np.array([
            553.93421795, 551.02918596, 545.81297397, 538.30746507, 528.54406252,
            516.56378888, 506.85181087, 506.40750073, 508.57359122, 511.41450948,
            514.02258691, 515.87725366, 516.65036719, 516.12420873, 514.15095359,
            510.63050328, 505.49750198, 498.71321538, 490.2604143, 480.13978746,
            468.36747184, 454.973485, 440.00027589, 423.50189151, 405.54332338,
            386.19977815, 365.55671479, 343.709479, 320.76336998, 296.8341639,
            272.04837717, 246.54473496, 220.47604586, 194.01174781, 167.34305754,
            140.69016619, 114.31490876, 88.54240315, 63.80108395, 40.70440853,
            20.24490036, 4.43498764, 0., 0., 0., 0., 0., 0.]), atol=1e-4))

        # other csol values
        self.assertTrue(np.allclose(solar(0.6, 1300, geometry=geometry), np.array([
            0.,          0.,           0.,           0.,
            2.30256929,  16.58242672,  35.58566559,  56.89183219,
            79.47609897, 102.75932685, 126.36068201, 150.00199764,
            173.46448986, 196.56643905, 219.15108349, 241.07964786,
            262.22702397, 282.47949664, 301.7327911 , 319.89143809,
            336.86818368, 352.58344167, 366.96545876, 379.95015129,
            391.48118796, 401.51027927, 409.99715357, 416.90997081,
            422.22561652, 425.93006404, 428.01897082, 428.49828971,
            427.38524116, 424.7096167 , 420.51571036, 414.86518702,
            407.84160341, 399.55771912, 390.16792141, 379.89073483,
            369.05250864, 358.18303379, 348.26343559, 341.67600518,
            345.64242972, 350.46165015, 353.81093342, 355.67622859]), atol=1e-4))
        
    def test_solar_gradients_isnan(self):
        """Test that we can calculate gradients of shortwave radiation without getting NaN values"""
        from jcm.physics.speedy.physical_constants import solc
        primals, f_vjp = jax.vjp(solar, 0.2, 4.*solc, geometry)
        input = jnp.ones_like(primals)
        df_dtyear, df_dcsol, df_dgeo = f_vjp(input)

        self.assertFalse(jnp.any(jnp.isnan(df_dtyear)))
        
    def test_solar_gradient_check(self): 
        from jcm.physics.speedy.physical_constants import solc
        tyear = 0.2
        csol = 4.*solc

        def f(tyear, csol):
            return solar(tyear, csol, geometry)

        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (tyear, csol), 
                                atol=None, rtol=1, eps=0.0001)
        check_jvp(f, f_jvp, args = (tyear, csol), 
                                atol=None, rtol=1, eps=0.000001)
        
class TestShortWaveRadiation(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global BoundaryData, SurfaceFluxData, HumidityData, ConvectionData, CondensationData, SWRadiationData, DateData, PhysicsData, \
               PhysicsState, PhysicsTendency, get_clouds, get_zonal_average_fields, get_shortwave_rad_fluxes, solar, epssw, solc, parameters, boundaries, geometry
        from jcm.boundaries import BoundaryData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, CondensationData, SWRadiationData, DateData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.shortwave_radiation import get_clouds, get_zonal_average_fields, get_shortwave_rad_fluxes, solar
        from jcm.physics.speedy.physical_constants import epssw, solc
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        parameters = Parameters.default()
        boundaries = BoundaryData.zeros((ix, il))
        geometry = Geometry.from_grid_shape((ix, il), kx)

    def test_shortwave_radiation(self):
        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat
        geopotential = jnp.arange(7, -1, -1, dtype = float)
        se = .1*geopotential

        xy = (ix, il)
        zxy = (kx, ix, il)
        broadcast = lambda a: jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1,) + xy)
        qa, qsat, rh, geopotential, se = broadcast(qa), broadcast(qsat), broadcast(rh), broadcast(geopotential), broadcast(se)

        psa = jnp.ones(xy)
        precnv = -1.0 * np.ones(xy)
        precls = 4.0 * np.ones(xy)
        iptop = 8 * np.ones(xy, dtype=int)
        fmask = .7 * np.ones(xy)

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx,compute_shortwave=True)

        date_data = DateData.zeros()
        date_data.tyear = 0.6

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        boundaries = BoundaryData.zeros(xy, fmask=fmask)
        _, physics_data = get_clouds(state, physics_data, parameters, boundaries, geometry)
        physics_data = get_zonal_average_fields(state, physics_data, boundaries, geometry)
        _, physics_data = get_shortwave_rad_fluxes(state, physics_data, parameters, boundaries, geometry)
        
        np.testing.assert_allclose(physics_data.shortwave_rad.rsds[0, :], [
            0., 0., 0., 0., 1.08102491, 7.9856262, 17.54767508, 28.67351887, 40.8631746, 53.79605732,
            67.22801389, 80.95422179, 94.79448489, 108.58701854, 122.18603817, 135.46087123, 148.29548103,
            160.58828119, 172.25138545, 183.21006299, 193.40177528, 202.77492961, 211.28786499, 218.90753726,
            225.60832278, 231.37096758, 236.18142325, 240.03003908, 242.91081466, 244.82085417, 245.76014002,
            245.7315415, 244.74127921, 242.79984604, 239.92358203, 236.13704304, 231.47654032, 225.99538369,
            219.77196135, 212.92314683, 205.62864786, 198.18231101, 191.12290959, 185.73622544, 185.85603776,
            186.12903619, 185.31120169, 183.42677496
        ], atol=1e-4)

        self.assertTrue(np.allclose(physics_data.shortwave_rad.rsns[0, :], [
            0., 0., 0., 0., 1.08102491, 7.9856262, 17.54767508, 28.67351887, 40.8631746, 53.79605732,
            67.22801389, 80.95422179, 94.79448489, 108.58701854, 122.18603817, 135.46087123, 148.29548103,
            160.58828119, 172.25138545, 183.21006299, 193.40177528, 202.77492961, 211.28786499, 218.90753726,
            225.60832278, 231.37096758, 236.18142325, 240.03003908, 242.91081466, 244.82085417, 245.76014002,
            245.7315415, 244.74127921, 242.79984604, 239.92358203, 236.13704304, 231.47654032, 225.99538369,
            219.77196135, 212.92314683, 205.62864786, 198.18231101, 191.12290959, 185.73622544, 185.85603776,
            186.12903619, 185.31120169, 183.42677496
        ], atol=1e-4))

        self.assertTrue(np.allclose(physics_data.shortwave_rad.ftop[0, :], [
            0., 0., 0., 0., 1.93599586, 13.84635135, 29.51685016, 46.89146027, 65.11718871, 83.73023451,
            102.44168978, 121.0533787, 139.41874296, 157.42199198, 174.96630874, 191.96679879, 208.34607385,
            224.03188495, 238.95538028, 253.05068109, 266.25471774, 278.50725748, 289.75158834, 299.9350031,
            309.0094909, 316.93257921, 323.66789347, 329.1860073, 333.46512965, 336.4917604, 338.26143437,
            338.77936105, 338.06122342, 336.13410495, 333.03776805, 328.82654748, 323.57244264, 317.37036698,
            310.3474896, 302.680774, 294.63184471, 286.62427534, 279.44504545, 275.01597523, 279.19413113,
            284.20954594, 288.1834261, 291.08877534
        ], atol=1e-4))

        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=0)[0, :], [
            0., 0., 0., 0., 0.10687137, 0.73259064, 1.49614688, 2.27724268, 3.03175176, 3.74177215,
            4.40170949, 5.01239461, 5.57803226, 6.10437168, 6.59753382, 7.06324094, 7.5063241, 7.93045047,
            8.33799935, 8.73007726, 9.10661781, 9.46654098, 9.80796542, 10.12843323, 10.42514601, 10.69520145,
            10.93580878, 11.14449603, 11.31928937, 11.45886328, 11.56266179, 11.63097744, 11.66499303,
            11.66678236, 11.63927325, 11.58618806, 11.51198779, 11.42187291, 11.32194103, 11.2197034,
            11.12539961, 11.05524554, 11.04026698, 11.15996872, 11.66726167, 12.26006372, 12.85902805,
            13.45775005
        ], atol=1e-4))

        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=2)[:, 0], [
            3.82887045, 7.81598669, 14.17718547, 5.65627818, 7.80939064, 12.48949685, 8.5056334, 5.21519786,
        ], atol=1e-4))

    def test_output_shapes(self):
        from datetime import datetime
        from jcm.date import Timestamp
        # Ensure that the output shapes are correct
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=Timestamp.from_datetime(datetime(2000, 3, 21)))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)
        state = PhysicsState.zeros(zxy)
        boundaries = BoundaryData.zeros(xy)

        new_data = get_zonal_average_fields(state, physics_data, boundaries, geometry)
        
        self.assertEqual(new_data.shortwave_rad.fsol.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.ozupp.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.ozone.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.stratz.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.zenit.shape, (ix, il))

    def test_solar_radiation_values(self):
        # Test that the solar radiation values are computed correctly
        from datetime import datetime
        from jcm.date import Timestamp
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=Timestamp.from_datetime(datetime(2000, 3, 21)))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)

        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
       
        physics_data = get_zonal_average_fields(state, physics_data, boundaries, geometry)

        topsr = solar(date_data.tyear, geometry=geometry)
        self.assertTrue(jnp.allclose(physics_data.shortwave_rad.fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        from datetime import datetime
        from jcm.date import Timestamp
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=Timestamp.from_datetime(datetime(2000, 3, 21)))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)

        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
        
        physics_data = get_zonal_average_fields(state, physics_data, boundaries, geometry)

        fs0 = 6.0
        self.assertTrue(jnp.all(physics_data.shortwave_rad.stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - physics_data.shortwave_rad.fsol, 0) == physics_data.shortwave_rad.stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        from datetime import datetime
        from jcm.date import Timestamp
        xy = (ix, il)
        zxy = (kx, ix, il)
        date_data = DateData.set_date(model_time=Timestamp.from_datetime(datetime(2000, 4, 1,12)))

        physics_data = PhysicsData.zeros(xy,kx,date=date_data)
        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
        physics_data = get_zonal_average_fields(state, physics_data, boundaries, geometry)

        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * geometry.sia**2 - 0.5
        expected_ozone = 0.4 * epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (date_data.tyear + 10.0 / 365.0)))  + 1.8 * flat2)
        np.testing.assert_allclose(physics_data.shortwave_rad.ozone[:, 0], physics_data.shortwave_rad.fsol[:, 0] * expected_ozone[0], atol=1e-4)

    def test_random_input_consistency(self):
        from datetime import datetime
        from jcm.date import Timestamp
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=Timestamp.from_datetime(datetime(2000, 3, 21)))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)
        state = PhysicsState.zeros(zxy)
        physics_data = get_zonal_average_fields(state, physics_data, boundaries, geometry)
        
        # Ensure outputs are consistent and within expected ranges
        self.assertTrue(jnp.all(physics_data.shortwave_rad.fsol >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.ozupp >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.ozone >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.stratz >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.zenit >= 0))
        
    def test_get_zonal_average_fields_gradients_isnan(self):
        """Test that we can calculate gradients of shortwave radiation without getting NaN values"""

        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat
        geopotential = jnp.arange(7, -1, -1, dtype = float)
        se = .1*geopotential

        xy = (ix, il)
        zxy = (kx, ix, il)
        broadcast = lambda a: jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1,) + xy)
        qa, qsat, rh, geopotential, se = broadcast(qa), broadcast(qsat), broadcast(rh), broadcast(geopotential), broadcast(se)

        psa = jnp.ones(xy)
        precnv = -1.0 * jnp.ones(xy)
        precls = 4.0 * jnp.ones(xy)
        iptop = 8 * jnp.ones(xy, dtype=int)
        fmask = .7 * jnp.ones(xy)

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx)

        date_data = DateData.zeros()
        date_data.tyear = 0.6

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)

        # Calculate gradient
        _, f_vjp = jax.vjp(get_zonal_average_fields, state, physics_data, boundaries, geometry)
        datas = PhysicsData.ones(xy,kx)
        df_dstates, df_ddatas, _, _ = f_vjp(datas)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())

    def test_get_shortwave_rad_fluxes_gradients_isnan_ones(self):
        """Test that we can calculate gradients of shortwave radiation without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        boundaries = BoundaryData.ones(xy)
        physics_data.shortwave_rad.compute_shortwave = True

        # Calculate gradient
        _, f_vjp = jax.vjp(get_shortwave_rad_fluxes, state, physics_data, parameters, boundaries, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dboundaries, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dboundaries.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())

    def test_clouds_gradients_isnan_with_realistic_values_grad(self):
        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat
        geopotential = jnp.arange(7, -1, -1, dtype = float)
        se = .1*geopotential

        xy = (ix, il)
        zxy = (kx, ix, il)
        broadcast = lambda a: jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1,) + xy)
        qa, qsat, rh, geopotential, se = broadcast(qa), broadcast(qsat), broadcast(rh), broadcast(geopotential), broadcast(se)

        psa = jnp.ones(xy)
        precnv = -1.0 * jnp.ones(xy)
        precls = 4.0 * jnp.ones(xy)
        iptop = 8 * jnp.ones(xy, dtype=int)
        fmask = .7 * jnp.ones(xy)

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx, compute_shortwave=True)

        date_data = DateData.zeros()
        date_data.tyear = 0.6

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        boundaries = BoundaryData.zeros(xy, fmask=fmask)
        # Calculate gradient
        primals, f_vjp = jax.vjp(get_clouds, state, physics_data, parameters, boundaries, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstate, df_ddatas, df_dparams, df_dboundaries, df_dgeometry = f_vjp(input)
        
        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dboundaries.isnan().any_true())

    @pytest.mark.skip(reason="JAX gradients are producing nans")
    def test_get_zonal_average_fields_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test whether gradients are close for shortwave radiation"""
        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat
        geopotential = jnp.arange(7, -1, -1, dtype = float)
        se = .1*geopotential
        xy = (ix, il)
        zxy = (kx, ix, il)
        broadcast = lambda a: jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1,) + xy)
        qa, qsat, rh, geopotential, se = broadcast(qa), broadcast(qsat), broadcast(rh), broadcast(geopotential), broadcast(se)
        psa = jnp.ones(xy)
        precnv = -1.0 * jnp.ones(xy)
        precls = 4.0 * jnp.ones(xy)
        iptop = 8 * jnp.ones(xy, dtype=int)
        fmask = .7 * jnp.ones(xy)

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx)
        date_data = DateData.zeros()
        date_data.tyear = 0.6
        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        _, physics_data = get_clouds(state, physics_data, parameters, boundaries, geometry)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, boundaries_f,geometry_f):
            data_out = get_zonal_average_fields(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.0001)

    def test_get_shortwave_rad_fluxes_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test whether gradients are close for shortwave radiation"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        boundaries = BoundaryData.ones(xy)
        physics_data.shortwave_rad.compute_shortwave = True

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, boundaries_f,geometry_f):
            tend_out, data_out = get_shortwave_rad_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.0001)

    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_clouds_gradient_check_realistic_values(self):
        from jcm.utils import convert_back, convert_to_float

        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat
        geopotential = jnp.arange(7, -1, -1, dtype = float)
        se = .1*geopotential

        xy = (ix, il)
        zxy = (kx, ix, il)
        broadcast = lambda a: jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1,) + xy)
        qa, qsat, rh, geopotential, se = broadcast(qa), broadcast(qsat), broadcast(rh), broadcast(geopotential), broadcast(se)

        psa = jnp.ones(xy)
        precnv = -1.0 * jnp.ones(xy)
        precls = 4.0 * jnp.ones(xy)
        iptop = 8 * jnp.ones(xy, dtype=int)
        fmask = .7 * jnp.ones(xy)

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx, compute_shortwave=True)

        date_data = DateData.zeros()
        date_data.tyear = 0.6

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        boundaries = BoundaryData.zeros(xy, fmask=fmask)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        boundaries_floats = convert_to_float(boundaries)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, boundaries_f,geometry_f):
            tend_out, data_out = get_clouds(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       boundaries=convert_back(boundaries_f, boundaries), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, boundaries_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)