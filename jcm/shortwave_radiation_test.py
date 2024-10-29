import unittest
import jax.numpy as jnp
import numpy as np
# truth for test cases are generated from https://github.com/duncanwp/speedy_test

class TestSolar(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)
        global sw
        from jcm import shortwave_radiation as sw

    def test_solar(self):
        self.assertTrue(np.allclose(sw.solar(0.2), np.array([
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
        self.assertTrue(np.allclose(sw.solar(0.4), np.array([
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
        self.assertTrue(np.allclose(sw.solar(0.6), np.array([
            0., 0., 0., 0., 2.42301138, 17.44981519, 37.44706963, 59.86771264,
            83.6333103, 108.1344301, 132.97031768, 157.84825598, 182.53801702,
            206.84837586, 230.61437093, 253.6899679, 275.94351445, 297.25534724,
            317.5157371, 336.62422101, 354.48898098, 371.02626785, 386.16057506,
            399.82446689, 411.95866549, 422.51235541, 431.44315853, 438.71756928,
            444.31126415, 448.20948277, 450.40765545, 450.9120464, 449.74077685,
            446.92519666, 442.51191674, 436.56582757, 429.17485652, 420.45766136,
            410.57670499, 399.7619425, 388.35679371, 376.91876172, 366.48029222,
            359.54828853, 363.72218759, 368.79349031, 372.31796687, 374.28083132]), atol=1e-4))
        self.assertTrue(np.allclose(sw.solar(0.8), np.array([
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
        self.assertTrue(np.allclose(sw.solar(1.0), np.array([
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
        self.assertTrue(np.allclose(sw.solar(0.6, 1300), np.array([
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
        
class TestShortWaveRadiation(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)
        global pd, phys, sw, pc, geo
        from jcm import physics_data as pd
        from jcm import physics as phys
        from jcm import shortwave_radiation as sw    
        from jcm import physical_constants as pc
        from jcm import geometry as geo

    def test_shortwave_radiation(self):        
        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat

        xy = (ix, il)
        broadcast = lambda a: jnp.tile(a[None, None, :], xy + (1,))
        qa, qsat, rh = broadcast(qa), broadcast(qsat), broadcast(rh)

        psa = jnp.ones(xy)
        precnv = -1 * np.ones(xy)
        precls = 4 * np.ones(xy)
        iptop = 8 * np.ones(xy)
        gse = .01 * np.ones(xy)
        fmask = .7 * np.ones(xy)
        tyear = 0.6

        surface_flux = pd.SurfaceFluxData(xy,fmask=fmask)
        humidity = pd.HumidityData(xy, kx, rh=rh, qsat=qsat)
        convection = pd.ConvectionData(xy, kx, psa=psa, iptop=iptop, precnv=precnv)
        condensation = pd.CondensationData(xy, kx, precls=precls)
        sw_data = pd.SWRadiationData(xy, kx, gse=gse)
        date_data = pd.DateData(tyear=tyear)

        physics_data = pd.PhysicsData(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = phys.PhysicsState(jnp.zeros_like(qa), jnp.zeros_like(qa), jnp.zeros_like(qa), specific_humidity=qa, geopotential=jnp.zeros_like(qa), surface_pressure=jnp.zeros(xy))

        _, physics_data = sw.clouds(state, physics_data)
        _, physics_data = sw.get_zonal_average_fields(state, physics_data)
        _, physics_data = sw.get_shortwave_rad_fluxes(state, physics_data)
        
        self.assertTrue(np.allclose(physics_data.shortwave_rad.rsds[0, :], [
            0., 0., 0., 0., 1.08102491, 7.9856262, 17.54767508, 28.67351887, 40.8631746, 53.79605732,
            67.22801389, 80.95422179, 94.79448489, 108.58701854, 122.18603817, 135.46087123, 148.29548103,
            160.58828119, 172.25138545, 183.21006299, 193.40177528, 202.77492961, 211.28786499, 218.90753726,
            225.60832278, 231.37096758, 236.18142325, 240.03003908, 242.91081466, 244.82085417, 245.76014002,
            245.7315415, 244.74127921, 242.79984604, 239.92358203, 236.13704304, 231.47654032, 225.99538369,
            219.77196135, 212.92314683, 205.62864786, 198.18231101, 191.12290959, 185.73622544, 185.85603776,
            186.12903619, 185.31120169, 183.42677496
        ], atol=1e-4))

        self.assertTrue(np.allclose(physics_data.shortwave_rad.ssr[0, :], [
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

        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=2)[0, :], [
            0., 0., 0., 0., 0.10687137, 0.73259064, 1.49614688, 2.27724268, 3.03175176, 3.74177215,
            4.40170949, 5.01239461, 5.57803226, 6.10437168, 6.59753382, 7.06324094, 7.5063241, 7.93045047,
            8.33799935, 8.73007726, 9.10661781, 9.46654098, 9.80796542, 10.12843323, 10.42514601, 10.69520145,
            10.93580878, 11.14449603, 11.31928937, 11.45886328, 11.56266179, 11.63097744, 11.66499303,
            11.66678236, 11.63927325, 11.58618806, 11.51198779, 11.42187291, 11.32194103, 11.2197034,
            11.12539961, 11.05524554, 11.04026698, 11.15996872, 11.66726167, 12.26006372, 12.85902805,
            13.45775005
        ], atol=1e-4))

        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=1)[0, :], [
            3.82887045, 7.81598669, 14.17718547, 5.65627818, 7.80939064, 12.48949685, 8.5056334, 5.21519786,
        ], atol=1e-4))

    def test_output_shapes(self):
        # Ensure that the output shapes are correct
        tyear = 0.25
        xy = (ix, il)
        xyz = (ix, il, kx)
        date_data = pd.DateData(tyear=tyear)
        physics_data = pd.PhysicsData(xy,kx,date=date_data)
        state = phys.PhysicsState(jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xy))
        _, new_data = sw.get_zonal_average_fields(state, physics_data)
        
        self.assertEqual(new_data.shortwave_rad.fsol.shape, (self.ix, self.il))
        self.assertEqual(new_data.shortwave_rad.ozupp.shape, (self.ix, self.il))
        self.assertEqual(new_data.shortwave_rad.ozone.shape, (self.ix, self.il))
        self.assertEqual(new_data.shortwave_rad.stratz.shape, (self.ix, self.il))
        self.assertEqual(new_data.shortwave_rad.zenit.shape, (self.ix, self.il))

    def test_solar_radiation_values(self):
        # Test that the sw.solar radiation values are computed correctly
        tyear = 0.25
        xy = (ix, il)
        xyz = (ix, il, kx)
        date_data = pd.DateData(tyear=tyear)
        physics_data = pd.PhysicsData(xy,kx,date=date_data)
        state = phys.PhysicsState(jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xy))
        _, physics_data = sw.get_zonal_average_fields(state, physics_data)

        topsr = sw.solar(tyear)
        self.assertTrue(jnp.allclose(physics_data.shortwave_rad.fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        tyear = 0.25
        xy = (ix, il)
        xyz = (ix, il, kx)
        date_data = pd.DateData(tyear=tyear)
        physics_data = pd.PhysicsData(xy,kx,date=date_data)
        state = phys.PhysicsState(jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xy))
        _, physics_data = sw.get_zonal_average_fields(state, physics_data)

        fs0 = 6.0
        self.assertTrue(jnp.all(physics_data.shortwave_rad.stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - physics_data.shortwave_rad.fsol, 0) == physics_data.shortwave_rad.stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        tyear = 0.25
        xy = (ix, il)
        xyz = (ix, il, kx)
        date_data = pd.DateData(tyear=tyear)
        physics_data = pd.PhysicsData(xy,kx,date=date_data)
        state = phys.PhysicsState(jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xy))
        _, physics_data = sw.get_zonal_average_fields(state, physics_data)

        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * geo.sia**2 - 0.5
        expected_ozone = 0.4 * pc.epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (tyear + 10.0 / 365.0)))  + 1.8 * flat2)
        self.assertTrue(jnp.allclose(physics_data.shortwave_rad.ozone[:, 0], physics_data.shortwave_rad.fsol[:, 0] * expected_ozone[0]))

    def test_random_input_consistency(self):
        tyear = 0.25
        xy = (ix, il)
        xyz = (ix, il, kx)
        date_data = pd.DateData(tyear=tyear)
        physics_data = pd.PhysicsData(xy,kx,date=date_data)
        state = phys.PhysicsState(jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xyz), jnp.zeros(xy))
        _, physics_data = sw.get_zonal_average_fields(state, physics_data)
        
        # Ensure outputs are consistent and within expected ranges
        self.assertTrue(jnp.all(physics_data.shortwave_rad.fsol >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.ozupp >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.ozone >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.stratz >= 0))
        self.assertTrue(jnp.all(physics_data.shortwave_rad.zenit >= 0))