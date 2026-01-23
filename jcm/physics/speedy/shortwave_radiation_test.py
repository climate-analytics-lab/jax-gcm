import unittest
import jax.numpy as jnp
import numpy as np
import jax
import jax_datetime as jdt
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
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx))

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

        global ForcingData, SurfaceFluxData, HumidityData, ConvectionData, CondensationData, SWRadiationData, DateData, PhysicsData, \
               PhysicsState, PhysicsTendency, get_clouds, get_zonal_average_fields, get_shortwave_rad_fluxes, solar, epssw, solc, parameters, forcing, geometry
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, CondensationData, SWRadiationData, DateData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.shortwave_radiation import get_clouds, get_zonal_average_fields, get_shortwave_rad_fluxes, solar
        from jcm.physics.speedy.physical_constants import epssw, solc
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        parameters = Parameters.default()
        forcing = ForcingData.zeros((ix, il))
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx))

    def test_shortwave_radiation(self):
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
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
        # Construct a varying iptop to catch layer-dependent effects and indexing bugs
        iptop = np.ones(xy, dtype=int) * jnp.linspace(0,kx,il).astype(int)[jnp.newaxis,:]
        fmask = .7 * np.ones(xy)

        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, fmask=fmask))

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx,compute_shortwave=True)

        date_data = DateData.zeros()
        date_data.tyear = 0.6

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        forcing = ForcingData.zeros(xy)
        _, physics_data = get_clouds(state, physics_data, parameters, forcing, geometry)
        physics_data = get_zonal_average_fields(state, physics_data, forcing, geometry)
        _, physics_data = get_shortwave_rad_fluxes(state, physics_data, parameters, forcing, geometry)

        # FIXME: testing against itself at the moment, need to get updated values from speedy.f90
        
        # surface downward shortwave radiation at all latitudes
        self.assertTrue(np.allclose(physics_data.shortwave_rad.rsds[0, :], [
            0.,          0.,          0.,          0.,          1.3527119,  10.074685,
            22.306503,   36.703987,   52.63945,    69.695274,   87.54011,   105.88558,
            88.18152,    101.30692,   114.275536,  126.95063,   139.21114,   150.95172,
            163.9722,    174.51448,   184.30727,   193.30157,   201.45921,   208.75128,
            218.38214,   223.96529,   228.6237,    232.34952,   235.13815,   236.98706,
            241.9083,    241.8789,    240.90105,   238.98369,   236.1414,    232.39674,
            231.47655,   225.99542,   219.77196,   212.92317,   205.62863,   198.1823,
            191.12292,   185.7362,    185.85603,   186.12903,   185.3112,    183.42676,
        ], atol=1e-4))

        # surface net (downward) shortwave radiation at all latitudes
        self.assertTrue(np.allclose(physics_data.shortwave_rad.rsns[0, :], [
            0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.35271192, 10.07468510,
            22.30650330, 36.70398712, 52.63945007, 69.69527435, 87.54010773,
            105.88558197, 88.18151855, 101.30692291, 114.27553558, 126.95063019,
            139.21113586, 150.95172119, 163.97219849, 174.51448059, 184.30726624,
            193.30157471, 201.45921326, 208.75128174, 218.38214111, 223.96528625,
            228.62370300, 232.34951782, 235.13815308, 236.98706055, 241.90829468,
            241.87890625, 240.90104675, 238.98368835, 236.14140320, 232.39674377,
            231.47654724, 225.99542236, 219.77195740, 212.92317200, 205.62863159,
            198.18229675, 191.12292480, 185.73620605, 185.85603333, 186.12902832,
            185.31120300, 183.42675781
        ], atol=1e-4))

        # top of atmosphere net shortwave radiation at all latitudes
        self.assertTrue(np.allclose(physics_data.shortwave_rad.ftop[0, :], [
            -0.29883093, -0.29883093, -0.29883093, -0.29883093, 2.09915209,
            16.96084595, 36.72006607, 58.85144806, 82.28882599, 106.43119812,
            130.88499451, 155.36427307, 130.87739563, 148.06996155, 164.85107422,
            181.12614441, 196.81008911, 211.82502747, 227.59419250, 241.13302612,
            253.80155945, 265.54354858, 276.30722046, 286.04598999, 297.72799683,
            305.37142944, 311.86950684, 317.19604492, 321.33135986, 324.26248169,
            330.78094482, 331.29727173, 330.60293579, 328.72210693, 325.69091797,
            321.55966187, 323.57238770, 317.37036133, 310.34744263, 302.68075562,
            294.63183594, 286.62426758, 279.44503784, 275.01593018, 279.19412231,
            284.20947266, 288.18334961, 291.08874512
        ], atol=1e-4))

        # column-mean absorbed shortwave flux at all latitudes
        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=0)[0, :], [
            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.13065891, 0.89812386,
            1.83904934, 2.80578613, 3.74352622, 4.62934351, 5.45546579, 6.22219181,
            5.33698463, 5.84537888, 6.32194138, 6.77193737, 7.19987106, 7.60916185,
            7.95274878, 8.32731819, 8.68678474, 9.03024769, 9.35600090, 9.66183662,
            9.91823387, 10.17576408, 10.40572453, 10.60581493, 10.77414989, 10.90942574,
            11.10908127, 11.17728901, 11.21273518, 11.21730423, 11.19368935, 11.14536285,
            11.51197910, 11.42187119, 11.32193184, 11.21969795, 11.12539673, 11.05524063,
            11.04026604, 11.15996361, 11.66725826, 12.26005840, 12.85902119, 13.45774651
        ], atol=1e-4))

        # mean across all latitudes of absorbed shortwave flux at each level
        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=2)[:, 0], [
            3.83171153, 7.95941877, 14.45124817, 6.03165197,
            7.87542248, 10.84506035, 8.45241356, 5.18130398
        ], atol=1e-4))

    def test_output_shapes(self):
        # Ensure that the output shapes are correct
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)
        state = PhysicsState.zeros(zxy)
        forcing = ForcingData.zeros(xy)

        new_data = get_zonal_average_fields(state, physics_data, forcing, geometry)
        
        self.assertEqual(new_data.shortwave_rad.fsol.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.ozupp.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.ozone.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.stratz.shape, (ix, il))
        self.assertEqual(new_data.shortwave_rad.zenit.shape, (ix, il))

    def test_solar_radiation_values(self):
        # Test that the solar radiation values are computed correctly
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)

        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
       
        physics_data = get_zonal_average_fields(state, physics_data, forcing, geometry)

        topsr = solar(date_data.tyear, geometry=geometry)
        self.assertTrue(jnp.allclose(physics_data.shortwave_rad.fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)

        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
        
        physics_data = get_zonal_average_fields(state, physics_data, forcing, geometry)

        fs0 = 6.0
        self.assertTrue(jnp.all(physics_data.shortwave_rad.stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - physics_data.shortwave_rad.fsol, 0) == physics_data.shortwave_rad.stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        xy = (ix, il)
        zxy = (kx, ix, il)
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-04-01 12:00:00'))

        physics_data = PhysicsData.zeros(xy,kx,date=date_data)
        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
        physics_data = get_zonal_average_fields(state, physics_data, forcing, geometry)

        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * geometry.sia**2 - 0.5
        expected_ozone = 0.4 * epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (date_data.tyear + 10.0 / 365.0)))  + 1.8 * flat2)
        np.testing.assert_allclose(physics_data.shortwave_rad.ozone[:, 0], physics_data.shortwave_rad.fsol[:, 0] * expected_ozone[0], atol=1e-4)

    def test_random_input_consistency(self):
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx,date=date_data)
        state = PhysicsState.zeros(zxy)
        physics_data = get_zonal_average_fields(state, physics_data, forcing, geometry)
        
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
        _, f_vjp = jax.vjp(get_zonal_average_fields, state, physics_data, forcing, geometry)
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
        forcing = ForcingData.ones(xy)
        physics_data.shortwave_rad.compute_shortwave = True

        # Calculate gradient
        _, f_vjp = jax.vjp(get_shortwave_rad_fluxes, state, physics_data, parameters, forcing, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())

    def test_clouds_gradients_isnan_with_realistic_values_grad(self):
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes

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

        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, fmask=fmask))

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx, compute_shortwave=True)

        date_data = DateData.zeros()
        date_data.tyear = 0.6

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        forcing = ForcingData.zeros(xy)
        # Calculate gradient
        primals, f_vjp = jax.vjp(get_clouds, state, physics_data, parameters, forcing, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstate, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input)
        
        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

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

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx)
        date_data = DateData.zeros()
        date_data.tyear = 0.6
        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, date=date_data)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        _, physics_data = get_clouds(state, physics_data, parameters, forcing, geometry)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, forcing_f,geometry_f):
            data_out = get_zonal_average_fields(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.0001)

    def test_get_shortwave_rad_fluxes_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test whether gradients are close for shortwave radiation"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)
        physics_data.shortwave_rad.compute_shortwave = True

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, forcing_f,geometry_f):
            tend_out, data_out = get_shortwave_rad_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
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
        forcing = ForcingData.zeros(xy, fmask=fmask)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, forcing_f,geometry_f):
            tend_out, data_out = get_clouds(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)