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

        global solar, terrain, speedy_coords
        from jcm.physics.radiation.speedy_shortwave import solar
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords

        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords)
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)

        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        terrain, speedy_coords = convert_to_speedy_latitudes(terrain, speedy_coords)

    def test_solar(self):
        self.assertTrue(np.allclose(solar(0.2, speedy_coords=speedy_coords), np.array([
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
        self.assertTrue(np.allclose(solar(0.4, speedy_coords=speedy_coords), np.array([
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
        self.assertTrue(np.allclose(solar(0.6, speedy_coords=speedy_coords), np.array([
            0., 0., 0., 0., 2.42301138, 17.44981519, 37.44706963, 59.86771264,
            83.6333103, 108.1344301, 132.97031768, 157.84825598, 182.53801702,
            206.84837586, 230.61437093, 253.6899679, 275.94351445, 297.25534724,
            317.5157371, 336.62422101, 354.48898098, 371.02626785, 386.16057506,
            399.82446689, 411.95866549, 422.51235541, 431.44315853, 438.71756928,
            444.31126415, 448.20948277, 450.40765545, 450.9120464, 449.74077685,
            446.92519666, 442.51191674, 436.56582757, 429.17485652, 420.45766136,
            410.57670499, 399.7619425, 388.35679371, 376.91876172, 366.48029222,
            359.54828853, 363.72218759, 368.79349031, 372.31796687, 374.28083132]), atol=1e-4))
        self.assertTrue(np.allclose(solar(0.8, speedy_coords=speedy_coords), np.array([
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
        self.assertTrue(np.allclose(solar(1.0, speedy_coords=speedy_coords), np.array([
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
        self.assertTrue(np.allclose(solar(0.6, speedy_coords, 1300), np.array([
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
        primals, f_vjp = jax.vjp(solar, 0.2, speedy_coords, 4.*solc)
        input = jnp.ones_like(primals)
        df_dtyear, df_dcoords, df_dcsol = f_vjp(input)

        self.assertFalse(jnp.any(jnp.isnan(df_dtyear)))
        
    def test_solar_gradient_check(self): 
        from jcm.physics.speedy.physical_constants import solc
        tyear = 0.2
        csol = 4.*solc

        def f(tyear, speedy_coords, csol):
            return solar(tyear, speedy_coords, csol)

        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (tyear, speedy_coords, csol), 
                                atol=None, rtol=1, eps=0.0001)
        check_jvp(f, f_jvp, args = (tyear, speedy_coords, csol), 
                                atol=None, rtol=1, eps=0.000001)
        
class TestShortWaveRadiation(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ForcingData, SurfaceFluxData, HumidityData, ConvectionData, CondensationData, SWRadiationData, DateData, PhysicsData, \
               PhysicsState, PhysicsTendency, get_clouds, get_zonal_average_fields, get_shortwave_rad_fluxes, solar, epssw, solc, parameters, forcing, terrain, speedy_coords, \
               convert_to_speedy_latitudes, TerrainData
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, CondensationData, SWRadiationData, PhysicsData
        from jcm.date import DateData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.radiation.speedy_shortwave import get_clouds, get_zonal_average_fields, get_shortwave_rad_fluxes, solar
        from jcm.physics.speedy.physical_constants import epssw, solc
        from jcm.physics.speedy.params import Parameters
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords

        parameters = Parameters.default()
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords)
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        forcing = ForcingData.zeros((ix, il))
        
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        terrain, speedy_coords = convert_to_speedy_latitudes(terrain, speedy_coords)

    # Golden values are self-generated (not from speedy.f90). They were
    # regenerated after the stratosphere-ozone migration: ozone absorption is now
    # distributed over the sigma<0.2 layers with SpeedyWeather's weight
    # 50*max(0,1/5-sigma)*dsigma instead of being placed at the fixed indices
    # k=0 (ozupp) and k=1 (ozone). The column-integrated ozone absorption is
    # preserved (so per-level dfabs is within ~2% of the old values), but the
    # vertical split changes, which shifts the surface fluxes; the values below
    # reflect the new distribution at nlev=8 (sigma<0.2 -> the top two layers).
    def test_shortwave_radiation(self):
        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa/qsat
        geopotential = 20000. * jnp.arange(7, -1, -1, dtype = float)
        se = .1*geopotential

        xy = (ix, il)
        zxy = (kx, ix, il)
        broadcast = lambda a: jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1,) + xy)
        qa, qsat, rh, geopotential, se = broadcast(qa), broadcast(qsat), broadcast(rh), broadcast(geopotential), broadcast(se)

        psa = jnp.ones(xy)
        precnv = 1.0 * np.ones(xy)
        precls = 4.0 * np.ones(xy)
        # Construct a varying iptop to catch layer-dependent effects and indexing bugs
        iptop = np.ones(xy, dtype=int) * jnp.linspace(0,kx,il).astype(int)[jnp.newaxis,:] + 1
        fmask = .7 * np.ones(xy)

        terrain_new = terrain.copy(fmask=fmask)
        terrain_new, speedy_c = convert_to_speedy_latitudes(terrain_new, speedy_coords)

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx,compute_shortwave=True)

        date_data = DateData.set_date(
            jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-08-08'))
        )

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, dt_seconds=date_data.dt_seconds, speedy_coords=speedy_c)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        # Mirror the per-step pipeline: solar geometry comes from
        # ForcingData.select(date) instead of being read off PhysicsData.date.
        forcing = ForcingData.zeros(xy).select(date_data, calendar='gregorian')
        physics_data = get_zonal_average_fields(state, physics_data, forcing, terrain_new)
        _, physics_data = get_clouds(state, physics_data, parameters, forcing, terrain_new)
        _, physics_data = get_shortwave_rad_fluxes(state, physics_data, parameters, forcing, terrain_new)
        
        # surface downward shortwave radiation at all latitudes
        self.assertTrue(np.allclose(physics_data.shortwave_rad.rsds[0, :], [
            0.0, 0.0, 0.0, 0.0, 0.934448, 6.969451, 15.64092, 25.736088, 36.90958,
            48.868553, 61.380684, 74.24382, 88.181465, 101.306694, 114.275154,
            126.950066, 139.210388, 150.950806, 163.971069, 174.513229, 184.305801,
            193.299942, 201.457474, 208.749451, 218.380264, 223.963364, 228.621826,
            232.347763, 235.136505, 236.98555, 241.906921, 241.877838, 240.900238,
            238.983093, 236.141022, 232.396622, 231.476761, 225.995895, 219.77272,
            212.92424, 205.63002, 198.183914, 191.124771, 185.738388, 185.858582,
            186.132034, 185.314651, 183.430634,
        ], atol=1e-4))

        # surface net (downward) shortwave radiation at all latitudes
        self.assertTrue(np.allclose(physics_data.shortwave_rad.rsns[0, :], [
            0.0, 0.0, 0.0, 0.0, 0.934448, 6.969451, 15.64092, 25.736088, 36.90958,
            48.868553, 61.380684, 74.24382, 88.181465, 101.306694, 114.275154,
            126.950066, 139.210388, 150.950806, 163.971069, 174.513229, 184.305801,
            193.299942, 201.457474, 208.749451, 218.380264, 223.963364, 228.621826,
            232.347763, 235.136505, 236.98555, 241.906921, 241.877838, 240.900238,
            238.983093, 236.141022, 232.396622, 231.476761, 225.995895, 219.77272,
            212.92424, 205.63002, 198.183914, 191.124771, 185.738388, 185.858582,
            186.132034, 185.314651, 183.430634
        ], atol=1e-4))

        # top of atmosphere net shortwave radiation at all latitudes
        self.assertTrue(np.allclose(physics_data.shortwave_rad.ftop[0, :], [
            0.0, 0.0, 0.0, 0.0, 1.738563, 12.500184, 27.122234, 43.247559, 60.267296,
            77.746223, 95.404114, 113.040649, 130.87735, 148.070038, 164.851227,
            181.126495, 196.810486, 211.825531, 227.594849, 241.133667, 253.802338,
            265.544373, 276.308167, 286.046936, 297.728943, 305.372406, 311.870422,
            317.19696, 321.332184, 324.263214, 330.781586, 331.297791, 330.603333,
            328.722351, 325.69104, 321.559631, 323.572296, 317.370117, 310.347107,
            302.680267, 294.631195, 286.623474, 279.444122, 275.014923, 279.192871,
            284.208099, 288.181763, 291.086945
        ], atol=1e-4))

        # column-mean absorbed shortwave flux at all latitudes
        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=0)[0, :], [
            0.0, 0.0, 0.0, 0.0, 0.100514, 0.691342, 1.435165, 2.188934, 2.919714,
            3.609709, 4.252928, 4.849604, 5.336984, 5.845419, 6.322011, 6.772051,
            7.200013, 7.609341, 7.952971, 8.327555, 8.687066, 9.030553, 9.356333,
            9.662186, 9.918588, 10.176131, 10.406075, 10.60615, 10.774457, 10.909705,
            11.109331, 11.177495, 11.212889, 11.217409, 11.193751, 11.145376, 11.511941,
            11.421775, 11.321795, 11.219502, 11.125146, 11.054942, 11.03992, 11.159565,
            11.666787, 12.259506, 12.858388, 13.457038
        ], atol=1e-4))

        # mean across all latitudes of absorbed shortwave flux at each level
        self.assertTrue(np.allclose(np.mean(physics_data.shortwave_rad.dfabs, axis=2)[:, 0], [
            3.711593, 8.028875, 14.355221, 5.895871,
            7.705781, 10.633465, 8.276639, 5.0749
        ], atol=1e-4))

    def test_output_shapes(self):
        # Ensure that the output shapes are correct
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx, dt_seconds=date_data.dt_seconds,speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy)
        # Mirror Model._get_step_fn_factory: solar geometry comes off
        # `forcing.solar`, populated by `select(date)`.
        forcing = ForcingData.zeros(xy).select(date_data, calendar='gregorian')

        new_data = get_zonal_average_fields(state, physics_data, forcing, terrain)
        
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
        physics_data = PhysicsData.zeros(xy,kx, dt_seconds=date_data.dt_seconds, speedy_coords=speedy_coords)

        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))

        forcing_now = forcing.select(date_data, calendar='gregorian')
        physics_data = get_zonal_average_fields(state, physics_data, forcing_now, terrain)

        topsr = solar(date_data.tyear(), speedy_coords=speedy_coords)
        self.assertTrue(jnp.allclose(physics_data.shortwave_rad.fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx, dt_seconds=date_data.dt_seconds, speedy_coords=speedy_coords)

        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))

        forcing_now = forcing.select(date_data, calendar='gregorian')
        physics_data = get_zonal_average_fields(state, physics_data, forcing_now, terrain)

        fs0 = 6.0
        self.assertTrue(jnp.all(physics_data.shortwave_rad.stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - physics_data.shortwave_rad.fsol, 0) == physics_data.shortwave_rad.stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        xy = (ix, il)
        zxy = (kx, ix, il)
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-04-01 12:00:00'))

        physics_data = PhysicsData.zeros(xy,kx, dt_seconds=date_data.dt_seconds, speedy_coords=speedy_coords)
        state = PhysicsState(jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(zxy), jnp.zeros(xy))
        forcing_now = forcing.select(date_data, calendar='gregorian')
        physics_data = get_zonal_average_fields(state, physics_data, forcing_now, terrain)

        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * physics_data.speedy_coords.sia**2 - 0.5
        expected_ozone = 0.4 * epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (date_data.tyear() + 10.0 / 365.0)))  + 1.8 * flat2)
        np.testing.assert_allclose(physics_data.shortwave_rad.ozone[:, 0], physics_data.shortwave_rad.fsol[:, 0] * expected_ozone[0], atol=1e-4)

    def test_random_input_consistency(self):
        xy = (ix, il)
        zxy = (kx, ix, il)
        # Provide a date that is equivalent to tyear=0.25
        date_data = DateData.set_date(model_time=jdt.to_datetime('2000-03-21'))
        physics_data = PhysicsData.zeros(xy,kx, dt_seconds=date_data.dt_seconds,speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy)
        forcing_now = forcing.select(date_data, calendar='gregorian')
        physics_data = get_zonal_average_fields(state, physics_data, forcing_now, terrain)
        
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

        date_data = DateData.set_date(
            jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-08-08'))
        )

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, dt_seconds=date_data.dt_seconds, speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)

        # Calculate gradient
        _, f_vjp = jax.vjp(get_zonal_average_fields, state, physics_data, forcing, terrain)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
        df_dstates, df_ddatas, _, _ = f_vjp(datas)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())

    def test_get_shortwave_rad_fluxes_gradients_isnan_ones(self):
        """Test that we can calculate gradients of shortwave radiation without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)
        physics_data.shortwave_rad.compute_shortwave = True

        # Calculate gradient
        _, f_vjp = jax.vjp(get_shortwave_rad_fluxes, state, physics_data, parameters, forcing, terrain)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())
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

        surface_flux = SurfaceFluxData.zeros(xy)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(xy, kx, iptop=iptop, precnv=precnv, se=se)
        condensation = CondensationData.zeros(xy, kx, precls=precls)
        sw_data = SWRadiationData.zeros(xy, kx, compute_shortwave=True)

        date_data = DateData.set_date(
            jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-08-08'))
        )

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, dt_seconds=date_data.dt_seconds, speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        forcing = ForcingData.zeros(xy)
        # Calculate gradient
        primals, f_vjp = jax.vjp(get_clouds, state, physics_data, parameters, forcing, terrain)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
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
        date_data = DateData.set_date(
            jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-08-08'))
        )
        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, dt_seconds=date_data.dt_seconds)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        _, physics_data = get_clouds(state, physics_data, parameters, forcing, terrain)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, forcing_f,terrain_f):
            data_out = get_zonal_average_fields(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.0001)

    @pytest.mark.skip(reason="finite differencing produces nans - pre-existing issue unrelated to exchange coefficients")
    def test_get_shortwave_rad_fluxes_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        """Test whether gradients are close for shortwave radiation"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)
        physics_data.shortwave_rad.compute_shortwave = True

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_shortwave_rad_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), 
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

        date_data = DateData.set_date(
            jdt.Datetime.from_pydatetime(jdt.to_datetime('2001-08-08'))
        )

        physics_data = PhysicsData.zeros(xy,kx,surface_flux=surface_flux, humidity=humidity, convection=convection, condensation=condensation, shortwave_rad=sw_data, dt_seconds=date_data.dt_seconds)
        state = PhysicsState.zeros(zxy, specific_humidity=qa, geopotential=geopotential, normalized_surface_pressure=psa)
        forcing = ForcingData.zeros(xy, fmask=fmask)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_clouds(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.000001)

class TestCloudDiagnosticsResolutionInvariance(unittest.TestCase):
    """The cloud diagnostics feeding the SW scheme are evaluated at fixed sigma
    surfaces, so on smooth analytic profiles they must agree across vertical
    grids (up to the linear-interpolation error of representing the profile on
    each grid). This is the property that keeps the stratocumulus SW feedback
    from drifting with the number of levels.
    """

    @staticmethod
    def _cloud_diagnostics(kx, ix=64, il=32):
        """Run get_clouds on analytic sigma-profiles discretized to kx levels."""
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.physics_data import (
            SurfaceFluxData, HumidityData, ConvectionData, CondensationData,
            SWRadiationData, PhysicsData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.physics.radiation.speedy_shortwave import get_clouds
        from jcm.physics.speedy.params import Parameters
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords

        parameters = Parameters.default()
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords)
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        sig = speedy_coords.fsg[:, jnp.newaxis, jnp.newaxis]

        # Smooth analytic profiles of sigma with a free-troposphere RH maximum
        # and a moist boundary layer, modulated horizontally so columns differ.
        xmod = 0.8 + 0.4 * jnp.arange(ix)[:, jnp.newaxis] / ix
        rh = xmod * (0.15 + 0.55 * jnp.exp(-((sig - 0.55) / 0.3) ** 2)
                     + 0.25 * sig ** 6)
        qsat = 25.0 * sig ** 3 + 1e-3            # [g/kg], smooth in sigma
        qa = rh * qsat
        phig = 70000.0 * (1.0 - sig) ** 1.2       # [J/kg], monotone in height
        se = 300000.0 + 0.4 * phig                # stable static-energy profile
        se, qa, qsat, rh, phig = (jnp.broadcast_to(f, (kx, ix, il))
                                  for f in (se, qa, qsat, rh, phig))

        xy = (ix, il)
        humidity = HumidityData.zeros(xy, kx, rh=rh, qsat=qsat)
        convection = ConvectionData.zeros(
            xy, kx, iptop=jnp.full(xy, kx, dtype=int),
            precnv=jnp.zeros(xy), se=se)
        condensation = CondensationData.zeros(xy, kx, precls=jnp.zeros(xy))
        sw_data = SWRadiationData.zeros(xy, kx, compute_shortwave=True)
        physics_data = PhysicsData.zeros(
            xy, kx, surface_flux=SurfaceFluxData.zeros(xy),
            humidity=humidity, convection=convection,
            condensation=condensation, shortwave_rad=sw_data,
            speedy_coords=speedy_coords)
        state = PhysicsState.zeros(
            (kx, ix, il), specific_humidity=qa, geopotential=phig,
            normalized_surface_pressure=jnp.ones(xy))

        _, data_out = get_clouds(state, physics_data, parameters,
                                 ForcingData.zeros(xy), terrain)
        return data_out.shortwave_rad

    def test_gse_cloudc_clstr_agree_across_level_counts(self):
        ref = self._cloud_diagnostics(kx=8)
        for kx in (16, 24):
            sw = self._cloud_diagnostics(kx=kx)
            # gse: static-energy gradient over a fixed sigma interval. se and
            # phig are linear in each other here, so gse is grid-independent.
            np.testing.assert_allclose(sw.gse, ref.gse, rtol=1e-4,
                                       err_msg=f"gse drifted at nlev={kx}")
            # cloudc/cloudstr: fixed-sigma RH sampling; agreement is limited
            # only by linear interpolation of the smooth RH profile.
            np.testing.assert_allclose(sw.cloudc, ref.cloudc, atol=0.05,
                                       err_msg=f"cloudc drifted at nlev={kx}")
            np.testing.assert_allclose(sw.cloudstr, ref.cloudstr, atol=0.05,
                                       err_msg=f"cloudstr drifted at nlev={kx}")
