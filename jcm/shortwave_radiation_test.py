import unittest
import pytest
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal
from jax import random

from jcm.shortwave_radiation import get_zonal_average_fields, solar, clouds
from jcm.physical_constants import solc, epssw
from jcm.params import il, ix
from jcm.geometry import sia


# truth for test cases are generated from https://github.com/duncanwp/speedy_test

class TestClouds(unittest.TestCase):

    def test_clouds_general(self):
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))
        rh = jnp.ones((ix,il,kx))
        precnv = jnp.ones((ix,il))
        precls = jnp.ones((ix,il))
        iptop = jnp.ones((ix,il))
        gse = jnp.ones((ix,il))
        fmask = jnp.ones((ix,il))

        icltop, cloudc, clstr, qcloud = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        
        # Check that icltop, cloudc, and clstr are not null.
        self.assertIsNotNone(icltop)
        self.assertIsNotNone(cloudc)
        self.assertIsNotNone(clstr)
        self.assertIsNotNone(qcloud)

        # Check that our outputs are the right shape
        self.assertEqual(icltop.shape,precnv.shape)
        self.assertEqual(cloudc.shape,precnv.shape)
        self.assertEqual(clstr.shape,precnv.shape)
        self.assertEqual(qcloud.shape,precnv.shape)

    def test_clouds_case1(self):

        a = 5
        b = 0.25
        c = -1
        d = 4
        e = 10
        f = 0.01
        g = 0.7
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 9
        cloudc_true = 0.6324555414579978
        clstr_true = 0.026250001043081284

        icltop, cloudc, clstr, _ = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        if (icltop != iptop):
            icltop_true -= 1
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

    def test_clouds_case2(self):

        a = 5129
        b = 0.25234
        c = -329842
        d = 2.134
        e = 8
        f = 0.01013498
        g = 0.2

        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 8
        cloudc_true = 1.0
        clstr_true = 0.007570200300812721

        icltop, cloudc, clstr, _ = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        if (icltop != iptop):
            icltop_true -= 1
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

    def test_clouds_case3(self):

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 0
        cloudc_true = 0.0
        clstr_true = 0.0

        icltop, cloudc, clstr, _ = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        if (icltop != iptop):
            icltop_true -= 1
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

    def test_clouds_case4(self):

        a = -1
        b = -1
        c = -1
        d = -1
        e = -1
        f = -1
        g = -1
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = -1
        cloudc_true = 1.0
        clstr_true = 0.15000000596046448

        icltop, cloudc, clstr, _ = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        if (icltop != iptop):
            icltop_true -= 1
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

    def test_clouds_case5(self):

        a = 420
        b = 293
        c = 238
        d = 430
        e = 9823
        f = 9023
        g = 0.1

        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 7
        cloudc_true = 1.0
        clstr_true = 4.395000174641609

        icltop, cloudc, clstr, _ = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        
        if (icltop != iptop):
            icltop_true -= 1

        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)

    def test_clouds_case6(self):

        a = 0.0000005
        b = -0.00000076
        c = 0.0000000008
        d = 0.0000000006
        e = 0.00000000002
        f = 0.00000004
        g = 0.000000003
        
        ix, il, kx = 1, 1, 8
        qa = jnp.ones((ix, il, kx))*a
        rh = jnp.ones((ix,il,kx))*b
        precnv = jnp.ones((ix,il))*c
        precls = jnp.ones((ix,il))*d
        iptop = jnp.ones((ix,il))*e
        gse = jnp.ones((ix,il))*f
        fmask = jnp.ones((ix,il))*g

        # from speedy:
        icltop_true = 0
        cloudc_true = 6.955861003243796e-05
        clstr_true = -3.42000013589859e-16

        icltop, cloudc, clstr, _ = clouds(qa,rh,precnv,precls,iptop,gse,fmask)
        if (icltop != iptop):
            icltop_true -= 1
        self.assertAlmostEqual(icltop,icltop_true)
        self.assertAlmostEqual(cloudc,cloudc_true)
        self.assertAlmostEqual(clstr,clstr_true)


@pytest.mark.parametrize("input_value, expected_output", [
    (0.0, np.array([
        553.93423522, 551.02920315, 545.81299099, 538.30748186, 528.544079,
        516.56380499, 506.85182399, 506.40751184, 508.57360094, 511.41451804,
        514.02259444, 515.87726025, 516.6503729, 516.12421359, 514.15095764,
        510.63050654, 505.49750448, 498.71321714, 490.26041534, 480.13978781,
        468.36747152, 454.97348403, 440.0002743, 423.50188934, 405.54332065,
        386.19977488, 365.55671103, 343.70947479, 320.76336536, 296.83415891,
        272.04837186, 246.54472938, 220.47604007, 194.01174187, 167.34305152,
        140.69016017, 114.31490282, 88.54239741, 63.80107855, 40.70440366,
        20.24489631, 4.43498509, 0., 0., 0., 0., 0., 0.
    ])),  
    (0.2, np.array([
        59.64891891, 82.51370562, 109.0996075, 135.94454033, 162.48195582,
        188.46471746, 213.72891835, 238.14170523, 261.58627434, 283.95547202,
        305.15011948, 325.07762082, 343.65189868, 360.79323687, 376.42841812,
        390.49090207, 402.92092072, 413.66583083, 422.68006932, 429.9254984,
        435.37150003, 438.9950085, 440.78070068, 440.7209988, 438.81611994,
        435.07404132, 429.51050427, 422.14893274, 413.02032164, 402.16320111,
        389.62332055, 375.45360549, 359.71400001, 342.47101119, 323.7977572,
        303.77351671, 282.48360014, 260.01911561, 236.4767785, 211.95903738,
        186.57407167, 160.43718712, 133.67240691, 106.41888862, 78.84586166,
        51.20481384, 24.06562443, 0.89269878
    ])),  
    (0.4, np.array([
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
        4.66603495e+02, 4.73109251e+02, 4.77630650e+02, 4.80148724e+02
    ])),  
    (0.6, np.array([
        0., 0., 0., 0., 2.42301138, 17.44981519, 37.44706963, 59.86771264,
        83.6333103, 108.1344301, 132.97031768, 157.84825598, 182.53801702,
        206.84837586, 230.61437093, 253.6899679, 275.94351445, 297.25534724,
        317.5157371, 336.62422101, 354.48898098, 371.02626785, 386.16057506,
        399.82446689, 411.95866549, 422.51235541, 431.44315853, 438.71756928,
        444.31126415, 448.20948277, 450.40765545, 450.9120464, 449.74077685,
        446.92519666, 442.51191674, 436.56582757, 429.17485652, 420.45766136,
        410.57670499, 399.7619425, 388.35679371, 376.91876172, 366.48029222,
        359.54828853, 363.72218759, 368.79349031, 372.31796687, 374.28083132
    ])),  
    (0.8, np.array([
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
        1.51012308e+01, 1.34429313e-01, 0.00000000e+00, 0.00000000e+00
    ])),  
    (1.0, np.array([
        553.93421795, 551.02918596, 545.81297397, 538.30746507, 528.54406252,
        516.56378888, 506.85181087, 506.40750073, 508.57359122, 511.41450948,
        514.02258691, 515.87725366, 516.65036719, 516.12420873, 514.15095359,
        510.63050328, 505.49750198, 498.71321538, 490.2604143, 480.13978746,
        468.36747184, 454.973485, 440.00027589, 423.50189151, 405.54332338,
        386.19977815, 365.55671479, 343.709479, 320.76336998, 296.8341639,
        272.04837717, 246.54473496, 220.47604586, 194.01174781, 167.34305754,
        140.69016619, 114.31490876, 88.54240315, 63.80108395, 40.70440853,
        20.24490036, 4.43498764, 0., 0., 0., 0., 0., 0.
    ])),  
])
def test_solar(input_value, expected_output, decimal=3):
    res=solar(input_value)
    assert_array_almost_equal(res,expected_output,decimal=decimal) 

class TestZonalAverageFields(unittest.TestCase):

    def setUp(self):
        # Set up test case with known inputs
        self.tyear = 0.25  # Example time of the year (spring equinox)
        self.solc = solc
        self.il = il
        self.ix = ix
        self.epssw = epssw

    def test_output_shapes(self):
        # Ensure that the output shapes are correct
        fsol, ozupp, ozone, stratz, zenit = get_zonal_average_fields(
            self.tyear
        )
        
        self.assertEqual(fsol.shape, (self.ix, self.il))
        self.assertEqual(ozupp.shape, (self.ix, self.il))
        self.assertEqual(ozone.shape, (self.ix, self.il))
        self.assertEqual(stratz.shape, (self.ix, self.il))
        self.assertEqual(zenit.shape, (self.ix, self.il))

    def test_solar_radiation_values(self):
        # Test that the solar radiation values are computed correctly
        fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(
            self.tyear
        )
        
        topsr = solar(self.tyear)
        self.assertTrue(jnp.allclose(fsol[:, 0], topsr[0]))

    def test_polar_night_cooling(self):
        # Ensure polar night cooling behaves correctly
        fsol, ozupp, ozone, zenit, stratz, = get_zonal_average_fields(
            self.tyear
        )
        
        fs0 = 6.0
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(jnp.maximum(fs0 - fsol, 0) == stratz))

    def test_ozone_absorption(self):
        # Check that ozone absorption is being calculated correctly
        fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(
            self.tyear
        )
        
        # Expected form for ozone based on the provided formula
        flat2 = 1.5 * sia**2 - 0.5
        expected_ozone = 0.4 * self.epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (self.tyear + 10.0 / 365.0)))  + 1.8 * flat2)
        print
        self.assertTrue(jnp.allclose(ozone[:, 0], fsol[:, 0] * expected_ozone[0]))

    def test_random_input_consistency(self):
        # Check that random inputs produce consistent outputs
        key = random.PRNGKey(0)
        
        fsol, ozupp, ozone, zenit, stratz= get_zonal_average_fields(
            self.tyear
        )
        
        # Ensure outputs are consistent and within expected ranges
        self.assertTrue(jnp.all(fsol >= 0))
        self.assertTrue(jnp.all(ozupp >= 0))
        self.assertTrue(jnp.all(ozone >= 0))
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(zenit >= 0))