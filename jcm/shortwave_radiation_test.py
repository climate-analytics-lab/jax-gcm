import unittest
import pytest
import jax.numpy as jnp
import numpy as np
from jax import random

from jcm.shortwave_radiation import solar, clouds, get_zonal_average_fields, get_shortwave_rad_fluxes
from jcm.physical_constants import solc, epssw
from jcm.params import il, ix, kx
from jcm.geometry import sia
from jcm.physics import SWRadiationData

# truth for test cases are generated from https://github.com/duncanwp/speedy_test

class TestSolar(unittest.TestCase):
    def test_solar(self):
        self.assertTrue(np.allclose(solar(0.2), np.array([
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
        self.assertTrue(np.allclose(solar(0.4), np.array([
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
        self.assertTrue(np.allclose(solar(0.6), np.array([
            0., 0., 0., 0., 2.42301138, 17.44981519, 37.44706963, 59.86771264,
            83.6333103, 108.1344301, 132.97031768, 157.84825598, 182.53801702,
            206.84837586, 230.61437093, 253.6899679, 275.94351445, 297.25534724,
            317.5157371, 336.62422101, 354.48898098, 371.02626785, 386.16057506,
            399.82446689, 411.95866549, 422.51235541, 431.44315853, 438.71756928,
            444.31126415, 448.20948277, 450.40765545, 450.9120464, 449.74077685,
            446.92519666, 442.51191674, 436.56582757, 429.17485652, 420.45766136,
            410.57670499, 399.7619425, 388.35679371, 376.91876172, 366.48029222,
            359.54828853, 363.72218759, 368.79349031, 372.31796687, 374.28083132]), atol=1e-4))
        self.assertTrue(np.allclose(solar(0.8), np.array([
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
        self.assertTrue(np.allclose(solar(1.0), np.array([
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
        self.assertTrue(np.allclose(solar(0.6, 1300), np.array([
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

        icltop, cloudc, clstr, qcloud = clouds(qa, rh, precnv, precls, iptop, gse, fmask)

        tyear = 0.6
        fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(tyear)

        sw_data = SWRadiationData(
            qcloud = qcloud,
            fsol = fsol,
            ozone = ozone,
            ozupp = ozupp,
            zenit = zenit,
            stratz = stratz
        )

        fsfcd, fsfc, ftop, dfabs = get_shortwave_rad_fluxes(psa, qa, icltop, cloudc, clstr, sw_data=sw_data)
        self.assertTrue(np.allclose(fsfcd[0, :], [
            0., 0., 0., 0., 1.08102491, 7.9856262, 17.54767508, 28.67351887, 40.8631746, 53.79605732,
            67.22801389, 80.95422179, 94.79448489, 108.58701854, 122.18603817, 135.46087123, 148.29548103,
            160.58828119, 172.25138545, 183.21006299, 193.40177528, 202.77492961, 211.28786499, 218.90753726,
            225.60832278, 231.37096758, 236.18142325, 240.03003908, 242.91081466, 244.82085417, 245.76014002,
            245.7315415, 244.74127921, 242.79984604, 239.92358203, 236.13704304, 231.47654032, 225.99538369,
            219.77196135, 212.92314683, 205.62864786, 198.18231101, 191.12290959, 185.73622544, 185.85603776,
            186.12903619, 185.31120169, 183.42677496
        ], atol=1e-4))

        self.assertTrue(np.allclose(fsfc[0, :], [
            0., 0., 0., 0., 1.08102491, 7.9856262, 17.54767508, 28.67351887, 40.8631746, 53.79605732,
            67.22801389, 80.95422179, 94.79448489, 108.58701854, 122.18603817, 135.46087123, 148.29548103,
            160.58828119, 172.25138545, 183.21006299, 193.40177528, 202.77492961, 211.28786499, 218.90753726,
            225.60832278, 231.37096758, 236.18142325, 240.03003908, 242.91081466, 244.82085417, 245.76014002,
            245.7315415, 244.74127921, 242.79984604, 239.92358203, 236.13704304, 231.47654032, 225.99538369,
            219.77196135, 212.92314683, 205.62864786, 198.18231101, 191.12290959, 185.73622544, 185.85603776,
            186.12903619, 185.31120169, 183.42677496
        ], atol=1e-4))

        self.assertTrue(np.allclose(ftop[0, :], [
            0., 0., 0., 0., 1.93599586, 13.84635135, 29.51685016, 46.89146027, 65.11718871, 83.73023451,
            102.44168978, 121.0533787, 139.41874296, 157.42199198, 174.96630874, 191.96679879, 208.34607385,
            224.03188495, 238.95538028, 253.05068109, 266.25471774, 278.50725748, 289.75158834, 299.9350031,
            309.0094909, 316.93257921, 323.66789347, 329.1860073, 333.46512965, 336.4917604, 338.26143437,
            338.77936105, 338.06122342, 336.13410495, 333.03776805, 328.82654748, 323.57244264, 317.37036698,
            310.3474896, 302.680774, 294.63184471, 286.62427534, 279.44504545, 275.01597523, 279.19413113,
            284.20954594, 288.1834261, 291.08877534
        ], atol=1e-4))

        self.assertTrue(np.allclose(np.mean(dfabs, axis=2)[0, :], [
            0., 0., 0., 0., 0.10687137, 0.73259064, 1.49614688, 2.27724268, 3.03175176, 3.74177215,
            4.40170949, 5.01239461, 5.57803226, 6.10437168, 6.59753382, 7.06324094, 7.5063241, 7.93045047,
            8.33799935, 8.73007726, 9.10661781, 9.46654098, 9.80796542, 10.12843323, 10.42514601, 10.69520145,
            10.93580878, 11.14449603, 11.31928937, 11.45886328, 11.56266179, 11.63097744, 11.66499303,
            11.66678236, 11.63927325, 11.58618806, 11.51198779, 11.42187291, 11.32194103, 11.2197034,
            11.12539961, 11.05524554, 11.04026698, 11.15996872, 11.66726167, 12.26006372, 12.85902805,
            13.45775005
        ], atol=1e-4))

        self.assertTrue(np.allclose(np.mean(dfabs, axis=1)[0, :], [
            3.82887045, 7.81598669, 14.17718547, 5.65627818, 7.80939064, 12.48949685, 8.5056334, 5.21519786,
        ], atol=1e-4))
        
    # def test_shortwave_radiation_general(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = jnp.ones((ix,il))*1
    #     qa = jnp.ones((ix, il, kx))*1
    #     cloudc = jnp.ones((ix,il))*1
    #     clstr = jnp.ones((ix,il))*1
    #     psa = jnp.ones((ix,il))*1

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     dummy_fdabs = jnp.ones((kx))

    #     self.assertIsNone(fsfcd)
    #     self.assertIsNone(fsfc)
    #     self.assertIsNone(ftop)
    #     self.assertIsNone(fdabs)

    #     self.assertEqual(fsfcd.shape,icltop.shape)
    #     self.assertEqual(fsfc.shape,icltop.shape)
    #     self.assertEqual(ftop.shape,icltop.shape)
    #     self.assertEqual(fdabs.shape,dummy_fdabs.shape)

    # def test_shortwave_radiation_case1(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = jnp.ones((ix,il))*1
    #     qa = jnp.ones((ix, il, kx))*1
    #     cloudc = jnp.ones((ix,il))*1
    #     clstr = jnp.ones((ix,il))*1
    #     psa = jnp.ones((ix,il))*1

    #     fsfcd_true = 0.0
    #     fsfc_true = 0.0
    #     ftop_true = -0.4300000071525574
    #     fdabs_true = jnp.ones((kx))*0

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     self.assertAlmostEqual(fsfcd,fsfcd_true)
    #     self.assertAlmostEqual(fsfc,fsfc_true)
    #     self.assertAlmostEqual(ftop,ftop_true)
    #     self.assertAlmostEqual(fdabs,fdabs_true)

    # def test_shortwave_radiation_case2(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = jnp.ones((ix,il))*1
    #     qa = jnp.ones((ix, il, kx))*5
    #     cloudc = jnp.ones((ix,il))*1.56
    #     clstr = jnp.ones((ix,il))*132.5
    #     psa = jnp.ones((ix,il))*3

    #     fsfcd_true = 0.0
    #     fsfc_true = 0.0
    #     ftop_true = -0.6708000111579895
    #     fdabs_true = jnp.ones((kx))*0

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     self.assertAlmostEqual(fsfcd,fsfcd_true)
    #     self.assertAlmostEqual(fsfc,fsfc_true)
    #     self.assertAlmostEqual(ftop,ftop_true)
    #     self.assertAlmostEqual(fdabs,fdabs_true)

    # def test_shortwave_radiation_case3(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = jnp.ones((ix,il))*1
    #     qa = jnp.ones((ix, il, kx))*2.3
    #     cloudc = jnp.ones((ix,il))*0.00001
    #     clstr = jnp.ones((ix,il))*1
    #     psa = jnp.ones((ix,il))*2345

    #     fsfcd_true = 0.0
    #     fsfc_true = 0.0
    #     ftop_true = -4.300000071525574e-06
    #     fdabs_true = jnp.ones((kx))*0

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     self.assertAlmostEqual(fsfcd,fsfcd_true)
    #     self.assertAlmostEqual(fsfc,fsfc_true)
    #     self.assertAlmostEqual(ftop,ftop_true)
    #     self.assertAlmostEqual(fdabs,fdabs_true)

    # def test_shortwave_radiation_case4(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = jnp.ones((ix,il))*0
    #     qa = jnp.ones((ix, il, kx))*0
    #     cloudc = jnp.ones((ix,il))*0
    #     clstr = jnp.ones((ix,il))*0
    #     psa = jnp.ones((ix,il))*0

    #     fsfcd_true = 0.0
    #     fsfc_true = 0.0
    #     ftop_true = 0.0
    #     fdabs_true = jnp.ones((kx))*0

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     self.assertAlmostEqual(fsfcd,fsfcd_true)
    #     self.assertAlmostEqual(fsfc,fsfc_true)
    #     self.assertAlmostEqual(ftop,ftop_true)
    #     self.assertAlmostEqual(fdabs,fdabs_true)

    # def test_shortwave_radiation_case5(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = np.ones((ix,il))*1
    #     qa = np.ones((ix, il, kx))*0.2
    #     cloudc = np.ones((ix,il))*0.3
    #     clstr = np.ones((ix,il))*0.1
    #     psa = np.ones((ix,il))*.005

    #     fsfcd_true = 0.0
    #     fsfc_true = 0.0
    #     ftop_true = -0.1290000021457672
    #     fdabs_true = jnp.ones((kx))*0

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     self.assertAlmostEqual(fsfcd,fsfcd_true)
    #     self.assertAlmostEqual(fsfc,fsfc_true)
    #     self.assertAlmostEqual(ftop,ftop_true)
    #     self.assertAlmostEqual(fdabs,fdabs_true)

    # def test_shortwave_radiation_case6(self):
    #     ix, il, kx = 1, 1, 8

    #     icltop = jnp.ones((ix,il))*2
    #     qa = jnp.ones((ix, il, kx))*1
    #     cloudc = jnp.ones((ix,il))*0.0005
    #     clstr = jnp.ones((ix,il))*0.00006
    #     psa = jnp.ones((ix,il))*0.0001

    #     fsfcd_true = 0.0
    #     fsfc_true = 0.0
    #     ftop_true = -0.0002150000035762787
    #     fdabs_true = jnp.ones((kx))*0

    #     fsfcd, fsfc, ftop, fdabs = get_shortwave_rad_fluxes(psa,qa,icltop,cloudc,clstr)

    #     self.assertAlmostEqual(fsfcd,fsfcd_true)
    #     self.assertAlmostEqual(fsfc,fsfc_true)
    #     self.assertAlmostEqual(ftop,ftop_true)
    #     self.assertAlmostEqual(fdabs,fdabs_true)

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

    def test_clouds_case7(self):
        qa = 0.5 * 1000. * jnp.array([0., 0.00035438, 0.00347954, 0.00472337, 0.00700214,0.01416442,0.01782708, 0.0216505])
        qsat = 1000. * jnp.array([0., 0.00037303, 0.00366268, 0.00787228, 0.01167024, 0.01490992, 0.01876534, 0.02279])
        rh = qa / qsat

        broadcast = lambda a: jnp.tile(a[None, :], (ix, il, 1))
        xy = (ix, il)

        precnv = jnp.ones(xy)
        precls = jnp.ones(xy)
        iptop = jnp.ones(xy) * 5
        gse = jnp.ones(xy)
        fmask = jnp.ones(xy)

        icltop, cloudc, clstr, qcloud = clouds(
            broadcast(qa),
            broadcast(rh),
            precnv,
            precls,
            iptop,
            gse,
            fmask
        )

        self.assertAlmostEqual(icltop[0, 0], 5)
        self.assertAlmostEqual(cloudc[0, 0], 0.69495568, places=4)
        self.assertAlmostEqual(clstr[0, 0], 0.07125001, places=4)

class TestZonalAverageFields(unittest.TestCase):
    def test_zonal_average_fields(self):
        zonal_avg_fields_test_cases = [
            (
                0.,
                (
                    [553.93423522, 551.02920315, 545.81299099, 538.30748186, 528.544079,
                    516.56380499, 506.85182399, 506.40751184, 508.57360094, 511.41451804,
                    514.02259444, 515.87726025, 516.6503729,  516.12421359, 514.15095764,
                    510.63050654, 505.49750448, 498.71321714, 490.26041534, 480.13978781,
                    468.36747152, 454.97348403, 440.0002743,  423.50188934, 405.54332065,
                    386.19977488, 365.55671103, 343.70947479, 320.76336536, 296.83415891,
                    272.04837186, 246.54472938, 220.47604007, 194.01174187, 167.34305152,
                    140.69016017, 114.31490282,  88.54239741,  63.80107855,  40.70440366,
                    20.24489631,   4.43498509,   0.,           0.,           0.,
                    0.,           0.,           0.],
                    [7.2976634,  6.92159552, 6.56751671, 6.23531793, 5.92359736, 5.62990397,
                    5.398223,   5.2947514,  5.24167706, 5.21473479, 5.20128668, 5.19301576,
                    5.18376882, 5.16873203, 5.14409293, 5.1069047,  5.05503364, 4.98713219,
                    4.90261001, 4.80158453, 4.68480551, 4.5535532,  4.40950605, 4.25458879,
                    4.0908038,  3.92005211, 3.74396128, 3.56371943, 3.37993336, 3.19252081,
                    3.00064098, 2.80268486, 2.59632838, 2.37866142, 2.146421,   1.89634433,
                    1.62570389, 1.33312115, 1.01988415, 0.69232674, 0.36693739, 0.08573562,
                    0.,         0.,         0.,         0.,         0.,         0.],
                    [10.56453547,  9.89266036,  9.1713976,   8.4176895,   7.64574606,  6.8674837,
                    6.14690587,  5.5538979,   4.9920504,   4.43739516,  3.88368477,  3.33262033,
                    2.79061919,  2.26716297,  1.77360326,  1.32209724,  0.92459193,  0.59186027,
                    0.332664,    0.1530815,   0.05607367,  0.04130872,  0.1052407,   0.24141559,
                    0.44093753,  0.69301942,  0.98551668,  1.30537204,  1.63889602,  1.97184665,
                    2.28932532,  2.57552098,  2.8134176,   2.9846069,   3.06938095,  3.04734991,
                    2.89886392,  2.60765791,  2.16544061,  1.58005528,  0.89195273,  0.22006198,
                    0.,          0.,          0.,          0.,          0.,          0.],
                    [1.31742418, 1.25612139, 1.20325404, 1.15831904, 1.1207386,  1.08987584,
                    1.06504956, 1.04555154, 1.03066246, 1.01966892, 1.01187902, 1.00663787,
                    1.00334176, 1.0014512,  1.00050247, 1.00011745, 1.00001161, 1.00000003,
                    1.00000122, 1.0000389,  1.00024146, 1.00083927, 1.00215986, 1.00462099,
                    1.0087218,  1.01503223, 1.02418072, 1.03684064, 1.05371554, 1.07552341,
                    1.10298071, 1.13678558, 1.17760118, 1.22603995, 1.28264725, 1.34788699,
                    1.42212772, 1.50563032, 1.59853751, 1.70086451, 1.81249333, 1.93316593,
                    2.06248469, 2.19990897, 2.34476065, 2.49622507, 2.65336251, 2.81511494],
                    [0.,         0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,         1.56501491,
                    6.,         6.,         6.,         6.,         6.,         6.]
                )
            ),
            (
                0.5,
                (
                    [ 0.,           0.,           0.,           0.,           0.,
                    0.,           4.08483387,  18.81602909,  37.904217,    59.46198854,
                    82.55993822, 106.62399298, 131.25338415, 156.14407514, 181.05126155,
                    205.76910765, 230.11888489, 253.94223501, 277.09635047, 299.45159043,
                    320.88972577, 341.30256481, 360.59170582, 378.66804698, 395.45172342,
                    410.87244868, 424.86947997, 437.39230003, 448.40122847, 457.86820141,
                    465.77812731, 472.13035389, 476.9409792,  480.24620882, 482.10714451,
                    482.61738607, 481.91518644, 480.20420937, 477.7914928,  475.16429924,
                    473.17273067, 473.63236889, 482.76897295, 493.96546896, 503.0901268,
                    510.10460768, 514.97956284, 517.69454081],
                    [ 0.,          0.,          0.,          0.,          0.,
                    0.,          0.07896661, 0.34103926, 0.64469936, 0.95052216,
                    1.24304743, 1.51632932, 1.76914724, 2.00277763, 2.21976075,
                    2.42313937, 2.61595823, 2.80093379, 2.98023605, 3.15536788,
                    3.327115,   3.49555499, 3.66012196, 3.81970706, 3.97279094,
                    4.11759867, 4.25226051, 4.37497901, 4.4841866,  4.57868748,
                    4.65778129, 4.72135827, 4.76996986, 4.80487509, 4.82806769,
                    4.84230166, 4.85114067, 4.85908555, 4.87189126, 4.89733994,
                    4.94726465, 5.04441936, 5.26158228, 5.53606156, 5.82738862,
                    6.13785418, 6.46876831, 6.82023295],
                    [ 0.,          0.,          0.,          0.,          0.,
                    0.,          0.14630309, 0.59336529, 1.04267738, 1.41350067,
                    1.67981229, 1.8384692,  1.89767413, 1.87122507, 1.77528326,
                    1.62653011, 1.44120066, 1.23468369, 1.02147612, 0.81530208,
                    0.62926409, 0.47591996, 0.36721106, 0.31422802, 0.32682146,
                    0.41310858, 0.57894545, 0.82744876, 1.1586499,  1.56934545,
                    2.05320333, 2.60112129, 3.20183795, 3.84276292, 4.5109358,
                    5.19409033, 5.88175178, 6.56640401, 7.24488267, 7.92050671,
                    8.60761631, 9.34590049, 10.28091192, 11.30402195, 12.32515566,
                    13.33254324, 14.3112653,  15.24258093],
                    [ 2.81511492, 2.65336249, 2.49622506, 2.34476064, 2.19990895,
                    2.06248468, 1.93316592, 1.81249332, 1.7008645,  1.5985375,
                    1.50563031, 1.42212771, 1.34788699, 1.28264724, 1.22603995,
                    1.17760117, 1.13678557, 1.1029807,  1.07552341, 1.05371554,
                    1.03684064, 1.02418072, 1.01503223, 1.0087218,  1.00462099,
                    1.00215986, 1.00083927, 1.00024146, 1.0000389,  1.00000122,
                    1.00000003, 1.00001161, 1.00011745, 1.00050247, 1.0014512,
                    1.00334176, 1.00663787, 1.01187902, 1.01966892, 1.03066246,
                    1.04555154, 1.06504956, 1.08987584, 1.1207386,  1.15831905,
                    1.20325404, 1.25612139, 1.31742419],
                    [ 6.,         6.,         6.,         6.,         6.,
                    6.,         1.91516613, 0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.,         0.,         0.,
                    0.,         0.,         0.]
                )
            ),
        ]
        for tyear, expected in zonal_avg_fields_test_cases:
            fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(tyear)
            self.assertTrue(np.allclose(fsol[0, :], expected[0], atol=1e-4))
            self.assertTrue(np.allclose(ozupp[0, :], expected[1], atol=1e-4))
            self.assertTrue(np.allclose(ozone[0, :], expected[2], atol=1e-4))
            self.assertTrue(np.allclose(zenit[0, :], expected[3], atol=1e-4))
            self.assertTrue(np.allclose(stratz[0, :], expected[4], atol=1e-4))

    # def test_polar_night_cooling(self):
    #     # Ensure polar night cooling behaves correctly
    #     fsol, ozupp, ozone, zenit, stratz, = get_zonal_average_fields(
    #         self.tyear
    #     )
        
    #     fs0 = 6.0
    #     self.assertTrue(jnp.all(stratz >= 0))
    #     self.assertTrue(jnp.all(jnp.maximum(fs0 - fsol, 0) == stratz))

    # def test_ozone_absorption(self):
    #     # Check that ozone absorption is being calculated correctly
    #     fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(
    #         self.tyear
    #     )
        
    #     # Expected form for ozone based on the provided formula
    #     flat2 = 1.5 * sia**2 - 0.5
    #     expected_ozone = 0.4 * self.epssw * (1.0 + jnp.maximum(0.0, jnp.cos(4.0 * jnp.arcsin(1.0) * (self.tyear + 10.0 / 365.0)))  + 1.8 * flat2)
    #     print
    #     self.assertTrue(jnp.allclose(ozone[:, 0], fsol[:, 0] * expected_ozone[0]))

    def test_random_input_consistency(self):
        # Check that random inputs produce consistent outputs
        key = random.PRNGKey(0)
        tyear = random.uniform(key, shape=(), minval=0, maxval=1)

        fsol, ozupp, ozone, zenit, stratz = get_zonal_average_fields(tyear)
        
        # Ensure outputs are consistent and within expected ranges
        self.assertTrue(jnp.all(fsol >= 0))
        self.assertTrue(jnp.all(ozupp >= 0))
        self.assertTrue(jnp.all(ozone >= 0))
        self.assertTrue(jnp.all(stratz >= 0))
        self.assertTrue(jnp.all(zenit >= 0))