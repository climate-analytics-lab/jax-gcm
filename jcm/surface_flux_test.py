import unittest
from jcm.surface_flux import get_surface_fluxes, set_orog_land_sfc_drag
import jax.numpy as jnp
import csv

class TestSurfaceFluxesUnit(unittest.TestCase):
    def test_updated_surface_flux(self):
        xy, xyz = (96, 48), (96, 48, 8)
        psa = jnp.ones(xy)
        ua = jnp.ones(xyz)
        va = jnp.ones(xyz)
        ta = jnp.ones(xyz) * 290
        qa = jnp.ones(xyz)
        rh = jnp.ones(xyz) * 0.5   
        phi = jnp.ones(xyz) * (jnp.arange(8))[None, None, ::-1]
        phi0 = jnp.zeros(xy)
        fmask = 0.5 * jnp.ones(xy)
        tsea = jnp.ones(xy) * 292
        ssrd = 400 * jnp.ones(xy)
        slrd = 400 * jnp.ones(xy)
        lfluxland = True

        ustr, vstr, shf, evap, slru, hfluxn, tsfc, tskin, u0, v0, t0 = get_surface_fluxes(psa, ua, va, ta, qa, rh, phi, phi0, fmask, tsea, ssrd, slrd, lfluxland)
        
        self.assertTrue(jnp.allclose(ustr[0, 0, :], jnp.array([-0.01493673, -0.00900353, -0.01197013]), atol=1e-4))
        self.assertTrue(jnp.allclose(vstr[0, 0, :], jnp.array([-0.01493673, -0.00900353, -0.01197013]), atol=1e-4))
        self.assertTrue(jnp.allclose(shf[0, 0, :], jnp.array([81.73508, 16.271175, 49.003124]), atol=1e-4))
        self.assertTrue(jnp.allclose(evap[0, 0, :], jnp.array([0.06291558, 0.10244954, 0.08268256]), atol=1e-4))
        self.assertTrue(jnp.allclose(slru[0, 0, :], jnp.array([459.7182, 403.96204, 431.84012]), atol=1e-4))
        self.assertTrue(jnp.allclose(hfluxn[0, 0, :], jnp.array([101.19495, 668.53546]), atol=1e-4))
        self.assertTrue(jnp.isclose(tsfc[0, 0], 290.0, atol=1e-4))
        self.assertTrue(jnp.isclose(tskin[0, 0], 297.22821044921875, atol=1e-4))
        self.assertTrue(jnp.isclose(u0[0, 0], 0.949999988079071, atol=1e-4))
        self.assertTrue(jnp.isclose(v0[0, 0], 0.949999988079071, atol=1e-4))
        self.assertTrue(jnp.isclose(t0[0, 0], 290.0, atol=1e-4))

    def test_surface_fluxes_test1(self):
        il, ix, kx = 96, 48, 8
        psa = jnp.ones((il, ix)) #surface pressure
        ua = jnp.ones(((il, ix, kx))) #zonal wind
        va = jnp.ones(((il, ix, kx))) #meridional wind
        ta = 288. * jnp.ones(((il, ix, kx))) #temperature
        qa = 5. * jnp.ones(((il, ix, kx))) #temperature
        rh = 0.8 * jnp.ones(((il, ix, kx))) #relative humidity
        phi = 5000. * jnp.ones(((il, ix, kx))) #geopotential
        phi0 = 500. * jnp.ones((il, ix)) #surface geopotential
        fmask = 0.5 * jnp.ones((il, ix)) #land fraction mask
        tsea = 290. * jnp.ones((il, ix)) #ssts
        ssrd = 400. * jnp.ones((il, ix)) #surface downward shortwave
        slrd = 400. * jnp.ones((il, ix)) #surface downward longwave
        lfluxland=True

        test_data = jnp.array([[-4.18139994e-03,-4.18139994e-03, 1.08220810e+02, 4.80042472e-02,
            4.87866394e+02, 4.80595490e+02, 2.89000000e+02, 2.98854797e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-1.50404554e-02,-1.50404554e-02, 7.55662489e+00, 2.64080837e-02,
            3.93007751e+02, 1.06054558e+02, 2.89000000e+02, 2.96575317e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-9.61105898e-03,-9.61105898e-03, 5.54379463e+01, 3.52742635e-02,
            4.32339783e+02, 2.97601044e+02, 2.89000000e+02, 2.97186432e+02,
            9.50001001e-01, 9.50001001e-01, 2.88000000e+02]])
            
        vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=1e-5
        ))

    def test_surface_fluxes_test2(self):
        il, ix, kx = 96, 48, 8
        psa = jnp.ones((il, ix)) #surface pressure
        ua = jnp.ones(((il, ix, kx))) #zonal wind
        va = jnp.ones(((il, ix, kx))) #meridional wind
        ta = 288. * jnp.ones(((il, ix, kx))) #temperature
        qa = 5. * jnp.ones(((il, ix, kx))) #temperature
        rh = 0.8 * jnp.ones(((il, ix, kx))) #relative humidity
        phi = 5000. * jnp.ones(((il, ix, kx))) #geopotential
        phi0 = 500. * jnp.ones((il, ix)) #surface geopotential
        fmask = 0.5 * jnp.ones((il, ix)) #land fraction mask
        tsea = 290. * jnp.ones((il, ix)) #ssts
        ssrd = 400. * jnp.ones((il, ix)) #surface downward shortwave
        slrd = 400. * jnp.ones((il, ix)) #surface downward longwave
        lfluxland=True

        test_data = jnp.array([[-4.18139994e-03,-4.18139994e-03, 1.08220810e+02, 4.80042472e-02,
            4.87866394e+02, 4.80595490e+02, 2.89000000e+02, 2.98854797e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-1.50404749e-02,-1.50404749e-02, 7.55662489e+00, 2.64080837e-02,
            3.93007751e+02, 1.06054558e+02, 2.89000000e+02, 2.96575317e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-9.61106271e-03,-9.61106271e-03, 5.54379463e+01, 3.52742635e-02,
            4.32339783e+02, 2.97601044e+02, 2.89000000e+02, 2.97186432e+02,
            9.50001001e-01, 9.50001001e-01, 2.88000000e+02]])
            
        vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=1e-5
        ))

    def test_surface_fluxes_test3(self):
        il, ix, kx = 96, 48, 8
        psa = jnp.ones((il, ix)) #surface pressure
        ua = jnp.ones(((il, ix, kx))) #zonal wind
        va = jnp.ones(((il, ix, kx))) #meridional wind
        ta = 288. * jnp.ones(((il, ix, kx))) #temperature
        qa = 5. * jnp.ones(((il, ix, kx))) #temperature
        rh = 0.8 * jnp.ones(((il, ix, kx))) #relative humidity
        phi = 5000. * jnp.ones(((il, ix, kx))) #geopotential
        phi0 = -10. * jnp.ones((il, ix)) #surface geopotential
        fmask = 0.5 * jnp.ones((il, ix)) #land fraction mask
        tsea = 290. * jnp.ones((il, ix)) #ssts
        ssrd = 400. * jnp.ones((il, ix)) #surface downward shortwave
        slrd = 400. * jnp.ones((il, ix)) #surface downward longwave
        lfluxland=True

        test_data = jnp.array([[-4.18139994e-03,-4.18139994e-03, 1.05182373e+02, 4.66440842e-02,
            4.92244934e+02, 4.80595490e+02, 2.89000000e+02, 2.99263367e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-1.50404554e-02,-1.50404554e-02, 7.55662489e+00, 2.64080837e-02,
            3.93007751e+02, 1.09651382e+02, 2.89000000e+02, 2.96832245e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-9.61105898e-03,-9.61105898e-03, 5.36600761e+01, 3.45076099e-02,
            4.33961243e+02, 2.99674957e+02, 2.89000000e+02, 2.97482452e+02,
            9.50001001e-01, 9.50001001e-01, 2.88000000e+02]])
    
        vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=1e-5
        ))

    def test_surface_fluxes_test4(self):
        il, ix, kx = 96, 48, 8
        psa = jnp.ones((il, ix)) #surface pressure
        ua = jnp.ones(((il, ix, kx))) #zonal wind
        va = jnp.ones(((il, ix, kx))) #meridional wind
        ta = 300. * jnp.ones(((il, ix, kx))) #temperature
        qa = 5. * jnp.ones(((il, ix, kx))) #temperature
        rh = 0.8 * jnp.ones(((il, ix, kx))) #relative humidity
        phi = 5000. * jnp.ones(((il, ix, kx))) #geopotential
        phi0 = 500. * jnp.ones((il, ix)) #surface geopotential
        fmask = 0.5 * jnp.ones((il, ix)) #land fraction mask
        tsea = 290. * jnp.ones((il, ix)) #ssts
        ssrd = 400. * jnp.ones((il, ix)) #surface downward shortwave
        slrd = 400. * jnp.ones((il, ix)) #surface downward longwave
        lfluxland=True

        test_data = jnp.array([[-1.98534015e-03,-1.98534015e-03, 3.40381584e+01, 2.68966686e-02,
            5.22806824e+02, 4.20411835e+02, 2.89000000e+02, 3.02115173e+02,
            9.49999988e-01, 9.49999988e-01, 3.00000000e+02],
            [-1.44388555e-02,-1.44388555e-02,-1.79395313e+01, 1.25386305e-02,
            3.93007751e+02, 1.78033081e+02, 2.89000000e+02, 3.01716644e+02,
            9.49999988e-01, 9.49999988e-01, 3.00000000e+02],
            [-8.21213890e-03,-8.21213890e-03, 7.37131262e+00, 1.92672648e-02,
            4.57831177e+02, 3.00025909e+02, 2.89000000e+02, 3.01831757e+02,
            9.50001001e-01, 9.50001001e-01, 3.00000000e+02]])

        vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=1e-5
        ))

    def test_surface_fluxes_test5(self):
        il, ix, kx = 96, 48, 8
        psa = jnp.ones((il, ix)) #surface pressure
        ua = jnp.ones(((il, ix, kx))) #zonal wind
        va = jnp.ones(((il, ix, kx))) #meridional wind
        ta = 285. * jnp.ones(((il, ix, kx))) #temperature
        qa = 5. * jnp.ones(((il, ix, kx))) #temperature
        rh = 0.8 * jnp.ones(((il, ix, kx))) #relative humidity
        phi = 5000. * jnp.ones(((il, ix, kx))) #geopotential
        phi0 = 500. * jnp.ones((il, ix)) #surface geopotential
        fmask = 0.5 * jnp.ones((il, ix)) #land fraction mask
        tsea = 290. * jnp.ones((il, ix)) #ssts
        ssrd = 400. * jnp.ones((il, ix)) #surface downward shortwave
        slrd = 400. * jnp.ones((il, ix)) #surface downward longwave
        lfluxland=True

        test_data = jnp.array([[-6.3609974e-03,-6.3609974e-03, 1.5656566e+02, 5.3803049e-02,
            4.6281613e+02, 5.3620532e+02, 2.8900000e+02, 2.9651727e+02,
            9.4999999e-01, 9.4999999e-01, 2.8500000e+02],
            [-1.5198796e-02,-1.5198796e-02, 2.8738983e+01, 4.0173572e-02,
            3.9300775e+02, 7.0954407e+01, 2.8900000e+02, 2.9406818e+02,
            9.4999999e-01, 9.4999999e-01, 2.8500000e+02],
            [-1.0780082e-02,-1.0780082e-02, 8.8576477e+01, 4.5306068e-02,
            4.1897797e+02, 3.0835028e+02, 2.8900000e+02, 2.9474951e+02,
            9.5000100e-01, 9.5000100e-01, 2.8500000e+02]])
        
        vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=1e-5
        ))

    def test_surface_fluxes_drag_test(self):
        il, ix, kx = 96, 48, 8

        phi0 = 500. * jnp.ones((il, ix)) #surface geopotential

        forog_test = set_orog_land_sfc_drag( phi0 )
        
        test_data = [1.0000012824780082, 1.0000012824780082, 1.0000012824780082]
        self.assertAlmostEqual(jnp.max(forog_test),test_data[0])
        self.assertAlmostEqual(jnp.min(forog_test),test_data[1])
