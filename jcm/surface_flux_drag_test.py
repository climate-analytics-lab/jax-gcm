## Unit test for surface fluxes module.
#
#
# Loads csv surface_fluxes_test.csv which has test case for values listed here
# from the speedy.f90 code

import unittest
from jcm.surface_flux import set_orog_land_sfc_drag
import jax.numpy as jnp

class TestSurfaceFlux(unittest.TestCase):

    def test_surface_flux(self):

        # grid
        il = 96 #latitude
        ix = 48 #longitude
        kx = 8 #vertical

        # make arrays
        psa = 1000. * np.ones((il, ix)) #surface pressure
        ua = np.ones(((il, ix, kx))) #zonal wind
        va = np.ones(((il, ix, kx))) #meridional wind
        ta = 288. * np.ones(((il, ix, kx))) #temperature
        qa = 5. * np.ones(((il, ix, kx))) # g/kg
        rh = 0.8 * np.ones(((il, ix, kx))) #relative humidity
        phi = 5000. * np.ones(((il, ix, kx))) #geopotential
        phi0 = 500. * np.ones((il, ix)) #surface geopotential
        fmask = 0.5 * np.ones((il, ix)) #land fraction mask
        tsea = 290. * np.ones((il, ix)) #ssts
        ssrd = 400. * np.ones((il, ix)) #surface downward shortwave
        slrd = 400. * np.ones((il, ix)) #surface downward longwave

        # default values, over land
        lfluxland="true"

        ustr,vstr,shf,evap,slru,hfluxn,tsfc,tskin,u0,v0,t0 = surface_fluxes.get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        # Check that returned values are not empty.
        self.assertIsNotNone(ustr)
        self.assertIsNotNone(vstr)
        self.assertIsNotNone(shf)
        self.assertIsNotNone(evap)
        self.assertIsNotNone(slru)
        self.assertIsNotNone(hfluxn)
        self.assertIsNotNone(tsfc)
        self.assertIsNotNone(tskin)
        self.assertIsNotNone(u0)
        self.assertIsNotNone(v0)
        self.assertIsNotNone(t0)
        
