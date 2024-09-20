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
        return
    
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
        lfluxland="true"

        with open("jcm/test_files/surface_flux_test1.csv", mode='r') as file:
            # reader = csv.reader(file)
        
            # # Read the header (keys)
            # keys = next(reader)
        
            # # Initialize an empty dictionary with keys
            # test_data = {key: [] for key in keys}
        
            # Read the rows and append values to the dictionary
            #for row in reader:
            #    for i, value in enumerate(row):
            #        #test_data[keys[i]].append(value if value.replace('.','',1).isdigit() else float(value))
            #        test_data[keys[i]].append(float(value) if value.replace('.', '', 1).isdigit() else value)
            #
            reader = csv.reader(file)
            columns = next(reader)
            colmap = dict(zip(columns, range(len(columns))))

            test_data = jnp.matrix(jnp.loadtxt(file, delimiter=",", skiprows=1))
            # Convert lists to JAX arrays
            test_data = {key: jnp.array(value) for key, value in test_data.items()}
    
        
        ustr,vstr,shf,evap,slru,hfluxn,tsfc,tskin,u0,v0,t0 = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertAlmostEqual(jnp.max(ustr),test_data["ustr"][0])
        self.assertAlmostEqual(jnp.min(ustr),test_data["ustr"][1])
        self.assertAlmostEqual(jnp.mean(ustr),test_data["ustr"][2])

        self.assertAlmostEqual(jnp.max(shf),test_data["shf"][0])
        self.assertAlmostEqual(jnp.min(shf),test_data["shf"][1])
        self.assertAlmostEqual(jnp.mean(shf),test_data["shf"][2])

        self.assertAlmostEqual(jnp.max(evap),test_data["evap"][0])
        self.assertAlmostEqual(jnp.min(evap),test_data["evap"][1])
        self.assertAlmostEqual(jnp.mean(evap),test_data["evap"][2])

        self.assertAlmostEqual(jnp.max(slru),test_data["slru"][0])
        self.assertAlmostEqual(jnp.min(slru),test_data["slru"][1])
        self.assertAlmostEqual(jnp.mean(slru),test_data["slru"][2])

        self.assertAlmostEqual(jnp.max(hfluxn),test_data["hfluxn"][0])
        self.assertAlmostEqual(jnp.min(hfluxn),test_data["hfluxn"][1])
        self.assertAlmostEqual(jnp.mean(hfluxn),test_data["hfluxn"][2])

        self.assertAlmostEqual(jnp.max(tsfc),test_data["tsfc"][0])
        self.assertAlmostEqual(jnp.min(tsfc),test_data["tsfc"][1])
        self.assertAlmostEqual(jnp.mean(tsfc),test_data["tsfc"][2])

        self.assertAlmostEqual(jnp.max(u0),test_data["u0"][0])
        self.assertAlmostEqual(jnp.min(u0),test_data["u0"][1])
        self.assertAlmostEqual(jnp.mean(u0),test_data["u0"][2])

        self.assertAlmostEqual(jnp.max(v0),test_data["v0"][0])
        self.assertAlmostEqual(jnp.min(v0),test_data["v0"][1])
        self.assertAlmostEqual(jnp.mean(v0),test_data["v0"][2])

        self.assertAlmostEqual(jnp.max(t0),test_data["t0"][0])
        self.assertAlmostEqual(jnp.min(t0),test_data["t0"][1])
        self.assertAlmostEqual(jnp.mean(t0),test_data["t0"][2])

    def test_surface_fluxes_test2(self):
        return
    
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
        lfluxland="false"

        with open("jcm/test_files/surface_flux_test2.csv", mode='r') as file:
            reader = csv.reader(file)
        
            # Read the header (keys)
            keys = next(reader)
        
            # Initialize an empty dictionary with keys
            test_data = {key: [] for key in keys}
        
            # Read the rows and append values to the dictionary
            for row in reader:
                for i, value in enumerate(row):
                    test_data[keys[i]].append(float(value) if value.replace('.','',1).isdigit() else value)
        
            # Convert lists to JAX arrays
            test_data = {key: jnp.array(value) for key, value in test_data.items()}
    
        
        ustr,vstr,shf,evap,slru,hfluxn,tsfc,tskin,u0,v0,t0 = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertAlmostEqual(jnp.max(ustr),test_data["ustr"][0])
        self.assertAlmostEqual(jnp.min(ustr),test_data["ustr"][1])
        self.assertAlmostEqual(jnp.mean(ustr),test_data["ustr"][2])

        self.assertAlmostEqual(jnp.max(shf),test_data["shf"][0])
        self.assertAlmostEqual(jnp.min(shf),test_data["shf"][1])
        self.assertAlmostEqual(jnp.mean(shf),test_data["shf"][2])

        self.assertAlmostEqual(jnp.max(evap),test_data["evap"][0])
        self.assertAlmostEqual(jnp.min(evap),test_data["evap"][1])
        self.assertAlmostEqual(jnp.mean(evap),test_data["evap"][2])

        self.assertAlmostEqual(jnp.max(slru),test_data["slru"][0])
        self.assertAlmostEqual(jnp.min(slru),test_data["slru"][1])
        self.assertAlmostEqual(jnp.mean(slru),test_data["slru"][2])

        self.assertAlmostEqual(jnp.max(hfluxn),test_data["hfluxn"][0])
        self.assertAlmostEqual(jnp.min(hfluxn),test_data["hfluxn"][1])
        self.assertAlmostEqual(jnp.mean(hfluxn),test_data["hfluxn"][2])

        self.assertAlmostEqual(jnp.max(tsfc),test_data["tsfc"][0])
        self.assertAlmostEqual(jnp.min(tsfc),test_data["tsfc"][1])
        self.assertAlmostEqual(jnp.mean(tsfc),test_data["tsfc"][2])

        self.assertAlmostEqual(jnp.max(u0),test_data["u0"][0])
        self.assertAlmostEqual(jnp.min(u0),test_data["u0"][1])
        self.assertAlmostEqual(jnp.mean(u0),test_data["u0"][2])

        self.assertAlmostEqual(jnp.max(v0),test_data["v0"][0])
        self.assertAlmostEqual(jnp.min(v0),test_data["v0"][1])
        self.assertAlmostEqual(jnp.mean(v0),test_data["v0"][2])

        self.assertAlmostEqual(jnp.max(t0),test_data["t0"][0])
        self.assertAlmostEqual(jnp.min(t0),test_data["t0"][1])
        self.assertAlmostEqual(jnp.mean(t0),test_data["t0"][2])

    def test_surface_fluxes_test3(self):
        return
    
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
        lfluxland="true"

        with open("jcm/test_files/surface_flux_test3.csv", mode='r') as file:
            reader = csv.reader(file)
        
            # Read the header (keys)
            keys = next(reader)
        
            # Initialize an empty dictionary with keys
            test_data = {key: [] for key in keys}
        
            # Read the rows and append values to the dictionary
            for row in reader:
                for i, value in enumerate(row):
                    test_data[keys[i]].append(float(value) if value.replace('.','',1).isdigit() else value)
        
            # Convert lists to JAX arrays
            test_data = {key: jnp.array(value) for key, value in test_data.items()}
    
        
        ustr,vstr,shf,evap,slru,hfluxn,tsfc,tskin,u0,v0,t0 = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertAlmostEqual(jnp.max(ustr),test_data["ustr"][0])
        self.assertAlmostEqual(jnp.min(ustr),test_data["ustr"][1])
        self.assertAlmostEqual(jnp.mean(ustr),test_data["ustr"][2])

        self.assertAlmostEqual(jnp.max(shf),test_data["shf"][0])
        self.assertAlmostEqual(jnp.min(shf),test_data["shf"][1])
        self.assertAlmostEqual(jnp.mean(shf),test_data["shf"][2])

        self.assertAlmostEqual(jnp.max(evap),test_data["evap"][0])
        self.assertAlmostEqual(jnp.min(evap),test_data["evap"][1])
        self.assertAlmostEqual(jnp.mean(evap),test_data["evap"][2])

        self.assertAlmostEqual(jnp.max(slru),test_data["slru"][0])
        self.assertAlmostEqual(jnp.min(slru),test_data["slru"][1])
        self.assertAlmostEqual(jnp.mean(slru),test_data["slru"][2])

        self.assertAlmostEqual(jnp.max(hfluxn),test_data["hfluxn"][0])
        self.assertAlmostEqual(jnp.min(hfluxn),test_data["hfluxn"][1])
        self.assertAlmostEqual(jnp.mean(hfluxn),test_data["hfluxn"][2])

        self.assertAlmostEqual(jnp.max(tsfc),test_data["tsfc"][0])
        self.assertAlmostEqual(jnp.min(tsfc),test_data["tsfc"][1])
        self.assertAlmostEqual(jnp.mean(tsfc),test_data["tsfc"][2])

        self.assertAlmostEqual(jnp.max(u0),test_data["u0"][0])
        self.assertAlmostEqual(jnp.min(u0),test_data["u0"][1])
        self.assertAlmostEqual(jnp.mean(u0),test_data["u0"][2])

        self.assertAlmostEqual(jnp.max(v0),test_data["v0"][0])
        self.assertAlmostEqual(jnp.min(v0),test_data["v0"][1])
        self.assertAlmostEqual(jnp.mean(v0),test_data["v0"][2])

        self.assertAlmostEqual(jnp.max(t0),test_data["t0"][0])
        self.assertAlmostEqual(jnp.min(t0),test_data["t0"][1])
        self.assertAlmostEqual(jnp.mean(t0),test_data["t0"][2])

    def test_surface_fluxes_test4(self):
        return
    
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
        lfluxland="true"

        with open("jcm/test_files/surface_flux_test4.csv", mode='r') as file:
            reader = csv.reader(file)
        
            # Read the header (keys)
            keys = next(reader)
        
            # Initialize an empty dictionary with keys
            test_data = {key: [] for key in keys}
        
            # Read the rows and append values to the dictionary
            for row in reader:
                for i, value in enumerate(row):
                    test_data[keys[i]].append(float(value) if value.replace('.','',1).isdigit() else value)
        
            # Convert lists to JAX arrays
            test_data = {key: jnp.array(value) for key, value in test_data.items()}
    
        
        ustr,vstr,shf,evap,slru,hfluxn,tsfc,tskin,u0,v0,t0 = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertAlmostEqual(jnp.max(ustr),test_data["ustr"][0])
        self.assertAlmostEqual(jnp.min(ustr),test_data["ustr"][1])
        self.assertAlmostEqual(jnp.mean(ustr),test_data["ustr"][2])

        self.assertAlmostEqual(jnp.max(shf),test_data["shf"][0])
        self.assertAlmostEqual(jnp.min(shf),test_data["shf"][1])
        self.assertAlmostEqual(jnp.mean(shf),test_data["shf"][2])

        self.assertAlmostEqual(jnp.max(evap),test_data["evap"][0])
        self.assertAlmostEqual(jnp.min(evap),test_data["evap"][1])
        self.assertAlmostEqual(jnp.mean(evap),test_data["evap"][2])

        self.assertAlmostEqual(jnp.max(slru),test_data["slru"][0])
        self.assertAlmostEqual(jnp.min(slru),test_data["slru"][1])
        self.assertAlmostEqual(jnp.mean(slru),test_data["slru"][2])

        self.assertAlmostEqual(jnp.max(hfluxn),test_data["hfluxn"][0])
        self.assertAlmostEqual(jnp.min(hfluxn),test_data["hfluxn"][1])
        self.assertAlmostEqual(jnp.mean(hfluxn),test_data["hfluxn"][2])

        self.assertAlmostEqual(jnp.max(tsfc),test_data["tsfc"][0])
        self.assertAlmostEqual(jnp.min(tsfc),test_data["tsfc"][1])
        self.assertAlmostEqual(jnp.mean(tsfc),test_data["tsfc"][2])

        self.assertAlmostEqual(jnp.max(u0),test_data["u0"][0])
        self.assertAlmostEqual(jnp.min(u0),test_data["u0"][1])
        self.assertAlmostEqual(jnp.mean(u0),test_data["u0"][2])

        self.assertAlmostEqual(jnp.max(v0),test_data["v0"][0])
        self.assertAlmostEqual(jnp.min(v0),test_data["v0"][1])
        self.assertAlmostEqual(jnp.mean(v0),test_data["v0"][2])

        self.assertAlmostEqual(jnp.max(t0),test_data["t0"][0])
        self.assertAlmostEqual(jnp.min(t0),test_data["t0"][1])
        self.assertAlmostEqual(jnp.mean(t0),test_data["t0"][2])

    def test_surface_fluxes_test5(self):
        return
    
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
        lfluxland="true"

        # with open("jcm/test_files/surface_flux_test5.csv", mode='r') as file:
        #     reader = csv.reader(file)
        
        #     # Read the header (keys)
        #     keys = next(reader)
        
        #     # Initialize an empty dictionary with keys
        #     test_data = {key: [] for key in keys}
        
        #     # Read the rows and append values to the dictionary
        #     for row in reader:
        #         for i, value in enumerate(row):
        #             test_data[keys[i]].append(float(value) if value.replace('.','',1).isdigit() else value)
        #         #row_ls = row.split(",")
        #         #for i, value in enumerate(row_ls)
        
        #     # Convert lists to JAX arrays
        #     test_data = {key: jnp.array(value) for key, value in test_data.items()}
        import pandas as pd
        test_data = pd.read_csv("jcm/test_files/surface_flux_test5.csv")
    
        
        ustr,vstr,shf,evap,slru,hfluxn,tsfc,tskin,u0,v0,t0 = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,ssrd,slrd,lfluxland)

        # Check that itop, precls, dtlsc, and dqlsc are not null.
        self.assertAlmostEqual(jnp.max(ustr),test_data["ustr"][0])
        self.assertAlmostEqual(jnp.min(ustr),test_data["ustr"][1])
        self.assertAlmostEqual(jnp.mean(ustr),test_data["ustr"][2])

        self.assertAlmostEqual(jnp.max(shf),test_data["shf"][0])
        self.assertAlmostEqual(jnp.min(shf),test_data["shf"][1])
        self.assertAlmostEqual(jnp.mean(shf),test_data["shf"][2])

        self.assertAlmostEqual(jnp.max(evap),test_data["evap"][0])
        self.assertAlmostEqual(jnp.min(evap),test_data["evap"][1])
        self.assertAlmostEqual(jnp.mean(evap),test_data["evap"][2])

        self.assertAlmostEqual(jnp.max(slru),test_data["slru"][0])
        self.assertAlmostEqual(jnp.min(slru),test_data["slru"][1])
        self.assertAlmostEqual(jnp.mean(slru),test_data["slru"][2])

        self.assertAlmostEqual(jnp.max(hfluxn),test_data["hfluxn"][0])
        self.assertAlmostEqual(jnp.min(hfluxn),test_data["hfluxn"][1])
        self.assertAlmostEqual(jnp.mean(hfluxn),test_data["hfluxn"][2])

        self.assertAlmostEqual(jnp.max(tsfc),test_data["tsfc"][0])
        self.assertAlmostEqual(jnp.min(tsfc),test_data["tsfc"][1])
        self.assertAlmostEqual(jnp.mean(tsfc),test_data["tsfc"][2])

        self.assertAlmostEqual(jnp.max(u0),test_data["u0"][0])
        self.assertAlmostEqual(jnp.min(u0),test_data["u0"][1])
        self.assertAlmostEqual(jnp.mean(u0),test_data["u0"][2])

        self.assertAlmostEqual(jnp.max(v0),test_data["v0"][0])
        self.assertAlmostEqual(jnp.min(v0),test_data["v0"][1])
        self.assertAlmostEqual(jnp.mean(v0),test_data["v0"][2])

        self.assertAlmostEqual(jnp.max(t0),test_data["t0"][0])
        self.assertAlmostEqual(jnp.min(t0),test_data["t0"][1])
        self.assertAlmostEqual(jnp.mean(t0),test_data["t0"][2])

    def test_surface_fluxes_drag_test(self):
        return
    
        il, ix, kx = 96, 48, 8

        hdrag = 2000.0 # Height scale for orographic correction        
        grav = 9.81 # gravity constant
        phi0 = 500. * jnp.ones((il, ix)) #surface geopotential

        forog_test = jnp.zeros((ix,il)) # Time-invariant fields (initial. in SFLSET)

        with open("jcm/test_files/surface_flux_drag_test.csv", mode='r') as file:
            reader = csv.reader(file)
        
            # Read the header (keys)
            keys = next(reader)
        
            # Initialize an empty dictionary with keys
            test_data = {key: [] for key in keys}
        
            # Read the rows and append values to the dictionary
            for row in reader:
                for i, value in enumerate(row):
                    test_data[keys[i]].append(float(value) if value.replace('.','',1).isdigit() else value)
        
            # Convert lists to JAX arrays
            test_data = {key: jnp.array(value) for key, value in test_data.items()}
    
        forog_test = set_orog_land_sfc_drag( phi0 )
        
        self.assertAlmostEqual(jnp.max(forog_test),test_data["forog"][0])
        self.assertAlmostEqual(jnp.min(forog_test),test_data["forog"][1])
        self.assertAlmostEqual(jnp.mean(forog_test),test_data["forog"][2])
