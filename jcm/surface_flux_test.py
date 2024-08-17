import unittest
from jcm.surface_flux import get_surface_fluxes, set_orog_land_sfc_drag
import jax.numpy as jnp
import csv

class TestSurfaceFluxesUnit(unittest.TestCase):

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
        lfluxland="true"

        with open("jcm/test_files/surface_flux_test1.csv", mode='r') as file:
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

        with open("jcm/test_files/surface_flux_test5.csv", mode='r') as file:
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

    def test_surface_fluxes_drag_test(self):
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
