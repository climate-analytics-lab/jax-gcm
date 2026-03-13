import unittest
import jax
import jax.numpy as jnp
import functools
from jax.test_util import check_vjp, check_jvp
class TestSurfaceFluxesUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ForcingData, SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData, \
               PhysicsState, get_surface_fluxes, get_orog_land_sfc_drag, PhysicsTendency, parameters, TerrainData, speedy_coords, coords, convert_to_speedy_latitudes, grav
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.params import Parameters
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes

        parameters = Parameters.default()
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)

        from jcm.constants import grav
        parameters = Parameters.default()
        
        from jcm.physics.speedy.surface_flux import get_surface_fluxes, get_orog_land_sfc_drag

    def test_grad_surface_flux(self):
        xy = (ix, il)
        zxy = (kx,ix,il)

        psa = jnp.ones((ix,il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 288. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #temperature
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature,soilw_am=soilw_am,stl_am=stl_am)

        _, f_vjp = jax.vjp(get_surface_fluxes, state, physics_data, parameters, forcing, terrain)

        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx,speedy_coords=speedy_c)

        input_tensors = (tends, datas)

        df_dstate, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp(input_tensors)
        
        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_updated_surface_flux(self):
        xy, zxy = (ix, il), (kx,ix,il)
        psa = jnp.ones(xy)
        ua = jnp.ones(zxy)
        va = jnp.ones(zxy)
        ta = jnp.ones(zxy) * 290
        qa = jnp.ones(zxy)
        rh = jnp.ones(zxy) * 0.5
        phi = jnp.ones(zxy) * (jnp.arange(kx))[::-1][:, jnp.newaxis, jnp.newaxis]
        phi0 = jnp.zeros(xy)
        fmask = 0.5 * jnp.ones(xy)
        sea_surface_temperature = jnp.ones((ix,il)) * 292
        rsds = 400 * jnp.ones(xy)
        rlds = 400 * jnp.ones(xy)
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        state = PhysicsState.zeros(zxy, ua, va, ta, qa, phi, psa)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.ones(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am, stl_am=stl_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        self.assertTrue(jnp.allclose(sflux_data.ustr[0, 0, :], jnp.array([-0.01493673, -0.00900353, -0.01197013]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.vstr[0, 0, :], jnp.array([-0.01493673, -0.00900353, -0.01197013]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.shf[0, 0, :], jnp.array([81.73508, 16.271175, 49.003124]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.evap[0, 0, :], jnp.array([0.06291558, 0.10244954, 0.08268256]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.rlus[0, 0, :], jnp.array([459.7182, 403.96204, 431.84012]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.hfluxn[0, 0, :], jnp.array([101.19495, 123.54051]), atol=1e-4))
        self.assertTrue(jnp.isclose(sflux_data.tsfc[0, 0], 290.0, atol=1e-4))
        self.assertTrue(jnp.isclose(sflux_data.tskin[0, 0], 297.22821044921875, atol=1e-4))
        self.assertTrue(jnp.isclose(sflux_data.u0[0, 0], 0.949999988079071, atol=1e-4))
        self.assertTrue(jnp.isclose(sflux_data.v0[0, 0], 0.949999988079071, atol=1e-4))
        self.assertTrue(jnp.isclose(sflux_data.t0[0, 0], 290.0, atol=1e-4))

    def test_surface_fluxes_test1(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix,il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 288. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #specific humidity
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        terrain = TerrainData.from_coords(coords,orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,sea_surface_temperature,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am)
        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        # old outputs: ustr, vstr, shf, evap, rlus, hfluxn, tsfc, tskin, u0, v0, t0
        test_data = jnp.array([[-4.1813999e-03, -1.5040455e-02, -9.6110590e-03],
                               [-4.1813999e-03, -1.5040455e-02, -9.6110590e-03],
                               [ 1.0822081e+02,  7.5566249e+00,  5.5437946e+01],
                               [ 4.8004247e-02,  2.6408084e-02,  3.5274263e-02],
                               [ 4.8786639e+02,  3.9300775e+02,  4.3233978e+02],
                               [ 3.3338889e+02,  1.0605444e+02,  2.2399849e+02],
                               [ 2.8900000e+02,  2.8900000e+02,  2.8900000e+02],
                               [ 2.9885480e+02,  2.9657532e+02,  2.9718643e+02],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 2.8800000e+02,  2.8800000e+02,  2.8800000e+02]])
        
        # pulling the subset of return values to be testsed against the test data
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]
        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data,
            rtol=2e-5
        ))

    def test_surface_fluxes_test2(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 288. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #temperature
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,sea_surface_temperature,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        test_data = jnp.array([[-4.1813999e-03, -1.5040475e-02, -9.6110627e-03],
                               [-4.1813999e-03, -1.5040475e-02, -9.6110627e-03],
                               [ 1.0822081e+02,  7.5566249e+00,  5.5437946e+01],
                               [ 4.8004247e-02,  2.6408084e-02,  3.5274263e-02],
                               [ 4.8786639e+02,  3.9300775e+02,  4.3233978e+02],
                               [ 3.3338889e+02,  1.0605444e+02,  2.2399849e+02],
                               [ 2.8900000e+02,  2.8900000e+02,  2.8900000e+02],
                               [ 2.9885480e+02,  2.9657532e+02,  2.9718643e+02],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 2.8800000e+02,  2.8800000e+02,  2.8800000e+02]])
            
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data,
            rtol=2e-5
        ))

    def test_surface_fluxes_test3(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 288. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #specific humidity
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = -10. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,sea_surface_temperature,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        test_data = jnp.array([[-4.1813999e-03, -1.5040455e-02, -9.6110590e-03],
                               [-4.1813999e-03, -1.5040455e-02, -9.6110590e-03],
                               [ 1.0518237e+02,  7.5566249e+00,  5.3660076e+01],
                               [ 4.6644084e-02,  2.6408084e-02,  3.4507610e-02],
                               [ 4.9224493e+02,  3.9300775e+02,  4.3396124e+02],
                               [ 3.3338889e+02,  1.0965121e+02,  2.2607188e+02],
                               [ 2.8900000e+02,  2.8900000e+02,  2.8900000e+02],
                               [ 2.9926337e+02,  2.9683224e+02,  2.9748245e+02],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 2.8800000e+02,  2.8800000e+02,  2.8800000e+02]])
    
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data,
            rtol=2e-5
        ))

    def test_surface_fluxes_test4(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 300. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #specific humidity
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave

        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,sea_surface_temperature,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am, stl_am=stl_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        test_data = jnp.array([[-1.98534015e-03, -1.44388555e-02, -8.21213890e-03],
                               [-1.98534015e-03, -1.44388555e-02, -8.21213890e-03],
                               [ 3.40381584e+01, -1.79395313e+01,  7.37131262e+00],
                               [ 2.68966686e-02,  1.25386305e-02,  1.92672648e-02],
                               [ 5.22806824e+02,  3.93007751e+02,  4.57831177e+02],
                               [ 3.93572662e+02,  1.78033020e+02,  2.86608398e+02],
                               [ 2.89000000e+02,  2.89000000e+02,  2.89000000e+02],
                               [ 3.02115173e+02,  3.01716644e+02,  3.01831757e+02],
                               [ 9.49999988e-01,  9.49999988e-01,  9.50001001e-01],
                               [ 9.49999988e-01,  9.49999988e-01,  9.50001001e-01],
                               [ 3.00000000e+02,  3.00000000e+02,  3.00000000e+02]])

        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data,
            rtol=2e-5
        ))

    def test_surface_fluxes_test5(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 285. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #specific humidity
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave

        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature,soilw_am=soilw_am, stl_am=stl_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux
        
        test_data = jnp.array([[-6.3609974e-03, -1.5198796e-02, -1.0780082e-02],
                               [-6.3609974e-03, -1.5198796e-02, -1.0780082e-02],
                               [ 1.5656566e+02,  2.8738983e+01,  8.8576477e+01],
                               [ 5.3803049e-02,  4.0173572e-02,  4.5306068e-02],
                               [ 4.6281613e+02,  3.9300775e+02,  4.1897797e+02],
                               [ 2.7777893e+02,  7.0954254e+01,  1.7913458e+02],
                               [ 2.8900000e+02,  2.8900000e+02,  2.8900000e+02],
                               [ 2.9651727e+02,  2.9406818e+02,  2.9474951e+02],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 9.4999999e-01,  9.4999999e-01,  9.5000100e-01],
                               [ 2.8500000e+02,  2.8500000e+02,  2.8500000e+02]])
        
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data,
            rtol=2e-5
        ))

    def test_surface_fluxes_drag_test(self):
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential

        forog_test = get_orog_land_sfc_drag(phi0, parameters.surface_flux.hdrag)

        test_data = [1.0000012824780082, 1.0000012824780082, 1.0000012824780082]
        self.assertAlmostEqual(jnp.max(forog_test),test_data[0])
        self.assertAlmostEqual(jnp.min(forog_test),test_data[1])


    def test_surface_fluxes_gradient_check_test1(self):
        from jcm.utils import convert_back, convert_to_float
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix,il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 288. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #temperature
        rh = 0.8 * jnp.ones(zxy) #relative humidity
        phi = 5000. * jnp.ones(zxy) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        soilw_am = 0.5* jnp.ones(((ix,il)))
        stl_am = 288* jnp.ones((ix,il))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,sea_surface_temperature,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        terrain = TerrainData.from_coords(coords, orography=phi0/grav, fmask=fmask, lfluxland=True)
        terrain, speedy_c = convert_to_speedy_latitudes(terrain, speedy_coords)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad,speedy_coords=speedy_c)
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f( state_f, physics_data_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_surface_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out.surface_flux)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (state_floats, physics_data_floats, parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (state_floats,physics_data_floats,  parameters_floats, forcing_floats, terrain_floats), 
                                atol=None, rtol=1, eps=0.000001)
        
    def test_surface_fluxes_drag_test_gradient_check(self):
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential

        def f(phi0, parameters_sf_hdrag):
            return get_orog_land_sfc_drag(phi0, parameters_sf_hdrag)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (phi0, parameters.surface_flux.hdrag),
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (phi0, parameters.surface_flux.hdrag),
                                atol=None, rtol=1, eps=0.000001)


class TestAquaplanetSurfaceFluxes(unittest.TestCase):
    """Tests for aquaplanet configuration (lfluxland=False, fmask=0).

    These tests verify that ocean surface fluxes are correctly computed
    when there is no land in the simulation (pure aquaplanet).
    """

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ForcingData, SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData, \
               PhysicsState, get_surface_fluxes, PhysicsTendency, parameters, convert_to_speedy_latitudes, grav, speedy_coords, terrain
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.params import Parameters
        from jcm.terrain import TerrainData
        from jcm.constants import grav
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes

        parameters = Parameters.default()
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords) 
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        terrain, speedy_coords = convert_to_speedy_latitudes(terrain, speedy_coords)
        parameters = Parameters.default()

        from jcm.physics.speedy.surface_flux import get_surface_fluxes

    def test_aquaplanet_ocean_evaporation_nonzero(self):
        """Test that ocean evaporation is computed when lfluxland=False.

        This test catches the bug where sea surface fluxes were zero in aquaplanet
        simulations because denvvs[:,:,2], t1[:,:,1], and q1[:,:,1] were only
        computed inside the land_fluxes conditional branch.
        """
        xy = (ix, il)
        zxy = (kx, ix, il)

        # Atmospheric state
        psa = jnp.ones((ix, il))
        ua = 5.0 * jnp.ones(zxy)
        va = 2.0 * jnp.ones(zxy)
        ta = 280.0 * jnp.ones(zxy)  # Cool atmosphere
        qa = 5.0 * jnp.ones(zxy)    # Some humidity
        rh = 0.7 * jnp.ones(zxy)
        phi = 5000.0 * jnp.ones(zxy)

        # Warm ocean - should drive evaporation
        sea_surface_temperature = 300.0 * jnp.ones((ix, il))
        rsds = 400.0 * jnp.ones((ix, il))
        rlds = 350.0 * jnp.ones((ix, il))

        state = PhysicsState.zeros(zxy, ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy, rlds=rlds)
        hum_data = HumidityData.zeros(xy, kx, rh=rh)
        conv_data = ConvectionData.zeros(xy, kx)
        sw_rad = SWRadiationData.zeros(xy, kx, rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy, kx)
        physics_data = PhysicsData.zeros(xy, kx, convection=conv_data, humidity=hum_data,
                                          surface_flux=sflux_data, shortwave_rad=sw_rad, longwave_rad=lw_rad,speedy_coords=speedy_coords)

        # Key: lfluxland=False for aquaplanet
        forcing = ForcingData.zeros(xy, sea_surface_temperature=sea_surface_temperature)

        tendencies, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        # Verify no NaNs in outputs
        self.assertFalse(jnp.any(jnp.isnan(sflux_data.evap)), "Evaporation contains NaNs")
        self.assertFalse(jnp.any(jnp.isnan(sflux_data.shf)), "Sensible heat flux contains NaNs")
        self.assertFalse(jnp.any(jnp.isnan(sflux_data.ustr)), "Wind stress contains NaNs")
        self.assertFalse(jnp.any(jnp.isnan(tendencies.specific_humidity)), "Humidity tendency contains NaNs")

        # Ocean evaporation (index 1) should be positive with warm SST and cool atmosphere
        self.assertTrue(jnp.all(sflux_data.evap[:, :, 1] > 0),
                        "Ocean evaporation should be positive with warm SST")

        # Weighted evaporation (index 2) should equal ocean evaporation when fmask=0
        self.assertTrue(jnp.allclose(sflux_data.evap[:, :, 2], sflux_data.evap[:, :, 1]),
                        "Weighted evaporation should equal ocean evaporation for pure ocean")

        # Humidity tendency should be positive (evaporation adds moisture)
        self.assertTrue(jnp.all(tendencies.specific_humidity[-1] > 0),
                        "Humidity tendency should be positive from ocean evaporation")

    def test_aquaplanet_sensible_heat_flux(self):
        """Test that ocean sensible heat flux is computed correctly in aquaplanet."""
        xy = (ix, il)
        zxy = (kx, ix, il)

        psa = jnp.ones((ix, il))
        ua = 5.0 * jnp.ones(zxy)
        va = 2.0 * jnp.ones(zxy)
        ta = 280.0 * jnp.ones(zxy)
        qa = 5.0 * jnp.ones(zxy)
        rh = 0.7 * jnp.ones(zxy)
        phi = 5000.0 * jnp.ones(zxy)

        # Warm ocean relative to atmosphere
        sea_surface_temperature = 300.0 * jnp.ones((ix, il))
        rsds = 400.0 * jnp.ones((ix, il))
        rlds = 350.0 * jnp.ones((ix, il))

        state = PhysicsState.zeros(zxy, ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy, rlds=rlds)
        hum_data = HumidityData.zeros(xy, kx, rh=rh)
        conv_data = ConvectionData.zeros(xy, kx)
        sw_rad = SWRadiationData.zeros(xy, kx, rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy, kx)
        physics_data = PhysicsData.zeros(xy, kx, convection=conv_data, humidity=hum_data,
                                          surface_flux=sflux_data, shortwave_rad=sw_rad, longwave_rad=lw_rad,speedy_coords=speedy_coords)
        forcing = ForcingData.zeros(xy, sea_surface_temperature=sea_surface_temperature)

        tendencies, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, terrain)
        sflux_data = physics_data.surface_flux

        # Sensible heat flux should be positive (ocean warming atmosphere)
        self.assertTrue(jnp.all(sflux_data.shf[:, :, 1] > 0),
                        "Ocean sensible heat flux should be positive with warm SST")

        # Temperature tendency should be positive at surface level
        self.assertTrue(jnp.all(tendencies.temperature[-1] > 0),
                        "Temperature tendency should be positive from warm ocean")

    def test_aquaplanet_gradient_check(self):
        """Test that gradients are valid for aquaplanet configuration."""
        xy = (ix, il)
        zxy = (kx, ix, il)
        psa = jnp.ones((ix, il))
        ua = 5.0 * jnp.ones(zxy)
        va = 2.0 * jnp.ones(zxy)
        ta = 290.0 * jnp.ones(zxy)
        qa = 5.0 * jnp.ones(zxy)
        rh = 0.7 * jnp.ones(zxy)
        phi = 5000.0 * jnp.ones(zxy)
        sea_surface_temperature = 295.0 * jnp.ones((ix, il))
        rsds = 400.0 * jnp.ones((ix, il))
        rlds = 350.0 * jnp.ones((ix, il))

        state = PhysicsState.zeros(zxy, ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy, rlds=rlds)
        hum_data = HumidityData.zeros(xy, kx, rh=rh)
        conv_data = ConvectionData.zeros(xy, kx)
        sw_rad = SWRadiationData.zeros(xy, kx, rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy, kx)
        physics_data = PhysicsData.zeros(xy, kx, convection=conv_data, humidity=hum_data,
                                          surface_flux=sflux_data, shortwave_rad=sw_rad, longwave_rad=lw_rad,speedy_coords=speedy_coords)
        forcing = ForcingData.zeros(xy, sea_surface_temperature=sea_surface_temperature)

        _, f_vjp = jax.vjp(get_surface_fluxes, state, physics_data, parameters, forcing, terrain)

        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx,speedy_coords=speedy_coords)

        df_dstate, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp((tends, datas))

        self.assertFalse(df_ddatas.isnan().any_true(), "Gradient w.r.t. physics_data contains NaNs")
        self.assertFalse(df_dstate.isnan().any_true(), "Gradient w.r.t. state contains NaNs")
        self.assertFalse(df_dparams.isnan().any_true(), "Gradient w.r.t. parameters contains NaNs")
        self.assertFalse(df_dforcing.isnan().any_true(), "Gradient w.r.t. forcing contains NaNs")

