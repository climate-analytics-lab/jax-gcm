import unittest
import jax
import jax.numpy as jnp
import functools
from jax.test_util import check_vjp, check_jvp
from jcm.physics.speedy.humidity import get_qsat

class TestSurfaceFluxesUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ForcingData, SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData, \
               PhysicsState, get_surface_fluxes, get_orog_land_sfc_drag, PhysicsTendency, parameters, Geometry, convert_to_speedy_latitudes, grav
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature,soilw_am=soilw_am,stl_am=stl_am,lfluxland=True)

        _, f_vjp = jax.vjp(get_surface_fluxes, state, physics_data, parameters, forcing, geometry)

        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx)

        input_tensors = (tends, datas)

        df_dstate, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input_tensors)
        
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
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.ones(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am, stl_am=stl_am, lfluxland=True)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, geometry)
        sflux_data = physics_data.surface_flux

        self.assertTrue(jnp.allclose(sflux_data.ustr[0, 0, :], jnp.array([-0.01493673, -0.00900353, -0.01197013]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.vstr[0, 0, :], jnp.array([-0.01493673, -0.00900353, -0.01197013]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.shf[0, 0, :], jnp.array([81.73508, 16.271175, 49.003124]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.evap[0, 0, :], jnp.array([0.06291558, 0.10244954, 0.08268256]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.rlus[0, 0, :], jnp.array([459.7182, 403.96204, 431.84012]), atol=1e-4))
        self.assertTrue(jnp.allclose(sflux_data.hfluxn[0, 0, :], jnp.array([101.19495, 668.53546]), atol=1e-4))
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am, lfluxland=True)
        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, geometry)
        sflux_data = physics_data.surface_flux

        # old outputs: ustr, vstr, shf, evap, rlus, hfluxn, tsfc, tskin, u0, v0, t0
        test_data = jnp.array([[-4.18139994e-03,-4.18139994e-03, 1.08220810e+02, 4.80042472e-02,
            4.87866394e+02, 4.80595490e+02, 2.89000000e+02, 2.98854797e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-1.50404554e-02,-1.50404554e-02, 7.55662489e+00, 2.64080837e-02,
            3.93007751e+02, 1.06054558e+02, 2.89000000e+02, 2.96575317e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-9.61105898e-03,-9.61105898e-03, 5.54379463e+01, 3.52742635e-02,
            4.32339783e+02, 2.97601044e+02, 2.89000000e+02, 2.97186432e+02,
            9.50001001e-01, 9.50001001e-01, 2.88000000e+02]])
        
        # pulling the subset of return values to be testsed against the test data
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]
        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am, lfluxland=True)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, geometry)
        sflux_data = physics_data.surface_flux


        test_data = jnp.array([[-4.18139994e-03,-4.18139994e-03, 1.08220810e+02, 4.80042472e-02,
            4.87866394e+02, 4.80595490e+02, 2.89000000e+02, 2.98854797e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-1.50404749e-02,-1.50404749e-02, 7.55662489e+00, 2.64080837e-02,
            3.93007751e+02, 1.06054558e+02, 2.89000000e+02, 2.96575317e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-9.61106271e-03,-9.61106271e-03, 5.54379463e+01, 3.52742635e-02,
            4.32339783e+02, 2.97601044e+02, 2.89000000e+02, 2.97186432e+02,
            9.50001001e-01, 9.50001001e-01, 2.88000000e+02]])
            
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=2e-5
        ))

    def test_surface_fluxes_test3(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 288. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #temperature
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am, lfluxland=True)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, geometry)
        sflux_data = physics_data.surface_flux

        test_data = jnp.array([[-4.18139994e-03,-4.18139994e-03, 1.05182373e+02, 4.66440842e-02,
            4.92244934e+02, 4.80595490e+02, 2.89000000e+02, 2.99263367e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-1.50404554e-02,-1.50404554e-02, 7.55662489e+00, 2.64080837e-02,
            3.93007751e+02, 1.09651382e+02, 2.89000000e+02, 2.96832245e+02,
            9.49999988e-01, 9.49999988e-01, 2.88000000e+02],
            [-9.61105898e-03,-9.61105898e-03, 5.36600761e+01, 3.45076099e-02,
            4.33961243e+02, 2.99674957e+02, 2.89000000e+02, 2.97482452e+02,
            9.50001001e-01, 9.50001001e-01, 2.88000000e+02]])
    
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=2e-5
        ))

    def test_surface_fluxes_test4(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 300. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #temperature
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am, stl_am=stl_am, lfluxland=True)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, geometry)
        sflux_data = physics_data.surface_flux

        test_data = jnp.array([[-1.98534015e-03,-1.98534015e-03, 3.40381584e+01, 2.68966686e-02,
            5.22806824e+02, 4.20411835e+02, 2.89000000e+02, 3.02115173e+02,
            9.49999988e-01, 9.49999988e-01, 3.00000000e+02],
            [-1.44388555e-02,-1.44388555e-02,-1.79395313e+01, 1.25386305e-02,
            3.93007751e+02, 1.78033081e+02, 2.89000000e+02, 3.01716644e+02,
            9.49999988e-01, 9.49999988e-01, 3.00000000e+02],
            [-8.21213890e-03,-8.21213890e-03, 7.37131262e+00, 1.92672648e-02,
            4.57831177e+02, 3.00025909e+02, 2.89000000e+02, 3.01831757e+02,
            9.50001001e-01, 9.50001001e-01, 3.00000000e+02]])

        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
            rtol=2e-5
        ))

    def test_surface_fluxes_test5(self):
        xy = (ix,il)
        zxy = (kx,ix,il)
        psa = jnp.ones((ix, il)) #surface pressure
        ua = jnp.ones(zxy) #zonal wind
        va = jnp.ones(zxy) #meridional wind
        ta = 285. * jnp.ones(zxy) #temperature
        qa = 5. * jnp.ones(zxy) #temperature
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature,soilw_am=soilw_am, stl_am=stl_am, lfluxland=True)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, forcing, geometry)
        sflux_data = physics_data.surface_flux
        
        test_data = jnp.array([[-6.3609974e-03,-6.3609974e-03, 1.5656566e+02, 5.3803049e-02,
            4.6281613e+02, 5.3620532e+02, 2.8900000e+02, 2.9651727e+02,
            9.4999999e-01, 9.4999999e-01, 2.8500000e+02],
            [-1.5198796e-02,-1.5198796e-02, 2.8738983e+01, 4.0173572e-02,
            3.9300775e+02, 7.0954407e+01, 2.8900000e+02, 2.9406818e+02,
            9.4999999e-01, 9.4999999e-01, 2.8500000e+02],
            [-1.0780082e-02,-1.0780082e-02, 8.8576477e+01, 4.5306068e-02,
            4.1897797e+02, 3.0835028e+02, 2.8900000e+02, 2.9474951e+02,
            9.5000100e-01, 9.5000100e-01, 2.8500000e+02]])
        
        vars = [sflux_data.ustr, sflux_data.vstr, sflux_data.shf, sflux_data.evap, sflux_data.rlus, sflux_data.hfluxn, sflux_data.tsfc, sflux_data.tskin, sflux_data.u0, sflux_data.v0, sflux_data.t0]

        self.assertTrue(jnp.allclose(
            jnp.array([[jnp.max(var), jnp.min(var), jnp.mean(var)] for var in vars]),
            test_data.T,
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
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.zeros(xy,sea_surface_temperature=sea_surface_temperature, soilw_am=soilw_am,stl_am=stl_am, lfluxland=True)


        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(geometry)

        def f( state_f, physics_data_f, parameters_f, forcing_f,geometry_f):
            tend_out, data_out = get_surface_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out.surface_flux)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (state_floats, physics_data_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (state_floats,physics_data_floats,  parameters_floats, forcing_floats, geometry_floats), 
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


class TestLandSurfaceFluxesIdealized(unittest.TestCase):
    """Idealized tests for land surface flux calculations to check physical plausibility."""

    def setUp(self):
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        from jcm.constants import grav
        from jcm.physics.speedy.surface_flux import compute_land_surface_fluxes
        from jcm.physics.speedy.physical_constants import p0, rgas, cp, sbc

        self.parameters = Parameters.default()
        self.compute_land_surface_fluxes = compute_land_surface_fluxes
        self.grav = grav
        self.p0 = p0
        self.rgas = rgas
        self.cp = cp
        self.sbc = sbc
        self.convert_to_speedy_latitudes = convert_to_speedy_latitudes
        self.Geometry = Geometry

        # Small grid for fast tests
        self.ix, self.il, self.kx = 64, 32, 8

    def test_evaporation_is_nonnegative(self):
        """Test that evaporation is always non-negative."""
        ix, il, kx = self.ix, self.il, self.kx

        # Create realistic inputs
        ua = jnp.ones((kx, ix, il)) * 5.0  # 5 m/s wind
        va = jnp.ones((kx, ix, il)) * 5.0
        ta = jnp.ones((kx, ix, il)) * 288.0  # 15°C
        qa = jnp.ones((kx, ix, il)) * 0.008  # ~8 g/kg
        rh = jnp.ones((kx, ix, il)) * 0.7  # 70% RH
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))  # 100% land
        stl_am = jnp.ones((ix, il)) * 285.0
        soilw_am = jnp.ones((ix, il)) * 0.5  # 50% soil moisture
        rsds = jnp.ones((ix, il)) * 400.0
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        # Create geometry
        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        ustr, vstr, shf, evap, rlus, hfluxn, tskin = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Evaporation should always be non-negative
        self.assertTrue(jnp.all(evap >= 0.0),
                       f"Evaporation should be non-negative, but got min={jnp.min(evap)}, max={jnp.max(evap)}")

    def test_evaporation_zero_with_zero_soil_moisture(self):
        """Test that evaporation is zero when soil moisture is zero."""
        ix, il, kx = self.ix, self.il, self.kx

        ua = jnp.ones((kx, ix, il)) * 5.0
        va = jnp.ones((kx, ix, il)) * 5.0
        ta = jnp.ones((kx, ix, il)) * 288.0
        qa = jnp.ones((kx, ix, il)) * 0.005
        rh = jnp.ones((kx, ix, il)) * 0.5
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))
        stl_am = jnp.ones((ix, il)) * 290.0
        soilw_am = jnp.zeros((ix, il))  # Zero soil moisture
        rsds = jnp.ones((ix, il)) * 400.0
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        ustr, vstr, shf, evap, rlus, hfluxn, tskin = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # With zero soil moisture, evaporation should be zero
        self.assertTrue(jnp.allclose(evap, 0.0, atol=1e-10),
                       f"Evaporation should be zero with zero soil moisture, but got {evap}")

    def test_evaporation_reasonable_values(self):
        """Test that evaporation has reasonable magnitude under normal conditions."""
        ix, il, kx = self.ix, self.il, self.kx

        # Normal midlatitude summer conditions
        ua = jnp.ones((kx, ix, il)) * 3.0  # Light wind
        va = jnp.ones((kx, ix, il)) * 3.0
        ta = jnp.ones((kx, ix, il)) * 293.0  # 20°C
        qa = jnp.ones((kx, ix, il)) * 0.010  # ~10 g/kg
        rh = jnp.ones((kx, ix, il)) * 0.6  # 60% RH
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))
        stl_am = jnp.ones((ix, il)) * 290.0
        soilw_am = jnp.ones((ix, il)) * 0.7  # Moist soil
        rsds = jnp.ones((ix, il)) * 600.0  # Strong solar radiation
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        ustr, vstr, shf, evap, rlus, hfluxn, tskin = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Evaporation should be positive and reasonable (0.01 to 1.0 kg/m^2/s roughly)
        self.assertTrue(jnp.all(evap >= 0.0), "Evaporation should be non-negative")
        self.assertTrue(jnp.all(evap < 1.0),
                       f"Evaporation seems unreasonably high: max={jnp.max(evap)} kg/m^2/s")
        # With moist soil and warm conditions, should have some evaporation
        self.assertTrue(jnp.all(evap > 0.0),
                       f"Expected positive evaporation with moist soil, got min={jnp.min(evap)}")

    def test_sensible_heat_flux_sign(self):
        """Test that sensible heat flux has correct sign based on temperature gradient."""
        ix, il, kx = self.ix, self.il, self.kx

        # Case 1: Surface warmer than air -> positive heat flux (into atmosphere)
        ua = jnp.ones((kx, ix, il)) * 3.0
        va = jnp.ones((kx, ix, il)) * 3.0
        ta = jnp.ones((kx, ix, il)) * 285.0  # Cool air
        qa = jnp.ones((kx, ix, il)) * 0.008
        rh = jnp.ones((kx, ix, il)) * 0.7
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))
        stl_am = jnp.ones((ix, il)) * 295.0  # Warm surface
        soilw_am = jnp.ones((ix, il)) * 0.5
        rsds = jnp.ones((ix, il)) * 600.0
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        ustr, vstr, shf, evap, rlus, hfluxn, tskin = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Surface warmer than air -> positive sensible heat flux
        self.assertTrue(jnp.all(shf > 0.0),
                       f"Expected positive heat flux with warm surface, got min={jnp.min(shf)}, max={jnp.max(shf)}")

    def test_fluxes_with_zero_wind(self):
        """Test that fluxes are computed reasonably even with zero wind (due to gustiness)."""
        ix, il, kx = self.ix, self.il, self.kx

        ua = jnp.zeros((kx, ix, il))  # Zero wind
        va = jnp.zeros((kx, ix, il))
        ta = jnp.ones((kx, ix, il)) * 288.0
        qa = jnp.ones((kx, ix, il)) * 0.008
        rh = jnp.ones((kx, ix, il)) * 0.7
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))
        stl_am = jnp.ones((ix, il)) * 290.0
        soilw_am = jnp.ones((ix, il)) * 0.5
        rsds = jnp.ones((ix, il)) * 400.0
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        ustr, vstr, shf, evap, rlus, hfluxn, tskin = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Even with zero wind, gustiness should produce some fluxes
        self.assertTrue(jnp.all(evap >= 0.0), "Evaporation should be non-negative")
        self.assertTrue(jnp.all(jnp.isfinite(shf)), "Heat flux should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(evap)), "Evaporation should be finite")
        # Wind stress should be zero with zero wind
        self.assertTrue(jnp.allclose(ustr, 0.0, atol=1e-10), "U wind stress should be zero with zero wind")
        self.assertTrue(jnp.allclose(vstr, 0.0, atol=1e-10), "V wind stress should be zero with zero wind")

    def test_evaporation_saturated_vs_dry_air(self):
        """Test that evaporation is higher with dry air than saturated air."""
        ix, il, kx = self.ix, self.il, self.kx

        # Common setup
        ua = jnp.ones((kx, ix, il)) * 5.0
        va = jnp.ones((kx, ix, il)) * 5.0
        ta = jnp.ones((kx, ix, il)) * 288.0
        rh = jnp.ones((kx, ix, il)) * 0.7
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))
        stl_am = jnp.ones((ix, il)) * 290.0
        soilw_am = jnp.ones((ix, il)) * 1.0  # Fully saturated soil
        rsds = jnp.ones((ix, il)) * 400.0
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        # Case 1: Dry air (30% RH)
        #rh_dry = jnp.ones((kx, ix, il)) * 0.3
        qa_dry = jnp.ones((kx, ix, il)) * get_qsat(ta, psa, 1) * 0.3
        _, _, _, evap_dry, _, _, _ = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa_dry, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Case 2: Nearly saturated air (95% RH)
        #rh_sat = jnp.ones((kx, ix, il)) * 0.95
        qa_moist = jnp.ones((kx, ix, il)) * get_qsat(ta, psa, 1) * 0.95
        _, _, _, evap_sat, _, _, _ = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa_moist, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Dry air should produce more evaporation than saturated air
        self.assertTrue(jnp.all(evap_dry > evap_sat),
                       f"Evaporation with dry air (mean={jnp.mean(evap_dry)}) should exceed "
                       f"evaporation with saturated air (mean={jnp.mean(evap_sat)})")

    def test_no_negative_values_in_all_outputs(self):
        """Test that certain outputs (evap, rlus) are never negative."""
        ix, il, kx = self.ix, self.il, self.kx

        # Use varied but realistic inputs
        ua = jnp.ones((kx, ix, il)) * 5.0
        va = jnp.ones((kx, ix, il)) * 3.0
        ta = jnp.ones((kx, ix, il)) * 288.0
        qa = jnp.ones((kx, ix, il)) * 0.008
        rh = jnp.ones((kx, ix, il)) * 0.6
        phi = jnp.ones((kx, ix, il)) * 5000.0
        phi0 = jnp.zeros((ix, il))
        psa = jnp.ones((ix, il))
        fmask = jnp.ones((ix, il))
        stl_am = jnp.ones((ix, il)) * 285.0
        soilw_am = jnp.ones((ix, il)) * 0.4
        rsds = jnp.ones((ix, il)) * 500.0
        rlds = jnp.ones((ix, il)) * 350.0
        alb_l = jnp.ones((ix, il)) * 0.2
        snowc = jnp.zeros((ix, il))

        geometry = self.convert_to_speedy_latitudes(
            self.Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/self.grav, fmask=fmask)
        )

        u0 = self.parameters.surface_flux.fwind0 * ua[kx-1]
        v0 = self.parameters.surface_flux.fwind0 * va[kx-1]
        esbc = self.parameters.mod_radcon.emisfc * self.sbc
        ghum0 = 1.0 - self.parameters.surface_flux.fhum0

        ustr, vstr, shf, evap, rlus, hfluxn, tskin = self.compute_land_surface_fluxes(
            u0=u0, v0=v0, ua=ua, va=va, ta=ta, qa=qa, rh=rh,
            phi=phi, phi0=phi0, psa=psa, fmask=fmask,
            stl_am=stl_am, soilw_am=soilw_am,
            rsds=rsds, rlds=rlds, alb_l=alb_l, snowc=snowc,
            phis0=geometry.phis0, wvi=geometry.wvi, sigl=geometry.sigl, coa=geometry.coa,
            parameters=self.parameters, esbc=esbc, ghum0=ghum0
        )

        # Check non-negativity of physically constrained variables
        self.assertTrue(jnp.all(evap >= 0.0), f"Evaporation should be non-negative, min={jnp.min(evap)}")
        self.assertTrue(jnp.all(rlus >= 0.0), f"Upward LW radiation should be non-negative, min={jnp.min(rlus)}")
        self.assertTrue(jnp.all(tskin > 0.0), f"Skin temperature should be positive, min={jnp.min(tskin)}")
        # Check all outputs are finite
        self.assertTrue(jnp.all(jnp.isfinite(ustr)), "Wind stress U should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(vstr)), "Wind stress V should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(shf)), "Sensible heat flux should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(evap)), "Evaporation should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(rlus)), "Upward LW radiation should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(hfluxn)), "Net heat flux should be finite")

