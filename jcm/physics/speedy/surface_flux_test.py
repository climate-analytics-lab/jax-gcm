import unittest
import jax
import jax.numpy as jnp
class TestSurfaceFluxesUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global BoundaryData, SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData, \
               PhysicsState, get_surface_fluxes, get_orog_land_sfc_drag, PhysicsTendency, parameters, geometry, grav
        from jcm.boundaries import BoundaryData
        from jcm.physics.speedy.physics_data import SurfaceFluxData, HumidityData, ConvectionData, SWRadiationData, LWRadiationData, PhysicsData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.params import Parameters
        from jcm.geometry import Geometry
        from jcm.constants import grav
        parameters = Parameters.default()
        geometry = Geometry.from_grid_shape((ix, il), kx)
        
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
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        lfluxland=True
            
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.zeros(xy,tsea=tsea,orog=phi0/grav, fmask=fmask, lfluxland=lfluxland)

        _, f_vjp = jax.vjp(get_surface_fluxes, state, physics_data, parameters, boundaries, geometry)

        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx)

        input_tensors = (tends, datas)

        df_dstate, df_ddatas, df_dparams, df_dboundaries, df_dgeometry = f_vjp(input_tensors)
        
        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstate.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dboundaries.isnan().any_true())

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
        tsea = jnp.ones((ix,il)) * 292 # this needs to overwrite what is in sea_model?
        rsds = 400 * jnp.ones(xy)
        rlds = 400 * jnp.ones(xy)
        lfluxland = True
        soilw_am = 0.5* jnp.ones(((ix,il,365)))
        state = PhysicsState.zeros(zxy, ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.ones(xy,tsea=tsea,orog=phi0/grav, fmask=fmask, lfluxland=lfluxland, soilw_am=soilw_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, boundaries, geometry)
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
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        lfluxland=True
        soilw_am = 0.5* jnp.ones(((ix,il,365)))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.zeros(xy,tsea=tsea, lfluxland=lfluxland,fmask=fmask, orog=phi0/grav,soilw_am=soilw_am)
        _, physics_data = get_surface_fluxes(state, physics_data, parameters, boundaries, geometry)
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
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        lfluxland=True
        soilw_am = 0.5* jnp.ones(((ix,il,365)))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.zeros(xy,tsea=tsea,orog=phi0/grav, fmask=fmask,lfluxland=lfluxland, soilw_am=soilw_am)
        
        _, physics_data = get_surface_fluxes(state, physics_data, parameters, boundaries, geometry)
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
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        lfluxland=True
        soilw_am = 0.5* jnp.ones(((ix,il,365)))
        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.zeros(xy,tsea=tsea,orog=phi0/grav, fmask=fmask,lfluxland=lfluxland, soilw_am=soilw_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, boundaries, geometry)
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
        soilw_am = 0.5* jnp.ones(((ix,il,365)))
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        lfluxland=True

        # vars = get_surface_fluxes(psa,ua,va,ta,qa,rh,phi,phi0,fmask,tsea,rsds,rlds,lfluxland)
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.zeros(xy,tsea=tsea,orog=phi0/grav,fmask=fmask,lfluxland=lfluxland, soilw_am=soilw_am)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, boundaries, geometry)
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
        soilw_am = 0.5* jnp.ones(((ix,il,365)))
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        lfluxland=True

        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy,rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)
        boundaries = BoundaryData.zeros(xy,tsea=tsea,orog=phi0/grav,fmask=fmask,soilw_am=soilw_am,lfluxland=lfluxland)

        _, physics_data = get_surface_fluxes(state, physics_data, parameters, boundaries, geometry)
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