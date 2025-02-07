import unittest
import jax.numpy as jnp
import jax

class TestHumidityUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)

        global ConvectionData, PhysicsData, PhysicsState, get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum, fsg, PhysicsTendency, \
        SurfaceFluxData, HumidityData, SWRadiationData, LWRadiationData, SeaModelData, parameters, BoundaryData
        from jcm.physics_data import ConvectionData, PhysicsData, SurfaceFluxData, HumidityData, SWRadiationData, LWRadiationData, SeaModelData
        from jcm.physics import PhysicsState, PhysicsTendency
        from jcm.humidity import get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum
        from jcm.geometry import fsg
        from jcm.boundaries import BoundaryData
        from jcm.params import Parameters
        parameters = Parameters.init()
        
        self.temp_standard = jnp.ones((ix,il,kx))*273
        self.pressure_standard = jnp.ones((ix,il))*0.5
        self.sigma = 4
        self.qg_standard = jnp.ones((ix,il,kx))*2

    def test_spec_hum_to_rel_hum_isnan_ones(self): 
        xy = (ix, il)
        xyz = (ix, il, kx)
        
        psa = jnp.ones((ix,il)) #surface pressure
        ua = jnp.ones(((ix, il, kx))) #zonal wind
        va = jnp.ones(((ix, il, kx))) #meridional wind
        ta = 288. * jnp.ones(((ix, il, kx))) #temperature
        qa = 5. * jnp.ones(((ix, il, kx))) #temperature
        rh = 0.8 * jnp.ones(((ix, il, kx))) #relative humidity
        phi = 5000. * jnp.ones(((ix, il, kx))) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave
        boundaries = BoundaryData.ones(xy,lfluxland=True)
            
        state = PhysicsState.zeros(xyz,ua, va, ta, qa, phi)
        sflux_data = SurfaceFluxData.zeros(xy)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx,psa=psa)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx,rlds=rlds)
        sea_data = SeaModelData.zeros(xy,tsea=tsea)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad, sea_model=sea_data)

        _, f_vjp = jax.vjp(spec_hum_to_rel_hum, state, physics_data, parameters, boundaries) 
        
        tends = PhysicsTendency.ones(xyz)
        datas = PhysicsData.ones(xy, kx)
        input = (tends, datas)
        
        df_dstates, df_ddatas, df_dparams, df_dboundaries = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.convection.isnan())
        self.assertFalse(df_dboundaries.isnan().any_true())

    def test_get_qsat(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        sigma = self.sigma
        qsat = get_qsat(temp[:,:,sigma], pressure, sigma)

        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values")

        # Edge case: Very low temperature
        temp = jnp.ones((ix,il))*100
        qsat = get_qsat(temp, pressure, sigma)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at low temperature")

        # Edge case: Very high temperature
        temp = jnp.ones((ix,il))*350
        qsat = get_qsat(temp, pressure, sigma)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at high temperature")

    def test_spec_hum_to_rel_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard
        xyz = (ix,il,kx)
        xy = (ix,il)

        convection_data = ConvectionData.zeros((ix,il), kx, psa=pressure)
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data)
        state = PhysicsState.zeros(xyz,temperature=temp, specific_humidity=qg)
        boundaries = BoundaryData.ones(xy)

        # Edge case: Zero Specific Humidity
        qg = jnp.ones((ix,il,kx))*0
        state = PhysicsState.zeros(xyz,temperature=temp, specific_humidity=qg)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, boundaries=boundaries)
        self.assertTrue((physics_data.humidity.rh == 0).all(), "Relative humidity should be 0 when specific humidity is 0")

        # Edge case: Very High Temperature
        temp = jnp.ones((ix,il,kx))*400
        state = PhysicsState.zeros(xyz,temperature=temp, specific_humidity=qg)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, boundaries=boundaries)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high temperatures")

        # Edge case: Extremely High Pressure
        pressure = jnp.ones((ix,il))*10
        convection_data = convection_data.copy(psa=pressure)
        physics_data = physics_data.copy(convection=convection_data)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, boundaries=boundaries)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high pressures")

        # Edge case: High Specific Humidity (near saturation)
        qg = jnp.ones((ix,il,kx))*(physics_data.humidity.qsat[0, 0, :] - 1e-6)
        state = PhysicsState.zeros(xyz,temperature=temp, specific_humidity=qg)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, boundaries=boundaries)
        self.assertTrue((physics_data.humidity.rh >= 0.99).all() and (physics_data.humidity.rh <= 1).all(), "Relative humidity should be close to 1 when specific humidity is near qsat")

    def test_rel_hum_to_spec_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard
        xyz = (ix,il,kx)
        xy = (ix,il)
        boundaries = BoundaryData.ones(xy)

        convection_data = ConvectionData.zeros((ix,il), kx, psa=pressure)
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data)
        state = PhysicsState.zeros(xyz,temperature=temp, specific_humidity=qg,surface_pressure=pressure)

        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, boundaries=boundaries)
        qa, qsat = rel_hum_to_spec_hum(temp[:,:,0], pressure, fsg[0], physics_data.humidity.rh[:,:,0])
        # Allow a small tolerance for floating point comparisons
        tolerance = 1e-6
        self.assertTrue(jnp.allclose(qa, qg[:,:,0], atol=tolerance), "QA should be close to the original QG when converted from RH")

if __name__ == '__main__':
    unittest.main()