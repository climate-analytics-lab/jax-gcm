import unittest
import jax.numpy as jnp
import jax
import functools
from jax.test_util import check_vjp, check_jvp

class TestHumidityUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ConvectionData, PhysicsData, PhysicsState, get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum, fsg, PhysicsTendency, \
        SurfaceFluxData, HumidityData, SWRadiationData, LWRadiationData, parameters, ForcingData, Geometry, convert_to_speedy_latitudes, default_geometry
        from jcm.physics.speedy.physics_data import ConvectionData, PhysicsData, SurfaceFluxData, HumidityData, SWRadiationData, LWRadiationData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.speedy.humidity import get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        parameters = Parameters.default()
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        default_geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx))

        self.temp_standard = jnp.ones((kx,ix,il))*273
        self.pressure_standard = jnp.ones((ix,il)) # normalized surface pressure
        self.sigma = 4
        self.qg_standard = jnp.ones((kx,ix,il))*2

    def test_spec_hum_to_rel_hum_isnan_ones(self):
        from jcm.constants import grav
        xy = (ix, il)
        zxy = (kx, ix, il)
        
        psa = jnp.ones((ix,il)) #surface pressure
        ua = jnp.ones(((kx, ix, il))) #zonal wind
        va = jnp.ones(((kx, ix, il))) #meridional wind
        ta = 288. * jnp.ones(((kx, ix, il))) #temperature
        qa = 5. * jnp.ones(((kx, ix, il))) #temperature
        rh = 0.8 * jnp.ones(((kx, ix, il))) #relative humidity
        phi = 5000. * jnp.ones(((kx, ix, il))) #geopotential
        phi0 = 500. * jnp.ones((ix, il)) #surface geopotential
        fmask = 0.5 * jnp.ones((ix, il)) #land fraction mask
        tsea = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave

        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx, orography=phi0/grav, fmask=fmask))
        forcing = ForcingData.ones(xy,tsea=tsea)
            
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy, rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad)

        _, f_vjp = jax.vjp(spec_hum_to_rel_hum, state, physics_data, parameters, forcing, geometry) 
        
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx)
        input = (tends, datas)
        
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_get_qsat(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        sigma = self.sigma
        qsat = get_qsat(temp[sigma], pressure, sigma)

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
        zxy = (kx,ix,il)
        xy = (ix,il)

        convection_data = ConvectionData.zeros((ix,il), kx)
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data)
        forcing = ForcingData.ones(xy)

        # Edge case: Zero Specific Humidity
        qg = jnp.ones((kx,ix,il))*0
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, geometry=default_geometry)
        self.assertTrue((physics_data.humidity.rh == 0).all(), "Relative humidity should be 0 when specific humidity is 0")

        # Edge case: Very High Temperature
        temp = jnp.ones((kx,ix,il))*400
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, geometry=default_geometry)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high temperatures")

        # Edge case: Extremely High Pressure
        pressure = jnp.ones((ix,il))*10
        state.normalized_surface_pressure = pressure
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, geometry=default_geometry)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high pressures")

        # Edge case: High Specific Humidity (near saturation)
        pressure = self.pressure_standard
        temp = self.temp_standard
        qg = jnp.ones((kx,ix,il))*(physics_data.humidity.qsat[:, 0, 0][:, jnp.newaxis, jnp.newaxis] - 1e-6)
        state = state.copy(specific_humidity=qg)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, geometry=default_geometry)
        self.assertTrue((physics_data.humidity.rh >= 0.99).all() and (physics_data.humidity.rh <= 1).all(), "Relative humidity should be close to 1 when specific humidity is near qsat")

    def test_rel_hum_to_spec_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard
        zxy = (kx,ix,il)
        xy = (ix,il)
        forcing = ForcingData.ones(xy)

        convection_data = ConvectionData.zeros((ix,il), kx)
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data)
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg,normalized_surface_pressure=pressure)

        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, geometry=default_geometry)
        qa, qsat = rel_hum_to_spec_hum(temp[0], pressure, default_geometry.fsg[0], physics_data.humidity.rh[0])
        # Allow a small tolerance for floating point comparisons
        tolerance = 1e-6
        self.assertTrue(jnp.allclose(qa, qg[0], atol=tolerance), "QA should be close to the original QG when converted from RH")

    # Gradient checks
        
    def test_get_qsat_gradient_check(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        sigma = self.sigma

        def f(temp_s, pressure):
            return get_qsat(temp_s, pressure, sigma) #(jnp.round(sigma)).astype(int)
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (temp[sigma], pressure), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (temp[sigma], pressure), 
                                atol=None, rtol=1, eps=0.000001)
        
        # Edge case: Very low temperature
        temp = jnp.ones((ix,il))*100
        check_vjp(f, f_vjp, args = (temp, pressure), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (temp, pressure), 
                                atol=None, rtol=1, eps=0.000001)


        # Edge case: Very high temperature
        temp = jnp.ones((ix,il))*350
        check_vjp(f, f_vjp, args = (temp, pressure), 
                                atol=None, rtol=1, eps=0.00001)
        # Test fails
        check_jvp(f, f_jvp, args = (temp, pressure), 
                                atol=None, rtol=1, eps=0.000001)


    def test_rel_hum_to_spec_hum_gradient_check(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard
        zxy = (kx,ix,il)
        xy = (ix,il)
        forcing = ForcingData.ones(xy)
        convection_data = ConvectionData.zeros((ix,il), kx)
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data)
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg,normalized_surface_pressure=pressure)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, geometry=default_geometry)

        def f(temp_0, pressure, geometry_fsg, physics_data_h_rh):
            return rel_hum_to_spec_hum(temp_0, pressure, geometry_fsg, physics_data_h_rh)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (temp[0], pressure, default_geometry.fsg[0], physics_data.humidity.rh[0]), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (temp[0], pressure, default_geometry.fsg[0], physics_data.humidity.rh[0]), 
                                atol=None, rtol=1, eps=0.000001)
        
    def test_spec_hum_to_rel_hum_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard
        zxy = (kx,ix,il)
        xy = (ix,il)
        # Set inputs
        convection_data = ConvectionData.ones((ix,il), kx)
        physics_data = PhysicsData.ones((ix,il), kx, convection=convection_data)
        forcing = ForcingData.ones(xy)
        # Edge case: Zero Specific Humidity
        qg = jnp.ones((kx,ix,il))*0
        state = PhysicsState.ones(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(default_geometry)

        def f(physics_data_f, state_f, parameters_f, forcing_f,geometry_f):
            tend_out, data_out = spec_hum_to_rel_hum(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, default_geometry)
                                       )
            return convert_to_float(data_out.humidity)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)


        # Edge case: Very High Temperature
        temp = jnp.ones((kx,ix,il))*400
        state = PhysicsState.ones(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)
        state_floats = convert_to_float(state)
        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.0001)


        # Edge case: Extremely High Pressure
        pressure = jnp.ones((ix,il))*10
        state.normalized_surface_pressure = pressure
        state_floats = convert_to_float(state)
        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)


        # Edge case: High Specific Humidity (near saturation)
        pressure = self.pressure_standard
        temp = self.temp_standard
        qg = jnp.ones((kx,ix,il))*(physics_data.humidity.qsat[:, 0, 0][:, jnp.newaxis, jnp.newaxis] - 1e-6)
        state = state.copy(specific_humidity=qg)
        state_floats = convert_to_float(state)
        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.000001)
        

if __name__ == '__main__':
    unittest.main()