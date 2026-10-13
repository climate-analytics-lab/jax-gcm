import unittest
import jax.numpy as jnp
import jax
import functools
from jax.test_util import check_vjp, check_jvp

from jcm.testing import check_gradients

class TestHumidityUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ConvectionData, PhysicsData, PhysicsState, get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum, fsg, PhysicsTendency, \
        SurfaceFluxData, HumidityData, SWRadiationData, LWRadiationData, parameters, ForcingData, coords, speedy_coords, terrain
        from jcm.physics.speedy.physics_data import ConvectionData, PhysicsData, SurfaceFluxData, HumidityData, SWRadiationData, LWRadiationData
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.physics.clouds.speedy_humidity import get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords

        parameters = Parameters.default()
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords)
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)

        self.temp_standard = jnp.ones((kx,ix,il))*273
        self.pressure_standard = jnp.ones((ix,il)) # normalized surface pressure
        self.sigma = 4
        self.qg_standard = jnp.ones((kx,ix,il))*2

    def test_spec_hum_to_rel_hum_isnan_ones(self):
        xy = (ix, il)
        zxy = (kx, ix, il)
        
        psa = jnp.ones((ix,il)) #surface pressure
        ua = jnp.ones(((kx, ix, il))) #zonal wind
        va = jnp.ones(((kx, ix, il))) #meridional wind
        ta = 288. * jnp.ones(((kx, ix, il))) #temperature
        qa = 5. * jnp.ones(((kx, ix, il))) #temperature
        rh = 0.8 * jnp.ones(((kx, ix, il))) #relative humidity
        phi = 5000. * jnp.ones(((kx, ix, il))) #geopotential
        sea_surface_temperature = 290. * jnp.ones((ix, il)) #ssts
        rsds = 400. * jnp.ones((ix, il)) #surface downward shortwave
        rlds = 400. * jnp.ones((ix, il)) #surface downward longwave

        forcing = ForcingData.ones(xy,sea_surface_temperature=sea_surface_temperature)
            
        state = PhysicsState.zeros(zxy,ua, va, ta, qa, phi, psa)
        sflux_data = SurfaceFluxData.zeros(xy, rlds=rlds)
        hum_data = HumidityData.zeros(xy,kx,rh=rh)
        conv_data = ConvectionData.zeros(xy,kx)
        sw_rad = SWRadiationData.zeros(xy,kx,rsds=rsds)
        lw_rad = LWRadiationData.zeros(xy,kx)
        physics_data = PhysicsData.zeros(xy,kx,convection=conv_data,humidity=hum_data,surface_flux=sflux_data,shortwave_rad=sw_rad,longwave_rad=lw_rad, speedy_coords=speedy_coords)

        _, f_vjp = jax.vjp(spec_hum_to_rel_hum, state, physics_data, parameters, forcing, terrain) 
        
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy, kx, speedy_coords=speedy_coords)
        input = (tends, datas)
        
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp(input)

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
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data, speedy_coords=speedy_coords)
        forcing = ForcingData.ones(xy)

        # Edge case: Zero Specific Humidity
        qg = jnp.ones((kx,ix,il))*0
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, terrain=terrain)
        self.assertTrue((physics_data.humidity.rh == 0).all(), "Relative humidity should be 0 when specific humidity is 0")

        # Edge case: Very High Temperature
        temp = jnp.ones((kx,ix,il))*400
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, terrain=terrain)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high temperatures")

        # Edge case: Extremely High Pressure
        pressure = jnp.ones((ix,il))*10
        state.normalized_surface_pressure = pressure
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, terrain=terrain)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high pressures")

        # Edge case: High Specific Humidity (near saturation)
        pressure = self.pressure_standard
        temp = self.temp_standard
        qg = jnp.ones((kx,ix,il))*(physics_data.humidity.qsat[:, 0, 0][:, jnp.newaxis, jnp.newaxis] - 1e-6)
        state = state.copy(specific_humidity=qg)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, terrain=terrain)
        self.assertTrue((physics_data.humidity.rh >= 0.99).all() and (physics_data.humidity.rh <= 1).all(), "Relative humidity should be close to 1 when specific humidity is near qsat")

    def test_rh_capped_at_cold_top_level(self):
        """Regression for #407: at SPEEDY's top sigma layer (0.025) in deep
        polar night, temperature can fall to ~125-150 K; the ice-Tetens
        ``qsat`` then collapses many orders of magnitude faster than any
        residual specific humidity, so the raw ratio would report millions
        of percent (measured 1.64e6 on the ``speedy-t31`` release-validation
        year). ``spec_hum_to_rel_hum`` must cap the *published* ``rh``
        in this (stratospheric) layer, leaving ``qsat`` itself -- and so
        every downstream tendency -- untouched.
        """
        from jcm.physics.clouds.speedy_humidity import _RH_SANITY_MAX

        top_sigma = float(speedy_coords.fsg[0])
        self.assertAlmostEqual(top_sigma, 0.025, places=3)

        zxy = (kx, ix, il)
        xy = (ix, il)

        temp = jnp.ones(zxy) * 273.0
        temp = temp.at[0].set(126.0)  # the release-validation extreme
        qg = jnp.ones(zxy) * 5e-4  # g/kg: a tiny but non-zero residual
        psa = self.pressure_standard

        convection_data = ConvectionData.zeros(xy, kx)
        physics_data = PhysicsData.zeros(xy, kx, convection=convection_data, speedy_coords=speedy_coords)
        forcing = ForcingData.ones(xy)
        state = PhysicsState.zeros(
            zxy, temperature=temp, specific_humidity=qg,
            normalized_surface_pressure=psa,
        )

        _, physics_data = spec_hum_to_rel_hum(
            physics_data=physics_data, state=state, parameters=parameters,
            forcing=forcing, terrain=terrain,
        )

        qsat_top = physics_data.humidity.qsat[0]
        rh_top = physics_data.humidity.rh[0]

        # qsat is NOT floored or capped -- it keeps following the reference
        # Fortran's formula verbatim, and is genuinely tiny at this T.
        self.assertTrue((qsat_top > 0).all())
        self.assertTrue((qsat_top < 1e-6).all())

        # The uncapped ratio at these inputs is several orders of magnitude
        # above the sanity ceiling; the published rh must sit at the
        # ceiling, not the diverging raw value.
        raw_ratio = qg[0] / qsat_top
        self.assertTrue((raw_ratio > 100.0 * _RH_SANITY_MAX).all())
        self.assertTrue(jnp.allclose(rh_top, _RH_SANITY_MAX))

        # Away from the pathological top level the cap must not perturb an
        # ordinary, physically unremarkable relative humidity.
        rh_mid = physics_data.humidity.rh[self.sigma]
        self.assertTrue((rh_mid < 1.0).all())

    def test_rh_cap_scoped_to_stratosphere_only(self):
        """The cap must not touch any level ``speedy_shortwave.py``'s
        ``get_clouds`` reads ``humidity.rh`` from for its own cloud-top
        search (#407's Codex review): that search excludes only the
        ``stratosphere_mask`` levels, so capping further down would change
        cloud diagnosis, not just the reported diagnostic. Reproduce the
        same pathology one level below the stratosphere (sigma ~0.2, still
        cold enough that qsat collapses) and confirm the raw, uncapped
        ratio survives even though it exceeds the sanity ceiling.
        """
        from jcm.physics.clouds.speedy_humidity import _RH_SANITY_MAX
        from jcm.physics.speedy.speedy_coords import stratosphere_mask

        strat = stratosphere_mask(speedy_coords.fsg)
        # The lowest non-stratospheric level (just below the mask) is the
        # one speedy_shortwave.py still searches over.
        level = int(jnp.argmin(jnp.where(strat, 1, speedy_coords.fsg)))
        self.assertFalse(bool(strat[level]))

        zxy = (kx, ix, il)
        xy = (ix, il)

        temp = jnp.ones(zxy) * 273.0
        temp = temp.at[level].set(140.0)  # cold enough to collapse qsat here too
        qg = jnp.ones(zxy) * 5e-4
        psa = self.pressure_standard

        convection_data = ConvectionData.zeros(xy, kx)
        physics_data = PhysicsData.zeros(xy, kx, convection=convection_data, speedy_coords=speedy_coords)
        forcing = ForcingData.ones(xy)
        state = PhysicsState.zeros(
            zxy, temperature=temp, specific_humidity=qg,
            normalized_surface_pressure=psa,
        )

        _, physics_data = spec_hum_to_rel_hum(
            physics_data=physics_data, state=state, parameters=parameters,
            forcing=forcing, terrain=terrain,
        )

        qsat_here = physics_data.humidity.qsat[level]
        rh_here = physics_data.humidity.rh[level]
        raw_ratio = qg[level] / qsat_here

        # The pathology is reproduced (raw ratio is pathologically large)...
        self.assertTrue((raw_ratio > _RH_SANITY_MAX).all())
        # ...but this level is left exactly as the reference computes it:
        # speedy_shortwave.py's cloud-top search still reads this value.
        self.assertTrue(jnp.allclose(rh_here, raw_ratio))

    def test_rel_hum_to_spec_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard
        zxy = (kx,ix,il)
        xy = (ix,il)
        forcing = ForcingData.ones(xy)

        convection_data = ConvectionData.zeros((ix,il), kx)
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data, speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg,normalized_surface_pressure=pressure)

        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, terrain=terrain)
        qa, qsat = rel_hum_to_spec_hum(temp[0], pressure, speedy_coords.fsg[0], physics_data.humidity.rh[0])
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
        physics_data = PhysicsData.zeros((ix,il), kx, convection=convection_data, speedy_coords=speedy_coords)
        state = PhysicsState.zeros(zxy,temperature=temp, specific_humidity=qg,normalized_surface_pressure=pressure)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state, parameters=parameters, forcing=forcing, terrain=terrain)

        def f(temp_0, pressure, geometry_fsg, physics_data_h_rh):
            return rel_hum_to_spec_hum(temp_0, pressure, geometry_fsg, physics_data_h_rh)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (temp[0], pressure, speedy_coords.fsg[0], physics_data.humidity.rh[0]), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (temp[0], pressure, speedy_coords.fsg[0], physics_data.humidity.rh[0]), 
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
        physics_data = PhysicsData.ones((ix,il), kx, convection=convection_data, speedy_coords=speedy_coords)
        forcing = ForcingData.ones(xy)
        # Edge case: Zero Specific Humidity
        qg = jnp.ones((kx,ix,il))*0
        state = PhysicsState.ones(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = spec_hum_to_rel_hum(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out.humidity)
        
        # Each edge case below re-checks the same f at a different state.
        # Worst agreement across the four edge cases and five perturbation
        # directions is 3.4e-3, against the rtol=1 the fixed-step pairs needed.
        # Tolerances here are set from the worst direction, not the committed
        # one: adding a PhysicsData leaf does not move existing leaves'
        # directions any more, but it does add a component to the projection.
        check_gradients(f, (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), rtol=1e-2)


        # Edge case: Very High Temperature
        temp = jnp.ones((kx,ix,il))*330
        state = PhysicsState.ones(zxy,temperature=temp, specific_humidity=qg, normalized_surface_pressure=pressure)
        state_floats = convert_to_float(state)
        check_gradients(f, (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), rtol=1e-2)


        # Edge case: Extremely High Pressure
        pressure = jnp.ones((ix,il))*10
        state.normalized_surface_pressure = pressure
        state_floats = convert_to_float(state)
        check_gradients(f, (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), rtol=1e-2)


        # Edge case: High Specific Humidity (near saturation)
        pressure = self.pressure_standard
        temp = self.temp_standard
        qg = jnp.ones((kx,ix,il))*(physics_data.humidity.qsat[:, 0, 0][:, jnp.newaxis, jnp.newaxis] - 1e-6)
        state = state.copy(specific_humidity=qg)
        state_floats = convert_to_float(state)
        check_gradients(f, (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), rtol=1e-2)
        

if __name__ == '__main__':
    unittest.main()