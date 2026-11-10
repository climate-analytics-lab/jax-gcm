import unittest
import jax.numpy as jnp
import numpy as np
import jax

from jcm.testing import check_gradients

def initialize_arrays(ix, il, kx):
    # Initialize arrays
    ta = jnp.zeros((kx, ix, il))
    rlds = jnp.zeros((ix, il))
    st4a = jnp.zeros((kx, ix, il, 2))     # Blackbody emission from full and half atmospheric levels
    flux = jnp.zeros((ix, il, 4))         # Radiative flux in different spectral bands

    # Set the min and max values
    min_val = 130.0
    max_val = 250.0
    
    # Calculate step size
    total_elements = ix * il * kx
    step_size = (max_val - min_val) / (total_elements - 1)

    ta = min_val + step_size*jnp.arange(total_elements).reshape((kx, il, ix)).transpose((0, 2, 1))
    
    return ta, rlds, st4a, flux

class TestLongwave(unittest.TestCase):
    
    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8

        global ModRadConData, LWRadiationData, SurfaceFluxData, PhysicsData, PhysicsState, PhysicsTendency, ForcingData, get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes, radset, parameters, terrain, speedy_coords
        from jcm.physics.speedy.physics_data import ModRadConData, LWRadiationData, SurfaceFluxData, PhysicsData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.forcing import ForcingData
        from jcm.physics.radiation.speedy_longwave import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes, radset
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords

        parameters = Parameters.default()
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords)
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)


    def test_downward_longwave_rad_fluxes(self):

        # FIXME: This array doesn't need to be this big once we fix the interfaces
        # -> We only test the first 5x5 elements
        zxy = (kx, ix, il)
        xy = (ix, il)
        ta, rlds, st4a, flux = initialize_arrays(ix, il, kx)
        mod_radcon = ModRadConData.zeros((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData.zeros((ix, il), kx, mod_radcon=mod_radcon,speedy_coords=speedy_coords)
        forcing = ForcingData.ones(xy)
        
        state = PhysicsState.zeros(zxy,temperature=ta)
        
        _, physics_data = get_downward_longwave_rad_fluxes(state, physics_data, parameters, forcing, terrain)

        # fortran values
        # print(rlds[:5, :5])
        f90_rlds = [[186.6984  , 187.670515, 188.646319, 189.625957, 190.609469],
                    [186.708473, 187.680627, 188.656572, 189.636231, 190.6197  ],
                    [186.718628, 187.69074 , 188.666658, 189.646441, 190.630014],
                    [186.728719, 187.700953, 188.676876, 189.656632, 190.640263],
                    [186.738793, 187.711066, 188.687129, 189.666908, 190.650495]]
        
        # print(dfabs[0, 0, :])
        f90_dfabs = [ -3.799531,
                     -20.11071 ,
                     -17.83563 ,
                     -17.667264,
                     -22.200773,
                     -27.997842,
                     -33.615657,
                     -47.10823 ]
        
        # print(np.mean(mod_radcon.st4a[:5,:5,:,:], axis=2))
        f90_st4a = [[[76.56151, 9.97944],
                     [77.0403 ,10.02566],
                     [77.5214 ,10.07201],
                     [78.0048 ,10.11851],
                     [78.49052,10.16516]],
                    [[76.56649, 9.97992],
                     [77.04531,10.02614],
                     [77.52642,10.0725 ],
                     [78.00985,10.119  ],
                     [78.4956 ,10.16564]],
                    [[76.57147, 9.9804 ],
                     [77.0503 ,10.02662],
                     [77.53144,10.07297],
                     [78.01489,10.11948],
                     [78.50067,10.16613]],
                    [[76.57644, 9.98088],
                     [77.0553 ,10.0271 ],
                     [77.53647,10.07346],
                     [78.01994,10.11996],
                     [78.50574,10.16662]],
                    [[76.58142, 9.98136],
                     [77.0603 ,10.02758],
                     [77.54149,10.07395],
                     [78.02499,10.12045],
                     [78.51081,10.1671 ]]]
        
        self.assertTrue(np.allclose(physics_data.surface_flux.rlds[:5, :5], np.asarray(f90_rlds), atol=1e-4))
        self.assertTrue(np.allclose(physics_data.longwave_rad.dfabs[:, 0, 0], f90_dfabs, atol=1e-4))
        self.assertTrue(np.allclose(np.mean(physics_data.mod_radcon.st4a[:, :5, :5, :], axis=0), np.asarray(f90_st4a), atol=1e-4))

    def test_upward_longwave_rad_fluxes(self):
        ta = jnp.ones((kx, ix, il)) * 300
        ts = jnp.ones((ix, il)) * 300
        rlds = jnp.ones((ix, il))
        rlus = jnp.ones((ix, il))
        dfabs = jnp.ones((kx, ix, il))
        st4a = jnp.ones((kx, ix, il, 2))
        flux = jnp.ones((ix, il, 4))
        tau2 = jnp.ones((kx, ix, il, 4)) + jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * .1
        stratc = jnp.ones((ix, il, 2))

        state = PhysicsState.zeros((ix, il), kx).copy(temperature=ta)
        input_physics_data = PhysicsData.zeros((ix, il), kx, speedy_coords=speedy_coords).copy(
            longwave_rad=LWRadiationData.zeros((ix, il), kx).copy(dfabs=dfabs),
            mod_radcon=ModRadConData.zeros((ix, il), kx).copy(st4a=st4a, flux=flux, tau2=tau2, emis2=1 - tau2, stratc=stratc),
            surface_flux=SurfaceFluxData.zeros((ix, il)).copy(rlus=rlus, rlds=rlds, tsfc=ts),
        )

        # skip testing ttend since we have access to dfabs
        _, output_physics_data = get_upward_longwave_rad_fluxes(state=state, physics_data=input_physics_data, parameters=parameters, forcing=ForcingData.zeros((ix, il)), terrain=terrain)

        fsfc = output_physics_data.surface_flux.rlns
        ftop = output_physics_data.longwave_rad.ftop
        dfabs = output_physics_data.longwave_rad.dfabs
        flux = output_physics_data.mod_radcon.flux

        fsfc_f90 = 0.0
        ftop_f90 = -2.037812334328966
        dfabs_f90 = jnp.array([-5e-2, 1.2938921, 1.6556535, 1.7784461, 1.7685201, 1.6721002, 1.5397001, 1.4595001])
        flux_f90 = jnp.array([-0.55618826, -0.35694631, -1.20774518, -1.05693259])

        self.assertTrue(jnp.allclose(fsfc[0, 0], fsfc_f90, atol=1e-5))
        self.assertTrue(jnp.allclose(ftop[0, 0], ftop_f90, atol=1e-5))
        self.assertTrue(jnp.allclose(dfabs[:, 0, 0], dfabs_f90, atol=1e-5))
        self.assertTrue(jnp.allclose(flux[0, 0, :], flux_f90, atol=1e-5))

    def test_get_downward_longwave_rad_fluxes_gradients_isnan_ones(self):
        """Test that we can calculate gradients of longwave radiation without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state = PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)
        # Calculate gradient
        _, f_vjp = jax.vjp(get_downward_longwave_rad_fluxes, state, physics_data, parameters, forcing, terrain)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())
       

    def test_get_upward_longwave_rad_fluxes_gradients_isnan_ones(self):
        """Test that we can calculate gradients of longwave radiation without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)  # Create PhysicsData object (parameter)
        state = PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)

        # Calculate gradient
        _, f_vjp = jax.vjp(get_upward_longwave_rad_fluxes, state, physics_data, parameters, forcing, terrain)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx,speedy_coords=speedy_coords)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dterrain = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_radset_gradient_check(self):
        zxy = (kx, ix, il)
        # Inside radset's 200-320 K clip, where it is differentiable in the
        # temperature as well as in its coefficients.
        temp = PhysicsState.ones(zxy).temperature * 260.0
        mod_radcon = parameters.mod_radcon

        def f(temp, mod_radcon):
            return radset(temp, mod_radcon)

        # Every band-fraction coefficient and epslw must carry a gradient:
        # they are what a fit of the scheme to a reference moves.
        check_gradients(
            f, (temp, mod_radcon), rtol=1e-2,
            live_inputs=["epslw", "fbco2_a", "fbco2_b", "fbco2_t0",
                         "fbwv1_a", "fbwv1_b", "fbwv1_t0",
                         "fbwv2_a", "fbwv2_b", "fbwv2_t0"])

    def test_radset_defaults_are_speedys_table(self):
        """The default coefficients are SPEEDY's radset table (radiation.f90)."""
        temp = jnp.array([180.0, 250.0, 300.0, 340.0])
        got = np.asarray(radset(temp, parameters.mod_radcon))
        t = np.clip(np.asarray(temp), 200.0, 320.0)
        f2 = 0.148 - 3.0e-6 * (t - 247.0) ** 2
        f3 = 0.356 - 5.2e-6 * (t - 282.0) ** 2
        f4 = 0.314 + 1.0e-5 * (t - 315.0) ** 2
        want = (1.0 - 0.05) * np.stack([1.0 - f2 - f3 - f4, f2, f3, f4], axis=-1)
        np.testing.assert_allclose(got, want, rtol=1e-6)
        # the four bands carry 1 - epslw of the spectrum at every temperature
        np.testing.assert_allclose(got.sum(axis=-1), 0.95, rtol=1e-6)

    def test_downward_longwave_rad_fluxes_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        # FIXME: This array doesn't need to be this big once we fix the interfaces
        # -> We only test the first 5x5 elements
        zxy = (kx, ix, il)
        xy = (ix, il)
        _, rlds, st4a, flux = initialize_arrays(ix, il, kx)
        # The shared 130-250 K ramp belongs to the Fortran regression test
        # above and is kept there, but it is not a point at which this scheme
        # is differentiable: ``radset`` clips its argument to [200, 320]
        # (speedy_longwave.py:265), so 58 % of that ramp sits on the constant
        # side of the clip and the level containing 200 K has ~60 of its 4608
        # points swept across the kink by the top rung. The one-sided secants
        # then disagree by 20-50 % at every rung. This ramp spans the same
        # 100 K over the same grid, placed wholly inside the clip's smooth
        # interval, and is an ordinary terrestrial column besides.
        n = kx * ix * il
        ta = 210.0 + (100.0 / (n - 1)) * jnp.arange(n).reshape(
            (kx, il, ix)).transpose((0, 2, 1))
        mod_radcon = ModRadConData.zeros((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData.zeros((ix, il), kx, mod_radcon=mod_radcon,speedy_coords=speedy_coords)
        forcing = ForcingData.ones(xy)
        state = PhysicsState.zeros(zxy,temperature=ta)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_downward_longwave_rad_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out)
        
        # The step is chosen by jcm.testing from the difference's own
        # self-consistency: st4a carries where-branches whose distance from
        # this point depends on the direction, so no fixed step is safe.
        # Measured agreement here is 7.1e-4.
        #
        # ``fsg`` alone is held fixed, for the same reason as in the upward
        # check below: it is the one sigma-grid leaf this scheme *selects* on,
        # through ``stratosphere_mask(fsg)`` whose sigma < 0.2 boundary the
        # 8-level grid sits exactly on. The rest of the grid (dhs, grdscp,
        # wvi) scales results and stays live, so its gradient is still
        # checked — freezing the whole subtree would have taken those three
        # out of both sides of the comparison as well.
        #
        # Two directions, neither the default seed. In the seed-0 direction
        # the projection all but cancels across the output leaves (-114,
        # against 1.7e3 and -9.2e3 here) while the float32 rounding of the
        # secants and their one-sided curvature stay the size they are in any
        # direction, so the coarse rungs fail the one-sided gate on curvature
        # alone and the ladder settles on a rung whose central difference is
        # rounding (2 % off; AD agrees with the 5e-4 rung to 2e-4). Both
        # directions used here converge on the first rung.
        for seed in (1, 2):
            check_gradients(f, (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), rtol=1e-2,
                            seed=seed, fixed_inputs=["speedy_coords/fsg"])


    def test_upward_longwave_rad_fluxes_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        ta = jnp.ones((kx, ix, il)) * 300
        ts = jnp.ones((ix, il)) * 300
        rlds = jnp.ones((ix, il))
        rlus = jnp.ones((ix, il))
        dfabs = jnp.ones((kx, ix, il))
        st4a = jnp.ones((kx, ix, il, 2))
        flux = jnp.ones((ix, il, 4))
        tau2 = jnp.ones((kx, ix, il, 4)) + jnp.arange(kx)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * .1
        stratc = jnp.ones((ix, il, 2))

        state = PhysicsState.zeros((ix, il), kx).copy(temperature=ta)
        input_physics_data = PhysicsData.zeros((ix, il), kx, speedy_coords=speedy_coords).copy(
            longwave_rad=LWRadiationData.zeros((ix, il), kx).copy(dfabs=dfabs),
            mod_radcon=ModRadConData.zeros((ix, il), kx).copy(st4a=st4a, flux=flux, tau2=tau2, emis2=1 - tau2, stratc=stratc),
            surface_flux=SurfaceFluxData.zeros((ix, il)).copy(rlus=rlus, rlds=rlds, tsfc=ts),
        )
        forcing = ForcingData.zeros((ix, il))

        # Set float inputs
        physics_data_floats = convert_to_float(input_physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        terrain_floats = convert_to_float(terrain)

        def f(physics_data_f, state_f, parameters_f, forcing_f,terrain_f):
            tend_out, data_out = get_upward_longwave_rad_fluxes(physics_data=convert_back(physics_data_f, input_physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       terrain=convert_back(terrain_f, terrain)
                                       )
            return convert_to_float(data_out)
        
        # Same branch structure as the downward check on a smaller output
        # tree; measured agreement 9.9e-4.
        #
        # ``fsg`` is held fixed, and only ``fsg``. It is the sigma-grid leaf
        # this scheme *selects* on: ``stratosphere_mask(fsg)`` has its
        # sigma < 0.2 boundary exactly where the 8-level grid sits
        # (fsg[2] = 0.2), so displacing fsg moves a whole layer between the
        # stratospheric and tropospheric blackbody treatments, jumping ftop
        # and dfabs by ~4.8 and leaving the secant reporting jump/eps at every
        # rung above 4e-6. The grid's scaling leaves — dhs, grdscp and the
        # interpolation weights wvi — are ordinary differentiable inputs here
        # and are left live.
        check_gradients(f, (physics_data_floats, state_floats, parameters_floats, forcing_floats, terrain_floats), rtol=1e-2,
                        fixed_inputs=["speedy_coords/fsg"])





class TestModRadConDataEmis2Default(unittest.TestCase):
    """``ModRadConData.zeros``/``.ones`` derive ``emis2`` from a supplied
    ``tau2`` rather than an unconditional class default (Codex review, PR
    #1124 thread on ``physics_data.py``): a caller passing ``tau2=`` without
    ``emis2=`` must get its actual complement, not whichever constant that
    factory uses for its own reference ``tau2``.
    """

    def test_zeros_derives_emis2_from_an_explicit_tau2(self):
        from jcm.physics.speedy.physics_data import ModRadConData

        tau2 = jnp.full((3, 2, 2, 4), 0.75)
        data = ModRadConData.zeros((2, 2), 3, tau2=tau2)
        np.testing.assert_allclose(np.asarray(data.emis2), 1.0 - np.asarray(tau2))

    def test_ones_derives_emis2_from_an_explicit_tau2(self):
        from jcm.physics.speedy.physics_data import ModRadConData

        tau2 = jnp.full((3, 2, 2, 4), 0.75)
        data = ModRadConData.ones((2, 2), 3, tau2=tau2)
        np.testing.assert_allclose(np.asarray(data.emis2), 1.0 - np.asarray(tau2))

    def test_an_explicit_emis2_is_not_overridden(self):
        from jcm.physics.speedy.physics_data import ModRadConData

        tau2 = jnp.full((3, 2, 2, 4), 0.75)
        emis2 = jnp.full((3, 2, 2, 4), 0.9)
        zeros_data = ModRadConData.zeros((2, 2), 3, tau2=tau2, emis2=emis2)
        ones_data = ModRadConData.ones((2, 2), 3, tau2=tau2, emis2=emis2)
        np.testing.assert_array_equal(np.asarray(zeros_data.emis2), np.asarray(emis2))
        np.testing.assert_array_equal(np.asarray(ones_data.emis2), np.asarray(emis2))

    def test_the_reference_defaults_are_unchanged(self):
        """No ``tau2``/``emis2`` given: each factory's own reference pair."""
        from jcm.physics.speedy.physics_data import ModRadConData

        zeros_data = ModRadConData.zeros((2, 2), 3)
        self.assertEqual(float(zeros_data.tau2[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(zeros_data.emis2[0, 0, 0, 0]), 1.0)

        ones_data = ModRadConData.ones((2, 2), 3)
        self.assertEqual(float(ones_data.tau2[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(ones_data.emis2[0, 0, 0, 0]), 0.0)


def _clear_sky_columns(ncol=6, kx=8):
    """Build a few cloud-free columns, from a moist tropical to a dry polar profile.

    Returns ``(state, physics_data)`` on a ``(kx, ncol, 1)`` grid in the
    units the translated SPEEDY routines take (humidity in g/kg).
    """
    from jcm.physics.speedy.physics_data import (
        ModRadConData, PhysicsData, SurfaceFluxData, SWRadiationData)
    from jcm.physics.speedy.speedy_coords import SpeedyCoords
    from jcm.physics_interface import PhysicsState
    import jcm.constants as c

    coords = SpeedyCoords.single_column_coords(num_levels=kx)
    shape = (ncol, 1)
    warmth = jnp.linspace(1.0, 0.0, ncol)[None, :, None]       # 1 tropical, 0 polar
    sigma = coords.fsg[:, None, None]
    t_surface = 250.0 + 50.0 * warmth
    temperature = jnp.maximum(t_surface * sigma ** 0.19, 200.0 + 15.0 * warmth)
    # g/kg: 18 at the tropical surface, falling with sigma**3, none in the stratosphere
    humidity = jnp.where(sigma > 0.15, (0.5 + 17.5 * warmth) * sigma ** 3, 0.0)
    zeros = jnp.zeros_like(temperature)
    state = PhysicsState(zeros, zeros, temperature, humidity, zeros,
                         jnp.ones(shape))
    tsfc = t_surface[0] + 1.0
    data = PhysicsData.zeros(
        shape, kx, speedy_coords=coords,
        shortwave_rad=SWRadiationData.zeros(
            shape, kx, zenit=jnp.ones(shape),
            icltop=jnp.full(shape, kx + 1, dtype=int)),
        mod_radcon=ModRadConData.zeros(shape, kx),
        surface_flux=SurfaceFluxData.zeros(shape).copy(
            tsfc=tsfc, rlus=0.98 * c.sbc * tsfc ** 4))
    return state, data


def _clear_sky_fluxes(state, data, shortwave_radiation, mod_radcon):
    """Return clear-sky (OLR, surface downward longwave, tau2) from the model's own passes."""
    from jcm.physics.speedy.params import Parameters
    from jcm.physics_interface import PhysicsTendency
    from jcm.physics.forcing.speedy_forcing import ablco2_from_co2_vmr
    from jcm.physics.radiation.speedy_longwave import (
        get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes)
    from jcm.physics.radiation.speedy_shortwave import shortwave_rad_fluxes
    import dataclasses

    parameters = dataclasses.replace(
        Parameters.default(), shortwave_radiation=shortwave_radiation,
        mod_radcon=mod_radcon)
    data = data.copy(mod_radcon=data.mod_radcon.copy(
        ablco2=ablco2_from_co2_vmr(jnp.asarray(360.0), mod_radcon.ablco2_ref)))
    tendency = PhysicsTendency.zeros(shape=state.temperature.shape)
    _, data, *_ = shortwave_rad_fluxes(
        (state, data, parameters, None, None, tendency))
    _, data = get_downward_longwave_rad_fluxes(state, data, parameters, None, None)
    rlds = data.surface_flux.rlds
    _, data = get_upward_longwave_rad_fluxes(state, data, parameters, None, None)
    return data.longwave_rad.ftop, rlds, data.mod_radcon.tau2


def c_sbc():
    """Return the Stefan-Boltzmann constant the scheme reads."""
    import jcm.constants as c
    return c.sbc


class TestClearSkyLongwaveParameters(unittest.TestCase):
    """The tunables of the clear-sky longwave: band absorptivities, the
    window's water-vapour continuum, the band fractions and ``epslw``.
    """

    def setUp(self):
        from jcm.physics.speedy.params import (
            ModRadConParameters, ShortwaveRadiationParameters)
        self.sw = ShortwaveRadiationParameters.default()
        self.mrc = ModRadConParameters.default()
        self.state, self.data = _clear_sky_columns()

    def test_the_continuum_is_off_at_the_reference(self):
        """The continuum defaults to off (ablwvc = 0): the window absorbs as SPEEDY's does."""
        self.assertEqual(float(self.sw.ablwvc), 0.0)
        _, _, tau2 = _clear_sky_fluxes(self.state, self.data, self.sw, self.mrc)
        coords = self.data.speedy_coords
        psa = self.state.normalized_surface_pressure
        want = jnp.exp(-psa[None] * coords.dhs[:, None, None] * self.sw.ablwin)
        np.testing.assert_allclose(np.asarray(tau2[..., 0]), np.asarray(want), rtol=1e-6)

    def test_the_continuum_thickens_the_window_as_q_squared_p(self):
        """Window optical depth gains ablwvc * q**2 * p/p0 per unit layer mass."""
        import dataclasses
        ablwvc = 0.004
        with_continuum = dataclasses.replace(self.sw, ablwvc=jnp.array(ablwvc))
        olr0, rlds0, tau0 = _clear_sky_fluxes(self.state, self.data, self.sw, self.mrc)
        olr1, rlds1, tau1 = _clear_sky_fluxes(self.state, self.data, with_continuum, self.mrc)
        coords = self.data.speedy_coords
        psa = self.state.normalized_surface_pressure[None]
        q = self.state.specific_humidity
        extra = (psa * coords.dhs[:, None, None]
                 * ablwvc * q * q * coords.fsg[:, None, None] * psa)
        np.testing.assert_allclose(
            np.asarray(tau1[..., 0]), np.asarray(tau0[..., 0] * jnp.exp(-extra)), rtol=1e-5)
        # the other bands do not see it
        np.testing.assert_array_equal(np.asarray(tau1[..., 1:]), np.asarray(tau0[..., 1:]))
        # a closing window traps surface emission and returns it downward,
        # most where the column is moist (column 0) and not at all where the
        # stratosphere-like dry column has no vapour to speak of
        d_olr = np.asarray(olr1 - olr0)[:, 0]
        d_rlds = np.asarray(rlds1 - rlds0)[:, 0]
        self.assertTrue(np.all(d_olr <= 1e-4) and np.all(d_rlds >= -1e-4))
        self.assertLess(d_olr[0], -1.0)
        self.assertGreater(d_rlds[0], 5.0)
        self.assertLess(abs(d_rlds[-1]), 0.1 * d_rlds[0])

    def test_the_column_budget_closes(self):
        """What the atmosphere absorbs is what enters at the surface less the OLR.

        The surface absorbs ``rlds`` and the upward pass returns the unabsorbed
        ``1 - emisfc`` of the downward band fluxes, which the downward pass
        hands it in ``mod_radcon.flux`` (as SPEEDY's module array does).
        """
        import dataclasses

        from jcm.physics.speedy.params import Parameters
        from jcm.physics_interface import PhysicsTendency
        from jcm.physics.radiation.speedy_longwave import (
            get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes)
        from jcm.physics.radiation.speedy_shortwave import shortwave_rad_fluxes

        parameters = dataclasses.replace(
            Parameters.default(), shortwave_radiation=self.sw, mod_radcon=self.mrc)
        state = self.state
        # a sunlit shortwave pass, which leaves its own band fluxes in the
        # shared array, as on a radiation step of the model
        data = self.data.copy(shortwave_rad=self.data.shortwave_rad.copy(
            fsol=jnp.full(state.normalized_surface_pressure.shape, 400.0)))
        tendency = PhysicsTendency.zeros(shape=state.temperature.shape)
        _, data, *_ = shortwave_rad_fluxes(
            (state, data, parameters, None, None, tendency))
        _, data = get_downward_longwave_rad_fluxes(state, data, parameters, None, None)
        down = data.mod_radcon.flux
        rlds = data.surface_flux.rlds
        black = self.mrc.epslw * self.mrc.emisfc * c_sbc() * state.temperature[-1] ** 4
        np.testing.assert_allclose(
            np.asarray(self.mrc.emisfc * down.sum(axis=-1) + black), np.asarray(rlds), rtol=1e-5)
        self.assertGreater(float(down.sum(axis=-1).min()), 50.0)
        _, data = get_upward_longwave_rad_fluxes(state, data, parameters, None, None)
        absorbed = data.longwave_rad.dfabs.sum(axis=0)
        net_surface = data.surface_flux.rlus - rlds
        np.testing.assert_allclose(
            np.asarray(absorbed), np.asarray(net_surface - data.longwave_rad.ftop), atol=2e-3)

    def test_clear_sky_fluxes_gradient_check(self):
        """OLR and surface downward flux are differentiable in every tunable."""
        def f(sw, mrc):
            olr, rlds, _ = _clear_sky_fluxes(self.state, self.data, sw, mrc)
            return olr, rlds

        import dataclasses
        # A point with the continuum switched on, so its derivative is not
        # taken at the edge of its range.
        sw = dataclasses.replace(self.sw, ablwvc=jnp.array(0.004))
        check_gradients(
            f, (sw, self.mrc), rtol=1e-2,
            live_inputs=["ablwin", "ablwvc", "ablwv1", "ablwv2", "epslw",
                         "ablco2_ref", "fbco2_a", "fbco2_b", "fbco2_t0",
                         "fbwv1_a", "fbwv1_b", "fbwv1_t0",
                         "fbwv2_a", "fbwv2_b", "fbwv2_t0"])
