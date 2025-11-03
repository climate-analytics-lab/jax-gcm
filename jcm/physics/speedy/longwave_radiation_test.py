import unittest
import jax.numpy as jnp
import numpy as np
import jax
import functools
from jax.test_util import check_vjp, check_jvp

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

        global ModRadConData, LWRadiationData, SurfaceFluxData, PhysicsData, PhysicsState, PhysicsTendency, ForcingData, get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes, radset, parameters, geometry
        from jcm.physics.speedy.physics_data import ModRadConData, LWRadiationData, SurfaceFluxData, PhysicsData
        from jcm.physics.speedy.params import Parameters
        from jcm.physics_interface import PhysicsState, PhysicsTendency
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes, radset
        from jcm.geometry import Geometry
        from jcm.physics.speedy.test_utils import convert_to_speedy_latitudes
        parameters = Parameters.default()
        geometry = convert_to_speedy_latitudes(Geometry.from_grid_shape(nodal_shape=(ix, il), num_levels=kx))

    def test_downward_longwave_rad_fluxes(self):

        # FIXME: This array doesn't need to be this big once we fix the interfaces
        # -> We only test the first 5x5 elements
        zxy = (kx, ix, il)
        xy = (ix, il)
        ta, rlds, st4a, flux = initialize_arrays(ix, il, kx)
        mod_radcon = ModRadConData.zeros((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData.zeros((ix, il), kx, mod_radcon=mod_radcon)
        forcing = ForcingData.ones(xy)
        
        state = PhysicsState.zeros(zxy,temperature=ta)
        
        _, physics_data = get_downward_longwave_rad_fluxes(state, physics_data, parameters, forcing, geometry)

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
        input_physics_data = PhysicsData.zeros((ix, il), kx).copy(
            longwave_rad=LWRadiationData.zeros((ix, il), kx).copy(dfabs=dfabs),
            mod_radcon=ModRadConData.zeros((ix, il), kx).copy(st4a=st4a, flux=flux, tau2=tau2, stratc=stratc),
            surface_flux=SurfaceFluxData.zeros((ix, il), kx).copy(rlus=jnp.zeros((ix,il,3)).at[:,:,2].set(rlus), rlds=rlds, tsfc=ts),
        )

        # skip testing ttend since we have access to dfabs
        _, output_physics_data = get_upward_longwave_rad_fluxes(state=state, physics_data=input_physics_data, parameters=parameters, forcing=ForcingData.zeros((ix, il)), geometry=geometry)

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
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state = PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)
        # Calculate gradient
        _, f_vjp = jax.vjp(get_downward_longwave_rad_fluxes, state, physics_data, parameters, forcing, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())
       

    def test_get_upward_longwave_rad_fluxes_gradients_isnan_ones(self):
        """Test that we can calculate gradients of longwave radiation without getting NaN values"""
        xy = (ix, il)
        zxy = (kx, ix, il)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state = PhysicsState.ones(zxy)
        forcing = ForcingData.ones(xy)

        # Calculate gradient
        _, f_vjp = jax.vjp(get_upward_longwave_rad_fluxes, state, physics_data, parameters, forcing, geometry)
        tends = PhysicsTendency.ones(zxy)
        datas = PhysicsData.ones(xy,kx)
        input = (tends, datas)
        df_dstates, df_ddatas, df_dparams, df_dforcing, df_dgeometry = f_vjp(input)

        self.assertFalse(df_ddatas.isnan().any_true())
        self.assertFalse(df_dstates.isnan().any_true())
        self.assertFalse(df_dparams.isnan().any_true())
        self.assertFalse(df_dforcing.isnan().any_true())

    def test_radset_gradient_check(self):
        zxy = (kx, ix, il)
        state = PhysicsState.ones(zxy)
        temp = state.temperature
        epslw = parameters.mod_radcon.epslw

        def f(temp, epslw):
            return radset(temp, epslw)

        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (temp, epslw), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (temp, epslw), 
                                atol=None, rtol=1, eps=0.000001)
        
    def test_downward_longwave_rad_fluxes_gradient_check(self):
        from jcm.utils import convert_back, convert_to_float
        # FIXME: This array doesn't need to be this big once we fix the interfaces
        # -> We only test the first 5x5 elements
        zxy = (kx, ix, il)
        xy = (ix, il)
        ta, rlds, st4a, flux = initialize_arrays(ix, il, kx)
        mod_radcon = ModRadConData.zeros((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData.zeros((ix, il), kx, mod_radcon=mod_radcon)
        forcing = ForcingData.ones(xy)
        state = PhysicsState.zeros(zxy,temperature=ta)

        # Set float inputs
        physics_data_floats = convert_to_float(physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, forcing_f,geometry_f):
            tend_out, data_out = get_downward_longwave_rad_fluxes(physics_data=convert_back(physics_data_f, physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.0001)


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
        input_physics_data = PhysicsData.zeros((ix, il), kx).copy(
            longwave_rad=LWRadiationData.zeros((ix, il), kx).copy(dfabs=dfabs),
            mod_radcon=ModRadConData.zeros((ix, il), kx).copy(st4a=st4a, flux=flux, tau2=tau2, stratc=stratc),
            surface_flux=SurfaceFluxData.zeros((ix, il), kx).copy(rlus=jnp.zeros((ix,il,3)).at[:,:,2].set(rlus), rlds=rlds, tsfc=ts),
        )
        forcing = ForcingData.zeros((ix, il))

        # Set float inputs
        physics_data_floats = convert_to_float(input_physics_data)
        state_floats = convert_to_float(state)
        parameters_floats = convert_to_float(parameters)
        forcing_floats = convert_to_float(forcing)
        geometry_floats = convert_to_float(geometry)

        def f(physics_data_f, state_f, parameters_f, forcing_f,geometry_f):
            tend_out, data_out = get_upward_longwave_rad_fluxes(physics_data=convert_back(physics_data_f, input_physics_data), 
                                       state=convert_back(state_f, state), 
                                       parameters=convert_back(parameters_f, parameters), 
                                       forcing=convert_back(forcing_f, forcing), 
                                       geometry=convert_back(geometry_f, geometry)
                                       )
            return convert_to_float(data_out)
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)  

        check_vjp(f, f_vjp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (physics_data_floats, state_floats, parameters_floats, forcing_floats, geometry_floats), 
                                atol=None, rtol=1, eps=0.0001)



