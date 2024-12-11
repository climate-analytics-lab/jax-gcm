import unittest
import jax.numpy as jnp
import numpy as np
import jax

def initialize_arrays(ix, il, kx):
    # Initialize arrays
    ta = jnp.zeros((ix, il, kx))
    fsfcd = jnp.zeros((ix, il))
    st4a = jnp.zeros((ix, il, kx, 2))     # Blackbody emission from full and half atmospheric levels
    flux = jnp.zeros((ix, il, 4))         # Radiative flux in different spectral bands

    # Set the min and max values
    min_val = 130.0
    max_val = 250.0
    
    # Calculate step size
    total_elements = ix * il * kx
    step_size = (max_val - min_val) / (total_elements - 1)

    ta = min_val + step_size*jnp.arange(total_elements).reshape((kx, il, ix)).transpose((2, 1, 0))
    
    return ta, fsfcd, st4a, flux

class TestDownwardLongwave(unittest.TestCase):
    
    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)

        global ModRadConData, PhysicsData, PhysicsState, PhysicsTendency, get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
        from jcm.physics_data import ModRadConData, PhysicsData
        from jcm.physics import PhysicsState, PhysicsTendency
        from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes

    def test_downward_longwave_rad_fluxes(self):        

        #FIXME: This array doens't need to be this big once we fix the interfaces
        # -> We only test teh first 5x5 elements
        xyz = (ix, il, kx)
        ta, fsfcd, st4a, flux = initialize_arrays(ix, il, kx)
        mod_radcon = ModRadConData.zeros((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData.zeros((ix, il), kx, mod_radcon=mod_radcon)

        state = PhysicsState.zeros(xyz,temperature=ta)
        
        _, physics_data = get_downward_longwave_rad_fluxes(state, physics_data)

        # fortran values
        # print(fsfcd[:5, :5])
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
        
        self.assertTrue(np.allclose(physics_data.longwave_rad.rlds[:5, :5], np.asarray(f90_rlds), atol=1e-4))
        self.assertTrue(np.allclose(physics_data.longwave_rad.dfabs[0, 0, :], f90_dfabs, atol=1e-4))
        self.assertTrue(np.allclose(np.mean(physics_data.mod_radcon.st4a[:5, :5, :, :], axis=2), np.asarray(f90_st4a), atol=1e-4))

    def test_upward_longwave_rad_fluxes(self):
        # TODO: Implement this test

        pass

    def test_get_downward_longwave_rad_fluxes_gradients(self):    
        """Test that we can calculate gradients of longwave radiation without getting NaN values"""
        xy = (ix, il)
        xyz = (ix, il, kx)
        physics_data = PhysicsData.zeros(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.zeros(xyz)

        # Calculate gradient
        primals, f_vjp = jax.vjp(get_downward_longwave_rad_fluxes, state, physics_data) 
        tends = PhysicsTendency.ones(xyz)
        datas = PhysicsData.ones(xy,kx) 
        input = (tends, datas)
        df_dstates, df_ddatas = f_vjp(input)

        # Checking if the function with respect to the input states is nan
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.surface_pressure)))

        # Checking if the function with respect to the input physics data is nan
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.rlds)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.dfabs)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.ftop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.slr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.qcloud)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.fsol)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.rsds)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ssr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ozone)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ozupp)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.zenit)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.stratz)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.gse)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.icltop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.cloudc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.cloudstr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ftop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.dfabs)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.psa)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.se)))
        # self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.iptop))) doesn't work bc current type is int
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.cbmf)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.precnv)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.alb_l)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.alb_s)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.albsfc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.snowc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.tau2)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.st4a)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.stratc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.flux)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.humidity.rh)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.humidity.qsat)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.precls)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.dtlsc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.dqlsc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.stl_am)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.soilw_am)))
        # Not testing df_ddatas.surface_flux.lfluxland because it is a bool type
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.ustr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.vstr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.shf)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.evap)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.slru)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.hfluxn)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.tsfc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.tskin)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.u0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.v0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.t0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.fmask)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.phi0)))
        # No testing df_ddatas.date
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.sea_model.tsea)))

    def test_get_upward_longwave_rad_fluxes_gradients(self):    
        """Test that we can calculate gradients of longwave radiation without getting NaN values"""
        xy = (ix, il)
        xyz = (ix, il, kx)
        physics_data = PhysicsData.ones(xy,kx)  # Create PhysicsData object (parameter)
        state =PhysicsState.ones(xyz)

        # Calculate gradient
        primals, f_vjp = jax.vjp(get_upward_longwave_rad_fluxes, state, physics_data) 
        tends = PhysicsTendency.ones(xyz)
        datas = PhysicsData.ones(xy,kx) 
        input = (tends, datas)
        df_dstates, df_ddatas = f_vjp(input)

        # Checking if the function with respect to the input states is nan
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstates.surface_pressure)))

        # Checking if the function with respect to the input physics data is nan
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.rlds)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.dfabs)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.ftop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.longwave_rad.slr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.qcloud)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.fsol)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.rsds)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ssr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ozone)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ozupp)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.zenit)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.stratz)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.gse)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.icltop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.cloudc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.cloudstr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.ftop)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.shortwave_rad.dfabs)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.psa)))  # Currently not working
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.se)))
        # self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.iptop))) doesn't work bc current type is int
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.cbmf)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.convection.precnv)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.alb_l)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.alb_s)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.albsfc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.snowc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.tau2)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.st4a)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.stratc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.mod_radcon.flux)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.humidity.rh)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.humidity.qsat)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.precls)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.dtlsc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.condensation.dqlsc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.stl_am)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.soilw_am)))
        # Not testing df_ddatas.surface_flux.lfluxland because it is a bool type
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.ustr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.vstr)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.shf)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.evap)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.slru)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.hfluxn)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.tsfc)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.tskin)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.u0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.v0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.t0)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.fmask)))
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.surface_flux.phi0)))
        # No testing df_ddatas.date
        self.assertFalse(jnp.any(jnp.isnan(df_ddatas.sea_model.tsea)))



