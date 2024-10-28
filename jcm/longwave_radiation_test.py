import unittest
import jax.numpy as jnp
import numpy as np
from jcm.model import initialize_modules
from jcm.physics_data import PhysicsData, ModRadConData
from jcm.physics import PhysicsState        
from jcm.longwave_radiation import get_downward_longwave_rad_fluxes

ix, il, kx = 96, 48, 8

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
        initialize_modules(kx=kx, il=il)

    def test_downward_longwave_rad_fluxes(self):

        #FIXME: This array doens't need to be this big once we fix the interfaces
        # -> We only test teh first 5x5 elements
        ta, fsfcd, st4a, flux = initialize_arrays(ix, il, kx)
        mod_radcon = ModRadConData((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData((ix, il), kx, mod_radcon=mod_radcon)
        
        state = PhysicsState(u_wind=jnp.zeros_like(ta),
                             v_wind=jnp.zeros_like(ta),
                             temperature=ta,
                             specific_humidity=jnp.zeros_like(ta),
                             geopotential=jnp.zeros_like(ta),
                             surface_pressure=jnp.zeros((ix, il)))
        
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
        
        self.assertTrue(np.allclose(physics_data.longwave_rad.rlds[:5, :5], jnp.asarray(f90_rlds), atol=1e-4))
        self.assertTrue(np.allclose(physics_data.longwave_rad.dfabs[0, 0, :], f90_dfabs, atol=1e-4))
        self.assertTrue(np.allclose(jnp.mean(physics_data.mod_radcon.st4a[:5, :5, :, :], axis=2), jnp.asarray(f90_st4a), atol=1e-4))

    def test_upward_longwave_rad_fluxes(self):
        # TODO: Implement this test

        pass


