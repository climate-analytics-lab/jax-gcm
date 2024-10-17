from numpy.testing import assert_array_almost_equal
import unittest
import jax.numpy as jnp
from jcm.params import ix, il, kx
from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes, radset
from jcm.physics_data import PhysicsData, ModRadConData
from jcm.physics import PhysicsState

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

    def test_downward_longwave_rad_fluxes(self):
        import numpy as np

        #FIXME: This array doens't need to be this big once we fix the interfaces
        # -> We only test teh first 5x5 elements
        ta, fsfcd, st4a, flux = initialize_arrays(ix, il, kx)
        mod_radcon = ModRadConData((ix, il), kx, flux=flux, st4a=st4a)
        physics_data = PhysicsData((ix, il), kx, mod_radcon=mod_radcon)

        physics_data = radset(physics_data)
        state = PhysicsState(u_wind=jnp.zeros_like(ta),
                             v_wind=jnp.zeros_like(ta),
                             temperature=ta,
                             specific_humidity=jnp.zeros_like(ta),
                             geopotential=jnp.zeros_like(ta),
                             surface_pressure=jnp.zeros((ix, il)))
        
        _, physics_data = get_downward_longwave_rad_fluxes(state, physics_data)

        # fortran values
        f90_rlds = [[186.6984  , 187.670515, 188.646319, 189.625957, 190.609469],
                    [186.708473, 187.680627, 188.656572, 189.636231, 190.6197  ],
                    [186.718628, 187.69074 , 188.666658, 189.646441, 190.630014],
                    [186.728719, 187.700953, 188.676876, 189.656632, 190.640263],
                    [186.738793, 187.711066, 188.687129, 189.666908, 190.650495]]

        f90_dfabs = [ -3.799531,
                     -20.11071 ,
                     -17.83563 ,
                     -17.667264,
                     -22.200773,
                     -27.997842,
                     -33.615657,
                     -47.10823 ]
        
        self.assertTrue(np.allclose(physics_data.longwave_rad.rlds[:5, :5], np.asarray(f90_rlds), atol=1e-4))
        self.assertTrue(np.allclose(physics_data.longwave_rad.dfabs[0, 0, :], f90_dfabs, atol=1e-4))

    def test_upward_longwave_rad_fluxes(self):
        # TODO: Implement this test

        pass


