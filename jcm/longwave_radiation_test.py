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

    def test_downard_longwave_rad_fluxes(self):
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
        # print(fsfcd[:5, :5])
        f90_rlds =[ [ 210.58037967609744, 210.59147844598601, 210.60257765343979, 210.61367729847015, 210.62477738108839],
                [ 211.64785852088454, 211.65899934932736, 211.67014061643681, 211.68128232222412, 211.69242446670108], 
                [ 212.71938001172822, 212.73056300453399, 212.74174643710919, 212.75293030946537, 212.76411462161374],
                [ 213.79495431257942, 213.80617957568973, 213.81740527967378, 213.82863142454275, 213.83985801030869],
                [ 214.87459160010974, 214.88585923959852, 214.89712732106619, 214.90839584452522, 214.91966480998622]]

        # print(dfabs[0, 0, :])   
        f90_dfabs = [  -3.5801730611349774, -17.861774929469838, -20.478947887745250, -17.260889773424999, -22.102412041367657, -27.772991217865744, -34.338080378956391, -71.309733057733894 ]

        # print(np.mean(mod_radcon.st4a[:5,:5,:,:], axis=2))
        f90_st4a = [[[76.11271,13.08427],
                     [76.58848,13.14281],
                     [77.06654,13.20152],
                     [77.54689,13.26041],
                     [78.02956,13.31949]],
 
                     [[76.11766,13.08487],
                     [76.59345,13.14342],
                     [77.07153,13.20214],
                     [77.55191,13.26103],
                     [78.0346 ,13.3201 ]],
 
                     [[76.1226 ,13.08549],
                     [76.59841,13.14403],
                     [77.07652,13.20274],
                     [77.55693,13.26164],
                     [78.03964,13.32072]],
 
                     [[76.12754,13.0861 ],
                     [76.60338,13.14464],
                     [77.08152,13.20336],
                     [77.56194,13.26225],
                     [78.04468,13.32133]],
 
                     [[76.13249,13.0867 ],
                     [76.60836,13.14525],
                     [77.08651,13.20398],
                     [77.56696,13.26287],
                     [78.04972,13.32195]]]

        # Note the transpose to match the fortran array order
        assert_array_almost_equal(physics_data.longwave_rad.rlds[:5, :5], np.asarray(f90_rlds).T, decimal=4)
        assert_array_almost_equal(physics_data.longwave_rad.dfabs[0, 0, :], f90_dfabs, decimal=4)
        assert_array_almost_equal(jnp.mean(physics_data.mod_radcon.st4a[:5,:5,:,:], axis=2), np.asarray(f90_st4a), decimal=4)


    def test_upward_longwave_rad_fluxes(self):
        # TODO: Implement this test

        pass


