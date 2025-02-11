from jax import jit
import jax.numpy as jnp
from jcm.params import Parameters

n_temperatures = 301
n_bands = 4

# Initialize empty fband array
fband_initial = jnp.zeros((n_temperatures, n_bands))

@jit
def radset(initial_array, parameters):
    """
    Set the energy fraction emitted in each LW band = f(T)
    """
    eps1 = 1.0 - parameters.mod_radcon.epslw
    t_min, t_max = 200, 320
    jtemp = jnp.arange(t_min, t_max + 1)
    
    # Calculate band components
    fband_2 = (0.148 - 3.0e-6 * (jtemp - 247) ** 2) * eps1
    fband_3 = (0.356 - 5.2e-6 * (jtemp - 282) ** 2) * eps1
    fband_4 = (0.314 + 1.0e-5 * (jtemp - 315) ** 2) * eps1
    fband_1 = eps1 - (fband_2 + fband_3 + fband_4)
    
    # Create working copy of the array
    result = initial_array.at[jtemp - 100, :4].set(
        jnp.stack((fband_1, fband_2, fband_3, fband_4), axis=-1)
    )
    
    # Handle boundary conditions
    jb = jnp.arange(4)
    result = result.at[:(t_min - 100), jb].set(result[t_min - 100, jb])
    result = result.at[(t_max + 1 - 100):, jb].set(result[t_max - 100, jb])
    
    return result

parameters = Parameters.default()
# Calculate final fband values
fband = radset(fband_initial, parameters)



