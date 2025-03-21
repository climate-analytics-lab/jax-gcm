from jcm.model import SpeedyModel
import jax
import jax.numpy as jnp
from jcm.params import Parameters

def loss_function(theta, forward_model, y, R_inv_sqrt, args = ()): 
    '''
    Returns data-model misfit (i.e. loss function to optimize over)

    Args: 
        theta: parameters of interest (find the parameters that minimize the loss function)
        forward_model: forward run through model with output the same shape as y
        y: data to compare model to (must be 1D vector)
        R_inv_sqrt: inverse square root of R (the assumed data errors associated to data)
        args: additional forward model function inputs
    '''

    return 0.5*jnp.linalg.norm(R_inv_sqrt@(y - forward_model(theta, *args)))