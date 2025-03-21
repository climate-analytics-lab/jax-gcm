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

    return 0.5*jnp.linalg.norm(R_inv_sqrt*(y - forward_model(theta, *args)))**2

def create_model(parameters, time_step = 30, save_interval = 1.0, total_time = 5.0, layers = 8): 
    '''
    Returns speedy model with given parameters and specifications

    Args: 
        parameters: parameters object type (jcm.paramters)
        time_step: Model time step in minutes
        save_interval: Save interval in days
        total_time: Total integration time in days
        layers: Number of vertical layers
    '''
    model = SpeedyModel(time_step = time_step, 
                        save_interval = save_interval, 
                        total_time = total_time, 
                        layers = layers,
                        parameters = parameters, 
                        post_process = True)
    return model

def forward_model_wrapper(theta, theta_keys, state = None, parameters = None, args = ()):
    '''
    Returns forward model run collapsed into a single vector
    '''
    if parameters is None:
        parameters = Parameters.default()

    ii = 0
    for attr, params in theta_keys.items():
        for param in params:
            setattr(getattr(parameters, attr), param, theta[ii])
            ii += 1
        
    model = create_model(parameters, *args) 
    if state is None: 
        state = model.get_initial_state()
    final_state, predictions = model.unroll(state)
    return predictions['dynamics'].temperature_variation.flatten()  # fix shape of this