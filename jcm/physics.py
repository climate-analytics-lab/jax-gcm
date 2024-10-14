'''
Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used. 
'''

from collections import abc
import jax.numpy as jnp
import tree_math
from typing import Callable
from jcm.physics_data import PhysicsData

from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, State, PrimitiveEquations

@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    surface_pressure: jnp.ndarray
    # relative_humidity: jnp.ndarray

@tree_math.struct
class PhysicsTendency:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray


def dynamics_state_to_physics_state(state: State, dynamics: PrimitiveEquations) -> PhysicsState:
    """
    Convert the state variables from the dynamics to the physics state variables.

    Args:
        state: Dynamic (dinosaur) State variables
        dynamics: PrimitiveEquations object containing the reference temperature and orography

    Returns:
        Physics state variables
    """
    # Calculate u and v from vorticity and divergence
    u, v = vor_div_to_uv_nodal(dynamics.coords.horizontal, state.vorticity, state.divergence)

    # Calculate geopotential
    phi_spectral = get_geopotential(
        state.temperature_variation,
        dynamics.reference_temperature,
        dynamics.orography,
        dynamics.coords.vertical,
    )
    # Z, X, Y
    t_spectral = state.temperature_variation + dynamics.coords.horizontal.to_modal(dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis])
    q_spectral = state.tracers['specific_humidity']

    t, q, phi, log_sp = dynamics.coords.horizontal.to_nodal(
        (t_spectral, q_spectral, phi_spectral, state.log_surface_pressure)
    )
    
    log_sp = 11.5 + 0*log_sp
    # TODO: figure out why log_sp values are ~45 (isn't this extremely large?) as well as why setting p1 to 0 prevents this from causing infs, but only with jit enabled...?
    sp = jnp.exp(log_sp)

    physics_state = PhysicsState(
        u.transpose(1, 2, 0),
        v.transpose(1, 2, 0),
        t.transpose(1, 2, 0),
        q.transpose(1, 2, 0),
        phi.transpose(1, 2, 0),
        sp.transpose(1, 2, 0),
    )
    return physics_state


def physics_tendency_to_dynamics_tendency(physics_tendency: PhysicsTendency, dynamics: PrimitiveEquations) -> State:
    """
    Convert the physics tendencies to the dynamics tendencies.

    Args:
        physics_tendency: Physics tendencies
        dynamics: PrimitiveEquations object containing the reference temperature and orography

    Returns:
        Dynamics tendencies
    """
    vor_tendency, div_tendency = uv_nodal_to_vor_div_modal(  # double check the math
        dynamics.coords.horizontal,
        physics_tendency.u_wind.transpose(2, 0, 1),
        physics_tendency.v_wind.transpose(2, 0, 1)
    )

    t_tendency = dynamics.coords.horizontal.to_modal(physics_tendency.temperature.transpose(2, 0, 1))
    q_tendency = dynamics.coords.horizontal.to_modal(physics_tendency.specific_humidity.transpose(2, 0, 1))
    
    log_sp_tendency = jnp.zeros_like(t_tendency[0, ...]) # This assumes the physics tendency is zero for log_surface_pressure

    # Create a new state object with the updated tendencies (which will be added to the current state)
    dynamics_tendency = State(vor_tendency, div_tendency, t_tendency, log_sp_tendency, {'specific_humidity': q_tendency})
    return dynamics_tendency


def get_physical_tendencies(
    state: State,
    dynamics: PrimitiveEquations,
    physics_terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]],
):
    """
    Computes the physical tendencies given the current state and a list of physics functions.

    Args:
        state: Dynamic (dinosaur) State variables
        dynamics: 
        physics_terms: List of physics functions that take a PhysicsState and return a PhysicsTendency

    Returns:
        Physical tendencies
    """
    physics_state = dynamics_state_to_physics_state(state, dynamics)

    # the 'physics_terms' return an instance of tendencies and data, data gets overwritten at each step 
    # and implicitly passed to the next physics_term. tendencies are summed 
    physics_tendency = PhysicsTendency(
        jnp.zeros_like(physics_state.u_wind),
        jnp.zeros_like(physics_state.u_wind),
        jnp.zeros_like(physics_state.u_wind),
        jnp.zeros_like(physics_state.u_wind))
    
    data = PhysicsData(physics_state.temperature.shape[0:2],physics_state.temperature.shape[2])
    # optionally initialize the physics data here if it needs to be 

    # TODO: revisit this and/or squeeze the physics_state pressure
    data.convection.psa = jnp.squeeze(physics_state.surface_pressure)
    
    for term in physics_terms:
        tend, data = term(data, physics_state)
        physics_tendency += tend

    #physics_tendency = sum(term(physics_state) for term in physics_terms)

    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics)
    return dynamics_tendency