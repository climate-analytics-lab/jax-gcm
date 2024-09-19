'''
Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used. 
'''

from collections import abc
import numpy as np
import jax.numpy as jnp
import tree_math
from typing import Callable

from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.sigma_coordinates import SigmaCoordinates
from dinosaur.scales import units

from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal, Grid
from dinosaur.primitive_equations import get_geopotential, State, PrimitiveEquations, PrimitiveEquationsSpecs
from dinosaur import primitive_equations_states

@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray
    fsol: jnp.ndarray
    ozone: jnp.ndarray
    ozupp: jnp.ndarray
    zenit: jnp.ndarray
    stratz: jnp.ndarray

@tree_math.struct
class ModRadConData:
    # Radiative properties of the surface (updated in fordate)
    # Albedo and snow cover arrays
    alb_l: jnp.ndarray  # Daily-mean albedo over land (bare-land + snow)
    alb_s: jnp.ndarray  # Daily-mean albedo over sea (open sea + sea ice)
    albsfc: jnp.ndarray # Combined surface albedo (land + sea)
    snowc: jnp.ndarray  # Effective snow cover (fraction)

    # Transmissivity and blackbody radiation (updated in radsw/radlw)
    tau2: jnp.ndarray   # Transmissivity of atmospheric layers
    st4a: jnp.ndarray   # Blackbody emission from full and half atmospheric levels
    stratc: jnp.ndarray # Stratospheric correction term
    flux: jnp.ndarray   # Radiative flux in different spectral bands

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
    t_spectral = state.temperature_variation + dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    q_spectral = state.tracers['specific_humidity']

    t, q, phi, log_sp = dynamics.coords.horizontal.to_nodal(
        (t_spectral, q_spectral, phi_spectral, state.log_surface_pressure)
    )
    
    
    sp = jnp.exp(log_sp)
    physics_state = PhysicsState(u, v, t, q, phi, sp)
    return physics_state


def physics_tendency_to_dynamics_tendency(physics_tendency: PhysicsTendency, dynamics: PrimitiveEquations) -> State:

    vor_tendency, div_tendency = uv_nodal_to_vor_div_modal(  # double check the math
        dynamics.coords.horizontal, physics_tendency.u_wind, physics_tendency.v_wind
    )
    """
    Convert the physics tendencies to the dynamics tendencies.

    Args:
        physics_tendency: Physics tendencies
        dynamics: PrimitiveEquations object containing the reference temperature and orography

    Returns:
        Dynamics tendencies
    """
    t_tendency = dynamics.coords.horizontal.to_modal(physics_tendency.temperature)
    q_tendency = dynamics.coords.horizontal.to_modal(physics_tendency.specific_humidity)
    
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

    physics_tendency = sum(term(physics_state) for term in physics_terms)

    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics)
    return dynamics_tendency