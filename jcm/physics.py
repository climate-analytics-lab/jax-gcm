'''
Date: 2/7/2024
Physics module.
'''

from collections import abc
import jax
import jax.numpy as jnp
import tree_math
from typing import Callable

from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, State, PrimitiveEquations


# Initialize physical parametrization routines
def initialize_physics():
    from physical_constants import grav, cp, p0, sigl, sigh, grdsig, grdscp, wvi
    from geometry import hsg, fsg, dhs

    '''
    kx = len(fsg)
    sigh = jnp.append(hsg[1:], hsg[-1]) # Slight adjustment to match the Fortran code

    for k in range(kx):
        sigl[k] = jnp.log(fsg[k])
        grdsig[k] = grav / (dhs[k] * p0)
        grdscp[k] = grdsig[k] / cp
    '''

    # 1.2 Functions of sigma and latitude
    sigh = jnp.concatenate((jnp.array([hsg[1]]), hsg[1:]))
    sigl = jnp.log(fsg)
    grdsig = grav / (dhs * p0)
    grdscp = grdsig / cp

    '''
    UNSURE ABOUT BUILT IN ARRAY OPERATIONS OUTPUT
    Code in loops
    # Weights for vertical interpolation at half-levels(1,kx) and surface
    for k in range(kx - 1):
        wvi[k, 0] = 1.0 / (sigl[k + 1] - sigl[k])
        wvi[k, 1] = (jnp.log(sigh[k]) - sigl[k]) * wvi[k, 0]

    wvi = jax.ops.index_update(wvi, kx - 1, jnp.array([0., (jnp.log(0.99) - sigl[kx - 1]) * wvi[kx - 2, 0]]))
    '''

    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
    # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
    wvi_1 = 1.0 / (sigl[1:] - sigl[:-1])
    wvi_2 = (jnp.log(sigh[:-1]) - sigl[:-1]) * wvi_1
    wvi = jnp.column_stack((wvi_1, wvi_2))


    wvi_last = jnp.array([0., (jnp.log(0.99) - sigl[-1]) * wvi[-1, 0]])
    wvi = jnp.vstack((wvi[:-1], wvi_last))

    return sigl, sigh, grdsig, grdscp, wvi


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
    # Calculate u and v from vorticity and divergence
    u, v = vor_div_to_uv_nodal(state, dynamics.coords.horizontal, state.vorticity, state.divergence)

    # Calculate geopotential
    phi_spectral = get_geopotential(
        state.temperature_variation,
        dynamics.reference_temperature,
        dynamics.orography,
        dynamics.coords.vertical,
    )
    # Z, X, Y
    t_spectral = state.temperature_variation + dynamics.reference_temperature[:, np.newaxis, np.newaxis]
    q_spectral = state['tracers']['specific_humidity']

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
    t_tendency = dynamics.coords.horizontal.to_modal(physics_tendency.temperature)
    q_tendency = dynamics.coords.horizontal.to_modal(physics_tendency.specific_humidity)
    log_sp_tendency = jnp.zeros_like(t_tendency)
    dynamics_tendency = State(vor_tendency, div_tendency, t_tendency, log_sp_tendency, {'specific_humidity': q_tendency})
    return dynamics_tendency


def get_physical_tendencies(
    state: State,
    dynamics: PrimitiveEquations,
    physics_terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]],
):
    """
    Computes the physical tendencies of the state variables.

    Args:
        state Dinosaur.State: State variables
        physics_terms [list]: Physical parametrization terms

    Returns:
        Physical tendencies
    """
    physics_state = dynamics_state_to_physics_state(state, dynamics)

    physics_tendency = sum(term(physics_state) for term in physics_terms)
    
    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics)
    return dynamics_tendency
