"""
Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used.
"""

from collections import abc
import jax.numpy as jnp
import tree_math
from typing import Callable
from jcm.physics_data import PhysicsData
from jcm.geometry import Geometry
from jcm.params import Parameters
from dinosaur import scales
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, compute_diagnostic_state, State, PrimitiveEquations
from jax import tree_util
from jcm.boundaries import BoundaryData

@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    surface_pressure: jnp.ndarray

    @classmethod
    def zeros(self, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, surface_pressure=None):
        return PhysicsState(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape),
            geopotential if geopotential is not None else jnp.zeros(shape),
            surface_pressure if surface_pressure is not None else jnp.zeros(shape[1:])
        )

    @classmethod
    def ones(self, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, surface_pressure=None):
        return PhysicsState(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape),
            geopotential if geopotential is not None else jnp.ones(shape),
            surface_pressure if surface_pressure is not None else jnp.ones(shape[1:])
        )

    def copy(self,u_wind=None,v_wind=None,temperature=None,specific_humidity=None,geopotential=None,surface_pressure=None):
        return PhysicsState(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity,
            geopotential if geopotential is not None else self.geopotential,
            surface_pressure if surface_pressure is not None else self.surface_pressure
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))

@tree_math.struct
class PhysicsTendency:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray

    @classmethod
    def zeros(self,shape,u_wind=None,v_wind=None,temperature=None,specific_humidity=None):
        return PhysicsTendency(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape)
        )

    @classmethod
    def ones(self,shape,u_wind=None,v_wind=None,temperature=None,specific_humidity=None):
        return PhysicsTendency(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape)
        )

    def copy(self,u_wind=None,v_wind=None,temperature=None,specific_humidity=None):
        return PhysicsTendency(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity
        )

def physics_state_to_dynamics_state(physics_state: PhysicsState, dynamics: PrimitiveEquations) -> State:
    
    # Calculate vorticity and divergence from u and v
    modal_vorticity, modal_divergence = uv_nodal_to_vor_div_modal(dynamics.coords.horizontal, physics_state.u, physics_state.v)

    # convert specific humidity to modal (and nondimensionalize)
    q = dynamics.physics_specs.nondimensionalize(physics_state.specific_humidity * units.gram / units.kilogram / units.second)
    q_modal = dynamics.coords.horizontal.to_modal(q)

    # convert temperature to a variation and then to modal
    temperature = physics_state.temperature - dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    temperature_modal = dynamics.coords.horizontal.to_modal(temperature)

    # convert log surface pressure to modal
    modal_log_sp = dynamics.coords.horizontal.to_modal(jnp.log(physics_state.surface_pressure))

    return State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=temperature_modal, # does this need to be referenced to ref_temp ? 
        log_surface_pressure=modal_log_sp,
        tracers={'specific_humidity': q_modal}
        )


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

    # Z, X, Y
    nodal_state = compute_diagnostic_state(state, dynamics.coords)
    t = nodal_state.temperature_variation
    q = nodal_state.tracers['specific_humidity']

    phi_spectral = get_geopotential(
        state.temperature_variation,
        dynamics.reference_temperature,
        dynamics.orography,
        dynamics.coords.vertical,
        dynamics.physics_specs.nondimensionalize(scales.GRAVITY_ACCELERATION),
        dynamics.physics_specs.nondimensionalize(scales.IDEAL_GAS_CONSTANT),
    )

    phi = dynamics.coords.horizontal.to_nodal(phi_spectral)
    log_sp = dynamics.coords.horizontal.to_nodal(state.log_surface_pressure)
    sp = jnp.exp(log_sp)

    t += dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    q = dynamics.physics_specs.dimensionalize(q, units.gram / units.kilogram).m

    return PhysicsState(u, v, t, q, phi, jnp.squeeze(sp))


def physics_tendency_to_dynamics_tendency(physics_tendency: PhysicsTendency, dynamics: PrimitiveEquations) -> State:
    """
    Convert the physics tendencies to the dynamics tendencies.

    Args:
        physics_tendency: Physics tendencies
        dynamics: PrimitiveEquations object containing the reference temperature and orography

    Returns:
        Dynamics tendencies
    """
    u_tend = physics_tendency.u_wind
    v_tend = physics_tendency.v_wind
    t_tend = physics_tendency.temperature
    q_tend = physics_tendency.specific_humidity
    
    q_tend = dynamics.physics_specs.nondimensionalize(q_tend * units.gram / units.kilogram / units.second)
    
    vor_tend_modal, div_tend_modal = uv_nodal_to_vor_div_modal(dynamics.coords.horizontal, u_tend, v_tend)
    t_tend_modal = dynamics.coords.horizontal.to_modal(t_tend)
    q_tend_modal = dynamics.coords.horizontal.to_modal(q_tend)
    
    log_sp_tend_modal = jnp.zeros_like(t_tend_modal[0, ...])

    # Create a new state object with the updated tendencies (which will be added to the current state)
    dynamics_tendency = State(vor_tend_modal,
                                      div_tend_modal,
                                      t_tend_modal,
                                      log_sp_tend_modal,
                                      sim_time=0.,
                                      tracers={'specific_humidity': q_tend_modal})
    return dynamics_tendency


def get_physical_tendencies(
    state: State,
    dynamics: PrimitiveEquations,
    time_step: int,
    physics_terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]],
    boundaries: BoundaryData,
    parameters: Parameters,
    geometry: Geometry,
    data: PhysicsData = None
    ) -> State:
    """
    Computes the physical tendencies given the current state and a list of physics functions.

    Args:
        state: Dynamic (dinosaur) State variables
        dynamics: PrimitiveEquations object
        physics_terms: List of physics functions that take a PhysicsState and return a PhysicsTendency

    Returns:
        Physical tendencies
    """
    physics_state = dynamics_state_to_physics_state(state, dynamics)

    # the 'physics_terms' return an instance of tendencies and data, data gets overwritten at each step
    # and implicitly passed to the next physics_term. tendencies are summed
    physics_tendency = PhysicsTendency.zeros(shape=physics_state.u_wind.shape)
    
    for term in physics_terms:
        tend, data = term(physics_state, data, parameters, boundaries, geometry)
        physics_tendency += tend

    # the actual timestep size seems to be 1/3 of time_step
    # so I'm setting the tendency to -q/(60s/min * 1/3 * time_step) to clamp q > 0
    dt_seconds = 20 * time_step
    physics_tendency = physics_tendency.copy(
        specific_humidity=jnp.where(
            physics_state.specific_humidity + dt_seconds * physics_tendency.specific_humidity >= 0,
            physics_tendency.specific_humidity,
            - physics_state.specific_humidity / dt_seconds
        )
    )

    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics)
    return dynamics_tendency