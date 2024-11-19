'''
Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used. 
'''

from collections import abc
import jax.numpy as jnp
import tree_math
from typing import Callable
from jcm.geometry import hsg, fsg, dhs
from jcm import physical_constants as pc
from jcm.physics_data import PhysicsData
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, compute_diagnostic_state, StateWithTime, PrimitiveEquations, PrimitiveEquationsSpecs

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
            surface_pressure if surface_pressure is not None else jnp.zeros(shape[0:2])
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
    
    def copy(self,u_wind=None,v_wind=None,temperature=None,specific_humidity=None):
        return PhysicsTendency(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity
        )

def initialize_physics():
    # 1.2 Functions of sigma and latitude
    pc.sigh = hsg
    pc.sigl = jnp.log(fsg)
    pc.grdsig = pc.grav/(dhs*pc.p0)
    pc.grdscp = pc.grdsig/pc.cp

    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
    # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
    pc.wvi = jnp.zeros((fsg.shape[0], 2))
    pc.wvi = pc.wvi.at[:-1, 0].set(1./(pc.sigl[1:]-pc.sigl[:-1]))
    pc.wvi = pc.wvi.at[:-1, 1].set((jnp.log(pc.sigh[1:-1])-pc.sigl[:-1])*pc.wvi[:-1, 0])
    pc.wvi = pc.wvi.at[-1, 1].set((jnp.log(0.99)-pc.sigl[-1])*pc.wvi[-2,0])

def dynamics_state_to_physics_state(state: StateWithTime, dynamics: PrimitiveEquations, specs: PrimitiveEquationsSpecs) -> PhysicsState:
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
    )
    phi = dynamics.coords.horizontal.to_nodal(phi_spectral)
    log_sp = dynamics.coords.horizontal.to_nodal(state.log_surface_pressure)
    sp = jnp.exp(log_sp)

    u = specs.dimensionalize(u, units.meter / units.second).m
    v = specs.dimensionalize(v, units.meter / units.second).m
    t = dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis] + t #specs.dimensionalize(t, units.kelvin).m
    # q = specs.dimensionalize(q, units.kilogram / units.kilogram).m
    # phi = specs.dimensionalize(phi, units.meter ** 2 / units.second ** 2).m
    # sp = specs.dimensionalize(sp, units.pascal).m
    print("u: ", jnp.min(u), jnp.max(u))
    print("v: ", jnp.min(v), jnp.max(v))
    print("t: ", jnp.min(t), jnp.max(t))
    print("q: ", jnp.min(q), jnp.max(q))

    physics_state = PhysicsState(
        u.transpose(1, 2, 0),
        v.transpose(1, 2, 0),
        t.transpose(1, 2, 0),
        q.transpose(1, 2, 0),
        phi.transpose(1, 2, 0),
        jnp.squeeze(sp.transpose(1, 2, 0))
    )
    return physics_state


def physics_tendency_to_dynamics_tendency(physics_tendency: PhysicsTendency, dynamics: PrimitiveEquations, specs: PrimitiveEquationsSpecs) -> StateWithTime:
    """
    Convert the physics tendencies to the dynamics tendencies.

    Args:
        physics_tendency: Physics tendencies
        dynamics: PrimitiveEquations object containing the reference temperature and orography

    Returns:
        Dynamics tendencies
    """
    u_tendency_nodal, v_tendency_nodal, t_tendency_nodal, q_tendency_nodal = (v.transpose(2, 0, 1)
                                                                                  for v in (physics_tendency.u_wind,
                                                                                            physics_tendency.v_wind,
                                                                                            physics_tendency.temperature,
                                                                                            physics_tendency.specific_humidity))
    u_tendency_nodal = specs.nondimensionalize(u_tendency_nodal * units.meter / units.second**2)
    v_tendency_nodal = specs.nondimensionalize(v_tendency_nodal * units.meter / units.second**2)
    # t_tendency_nodal = specs.nondimensionalize(t_tendency_nodal * units.kelvin / units.second)
    # q_tendency_nodal = specs.nondimensionalize(q_tendency_nodal * units.kilogram / units.kilogram / units.second)

    vor_tendency, div_tendency = uv_nodal_to_vor_div_modal(
        dynamics.coords.horizontal,
        u_tendency_nodal,# * 21600/jnp.pi,
        v_tendency_nodal# * 21600/jnp.pi
    )
    t_tendency = dynamics.coords.horizontal.to_modal(t_tendency_nodal * 21600/jnp.pi)
    q_tendency = dynamics.coords.horizontal.to_modal(q_tendency_nodal * 21600/jnp.pi)
    
    log_sp_tendency = jnp.zeros_like(t_tendency[0, ...]) # This assumes the physics tendency is zero for log_surface_pressure

    # Create a new state object with the updated tendencies (which will be added to the current state)
    dynamics_tendency = StateWithTime(vor_tendency, div_tendency, t_tendency, log_sp_tendency, sim_time=0., 
                                      tracers={'specific_humidity': q_tendency})
    return dynamics_tendency


def get_physical_tendencies(
    state: StateWithTime,
    dynamics: PrimitiveEquations,
    specs: PrimitiveEquationsSpecs,
    physics_terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]],
    data: PhysicsData = None
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
    physics_state = dynamics_state_to_physics_state(state, dynamics, specs)

    # the 'physics_terms' return an instance of tendencies and data, data gets overwritten at each step 
    # and implicitly passed to the next physics_term. tendencies are summed 
    physics_tendency = PhysicsTendency.zeros(shape=physics_state.u_wind.shape)
    
    print("testing 1")

    for term in physics_terms:
        tend, data = term(physics_state, data)
        physics_tendency += tend

    print("testing 2")

    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics, specs)
    return dynamics_tendency