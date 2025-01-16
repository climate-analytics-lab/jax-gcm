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
from jcm.params import Parameters
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, compute_diagnostic_state, StateWithTime, PrimitiveEquations, PrimitiveEquationsSpecs
from jax import tree_util

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
    
    @classmethod
    def ones(self, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, surface_pressure=None):
        return PhysicsState(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape),
            geopotential if geopotential is not None else jnp.ones(shape),
            surface_pressure if surface_pressure is not None else jnp.ones(shape[0:2])
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

def dynamics_state_to_physics_state(state: StateWithTime, dynamics: PrimitiveEquations) -> PhysicsState:
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

    # note: surface pressure is nondimensionalized by assuming mean(log_sp) = 0, which means it uses a pressure scale of 1e5, different from the scales used by other quantities, so we don't dimensionalize it here
    u = dynamics.physics_specs.dimensionalize(u, units.meter / units.second).m
    v = dynamics.physics_specs.dimensionalize(v, units.meter / units.second).m
    t = dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis] + dynamics.physics_specs.dimensionalize(t, units.kelvin).m
    q = dynamics.physics_specs.dimensionalize(q, units.gram / units.kilogram).m
    phi = dynamics.physics_specs.dimensionalize(phi, units.meter ** 2 / units.second ** 2).m

    physics_state = PhysicsState(
        u.transpose(1, 2, 0),
        v.transpose(1, 2, 0),
        t.transpose(1, 2, 0),
        q.transpose(1, 2, 0),
        phi.transpose(1, 2, 0),
        jnp.squeeze(sp.transpose(1, 2, 0))
    )
    return physics_state


def physics_tendency_to_dynamics_tendency(physics_tendency: PhysicsTendency, dynamics: PrimitiveEquations) -> StateWithTime:
    """
    Convert the physics tendencies to the dynamics tendencies.

    Args:
        physics_tendency: Physics tendencies
        dynamics: PrimitiveEquations object containing the reference temperature and orography

    Returns:
        Dynamics tendencies
    """
    nondimensionalize = lambda tend, unit: dynamics.physics_specs.nondimensionalize(tend.transpose(2, 0, 1) * unit / units.second)
    u_tend = nondimensionalize(physics_tendency.u_wind, units.meter / units.second)
    v_tend = nondimensionalize(physics_tendency.v_wind, units.meter / units.second)
    t_tend = nondimensionalize(physics_tendency.temperature, units.kelvin)
    q_tend = nondimensionalize(physics_tendency.specific_humidity, units.gram / units.kilogram)
    
    vor_tend_modal, div_tend_modal = uv_nodal_to_vor_div_modal(dynamics.coords.horizontal, u_tend, v_tend)
    t_tend_modal = dynamics.coords.horizontal.to_modal(t_tend)
    q_tend_modal = dynamics.coords.horizontal.to_modal(q_tend)
    
    log_sp_tend_modal = jnp.zeros_like(t_tend_modal[0, ...])

    # Create a new state object with the updated tendencies (which will be added to the current state)
    dynamics_tendency = StateWithTime(vor_tend_modal,
                                      div_tend_modal,
                                      t_tend_modal,
                                      log_sp_tend_modal,
                                      sim_time=0., 
                                      tracers={'specific_humidity': q_tend_modal})
    return dynamics_tendency


def get_physical_tendencies(
    state: StateWithTime,
    dynamics: PrimitiveEquations,
    time_step: int,
    physics_terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]],
    data: PhysicsData = None,
    boundaries: BoundaryData = None,
    parameters: Parameters = None
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
    physics_tendency = PhysicsTendency.zeros(shape=physics_state.u_wind.shape)
    
    for term in physics_terms:
        tend, data = term(physics_state, data, boundaries)
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

def spectral_truncation(dynamics: PrimitiveEquations, fg1, trunc):
    # given fsp, a spectral representation of a field, return a truncated version
    nx = trunc+2 # Number of total wavenumbers for spectral storage arrays
    mx = trunc+1 # Number of zonal wavenumbers for spectral storage arrays

    fsp = dynamics.coords.horizontal.to_modal(fg1)

    n_indices, m_indices = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx), indexing='ij')
    total_wavenumber = m_indices + n_indices
    fsp = jnp.where(total_wavenumber > trunc, 0.0, fsp)

    fg2 = dynamics.coords.horizontal.to_nodal(fg1)

    return fg2