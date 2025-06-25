"""
Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used.
"""

from collections import abc
import jax
import jax.numpy as jnp
import tree_math
from typing import Callable
from jcm.physics_data import PhysicsData, PhysicsOutputData, SWRadiationOutputData, ConvectionOutputData
from jcm.geometry import Geometry
from jcm.params import Parameters
from dinosaur import scales
from dinosaur.typing import ModelState
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, compute_diagnostic_state, State, PrimitiveEquations
from jax import tree_util
from jcm.boundaries import BoundaryData
import dataclasses

@dataclasses.dataclass
class SpeedyPrimitiveEquations(PrimitiveEquations):
    @jax.named_call
    def explicit_terms(self, state: ModelState) -> ModelState:
        return ModelState(
           state=super().explicit_terms(state.state),
           diagnostics=tree_util.tree_map(lambda x: jnp.zeros_like(x), state.diagnostics)
        )

    @jax.named_call
    def implicit_terms(self, state: ModelState) -> ModelState:
        return ModelState(
           state=super().implicit_terms(state.state),
           diagnostics=tree_util.tree_map(lambda x: jnp.zeros_like(x), state.diagnostics)
        )
    
    @jax.named_call
    def implicit_inverse(self, state: State, step_size: float) -> State:
        # FIXME: double check if this is correct
        return ModelState(
           state=super().implicit_inverse(state.state, step_size),
           diagnostics=state.diagnostics
        )

@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    surface_pressure: jnp.ndarray  # normalized surface pressure (normalized by p0)

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


def physics_state_to_dynamics_state(physics_state: PhysicsState, dynamics: PrimitiveEquations) -> State:

    # Calculate vorticity and divergence from u and v
    modal_vorticity, modal_divergence = uv_nodal_to_vor_div_modal(dynamics.coords.horizontal, physics_state.u_wind, physics_state.v_wind)

    # convert specific humidity to modal (and nondimensionalize)
    q = dynamics.physics_specs.nondimensionalize(physics_state.specific_humidity * units.gram / units.kilogram / units.second)
    q_modal = dynamics.coords.horizontal.to_modal(q)

    # convert temperature to a variation and then to modal
    temperature = physics_state.temperature - dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    temperature_modal = dynamics.coords.horizontal.to_modal(temperature)

    # take the log of normalized surface pressure and convert to modal
    log_surface_pressure = jnp.log(physics_state.surface_pressure)
    modal_log_sp = dynamics.coords.horizontal.to_modal(log_surface_pressure)

    return State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=temperature_modal, # does this need to be referenced to ref_temp ? 
        log_surface_pressure=modal_log_sp,
        tracers={'specific_humidity': q_modal}
    )

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
    dynamics_tendency = State(
        vor_tend_modal,
        div_tend_modal,
        t_tend_modal,
        log_sp_tend_modal,
        sim_time=0.,
        tracers={'specific_humidity': q_tend_modal}
    )
    return dynamics_tendency

def verify_state(state: PhysicsState) -> PhysicsState:
    # set specific humidity to 0.0 if it became negative during the dynamics evaluation
    qa = jnp.where(state.specific_humidity < 0.0, 0.0, state.specific_humidity) 
    updated_state = state.copy(specific_humidity=qa)

    return updated_state

def verify_tendencies(state: PhysicsState, tendencies: PhysicsTendency, time_step) -> PhysicsTendency:
    # set specific humidity tendency such that the resulting specific humidity is non-negative
    dt_seconds = 60 * time_step
    updated_tendencies = tendencies.copy(
        specific_humidity=jnp.where(
            state.specific_humidity + dt_seconds * tendencies.specific_humidity >= 0,
            tendencies.specific_humidity,
            - state.specific_humidity / dt_seconds
        )
    )

    return updated_tendencies

def get_physical_tendencies(
    state: ModelState,
    dynamics: PrimitiveEquations,
    time_step: int,
    physics_terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]],
    boundaries: BoundaryData,
    parameters: Parameters,
    geometry: Geometry,
    data: PhysicsData = None
) -> ModelState:
    """
    Computes the physical tendencies given the current state and a list of physics functions.

    Args:
        state: Dynamic (dinosaur) State variables
        dynamics: PrimitiveEquations object
        physics_terms: List of physics functions that take a PhysicsState and return a PhysicsTendency

    Returns:
        Physical tendencies
    """
    unclamped_physics_state = dynamics_state_to_physics_state(state.state, dynamics)
    physics_state = verify_state(unclamped_physics_state)

    # the 'physics_terms' return an instance of tendencies and data, data gets overwritten at each step
    # and implicitly passed to the next physics_term. tendencies are summed
    physics_tendency = PhysicsTendency.zeros(shape=physics_state.u_wind.shape)
    for term in physics_terms:
        tend, data = term(physics_state, data, parameters, boundaries, geometry)
        physics_tendency += tend
    physics_tendency = verify_tendencies(unclamped_physics_state, physics_tendency, time_step)
    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics)

    output_data = PhysicsOutputData(
        shortwave_rad=SWRadiationOutputData(
            qcloud=data.shortwave_rad.qcloud,
            fsol=data.shortwave_rad.fsol,
            rsds=data.shortwave_rad.rsds,
            rsns=data.shortwave_rad.rsns,
            ozone=data.shortwave_rad.ozone,
            ozupp=data.shortwave_rad.ozupp,
            zenit=data.shortwave_rad.zenit,
            stratz=data.shortwave_rad.stratz,
            gse=data.shortwave_rad.gse,
            cloudc=data.shortwave_rad.cloudc,
            cloudstr=data.shortwave_rad.cloudstr,
            ftop=data.shortwave_rad.ftop,
            dfabs=data.shortwave_rad.dfabs,
        ),
        longwave_rad=data.longwave_rad,
        convection=ConvectionOutputData(
            se=data.convection.se,
            cbmf=data.convection.cbmf,
            precnv=data.convection.precnv,
        ),
        mod_radcon=data.mod_radcon,
        humidity=data.humidity,
        condensation=data.condensation,
        surface_flux=data.surface_flux,
        land_model=data.land_model,
    )

    output_tendency = ModelState(
        state = dynamics_tendency,
        diagnostics = output_data / (time_step * 60.) # FIXME: can we return the output_data directly rather than as a tendency that gets accumulated over the substeps of the rk timestep?
    )
    
    return output_tendency