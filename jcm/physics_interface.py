"""
Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used.
"""

import jax.numpy as jnp
import tree_math
from jcm.geometry import Geometry
from dinosaur import scales
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, compute_diagnostic_state, State, PrimitiveEquations
from jax import tree_util
from jcm.boundaries import BoundaryData
from jcm.date import DateData
from typing import Tuple, Dict, Any

@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray # Normalized by global mean sea level pressure

    @classmethod
    def zeros(cls, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, normalized_surface_pressure=None):
        return cls(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape),
            geopotential if geopotential is not None else jnp.zeros(shape),
            normalized_surface_pressure if normalized_surface_pressure is not None else jnp.zeros(shape[1:])
        )

    @classmethod
    def ones(cls, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, normalized_surface_pressure=None):
        return cls(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape),
            geopotential if geopotential is not None else jnp.ones(shape),
            normalized_surface_pressure if normalized_surface_pressure is not None else jnp.ones(shape[1:])
        )

    def copy(self,u_wind=None,v_wind=None,temperature=None,specific_humidity=None,geopotential=None,normalized_surface_pressure=None):
        return PhysicsState(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity,
            geopotential if geopotential is not None else self.geopotential,
            normalized_surface_pressure if normalized_surface_pressure is not None else self.normalized_surface_pressure
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))

PhysicsState.__doc__ = """Represents the state of the atmosphere in physical (nodal) space.

This structure holds the atmospheric variables on a grid, which are used as
inputs for the physics parameterizations.

Attributes:
    u_wind : jnp.ndarray
        Zonal (east-west) component of wind.
    v_wind : jnp.ndarray
        Meridional (north-south) component of wind.
    temperature : jnp.ndarray
        Atmospheric temperature.
    specific_humidity : jnp.ndarray
        The mass of water vapor per unit mass of moist air.
    geopotential : jnp.ndarray
        The gravitational potential energy per unit mass at a given height.
    normalized_surface_pressure : jnp.ndarray
        Surface pressure normalized by a reference pressure p0.
"""

@tree_math.struct
class PhysicsTendency:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray

    @classmethod
    def zeros(cls,shape,u_wind=None,v_wind=None,temperature=None,specific_humidity=None):
        return cls(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape)
        )

    @classmethod
    def ones(cls,shape,u_wind=None,v_wind=None,temperature=None,specific_humidity=None):
        return cls(
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

PhysicsTendency.__doc__ = """Represents the tendencies (rates of change) of physical variables.
These tendencies are computed by the physics parameterizations and are used
to update the model state over a time step.

Attributes:
    u_wind : jnp.ndarray
        Tendency of the zonal wind component.
    v_wind : jnp.ndarray
        Tendency of the meridional wind component.
    temperature : jnp.ndarray
        Tendency of temperature.
    specific_humidity : jnp.ndarray
        Tendency of specific humidity.
"""

class Physics:
    write_output: bool
    
    def compute_tendencies(self, state: PhysicsState, boundaries: BoundaryData, geometry: Geometry, date: DateData) -> Tuple[PhysicsTendency, Any]:
        """
        Compute the physical tendencies given the current state and data structs.

        Args:
            state: Current state variables
            boundaries: Boundary data
            geometry: Geometry data
            date: Date data

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data
        """
        raise NotImplementedError("Physics compute_tendencies method not implemented.")

    def data_struct_to_dict(self, struct: Any, geometry: Geometry, sep: str = ".") -> dict[str, Any]:
        """
        Flattens a physics data struct into a dictionary.

        Args:
            struct: The struct to flatten.
            geometry: Geometry object.
            sep: Separator to use for constructing hierarchical keys.

        Returns:
            A dictionary representation of the struct, without nesting.
        """
        if struct is None:
            return {}
        
        def _to_dict_recursive(obj, parent_key=""):
            items = {}
            for key, val in obj.__dict__.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if hasattr(val, "__dict__") and val.__dict__:
                    items.update(_to_dict_recursive(val, parent_key=new_key))
                else:
                    items[new_key] = val
            return items
        
        items = _to_dict_recursive(struct)

        # replace multi-channel fields with a field for each channel
        _original_keys = list(items.keys())
        for k in _original_keys:
            s = items[k].shape
            if len(s) == 5 and s[1:-1] == geometry.nodal_shape or len(s) == 4 and s[1:-1] == geometry.nodal_shape[1:]:
                items.update({f"{k}{sep}{i}": items[k][..., i] for i in range(s[-1])})
                del items[k]

        return items

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
    """
    Converts state variables from the physics (nodal space) back to the dynamical core (spectral space).
    This is the inverse of `dynamics_state_to_physics_state`. It is currently not used in the main
    time-stepping loop but can be useful for diagnostics or model initialization.
    
    Args:
        physics_state: The `PhysicsState` object containing the atmospheric state on the model grid.
        dynamics: The `PrimitiveEquations` object containing model configuration.
    
    Returns:
        A `State` object for the dynamical core.
    """
    # Calculate vorticity and divergence from u and v
    modal_vorticity, modal_divergence = uv_nodal_to_vor_div_modal(dynamics.coords.horizontal, physics_state.u_wind, physics_state.v_wind)

    # convert specific humidity to modal (and nondimensionalize)
    q = dynamics.physics_specs.nondimensionalize(physics_state.specific_humidity * units.gram / units.kilogram / units.second)
    q_modal = dynamics.coords.horizontal.to_modal(q)

    # convert temperature to a variation and then to modal
    temperature = physics_state.temperature - dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    temperature_modal = dynamics.coords.horizontal.to_modal(temperature)

    # take the log of normalized surface pressure and convert to modal
    log_surface_pressure = jnp.log(physics_state.normalized_surface_pressure)
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
    """
    Ensures the physical validity of the state variables.
    
    Args:
        state: The `PhysicsState` object.
    
    Returns:
        The verified and potentially corrected `PhysicsState` object.
    """
    # set specific humidity to 0.0 if it became negative during the dynamics evaluation
    qa = jnp.where(state.specific_humidity < 0.0, 0.0, state.specific_humidity)
    updated_state = state.copy(specific_humidity=qa)

    return updated_state

def verify_tendencies(state: PhysicsState, tendencies: PhysicsTendency, time_step) -> PhysicsTendency:
    """
    Adjusts tendencies to prevent the state from becoming physically invalid in the next time step.
    
    Args:
        state: The current `PhysicsState`.
        tendencies: The computed `PhysicsTendency`.
        time_step: The model time step in seconds.
    
    Returns:
        The verified and potentially corrected `PhysicsTendency` object.
    """
    # set specific humidity tendency such that the resulting specific humidity is non-negative
    updated_tendencies = tendencies.copy(
        specific_humidity=jnp.where(
            state.specific_humidity + time_step * tendencies.specific_humidity >= 0,
            tendencies.specific_humidity,
            - state.specific_humidity / time_step
        )
    )

    return updated_tendencies

def get_physical_tendencies(
    state: State,
    dynamics: PrimitiveEquations,
    time_step: float,
    physics: Physics,
    boundaries: BoundaryData,
    geometry: Geometry,
    date: DateData,
) -> State:
    """
    Computes the physical tendencies given the current state and a list of physics functions.

    Args:
        state: Dynamic (dinosaur) State variables
        dynamics: PrimitiveEquations object
        time_step: Time step in seconds
        physics: Physics object (e.g. HeldSuarezPhysics, SpeedyPhysics)
        boundaries: BoundaryData object
        geometry: Geometry object
        date: DateData object

    Returns:
        Physical tendencies in dinosaur.primitive_equations.State format
    """
    physics_state = dynamics_state_to_physics_state(state, dynamics)

    clamped_physics_state = verify_state(physics_state)
    physics_tendency, _ = physics.compute_tendencies(clamped_physics_state, boundaries, geometry, date)

    physics_tendency = verify_tendencies(physics_state, physics_tendency, time_step)
    dynamics_tendency = physics_tendency_to_dynamics_tendency(physics_tendency, dynamics)
    return dynamics_tendency