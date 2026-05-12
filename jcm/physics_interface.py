"""Date: 2/7/2024
Physics module that interfaces between the dynamics and the physics of the model. Should be agnostic
to the specific physics being used.
"""

import jax
import jax.numpy as jnp
import tree_math
from dinosaur import scales
from dinosaur.scales import units
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import (
    compute_diagnostic_state, compute_diagnostic_state_hybrid,
    State, PrimitiveEquations,
    get_geopotential_on_sigma, get_geopotential_on_hybrid,
)
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.filtering import horizontal_diffusion_filter
from jax import tree_util
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from typing import Tuple, Any, Dict, TypeAlias
from jcm.diffusion import DiffusionFilter
import logging

# The cross-step physics carry threaded through the integration scan.
# In the operator-split path (issue #471) the carry is built once at
# ``Model`` construction time by :meth:`Physics.initial_carry_state` and
# threaded through ``Model.run / .resume`` as an explicit pytree.
#
# Today the carry is a ``dict`` of typed sub-structs keyed by name
# (``radiation``, ``vertical_diffusion``, ``clouds``, …). The alias
# documents intent and gives a single name to update if/when we promote
# the carry to a typed ``@tree_math.struct``.
PhysicsCarryState: TypeAlias = Dict[str, Any]

logger = logging.getLogger(__name__)

@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray # Normalized by global mean sea level pressure
    tracers: Dict[str, jnp.ndarray]  # Additional tracers beyond specific_humidity

    def __init__(self, u_wind, v_wind, temperature, specific_humidity, geopotential, normalized_surface_pressure, tracers=None):
        """Initialize PhysicsState with atmospheric variables."""
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.temperature = temperature
        self.specific_humidity = specific_humidity
        self.geopotential = geopotential
        self.normalized_surface_pressure = normalized_surface_pressure
        self.tracers = tracers if tracers is not None else {}

    @classmethod
    def zeros(cls, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, normalized_surface_pressure=None, tracers=None):
        return cls(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape),
            geopotential if geopotential is not None else jnp.zeros(shape),
            normalized_surface_pressure if normalized_surface_pressure is not None else jnp.zeros(shape[1:]),
            tracers if tracers is not None else {}
        )

    @classmethod
    def ones(cls, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, normalized_surface_pressure=None, tracers=None):
        return cls(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape),
            geopotential if geopotential is not None else jnp.ones(shape),
            normalized_surface_pressure if normalized_surface_pressure is not None else jnp.ones(shape[1:]),
            tracers if tracers is not None else {}
        )

    def copy(self,u_wind=None,v_wind=None,temperature=None,specific_humidity=None,geopotential=None,normalized_surface_pressure=None,tracers=None):
        return PhysicsState(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity,
            geopotential if geopotential is not None else self.geopotential,
            normalized_surface_pressure if normalized_surface_pressure is not None else self.normalized_surface_pressure,
            tracers if tracers is not None else self.tracers
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(jnp.any, self))

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
    tracers: Dict[str, jnp.ndarray]  # Tendencies for additional tracers

    def __init__(self, u_wind, v_wind, temperature, specific_humidity, tracers=None):
        """Initialize PhysicsTendency with tendency fields."""
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.temperature = temperature
        self.specific_humidity = specific_humidity
        self.tracers = tracers if tracers is not None else {}

    @classmethod
    def zeros(cls,shape,u_wind=None,v_wind=None,temperature=None,specific_humidity=None,tracers=None):
        return cls(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape),
            tracers if tracers is not None else {}
        )

    @classmethod
    def ones(cls,shape,u_wind=None,v_wind=None,temperature=None,specific_humidity=None,tracers=None):
        return cls(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape),
            tracers if tracers is not None else {}
        )

    def copy(self,u_wind=None,v_wind=None,temperature=None,specific_humidity=None,tracers=None):
        return PhysicsTendency(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity,
            tracers if tracers is not None else self.tracers
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
    UNITS_TABLE_CSV_PATH = None
    cached_coords = None

    def cache_coords(self, coords: CoordinateSystem):
        return None

    def required_tracers(self):
        """Return a tuple of TracerSpec objects this physics needs in state.tracers.

        Default is empty — only ``specific_humidity`` is assumed. Composable
        physics packages override this to aggregate declarations from terms.
        """
        return ()

    def compute_tendencies(self, state: PhysicsState, forcing: ForcingData, terrain: TerrainData, prev_physics_data=None) -> Tuple[PhysicsTendency, Any]:
        """Compute the physical tendencies given the current state and data structs.

        Args:
            state: Current state variables
            forcing: Forcing data — pre-sliced for the current step;
                ``forcing.solar`` carries the orbital geometry physics
                needs, so no calendar is plumbed in here.
            terrain: Terrain data (boundary conditions)
            prev_physics_data: Previous step's physics carry (a
                :data:`PhysicsCarryState`) — used by radiation sub-cycling,
                the analytic TKE source update, etc. ``None`` means "no
                carry available" (snapshot mode under the legacy path, or
                op-split's first ``dt``).

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data

        """
        raise NotImplementedError("Physics compute_tendencies method not implemented.")

    def initial_carry_state(self, coords) -> PhysicsCarryState:
        """Build the cross-step physics carry at ``Model`` construction time.

        Default returns ``{}``. ``ComposablePhysics`` aggregates per-term
        slots; raw subclasses can return whatever ``compute_tendencies``
        expects as ``prev_physics_data``.
        """
        return {}

    def get_empty_data(self, coords) -> Any:
        """Return a zero-shape diagnostics dict (deprecated).

        Used by the legacy ``output_averages=True`` path as the
        stacked-running-mean accumulator seed. Will be removed in Phase
        4 of issue #471 when the legacy path is deleted.
        """
        return None

    def data_struct_to_dict(self, struct: Any, nodal_shape, sep: str = ".") -> dict[str, Any]:
        """Flattens a physics data struct into a dictionary.

        Args:
            struct: The struct to flatten.
            nodal_shape: Shape of the nodal grid (kx, ix, il).
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
                if isinstance(val, jax.Array):
                    items[new_key] = val
                elif hasattr(val, "__dict__") and val.__dict__:
                    items.update(_to_dict_recursive(val, parent_key=new_key))
                else:
                    raise ValueError(f"Unsupported type for key {new_key}: {type(val)}")
            return items

        items = _to_dict_recursive(struct)

        # replace multi-channel fields with a field for each channel
        _original_keys = list(items.keys())
        for k in _original_keys:
            s = items[k].shape
            if len(s) == 5 and s[1:-1] == nodal_shape or len(s) == 4 and s[1:-1] == nodal_shape[1:]:
                items.update({f"{k}{sep}{i}": items[k][..., i] for i in range(s[-1])})
                del items[k]

        return items

def dynamics_state_to_physics_state(
    state: State,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
) -> PhysicsState:
    """Convert the state variables from the dynamics to the physics state variables.

    Args:
        state: Dynamic (dinosaur) State variables
        dynamics: PrimitiveEquations object containing the reference temperature and orography
        tracer_specs: Optional mapping ``name -> TracerSpec``. Tracers whose spec
            has ``nondimensionalize=False`` bypass the gram/kg conversion. When
            ``None`` or a tracer is absent from the mapping, the default gram/kg
            conversion is applied (existing behavior).

    Returns:
        Physics state variables

    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    jax.debug.callback(lambda: logger.debug("Converting state variables from dynamics to physics state variables"))
    # Calculate u and v from vorticity and divergence
    u, v = vor_div_to_uv_nodal(dynamics.coords.horizontal, state.vorticity, state.divergence)

    # Z, X, Y — dispatch to the hybrid variant when the vertical coord is hybrid
    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        nodal_state = compute_diagnostic_state_hybrid(state, dynamics.coords)
    else:
        nodal_state = compute_diagnostic_state(state, dynamics.coords)
    t = nodal_state.temperature_variation
    q = nodal_state.tracers['specific_humidity']

    # Compute geopotential - different approaches for sigma vs hybrid coordinates
    nodal_orography = dynamics.coords.horizontal.to_nodal(dynamics.orography)
    log_sp = dynamics.coords.horizontal.to_nodal(state.log_surface_pressure)
    sp = jnp.exp(log_sp)

    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        # For hybrid coordinates, the state's `log_surface_pressure` already
        # stores log(P_s in nondim Pa); `sp = exp(log_sp)` is therefore the
        # actual surface pressure in the same units as the `a_boundaries`.
        #
        # `get_geopotential_on_hybrid` internally uses method='sparse' which
        # requires surface_pressure with a leading vertical dim (shape
        # (1, lon, lat)); keep `sp` un-squeezed to match.
        full_temperature = nodal_state.temperature_variation + dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
        phi = get_geopotential_on_hybrid(
            temperature=full_temperature,
            surface_pressure=sp,
            specific_humidity=None,
            nodal_orography=nodal_orography,
            coordinates=dynamics.nondim_levels,
            gravity_acceleration=dynamics.physics_specs.nondimensionalize(scales.GRAVITY_ACCELERATION),
            ideal_gas_constant=dynamics.physics_specs.nondimensionalize(scales.IDEAL_GAS_CONSTANT),
            sharding=None,
        )
    else:
        # For sigma coordinates, use the full geopotential calculation in nodal space
        full_temperature = nodal_state.temperature_variation + dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
        phi = get_geopotential_on_sigma(
            temperature=full_temperature,
            specific_humidity=None,
            nodal_orography=nodal_orography,
            sigma=dynamics.coords.vertical,
            gravity_acceleration=dynamics.physics_specs.nondimensionalize(scales.GRAVITY_ACCELERATION),
            ideal_gas_constant=dynamics.physics_specs.nondimensionalize(scales.IDEAL_GAS_CONSTANT),
            sharding=None
        )

    t += dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    q = dynamics.physics_specs.dimensionalize(q, units.gram / units.kilogram).m

    # Extract all tracers from the nodal state (except specific_humidity which is handled separately).
    # Tracers whose TracerSpec has nondimensionalize=False (e.g. number concentrations) pass through
    # untouched; everything else is treated as a mass mixing ratio in gram/kilogram.
    all_tracers = {}
    for tracer_name, tracer_value in nodal_state.tracers.items():
        if tracer_name == 'specific_humidity':
            continue
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            all_tracers[tracer_name] = tracer_value
        else:
            all_tracers[tracer_name] = dynamics.physics_specs.dimensionalize(
                tracer_value, units.gram / units.kilogram
            ).m

    # Produce a PhysicsState with `normalized_surface_pressure = P_s / p0`.
    # For sigma coords state stores log(P_s / p0) already, so sp = P_s/p0.
    # For hybrid coords state stores log(P_s_in_Pa), so we divide by p0 here
    # to put PhysicsState on a common scale the physics routines expect.
    if isinstance(dynamics.coords.vertical, HybridCoordinates):
        from jcm.constants import p0 as P0_PA
        p0_nondim = dynamics.physics_specs.nondimensionalize(P0_PA * units.pascal)
        nsp = jnp.squeeze(sp, axis=-3) / p0_nondim
    else:
        nsp = jnp.squeeze(sp, axis=-3)

    return PhysicsState(u, v, t, q, phi, nsp, all_tracers)

def physics_state_to_dynamics_state(
    physics_state: PhysicsState,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
) -> State:
    """Convert state variables from the physics (nodal space) back to the dynamical core (spectral space).
    This is the inverse of `dynamics_state_to_physics_state`. It is currently not used in the main
    time-stepping loop but can be useful for diagnostics or model initialization.

    Args:
        physics_state: The `PhysicsState` object containing the atmospheric state on the model grid.
        dynamics: The `PrimitiveEquations` object containing model configuration.
        tracer_specs: Optional mapping ``name -> TracerSpec``. Tracers whose spec has
            ``nondimensionalize=False`` are carried through without gram/kg scaling.

    Returns:
        A `State` object for the dynamical core.

    """
    # Calculate vorticity and divergence from u and v
    modal_vorticity, modal_divergence = uv_nodal_to_vor_div_modal(dynamics.coords.horizontal, physics_state.u_wind, physics_state.v_wind)

    # convert specific humidity to modal (and nondimensionalize)
    q = dynamics.physics_specs.nondimensionalize(physics_state.specific_humidity * units.gram / units.kilogram)
    q_modal = dynamics.coords.horizontal.to_modal(q)

    # convert temperature to a variation and then to modal
    temperature = physics_state.temperature - dynamics.reference_temperature[:, jnp.newaxis, jnp.newaxis]
    temperature_modal = dynamics.coords.horizontal.to_modal(temperature)

    # take the log of normalized surface pressure and convert to modal
    log_surface_pressure = jnp.log(physics_state.normalized_surface_pressure)
    modal_log_sp = dynamics.coords.horizontal.to_modal(log_surface_pressure)

    # Convert all tracers to modal; respect TracerSpec.nondimensionalize to
    # decide whether the gram/kg scaling applies.
    tracers_modal = {'specific_humidity': q_modal}
    for tracer_name, tracer_value in physics_state.tracers.items():
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            tracer_nd = tracer_value
        else:
            tracer_nd = dynamics.physics_specs.nondimensionalize(
                tracer_value * units.gram / units.kilogram
            )
        tracers_modal[tracer_name] = dynamics.coords.horizontal.to_modal(tracer_nd)

    return State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=temperature_modal, # does this need to be referenced to ref_temp ?
        log_surface_pressure=modal_log_sp[..., jnp.newaxis, :, :], # Dinosaur expects log_sp to have a vertical dimension
        tracers=tracers_modal
    )

def physics_tendency_to_dynamics_tendency(
    physics_tendency: PhysicsTendency,
    dynamics: PrimitiveEquations,
    tracer_specs: dict | None = None,
) -> State:
    """Convert the physics tendencies to the dynamics tendencies.

    Args:
        physics_tendency: Physics tendencies
        dynamics: PrimitiveEquations object containing the reference temperature and orography
        tracer_specs: Optional mapping ``name -> TracerSpec``. Tracer tendencies whose spec
            has ``nondimensionalize=False`` are carried through without gram/kg/second scaling.

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

    # dinosaur ``State.log_surface_pressure`` is shape ``(1, n_lat_modes,
    # n_lon_modes)`` (a leading vertical axis of size 1 for broadcasting
    # against full vertical fields). The op-split path adds the tendency
    # directly to the state via tree_math, which checks for *exact*
    # shape matches. Keep the leading axis here so the same tendency
    # object works for both the legacy ``compose_equations`` path
    # (which sums via numpy broadcasting) and the op-split path.
    log_sp_tend_modal = jnp.zeros_like(t_tend_modal[:1, ...])

    # Convert all tracer tendencies to modal; TracerSpec.nondimensionalize=False
    # tendencies carry the same (pre-nondimensional) units as the tracer itself
    # divided by model time, so we only strip the /second for the default path.
    tracers_tend_modal = {'specific_humidity': q_tend_modal}
    for tracer_name, tracer_tend in physics_tendency.tracers.items():
        spec = tracer_specs.get(tracer_name) if tracer_specs else None
        if spec is not None and not spec.nondimensionalize:
            tracer_tend_nd = tracer_tend
        else:
            tracer_tend_nd = dynamics.physics_specs.nondimensionalize(
                tracer_tend * units.gram / units.kilogram / units.second
            )
        tracers_tend_modal[tracer_name] = dynamics.coords.horizontal.to_modal(tracer_tend_nd)

    # Create a new state object with the updated tendencies (which will be added to the current state)
    dynamics_tendency = State(
        vor_tend_modal,
        div_tend_modal,
        t_tend_modal,
        log_sp_tend_modal,
        sim_time=0.,
        tracers=tracers_tend_modal
    )
    return dynamics_tendency

# Tracer names that are physically non-negative (mass mixing ratios,
# number concentrations, fractions). Any tracer in this set gets clipped
# to ``>= 0`` on its way into and out of physics. The clip is applied as
# a positive-definite filter so that small negatives produced by the
# horizontal-spectral round-trip of advected fields don't propagate into
# downstream physics terms. We deliberately do NOT clip to an upper bound:
# unphysically large values should surface as a visible regression rather
# than be silently masked.
_NON_NEGATIVE_TRACERS = frozenset({
    "specific_humidity", "qc", "qi", "qr", "qs", "qnc", "qni",
    "co2_vmr", "methane_vmr", "ozone_vmr",
})


def _clip_non_negative_tracers(tracers: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Return a copy of ``tracers`` with positive-definite ones clamped to ``>= 0``."""
    return {
        name: (jnp.maximum(value, 0.0) if name in _NON_NEGATIVE_TRACERS else value)
        for name, value in tracers.items()
    }


def verify_state(state: PhysicsState) -> PhysicsState:
    """Ensure the physical validity of the state variables.

    Clips ``specific_humidity`` and every positive-definite tracer (cloud
    water, ice, rain, snow, droplet- and ice-number concentrations, GHG
    volume mixing ratios) to ``>= 0``. We deliberately do NOT clip to an
    upper bound — aggressive caps hide bugs in the physics (particularly
    convection) that should surface as unphysical values rather than be
    silently masked. Individual physics routines apply local NaN-avoidance
    guards on their own narrow scopes (e.g. the ``q / (1-q)`` conversion
    in radiation).

    The clip is the visible side of a positive-definite filter that
    catches the small negatives the spectral horizontal-advection round-
    trip leaves on advected scalars (see the q-ringing fix in PR #458 for
    why this matters for the moisture cycle). It runs once at the start
    of every physics step on the gridpoint state.

    Args:
        state: The ``PhysicsState`` object.

    Returns:
        The verified and potentially corrected ``PhysicsState``.

    """
    qa = jnp.maximum(state.specific_humidity, 0.0)
    return state.copy(
        specific_humidity=qa,
        tracers=_clip_non_negative_tracers(state.tracers),
    )


def verify_tendencies(state: PhysicsState, tendencies: PhysicsTendency, time_step) -> PhysicsTendency:
    """Adjust tendencies to prevent the state from becoming physically invalid in the next time step.

    For every positive-definite scalar (``specific_humidity`` plus the
    set of tracers in ``_NON_NEGATIVE_TRACERS``) we cap the negative part
    of the tendency at ``-state / dt``, i.e. just enough to drive the
    field to zero rather than below. This mirrors what an implicit step
    on a linear sink would do for the same field.

    Args:
        state: The current ``PhysicsState`` (already passed through
            ``verify_state``).
        tendencies: The physics tendencies.
        time_step: The model time step in seconds.

    Returns:
        The verified ``PhysicsTendency``.

    """
    def _cap_negative_tend(value, tend):
        next_value = value + time_step * tend
        return jnp.where(next_value < 0, -value / time_step, tend)

    clipped_dqdt = _cap_negative_tend(
        state.specific_humidity, tendencies.specific_humidity,
    )
    clipped_tracer_tends = {
        name: (
            _cap_negative_tend(state.tracers[name], tend)
            if name in _NON_NEGATIVE_TRACERS and name in state.tracers
            else tend
        )
        for name, tend in tendencies.tracers.items()
    }
    return tendencies.copy(
        specific_humidity=clipped_dqdt,
        tracers=clipped_tracer_tends,
    )

def compute_physics_step(
    state: State,
    dynamics: PrimitiveEquations,
    time_step: float,
    physics: Physics,
    forcing: ForcingData,
    terrain: TerrainData,
    physics_state,
) -> tuple[State, Any]:
    """Compute the physics dynamics-tendency for an operator-split timestep.

    The operator-split path (issue #471) calls physics exactly once per
    ``dt`` rather than once per IMEX RK substage. There is no substage
    cache to gate against — ``physics_state`` is an explicit pytree
    carry threaded through the integration scan.

    Args:
        state: Dynamic (dinosaur) ``State`` in spectral space.
        dynamics: ``PrimitiveEquations`` instance (provides coords, ref
            temperature, orography, etc.).
        time_step: Model timestep in seconds (used by
            ``verify_tendencies`` to cap negative-going tracer
            tendencies).
        physics: ``Physics`` instance (e.g. :class:`ComposablePhysics`).
        forcing: Time-sliced ``ForcingData`` for this step. ``solar``
            already carries the per-step orbital geometry, so physics
            never needs the calendar / model step.
        terrain: ``TerrainData`` (orography, land-sea mask, …).
        physics_state: The cross-step physics carry — the dict returned
            by the previous step's ``compute_tendencies`` call, or the
            initial value produced by ``physics.initial_carry_state``.

    Returns:
        Tuple of ``(dynamics_tendency, new_physics_state)``. The first
        is in dinosaur ``State`` form ready to be added to the
        primitive-equations state; the second is the carry to thread
        into the next step.

    """
    tracer_specs = {spec.name: spec for spec in physics.required_tracers()}
    physics_grid_state = dynamics_state_to_physics_state(
        state, dynamics, tracer_specs=tracer_specs,
    )
    clamped_physics_state = verify_state(physics_grid_state)

    physics_tendency, new_physics_state = physics.compute_tendencies(
        clamped_physics_state, forcing, terrain,
        prev_physics_data=physics_state,
    )
    physics_tendency = verify_tendencies(
        clamped_physics_state, physics_tendency, time_step,
    )
    dynamics_tendency = physics_tendency_to_dynamics_tendency(
        physics_tendency, dynamics, tracer_specs=tracer_specs,
    )
    return dynamics_tendency, new_physics_state


def filter_tendencies(dynamics_tendency: State,
                      diffusion: DiffusionFilter,
                      time_step, 
                      grid) -> State:
    """Apply dinosaur horizontal diffusion filter to the dynamics divergence tendency

    Args:
        dynamics_tendency: Dynamics tendencies in dinosaur.primitive_equations.State format
        diffusion: DiffusionFilter object containing the diffusion parameters
        time_step: Time step in seconds
        grid: dinosaur.spherical_harmonic.Grid object
    
    Returns:
        Filtered dynamics tendencies in dinosaur.primitive_equations.State format

    """
    # Hyperdiffuse every spectral prognostic, with the timescale + order
    # ECHAM uses for each (divergence shortest, vorticity / specific
    # humidity intermediate, temperature longest). The previous version
    # only filtered ``divergence`` — vorticity, T', log_ps and all tracers
    # passed through unfiltered, so high-wavenumber spectral content from
    # sharp gradients (in particular the surface-evap PBL profile)
    # accumulated step after step. For ``specific_humidity`` on T63L47
    # hybrid + real terrain that grew the round-off (~1e-24 g/kg) into a
    # 1e-3 g/kg negative-q hole within ~20 steps, which drove the supersat
    # / convective-heating runaway. Microphysics tracers (qc, qi, qr, qs,
    # qnc, qni) are deliberately NOT filtered — they live on cloud bases /
    # fronts and hyperdiffusion would over-smear them.
    eig_max_abs = abs(grid.laplacian_eigenvalues[-1])

    div_scale = time_step / (
        diffusion.div_timescale * eig_max_abs ** diffusion.div_order
    )
    div_filter = horizontal_diffusion_filter(
        grid, scale=div_scale, order=int(diffusion.div_order),
    )
    vor_q_scale = time_step / (
        diffusion.vor_q_timescale * eig_max_abs ** diffusion.vor_q_order
    )
    vor_q_filter = horizontal_diffusion_filter(
        grid, scale=vor_q_scale, order=int(diffusion.vor_q_order),
    )
    temp_scale = time_step / (
        diffusion.temp_timescale * eig_max_abs ** diffusion.temp_order
    )
    temp_filter = horizontal_diffusion_filter(
        grid, scale=temp_scale, order=int(diffusion.temp_order),
    )

    # Each filter is a tree_map of `scale * x` over leaves with the matching
    # spectral shape. We apply each to the appropriate variable explicitly,
    # ignoring the filtered values for the other fields.
    filtered_div = div_filter(dynamics_tendency).divergence
    filtered_vor = vor_q_filter(dynamics_tendency).vorticity
    filtered_temp = temp_filter(dynamics_tendency).temperature_variation

    filtered_tracers = dict(dynamics_tendency.tracers)
    if "specific_humidity" in filtered_tracers:
        filtered_q = vor_q_filter(dynamics_tendency).tracers["specific_humidity"]
        filtered_tracers["specific_humidity"] = filtered_q

    return State(
        vorticity=filtered_vor,
        divergence=filtered_div,
        temperature_variation=filtered_temp,
        log_surface_pressure=dynamics_tendency.log_surface_pressure,
        sim_time=dynamics_tendency.sim_time,
        tracers=filtered_tracers,
    )