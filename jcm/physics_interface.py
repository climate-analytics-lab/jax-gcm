"""Dycore-agnostic physics interface types and helpers.

This module defines the gridpoint-space data structures that physics packages
consume and produce — :class:`PhysicsState` and :class:`PhysicsTendency` — and
the :class:`Physics` base class that they implement.

It is **dycore-agnostic**: no spectral transforms or ``dinosaur`` symbols
appear here. The actual dycore↔physics conversion is owned by each backend
under ``jcm/dycore/<backend>/state_bridge.py`` (see
:mod:`jcm.dycore.dinosaur.state_bridge` for the canonical example).
"""

import jax
import jax.numpy as jnp
import tree_math
from jax import tree_util
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from typing import Tuple, Any, Dict, TypeAlias
import logging

# The cross-step physics carry threaded through the integration scan. In the
# operator-split path (issue #471) the carry is built once at ``Model``
# construction time by :meth:`Physics.initial_carry_state` and threaded
# through ``Model.run / .resume`` as an explicit pytree.
#
# Today the carry is a ``dict`` of typed sub-structs keyed by name
# (``radiation``, ``vertical_diffusion``, ``clouds``, …). The alias documents
# intent and gives a single name to update if/when we promote the carry to a
# typed ``@tree_math.struct``.
PhysicsCarryState: TypeAlias = Dict[str, Any]

logger = logging.getLogger(__name__)


# ``PhysicsState`` is the *physics-facing* common gridpoint type. Every
# physics scheme — radiation, convection, vertical diffusion, surface flux
# — consumes winds in physical space, so the boundary is u/v. A dycore
# whose native prognostic representation is something else (e.g. spectral
# vorticity / divergence for the dinosaur backend) is free to do so — the
# conversion to u/v happens inside the dycore's ``to_physics_state`` and
# is not visible to physics packages.


@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    geopotential: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray  # Normalized by global mean sea level pressure
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

    def copy(self, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, geopotential=None, normalized_surface_pressure=None, tracers=None):
        return PhysicsState(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity,
            geopotential if geopotential is not None else self.geopotential,
            normalized_surface_pressure if normalized_surface_pressure is not None else self.normalized_surface_pressure,
            tracers if tracers is not None else self.tracers,
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
    def zeros(cls, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, tracers=None):
        return cls(
            u_wind if u_wind is not None else jnp.zeros(shape),
            v_wind if v_wind is not None else jnp.zeros(shape),
            temperature if temperature is not None else jnp.zeros(shape),
            specific_humidity if specific_humidity is not None else jnp.zeros(shape),
            tracers if tracers is not None else {}
        )

    @classmethod
    def ones(cls, shape, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, tracers=None):
        return cls(
            u_wind if u_wind is not None else jnp.ones(shape),
            v_wind if v_wind is not None else jnp.ones(shape),
            temperature if temperature is not None else jnp.ones(shape),
            specific_humidity if specific_humidity is not None else jnp.ones(shape),
            tracers if tracers is not None else {}
        )

    def copy(self, u_wind=None, v_wind=None, temperature=None, specific_humidity=None, tracers=None):
        return PhysicsTendency(
            u_wind if u_wind is not None else self.u_wind,
            v_wind if v_wind is not None else self.v_wind,
            temperature if temperature is not None else self.temperature,
            specific_humidity if specific_humidity is not None else self.specific_humidity,
            tracers if tracers is not None else self.tracers,
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

    def cache_coords(self, coords):
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
            state: Current state variables.
            forcing: Forcing data — pre-sliced for the current step (the
                Model collapses every time-varying leaf, including
                ``solar`` and ``nudging_target``, before calling here).
            terrain: Terrain data (boundary conditions).
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


def compute_physics_step_gridpoint(
    physics_state: PhysicsState,
    forcing: ForcingData,
    terrain: TerrainData,
    physics_state_carry,
    *,
    physics: Physics,
    time_step: float,
) -> Tuple[PhysicsTendency, Any]:
    """Run the operator-split physics step in gridpoint space.

    Pure gridpoint flow: :func:`verify_state` →
    ``physics.compute_tendencies`` → :func:`verify_tendencies`. The dycore
    is responsible for the gridpoint↔native conversions either side; this
    function carries no dycore knowledge.

    Args:
        physics_state: Current gridpoint state (already projected from the
            dycore via :meth:`DynamicalCore.to_physics_state`).
        forcing: Time-sliced forcing for this step.
        terrain: Boundary conditions.
        physics_state_carry: Cross-step physics carry (the dict returned by
            the previous step's :meth:`Physics.compute_tendencies`).
        physics: The active physics package.
        time_step: Model timestep in seconds. Used by :func:`verify_tendencies`
            to cap negative-going tracer tendencies.

    Returns:
        ``(physics_tendency, new_physics_state_carry)``. The dycore is
        responsible for converting ``physics_tendency`` into its own native
        tendency representation before integrating.

    """
    clamped_physics_state = verify_state(physics_state)
    physics_tendency, new_carry = physics.compute_tendencies(
        clamped_physics_state, forcing, terrain,
        prev_physics_data=physics_state_carry,
    )
    physics_tendency = verify_tendencies(
        clamped_physics_state, physics_tendency, time_step,
    )
    return physics_tendency, new_carry
