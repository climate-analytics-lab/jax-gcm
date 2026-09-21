"""Dycore-agnostic physics interface types and helpers.

This module defines the gridpoint-space data structures that physics packages
consume and produce — :class:`PhysicsState` and :class:`PhysicsTendency` — and
the :class:`Physics` base class that they implement.

It is **dycore-agnostic**: no spectral transforms or ``dinosaur`` symbols
appear here. The actual dycore↔physics conversion is owned by each backend
under ``jcm/dycore/<backend>/state_bridge.py`` (see
:mod:`jcm.dycore.dinosaur.state_bridge` for the canonical example).
"""

import logging
from typing import Any, Dict, Tuple, TypeAlias

import jax
import jax.numpy as jnp
import tree_math
from jax import tree_util

from jcm import constants as physical_constants
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData

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
inputs for the physics parameterizations. All fields are dimensional. In
particular, ``specific_humidity`` has one canonical representation throughout
the public physics API: the dimensionless mass fraction kg/kg. Backends and
legacy schemes with another native convention must convert at their boundary.

Attributes:
    u_wind : jnp.ndarray
        Zonal (east-west) component of wind.
    v_wind : jnp.ndarray
        Meridional (north-south) component of wind.
    temperature : jnp.ndarray
        Atmospheric temperature [K].
    specific_humidity : jnp.ndarray
        Mass of water vapor per unit mass of moist air [kg/kg].
    geopotential : jnp.ndarray
        Gravitational potential energy per unit mass [m2/s2].
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
to update the model state over a time step. Fields use the same dimensional
conventions as :class:`PhysicsState` per second; specific humidity is therefore
kg/kg/s.

Attributes:
    u_wind : jnp.ndarray
        Tendency of the zonal wind component.
    v_wind : jnp.ndarray
        Tendency of the meridional wind component.
    temperature : jnp.ndarray
        Tendency of temperature.
    specific_humidity : jnp.ndarray
        Tendency of specific humidity [kg/kg/s].
"""


class Physics:
    UNITS_TABLE_CSV_PATH = None
    cached_coords = None

    def units_table_paths(self) -> tuple:
        """Units/description CSVs to attach to ``to_xarray`` output.

        A container physics gathers the tables of the terms it holds, so
        each term ships the metadata for the diagnostics it publishes.
        """
        if self.UNITS_TABLE_CSV_PATH is None:
            return ()
        return (self.UNITS_TABLE_CSV_PATH,)

    def cache_coords(self, coords):
        return None

    def required_tracers(self):
        """Return a tuple of TracerSpec objects this physics needs in state.tracers.

        Default is empty — only ``specific_humidity`` is assumed. Composable
        physics packages override this to aggregate declarations from terms.
        """
        return ()

    def prognostic_carry_slots(self):
        """Carry keys holding prognostic state rather than diagnostics.

        Default is empty — a package whose whole cross-step carry is
        recomputed each step declares nothing. ``ComposablePhysics``
        aggregates the per-term declarations. A checkpoint restore refuses
        to seed or drop these when migrating a changed carry field set
        (``docs/source/design/checkpoint_compatibility.md``).
        """
        return ()

    def stable_time_step_minutes(self, coords) -> float | None:
        """Largest numerically-stable model time step (minutes), or ``None``.

        ``Model`` consults this when the user does not pass ``time_step`` and
        the Model is building its own dycore, so grid-dependent explicit-
        tendency stability limits (e.g. SPEEDY's surface drag in a thin
        bottom sigma layer) yield a stable default automatically. ``None``
        means "no physics-imposed limit" (the historical 30-minute default
        applies). ``ComposablePhysics`` aggregates per-term limits.
        """
        return None

    def required_dycore_fields(self):
        """Names of dycore-supplied fields this physics needs each step.

        See :meth:`jcm.dycore.base.DynamicalCore.physics_fields`.
        Default is empty; ``ComposablePhysics`` aggregates per-term
        declarations (``PhysicsTerm.requires_dycore_fields``).
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

    def _finalize_tendency_verification(
        self,
        state: PhysicsState,
        raw_tendencies: PhysicsTendency,
        applied_tendencies: PhysicsTendency,
        water_corrections: dict[str, jnp.ndarray],
        physics_data: Any,
    ) -> Any:
        """Finalize carry data after the interface positivity verification.

        Most physics implementations have no post-verification carry work.
        Containers can override this protected hook when diagnostics must
        describe the tendency that the host actually integrates rather than
        the raw term sum.
        """
        del state, raw_tendencies, applied_tendencies, water_corrections
        return physics_data

    def initial_carry_state(self, coords) -> PhysicsCarryState:
        """Build the cross-step physics carry at ``Model`` construction time.

        Default returns ``{}``. ``ComposablePhysics`` aggregates per-term
        slots; raw subclasses can return whatever ``compute_tendencies``
        expects as ``prev_physics_data``.
        """
        return {}

    def get_empty_data(self, coords) -> Any:
        """Return a zero-filled diagnostics structure.

        ``Model`` uses this as the structural template for the cross-step
        physics carry and as the running-mean accumulator seed when
        ``output_averages=True``. Implementations should return the same
        pytree structure that ``compute_tendencies`` returns as its updated
        physics data.
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

# Water mass fields whose positivity correction contributes to the water
# budget. qnc/qni are number concentrations and the VMR fields are gases, so
# their positivity caps must not be folded into a water-mass diagnostic.
_WATER_MASS_TRACERS = frozenset({"qc", "qi", "qr", "qs"})

# Water-mass fields whose positivity cap is made column-conservative when the
# layer masses are available (#806). A bare per-cell cap raises the negative
# half of a conservative vertical redistribution (vertical diffusion mixes
# ``q``/``qc``/``qi``) while the receiving layers keep the gain, creating
# column water. For these fields the cap's spurious column-integrated source is
# removed again, spread over the layers that still hold water after the cap, so
# the column water path is conserved to round-off without ever driving a layer
# below zero. ``qnc``/``qni`` (numbers) and the VMR gases are excluded: they
# are not water mass and their redistribution conservation is a separate
# concern, so they keep the bare cap.
_WATER_CONSERVED_FIELDS = frozenset({"specific_humidity"}) | _WATER_MASS_TRACERS

# Deliberately just the membership test above: JAM aerosol and gas tracers
# are NOT capped. Their tendency sums conservative redistributions (tracer
# vertical diffusion, convective transport) with paired transfers (sulfur
# chemistry, activation exchange), and clipping one side of a conserved
# pair creates mass. Their removal is bounded where it is produced, by the
# operator split in ``aerosol/jam/removal_split.py``. Do not re-add a name
# family here without re-reading that argument (and #806, the same defect
# in the retained water fields).


def has_non_negative_tendency(name: str) -> bool:
    """Whether a tracer's tendency must not drive it below zero."""
    return name in _NON_NEGATIVE_TRACERS


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
    volume mixing ratios) to ``>= 0``. Aerosol and gas tracers are
    deliberately left alone, here and on the tendency side (see
    :func:`has_non_negative_tendency`). We deliberately do NOT clip to an
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


def _conserve_water_column(value, capped, raw, pressure_thickness, time_step):
    """Reallocate the positivity cap's column water source within the column.

    ``capped - raw`` is the mixing-ratio rate the per-cell cap ADDED to keep a
    layer non-negative. Pressure-weighted and summed over the column it is a
    spurious water source (#806): where a conservative vertical-diffusion
    redistribution overdrew a donor layer, the cap invents the water the
    receiving layers already hold. This removes exactly that column-integrated
    source again, distributed over the water left after the cap (proportional
    to each layer's remaining mass — the standard hole-filling choice, and the
    only defensible one, since the individual receiving layers are not
    identifiable from the summed tendency the interface sees). Each layer is
    scaled toward — never below — zero, so the cap's non-negativity is
    preserved. A column that cannot supply the whole deficit is drained to zero
    and the bounded residual stays in the water-positivity ledger.

    Vertical is axis 0 (broadcasting-native); the reduction is per column.
    ``pressure_thickness`` is |Δp| (Pa); the 1/g mass factor cancels in the
    ratio, so the bare Δp is the correct weight.
    """
    qpos = jnp.maximum(value, 0.0)
    added = capped - raw                        # >= 0 mixing-ratio source rate
    source = jnp.sum(added * pressure_thickness, axis=0, keepdims=True)
    qnext = qpos + time_step * capped           # >= 0 post-cap next-step value
    removable = jnp.sum(qnext * pressure_thickness, axis=0, keepdims=True)
    # Safe denominator so a dry column (removable == 0) gives frac == 0 with no
    # 0/0 in the reverse-mode graph (#558/#559 poison-free guard).
    safe_removable = jnp.where(removable > 0.0, removable, 1.0)
    # Fraction of each layer's post-cap water to remove — uniform across the
    # column (proportional to mass) and bounded to [0, 1].
    frac = jnp.where(
        removable > 0.0,
        jnp.clip(time_step * source / safe_removable, 0.0, 1.0),
        0.0,
    )
    return capped - frac * qnext / time_step


def _verify_tendencies_with_water_corrections(
    state: PhysicsState,
    tendencies: PhysicsTendency,
    time_step,
    pressure_thickness: jnp.ndarray | None = None,
) -> tuple[PhysicsTendency, dict[str, jnp.ndarray]]:
    """Return applied tendencies and stop-gradient water corrections.

    ``pressure_thickness`` — the layer masses |Δp| in the same layout and with
    the same level axis (0) as the tendencies — enables the column-conservative
    reallocation for the water-mass fields (#806). Without it those fields fall
    back to the bare per-cell positivity cap (a host that does not expose its
    vertical geometry, or the standalone :func:`verify_tendencies` entry
    point).
    """

    def _positivity(value, tend, conserve):
        # Per-cell non-negativity cap: floor the tendency at the drain rate
        # that empties the tracer and no further. ``max(tend, -max(value,0)/dt)``
        # equals the plain ``-value/dt`` cap wherever ``value >= 0``, leaves any
        # source untouched, and on a tracer that arrives negative (aerosol is
        # not entry-clipped) stops the sink rather than inventing mass.
        capped = jnp.maximum(tend, -jnp.maximum(value, 0.0) / time_step)
        if conserve and pressure_thickness is not None:
            result = _conserve_water_column(
                value, capped, tend, pressure_thickness, time_step,
            )
        else:
            result = capped
        # Straight-through estimator (maintainability review B.1 cross-
        # cutting): the primal keeps the hard cap (plus the conservative
        # reallocation), but the cotangent passes through to the producing
        # tendency unchanged. The bare where() rerouted gradients from the
        # physics that produced the tendency onto the STATE whenever it fired
        # — and it fires routinely wherever precip/evaporation drives q toward
        # 0, silently detaching those cells from any parameter being
        # calibrated. The exact-primal form ``stop_grad(result) + (tend -
        # stop_grad(tend))`` is bitwise ``result`` in the forward pass (the
        # tend terms cancel exactly), unlike ``tend + stop_grad(result - tend)``
        # whose re-association can undershoot the drain by an ulp and produce
        # q < 0.
        return jax.lax.stop_gradient(result) + (
            tend - jax.lax.stop_gradient(tend)
        )

    clipped_dqdt = _positivity(
        state.specific_humidity, tendencies.specific_humidity, conserve=True,
    )
    clipped_tracer_tends = {
        name: (
            _positivity(
                state.tracers[name], tend,
                conserve=name in _WATER_CONSERVED_FIELDS,
            )
            if has_non_negative_tendency(name) and name in state.tracers
            else tend
        )
        for name, tend in tendencies.tracers.items()
    }
    applied = tendencies.copy(
        specific_humidity=clipped_dqdt,
        tracers=clipped_tracer_tends,
    )

    # Net (post-reallocation) positivity source per water field. Detaching this
    # diagnostic is intentional: it records the residual artificial source in
    # the primal water budget without opening a second optimization path around
    # the straight-through estimator. With Δp available and the column able to
    # supply the deficit this is ~0 to round-off; the bare cap (no Δp) or a
    # fully-drained column leaves the same gross correction #824 recorded.
    corrections = {
        "specific_humidity": jax.lax.stop_gradient(
            applied.specific_humidity - tendencies.specific_humidity
        ),
    }
    corrections.update({
        name: jax.lax.stop_gradient(
            applied.tracers[name] - tendencies.tracers[name]
        )
        for name in tendencies.tracers
        if name in _WATER_MASS_TRACERS and name in applied.tracers
    })
    return applied, corrections


def verify_tendencies(state: PhysicsState, tendencies: PhysicsTendency, time_step) -> PhysicsTendency:
    """Adjust tendencies to prevent the state from becoming physically invalid in the next time step.

    For every positive-definite scalar (``specific_humidity`` plus every
    tracer :func:`has_non_negative_tendency` accepts) we cap the negative
    part of the tendency at ``-state / dt``, i.e. just enough to drive the
    field to zero rather than below. This mirrors what an implicit step
    on a linear sink would do for the same field.

    The cap is only sound for a field whose tendency is a pure sink plus
    sources: clipping the donor half of a conservative redistribution while
    its receivers keep their gain CREATES mass. Aerosol and gas tracers are
    excluded for that reason. The retained water fields do not strictly
    satisfy it either — vertical diffusion redistributes q/qc/qi — so through
    the gridpoint driver (:func:`compute_physics_step_gridpoint`), where the
    layer masses Δp are available, the water-mass fields' cap is made
    column-conservative: the spurious column-integrated source is removed
    again from the water left after the cap, so column water is conserved to
    round-off while every layer stays >= 0 (#806). This standalone entry point
    has no Δp and therefore applies only the bare per-cell cap; it is used for
    unit tests and hosts that do not expose their vertical geometry.

    Args:
        state: The current ``PhysicsState`` (already passed through
            ``verify_state``).
        tendencies: The physics tendencies.
        time_step: The model time step in seconds.

    Returns:
        The verified ``PhysicsTendency``.

    """
    applied, _ = _verify_tendencies_with_water_corrections(
        state, tendencies, time_step,
    )
    return applied


def _record_water_positivity_corrections(
    diagnostics: dict,
    raw_tendencies: PhysicsTendency,
    applied_tendencies: PhysicsTendency,
    corrections: dict[str, jnp.ndarray],
) -> dict:
    """Attach applied-tendency water accounting to composable diagnostics."""
    prev_step = diagnostics.get("_prev_step")
    reference = (
        prev_step.get("q_tendency")
        if isinstance(prev_step, dict)
        else raw_tendencies.specific_humidity
    )

    # Column-vectorized physics returns grid-shaped tendencies but keeps its
    # diagnostic carry in (level, column) layout. Reshaping to the existing
    # _prev_step reference preserves that package-native layout and therefore
    # the fixed pytree shapes required by lax.scan.
    def _diagnostic_layout(value):
        return jnp.reshape(value, reference.shape)

    correction_diagnostics = {
        f"{name}_tendency": _diagnostic_layout(value)
        for name, value in corrections.items()
    }
    total = sum(
        correction_diagnostics.values(),
        jnp.zeros_like(_diagnostic_layout(raw_tendencies.specific_humidity)),
    )
    correction_diagnostics["total_water_tendency"] = total

    pressure_thickness = diagnostics.get("pressure_thickness")
    if pressure_thickness is not None:
        total_for_pressure = jnp.reshape(total, pressure_thickness.shape)
        correction_diagnostics["column_water_source"] = jnp.sum(
            total_for_pressure * pressure_thickness / physical_constants.grav,
            axis=0,
        )

    updated = {
        **diagnostics,
        "water_positivity_correction": correction_diagnostics,
    }
    if isinstance(prev_step, dict):
        updated["_prev_step"] = {
            **prev_step,
            "q_tendency": _diagnostic_layout(
                applied_tendencies.specific_humidity
            ),
        }
    return updated


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
        ``(physics_tendency, new_physics_state_carry)``. The tendency is the
        verified value the host must integrate. Composable carries include
        per-field water positivity corrections and, when pressure thickness
        is available, their pressure-weighted column source. The dycore is
        responsible for converting ``physics_tendency`` into its own native
        tendency representation before integrating.

    """
    clamped_physics_state = verify_state(physics_state)
    raw_physics_tendency, new_carry = physics.compute_tendencies(
        clamped_physics_state, forcing, terrain,
        prev_physics_data=physics_state_carry,
    )
    # Layer masses Δp for the column-conservative water-positivity limiter
    # (#806). ``None`` for a physics package that does not expose its vertical
    # geometry (or a non-column host); the limiter then falls back to the bare
    # per-cell cap.
    thickness_fn = getattr(physics, "pressure_thickness", None)
    pressure_thickness = thickness_fn(clamped_physics_state) if callable(
        thickness_fn
    ) else None
    physics_tendency, corrections = _verify_tendencies_with_water_corrections(
        clamped_physics_state, raw_physics_tendency, time_step,
        pressure_thickness=pressure_thickness,
    )
    finalize = getattr(physics, "_finalize_tendency_verification", None)
    if callable(finalize):
        new_carry = finalize(
            clamped_physics_state,
            raw_physics_tendency,
            physics_tendency,
            corrections,
            new_carry,
        )
    return physics_tendency, new_carry
