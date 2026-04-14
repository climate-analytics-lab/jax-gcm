"""ComposablePhysics: a Physics implementation built from composable terms.

ComposablePhysics holds an ordered list of PhysicsTerm instances and iterates
through them in ``compute_tendencies``, summing tendencies and threading a
``diagnostics`` dict forward. It implements the ``Physics`` interface so that
``Model`` can use it as a drop-in replacement for ``SpeedyPhysics`` or
``IconPhysics``.

See docs/design/composable_physics.md for the full design.

Date: 2026-04-12
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import Physics, PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData
from jcm.physics.physics_term import PhysicsTerm


class ComposablePhysics(nnx.Module, Physics):
    """A physics package built from an ordered list of PhysicsTerm modules.

    Terms are called in order; each receives the diagnostics dict produced by
    all preceding terms. Tendencies are summed.

    When ``vectorize_columns=True``, the 3D state ``(nlev, nlon, nlat)`` is
    reshaped to column format ``(nlev, ncols)`` before iterating terms, and
    accumulated tendencies are reshaped back to 3D afterward. This is the
    standard pattern for column-based physics schemes (ICON, and most
    comprehensive physics packages). SPEEDY operates directly on 3D arrays
    because its low resolution makes the reshape overhead not worthwhile.

    Composition operators (``__add__``, ``replace``, ``remove``) return new
    ``ComposablePhysics`` instances.
    """

    def __init__(
        self,
        terms: list[PhysicsTerm],
        checkpoint_terms: bool = True,
        vectorize_columns: bool = False,
    ):
        """Initialize ComposablePhysics.

        Args:
            terms: Ordered list of PhysicsTerm instances.
            checkpoint_terms: Whether to checkpoint each term for memory
                efficiency during backpropagation (default True).
            vectorize_columns: Whether to reshape state from 3D to column
                format before iterating terms. Use True for column-based
                physics (ICON, etc.), False for grid-based (SPEEDY).

        """
        self.terms = nnx.List(terms)
        self.checkpoint_terms = checkpoint_terms
        self.vectorize_columns = vectorize_columns
        self._validate_ordering()

    # ------------------------------------------------------------------
    # Physics interface
    # ------------------------------------------------------------------

    def cache_coords(self, coords) -> None:
        """Delegate cache_coords to each term."""
        for term in self.terms:
            term.cache_coords(coords)

    def compute_tendencies(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        terrain: TerrainData,
        date: DateData,
        prev_physics_data=None,
    ) -> tuple[PhysicsTendency, dict[str, jnp.ndarray]]:
        """Compute total physics tendencies by iterating over terms.

        Args:
            state: Current atmospheric state.
            forcing: Boundary condition forcing data.
            terrain: Terrain boundary conditions.
            date: Current model date/time info.
            prev_physics_data: Previous step's diagnostics dict for caching
                expensive computations (e.g. radiation sub-stepping).
                None on the first step.

        Returns:
            Summed tendencies and the final diagnostics dict.

        """
        if self.vectorize_columns:
            return self._compute_tendencies_columns(
                state, forcing, terrain, date, prev_physics_data,
            )
        return self._compute_tendencies_3d(
            state, forcing, terrain, date, prev_physics_data,
        )

    def _compute_tendencies_3d(
        self, state, forcing, terrain, date, prev_physics_data=None,
    ):
        """Iterate terms on the full 3D grid (e.g. SPEEDY)."""
        diagnostics: dict[str, jnp.ndarray] = {}
        if prev_physics_data is not None:
            diagnostics = {**prev_physics_data}

        diagnostics["_date"] = date

        tendencies = PhysicsTendency.zeros(state.temperature.shape)

        for term in self.terms:
            call_fn = jax.checkpoint(term) if self.checkpoint_terms else term
            tend, diagnostics = call_fn(state, diagnostics, forcing, terrain)
            tendencies += tend

        return tendencies, diagnostics

    def _compute_tendencies_columns(
        self, state, forcing, terrain, date, prev_physics_data=None,
    ):
        """Column-vectorized term iteration.

        Reshapes state from 3D (nlev, nlon, nlat) to columns (nlev, ncols)
        before iterating terms, then reshapes accumulated tendencies back
        to 3D. This is the standard pattern for column-based physics schemes.
        """
        nlev, nlon, nlat = state.temperature.shape
        ncols = nlat * nlon

        vectorized_state = _reshape_state_to_columns(state, nlev, ncols)

        diagnostics: dict = {}
        if prev_physics_data is not None:
            diagnostics = {**prev_physics_data}

        diagnostics["_date"] = date

        tracer_tends = {
            name: jnp.zeros((nlev, ncols))
            for name in state.tracers
        }
        acc = {
            "u_wind": jnp.zeros((nlev, ncols)),
            "v_wind": jnp.zeros((nlev, ncols)),
            "temperature": jnp.zeros((nlev, ncols)),
            "specific_humidity": jnp.zeros((nlev, ncols)),
            "tracers": tracer_tends,
        }

        for term in self.terms:
            call_fn = (
                jax.checkpoint(term)
                if self.checkpoint_terms
                else term
            )
            tend, diagnostics = call_fn(
                vectorized_state, diagnostics, forcing, terrain,
            )
            acc = _accumulate(acc, tend)

        tendencies = _reshape_tendencies_to_3d(acc, nlev, nlat, nlon)
        return tendencies, diagnostics

    def get_empty_data(self, coords) -> dict[str, jnp.ndarray]:
        """Return an empty diagnostics dict suitable for DiagnosticsCollector.

        This runs compute_tendencies once with zero state to discover the
        diagnostic keys and their shapes, then zeros them out.
        """
        from jax.tree_util import tree_map

        # Build minimal zero state to probe diagnostic shapes
        nodal_shape = coords.horizontal.nodal_shape
        nlev = coords.nodal_shape[0]
        shape_3d = (nlev,) + nodal_shape

        zero_state = PhysicsState.zeros(shape_3d)
        zero_forcing = ForcingData.zeros(nodal_shape)
        zero_terrain = TerrainData.aquaplanet(coords)
        zero_date = DateData.zeros()

        _, diagnostics = self.compute_tendencies(
            zero_state, zero_forcing, zero_terrain, zero_date
        )
        return tree_map(jnp.zeros_like, diagnostics)

    def data_struct_to_dict(
        self, struct: Any, nodal_shape=None, sep: str = "."
    ) -> dict[str, Any]:
        """Convert diagnostics to a flat dict for xarray output.

        Since ComposablePhysics already uses a dict, this is mostly a
        pass-through, filtering out internal keys (prefixed with ``_``)
        and handling multi-channel fields.
        """
        if struct is None:
            return {}
        if not isinstance(struct, dict):
            return super().data_struct_to_dict(struct, nodal_shape, sep)

        items = {k: v for k, v in struct.items() if not k.startswith("_")}

        # Expand multi-channel fields (trailing dim beyond nodal_shape)
        if nodal_shape is not None:
            original_keys = list(items.keys())
            for k in original_keys:
                v = items[k]
                if not isinstance(v, jax.Array):
                    continue
                s = v.shape
                if (
                    len(s) == 5
                    and s[1:-1] == nodal_shape
                    or len(s) == 4
                    and s[1:-1] == nodal_shape[1:]
                ):
                    items.update(
                        {f"{k}{sep}{i}": v[..., i] for i in range(s[-1])}
                    )
                    del items[k]

        return items

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __add__(self, other: ComposablePhysics | PhysicsTerm) -> ComposablePhysics:
        """Concatenate term lists from two physics objects."""
        if hasattr(other, 'terms'):
            other_terms = list(other.terms)
        elif hasattr(other, 'category') and callable(other):
            other_terms = [other]
        else:
            return NotImplemented
        return ComposablePhysics(
            terms=list(self.terms) + other_terms,
            checkpoint_terms=self.checkpoint_terms,
            vectorize_columns=self.vectorize_columns,
        )

    def __radd__(self, other):
        """Support sum() by handling 0 + ComposablePhysics."""
        if other == 0:
            return self
        return NotImplemented

    def replace(self, category: str, new_term: PhysicsTerm) -> ComposablePhysics:
        """Replace all terms of a given category with a single new term.

        The new term is inserted at the position of the first replaced term.
        """
        new_terms = []
        inserted = False
        for t in self.terms:
            if t.category == category:
                if not inserted:
                    new_terms.append(new_term)
                    inserted = True
                # skip original term
            else:
                new_terms.append(t)
        if not inserted:
            raise ValueError(
                f"No term with category {category!r} found. "
                f"Available categories: {[t.category for t in self.terms]}"
            )
        return ComposablePhysics(
            terms=new_terms,
            checkpoint_terms=self.checkpoint_terms,
            vectorize_columns=self.vectorize_columns,
        )

    def remove(self, category: str) -> ComposablePhysics:
        """Remove all terms of a given category."""
        return ComposablePhysics(
            terms=[t for t in self.terms if t.category != category],
            checkpoint_terms=self.checkpoint_terms,
            vectorize_columns=self.vectorize_columns,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_ordering(self) -> None:
        """Check that each term's ``requires`` are satisfied by upstream ``provides``.

        Raises ValueError if a term requires a diagnostic key that no
        upstream term provides.
        """
        available: set[str] = set()
        for term in self.terms:
            missing = set(term.requires) - available
            if missing:
                raise ValueError(
                    f"Term {term.name!r} requires diagnostics {missing} "
                    f"but no upstream term provides them. "
                    f"Available at this point: {available}"
                )
            available.update(term.provides)


# ------------------------------------------------------------------
# Column vectorization helpers
# ------------------------------------------------------------------

def _reshape_state_to_columns(state, nlev, ncols):
    """Reshape PhysicsState fields from 3D (nlev, nlon, nlat) to columns (nlev, ncols)."""
    from jcm.physics_interface import PhysicsState as PS

    def reshape_field(field):
        if field.ndim == 3:
            return field.reshape(nlev, ncols)
        elif field.ndim == 2:
            return field.reshape(ncols)
        return field

    reshaped = jax.tree_util.tree_map(reshape_field, {
        "u_wind": state.u_wind,
        "v_wind": state.v_wind,
        "temperature": state.temperature,
        "specific_humidity": state.specific_humidity,
        "geopotential": state.geopotential,
        "normalized_surface_pressure": state.normalized_surface_pressure,
    })
    tracers = {
        name: tracer.reshape(nlev, ncols)
        for name, tracer in state.tracers.items()
    }
    return PS(**reshaped, tracers=tracers)


def _accumulate(acc, tend):
    """Accumulate column-format tendencies."""
    return {
        "u_wind": acc["u_wind"] + tend.u_wind,
        "v_wind": acc["v_wind"] + tend.v_wind,
        "temperature": acc["temperature"] + tend.temperature,
        "specific_humidity": (
            acc["specific_humidity"] + tend.specific_humidity
        ),
        "tracers": {
            name: acc["tracers"][name] + tend.tracers.get(name, 0.0)
            for name in acc["tracers"]
        },
    }


def _reshape_tendencies_to_3d(tendencies, nlev, nlat, nlon):
    """Reshape column tendencies back to 3D (nlev, nlon, nlat)."""
    from jcm.physics_interface import PhysicsTendency as PT

    def reshape_to_3d(field):
        if field.ndim == 2:
            return field.reshape(nlev, nlon, nlat)
        return field

    return PT(
        u_wind=reshape_to_3d(tendencies["u_wind"]),
        v_wind=reshape_to_3d(tendencies["v_wind"]),
        temperature=reshape_to_3d(tendencies["temperature"]),
        specific_humidity=reshape_to_3d(
            tendencies["specific_humidity"],
        ),
        tracers={
            name: field.reshape(nlev, nlon, nlat)
            for name, field in tendencies["tracers"].items()
        },
    )
