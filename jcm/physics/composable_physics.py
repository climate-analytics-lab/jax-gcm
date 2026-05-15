"""ComposablePhysics: a Physics implementation built from composable terms.

ComposablePhysics holds an ordered list of PhysicsTerm instances and iterates
through them in ``compute_tendencies``, summing tendencies and threading a
``diagnostics`` dict forward. It implements the ``Physics`` interface that
``Model`` consumes.

The threaded ``diagnostics`` dict serves a dual role:

- Keys without a leading underscore (e.g. ``"cloud_fraction"``,
  ``"sw_heating_rate"``) are user-facing diagnostic outputs. They appear in
  the xarray Dataset returned by ``Model.run().to_xarray()``.
- Keys prefixed with ``_`` (e.g. ``"_radiation"``, ``"_chemistry"``) are
  internal inter-term state — typed PhysicsData sub-structs that downstream
  terms read but the user never sees. ``data_struct_to_dict`` filters them
  out of the user-facing output.

See docs/design/composable_physics.md for the full design.
"""

from __future__ import annotations

from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import Physics, PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics.radiation.band_config import RadiationBandConfig


class ComposablePhysics(nnx.Module, Physics):
    """A physics package built from an ordered list of PhysicsTerm modules.

    Terms are called in order; each receives the diagnostics dict produced by
    all preceding terms (see module docstring for the dict's dual role) and
    returns a (tendency, updated_diagnostics) pair. Tendencies are summed.

    When ``vectorize_columns=True``, the 3D state ``(nlev, nlon, nlat)`` is
    reshaped to column format ``(nlev, ncols)`` before iterating terms, and
    accumulated tendencies are reshaped back to 3D afterward. This is the
    standard pattern for column-based physics schemes (ECHAM, and most
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
        dt_seconds: float = 1800.0,
        band_config: RadiationBandConfig | None = None,
    ):
        """Initialize ComposablePhysics.

        Args:
            terms: Ordered list of PhysicsTerm instances.
            checkpoint_terms: Whether to checkpoint each term for memory
                efficiency during backpropagation (default True).
            vectorize_columns: Whether to reshape state from 3D to column
                format before iterating terms. Use True for column-based
                physics (ECHAM, etc.), False for grid-based (SPEEDY).
            dt_seconds: Physics timestep in seconds. Injected into the
                diagnostics dict as ``"_dt_seconds"`` at the top of every
                ``compute_tendencies`` call so terms that integrate by
                ``dt`` (chemistry, microphysics, vertical diffusion, …)
                read a single source of truth without needing model-wide
                date plumbing. ``Model.__init__`` overrides this to its
                own ``dt_si`` after the physics object is constructed.
            band_config: Shared radiation band centers (LW + SW). Owned
                here so wavelength-dependent terms (MACv2-SP aerosols,
                future banded chemistry) read the *active* radiation
                backend's bands instead of hardcoding constants. Injected
                into ``diagnostics["_band_config"]`` each step. Defaults
                to a single 550 nm SW band — runners override this via
                ``RadiationBandConfig.from_rrtmgp(...)`` when RRTMGP is
                in use, so terms see the real 14-band SW structure.

        """
        self.terms = nnx.List(terms)
        self.checkpoint_terms = checkpoint_terms
        self.vectorize_columns = vectorize_columns
        self.dt_seconds = float(dt_seconds)
        self.band_config = (
            band_config if band_config is not None
            else RadiationBandConfig.broadband()
        )
        self._validate_ordering()

    # ------------------------------------------------------------------
    # Physics interface
    # ------------------------------------------------------------------

    def cache_coords(self, coords) -> None:
        """Delegate cache_coords (and band_config) to each term.

        Also pushes ``self.band_config`` into every term that opts in via
        :meth:`PhysicsTerm.cache_band_config`. Done here (not in
        ``initial_carry_state``) because terms with band-shaped carry
        slots — e.g. MACv2-SP aerosol's per-SW-band optics — need the
        band count *before* their ``initial_carry_state`` runs.
        """
        for term in self.terms:
            term.cache_coords(coords)
            term.cache_band_config(self.band_config)

    def required_tracers(self) -> tuple[TracerSpec, ...]:
        """Union of TracerSpecs declared by every term.

        Raises ``ValueError`` if two terms declare the same tracer name with
        different specs — ambiguity should be resolved at composition time,
        not silently.
        """
        seen: dict[str, TracerSpec] = {}
        for term in self.terms:
            for spec in term.required_tracers():
                if spec.name in seen and seen[spec.name] != spec:
                    raise ValueError(
                        f"Conflicting TracerSpec for {spec.name!r}: "
                        f"{seen[spec.name]} vs {spec}"
                    )
                seen[spec.name] = spec
        return tuple(seen.values())

    def compute_tendencies(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        terrain: TerrainData,
        prev_physics_data=None,
    ) -> tuple[PhysicsTendency, dict[str, jnp.ndarray]]:
        """Compute total physics tendencies by iterating over terms.

        Args:
            state: Current atmospheric state.
            forcing: Boundary condition forcing data.
            terrain: Terrain boundary conditions.
            prev_physics_data: Previous step's diagnostics dict for caching
                expensive computations (e.g. radiation sub-stepping).
                None on the first step.

        Returns:
            Summed tendencies and the final diagnostics dict.

        """
        if self.vectorize_columns:
            tendencies, diagnostics = self._compute_tendencies_columns(
                state, forcing, terrain, prev_physics_data,
            )
        else:
            tendencies, diagnostics = self._compute_tendencies_3d(
                state, forcing, terrain, prev_physics_data,
            )
        # Strip pure-plumbing keys (timestep snapshot, sliced forcing,
        # parameter snapshot) before returning. These are re-injected at
        # the top of the next ``compute_tendencies`` call from
        # authoritative sources, so they don't need to ride in the saved
        # trajectory and would otherwise bloat the prediction dict and
        # break tree_map averaging tests against the legacy
        # ``PhysicsData``-shaped output.
        diagnostics = {
            k: v for k, v in diagnostics.items()
            if k not in self._INTERNAL_DIAGNOSTIC_KEYS
        }
        return tendencies, diagnostics

    def _compute_tendencies_3d(
        self, state, forcing, terrain, prev_physics_data=None,
    ):
        """Iterate terms on the full 3D grid (e.g. SPEEDY)."""
        diagnostics: dict[str, jnp.ndarray] = {}
        if prev_physics_data is not None:
            diagnostics = {**prev_physics_data}

        diagnostics["_dt_seconds"] = self.dt_seconds
        diagnostics["_band_config"] = self.band_config

        tendencies = PhysicsTendency.zeros(state.temperature.shape)

        for term in self.terms:
            call_fn = jax.checkpoint(term) if self.checkpoint_terms else term
            tend, diagnostics = call_fn(state, diagnostics, forcing, terrain)
            tendencies += tend

        return tendencies, diagnostics

    def _compute_tendencies_columns(
        self, state, forcing, terrain, prev_physics_data=None,
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

        diagnostics["_dt_seconds"] = self.dt_seconds
        diagnostics["_band_config"] = self.band_config

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
        """Return a zero-filled template of the per-step diagnostics dict.

        Internal helper used by ``Model._build_initial_physics_carry``
        and ``Model._get_op_split_integrate_fn`` to discover the
        pytree structure of ``compute_tendencies``' output dict. The
        ``lax.scan`` carry needs to be that exact structure on
        iteration 1; ``initial_carry_state`` alone only covers the
        cross-step slots, so we union it with this structural template
        (zero values, used only for shape).

        Implementation: runs ``compute_tendencies`` once at Model
        construction time with a non-zero isothermal probe state
        (288 K, q=0, etc.) so radiation terms don't hit 0/0=NaN.
        The result is *only ever used as a zero-filled template* —
        never as live cross-step physics state. (Mis-using its
        output as live state was the architectural bug `#470
        <https://github.com/climate-analytics-lab/jax-gcm/issues/470>`_
        tracks; the operator-split refactor in `#471` moved live
        state to ``initial_carry_state``.)

        """
        from jax.tree_util import tree_map

        # Probe at a well-conditioned isothermal state. Using
        # ``PhysicsState.zeros`` here produces 0/0 = NaN in radiation
        # terms; the non-zero T=288 K probe walks the same code paths
        # without the division-by-zero. We zero the output anyway
        # so the probe values themselves don't matter — only the
        # pytree structure / shapes do.
        nodal_shape = coords.horizontal.nodal_shape
        nlev = coords.nodal_shape[0]
        shape_3d = (nlev,) + nodal_shape

        probe_state = PhysicsState.zeros(shape_3d).copy(
            temperature=jnp.full(shape_3d, 288.0),
            normalized_surface_pressure=jnp.ones(nodal_shape),
        )
        probe_forcing = ForcingData.zeros(nodal_shape)
        probe_terrain = TerrainData.aquaplanet(coords)

        _, diagnostics = self.compute_tendencies(
            probe_state, probe_forcing, probe_terrain,
        )
        return tree_map(jnp.zeros_like, diagnostics)

    def initial_carry_state(self, coords) -> dict[str, jnp.ndarray]:
        """Aggregate per-term cross-step carry-state slots.

        Iterates over the term list and merges each term's
        :meth:`PhysicsTerm.initial_carry_state` output into a single
        dict. Terms that don't override get ``{}`` and contribute
        nothing. Conflicting keys (same key written by multiple terms)
        raise ``ValueError`` — keys should be namespaced per-term.

        This is the deterministic replacement for :meth:`get_empty_data`
        in the operator-split path. It runs once at ``Model.__init__``
        time and the result becomes the initial value of the
        ``physics_state`` carry threaded through the scan.

        Args:
            coords: model :class:`dinosaur.coordinate_systems.CoordinateSystem`.

        Returns:
            A dict ``{"<key>": <slot>}`` containing every term's
            cross-step carry slots, ready to be passed as
            ``prev_physics_data`` to :meth:`compute_tendencies` on the
            first step.

        """
        carry: dict[str, jnp.ndarray] = {}
        for term in self.terms:
            slot = term.initial_carry_state(coords)
            if not slot:
                continue
            overlap = carry.keys() & slot.keys()
            if overlap:
                raise ValueError(
                    f"Term {term.name!r} initial_carry_state collides with "
                    f"an upstream term on keys {sorted(overlap)}. "
                    "Namespace per-term keys (e.g. ``_radiation``, ``_clouds``)."
                )
            carry.update(slot)
        return carry

    # Underscore-prefixed keys that are pure plumbing (timestep, sliced
    # forcing snapshots, parameter snapshots) and must NOT be flattened into
    # the user-facing xarray output.
    _INTERNAL_DIAGNOSTIC_KEYS: ClassVar[frozenset[str]] = frozenset({
        "_dt_seconds",
        "_band_config",
        "_forcing_2d",
        "_echam_params",
        "_echam_coords",
        "_speedy_coords",
    })

    # Sub-struct fields that survive the flatten step but should be dropped
    # before reaching xarray. These are typically per-RRTMGP-band optical
    # properties — they're plumbing for the radiation coupling, have a
    # 14-element ``sw_band`` axis that ``data_to_xarray`` has no dim entry
    # for, and balloon save_intervals netCDFs by ~3× without adding
    # scientific value. Filter is applied to the full dotted key
    # (e.g. ``aerosol.aod_sw_per_band``).
    _EXCLUDED_OUTPUT_KEYS: ClassVar[frozenset[str]] = frozenset({
        "aerosol.aod_sw_per_band",
        "aerosol.ssa_sw_per_band",
        "aerosol.asy_sw_per_band",
    })

    def data_struct_to_dict(
        self, struct: Any, nodal_shape=None, sep: str = "."
    ) -> dict[str, Any]:
        """Convert diagnostics to a flat dict for xarray output.

        The threaded diagnostics dict mixes three kinds of values:

        - Top-level array diagnostics (no leading underscore) — kept as-is.
        - Typed sub-structs of arrays stashed under ``_<name>`` for inter-term
          communication (``_radiation``, ``_humidity``, ...) — flattened into
          ``<name>.<field>`` user-facing keys (matches the legacy SPEEDY /
          ECHAM ``PhysicsData`` xarray layout).
        - Infrastructure objects (``_dt_seconds``, ``_echam_params``, ...)
          that are listed in :attr:`_INTERNAL_DIAGNOSTIC_KEYS` or that fail
          array-only flattening — silently dropped from user output.
        """
        if struct is None:
            return {}
        if not isinstance(struct, dict):
            return super().data_struct_to_dict(struct, nodal_shape, sep)

        items: dict[str, Any] = {}
        for k, v in struct.items():
            if k in self._INTERNAL_DIAGNOSTIC_KEYS:
                continue
            out_key = k.lstrip("_") if k.startswith("_") else k
            if not out_key:
                continue
            if isinstance(v, jax.Array):
                items[out_key] = v
            elif hasattr(v, "__dict__") and v.__dict__:
                # Typed sub-struct (e.g. PhysicsData.radiation). Flatten via
                # the parent recursive helper; skip if it raises (sub-structs
                # that contain non-array fields).
                try:
                    sub = super().data_struct_to_dict(v, nodal_shape, sep)
                except (ValueError, AttributeError):
                    continue
                for sk, sv in sub.items():
                    full_key = f"{out_key}{sep}{sk}"
                    if full_key in self._EXCLUDED_OUTPUT_KEYS:
                        continue
                    items[full_key] = sv

        # Reshape column-vectorized diagnostics (a flattened ncols axis,
        # produced when ``vectorize_columns=True``) back to ``(lon, lat)``
        # before xarray sees them. The ``nodal_shape`` argument from
        # ``Predictions.to_xarray`` may include leading non-spatial axes
        # (level, time, ...); the spatial portion is the trailing pair.
        if nodal_shape is not None and len(nodal_shape) >= 2:
            spatial_shape = tuple(nodal_shape[-2:])
            ncols = spatial_shape[0] * spatial_shape[1]
            for k in list(items.keys()):
                v = items[k]
                if not isinstance(v, jax.Array):
                    continue
                if ncols not in v.shape:
                    continue
                # Find the (single) ncols axis and unflatten it.
                ncols_axes = [i for i, n in enumerate(v.shape) if n == ncols]
                if len(ncols_axes) != 1:
                    continue
                axis = ncols_axes[0]
                new_shape = (v.shape[:axis] + spatial_shape
                             + v.shape[axis + 1:])
                items[k] = v.reshape(new_shape)

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
                    or len(s) == 3
                    and s[:-1] == nodal_shape
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
            dt_seconds=self.dt_seconds,
            band_config=self.band_config,
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
            dt_seconds=self.dt_seconds,
            band_config=self.band_config,
        )

    def remove(self, category: str) -> ComposablePhysics:
        """Remove all terms of a given category."""
        return ComposablePhysics(
            terms=[t for t in self.terms if t.category != category],
            checkpoint_terms=self.checkpoint_terms,
            vectorize_columns=self.vectorize_columns,
            dt_seconds=self.dt_seconds,
            band_config=self.band_config,
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
