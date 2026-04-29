"""Single-column physics driver.

``SingleColumnModel`` evolves prognostic tracers (cloud water/ice, aerosols,
chemistry species, etc.) with ``lax.scan`` while large-scale atmospheric
state is supplied externally as a time series for *one column at one
location*. The dynamical core does not run; physics tendencies decide
state evolution.

The user supplies a vertical coordinate (``SigmaCoordinates`` or
``HybridCoordinates``) and a single ``(lat_deg, lon_deg)`` location, plus a
single-column ``TerrainData`` and ``ForcingData``. Internally the SCM
builds a duck-typed ``(1, 1)`` coords stub so column-based physics can
cache its coord-dependent transforms (lat, vertical-level transforms,
etc.) without dragging in a full horizontal grid.

Multiple columns at unrelated locations should be run in parallel one
layer above the SCM (e.g. ``jax.vmap`` over a list of
``(lat, lon, column_state)`` triples).

The companion ``PrescribedStateModel`` (in ``jcm.prescribed_state_model``)
is the multi-column equivalent: it accepts a full-grid prescribed state
and computes tendencies for every cell with ``vmap``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
from jax import lax
from jax.tree_util import tree_map

from jcm.date import DateData
from jcm.forcing import ForcingData
from jcm.physics_interface import (
    Physics,
    PhysicsState,
    PhysicsTendency,
    verify_state,
)
from jcm.terrain import TerrainData


@tree_math.struct
class SCMPredictions:
    """Container for ``SingleColumnModel.run`` outputs.

    All array fields are 1-D in the level dimension with a leading time
    axis: ``(n_times, nlev)`` for column profiles, ``(n_times,)`` for
    surface scalars.

    Attributes:
        prescribed_states: Time series of input column states (1-D).
        tracer_states: Time series of evolved tracers (dict of 1-D arrays).
        relaxed_states: Time series of relaxed prognostic variables (dict;
            empty when no relaxation is configured).
        tendencies: Physics tendencies at each step (1-D).
        physics_data: Per-step diagnostics dict from the physics package.
        times: Times in days since ``start_date``.

    """

    prescribed_states: PhysicsState
    tracer_states: dict
    relaxed_states: dict
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any


def _vertical_nlev(vertical) -> int:
    if hasattr(vertical, "centers"):
        return int(np.asarray(vertical.centers).shape[0])
    if hasattr(vertical, "a_boundaries"):
        return int(np.asarray(vertical.a_boundaries).shape[0]) - 1
    raise TypeError(
        f"Unsupported vertical coordinate type {type(vertical).__name__!r}; "
        "expected SigmaCoordinates or HybridCoordinates."
    )


def _make_single_column_coords(vertical, lat_deg: float, lon_deg: float):
    """Duck-typed ``CoordinateSystem`` analogue at the user's column.

    The SCM's physics packages only read ``coords.vertical``,
    ``coords.horizontal.{latitudes, longitudes, nodal_shape}`` and
    ``coords.nodal_shape`` from whatever they're handed, so a
    ``SimpleNamespace`` with those attributes is enough — no real
    horizontal grid needed.

    The horizontal shape is ``(1, 1)``: a single column at the requested
    ``(lat_deg, lon_deg)``. ICON's term setup (e.g.
    ``IconTermBase.cache_coords``) assumes a 3-tuple ``(nlev, nlon, nlat)``
    nodal shape, so we keep that convention rather than collapsing to
    ``(nlev, 1)``.
    """
    nlev = _vertical_nlev(vertical)
    lat_rad = jnp.asarray([float(np.deg2rad(lat_deg))])
    lon_rad = jnp.asarray([float(np.deg2rad(lon_deg))])
    horizontal = SimpleNamespace(
        nodal_shape=(1, 1),
        latitudes=lat_rad,
        longitudes=lon_rad,
        # ``nodal_axes`` returns (lon, sin(lat)) by convention; included so
        # any helper that touches it on a stub coord still works.
        nodal_axes=(lon_rad, jnp.sin(lat_rad)),
    )
    return SimpleNamespace(
        horizontal=horizontal,
        vertical=vertical,
        nodal_shape=(nlev, 1, 1),
    )


def _expand_field(value: jnp.ndarray, nlev: int) -> jnp.ndarray:
    """Reshape a 1-D column field to ``(nlev, 1, 1)`` (or scalar surface to ``(1, 1)``)."""
    arr = jnp.asarray(value)
    if arr.ndim == 1:
        return arr.reshape(nlev, 1, 1)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    return arr


def _column_state_to_grid(column_state: PhysicsState, nlev: int) -> PhysicsState:
    """Reshape a 1-D column ``PhysicsState`` to the internal ``(nlev, 1, 1)`` grid."""
    grid_args = {}
    for field, value in column_state.asdict().items():
        if field == "tracers":
            grid_args["tracers"] = {
                k: _expand_field(v, nlev) for k, v in value.items()
            }
        elif field == "normalized_surface_pressure":
            arr = jnp.asarray(value)
            grid_args[field] = arr.reshape(1, 1) if arr.ndim == 0 else arr
        else:
            grid_args[field] = _expand_field(value, nlev)
    return type(column_state)(**grid_args)


def _squeeze_field(value: jnp.ndarray) -> jnp.ndarray:
    """Squeeze the ``(1, 1)`` grid axes off a per-cell array."""
    arr = jnp.asarray(value)
    if arr.ndim >= 2:
        return arr[..., 0, 0]
    return arr


def _squeeze_tendency(tend: PhysicsTendency) -> PhysicsTendency:
    args = {}
    for field, value in tend.asdict().items():
        if field == "tracers":
            args["tracers"] = {k: _squeeze_field(v) for k, v in value.items()}
        else:
            args[field] = _squeeze_field(value)
    return type(tend)(**args)


class SingleColumnModel:
    """Evolve physics tracers for one column at one ``(lat, lon)`` location.

    Example::

        from dinosaur.sigma_coordinates import SigmaCoordinates
        from jcm.physics.icon.icon_terms import icon_physics
        scm = SingleColumnModel(
            physics=icon_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0, lon_deg=180.0,
        )
        # column_state is a PhysicsState whose array fields are 1-D (nlev,)
        # and normalized_surface_pressure is a scalar.
        predictions = scm.run([column_state, column_state, ...])

    Args:
        physics: Physics package whose ``compute_tendencies`` drives evolution.
        vertical: Vertical coordinate (``SigmaCoordinates`` or
            ``HybridCoordinates``) — the only required spatial input.
        lat_deg: Column latitude in degrees (default 0).
        lon_deg: Column longitude in degrees (default 0).
        terrain: Optional single-column ``TerrainData`` (shape ``(1, 1)``);
            defaults to ``TerrainData.single_column()`` (flat, all ocean).
        forcing: Optional single-column ``ForcingData`` (shape ``(1, 1)``);
            defaults to ``ForcingData.zeros((1, 1))``.
        start_date: Starting date for the time series (default 2000-01-01).
        dt_seconds: Physics timestep in seconds (default 1800).
        apply_tracer_tendencies: When ``False`` tracers are reported
            diagnostically but not advanced.
        relaxation_timescales: Optional ``{var_name: tau_seconds}`` mapping.
            Listed prognostic variables (``u_wind``, ``v_wind``,
            ``temperature``, ``specific_humidity``) are nudged toward the
            prescribed state with timescale ``tau`` while still receiving
            their physics tendency.

    """

    def __init__(
        self,
        physics: Physics,
        vertical,
        lat_deg: float = 0.0,
        lon_deg: float = 0.0,
        terrain: TerrainData | None = None,
        forcing: ForcingData | None = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
        dt_seconds: float = 1800.0,
        apply_tracer_tendencies: bool = True,
        relaxation_timescales: dict[str, float] | None = None,
    ) -> None:
        """Initialise (see class docstring for argument descriptions)."""
        self.physics = physics
        self.vertical = vertical
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.start_date = start_date
        self.dt_seconds = float(dt_seconds)
        self.apply_tracer_tendencies = apply_tracer_tendencies
        self.relaxation_timescales = dict(relaxation_timescales or {})

        self.coords = _make_single_column_coords(vertical, lat_deg, lon_deg)
        self.terrain = terrain if terrain is not None else TerrainData.single_column()
        self.forcing = forcing if forcing is not None else ForcingData.zeros((1, 1))

        self.physics.cache_coords(self.coords)
        from jcm.physics.icon.icon_terms import ComposableIconPhysics
        if isinstance(self.physics, ComposableIconPhysics):
            self.physics.apply_timestep(self.dt_seconds)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _date(self, time_idx) -> DateData:
        sim_time_seconds = time_idx * self.dt_seconds
        seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(seconds=seconds_int),
            model_step=jnp.asarray(time_idx).astype(jnp.int32),
            dt_seconds=self.dt_seconds,
        )

    @staticmethod
    def _stack_states(states: list[PhysicsState]) -> PhysicsState:
        return tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *states)

    def _make_step_fn(
        self,
        forcing: ForcingData,
        apply_tendencies: bool,
        tracer_names: tuple[str, ...],
        relaxed_var_params: tuple[tuple[str, float], ...],
    ) -> Callable:
        physics = self.physics
        terrain = self.terrain
        nlev = self.coords.nodal_shape[0]
        dt_seconds = self.dt_seconds
        start_date = self.start_date

        def compute_date(time_idx):
            sim_time_seconds = time_idx * dt_seconds
            seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
            return DateData.set_date(
                model_time=start_date + jdt.Timedelta(seconds=seconds_int),
                model_step=jnp.asarray(time_idx).astype(jnp.int32),
                dt_seconds=dt_seconds,
            )

        def step_fn(prescribed_column, tracers, relaxed_vars, physics_data, time_idx):
            full_state_args = prescribed_column.asdict()
            full_state_args.pop("tracers", None)
            for name, _ in relaxed_var_params:
                full_state_args[name] = relaxed_vars[name]
            full_state_args["tracers"] = tracers
            column_state = type(prescribed_column)(**full_state_args)

            grid_state = _column_state_to_grid(column_state, nlev)
            clamped = verify_state(grid_state)
            tendencies_grid, new_physics_data = physics.compute_tendencies(
                clamped, forcing, terrain, compute_date(time_idx),
                prev_physics_data=physics_data,
            )
            tendencies = _squeeze_tendency(tendencies_grid)

            if apply_tendencies:
                updated_tracers = {}
                for name in tracer_names:
                    tracer = tracers[name]
                    tracer_tend = tendencies.tracers.get(name, jnp.zeros_like(tracer))
                    updated_tracers[name] = jnp.maximum(
                        tracer + dt_seconds * tracer_tend, 0.0,
                    )
            else:
                updated_tracers = tracers

            updated_relaxed_vars = {}
            for name, tau in relaxed_var_params:
                current_val = relaxed_vars[name]
                target_val = getattr(prescribed_column, name)
                phys_tend = getattr(tendencies, name)
                nudging_tend = (target_val - current_val) / tau
                updated_relaxed_vars[name] = (
                    current_val + dt_seconds * (phys_tend + nudging_tend)
                )

            return tendencies, updated_tracers, updated_relaxed_vars, new_physics_data

        return step_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prescribed_states: list[PhysicsState] | PhysicsState,
        forcing: ForcingData | None = None,
        initial_tracers: dict | None = None,
        initial_physics_data: Any = None,
        times: jnp.ndarray | None = None,
        initial_relaxed_vars: dict | None = None,
    ) -> SCMPredictions:
        """Run the SCM with evolving tracers.

        Args:
            prescribed_states: List of column ``PhysicsState`` snapshots, or
                a single ``PhysicsState`` whose leading axis is time. Array
                fields must be 1-D ``(nlev,)`` per snapshot.
            forcing: Optional override for the single-column forcing supplied
                at construction.
            initial_tracers: Initial values for prognostic tracers
                (1-D ``(nlev,)`` per tracer). Defaults to the first
                prescribed state's tracers (or ``{}``).
            initial_physics_data: Optional initial diagnostics dict.
            times: Optional days-since-start array.
            initial_relaxed_vars: Initial values for relaxed prognostic
                variables (1-D ``(nlev,)`` per variable).

        Returns:
            ``SCMPredictions``.

        """
        if forcing is None:
            forcing = self.forcing

        if isinstance(prescribed_states, list):
            prescribed_states = self._stack_states(prescribed_states)

        n_times = prescribed_states.u_wind.shape[0]

        if initial_tracers is None:
            first_tracers = tree_map(lambda x: x[0], prescribed_states.tracers)
            initial_tracers = first_tracers if first_tracers else {}

        relaxed_var_params = tuple(sorted(self.relaxation_timescales.items()))
        if initial_relaxed_vars is None:
            first_state_slice = tree_map(lambda x: x[0], prescribed_states)
            initial_relaxed_vars = {
                name: getattr(first_state_slice, name)
                for name, _ in relaxed_var_params
            }

        # Bootstrap the diagnostics-dict pytree shape by running one step.
        if initial_physics_data is None:
            first_state = tree_map(lambda x: x[0], prescribed_states)
            state_args = first_state.asdict()
            state_args.pop("tracers", None)
            for name, val in initial_relaxed_vars.items():
                state_args[name] = val
            state_args["tracers"] = initial_tracers
            first_state_combined = type(first_state)(**state_args)
            nlev = self.coords.nodal_shape[0]
            grid_state = _column_state_to_grid(first_state_combined, nlev)
            clamped = verify_state(grid_state)
            _, initial_physics_data = self.physics.compute_tendencies(
                clamped, forcing, self.terrain, self._date(0),
            )

        if times is None:
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)

        step_fn = self._make_step_fn(
            forcing=forcing,
            apply_tendencies=self.apply_tracer_tendencies,
            tracer_names=tuple(initial_tracers.keys()),
            relaxed_var_params=relaxed_var_params,
        )

        def scan_step(carry, time_idx):
            tracers, relaxed_vars, physics_data = carry
            prescribed_column = tree_map(lambda x: x[time_idx], prescribed_states)
            prescribed_column = prescribed_column.copy(tracers={})
            tendencies, new_tracers, new_relaxed_vars, new_physics_data = step_fn(
                prescribed_column, tracers, relaxed_vars, physics_data, time_idx,
            )
            new_carry = (new_tracers, new_relaxed_vars, new_physics_data)
            return new_carry, (tendencies, new_tracers, new_relaxed_vars, new_physics_data)

        initial_carry = (initial_tracers, initial_relaxed_vars, initial_physics_data)

        @jax.jit
        def run_scan():
            return lax.scan(scan_step, initial_carry, jnp.arange(n_times))

        _, (tendencies, tracer_history, relaxed_vars_history, physics_data_history) = run_scan()

        return SCMPredictions(
            prescribed_states=prescribed_states,
            tracer_states=tracer_history,
            relaxed_states=relaxed_vars_history,
            tendencies=tendencies,
            physics_data=physics_data_history,
            times=times,
        )
