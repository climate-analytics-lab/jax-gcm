"""Single-column / prescribed-state physics driver.

``SingleColumnModel`` evolves prognostic tracers (cloud water/ice, aerosols,
chemistry species, etc.) with ``lax.scan`` while large-scale atmospheric
state is supplied externally as a time series. The dynamical core does not
run; this is a process-level integrator where physics tendencies decide
state evolution.

Useful for:
- Process-level cloud/aerosol scheme validation against reference data.
- Reanalysis-driven offline physics runs.
- Faster-than-real-time tracer experiments (no spectral transform overhead).

The companion ``PrescribedStateModel`` (in
``jcm.prescribed_state_model``) performs the same diagnostic step but uses
``vmap`` because each step's tendencies are independent of every other.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import tree_math
from jax import lax
from jax.tree_util import tree_map

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.date import DateData
from jcm.forcing import ForcingData, default_forcing
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

    Attributes:
        prescribed_states: Time series of prescribed atmospheric states.
        tracer_states: Time series of evolved prognostic tracers (dict of arrays).
        relaxed_states: Time series of relaxed prognostic variables (dict of arrays;
            empty when no relaxation is configured).
        tendencies: Physics tendencies computed at each step.
        physics_data: Per-step diagnostics dict from the physics package.
        times: Times in days since ``start_date``.

    """

    prescribed_states: PhysicsState
    tracer_states: dict
    relaxed_states: dict
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any


class SingleColumnModel:
    """Evolve physics tracers with prescribed atmospheric state.

    Examples
    --------
    >>> from jcm.physics.icon.icon_terms import icon_physics
    >>> from jcm.physics.icon.icon_levels import get_icon_levels
    >>> from jcm.utils import get_coords
    >>> coords = get_coords(get_icon_levels(8), spectral_truncation=21)
    >>> model = SingleColumnModel(physics=icon_physics(), coords=coords)
    >>> predictions = model.run(prescribed_states, initial_tracers={'qc': ..., 'qi': ...})

    Args:
        physics: Physics package whose ``compute_tendencies`` will drive evolution.
        coords: ``CoordinateSystem`` used to size grids and to call
            ``physics.cache_coords``. Required.
        terrain: Optional ``TerrainData`` boundary conditions. Defaults to
            ``TerrainData.aquaplanet(coords)``.
        start_date: Starting date for the time series (default 2000-01-01).
        dt_seconds: Physics timestep in seconds (default 1800).
        apply_tracer_tendencies: When ``False`` tracers are reported diagnostically
            but not advanced — useful for sanity-checking tendencies in isolation.
        relaxation_timescales: Optional ``{var_name: tau_seconds}`` mapping.
            Listed prognostic variables (``u_wind``, ``v_wind``, ``temperature``,
            ``specific_humidity``) are nudged toward the prescribed state with
            timescale ``tau`` while still receiving their physics tendency.

    """

    def __init__(
        self,
        physics: Physics,
        coords: CoordinateSystem,
        terrain: TerrainData | None = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
        dt_seconds: float = 1800.0,
        apply_tracer_tendencies: bool = True,
        relaxation_timescales: dict[str, float] | None = None,
    ) -> None:
        self.physics = physics
        self.coords = coords
        self.terrain = terrain if terrain is not None else TerrainData.aquaplanet(coords)
        self.start_date = start_date
        self.dt_seconds = float(dt_seconds)
        self.apply_tracer_tendencies = apply_tracer_tendencies
        self.relaxation_timescales = dict(relaxation_timescales or {})

        # Cache coord-dependent transforms once. ICON physics also needs the
        # timestep so radiation sub-stepping / accumulators are correct.
        self.physics.cache_coords(coords)
        from jcm.physics.icon.icon_terms import ComposableIconPhysics
        if isinstance(self.physics, ComposableIconPhysics):
            self.physics.apply_timestep(self.dt_seconds)

    # ------------------------------------------------------------------
    # Date helper
    # ------------------------------------------------------------------

    def _date_from_time_index(self, time_index) -> DateData:
        sim_time_seconds = time_index * self.dt_seconds
        seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(seconds=seconds_int),
            model_step=jnp.asarray(time_index).astype(jnp.int32),
            dt_seconds=self.dt_seconds,
        )

    # ------------------------------------------------------------------
    # Pure step factory (closed over static config)
    # ------------------------------------------------------------------

    def _make_step_fn(
        self,
        forcing: ForcingData,
        terrain: TerrainData,
        apply_tendencies: bool,
        dt_seconds: float,
        tracer_names: tuple[str, ...],
        relaxed_var_params: tuple[tuple[str, float], ...],
    ) -> Callable:
        physics = self.physics
        start_date = self.start_date

        def compute_date(time_idx):
            sim_time_seconds = time_idx * dt_seconds
            seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
            return DateData.set_date(
                model_time=start_date + jdt.Timedelta(seconds=seconds_int),
                model_step=jnp.asarray(time_idx).astype(jnp.int32),
                dt_seconds=dt_seconds,
            )

        def step_fn(prescribed_state, tracers, relaxed_vars, physics_data, time_idx):
            full_state_args = prescribed_state.asdict()
            full_state_args.pop("tracers", None)
            for name, _ in relaxed_var_params:
                full_state_args[name] = relaxed_vars[name]
            full_state_args["tracers"] = tracers
            full_state = type(prescribed_state)(**full_state_args)
            clamped_state = verify_state(full_state)

            tendencies, new_physics_data = physics.compute_tendencies(
                clamped_state, forcing, terrain, compute_date(time_idx),
                prev_physics_data=physics_data,
            )

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
                target_val = getattr(prescribed_state, name)
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

    @staticmethod
    def _stack_states(states: list[PhysicsState]) -> PhysicsState:
        return tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *states)

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
            prescribed_states: List of ``PhysicsState`` snapshots, or a single
                ``PhysicsState`` whose leading axis is time.
            forcing: Surface forcing data; defaults to the aquaplanet forcing
                derived from ``self.coords.horizontal``.
            initial_tracers: Initial values for prognostic tracers. Defaults
                to the first prescribed state's tracers (or ``{}``).
            initial_physics_data: Optional initial diagnostics dict; if ``None``
                one physics step is run upfront to materialise the right pytree
                structure for ``lax.scan``.
            times: Optional days-since-start array; defaults to
                ``jnp.arange(n_times) * dt_seconds / 86400``.
            initial_relaxed_vars: Initial values for relaxed prognostic variables.

        Returns:
            ``SCMPredictions``.

        """
        if forcing is None:
            forcing = default_forcing(self.coords.horizontal)

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
            first_date = self._date_from_time_index(0)
            clamped_state = verify_state(first_state_combined)
            _, initial_physics_data = self.physics.compute_tendencies(
                clamped_state, forcing, self.terrain, first_date,
            )

        if times is None:
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)

        step_fn = self._make_step_fn(
            forcing=forcing,
            terrain=self.terrain,
            apply_tendencies=self.apply_tracer_tendencies,
            dt_seconds=self.dt_seconds,
            tracer_names=tuple(initial_tracers.keys()),
            relaxed_var_params=relaxed_var_params,
        )

        def scan_step(carry, time_idx):
            tracers, relaxed_vars, physics_data = carry
            prescribed_state = tree_map(lambda x: x[time_idx], prescribed_states)
            prescribed_state = prescribed_state.copy(tracers={})
            tendencies, new_tracers, new_relaxed_vars, new_physics_data = step_fn(
                prescribed_state, tracers, relaxed_vars, physics_data, time_idx,
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

    def run_single(
        self,
        prescribed_state: PhysicsState,
        forcing: ForcingData | None = None,
        tracers: dict | None = None,
        physics_data: Any = None,
        time_index: int = 0,
    ) -> tuple[PhysicsTendency, dict, dict, Any]:
        """Compute one physics step against ``prescribed_state``.

        Returns ``(tendencies, updated_tracers, updated_relaxed_vars, physics_data)``.
        """
        if forcing is None:
            forcing = default_forcing(self.coords.horizontal)
        if tracers is None:
            tracers = prescribed_state.tracers if prescribed_state.tracers else {}

        relaxed_var_params = tuple(sorted(self.relaxation_timescales.items()))
        relaxed_vars = {
            name: getattr(prescribed_state, name) for name, _ in relaxed_var_params
        }

        if physics_data is None:
            temp_state = prescribed_state.copy(tracers=tracers)
            clamped_state = verify_state(temp_state)
            _, physics_data = self.physics.compute_tendencies(
                clamped_state, forcing, self.terrain,
                self._date_from_time_index(time_index),
            )

        step_fn = self._make_step_fn(
            forcing=forcing,
            terrain=self.terrain,
            apply_tendencies=self.apply_tracer_tendencies,
            dt_seconds=self.dt_seconds,
            tracer_names=tuple(tracers.keys()),
            relaxed_var_params=relaxed_var_params,
        )

        return step_fn(
            prescribed_state.copy(tracers={}),
            tracers,
            relaxed_vars,
            physics_data,
            time_index,
        )
