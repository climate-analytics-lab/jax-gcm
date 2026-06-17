"""User-facing :class:`Model` class.

The :class:`Model` orchestrates a simulation: forcing, run/resume/run_from_state,
chunked op-split scan, post-processing, and xarray conversion. The two
component contracts it routes between are:

* the :class:`DynamicalCore` (state initialisation, the per-``dt`` step,
  the gridpoint↔native bridge) — see :mod:`jcm.dycore`;
* the :class:`Physics` (parameterizations producing gridpoint
  :class:`PhysicsTendency` from a :class:`PhysicsState`) — see
  :mod:`jcm.physics`.

The Model itself owns nothing dynamics- or physics-specific; it just
threads the cross-step physics carry through the scan, handles the
sim-time / date bookkeeping, and produces an xarray trajectory.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import jax_datetime as jdt
from numpy import timedelta64
from typing import Callable, Any
from dinosaur.scales import units
import pandas as pd
from functools import partial
import logging

from jcm.date import DateData, parse_duration_days
from jcm.forcing import ForcingData, default_forcing
from jcm.physics_interface import (
    PhysicsState, Physics, compute_physics_step_gridpoint, verify_state,
)
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.terrain import TerrainData
from jcm.utils import DYNAMICS_UNITS_TABLE_CSV_PATH
from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.dinosaur.dycore import DinosaurDycore


logger = logging.getLogger(__name__)


class ModelPredictions:
    """User-facing container for model prediction outputs.

    Wraps the internal :class:`Predictions` pytree with the coordinate system
    and physics module needed for xarray conversion. Returned by
    :meth:`Model.run`, :meth:`Model.resume`, and :meth:`Model.run_from_state`.

    Attributes:
        dynamics (PhysicsState): The physical state variables.
        physics (Any): Diagnostic physics data.
        times (Any): Timestamps of the predictions.

    """

    def __init__(self, predictions: Predictions, coords, physics: Physics):  # noqa: D107
        self._predictions = predictions
        self._coords = coords
        self._physics = physics

    @property
    def dynamics(self):
        return self._predictions.dynamics

    @property
    def physics(self):
        return self._predictions.physics

    @property
    def times(self):
        return self._predictions.times

    def to_xarray(self):
        """Convert the full prediction trajectory to an xarray.Dataset.

        Returns:
            An xarray.Dataset ready for analysis and plotting.

        """
        from jcm.utils import data_to_xarray

        # float0s are placeholders representing the lack of tangent space for non-differentiable variables.
        # jax.numpy arrays cannot have float0 dtype, so jcm handles them with numpy arrays;
        # substituting jax.numpy arrays here allows us to handle Predictions objects that contain derivatives.
        float0s_to_nans = lambda pytree: tree_map(
            lambda x: jnp.full_like(x, jnp.nan, dtype=float) if x.dtype == jax.dtypes.float0 else x,
            pytree,
        )

        dynamics_predictions = float0s_to_nans(self.dynamics)
        physics_predictions = float0s_to_nans(self.physics)

        nodal_shape = dynamics_predictions.u_wind.shape[1:]

        # Per-physics flattening of the diagnostic struct into a dict of named fields.
        physics_preds_dict = self._physics.data_struct_to_dict(physics_predictions, nodal_shape=nodal_shape)

        times = jax.device_get(self.times)
        coords = jax.device_get(self._coords)

        additional_coords = {}
        if self._physics.cached_coords is not None and hasattr(self._physics.cached_coords, 'xarray_additional_coords'):
            additional_coords = self._physics.cached_coords.xarray_additional_coords()

        pred_ds = data_to_xarray(
            dynamics_predictions.asdict() | physics_preds_dict,
            coords=coords, serialize_coords_to_attrs=False,
            times=times - times[0],
            additional_coords=additional_coords,
        )

        # Attach units / descriptions from the physics-specific units table.
        units_df = pd.read_csv(DYNAMICS_UNITS_TABLE_CSV_PATH)
        if self._physics.UNITS_TABLE_CSV_PATH is not None:
            units_df = pd.concat([units_df, pd.read_csv(self._physics.UNITS_TABLE_CSV_PATH)], ignore_index=True)
        for var, unit, desc in zip(units_df["Variable"], units_df["Units"], units_df["Description"]):
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
                pred_ds[var].attrs["description"] = desc

        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere.
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # Convert sim-day timestamps to datetimes.
        pred_ds['time'] = (
            times * (timedelta64(1, 'D') / timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')

        return pred_ds


def _model_predictions_flatten(mp):
    """Flatten ModelPredictions for JAX pytree operations (tree_map, etc.).

    Only the internal Predictions pytree is treated as array data. Coords and
    physics are not in aux_data so that ``tree_map`` works across ModelPredictions
    from different Model instances.
    """
    children = (mp._predictions,)
    return children, None


def _model_predictions_unflatten(aux_data, children):
    return ModelPredictions(children[0], None, None)


jax.tree_util.register_pytree_node(
    ModelPredictions,
    _model_predictions_flatten,
    _model_predictions_unflatten,
)


def _op_split_trajectory(
    step_fn: Callable[[Any, Any], tuple[Any, Any]],
    initial_physics_state: Any,
    empty_diagnostics: Any,
    outer_steps: int,
    inner_steps: int,
    post_process_fn: Callable[[Any, Any], Any] = lambda x, ps: x,
    output_averages: bool = False,
) -> Callable[[Any], tuple[Any, Any, Any]]:
    """Trajectory builder for the operator-split path.

    The op-split ``step_fn`` has signature ``(state, physics_state) ->
    (state_next, physics_state_next)``. ``physics_state`` is the cross-step
    physics carry (radiation flux for sub-cycling, prev TKE for the analytic
    source update, etc.) and flows through the ``lax.scan`` as a first-class
    pytree.

    ``post_process_fn`` takes ``(state, physics_state)``. In snapshot mode the
    saved physics carry is exactly the one used by the integration — radiation
    sub-cycle cache, TKE memory, etc. — so diagnostics reported in
    ``predictions.physics`` match what the dycore actually consumed.

    In averaged mode, the per-step physics dict (the same dict that becomes
    ``physics_state_next``) is accumulated as a running mean across the inner
    steps and saved per outer step. The running mean uses POST-step states
    (``x_next``); this matches the snapshot path's end-of-step samples, so
    ``mean(snapshots)`` and ``averaged(...)`` agree to numerical roundoff.

    Args:
        step_fn: Operator-split per-``dt`` step.
        initial_physics_state: Cross-step carry initial value (built via
            :meth:`ComposablePhysics.initial_carry_state` unioned with a
            structural template from :meth:`ComposablePhysics.get_empty_data`).
        empty_diagnostics: Zero-shaped diagnostics dict used to seed the
            running-mean accumulator in averaged mode. Same structure as the
            per-step ``physics_state_next``.
        outer_steps: Number of saved frames.
        inner_steps: Inner ``dt`` steps between saved frames.
        post_process_fn: Applied to the state at save time (snapshot mode) or
            to the running mean (averaged mode).
        output_averages: When True, the saved frame is the running mean of
            ``post_process_fn(state)`` over the inner steps.

    Returns:
        A function ``initial_state -> (final_state, final_physics_state,
        saved_trajectory)`` where ``saved_trajectory`` has a leading axis of
        length ``outer_steps``. ``final_physics_state`` is the cross-step carry
        coming out of the last ``dt`` — exposing it lets callers (e.g.
        ``Model.resume``) thread a continuous carry across API boundaries so
        a 5d + resume(5d) integration matches a single 10d integration. In
        averaged mode the returned trajectory's ``physics`` field is the
        time-averaged diagnostics dict.

    """
    # Snapshot and averaged modes only differ in what the inner scan
    # accumulates and what the outer step saves; the surrounding outer
    # ``lax.scan`` over ``(state, physics_state)`` and the
    # ``(x_final, ps_final, preds)`` return are identical, so define them
    # once.
    def _averaged_outer_step():
        @jax.checkpoint
        def inner_step(carry, _):
            x, physics_state, x_sum, diag_sum = carry
            x_next, physics_state_next = step_fn(x, physics_state)
            # Sum POST-step states so that mean(state_1..state_N) matches the
            # snapshot path (which saves state_N at outer steps). Summing
            # pre-step states would be off by one timestep — tolerable for
            # slow fields, but the op-split per-step transient is large
            # enough to surface as test failures at the rtol=1e-3 the
            # averaging test runs at.
            x_sum = tree_map(lambda a, b: a + b, x_sum, x_next)
            diag_sum = tree_map(
                lambda acc, new: acc + new / inner_steps,
                diag_sum, physics_state_next,
            )
            return (x_next, physics_state_next, x_sum, diag_sum), None

        def outer_step(carry, _, empty_sum, empty_diag_sum):
            x, physics_state = carry
            init = (x, physics_state, empty_sum, empty_diag_sum)
            (x_next, ps_next, x_sum, diag_sum), _ = jax.lax.scan(
                inner_step, init, None, length=inner_steps,
            )
            averaged_state = tree_map(lambda s: s / inner_steps, x_sum)
            preds = post_process_fn(averaged_state, ps_next)
            preds = preds.replace(physics=diag_sum)
            return (x_next, ps_next), preds

        return outer_step

    def _snapshot_outer_step():
        @jax.checkpoint
        def inner_step(carry, _):
            x, physics_state = carry
            x_next, physics_state_next = step_fn(x, physics_state)
            return (x_next, physics_state_next), None

        def outer_step(carry, _):
            (x_final, ps_final), _ = jax.lax.scan(
                inner_step, carry, None, length=inner_steps,
            )
            # Save the carried physics state alongside the dynamics state.
            # Calling ``post_process_fn`` with ``ps_final`` lets snapshot
            # diagnostics reflect the sub-cycled radiation cache / TKE
            # memory the dycore actually consumed — recomputing physics at
            # save time with a freshly-seeded carry would zero out radiation
            # on non-radiation outer steps (default 2-hour
            # ``radiation_interval``).
            return (x_final, ps_final), post_process_fn(x_final, ps_final)

        return outer_step

    def integrate(x_initial):
        if output_averages:
            empty_sum = tree_map(jnp.zeros_like, x_initial)
            # Cast accumulator leaves to float so that ``acc + new / N`` doesn't
            # promote dtype mid-scan — jax.lax.scan rejects type changes in the
            # carry.
            empty_diag_sum = tree_map(
                lambda x: jnp.zeros(jnp.shape(x), dtype=float),
                empty_diagnostics,
            )
            outer_step_fn = _averaged_outer_step()
            outer_step = lambda c, _: outer_step_fn(
                c, _, empty_sum, empty_diag_sum,
            )
        else:
            outer_step = _snapshot_outer_step()

        (x_final, ps_final), preds = jax.lax.scan(
            outer_step,
            (x_initial, initial_physics_state),
            None, length=outer_steps,
        )
        return x_final, ps_final, preds

    return integrate


class Model:
    """Top level class for a JAX-GCM simulation.

    The Model orchestrates the run (timestep, forcing, op-split scan,
    post-processing). Dynamics-specific work (state init, the per-``dt`` step,
    the spectral↔gridpoint bridge) is delegated to a :class:`DynamicalCore`;
    physics-specific work to a :class:`Physics`.
    """

    def __init__(self,
                 dycore: DynamicalCore | None = None,
                 *,
                 coords=None,
                 time_step: float = 30.0,
                 terrain: TerrainData = None,
                 physics: Physics = None,
                 start_date: jdt.Datetime = jdt.to_datetime('2000-01-01'),
                 calendar: str = "365_day",
                 log_level=logging.CRITICAL) -> None:
        """Initialise the model.

        Args:
            dycore: The :class:`DynamicalCore` driving the integration. When
                ``None``, a default :class:`DinosaurDycore` is constructed
                from ``coords`` and ``terrain`` for convenience. Backend-
                specific knobs (diffusion, nudging-as-PhysicsTerm targets,
                IMEX stepper details) belong to the dycore's own constructor
                — wire them there, then pass the dycore in.
            coords: CoordinateSystem. Required when ``dycore`` is ``None``.
                To enable SPMD parallelization, pass ``spmd_mesh`` to the
                coords helper (e.g. :func:`get_speedy_coords`).
            time_step: Model time step in minutes.
            terrain: :class:`TerrainData` (orography, land-sea mask, etc.).
                Defaults to an aquaplanet when building the default dycore.
            physics: :class:`Physics` describing the model physics. Defaults
                to :func:`speedy_physics`. Add nudging via the
                :class:`jcm.nudging.NudgingTerm` PhysicsTerm.
            start_date: ``jax_datetime.Datetime`` for the start of the run.
                Used to convert ``state.sim_time`` to a :class:`DateData`
                that's threaded into the physics-step diagnostics dict (so
                forcing-driven and date-aware terms can read it).
            calendar: Calendar string (``"365_day"`` or ``"gregorian"``) for
                the same date conversion.
            log_level: Logging verbosity level.

        """
        logging.getLogger().setLevel(log_level)
        self.calendar = calendar
        self.start_date = start_date

        self.dt_si = (time_step * units.minute).to(units.second)
        self.physics = physics if physics is not None else speedy_physics()

        tracer_specs = {spec.name: spec for spec in self.physics.required_tracers()}
        if dycore is None:
            if coords is None:
                raise ValueError(
                    "Model requires either an explicit ``dycore`` or a "
                    "``coords`` argument (used to build the default "
                    "DinosaurDycore)."
                )
            terrain = terrain if terrain is not None else TerrainData.aquaplanet(coords)
            dycore = DinosaurDycore(
                coords=coords,
                terrain=terrain,
                dt_seconds=float(self.dt_si.m),
                tracer_specs=tracer_specs,
            )
        self.dycore = dycore
        # Synchronise the dycore's tracer specs with the active physics so
        # the explicit-dycore path can ship with default (empty) specs and
        # still mis-scale-correctly on tracers whose
        # ``TracerSpec.nondimensionalize=False``.
        self.dycore.required_tracers_ok(self.physics.required_tracers())
        self.dycore.tracer_specs = tracer_specs
        # Convenience aliases so callers don't have to type ``self.dycore.coords``.
        self.coords = dycore.coords
        self.terrain = dycore.terrain

        self.physics.cache_coords(self.coords)
        # Hand the model's timestep to the physics. ``ComposablePhysics``
        # injects it into the diagnostics dict every step under
        # ``"_dt_seconds"`` so any term that integrates by ``dt`` (chemistry,
        # microphysics, vertical diffusion, …) reads a single source of truth
        # instead of going through date plumbing.
        if hasattr(self.physics, "dt_seconds"):
            self.physics.dt_seconds = float(self.dt_si.m)

        # Initial gridpoint state set upon calling model.run.
        self.initial_nodal_state = None

        # Dycore-native state at end of last run/resume.
        self._final_dycore_state = None

        # Cross-step physics carry threaded through op-split run/resume.
        # ``None`` means "build a fresh carry on the next call"; set by
        # ``bootstrap_state`` so that ``run() + resume()`` matches a single
        # ``run()`` of the combined duration.
        self._final_physics_state = None

    def _date_from_sim_time(self, sim_time) -> DateData:
        # Stop gradient: date/calendar computations use non-differentiable ops
        # (floor, round, int casts) and should not be part of the AD graph.
        sim_time = jax.lax.stop_gradient(sim_time)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(
                days=jnp.floor(sim_time / 86400).astype(jnp.int32),
                seconds=jnp.round(sim_time % 86400).astype(jnp.int32),
            ),
            model_step=jnp.int32(sim_time / self.dt_si.m),
            dt_seconds=float(self.dt_si.m),
            calendar=self.calendar,
        )

    def _prepare_initial_dycore_state(self, physics_state: PhysicsState = None,
                                      random_seed=0, sim_time=0.0):
        """Build the dycore-native initial state.

        Thin wrapper around :meth:`DynamicalCore.initial_state` that supplies
        the tracer specs aggregated from the active physics package.
        """
        tracer_specs = {spec.name: spec for spec in self.physics.required_tracers()}
        return self.dycore.initial_state(
            physics_state,
            sim_time=sim_time,
            random_seed=random_seed,
            tracer_specs=tracer_specs,
        )

    def _get_op_split_step_fn(self, forcing: ForcingData):
        """Build the operator-split single-step function (Lie split a).

        One call: ``(state, physics_state) -> (state_next, physics_state_next)``.

        Order: ``state → gridpoint projection → physics_tendency → dycore.step``
        (which itself does the forward-Euler add, the dynamics step, and the
        spectral filters). Mirrors ECHAM6's ``physc`` → ``sccd``/``scctp`` →
        ``hdiff`` chain.
        """
        def step(state, physics_state):
            date = self._date_from_sim_time(self.dycore.sim_time(state))
            forcing_now = forcing.select(date, calendar=self.calendar)
            physics_state_grid = self.dycore.to_physics_state(state)
            physics_tendency, new_physics_state = compute_physics_step_gridpoint(
                physics_state_grid, forcing_now, self.terrain, physics_state,
                physics=self.physics,
                time_step=self.dt_si.m,
            )
            state_next = self.dycore.step(state, physics_tendency)
            return state_next, new_physics_state

        return step

    def _post_process(
        self,
        state,
        physics_state: Any,
        output_averages: bool,
    ) -> Predictions:
        """Post-process a single saved state from the op-split trajectory.

        The op-split scan threads ``physics_state`` — the cross-step carry
        returned by the prior ``compute_tendencies`` call — into this function
        at save time. We use it directly as the ``predictions.physics`` payload
        in snapshot mode rather than re-running physics with a freshly-seeded
        carry. That avoids the bug where sub-cycled radiation diagnostics
        (default ``radiation_interval=7200s``) would be reported as zero on
        non-radiation outer steps because the recompute path didn't see the
        cached radiation fields the dycore was actually consuming.

        In averaged mode the caller overrides ``predictions.physics`` with the
        inner-step running mean, so the value attached here is discarded — we
        leave it as ``physics_state`` for symmetry and pytree-structure stability.

        Non-negative tracers (``specific_humidity``, ``qc``/``qi``, GHG VMRs)
        get a final ``verify_state`` clamp at the dycore→gridpoint output
        boundary so spectral Gibbs ringing of the physics tendency doesn't leak
        negative values into user-visible output.
        """
        jax.debug.callback(
            lambda t: logger.info("Post processing: %s simulated seconds", t),
            self.dycore.sim_time(state),
        )
        return Predictions(
            dynamics=verify_state(self.dycore.to_physics_state(state)),
            physics=physics_state if not output_averages else None,
            times=None,
        )

    def _build_initial_physics_carry(self) -> Any:
        """Build the cross-step physics carry seed for an op-split run.

        Pulls per-term initial state from :meth:`Physics.initial_carry_state`
        (deterministic, no zero-state probe). Unions with the *structural
        template* from :meth:`Physics.get_empty_data` so the ``lax.scan`` carry
        pytree matches the post-step ``compute_tendencies`` output structure
        on iteration 1 (within-step diagnostic keys terms write are
        zero-filled). ``get_empty_data`` is internal-only in this role.
        """
        template = self.physics.get_empty_data(self.coords)
        initial_carry = self.physics.initial_carry_state(self.coords)
        if isinstance(initial_carry, dict) and isinstance(template, dict):
            return {**template, **initial_carry}
        # Explicit ``is None`` check: ``initial_carry or template`` would
        # trigger ``bool(carry)`` and raise an ambiguous-truth ``ValueError``
        # if a ``Physics`` subclass returns a JAX array (or any object with
        # non-scalar truth semantics).
        return template if initial_carry is None else initial_carry

    def _get_op_split_integrate_fn(
        self,
        step_fn,
        outer_steps,
        inner_steps,
        post_process_fn,
        output_averages,
    ):
        """Integrate-fn builder for the operator-split path.

        Returns a closure ``(state, initial_physics_state) -> (final_state,
        final_physics_state, predictions)``. The running-mean accumulator
        template comes from :meth:`Physics.get_empty_data` — a zero-filled
        snapshot of the dict ``compute_tendencies`` produces, which is exactly
        the pytree structure the scan carries.
        """
        template = self.physics.get_empty_data(self.coords)

        def _integrate_fn(state, initial_physics_state):
            trajectory = _op_split_trajectory(
                step_fn=step_fn,
                initial_physics_state=initial_physics_state,
                empty_diagnostics=template,
                outer_steps=outer_steps,
                inner_steps=inner_steps,
                post_process_fn=post_process_fn,
                output_averages=output_averages,
            )
            return trajectory(state)

        return _integrate_fn

    @partial(jax.jit, static_argnums=(0, 4, 5, 6))  # Note: changing fields assumed static won't propagate.
    def _run_from_state(self,
                        initial_state,
                        initial_physics_state: Any,
                        forcing: ForcingData,
                        save_interval=10.0,
                        total_time=120.0,
                        output_averages=False,
    ):
        """JIT-compiled simulation loop. Returns raw :class:`Predictions` pytree.

        Physics is computed once per ``dt`` outside the dycore's stage loop
        and applied as a gridpoint :class:`PhysicsTendency` that the dycore
        adds via forward-Euler (operator-split Lie a from #471). The
        cross-step physics carry is first-class — threaded in as
        ``initial_physics_state`` and returned as the final carry so callers
        can continue a run across API boundaries without re-seeding (e.g.
        :meth:`Model.resume`).
        """
        inner_steps = int(save_interval / self.dt_si.to(units.day).m)
        outer_steps = int(total_time / save_interval)
        # Op-split saves end-of-step states (snapshot mode) or post-step
        # running means (averaged mode), so the first saved frame is at
        # ``initial_state.sim_time + save_interval``, not ``+ 0``. Index by
        # ``arange(outer_steps) + 1`` to label frames at the times they
        # actually correspond to.
        times = self.start_date.delta.days \
                + (self.dycore.sim_time(initial_state) * units.second).to(units.day).m \
                + save_interval * (jnp.arange(outer_steps) + 1)

        op_split_step = self._get_op_split_step_fn(forcing)
        integrate = self._get_op_split_integrate_fn(
            op_split_step,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            post_process_fn=lambda state, physics_state: self._post_process(
                state, physics_state, output_averages,
            ),
            output_averages=output_averages,
        )
        final_dycore_state, final_physics_state, predictions = integrate(
            initial_state, initial_physics_state,
        )

        return final_dycore_state, final_physics_state, predictions.replace(times=times)

    def run_from_state(self,
                       initial_state,
                       forcing: ForcingData,
                       save_interval=10.0,
                       total_time=120.0,
                       output_averages=False,
    ):
        """Run the simulation forward from a given dycore-native initial state.

        Alternative to ``model.run`` / ``model.resume`` which does not read or
        write the model's internal state.

        Note: the operator-split path carries a cross-step physics state
        (radiation cache, prior-step TKE, …). This method rebuilds that carry
        from scratch at every call. For chaining runs continuously (so the
        carry persists across API boundaries), use ``run`` / ``resume`` —
        those thread ``self._final_physics_state`` automatically. For an
        advanced caller that wants explicit control of the carry, use
        :meth:`run_from_state_with_carry`.

        Args:
            initial_state: Dycore-native initial state (e.g.
                ``primitive_equations.State`` for the dinosaur backend).
            forcing: :class:`ForcingData` containing forcing for the run.
            save_interval: Interval at which to save outputs. Number of days
                (float) or a calendar string like ``'1 month'``.
            total_time: Total time to run. Same units as ``save_interval``.
            output_averages: Whether to output time-averaged quantities.

        Returns:
            A tuple ``(final_dycore_state, ModelPredictions)``.

        """
        final_state, _, predictions = self.run_from_state_with_carry(
            initial_state,
            forcing,
            save_interval=save_interval,
            total_time=total_time,
            output_averages=output_averages,
        )
        return final_state, predictions

    def run_from_state_with_carry(self,
                                  initial_state,
                                  forcing: ForcingData,
                                  save_interval=10.0,
                                  total_time=120.0,
                                  output_averages=False,
                                  initial_physics_state: Any = None,
    ):
        """Lower-level ``run_from_state`` that exposes the cross-step physics carry."""
        save_interval_days = parse_duration_days(save_interval, calendar=self.calendar)
        total_time_days = parse_duration_days(total_time, calendar=self.calendar)
        if initial_physics_state is None:
            initial_physics_state = self._build_initial_physics_carry()
        final_dycore_state, final_physics_state, predictions = self._run_from_state(
            initial_state, initial_physics_state, forcing,
            save_interval_days, total_time_days,
            output_averages,
        )
        return (
            final_dycore_state,
            final_physics_state,
            ModelPredictions(predictions, self.coords, self.physics),
        )

    def resume(self,
               forcing: ForcingData = None,
               save_interval=10.0,
               total_time=120.0,
               output_averages=False,
    ) -> ModelPredictions:
        """Continue from end of previous ``run`` / ``resume``.

        Continues the cross-step physics carry across the call boundary:
        ``self._final_physics_state`` from the previous ``run``/``resume`` is
        threaded back in so sub-cycled radiation, prior-step TKE, etc. don't
        reset at the API seam. A run broken into ``run()`` then ``resume()``
        for the same total duration therefore matches a single ``run()`` of
        the combined duration (to numerical roundoff).
        """
        jax.debug.callback(
            lambda: logger.info(
                "Model starting with params: save_interval: %s, total_time: %s, output_averages: %s",
                save_interval, total_time, output_averages),
        )
        final_dycore_state, final_physics_state, predictions = self.run_from_state_with_carry(
            initial_state=self._final_dycore_state,
            forcing=forcing or default_forcing(self.coords.horizontal),
            save_interval=save_interval,
            total_time=total_time,
            output_averages=output_averages,
            initial_physics_state=self._final_physics_state,
        )
        jax.debug.callback(lambda: logger.info("Run completed."))
        self._final_dycore_state = final_dycore_state
        self._final_physics_state = final_physics_state
        return predictions

    def run(self,
            initial_state=None,
            forcing: ForcingData = None,
            save_interval=10.0,
            total_time=120.0,
            output_averages=False,
    ) -> ModelPredictions:
        """Set the initial state and run the full simulation forward in time.

        ``initial_state`` may be:
            * ``None`` — the dycore builds its own default initial state.
            * a :class:`PhysicsState` — gridpoint state, projected onto the
              dycore via :meth:`DynamicalCore.initial_state`.
            * a dycore-native state (e.g. ``primitive_equations.State`` for
              the dinosaur backend) — used directly.
        """
        self.bootstrap_state(initial_state)
        return self.resume(
            forcing=forcing, save_interval=save_interval,
            total_time=total_time, output_averages=output_averages,
        )

    def bootstrap_state(self, initial_state=None) -> None:
        """Populate ``_final_dycore_state`` and ``_final_physics_state`` without integrating.

        Equivalent to the prep that ``run`` does before its first ``resume``
        call, but exposed as a standalone method so callers that need only the
        initial pytrees — checkpoint restore (where ``flax.serialization.from_bytes``
        requires a template), state introspection, or a bring-your-own-stepper
        workflow — don't have to spin up a zero-length integration to get them.

        ``initial_state`` may be ``None``, a gridpoint :class:`PhysicsState`,
        or a dycore-native state.
        """
        if initial_state is None:
            self.initial_nodal_state = None
            self._final_dycore_state = self._prepare_initial_dycore_state(None)
        elif isinstance(initial_state, PhysicsState):
            self.initial_nodal_state = initial_state
            self._final_dycore_state = self._prepare_initial_dycore_state(initial_state)
        else:
            # Assume the caller has supplied a dycore-native state object.
            self.initial_nodal_state = self.dycore.to_physics_state(initial_state)
            self._final_dycore_state = initial_state

        # Eagerly build the physics carry. ``resume`` would otherwise build it
        # lazily on first call, but materialising it here makes the pytree
        # available as a checkpoint-restore template and to any caller that
        # wants to inspect / mutate the seed state before stepping.
        self._final_physics_state = self._build_initial_physics_carry()
