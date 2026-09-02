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
from typing import Callable, Any
from dinosaur.scales import units
from functools import partial
import logging

from jcm import profiling, provenance
from jcm.date import DateData, parse_duration_days
from jcm.forcing import ForcingData, default_forcing
from jcm.predictions import ModelPredictions
from jcm.physics_interface import (
    PhysicsState, Physics, compute_physics_step_gridpoint, verify_state,
)
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.terrain import TerrainData
from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.dinosaur.dycore import DinosaurDycore


logger = logging.getLogger(__name__)

#: Per-trace parameter records kept for provenance (#732). A model
#: retraces once per distinct combination of static arguments and dynamic
#: avals, which is a handful in normal use; the cap bounds the memory a
#: long-lived model can accumulate. Each record is a few kB.
_MAX_TRACED_PARAM_RECORDS = 64


# ---------------------------------------------------------------------------
# Explicit-mesh (multi-device) support
# ---------------------------------------------------------------------------
# The pySES backend shards its element axis across devices under an *explicit*
# JAX mesh (``jax.set_mesh`` in pyses' JaxBackend). Physics columns are
# element-major, so the element sharding IS a block sharding of the trailing
# column axis — but explicit-mode semantics reject the replicated-scratch
# patterns column physics is written with (``jnp.zeros((nlev, ncols))`` meeting
# a sharded field, etc.). The physics call therefore runs under
# ``jax.sharding.auto_axes``: classic GSPMD propagation inside, with the
# declared column sharding re-imposed on every output that has a trailing
# column axis. On a single device (or the dinosaur SPMD path, which uses
# ordinary auto meshes) there is no ambient explicit axis and all of this is
# inert.

def _ambient_explicit_axis():
    """Name of the ambient explicit mesh axis (pyses' element axis), or None."""
    mesh = jax.sharding.get_abstract_mesh()
    explicit = getattr(mesh, "explicit_axes", ())
    return explicit[0] if explicit else None


def _column_partition_specs(tree, ncols: int, axis: str):
    """Per-leaf PartitionSpec: shard a trailing ``ncols`` axis, else replicate."""
    from jax.sharding import PartitionSpec

    def spec(leaf):
        shape = jnp.shape(leaf)
        if len(shape) >= 1 and shape[-1] == ncols:
            return PartitionSpec(*((None,) * (len(shape) - 1)), axis)
        return PartitionSpec()

    return tree_map(spec, tree)


def _reshard_columns(tree, ncols: int, axis: str):
    """Reshard every trailing-column leaf onto the explicit axis.

    Used on the scan-entry physics carry (and the averaged-diagnostics
    accumulator template): the per-step physics outputs are column-sharded, and
    ``lax.scan`` requires the carry type — sharding included — to be
    invariant, so the *initial* carry must already be sharded the same way.
    This also covers carries restored from a checkpoint (host numpy arrays).
    """
    from jax.sharding import reshard

    return reshard(tree, _column_partition_specs(tree, ncols, axis))


def _neutralize_mesh_typing(physics) -> None:
    """Strip the ambient-mesh typing from a physics module's array state.

    Arrays created while a concrete explicit mesh is set (``jax.set_mesh`` in
    pyses' backend) carry that mesh on their aval; used as closure constants
    inside an ``auto_axes`` physics region they raise "context mesh should
    match the aval mesh". Recreating them with the mesh temporarily unset
    gives the mesh-less, uncommitted typing that every pre-mesh module
    constant has — freely usable in both explicit and auto regions. In-place
    via ``nnx.update``; dtypes are preserved (the float32 physics cast in
    ``_build_initial_physics_carry`` is unaffected).
    """
    import numpy as np
    from flax import nnx

    graphdef, state = nnx.split(physics)
    prev_mesh = jax.sharding.get_mesh()
    jax.set_mesh(None)
    try:
        state = tree_map(
            lambda x: jnp.asarray(np.asarray(x))
            if isinstance(x, jax.Array) else x,
            state,
        )
    finally:
        jax.set_mesh(prev_mesh)
    nnx.update(physics, state)


def _op_split_trajectory(
    step_fn: Callable[[Any, Any], tuple[Any, Any]],
    initial_physics_state: Any,
    empty_diagnostics: Any,
    outer_steps: int,
    inner_steps: int,
    post_process_fn: Callable[[Any, Any], Any] = lambda x, ps: x,
    output_averages: bool = False,
    observe_fn: Callable[[Any, Any], Any] | None = None,
    observer_xs: Any = None,
    snapshot_stride: int = 0,
    snapshot_fields: tuple[str, ...] = (),
) -> Callable[[Any], tuple[Any, Any, Any, Any]]:
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
        observe_fn: Optional per-``dt`` virtual-observation sampler
            ``(physics_state_next, xs_slice) -> samples`` (see
            :mod:`jcm.observers`). Its output is emitted as inner-scan ``ys``
            every timestep — the only channel that survives at ``dt``
            resolution rather than being decimated to ``save_interval``.
        observer_xs: Pytree of per-step sampling tables whose leaves have a
            leading axis of ``outer_steps * inner_steps``; sliced per ``dt``
            and fed to ``observe_fn``. Required when ``observe_fn`` is set.

    Returns:
        A function ``initial_state -> (final_state, final_physics_state,
        saved_trajectory, observations)`` where ``saved_trajectory`` has a
        leading axis of length ``outer_steps`` and ``observations`` (``None``
        without ``observe_fn``) has leaves with a leading axis of
        ``outer_steps * inner_steps``. ``final_physics_state`` is the cross-step carry
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
    have_observers = observe_fn is not None
    # Interval-instantaneous snapshots of selected 2-D diagnostics
    # (AeroCom 3-hourly output, jax-gcm#586): a strided buffer rides the
    # inner-scan carry — the scan's ys are structurally one-per-step, so a
    # cheaper cadence has to accumulate in the carry exactly the way the
    # interval mean does. Averaged mode only: the snapshot cadence divides
    # the save interval, and the snapshot-mode path IS already
    # instantaneous at its own cadence.
    have_snapshots = snapshot_stride > 0 and len(snapshot_fields) > 0
    if have_snapshots:
        if not output_averages:
            raise ValueError(
                "snapshot_fields need output_averages=True; the snapshot "
                "path is already instantaneous at save_interval.")
        if inner_steps % snapshot_stride:
            raise ValueError(
                f"snapshot stride {snapshot_stride} must divide the "
                f"{inner_steps} inner steps per save interval.")
        n_snaps = inner_steps // snapshot_stride

        def _resolve_snap(diag, name):
            if "." in name:
                head, _, attr = name.partition(".")
                return getattr(diag[head], attr)
            return diag[name]

        empty_snaps = {}
        for name in snapshot_fields:
            tmpl = _resolve_snap(empty_diagnostics, name)
            # Horizontal-only fields: flat (ncols,) under vectorized
            # physics, (nlon, nlat) under grid-layout physics (SPEEDY).
            if tmpl.ndim not in (1, 2):
                raise ValueError(
                    f"snapshot field {name!r} has shape {tmpl.shape}; only "
                    "2-D horizontal fields are snapshot-able — 3-D fields "
                    "at 3-hourly cadence belong in a dedicated run.")
            empty_snaps[name] = jnp.zeros(
                (n_snaps,) + tmpl.shape, dtype=jnp.result_type(tmpl.dtype, jnp.float32))

    # The saved-trajectory physics payload must not carry the per-step
    # ``_sampler_state`` snapshot (state fields the StateSampler term
    # publishes for the observers) — that would duplicate the dynamics
    # fields in every saved frame. It stays in the *carry* (the scan needs a
    # structure-stable pytree and the observers read it every ``dt``) but is
    # stripped from what gets saved.
    def _strip_sampler(diag):
        if isinstance(diag, dict) and "_sampler_state" in diag:
            return {k: v for k, v in diag.items() if k != "_sampler_state"}
        return diag

    def _averaged_outer_step():
        @jax.checkpoint
        def inner_step(carry, xs):
            i_inner, obs_x = xs
            x, physics_state, x_sum, diag_sum, snaps = carry
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
            obs = observe_fn(physics_state_next, obs_x) if have_observers else None
            if have_snapshots:
                # Write the POST-step instantaneous value into its slot on
                # stride boundaries; off-stride steps rewrite the slot with
                # its own value (a no-op) so the carry stays branch-free.
                is_snap = ((i_inner + 1) % snapshot_stride) == 0
                idx = jnp.clip((i_inner + 1) // snapshot_stride - 1,
                               0, n_snaps - 1)
                new_snaps = {}
                for name in snapshot_fields:
                    val = _resolve_snap(physics_state_next, name).astype(
                        snaps[name].dtype)
                    slot = jnp.where(is_snap, val, snaps[name][idx])
                    new_snaps[name] = snaps[name].at[idx].set(slot)
                snaps = new_snaps
            return (x_next, physics_state_next, x_sum, diag_sum, snaps), obs

        def outer_step(carry, obs_x_frame, empty_sum, empty_diag_sum):
            x, physics_state = carry
            init = (x, physics_state, empty_sum, empty_diag_sum,
                    empty_snaps if have_snapshots else {})
            inner_xs = (jnp.arange(inner_steps), obs_x_frame)
            (x_next, ps_next, x_sum, diag_sum, snaps), obs = jax.lax.scan(
                inner_step, init, inner_xs, length=inner_steps,
            )
            averaged_state = tree_map(lambda s: s / inner_steps, x_sum)
            preds = post_process_fn(averaged_state, ps_next)
            preds = preds.replace(physics=_strip_sampler(diag_sum))
            return (x_next, ps_next), (preds, obs, snaps)

        return outer_step

    def _snapshot_outer_step():
        @jax.checkpoint
        def inner_step(carry, obs_x):
            x, physics_state = carry
            x_next, physics_state_next = step_fn(x, physics_state)
            obs = observe_fn(physics_state_next, obs_x) if have_observers else None
            return (x_next, physics_state_next), obs

        def outer_step(carry, obs_x_frame):
            (x_final, ps_final), obs = jax.lax.scan(
                inner_step, carry, obs_x_frame, length=inner_steps,
            )
            # Save the carried physics state alongside the dynamics state.
            # Calling ``post_process_fn`` with ``ps_final`` lets snapshot
            # diagnostics reflect the sub-cycled radiation cache / TKE
            # memory the dycore actually consumed — recomputing physics at
            # save time with a freshly-seeded carry would zero out radiation
            # on non-radiation outer steps (default 2-hour
            # ``radiation_interval``).
            return (x_final, ps_final), (post_process_fn(x_final, ps_final), obs, {})

        return outer_step

    def integrate(x_initial):
        if output_averages:
            empty_sum = tree_map(jnp.zeros_like, x_initial)
            # Cast accumulator leaves to float so that ``acc + new / N`` doesn't
            # promote dtype mid-scan — jax.lax.scan rejects type changes in the
            # carry. ``zeros_like`` (not ``zeros(shape)``) so any device
            # sharding on the template survives into the accumulator — under
            # an explicit mesh a replicated accumulator could not absorb the
            # column-sharded per-step diagnostics.
            empty_diag_sum = tree_map(
                lambda x: jnp.zeros_like(x, dtype=float),
                empty_diagnostics,
            )
            outer_step_fn = _averaged_outer_step()
            outer_step = lambda c, xs: outer_step_fn(
                c, xs, empty_sum, empty_diag_sum,
            )
        else:
            outer_step = _snapshot_outer_step()

        # Observer sampling tables enter the scans as ``xs``: the leading
        # per-``dt`` axis is folded to (outer, inner, ...) so the outer scan
        # slices whole frames and the inner scan slices single steps.
        scan_xs = None
        if have_observers:
            scan_xs = tree_map(
                lambda a: a.reshape(
                    (outer_steps, inner_steps) + a.shape[1:]),
                observer_xs,
            )

        (x_final, ps_final), (preds, observations, snapshots) = jax.lax.scan(
            outer_step,
            (x_initial, initial_physics_state),
            scan_xs, length=outer_steps,
        )
        if have_observers:
            # (outer, inner, ...) -> (n_steps, ...) for the per-dt channel.
            observations = tree_map(
                lambda a: a.reshape((-1,) + a.shape[2:]), observations,
            )
        if have_snapshots:
            # (outer, n_snaps, ...) -> (total_snaps, ...).
            snapshots = tree_map(
                lambda a: a.reshape((-1,) + a.shape[2:]), snapshots,
            )
        return x_final, ps_final, preds, observations, snapshots

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
                 time_step: float | None = None,
                 terrain: TerrainData = None,
                 physics: Physics = None,
                 start_date: jdt.Datetime | None = None,
                 calendar: str = "365_day",
                 observers=(),
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
            time_step: Model time step in minutes. When ``None`` (the default)
                it is resolved from a single source of truth:

                * with an explicit ``dycore``, the dycore's own
                  ``dt_seconds`` is adopted — whoever constructs the dycore
                  owns the timestep, and physics/dates/saves follow it, so
                  the two can never silently disagree;
                * when the Model builds its own dycore from ``coords``, the
                  active physics is consulted via
                  :meth:`Physics.stable_time_step_minutes` (aggregated over
                  terms by ``ComposablePhysics``), so grid-dependent
                  explicit-tendency stability limits — e.g. SPEEDY's surface
                  drag in the thin bottom sigma layer of high-``nlev`` grids,
                  see docs/source/design/speedy_variable_levels.md — shrink the default below
                  the historical 30 minutes only where needed. Physics
                  without such a limit (ECHAM, Held-Suarez, ...) keeps
                  30 minutes; SPEEDY's standard 7/8-level runs sit on the
                  stable plateau and keep 30 minutes exactly.

                Pass an explicit value to override; with an explicit dycore
                the value must match ``dycore.dt_seconds`` (a mismatch
                raises, since dynamics would otherwise advance by a
                different step than physics/dates/saves assume).
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
            observers: Sequence of :class:`jcm.observers.Observer` — virtual
                observation operators sampled every ``dt`` (stations, moving
                platforms, solar-time swaths). Fixed at construction, like
                ``physics`` (``_run_from_state`` treats the Model as a static
                jit argument, so mutating them later would not retrace).
                When present, a :class:`~jcm.physics.diagnostics.
                state_sampler.StateSampler` term is appended to the physics
                automatically so state fields are sampleable. Results ride
                on :class:`~jcm.predictions.ModelPredictions` — see
                :meth:`~jcm.predictions.ModelPredictions.observation_datasets`.
            log_level: Logging verbosity level.

        """
        logging.getLogger().setLevel(log_level)
        self.calendar = calendar
        # Default built HERE, not as a def-time default: a def-time
        # ``jdt.to_datetime(...)`` freezes its array dtypes at import time,
        # and a backend that enables jax_enable_x64 later (pySES does,
        # process-wide) then mixes 32-bit datetime internals with 64-bit
        # arithmetic inside the checkpointed scan — an MLIR verifier error
        # under JAX >= 0.8 (ordering-dependent: only bites when jcm.model
        # is imported before the x64 flag flips).
        self.start_date = (start_date if start_date is not None
                           else jdt.to_datetime('2000-01-01'))

        self.physics = physics if physics is not None else speedy_physics()
        time_step = self._resolve_time_step_minutes(time_step, dycore, coords)
        self.dt_si = (time_step * units.minute).to(units.second)

        self.observers = tuple(observers)
        # Parameters as bound into each compiled trace, keyed by a trace
        # id that the executable itself carries back (#732). See the
        # capture in ``_run_from_state`` for why the live values will not do.
        self._traced_params: dict = {}
        self._trace_counter: int = 0
        if len({obs.name for obs in self.observers}) != len(self.observers):
            raise ValueError("Observer names must be unique.")
        if self.observers:
            # Observers sample state fields / vertical coordinates through
            # the diagnostics dict; the StateSampler term publishes them.
            from jcm.physics.diagnostics.state_sampler import StateSampler
            has_sampler = any(
                getattr(t, "name", "") == StateSampler.name
                for t in getattr(self.physics, "terms", ())
            )
            if not has_sampler:
                if not hasattr(self.physics, "terms"):
                    raise ValueError(
                        "observers require a composable physics package (the "
                        "StateSampler term is appended to physics.terms).")
                self.physics = self.physics + StateSampler()

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

        # Satisfy, then validate, the dycore-field contract at construction:
        # every field a term declares in ``requires_dycore_fields`` must be
        # supplied by the backend (physics_field_names) or an upstream term's
        # ``provides`` — settled here, not deep inside the first traced step.
        #
        # A backend that CAN produce a required field but has its provider
        # switched off is turned on rather than rejected: the provider flags
        # (``compute_omega``, ``compute_frontogenesis``) are pure cost knobs,
        # and a term that declares a field cannot function without it, so
        # there is no configuration in which "off" is the right answer. The
        # alternative — making every caller of ``Model(physics=echam_physics())``
        # hand-construct ``DinosaurDycore(compute_omega=True)`` — is a tax
        # that buys nothing, and the silent-fallback alternative is worse
        # still (a term losing an input it declared, invisibly). The Hydra
        # path has resolved providers this way since jax-gcm#409
        # (``runners._want_omega``); this makes the policy uniform for
        # library callers, and leaves genuine incapability (a backend with no
        # such provider at all, e.g. pySES and omega — #698) as the only
        # failure.
        required = tuple(getattr(self.physics, "required_dycore_fields",
                                 lambda: ())())
        for field in required:
            if field in self.dycore.physics_field_names():
                continue
            flag = f"compute_{field}"
            if hasattr(self.dycore, flag):
                setattr(self.dycore, flag, True)
        self._dycore_field_names = tuple(self.dycore.physics_field_names())
        missing = [f for f in required if f not in self._dycore_field_names]
        if missing:
            raise ValueError(
                f"The composed physics requires dycore-supplied fields "
                f"{missing}, but this backend provides "
                f"{list(self._dycore_field_names) or 'none'}, and has no "
                f"compute_<field> provider to switch on. Add a physics-side "
                "provider term whose ``provides`` names the field, or turn "
                "off the feature that declares it — each requiring term's "
                "docstring names its switch (e.g. TiedtkeConvection's "
                "``cu_lmfmid`` for ``omega``)."
            )

        for observer in self.observers:
            observer.cache_grid(self.coords)

        self.physics.cache_coords(self.coords)
        if _ambient_explicit_axis() is not None:
            # Multi-device explicit mesh (pySES): the physics' cached arrays
            # (hybrid tables, per-column lat/lon, parameter scalars) were just
            # created under the ambient explicit mesh and would clash as
            # explicit-typed closure constants inside the auto-mode physics
            # region (see ``_ambient_explicit_axis``). Recreate them mesh-less
            # (the typing every pre-mesh module constant has) so they behave
            # as ordinary replicated constants there, while staying concrete
            # Python values for trace-time configuration reads.
            _neutralize_mesh_typing(self.physics)
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

    def __repr__(self) -> str:
        """One-line summary: backend, grid, levels, dt, physics terms (#322)."""
        horiz = getattr(self.coords, "horizontal", None)
        shape = getattr(horiz, "nodal_shape", None)
        grid = f"{shape[0]}x{shape[1]}" if shape is not None else "?"
        layers = getattr(getattr(self.coords, "vertical", None), "layers", "?")
        terms = getattr(self.physics, "terms", None)
        physics = (
            "[" + ", ".join(getattr(t, "name", type(t).__name__)
                            for t in terms) + "]"
            if terms is not None else type(self.physics).__name__
        )
        return (
            f"{type(self).__name__}(dycore={type(self.dycore).__name__}, "
            f"grid={grid}, levels={layers}, dt={float(self.dt_si.m):g}s, "
            f"physics={physics})"
        )

    # Historical default model time step; also the ceiling for physics-
    # suggested stable steps (a physics limit can only shrink the default,
    # never silently enlarge it).
    _DEFAULT_TIME_STEP_MINUTES = 30.0

    def _resolve_time_step_minutes(self, time_step, dycore, coords) -> float:
        """Resolve the model time step (minutes) from a single source of truth.

        The timestep is needed by both the dycore (dynamics integrator) and
        the Model (physics cadence, dates, save intervals), but the two are
        supplied independently — this method is the one place their
        consistency is enforced:

        * Explicit ``time_step``: used as given. If an explicit ``dycore``
          was also supplied, the two must agree (the dycore bakes its step
          into its integrator at construction, so a silent mismatch would
          advance dynamics by a different ``dt`` than physics/dates assume —
          raise instead).
        * ``time_step is None`` with an explicit ``dycore``: adopt the
          dycore's ``dt_seconds``. Whoever constructed the dycore owns the
          step.
        * ``time_step is None`` on the ``coords`` path (Model builds the
          dycore itself): consult the active physics'
          :meth:`Physics.stable_time_step_minutes` — the numerically binding
          constraint is a property of the physics scheme (e.g. SPEEDY's
          explicit surface drag in a thin bottom sigma layer), so the scheme
          that imposes it owns the limit. The default is the historical
          30 minutes, shrunk to the physics limit where one applies.
        """
        dycore_dt_seconds = (
            getattr(dycore, "dt_seconds", None) if dycore is not None else None
        )
        if time_step is not None:
            if (dycore_dt_seconds is not None
                    and abs(time_step * 60.0 - float(dycore_dt_seconds)) > 1e-6):
                raise ValueError(
                    f"time_step={time_step} min conflicts with the explicit "
                    f"dycore's dt_seconds={float(dycore_dt_seconds)} "
                    f"({float(dycore_dt_seconds) / 60.0} min). The dycore "
                    "bakes its step into its integrator at construction; "
                    "either drop time_step= (the Model adopts the dycore's "
                    "step) or rebuild the dycore with the intended dt_seconds."
                )
            return float(time_step)
        if dycore is not None:
            if dycore_dt_seconds is not None:
                return float(dycore_dt_seconds) / 60.0
            return self._DEFAULT_TIME_STEP_MINUTES
        limit = self.physics.stable_time_step_minutes(coords)
        if limit is None:
            return self._DEFAULT_TIME_STEP_MINUTES
        return min(self._DEFAULT_TIME_STEP_MINUTES, float(limit))

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
        ``hdiff`` chain. The dynamics→physics gridpoint projection
        (:meth:`DynamicalCore.to_physics_state`) is where any dycore-side tracer
        cleaning (e.g. a spectral core's mass-conserving positivity filter)
        happens, so the step body stays agnostic to it.
        """
        # Align the forcing to the model's working float precision once, up
        # front, before the scan time-slices it. ``ForcingData`` is often built
        # before the MAM4-JAX core enables ``jax_enable_x64`` (so its arrays are
        # float32), whereas the working model state is float64; a per-step
        # float32 forcing feeding the physics diagnostics would then clash with
        # the float64 scan carry ("carry input/output types differ"). Casting
        # here only ever upcasts (x64 path) or is a no-op (non-x64, where forcing
        # and state share float32), so the selection time axis is unaffected.
        work_dtype = jnp.zeros(()).dtype
        forcing = jax.tree_util.tree_map(
            lambda x: x.astype(work_dtype)
            if (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating))
            else x,
            forcing,
        )

        def step(state, physics_state):
            date = self._date_from_sim_time(self.dycore.sim_time(state))
            forcing_now = forcing.select(date, calendar=self.calendar)
            # The scopes opened here and in ComposablePhysics's term loop label
            # this step's HLO, so that a profiler trace can be split into
            # dynamics / bridge / per-term cost. See jcm.profiling.
            with profiling.scope(profiling.BRIDGE_TO_PHYSICS):
                physics_state_grid = self.dycore.to_physics_state(state)
                if self._dycore_field_names:
                    # Dycore-supplied diagnostic fields (frontogenesis, ...):
                    # re-injected every step under a plumbing key that
                    # ComposablePhysics strips from its output, so the scan
                    # carry's pytree structure is unaffected (the codex-P1
                    # lesson from the observers work: anything that rides the
                    # carry must exist in the construction-time template).
                    extra = self.dycore.physics_fields(state,
                                                       physics_state_grid)
                    physics_state = {**physics_state,
                                     "_dycore_fields": extra}
            call = partial(
                compute_physics_step_gridpoint,
                physics=self.physics, time_step=self.dt_si.m,
            )
            # Scope the physics call as a whole. It ENCLOSES the per-term
            # scopes, so under the innermost-wins attribution rule it retains
            # only the driver's own overhead: verification, the column
            # reshapes and the tendency accumulation between terms. Applied to
            # the callable rather than at the call sites so that the sharding
            # branch below stays as it was.
            call = profiling.scoped(call, profiling.BRIDGE_TO_DYNAMICS)
            args = (physics_state_grid, forcing_now, self.terrain,
                    physics_state)
            axis = _ambient_explicit_axis()
            if axis is None:
                physics_tendency, new_physics_state = call(*args)
            else:
                # Multi-device explicit mesh (pySES element sharding): run the
                # physics under auto sharding semantics — see the module-level
                # note at ``_ambient_explicit_axis``. The physics module's own
                # cached arrays were made mesh-less at construction
                # (``_neutralize_mesh_typing``), so they pass as ordinary
                # replicated closure constants; the array ARGUMENTS are
                # re-typed by auto_axes itself.
                from jax.sharding import auto_axes

                ncols = physics_state_grid.temperature.shape[-1]
                # The shape-only trace must not see the explicit shardings —
                # tracing the unwrapped physics with explicit-typed inputs
                # hits the very type errors auto_axes exists to avoid — so
                # eval_shape runs on bare ShapeDtypeStructs (replicated
                # typing). Only the output SHAPES are consumed.
                strip = lambda a: (jax.ShapeDtypeStruct(jnp.shape(a), a.dtype)  # noqa: E731
                                   if hasattr(a, "dtype") else a)
                out_shapes = jax.eval_shape(call, *tree_map(strip, args))
                specs = _column_partition_specs(out_shapes, ncols, axis)
                physics_tendency, new_physics_state = auto_axes(
                    call, axes=axis, out_sharding=specs,
                )(*args)
            with profiling.scope(profiling.DYNAMICS):
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
        if isinstance(physics_state, dict) and "_sampler_state" in physics_state:
            # The StateSampler's per-step state snapshot exists only for the
            # per-dt observer channel; saving it would duplicate the dynamics
            # fields in every frame.
            physics_state = {
                k: v for k, v in physics_state.items() if k != "_sampler_state"
            }
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
            carry = {**template, **initial_carry}
        else:
            # Explicit ``is None`` check: ``initial_carry or template`` would
            # trigger ``bool(carry)`` and raise an ambiguous-truth
            # ``ValueError`` if a ``Physics`` subclass returns a JAX array
            # (or any object with non-scalar truth semantics).
            carry = template if initial_carry is None else initial_carry
        # Dycores that run their dynamics at a different precision than the
        # physics (the pySES CAM-SE backend: float64 dynamics under
        # jax_enable_x64, float32 physics) expose ``physics_dtype``; the
        # scan carry must match the dtype the per-step compute produces, or
        # iteration 1 fails to type-check. The template above was built at
        # the process default, so cast its float leaves down here. Backends
        # without the attribute (dinosaur) are untouched.
        physics_dtype = getattr(self.dycore, "physics_dtype", None)
        if physics_dtype is not None:
            carry = jax.tree.map(
                lambda x: x.astype(physics_dtype)
                if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
                else x,
                carry,
            )
        return carry

    def _get_op_split_integrate_fn(
        self,
        step_fn,
        outer_steps,
        inner_steps,
        post_process_fn,
        output_averages,
        observer_xs=(),
        snapshot_stride=0,
        snapshot_fields=(),
    ):
        """Integrate-fn builder for the operator-split path.

        Returns a closure ``(state, initial_physics_state) -> (final_state,
        final_physics_state, predictions)``. The running-mean accumulator
        template comes from :meth:`Physics.get_empty_data` — a zero-filled
        snapshot of the dict ``compute_tendencies`` produces, which is exactly
        the pytree structure the scan carries.
        """
        template = self.physics.get_empty_data(self.coords)

        observe_fn = None
        if self.observers:
            observers = self.observers

            def observe_fn(physics_state_next, obs_x):
                return tuple(
                    obs.sample(physics_state_next, x)
                    for obs, x in zip(observers, obs_x)
                )

        def _integrate_fn(state, initial_physics_state):
            axis = _ambient_explicit_axis()
            empty_diagnostics = template
            if axis is not None:
                # Explicit mesh: the per-step physics outputs are
                # column-sharded, and the scan carry type must be invariant —
                # shard the initial carry (which may be a host-built template
                # or a checkpoint-restored numpy pytree) and the averaging
                # template the same way up front.
                ncols = int(self.coords.horizontal.nodal_shape[-1])
                initial_physics_state = _reshard_columns(
                    initial_physics_state, ncols, axis)
                empty_diagnostics = _reshard_columns(template, ncols, axis)
            trajectory = _op_split_trajectory(
                step_fn=step_fn,
                initial_physics_state=initial_physics_state,
                empty_diagnostics=empty_diagnostics,
                outer_steps=outer_steps,
                inner_steps=inner_steps,
                post_process_fn=post_process_fn,
                output_averages=output_averages,
                observe_fn=observe_fn,
                observer_xs=observer_xs if self.observers else None,
                snapshot_stride=snapshot_stride,
                snapshot_fields=snapshot_fields,
            )
            return trajectory(state)

        return _integrate_fn

    def _remember_traced_params(self, trace_id: int, record: dict) -> None:
        """Store one trace's parameter record, oldest evicted past the cap.

        Bounded because nothing else here can be: a model that keeps
        meeting new input shapes retraces indefinitely, and
        ``jax.clear_caches()`` does not reach this dict (#733 review).
        Evicting the oldest degrades that executable's record to empty if
        it is ever run again, which is the safe direction — a missing
        record, never another executable's values.
        """
        self._traced_params[trace_id] = record
        while len(self._traced_params) > _MAX_TRACED_PARAM_RECORDS:
            self._traced_params.pop(next(iter(self._traced_params)))

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 8, 9))  # Note: changing fields assumed static won't propagate.
    def _run_from_state(self,
                        initial_state,
                        initial_physics_state: Any,
                        forcing: ForcingData,
                        save_interval=10.0,
                        total_time=120.0,
                        output_averages=False,
                        observer_xs=(),
                        snapshot_stride=0,
                        snapshot_fields=(),
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
        # Capture the parameters HERE, at trace time, not from the live
        # module afterwards (#732). ``self`` is a static argument, so the
        # parameter values are baked into this executable as constants and
        # mutating one in place afterwards changes nothing the compiled
        # function does (see the note on the decorator). Reading the module
        # at the model-to-user handoff would therefore stamp a trajectory
        # with values that did not produce it. On a cache hit this does not
        # re-run, which is right: the reused executable still holds the
        # parameters captured at its own trace.
        trace_id = self._trace_counter
        self._trace_counter += 1
        try:
            self._remember_traced_params(
                trace_id, provenance.describe_params(self.physics))
        except Exception:  # noqa: BLE001 — provenance never fails a run
            logger.warning("provenance: trace-time parameter capture failed",
                           exc_info=True)

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
            observer_xs=observer_xs,
            snapshot_stride=snapshot_stride,
            snapshot_fields=snapshot_fields,
        )
        (final_dycore_state, final_physics_state, predictions, observations,
         snapshots) = integrate(initial_state, initial_physics_state)

        # The id rides back as a traced constant, so a cache hit returns
        # the id of the executable that actually ran. Keying the store on
        # the static arguments instead was not enough: one static
        # signature can own several executables (a forcing of a different
        # shape or dtype retraces), and the later trace overwrote the
        # earlier one's record.
        return (final_dycore_state, final_physics_state,
                predictions.replace(times=times), observations, snapshots,
                jnp.int32(trace_id))

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
                                  snapshot_interval=None,
                                  snapshot_variables=(),
    ):
        """Lower-level ``run_from_state`` that exposes the cross-step physics carry.

        ``snapshot_interval`` / ``snapshot_variables`` (jax-gcm#586) add an
        interval-INSTANTANEOUS output stream of selected 2-D diagnostics
        alongside the interval-mean fields: e.g. 3-hourly ``clt``/``lwp``
        snapshots riding a monthly-mean AeroCom run. Averaged mode only;
        the snapshot interval must divide ``save_interval``. Fields are
        top-level diagnostics keys or dotted struct fields
        (``"radiation.toa_sw_up"``); retrieve the stream with
        :meth:`ModelPredictions.snapshot_dataset`.
        """
        save_interval_days = parse_duration_days(save_interval, calendar=self.calendar)
        total_time_days = parse_duration_days(total_time, calendar=self.calendar)
        snapshot_stride = 0
        if snapshot_interval is not None and snapshot_variables:
            snap_days = parse_duration_days(snapshot_interval,
                                            calendar=self.calendar)
            dt_days = self.dt_si.to(units.day).m
            snapshot_stride = int(round(snap_days / dt_days))
            if abs(snapshot_stride * dt_days - snap_days) > 1e-9:
                raise ValueError(
                    f"snapshot_interval {snapshot_interval!r} is not a "
                    f"multiple of the model timestep ({self.dt_si.m} s).")
        if initial_physics_state is None:
            initial_physics_state = self._build_initial_physics_carry()

        # Build the observers' per-step sampling tables for this window
        # (offline numpy; horizontal weights are resolved here once and only
        # the vertical interpolation remains state-dependent in the scan).
        # Absolute start time in days since 1970 — the same axis the
        # trajectory ``times`` use — so chunked run/resume sequences slice
        # the observation tracks consistently.
        observer_xs = ()
        obs_t0_days = None
        if self.observers:
            dt_days = self.dt_si.to(units.day).m
            n_steps = (int(total_time_days / save_interval_days)
                       * int(save_interval_days / dt_days))
            obs_t0_days = float(
                self.start_date.delta.days
                + float(jax.device_get(self.dycore.sim_time(initial_state)))
                / 86400.0
            )
            observer_xs = tuple(
                obs.prepare(obs_t0_days, float(self.dt_si.m), n_steps)
                for obs in self.observers
            )

        (final_dycore_state, final_physics_state, predictions, observations,
         snapshots, trace_id) = self._run_from_state(
                initial_state, initial_physics_state, forcing,
                save_interval_days, total_time_days,
                output_averages, observer_xs,
                snapshot_stride, tuple(snapshot_variables),
        )
        return (
            final_dycore_state,
            final_physics_state,
            ModelPredictions(
                predictions, self.coords, self.physics,
                dycore=self.dycore,
                params=self._traced_params.get(int(trace_id)),
                observations=observations,
                observers=self.observers,
                obs_t0_days=obs_t0_days,
                obs_dt_seconds=float(self.dt_si.m),
                snapshots=snapshots,
                snapshot_variables=tuple(snapshot_variables),
                snapshot_interval_days=(
                    snapshot_stride * self.dt_si.to(units.day).m
                    if snapshot_stride else None),
            ),
        )

    def resume(self,
               forcing: ForcingData = None,
               save_interval=10.0,
               total_time=120.0,
               output_averages=False,
               snapshot_interval=None,
               snapshot_variables=(),
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
            snapshot_interval=snapshot_interval,
            snapshot_variables=snapshot_variables,
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
            snapshot_interval=None,
            snapshot_variables=(),
            initial_physics_state: Any = None,
    ) -> ModelPredictions:
        """Set the initial state and run the full simulation forward in time.

        ``initial_state`` may be:
            * ``None`` — the dycore builds its own default initial state.
            * a :class:`PhysicsState` — gridpoint state, projected onto the
              dycore via :meth:`DynamicalCore.initial_state`.
            * a dycore-native state (e.g. ``primitive_equations.State`` for
              the dinosaur backend) — used directly.

        ``initial_physics_state`` seeds the cross-step physics carry (the
        radiation sub-cycle cache, prior-step TKE, …) that :meth:`resume`
        threads through the integration. When ``None`` (the default),
        :meth:`bootstrap_state` builds a fresh carry from the composed physics
        + coords. When supplied — a warm start off a donor checkpoint whose
        carry we want to preserve rather than reset — it *replaces* the
        freshly-built carry after ``bootstrap_state`` and before the first
        ``resume``. It must be a carry that structurally matches what
        :meth:`_build_initial_physics_carry` builds for this model (same
        physics composition + coords): ``resume`` uses the freshly-built carry
        as the pytree template and unflattens the checkpoint against it, so a
        carry from a different composition would not line up. In practice this
        is the value returned by :func:`jcm.initial_states.checkpoint_state`,
        which loads it through that same template.
        """
        self.bootstrap_state(initial_state)
        if initial_physics_state is not None:
            # ``bootstrap_state`` just built a fresh carry; a warm start wants
            # the donor's restored carry instead so sub-cycled radiation /
            # prior-step TKE don't reset at the run seam. Replace it before the
            # first ``resume`` threads ``self._final_physics_state`` in.
            self._final_physics_state = initial_physics_state
        return self.resume(
            forcing=forcing, save_interval=save_interval,
            total_time=total_time, output_averages=output_averages,
            snapshot_interval=snapshot_interval,
            snapshot_variables=snapshot_variables,
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
