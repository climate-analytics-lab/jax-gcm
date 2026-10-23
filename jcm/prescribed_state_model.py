"""Diagnose physics tendencies from a prescribed atmospheric state series.

.. note:: Carry-resident physics state (jax-gcm#623): a composition that
   declares cross-step carry slots (TTE-TKE turbulence; the carry-stored
   cloud-borne aerosol phase, #602 item 3, the JAM storage) has that state
   threaded SEQUENTIALLY across the prescribed times with ``lax.scan`` —
   each evaluated time's physics genuinely sees the carry the physics
   produced diagnosing the previous prescribed time, the same flow a live
   run has. The threaded carry is the FINALIZED one — the scan step runs
   ``physics.compute_tendencies`` and then
   :func:`jcm.physics_interface.finalize_physics_tendencies_and_carry`, the
   SAME water-positivity verification a live :class:`~jcm.model.Model` run
   applies via ``compute_physics_step_gridpoint`` — so a carry-reading term
   (Tiedtke's ``_prev_step.q_tendency``) sees the applied, post-cap value a
   live run would have handed it, not ``compute_tendencies``' raw,
   provisional one. The per-time TENDENCY this module reports stays raw
   (unverified) in both carry modes, so it means the same thing regardless
   of whether the physics has a carry. What differs from a live run is the
   SEED: with no donor carry supplied (``run(..., initial_physics_state=
   ...)``), the carry is cold-started from ``physics.initial_carry_state``
   at the FIRST prescribed time rather than inherited from any true prior
   state, so carry-dependent terms are in a spin-up for the first few
   times — read a saved run's own ``jam_cloud_borne.*`` output for
   cloud-borne analysis that needs the exact record rather than a
   re-diagnosis. Physics with no carry slots (Held-Suarez, SPEEDY) is
   unaffected: every time is already independent of every other, so it
   keeps the historical ``vmap`` path. Construction warns once, naming any
   slots a non-dict (raw, non-``ComposablePhysics``) carry cannot be
   threaded for — see :meth:`PrescribedStateModel._classify_carry`.


``PrescribedStateModel`` computes physics tendencies for a prescribed state
time series, either independently per time (``vmap``, when the physics has
no cross-step carry) or sequentially (``lax.scan``, threading the carry,
when it does) — see the module note above. There is no tracer evolution in
either case: states are externally prescribed, not advanced by the
physics' own tendencies. Useful for offline diagnostics, validation
against reference data, and re-diagnosing carry-free physics independently
per state (e.g. for a lookup table). For tracer evolution (where each
step's STATE, not just its carry, depends on the previous one), use
``SingleColumnModel`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math
import warnings

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
from jax.tree_util import tree_map

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.date import DateData, SECONDS_PER_DAY, to_datetime
from jcm.forcing import ForcingData, default_forcing
from jcm.physics_interface import (
    Physics,
    PhysicsState,
    PhysicsTendency,
    build_initial_physics_carry,
    finalize_physics_tendencies_and_carry,
    verify_state,
)
from jcm.terrain import TerrainData


@tree_math.struct
class PrescribedStatePredictions:
    """Container for ``PrescribedStateModel.run`` outputs.

    Attributes:
        states: The prescribed atmospheric state time series.
        tendencies: Physics tendencies for each step.
        physics_data: Per-step diagnostics dict from the physics package.
        times: Times in days since the start of the run.
        a_boundaries_pa: TOA-first hybrid ``a`` interface table [Pa], as
            :func:`jcm.cf_metadata.hybrid_boundaries` returns it. ``None`` when
            the predictions were built without a vertical coordinate table
            (e.g. constructed directly in a test); ``to_xarray`` then falls
            back to a bare (flipped) ``level`` index.
        b_boundaries: TOA-first hybrid ``b`` interface table, paired with
            ``a_boundaries_pa``.

    """

    states: PhysicsState
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any
    # Vertical coordinate tables threaded through so ``to_xarray`` can write
    # the file convention (#739). Default ``None`` keeps direct constructions
    # (tests, callers that don't have ``coords``) working.
    a_boundaries_pa: Any = None
    b_boundaries: Any = None

    def to_xarray(self):
        """Dump states + tendencies + physics_data into an ``xr.Dataset``.

        Minimal serialiser for offline diagnostics — does NOT use the
        coord-aware ``data_to_xarray`` path that ``ModelPredictions``
        uses (which would require threading ``physics`` and the full
        ``additional_coords`` machinery through the predictions struct).
        Variables are emitted with raw positional dim names
        ``time, level, lon, lat`` and no per-variable units metadata; the
        intent is to support quick NaN-checks and column plots, not
        production climatology.

        The ``level`` axis follows the **surface-first** file convention
        every JCM product uses (``jcm.cf_metadata``, #710, #739). The data
        is built in the top-first physics frame and then finalised through
        :func:`jcm.cf_metadata.finalize_output`, which flips both vertical
        axes surface-first and — when the hybrid ``(a, b)`` tables were
        threaded in (``a_boundaries_pa``/``b_boundaries``, set by
        ``PrescribedStateModel.run``) — attaches real descending sigma
        coordinates and CF attributes. A file written this way round-trips
        through :func:`jcm.utils.load_states_from_xarray`, which detects the
        surface-first orientation and returns a top-first ``PhysicsState``.

        When the tables are absent (a struct built directly, e.g. in a
        test), the ``level`` axis is still flipped so its *direction*
        matches the file convention, but it stays a bare (flipped) integer
        index with no sigma values — orientation is then not self-describing
        and ``load_states_from_xarray`` cannot auto-detect it.

        Use ``ModelPredictions.to_xarray()`` when you need the full
        coord-aware serialisation (units, ``additional_coords``,
        surface-first vertical axes with CF metadata, etc.).
        """
        import numpy as np
        import xarray as xr

        from jcm import cf_metadata
        from jcm import constants as c
        from jcm.predictions import output_time_labels
        from jcm.temporal_aggregation import set_cf_datetime_encoding

        # Hardcoded positional dim names matching the prescribed-mode
        # vmap layout: (time, level, lon, lat) for column variables,
        # (time, lon, lat) for surface fields.
        def _dims_for(arr):
            shape = np.shape(arr)
            if len(shape) == 4:
                return ("time", "level", "lon", "lat")
            if len(shape) == 3:
                return ("time", "lon", "lat")
            if len(shape) == 2:
                return ("time", "level")
            if len(shape) == 1:
                return ("time",)
            return tuple(f"dim_{i}" for i in range(len(shape)))

        data_vars: dict[str, tuple] = {}

        def _add(prefix, struct):
            for k, v in struct.asdict().items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        data_vars[f"{prefix}{k}.{sk}"] = (_dims_for(sv), np.asarray(sv))
                else:
                    data_vars[f"{prefix}{k}"] = (_dims_for(v), np.asarray(v))

        _add("state.", self.states)
        _add("tend.", self.tendencies)

        if isinstance(self.physics_data, dict):
            for k, v in self.physics_data.items():
                if k.startswith("_"):
                    continue
                if hasattr(v, "asdict"):
                    _add(f"diag.{k.lstrip('_')}.", v)
                else:
                    arr = np.asarray(v)
                    data_vars[f"diag.{k}"] = (_dims_for(arr), arr)

        host_times = output_time_labels(self.times)
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": np.asarray(host_times)},
        )

        # Data is still top-first here. Bring it into the file convention.
        if self.a_boundaries_pa is not None:
            ds = cf_metadata.finalize_output(
                ds,
                a_boundaries_pa=self.a_boundaries_pa,
                b_boundaries=self.b_boundaries,
                p0=float(c.p0),
                flip_vertical=True,
            )
        else:
            # No vertical table available: still flip the level axis so its
            # direction matches the surface-first file convention, but without
            # the table the ``level`` axis stays a bare (flipped) integer index
            # with no sigma coordinate values to make the orientation
            # self-describing.
            ds = cf_metadata.orient_surface_first(ds)
            ds = cf_metadata.apply_cf_attributes(ds)
        return set_cf_datetime_encoding(ds, "time")


class PrescribedStateModel:
    """Compute physics tendencies for a prescribed state time series.

    How the physics' cross-step carry is handled depends on whether the
    composition declares any (``Physics.initial_carry_state``), decided once
    at construction (:meth:`_classify_carry`):

    * No carry slots (Held-Suarez, SPEEDY): every time is independent of
      every other, so :meth:`run` evaluates them with ``vmap`` — exact and
      embarrassingly parallel, and unaffected by any of the below.
    * A named (``dict``) carry (TTE-TKE turbulence; JAM's cloud-borne
      aerosol): :meth:`run` threads it SEQUENTIALLY across the prescribed
      times with ``lax.scan``, so each time's physics genuinely sees the
      carry the physics produced diagnosing the PREVIOUS prescribed time —
      the same flow a live run has, including the SAME water-positivity
      verification/finalization step a live run applies (see the module
      note above) rather than the raw, provisional
      ``physics.compute_tendencies`` carry. What differs from a live run is
      the seed: with no donor supplied (``run(..., initial_physics_state=
      ...)``),
      the carry is cold-started from ``initial_carry_state`` at the FIRST
      prescribed time, so carry-dependent terms are in a spin-up for the
      first few times rather than inheriting a true prior state.
      :meth:`run` warns about this every call that cold-starts (i.e. every
      call without a donor).
    * An unnamed (non-``dict``) carry: a raw, non-composable ``Physics`` may
      return any pytree here (``ComposablePhysics`` — "composable physics is
      the only physics API" — always returns a dict, so this is a
      hypothetical fallback, not the common case). Without slot names there
      is no principled way to union it with the structural template a scan
      needs to seed (:func:`jcm.physics_interface.build_initial_physics_carry`'s
      dict-merge branch requires a ``Mapping``), so this carry is NOT
      threaded: ``run`` falls back to ``vmap``, re-diagnosing it from a cold
      start at every time, and construction warns once.

    **The re-diagnosis approximation in "scan" mode.** Threading the carry
    between saved states is exact in the sense that it is the carry the
    composed physics actually produces from those states; it is NOT exact
    in the sense of matching a live run, because a live run advances the
    carry once per ``dt_seconds`` while ``run`` (by default) advances it
    once per SAVED state — typically much coarser (daily snapshots against
    an hourly-to-half-hourly ``dt``). The scan consequently lets TTE-TKE,
    the JAM cloud-borne reservoir, etc. evolve at the save cadence rather
    than the model cadence, which is a coarser approximation of the carry's
    true trajectory the sparser the saved states are. ``run`` warns once
    per call when the saved times' median spacing exceeds ``dt_seconds`` by
    more than 2x (see :meth:`_warn_coarse_save_cadence`).
    ``carry_steps_per_state`` (below) is the knob for improving that
    approximation, NOT a way to eliminate it — see its own docstring. The
    saved run's own carry-derived output (e.g. ``jam_cloud_borne.*``, TKE)
    remains the exact record for analysis that cannot tolerate either
    approximation.

    Args:
        physics: Physics package whose ``compute_tendencies`` is called per step.
        coords: ``CoordinateSystem`` used for grids and ``physics.cache_coords``.
        terrain: Optional ``TerrainData`` boundary conditions; defaults to the
            aquaplanet derived from ``coords``.
        dt_seconds: Physics timestep in seconds (default 1800).
        carry_steps_per_state: In "scan" mode, how many times to evaluate
            the physics at each saved state before recording its tendencies/
            diagnostics (default 1, i.e. unchanged from the historical
            behaviour). With N > 1, the state is held FIXED for N
            evaluations and only the carry is threaded between them, so a
            carry far from where it would equilibrate with that state (TKE,
            the cloud-borne reservoir) gets N-1 extra chances to relax
            toward it before the diagnostics are read — the Nth
            evaluation's tendencies/diagnostics are what ``run`` records,
            the first N-1 are discarded. This does NOT make the result
            exact: it is still a re-diagnosis of ONE state N times, not of
            the N physical ``dt_seconds`` steps a live run would actually
            take between two saved states (whose intermediate states this
            model never sees) — it only lets the carry approach the
            fixed-state equilibrium a live run's carry would be drifting
            toward, which is the right qualitative correction when
            ``dt_seconds`` is much finer than the save interval. A sensible
            value is of order ``save_interval / dt_seconds`` (the number of
            model steps a live run would actually take), capped so the
            per-state cost stays reasonable (e.g. a few tens at most); the
            default of 1 keeps the pre-#623-follow-up cost and behaviour
            for anyone not told to change it. Ignored outside "scan" mode,
            where there is no carry to relax (repeating a carry-free
            evaluation would just recompute the same answer).

    """

    def __init__(
        self,
        physics: Physics,
        coords: CoordinateSystem,
        terrain: TerrainData | None = None,
        dt_seconds: float = 1800.0,
        start_time: jdt.Datetime | str | None = None,
        carry_steps_per_state: int = 1,
    ) -> None:
        """Initialise (see class docstring for argument descriptions).

        ``start_time`` mirrors :class:`jcm.model.Model` so
        each prescribed state can collapse ``TimeSeries`` forcing leaves
        (sea ice, SST, ozone climatology, ...) to the slice valid at that
        state's time before physics is evaluated. Without this,
        from-file forcings stay as ``TimeSeries`` structs and physics
        terms that arithmetic-combine them with plain arrays raise
        ``TypeError: non-tree_math.VectorMixin argument is not a
        scalar``.

        ``carry_steps_per_state`` is a static Python int (an unrolled
        Python loop inside the scan step, not a traced value), so it is a
        compile-time choice, not a per-call one — see the class docstring
        and :meth:`run` for what it does and why it exists (#623 follow-up:
        the save cadence is typically much coarser than ``dt_seconds``).
        """
        self.physics = physics
        self.coords = coords
        self.terrain = terrain if terrain is not None else TerrainData.aquaplanet(coords)
        if not math.isfinite(float(dt_seconds)) or float(dt_seconds) <= 0:
            raise ValueError("dt_seconds must be finite and positive.")
        if float(dt_seconds) != round(float(dt_seconds)):
            raise ValueError("dt_seconds must be representable as whole seconds.")
        self.dt_seconds = int(dt_seconds)
        if int(carry_steps_per_state) != carry_steps_per_state or carry_steps_per_state < 1:
            raise ValueError(
                "carry_steps_per_state must be a positive integer, got "
                f"{carry_steps_per_state!r}.")
        self.carry_steps_per_state = int(carry_steps_per_state)
        self.start_time = to_datetime(
            "2000-01-01" if start_time is None else start_time,
            name="start_time")
        self.physics.cache_coords(coords)
        (self._carry_mode, self._carry_slot_names,
         self._prognostic_carry_slots) = self._classify_carry(coords)
        if self._carry_mode == "unthreadable":
            self._warn_carry_unthreadable()
        # Hand the timestep down to the composable-physics container so its
        # terms read a single ``dt`` source — mirrors the wiring in ``Model``
        # and ``SingleColumnModel``.
        if hasattr(self.physics, "dt_seconds"):
            self.physics.dt_seconds = self.dt_seconds

    def _classify_carry(self, coords) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Decide once how ``self.physics``' cross-step carry is handled (#623).

        Returns ``(mode, slot_names, prognostic_slots)`` — see the class
        docstring for what each ``mode`` ("none" / "scan" / "unthreadable")
        means for :meth:`run`. ``slot_names`` is empty for "none", the
        sorted carry keys for "scan", and ``("an unnamed carry pytree",)``
        for "unthreadable" (there are no keys to report). ``prognostic_slots``
        is the subset of ``slot_names`` the physics declares prognostic
        (``Physics.prognostic_carry_slots``) — the only copy of a physical
        quantity, such as JAM's cloud-borne aerosol, as opposed to a slot a
        term rebuilds every step.
        """
        initial_carry_state = getattr(self.physics, "initial_carry_state", None)
        if initial_carry_state is None:
            return "none", (), ()
        # Only the carry's structure is needed; ``eval_shape`` gets it
        # without allocating the seed (tens of MB for a JAM composition at
        # T63). ``Physics.initial_carry_state`` may return any pytree
        # ``compute_tendencies`` accepts (``ComposablePhysics`` returns a dict
        # of named slots; a raw physics may return an array, a struct or
        # ``None``), so only a mapping has slot names to report.
        carry = jax.eval_shape(lambda: initial_carry_state(coords))
        if not jax.tree_util.tree_leaves(carry):
            return "none", (), ()
        if not isinstance(carry, Mapping):
            return "unthreadable", ("an unnamed carry pytree",), ()
        slots = tuple(sorted(str(key) for key in carry))
        prognostic = tuple(
            slot for slot in getattr(self.physics, "prognostic_carry_slots", tuple)()
            if slot in slots)
        return "scan", slots, prognostic

    def _warn_carry_unthreadable(self) -> None:
        """Warn once for the "unthreadable" (non-``dict``) carry case.

        Every other carry shape is now threaded sequentially by :meth:`run`
        (see the class docstring); this is the one case :meth:`run` still
        falls back to the pre-#623 ``vmap`` behaviour for, because an
        unnamed pytree carry cannot be unioned with the structural template
        a scan needs to seed. So it warns rather than refuses.
        """
        detail = (
            f" {', '.join(self._prognostic_carry_slots)} holds prognostic "
            "state (its only copy lives in the carry), so it is "
            "re-diagnosed as empty."
            if self._prognostic_carry_slots else "")
        warnings.warn(
            "PrescribedStateModel evaluates each time independently, so the "
            "physics' cross-step carry slots "
            f"({', '.join(self._carry_slot_names)}) are not threaded from "
            "one time to the next: each time is re-diagnosed from a cold "
            "start, with every slot at the value its term assumes when no "
            "carry is passed. Slots a term rebuilds every step are "
            "unaffected; state that accumulates across steps (such as "
            f"TTE-TKE turbulence) takes its spin-up value.{detail} To "
            "analyse such state from a saved run, read that run's own "
            "output of it (e.g. jam_cloud_borne.*).",
            UserWarning, stacklevel=3)

    def _warn_carry_spin_up(self) -> None:
        """Warn (every cold-start call) that the threaded carry is spinning up.

        Called from :meth:`run` only in "scan" mode and only when the
        caller did not supply ``initial_physics_state`` — a donor carry
        makes this warning moot, so it is not raised then. Unlike the
        construction-time warnings above, this is NOT a once-per-instance
        warning: whether a given ``run`` call cold-starts is a per-call
        fact (one call might pass a donor, the next might not), so each
        cold-starting call is told so.
        """
        detail = (
            f" {', '.join(self._prognostic_carry_slots)} holds prognostic "
            "state (its only copy lives in the carry), so it starts from "
            "an EMPTY reservoir rather than a live run's accumulated value."
            if self._prognostic_carry_slots else "")
        warnings.warn(
            "PrescribedStateModel threads the physics' cross-step carry "
            "sequentially across the prescribed times (lax.scan, #623): "
            f"carry slots ({', '.join(self._carry_slot_names)}) in the "
            "physics seen diagnosing time i are handed to time i+1, "
            "exactly as a live run's carry flows from one step to the "
            "next. What differs from a live run is the SEED — with no "
            "initial_physics_state given, the carry is cold-started from "
            "physics.initial_carry_state at the FIRST prescribed time "
            "rather than inherited from any true prior state, so carry-"
            "dependent terms are in a spin-up for the first few times "
            "(e.g. TTE-TKE turbulence approaches its equilibrium estimate "
            f"over several steps).{detail} Pass "
            "run(..., initial_physics_state=<donor carry>) — e.g. "
            "jcm.initial_states.checkpoint_state(...)'s returned "
            "physics_carry, built against the SAME physics+coords — to "
            "seed from a live run's own carry and skip the spin-up. To "
            "analyse the cloud-borne reservoir exactly as the original run "
            "saw it, read that run's own output of it (jam_cloud_borne.*) "
            "rather than re-diagnosing.",
            UserWarning, stacklevel=3)

    @staticmethod
    def _exact_offsets(times):
        """Integer ``(days, seconds)`` offsets from ``start_time`` of ``times``.

        ``times`` is fixed-duration days since ``start_time``. A concrete axis
        is rounded to whole seconds on the host in float64, so a state ten
        years into a run is still placed on its exact second (a float32
        ``days * 86400`` would resolve only ~32 s there). A traced axis falls
        back to the same arithmetic in JAX. Returns ``(days, seconds, exact)``
        where ``exact`` is ``False`` for the traced fallback.
        """
        if isinstance(times, jax.core.Tracer):
            sim = jnp.asarray(times) * SECONDS_PER_DAY
            return (jnp.floor(sim / SECONDS_PER_DAY).astype(jnp.int32),
                    jnp.round(sim % SECONDS_PER_DAY).astype(jnp.int32), False)
        seconds = np.rint(np.asarray(jax.device_get(times), dtype=np.float64)
                          * SECONDS_PER_DAY).astype(np.int64)
        days, secs = np.divmod(seconds, SECONDS_PER_DAY)
        return (jnp.asarray(days, dtype=jnp.int32),
                jnp.asarray(secs, dtype=jnp.int32), True)

    def _run_window_seconds(self, times):
        """``(start, end)`` seconds since 1970-01-01 the states span.

        ``times`` is days since ``start_time`` (the clock ``run`` evaluates
        each state on); the window is computed from the same exact offsets,
        in the representation :func:`jcm.forcing.by_date_coverage_error`
        compares date-aligned forcing axes against. ``None`` when ``times``
        is traced: there is then no concrete window, and validation must not
        force a host read.
        """
        days, secs, exact = self._exact_offsets(times)
        if not exact:
            return None
        base = (int(self.start_time.delta.days) * SECONDS_PER_DAY
                + int(self.start_time.delta.seconds))
        offsets = (np.asarray(days, dtype=np.int64) * SECONDS_PER_DAY
                   + np.asarray(secs, dtype=np.int64))
        return float(base + offsets.min()), float(base + offsets.max())

    def _check_scan_time_order(self, day_offsets, second_offsets) -> None:
        """Refuse an out-of-order time axis in "scan" mode.

        Threading the carry sequentially (``lax.scan``) hands the carry
        diagnosing array index ``i`` to index ``i + 1`` AS THOUGH it were
        that time's true predecessor. That is only a sound re-diagnosis of
        a saved trajectory when the states/times are already in
        non-decreasing time order; a shuffled or decreasing axis (e.g.
        independent states assembled for a lookup table) would thread a
        carry between unrelated times. Skipped when the offsets are traced
        (no host values to compare) — a jitted/vmapped caller is
        responsible for its own ordering.
        """
        if isinstance(day_offsets, jax.core.Tracer):
            return
        days = np.asarray(jax.device_get(day_offsets), dtype=np.int64)
        secs = np.asarray(jax.device_get(second_offsets), dtype=np.int64)
        total_seconds = days * SECONDS_PER_DAY + secs
        if np.any(np.diff(total_seconds) < 0):
            raise ValueError(
                "PrescribedStateModel.run: physics declares cross-step "
                "carry slots that are threaded SEQUENTIALLY in array order "
                "(lax.scan, #623), so 'states'/'times' must be in "
                "non-decreasing time order — each state's carry is seeded "
                "from the state that precedes it. Sort states and times by "
                "time before calling run, or compose carry-free physics "
                "(Held-Suarez, SPEEDY) for order-independent (vmap) "
                "evaluation."
            )

    def _warn_coarse_save_cadence(self, day_offsets, second_offsets) -> None:
        """Warn once per call when the saved cadence is much coarser than ``dt``.

        Only meaningful with ``carry_steps_per_state == 1`` (the default):
        with a larger value the caller has already taken the mitigation
        this warning would recommend, so it stays silent — the warning's
        own claim ("advanced once per saved state") would otherwise be
        wrong. Skipped when the offsets are traced, like
        :meth:`_check_scan_time_order`.

        ``run.mode=prescribed``'s normal input (``jcm/runners.py``) is
        daily snapshots from a saved run whose ``dt_seconds`` is hours, so
        this is expected to fire often for the common case — it is
        informational, not a defect report, which is why it names the
        mitigation (``carry_steps_per_state``) rather than suggesting the
        gaps be "fixed".
        """
        if self.carry_steps_per_state != 1:
            return
        if isinstance(day_offsets, jax.core.Tracer):
            return
        days = np.asarray(jax.device_get(day_offsets), dtype=np.int64)
        secs = np.asarray(jax.device_get(second_offsets), dtype=np.int64)
        total_seconds = days * SECONDS_PER_DAY + secs
        gaps = np.diff(total_seconds)
        if gaps.size == 0:
            return
        median_gap = float(np.median(gaps))
        if median_gap <= 2 * self.dt_seconds:
            return
        warnings.warn(
            "PrescribedStateModel: the saved times' median spacing "
            f"({median_gap:g} s) is {median_gap / self.dt_seconds:.1f}x "
            f"dt_seconds ({self.dt_seconds} s). The carry slots "
            f"({', '.join(self._carry_slot_names)}) are advanced ONCE per "
            "saved state, not once per dt_seconds the way a live run "
            "advances them between these times, so carry-dependent "
            "diagnostics (TTE-TKE turbulence; the cloud-borne aerosol "
            "reservoir) are a save-cadence-rate approximation of the "
            "carry's true trajectory, not the model-cadence one. Construct "
            "with carry_steps_per_state=<N> (order save_interval/"
            "dt_seconds, capped sensibly) to evaluate the physics N times "
            "per saved state — state held fixed, only the carry threaded "
            "— so it relaxes toward that state's equilibrium before the "
            "diagnostics are read; this is a better approximation, not an "
            "exact one (see the class docstring). The saved run's own "
            "carry-derived output (e.g. jam_cloud_borne.*) remains the "
            "exact record.",
            UserWarning, stacklevel=3)

    def run(
        self,
        states: list[PhysicsState] | PhysicsState,
        forcing: ForcingData | None = None,
        times: jnp.ndarray | None = None,
        initial_physics_state: Any = None,
    ) -> PrescribedStatePredictions:
        """Compute physics tendencies for each prescribed state.

        Args:
            states: List of ``PhysicsState`` snapshots, or a single
                ``PhysicsState`` whose leading axis is time.
            forcing: Surface forcing; defaults to aquaplanet from ``coords``.
            times: Fixed-duration days since ``start_time`` at which each
                state is valid, one per state. Date-aligned forcing is
                selected, and its coverage checked, at these times (rounded
                to whole seconds); ``None`` means consecutive ``dt_seconds``
                steps from ``start_time``. Output labels are the exact
                :class:`jax_datetime.Datetime` values of these instants.
            initial_physics_state: Donor cross-step physics carry to seed a
                "scan"-mode re-diagnosis from (ignored in "none"/
                "unthreadable" mode, where there is nothing to seed).
                ``None`` (the default) cold-starts from
                ``physics.initial_carry_state`` instead, which warns (every
                such call — see :meth:`_warn_carry_spin_up`) that the first
                few times are a spin-up. Must structurally match
                :func:`jcm.physics_interface.build_initial_physics_carry`
                for THIS ``physics``/``coords`` — exactly what
                :func:`jcm.initial_states.checkpoint_state` returns as its
                ``physics_carry`` when the states being re-diagnosed are
                themselves a checkpointed run's output, and what
                :meth:`jcm.model.Model.initial_physics_carry` / the final
                carry of a prior ``run`` also produce — so a warm carry from
                any of those sources threads straight through.

        In "scan" mode, when the saved times are spaced much coarser than
        ``dt_seconds`` — the normal case for ``run.mode=prescribed`` input,
        daily snapshots against an hourly-to-half-hourly ``dt`` — this
        warns once per call naming the constructor's ``carry_steps_per_state``
        parameter (see the class docstring) as the knob for a better
        (still approximate) carry trajectory.

        Returns:
            ``PrescribedStatePredictions``.

        """
        if forcing is None:
            forcing = default_forcing(self.coords.horizontal)

        if isinstance(states, list):
            states = tree_map(lambda *a: jnp.stack(a, axis=0), *states)

        n_times = states.u_wind.shape[0]
        if times is None:
            # No time axis given: the states are taken as consecutive model
            # steps. A caller holding real snapshot times (e.g. the state
            # file's own time coordinate) must pass them, or date-aligned
            # forcing is selected on this synthetic step clock.
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)
        elif int(jnp.shape(times)[0]) != n_times:
            raise ValueError(
                f"PrescribedStateModel.run: {int(jnp.shape(times)[0])} times "
                f"for {n_times} states; pass one time (days since start_time) "
                "per state.")

        # Both directions of the forced-mode contract (#301), as at every run
        # entry point, over the window the prescribed states actually span.
        from jcm.physics.surface.prescribed_flux import validate_run_forcing
        validate_run_forcing(self.physics, forcing,
                             run_window=self._run_window_seconds(times))

        physics = self.physics
        terrain = self.terrain
        start_time = self.start_time
        dt_seconds = self.dt_seconds

        # ``times`` is days-since-``start_time``; place each state on its
        # exact whole-second offset (the same instants the coverage window
        # above was computed from).
        day_offsets, second_offsets, _ = self._exact_offsets(times)

        def _date_for(day_offset, second_offset):
            model_time = start_time + jdt.Timedelta(days=day_offset,
                                                    seconds=second_offset)
            elapsed = (day_offset.astype(jnp.float32) * SECONDS_PER_DAY
                       + second_offset)
            return DateData.set_date(
                model_time=model_time,
                model_step=jnp.int32(elapsed / dt_seconds),
                dt_seconds=dt_seconds,
            )

        if self._carry_mode == "scan":
            self._check_scan_time_order(day_offsets, second_offsets)
            self._warn_coarse_save_cadence(day_offsets, second_offsets)
            if initial_physics_state is None:
                # ``ComposablePhysics.compute_tendencies`` pins every
                # floating leaf of its output (the next carry) to
                # ``state.temperature.dtype`` (the "Invariant" comment in
                # ``compute_tendencies``) — the prescribed states' own
                # working dtype, NOT the process default ``jnp.zeros``
                # would use. Under ``jax_enable_x64`` with float32
                # prescribed states (the supported pySES float64-dynamics/
                # float32-physics split; JAM/MAM4 enables x64 on import),
                # those differ, and an uncast carry fails ``lax.scan``'s
                # iteration-1 type check ("carry input and carry output
                # must have equal types") on every floating slot. Casting
                # here, like ``Model.initial_physics_carry`` already casts
                # to its dycore's ``physics_dtype``, keeps the two in sync.
                initial_physics_state = build_initial_physics_carry(
                    physics, self.coords,
                    physics_dtype=states.temperature.dtype)
                self._warn_carry_spin_up()

            carry_steps_per_state = self.carry_steps_per_state

            def scan_step(carry, xs):
                state, day_offset, second_offset = xs
                clamped = verify_state(state)
                forcing_now = forcing.select(
                    _date_for(day_offset, second_offset))
                # ``carry_steps_per_state`` is a plain Python int (a static
                # construction-time choice — see its docstring), so this is
                # an UNROLLED Python loop, not a traced one: the default of
                # 1 compiles to exactly the single call the pre-this-knob
                # code made. State and forcing are held fixed across the
                # iterations; only ``carry`` (renamed ``diagnostics`` after
                # the first pass) threads between them, and only the LAST
                # iteration's tendencies/diagnostics are kept — the ones
                # earlier iterations produced existed only to let the carry
                # relax toward this state's equilibrium.
                diagnostics = carry
                for _ in range(carry_steps_per_state):
                    tendencies, raw_diagnostics = physics.compute_tendencies(
                        clamped, forcing_now, terrain,
                        prev_physics_data=diagnostics)
                    # Codex P1 (#623 follow-up): ``raw_diagnostics`` is what
                    # ``compute_tendencies`` actually returns — carrying the
                    # UNAPPLIED ``_prev_step.q_tendency`` and a frozen,
                    # never-updated ``water_positivity_correction`` entry
                    # (see ``_compute_tendencies_3d``/``_compute_tendencies_columns``:
                    # both say outright "the gridpoint interface replaces
                    # q_tendency with its applied post-cap value before the
                    # carry leaves the host" — a live ``Model`` run makes
                    # that replacement via ``compute_physics_step_gridpoint``
                    # on every step, but this scan used to thread
                    # ``raw_diagnostics`` on unchanged). The THREADED carry
                    # must instead be the one ``finalize_physics_tendencies_
                    # and_carry`` produces — the SAME verification/finalize
                    # step a live run applies — so a carry-reading term (e.g.
                    # Tiedtke reading ``_prev_step.q_tendency`` as the
                    # dynamics moisture tendency) sees what a live run would
                    # have handed it. The per-step TENDENCY recorded below
                    # stays the raw ``compute_tendencies`` value on purpose,
                    # matching the "none"/"unthreadable" (vmap) branch below,
                    # which never verifies at all — tendencies mean the same
                    # thing (physics' raw diagnosis) regardless of carry mode.
                    _, diagnostics = finalize_physics_tendencies_and_carry(
                        clamped, tendencies, raw_diagnostics,
                        dt_seconds, physics)
                # ``diagnostics`` IS the next outer step's carry, by the
                # same operator-split contract ``Model``'s scan relies on
                # (``jcm.physics.composable_physics`` module docstring) —
                # the carry fed into step i+1 and the per-step output saved
                # for step i are the identical dict.
                return diagnostics, (tendencies, diagnostics)

            @jax.jit
            def scanned():
                return jax.lax.scan(
                    scan_step, initial_physics_state,
                    (states, day_offsets, second_offsets))

            _, (tendencies, physics_data) = scanned()
        else:
            # "none" (no carry: independent in time, exact) or
            # "unthreadable" (a non-dict carry: see the class docstring —
            # kept on this path, with the construction-time warning, because
            # there is no principled way to seed a scan for it).
            def step(state, day_offset, second_offset):
                clamped = verify_state(state)
                # Collapse any TimeSeries forcing leaves to the slice valid
                # at this state's exact time.
                forcing_now = forcing.select(
                    _date_for(day_offset, second_offset))
                return physics.compute_tendencies(clamped, forcing_now, terrain)

            @jax.jit
            def vmapped():
                return jax.vmap(step)(states, day_offsets, second_offsets)

            tendencies, physics_data = vmapped()
        exact_times = start_time + jdt.Timedelta(days=day_offsets,
                                                 seconds=second_offsets)
        # Carry the TOA-first hybrid (a, b) interface tables so ``to_xarray``
        # can write real surface-first sigma coordinates + CF metadata (#739).
        from jcm import cf_metadata
        a_half, b_half = cf_metadata.hybrid_boundaries(self.coords.vertical)
        return PrescribedStatePredictions(
            states=states,
            tendencies=tendencies,
            physics_data=physics_data,
            times=exact_times,
            a_boundaries_pa=a_half,
            b_boundaries=b_half,
        )
