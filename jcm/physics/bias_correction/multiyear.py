"""Assemble multi-year offline training data from the per-year ERA5 cache.

The single-year path (notebook 07) holds one nudged trajectory in memory and
flattens it. Twenty-one years of that is ~28 million rows, far more than an
8352-parameter per-column MLP needs and more than fits comfortably alongside
the model, so this reads the cache one year at a time, subsamples frames, and
accumulates on the host.

Subsampling is by frame rather than by column on purpose. Columns within a
frame are the diversity the network actually learns from -- it sees each as an
independent sample -- whereas frames 1.25 days apart are strongly correlated.
Taking every ``frame_stride``-th frame therefore spreads the sample across
seasons and years while keeping the full horizontal variety of each frame
retained.

The labels are recomputed with the *same* :class:`NudgingConfig` the cache was
built with, so the recorded target is the nudge the run actually applied
rather than an approximation of it.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from jcm.nudging import NudgingConfig, NudgingTarget
from jcm.physics.bias_correction.era5_cache import (
    DAYS_PER_MODEL_YEAR,
    load_year,
    state_slot_indices,
)
from jcm.physics.bias_correction.era5_data import VARS_3D
from jcm.physics.bias_correction.offline_training import (
    assemble_training_arrays,
    recompute_target_tendencies,
)
from jcm.physics_interface import PhysicsState

# Cached PhysicsState leaves that are not part of the 4-field feature vector
# but are needed to rebuild a state the nudging tendency can be taken against.
_EXTRA_STATE = ("geopotential", "normalized_surface_pressure")


def _state_from_cache(states: dict, frames) -> PhysicsState:
    """Rebuild a PhysicsState from cached arrays.

    ``frames`` may be an array of indices, giving a trajectory with a leading
    time axis, or a single integer, giving one start state with no time axis.
    Both callers need the same field list and the same empty tracer dict, so
    they share this rather than each spelling it out.
    """
    take = {name: jnp.asarray(states[name][frames])
            for name in VARS_3D + _EXTRA_STATE}
    return PhysicsState(**take, tracers={})


def frames_per_year(meta: dict) -> int:
    """Return the number of saved state frames in one cached year.

    Derived from the cache's own save cadence rather than assumed. Hardcoding
    it (365 / 1.25 = 292) silently mis-spaces the start selection for any cache
    built with a different ``save_days``.
    """
    return int(DAYS_PER_MODEL_YEAR / meta["save_days"])


def parse_year_span(spec: str) -> list[int]:
    """Parse ``'1995-2015'`` or ``'2001,2002'`` into a sorted list of years.

    Shared by every multi-year driver so a span means the same thing to the
    cache builder, the offline trainer and the fine-tuner.
    """
    years = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"descending year range: {part}")
            years.update(range(lo, hi + 1))
        else:
            years.add(int(part))
    if not years:
        raise ValueError(f"no years in {spec!r}")
    return sorted(years)


def spinup_frames(spinup_days: float, save_days: float) -> int:
    """Count the leading frames to discard from a year.

    Every cached year is an independent nudged run started from bootstrap, so
    its opening days are the model being pulled onto ERA5 rather than a
    converged near-ERA5 trajectory. The nudging tendency there is large and
    unusually easy to predict, which flatters the offline fit while teaching
    the term about a transient it will never see in a free run.

    Frame ``i`` sits at day ``(i + 1) * save_days``, so this counts the frames
    at or inside ``spinup_days``.
    """
    return int(np.floor(spinup_days / save_days + 1e-9))


def year_pairs(root, year: int, *, meta: dict, frame_stride: int = 4,
               spinup_days: float | None = None):
    """Feature/target rows for one cached year.

    Args:
        root: cache directory.
        year: calendar year to read.
        meta: settings the cached year must match.
        frame_stride: keep every Nth surviving frame.
        spinup_days: leading days to discard; defaults to the value the cache
            recorded.

    Returns:
        ``(feats, targets)``, both ``(N, 4*nlev)`` float32 host arrays, where
        ``N`` is the retained frame count times the number of columns.

    """
    fields, seconds, states = load_year(root, year, meta=meta)
    n_frames = states["temperature"].shape[0]

    # Frame i is the state after i+1 saves; the tail frames of a year have no
    # in-year target and are dropped. See state_slot_indices for why this is
    # not a clip.
    slots = state_slot_indices(n_frames, meta["save_days"],
                               meta["cadence_hours"], len(seconds))
    drop = spinup_frames(
        meta["spinup_days"] if spinup_days is None else spinup_days,
        meta["save_days"])
    if drop >= len(slots):
        raise ValueError(
            f"{year}: spin-up of {drop} frames leaves nothing of "
            f"{len(slots)} usable frames")
    frames = np.arange(drop, len(slots))[::frame_stride]
    slots = slots[drop:][::frame_stride]

    state = _state_from_cache(states, frames)
    target = NudgingTarget(**{v: jnp.asarray(fields[v][slots])
                              for v in VARS_3D})

    config = NudgingConfig.temp_humidity(
        int(states["temperature"].shape[1]),
        tau_seconds=meta["tau_seconds"])
    tendencies = recompute_target_tendencies(state, target, config)
    feats, targets = assemble_training_arrays(state, tendencies)
    return np.asarray(feats, dtype=np.float32), np.asarray(targets,
                                                           dtype=np.float32)


def climatology_starts(root, years, *, meta: dict, n_starts: int,
                       win_slots: int, lead_slots: int | None = None,
                       spinup_days: float | None = None,
                       days: Sequence[float] | None = None):
    """Start states and matching ERA5 window means, drawn across cached years.

    The single-year driver takes consecutive frames a day and a quarter apart,
    so its "pool" is really one problem seen a few times. Here the pool cycles
    **years first**: with 21 cached years, a pool of 21 is the same day of year
    in each of them, and only a larger pool starts stepping the day of year
    forward. That is the multi-year version of a climatology -- one seasonal
    window, many independent realisations of the weather inside it -- rather
    than a wider spread of seasons, which would make each start a different
    problem and the gradient noisier.

    The calendar year of a start does not reach the model: the ocean forcing is
    a repeating climatology and insolation depends only on fraction-of-year, so
    a start is fully described by its day of year. Only ``sim_time`` is set.

    Every window is required to fit inside its own year, which keeps the cache
    read per-year and avoids stitching a window across a year boundary.

    ``days`` overrides that single-window default with an explicit list of
    days of year, cycling years first WITHIN each day so every season still
    gets many realisations rather than one noisy sample. It exists because the
    default pool cannot reach a second season: the frame advances by one save
    (``save_days``, 1.25 days) per completed year-cycle, so even a pool of 84
    spans 11.25 to 15.0 days of year. Reaching mid-July would need a pool of
    ~6000.

    Repeat a day to WEIGHT it. Weighting has to be done by sampling frequency
    rather than by scaling the loss, because the driver runs one start per
    update and Adam divides the step by ``sqrt(v)``: scaling a single update's
    loss scales its gradient and its second moment together, so the step size
    is very nearly unchanged. Appearing twice per epoch genuinely doubles a
    window's influence; a 2x loss multiplier very nearly does not.

    Returns:
        A list of ``n_starts`` dicts with ``year``, ``frame``, ``sim_time``
        (seconds), ``state`` (a single-frame :class:`PhysicsState`), ``climT``,
        ``climq``, and -- when ``lead_slots`` is given -- ``leadT``/``leadq``
        for the combined rollout term.

    """
    save_days = meta["save_days"]
    slots_per_day = 24 // meta["cadence_hours"]
    spin = meta["spinup_days"] if spinup_days is None else spinup_days
    first = spinup_frames(spin, save_days)

    years = list(years)
    if days is None:
        picks = [(years[k % len(years)], first + k // len(years))
                 for k in range(n_starts)]
    else:
        # day = (frame + 1) * save_days, matching `sim_time` and the `s0`
        # slot arithmetic below, so invert it the same way.
        wanted = []
        for d in days:
            frame = int(round(d / save_days)) - 1
            if frame < first:
                raise ValueError(
                    f"day {d} is frame {frame}, inside the {spin}-day spinup "
                    f"(first usable frame {first}, day "
                    f"{(first + 1) * save_days}).")
            wanted.append(frame)
        picks = [(years[k % len(years)],
                  wanted[(k // len(years)) % len(wanted)])
                 for k in range(n_starts)]

    # Group by year so each year's cache is read once, but carry every pick's
    # position in the cycle. Keying the final sort on (year, frame) instead
    # would collapse duplicates -- and duplicates are exactly how `days`
    # weights a season -- landing both copies of a repeated window in the same
    # slot, so the driver would hand the optimiser the identical start on two
    # consecutive updates instead of spreading them through the epoch.
    by_year: dict = {}
    for i, (y, f) in enumerate(picks):
        by_year.setdefault(y, []).append((f, i))

    out = []
    for year in sorted(by_year):
        frames = sorted(by_year[year])
        fields, seconds, states = load_year(root, year, meta=meta)
        n_slots = len(seconds)
        for frame, pick_index in frames:
            # Frame f is the state after f+1 saves; the ERA5 window starts at
            # that same instant. Same +1 as state_slot_indices, for the same
            # reason.
            s0 = int(round((frame + 1) * save_days * slots_per_day))
            need = s0 + win_slots + (lead_slots or 0)
            if need > n_slots:
                raise ValueError(
                    f"{year}: start frame {frame} needs ERA5 slot {need} of "
                    f"{n_slots}. The window runs past the end of the year; "
                    f"use a shorter window or fewer starts.")
            rec = {
                "_order": pick_index,
                "year": year,
                "frame": frame,
                "sim_time": (frame + 1) * save_days * 86400.0,
                # A single frame index, so no leading time axis: this is one
                # start, not a trajectory.
                "state": _state_from_cache(states, frame),
                "climT": jnp.asarray(
                    fields["temperature"][s0:s0 + win_slots].mean(0)),
                "climq": jnp.asarray(
                    fields["specific_humidity"][s0:s0 + win_slots].mean(0)),
            }
            if lead_slots is not None:
                rec["leadT"] = jnp.asarray(
                    fields["temperature"][s0 + lead_slots])
                rec["leadq"] = jnp.asarray(
                    fields["specific_humidity"][s0 + lead_slots])
            out.append(rec)
        del fields, states

    # Return in the requested cycle order, not grouped by year, so a run that
    # stops early has still seen a spread of years rather than only the first.
    out.sort(key=lambda r: r["_order"])
    for rec in out:
        del rec["_order"]
    return out


def rollout_starts(root, years, *, meta: dict, n_starts: int,
                   lead_slots: Sequence[int], spinup_days: float | None = None,
                   n_frames: int | None = None):
    """Start states plus short-lead ERA5 targets, drawn across cached years.

    The rollout curriculum trains the same starts against several lead times
    (6 h, 12 h, 24 h, ...), so every lead is collected in one pass over the
    cache rather than re-reading a year per stage.

    Selection differs from :func:`climatology_starts` on purpose: starts are
    spread over the SEASONAL CYCLE as well as over years. Notebook 08 staggered
    its starts across a whole year, and that coverage is load-bearing. A term
    trained only on mid-January states has never seen July, and the 450-day
    free run used for evaluation passes through every season, so it goes
    non-finite. The climatology stage can keep a fixed seasonal window because
    its loss is an average over one; a rollout term cannot.

    Returns:
        A list of dicts with ``year``, ``frame``, ``sim_time``, ``state``, and
        ``targets``: a mapping ``lead -> (T, q)`` of ERA5 snapshots that many
        6-hourly slots after the start.

    """
    save_days = meta["save_days"]
    slots_per_day = 24 // meta["cadence_hours"]
    slots_per_frame = int(save_days * slots_per_day)
    spin = meta["spinup_days"] if spinup_days is None else spinup_days
    first = spinup_frames(spin, save_days)
    leads = sorted(set(int(x) for x in lead_slots))
    if n_frames is None:
        n_frames = frames_per_year(meta)

    years = list(years)
    # Walk the day of year and the calendar year together, so consecutive
    # starts differ in BOTH. With 21 years and 21 starts that is 21 distinct
    # seasons in 21 distinct years rather than one week repeated. The last
    # frame is excluded because the longest stage still needs a target after
    # it, converted from slots into frames.
    usable = max(1, n_frames - first - max(leads) // slots_per_frame - 1)
    step = max(1, usable // max(1, n_starts))
    picks = [(years[k % len(years)], first + k * step)
             for k in range(n_starts)]

    out = []
    for year in sorted({y for y, _ in picks}):
        frames = sorted(f for y, f in picks if y == year)
        fields, seconds, states = load_year(root, year, meta=meta)
        n_slots = len(seconds)
        for frame in frames:
            s0 = int(round((frame + 1) * save_days * slots_per_day))
            if s0 + max(leads) >= n_slots:
                raise ValueError(
                    f"{year}: start frame {frame} with lead {max(leads)} needs "
                    f"slot {s0 + max(leads)} of {n_slots}; the longest stage "
                    f"runs past the end of the year.")
            out.append({
                "year": year,
                "frame": frame,
                "sim_time": (frame + 1) * save_days * 86400.0,
                "state": _state_from_cache(states, frame),
                "targets": {
                    lead: (jnp.asarray(fields["temperature"][s0 + lead]),
                           jnp.asarray(fields["specific_humidity"][s0 + lead]))
                    for lead in leads},
            })
        del fields, states

    order = {(y, f): i for i, (y, f) in enumerate(picks)}
    out.sort(key=lambda r: order[(r["year"], r["frame"])])
    return out


def offline_dataset(root, years, *, meta: dict, frame_stride: int = 4,
                    spinup_days: float | None = None, verbose: bool = True):
    """Concatenate :func:`year_pairs` across ``years``.

    Args:
        root: cache directory.
        years: calendar years to include.
        meta: the settings the cache must have been built with; a year built
            differently raises rather than being blended in silently.
        frame_stride: keep every Nth saved frame. 1 uses everything.
        spinup_days: leading days to discard per year; defaults to the cached
            value.
        verbose: print a per-year progress line.

    Returns:
        ``(feats, targets)``, both ``(N, 4*nlev)`` float32.

    """
    all_feats, all_targets, total = [], [], 0
    for year in years:
        f, t = year_pairs(root, year, meta=meta, frame_stride=frame_stride,
                          spinup_days=spinup_days)
        all_feats.append(f)
        all_targets.append(t)
        total += f.shape[0]
        if verbose:
            print(f"  {year}: {f.shape[0]:,} rows (running {total:,})",
                  flush=True)
    feats = np.concatenate(all_feats, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    if not np.isfinite(feats).all():
        raise ValueError("assembled features contain non-finite values")
    if not np.isfinite(targets).all():
        raise ValueError("assembled targets contain non-finite values")
    return feats, targets
