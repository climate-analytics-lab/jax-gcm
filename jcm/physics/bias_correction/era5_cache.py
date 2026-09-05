"""Per-year on-disk cache of ERA5 targets and nudged start states.

Training on two decades cannot hold its inputs in memory. One year of
6-hourly ERA5 on the T31 grid is ~860 MB and a year of saved nudged states
another ~215 MB; notebook 08 already deletes its target mid-run at *one*
year. So the multi-year path builds the same per-year products once, writes
them to local disk, and every later experiment reads from there instead of
re-streaming the cloud store.

Each year is independent: it loads that year's ERA5, runs a nudged model
from 1 January to get near-ERA5 start states, and writes both. Years are
restarted rather than chained into one continuous run, which makes the build
resumable and parallelisable across GPUs; the cost is the first
``spinup_days`` of each year's states being unconverged, which the training
side already discards.

Three safeguards, each earned by a past failure in this project:

* **Atomic writes.** Arrays go to a temporary name and are renamed into
  place, and a year's manifest is written *last*. An interrupted build leaves
  no file that looks complete but is not.
* **Metadata validation.** Every year records the grid, cadence, nudging
  strength and time step it was built with, and :func:`load_year` refuses a
  mismatch. Silently reusing products built for a different grid is exactly
  what produced all-NaN scores when a T63 evaluation read T31 references.
* **Finiteness checks.** A nudged year that went non-finite is never written.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.forcing import BY_DATE, make_time_series
from jcm.model import Model
from jcm.nudging import NudgingConfig, NudgingTarget, with_nudging
from jcm.physics.bias_correction.era5_data import (
    DAYS_PER_MODEL_YEAR,
    VARS_3D,
    load_era5,
)
from jcm.physics.speedy.speedy_terms import speedy_physics

# Bumped when the on-disk layout or the meaning of a stored product changes,
# so an old cache is rejected rather than silently reinterpreted.
#   2: store fields as float32 (see STORE_DTYPE).
CACHE_VERSION = 2

# The regrid promotes to float64 -- xarray's interp and the log-pressure remap
# both run against float64 pressure axes -- but the model is float32, so the
# extra bits are discarded on the first jnp.asarray. Storing them doubled the
# cache (1.94 GB/year measured against 1.08 predicted). Casting here is exactly
# what JAX would do on load, so it changes no result, and it halves both the
# footprint and the read time of every later experiment.
STORE_DTYPE = np.float32

# PhysicsState leaves that are persisted. `tracers` is deliberately excluded:
# SPEEDY carries humidity as its own field and the dict is empty in every run
# this cache serves, so storing it would add an empty branch that later reads
# would have to special-case.
STATE_FIELDS = ("u_wind", "v_wind", "temperature", "specific_humidity",
                "geopotential", "normalized_surface_pressure")


def cache_meta(coords, *, cadence_hours: int, save_days: float,
               tau_seconds: float, time_step_minutes: float,
               spinup_days: float) -> dict:
    """Describe the settings a cached year is only valid for."""
    nlon, nlat = coords.horizontal.nodal_shape
    return {
        "cache_version": CACHE_VERSION,
        "nlon": int(nlon),
        "nlat": int(nlat),
        "layers": int(coords.vertical.layers),
        "cadence_hours": int(cadence_hours),
        "save_days": float(save_days),
        "tau_seconds": float(tau_seconds),
        "time_step_minutes": float(time_step_minutes),
        "spinup_days": float(spinup_days),
    }


def _era5_path(root, year):
    return Path(root) / f"era5_{year}.npz"


def _states_path(root, year):
    return Path(root) / f"states_{year}.npz"


def _manifest_path(root, year):
    return Path(root) / f"year_{year}.json"


def _save_atomic(path: Path, **arrays):
    """Write an npz via a temporary name so a partial file is never visible."""
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def is_cached(root, year: int, meta: dict | None = None) -> bool:
    """Is ``year`` present, complete, and built with matching settings?

    The manifest is written last, so its presence is what marks a year done.
    """
    manifest = _manifest_path(root, year)
    if not (manifest.exists() and _era5_path(root, year).exists()
            and _states_path(root, year).exists()):
        return False
    if meta is None:
        return True
    return json.loads(manifest.read_text()) == meta


def build_year(year: int, *, root, coords, terrain, base_forcing,
               cadence_hours: int = 6, save_days: float = 1.25,
               tau_seconds: float = 21600.0, time_step_minutes: float = 30.0,
               spinup_days: float = 10.0, overwrite: bool = False) -> dict:
    """Stream, nudge and cache a single calendar year.

    Returns the year's metadata. Re-running with a cached year is a no-op
    unless ``overwrite``, which is what makes a multi-hour span build safe to
    interrupt and restart.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta = cache_meta(coords, cadence_hours=cadence_hours, save_days=save_days,
                      tau_seconds=tau_seconds,
                      time_step_minutes=time_step_minutes,
                      spinup_days=spinup_days)

    if is_cached(root, year, meta) and not overwrite:
        print(f"  {year}: cached, skipping", flush=True)
        return meta

    # Each stage is announced before it starts and timed after. A year takes
    # minutes, dominated by cloud streaming and one JIT compile, so without
    # this a long build is indistinguishable from a hang -- and someone
    # watching a silent terminal reasonably kills it.
    t0 = time.time()
    print(f"  {year}: streaming ERA5 ...", flush=True)

    # A year is exactly 365 days on the model's calendar; load_era5 drops
    # 29 February and returns the axis on the model clock, so the ERA5 slot
    # index and the model day agree for every year, leap or not. It also
    # raises on a year the source cannot fill (check_year_is_complete), which
    # is what stops a partial year reaching the expensive nudged run below and
    # then being written out as if it were whole.
    fields, time_seconds = load_era5(coords, year, DAYS_PER_MODEL_YEAR,
                                     cadence_hours=cadence_hours)
    t_era5 = time.time() - t0
    print(f"  {year}: {len(time_seconds)} era5 slots in {t_era5:.0f}s;"
          f" nudged run (compiles first, then ~1 model year) ...", flush=True)

    target = NudgingTarget(**{
        k: make_time_series(jnp.asarray(v), jnp.asarray(time_seconds),
                            align_mode=BY_DATE)
        for k, v in fields.items()})

    config = NudgingConfig.temp_humidity(int(coords.vertical.layers),
                                         tau_seconds=tau_seconds)
    nudged = Model(coords=coords, terrain=terrain,
                   physics=with_nudging(speedy_physics(), config),
                   start_date=jdt.to_datetime(f"{year}-01-01"),
                   calendar="365_day", time_step=time_step_minutes)
    t1 = time.time()
    preds = nudged.run(forcing=base_forcing.replace(nudging_target=target),
                       save_interval=save_days,
                       total_time=float(DAYS_PER_MODEL_YEAR))
    states = preds.dynamics
    del preds, target

    state_arrays = {f"state__{name}": np.asarray(getattr(states, name),
                                                 dtype=STORE_DTYPE)
                    for name in STATE_FIELDS}
    for name, a in state_arrays.items():
        if not np.isfinite(a).all():
            raise ValueError(f"{year}: nudged run went non-finite in {name}")
    n_frames = state_arrays["state__temperature"].shape[0]
    print(f"  {year}: {n_frames} state frames in {time.time() - t1:.0f}s,"
          f" all finite; writing ...", flush=True)

    t2 = time.time()
    _save_atomic(_era5_path(root, year),
                 # The clock stays float64: it counts seconds since 1970, which
                 # is ~1e9 and would lose sub-hour resolution in float32.
                 time_seconds=np.asarray(time_seconds, dtype=np.float64),
                 **{f"era5__{k}": np.asarray(v, dtype=STORE_DTYPE)
                    for k, v in fields.items()})
    _save_atomic(_states_path(root, year), **state_arrays)
    # Written last: this is what `is_cached` treats as the completion marker.
    _manifest_path(root, year).write_text(json.dumps(meta, indent=2))

    written = (_era5_path(root, year).stat().st_size
               + _states_path(root, year).stat().st_size) / 1e9
    print(f"  {year}: wrote {written:.2f} GB in {time.time() - t2:.0f}s"
          f"  [year total {time.time() - t0:.0f}s]", flush=True)
    return meta


def build_span(years, **kwargs) -> list[int]:
    """Cache every year in ``years``, skipping those already complete."""
    built = []
    for year in years:
        build_year(year, **kwargs)
        built.append(year)
    return built


def state_slot_indices(n_frames: int, save_days: float, cadence_hours: int,
                       n_slots: int) -> np.ndarray:
    """ERA5 slot index matching each saved state frame.

    A run's frame ``i`` is the state after ``i + 1`` saves, not ``i`` -- the
    first save lands at ``t = save_interval``, never at ``t = 0``. Pairing
    frame ``i`` with the slot at day ``i * save_days`` is therefore off by one
    save interval, which is the label-lag bug: nothing errors, every label
    is just shifted, and the trained term is quietly wrong.

    The last frames of a year have no target inside that year (frame 291 of a
    292-frame year wants slot 1460 of 1460) and are dropped rather than
    clipped, since clipping would silently pair the final frames with the same
    stale target.

    Returns:
        ``(n_paired,)`` slot indices, one per usable frame, in frame order.

    """
    slots_per_save = save_days * 24.0 / cadence_hours
    if not float(slots_per_save).is_integer():
        raise ValueError(
            f"save_days={save_days} at {cadence_hours} h cadence gives "
            f"{slots_per_save} slots per save; the pairing needs a whole "
            f"number of ERA5 slots per saved frame")
    idx = (np.arange(n_frames) + 1) * int(slots_per_save)
    return idx[idx < n_slots]


def load_year(root, year: int, *, meta: dict | None = None,
              with_states: bool = True):
    """Read a cached year back.

    Args:
        root: cache directory.
        year: calendar year to read.
        meta: if given, the settings the caller requires. A cached year built
            with anything else raises rather than being silently reused.
        with_states: skip loading the nudged states when only the ERA5
            targets are wanted (the offline stage does not need them).

    Returns:
        ``(fields, time_seconds, states)`` where ``states`` is a dict of
        PhysicsState arrays, or ``None`` when ``with_states`` is false.

    """
    root = Path(root)
    manifest = _manifest_path(root, year)
    if not manifest.exists():
        raise FileNotFoundError(
            f"{year} is not cached under {root} (no manifest; an interrupted "
            f"build leaves the arrays but no marker, so rebuild that year)")
    stored = json.loads(manifest.read_text())
    if meta is not None and stored != meta:
        diff = {k: (stored.get(k), v) for k, v in meta.items()
                if stored.get(k) != v}
        raise ValueError(
            f"{year} was cached with different settings; stored vs required: "
            f"{diff}. Rebuild it rather than mixing products.")

    with np.load(_era5_path(root, year)) as z:
        fields = {v: z[f"era5__{v}"] for v in VARS_3D}
        time_seconds = z["time_seconds"]

    states = None
    if with_states:
        with np.load(_states_path(root, year)) as z:
            states = {name: z[f"state__{name}"] for name in STATE_FIELDS}
    return fields, time_seconds, states
