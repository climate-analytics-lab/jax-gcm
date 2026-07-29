"""Drive SPEEDY physics as a single column at SGP with ARMBE state.

Diagnostic mode: the observed profiles are prescribed at every step and the
physics diagnoses radiation, convection, condensation and surface fluxes on
them. The dynamical core never runs. Only tracers evolve.

    source env.sh
    python run_scm.py                       # synthetic fixture
    python run_scm.py --atm data/sgparmbeatmC1.c1 --cldrad data/sgparmbecldradC1.c1

Writes ``outputs/scm_run.npz`` for ``evaluate.py``.

Two things worth knowing about this setup:

* ``start_date`` is taken from the data and matters a lot. The SCM selects
  forcing per step from it, and SPEEDY's insolation keys off fraction-of-year;
  getting it wrong silently gives you the wrong season's sun (see
  ``SCM_FORCING_PATCH_NOTE.md`` at the repo root).
* Land surface temperature follows the retained input record through a
  date-aligned ``TimeSeries``. Soil moisture and albedo remain static, so they
  are the next surface-forcing candidates if real-data fluxes are biased.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.forcing import BY_DATE, ForcingData, make_time_series
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData

from armbe_io import (
    SGP_LAT_DEG,
    SGP_LON_DEG,
    SGP_OROG_M,
    load_armbe,
    pick,
    to_obs_targets,
    to_state_series,
)

# physics_data term -> fields we pull out as diagnostics.
DIAGNOSTICS = {
    "_condensation": ("precls",),
    "_convection": ("precnv",),
    "_shortwave_rad": ("rsds", "rsns", "fsol", "cloudc", "ftop"),
    "_surface_flux": ("rlds", "rlus", "shf", "evap", "tsfc"),
}


def unwrap(o):
    """physics_data stores each term as a 0-d object array around a dataclass."""
    if getattr(o, "dtype", None) is object and getattr(o, "shape", None) == ():
        return o.item()
    return o


def extract(physics_data) -> dict[str, np.ndarray]:
    out = {}
    for term, fields in DIAGNOSTICS.items():
        obj = unwrap(physics_data[term])
        d = vars(obj) if hasattr(obj, "__dict__") else obj
        for f in fields:
            if f in d:
                out[f"{term.lstrip('_')}.{f}"] = np.asarray(d[f])
    return out


def _epoch_seconds(times: np.ndarray) -> np.ndarray:
    """ARMBE timestamps -> seconds since 1970-01-01, which is what TimeSeries
    BY_DATE indexing expects (mirrors forcing._time_axis_seconds_from_ds).
    """
    t = np.asarray(times).astype("datetime64[s]")
    return (t - np.datetime64("1970-01-01T00:00:00")).astype(np.int64).astype(float)


def start_date_from_timestamp(timestamp) -> jdt.Datetime:
    """Build the SCM start date without discarding its hour/minute/second."""
    timestamp = np.datetime64(timestamp, "s")
    return jdt.to_datetime(np.datetime_as_string(timestamp, unit="s"))


def validate_cadence(times: np.ndarray, dt_seconds: float) -> None:
    """Require retained prescribed states to match the SCM integration step.

    A gap cannot safely be represented by merely advancing a regular SCM scan:
    that would put the profile, date-selected forcing, and diagnostics on
    different clocks. Reject it explicitly until irregular timestamp support is
    implemented.
    """
    times = np.asarray(times).astype("datetime64[s]")
    if len(times) < 2:
        return
    deltas = np.diff(times).astype("timedelta64[s]").astype(np.int64)
    if not np.all(deltas == int(dt_seconds)):
        raise ValueError(
            "retained ARMBE timestamps must be regularly spaced at --dt; "
            f"got intervals {np.unique(deltas).tolist()} s with --dt={dt_seconds}. "
            "Choose a contiguous window or add irregular-timestep support."
        )


def filter_regular_cadence(states, times: np.ndarray, meta: dict,
                           dt_seconds: float):
    """Retain the dominant timestamp phase on a requested regular cadence.

    ARMBEATM can contain otherwise valid profile records between its nominal
    six-hour analysis times. This opt-in filter removes those off-cadence
    records; :func:`validate_cadence` still rejects an actual missing step on
    the retained phase rather than interpolating across it.
    """
    times = np.asarray(times).astype("datetime64[s]")
    if len(times) < 2:
        return states, times, meta
    seconds = times.astype(np.int64)
    phases, counts = np.unique(np.mod(seconds, int(dt_seconds)), return_counts=True)
    dominant_phase = phases[np.argmax(counts)]
    keep = np.mod(seconds, int(dt_seconds)) == dominant_phase
    if np.all(keep):
        return states, times, meta

    retained_indices = meta["retained_indices"][keep]
    filtered_meta = {
        **meta,
        "n_states": int(keep.sum()),
        "retained_indices": retained_indices,
        "n_off_cadence_dropped": int((~keep).sum()),
        "cadence_phase_seconds": int(dominant_phase),
    }
    return [state for state, use in zip(states, keep) if use], times[keep], filtered_meta


def _git_revision(repo_root: Path) -> str | None:
    """Return the checked-out revision when this experiment is run from git."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, args, argv, meta: dict, times: np.ndarray) -> None:
    """Write the provenance needed to reproduce an SCM archive."""
    times = np.asarray(times).astype("datetime64[s]")
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "archive": str(args.output.resolve()),
        "inputs": {
            "atm": str(Path(args.atm).resolve()),
            "cldrad": str(Path(args.cldrad).resolve()),
            "resolved_variables": meta["resolved"],
        },
        "retained_times": {
            "start": np.datetime_as_string(times[0], unit="s"),
            "end": np.datetime_as_string(times[-1], unit="s"),
            "n_states": meta["n_states"],
            "n_input_times": meta["n_input_times"],
            "n_dropped": meta["n_dropped"],
        },
        "configuration": {
            "nlev": args.nlev,
            "dt_seconds": args.dt,
            "window_start": args.start,
            "window_end": args.end,
            "static_forcing": args.static_forcing,
            "regular_cadence": args.regular_cadence,
            "calendar": "gregorian",
            "site": "SGP C1",
            "land_surface_temperature": (
                "static record mean" if args.static_forcing else "BY_DATE time series"
            ),
            "soil_moisture": 0.30,
            "bare_land_albedo": 0.20,
            "co2_vmr_ppmv": 407.0,
        },
        "command": ["python", Path(__file__).name, *argv],
        "git_revision": _git_revision(repo_root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_forcing(ds, times, nodal_shape=(1, 1), static: bool = False):
    """Land-surface forcing for SGP, with surface temperature following the obs.

    Why this isn't static: sensible heat is driven by the land-air temperature
    difference. Pinning ``stl_am`` at the record mean makes that difference
    average to zero, so the flux averages to zero too — the scheme is fine, it's
    just being handed a zero gradient. Verified by sweeping ``stl_am`` against a
    fixed 300 K air temperature: shf goes -25.6, -13.9, -2.3, +9.4, +21.1 W/m²
    for dT of -10, -5, 0, +5, +10 K. Clean and linear at ~2.3 W/m²/K.

    So ``stl_am`` is a ``TimeSeries`` leaf: the patched ``SingleColumnModel``
    calls ``ForcingData.select(date)`` every step, which slices it to that step.
    ``static=True`` restores the old (broken) behaviour for comparison.
    """
    t_name = pick(ds, "surface_temperature", required=False)
    if t_name is not None:
        t_obs = np.asarray(ds[t_name].values, dtype=float)
        if np.nanmax(t_obs) < 100.0:      # Celsius -> Kelvin
            t_obs = t_obs + 273.15
    else:
        t_obs = np.full(len(times), 295.0)
    t_obs = np.nan_to_num(t_obs, nan=float(np.nanmean(t_obs)))

    n = min(len(t_obs), len(times))
    t_obs, times = t_obs[:n], times[:n]
    ones = jnp.ones(nodal_shape)

    if static:
        stl = float(np.mean(t_obs)) * ones
        sst = stl
    else:
        # (nt, 1, 1): time on axis 0, the single column on the rest.
        stl = make_time_series(
            jnp.asarray(t_obs.reshape(n, *nodal_shape)),
            jnp.asarray(_epoch_seconds(times)),
            align_mode=BY_DATE,
        )
        sst = float(np.mean(t_obs)) * ones     # land point; SST is unused

    return ForcingData.zeros(
        nodal_shape,
        alb0=0.20 * ones,           # bare-land albedo, typical for SGP pasture
        stl_am=stl,                 # land surface temperature (follows obs)
        sea_surface_temperature=sst,
        soilw_am=0.30 * ones,
        snowc_am=jnp.zeros(nodal_shape),
        sice_am=jnp.zeros(nodal_shape),
        co2_vmr=jnp.asarray(407.0),   # ~2018 global mean, ppmv
    ), float(np.mean(t_obs))


def main(argv=None) -> int:
    here = Path(__file__).parent
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atm", default=str(here / "data/synthetic/sgparmbeatmC1.c1.synthetic.nc"))
    ap.add_argument("--cldrad", default=str(here / "data/synthetic/sgparmbecldradC1.c1.synthetic.nc"))
    ap.add_argument("--nlev", type=int, default=8)
    ap.add_argument("--dt", type=float, default=3600.0, help="seconds; match obs cadence")
    ap.add_argument("--start", default=None, help="window start, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="window end, YYYY-MM-DD")
    ap.add_argument("--output", type=Path, default=here / "outputs" / "scm_run.npz")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="JSON provenance sidecar (default: <output>.manifest.json)")
    ap.add_argument("--static-forcing", action="store_true",
                    help="pin surface temp at the record mean (the old, broken "
                          "behaviour) — for comparison only")
    ap.add_argument("--regular-cadence", action="store_true",
                    help="retain only the dominant timestamp phase at --dt before "
                         "rejecting genuine missing steps")
    args = ap.parse_args(argv)

    ds = load_armbe(args.atm, args.cldrad, args.start, args.end)
    states, times, meta = to_state_series(ds, nlev=args.nlev)
    if args.regular_cadence:
        states, times, meta = filter_regular_cadence(states, times, meta, args.dt)
    if not states:
        raise SystemExit("no usable states built from the input — check the loader "
                         "report above for unresolved variables.")
    retained_ds = ds.isel(time=meta["retained_indices"])
    obs = to_obs_targets(ds, indices=meta["retained_indices"])
    validate_cadence(times, args.dt)

    # The date the states actually correspond to. This drives insolation.
    t0 = np.asarray(times)[0]
    start_date = start_date_from_timestamp(t0)

    forcing, t_sfc = build_forcing(retained_ds, times, static=args.static_forcing)
    # SGP is land: fmask=1 (fmask is the LAND fraction — speedy_surface_flux.py:274
    # blends tsfc = sst + fmask*(stl_am - sst)). lfluxland=True is essential and
    # defaults to False: the land flux branch is behind
    # `jax.lax.cond(lfluxland, land_fluxes, pass_fn)` (line 228), so leaving it off
    # means the land tile stays zeros while fmask=1 selects exactly that tile —
    # i.e. the atmosphere sees no surface fluxes at all, silently.
    terrain = TerrainData.single_column(orog=SGP_OROG_M, fmask=1.0, lfluxland=True)

    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(args.nlev),
        lat_deg=SGP_LAT_DEG,
        lon_deg=SGP_LON_DEG,
        terrain=terrain,
        forcing=forcing,
        dt_seconds=args.dt,
        start_date=start_date,
        calendar="gregorian",
    )

    print(f"site      : SGP C1  ({SGP_LAT_DEG}N, {SGP_LON_DEG}E)  orog={SGP_OROG_M}m  land")
    print(f"physics   : SPEEDY, {args.nlev} sigma levels, dt={args.dt:.0f}s")
    print(f"start_date: {np.datetime64(t0, 's')}  (drives insolation)")
    print(f"steps     : {len(states)}  ({len(states)*args.dt/86400:.1f} days)")
    print(f"sfc temp  : {t_sfc:.1f} K mean, "
          f"{'STATIC (broken, comparison only)' if args.static_forcing else 'following obs per step'}")
    print(f"dropped   : {meta['n_dropped']} invalid profile times, "
          f"{meta.get('n_off_cadence_dropped', 0)} off-cadence times")

    pred = scm.run(states)
    diags = extract(pred.physics_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        times=np.asarray(times).astype("datetime64[s]").astype(np.int64),
        dt_seconds=args.dt,
        **{f"model.{k}": v for k, v in diags.items()},
        **{f"obs.{k}": v for k, v in obs.items()},
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(manifest_path, args, argv, meta, times)
    print(f"\nwrote {args.output}")
    print(f"wrote {manifest_path}")
    print("\nmodel diagnostics:")
    for k, v in sorted(diags.items()):
        flat = np.asarray(v).reshape(v.shape[0], -1)
        print(f"  {k:28s} shape={str(v.shape):16s} "
              f"mean={np.nanmean(flat):10.4g}  finite={np.all(np.isfinite(v))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
