"""Run independent unnudged six-hour SPEEDY physics hindcasts from ARMBE.

Each window starts from an observed atmospheric profile, advances temperature,
humidity, and horizontal winds with SPEEDY physics for twelve 30-minute steps,
then resets all state, tracers, and physics carry at the following observed
profile. This is a physics-only hindcast, not a continuous free-running SCM:
surface pressure and geopotential are held at the initial observed values and
the dynamical core / large-scale forcing are absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dinosaur.sigma_coordinates import SigmaCoordinates

from armbe_io import SGP_LAT_DEG, SGP_LON_DEG, SGP_OROG_M, load_armbe, to_obs_targets, to_state_series
from evaluate import PAIRS, metrics, model_series, to_daily
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData
from run_scm import (
    _git_revision,
    build_forcing,
    extract,
    filter_regular_cadence,
    start_date_from_timestamp,
    validate_cadence,
)


def _interval_observations(ds, window_starts, window_ends) -> dict[str, np.ndarray]:
    """Mean full-resolution ARM targets over each half-open forecast window."""
    raw_targets = to_obs_targets(ds)
    data_times = np.asarray(ds.time.values).astype("datetime64[s]")
    out = {name: [] for name in raw_targets}
    for start, end in zip(window_starts, window_ends):
        mask = (data_times >= start) & (data_times < end)
        for name, values in raw_targets.items():
            out[name].append(float(np.nanmean(values[mask])) if np.any(mask) else np.nan)
    return {name: np.asarray(values) for name, values in out.items()}


def _print_comparison(archive: dict[str, np.ndarray]) -> None:
    """Print interval-mean process diagnostics and final-profile errors."""
    model = model_series(archive)
    print(f"\nSix-hour interval-mean comparison, {len(archive['times'])} windows\n")
    header = f"{'field':24s}{'obs':>10s}{'model':>10s}{'bias':>10s}{'rmse':>10s}{'corr':>8s}"
    print(header)
    print("-" * len(header))
    for label, (model_key, obs_key) in PAIRS.items():
        if obs_key not in archive:
            continue
        summary = metrics(model[model_key], archive[obs_key])
        corr = f"{summary['corr']:8.2f}" if np.isfinite(summary["corr"]) else f"{'n/a':>8s}"
        print(f"{label:24s}{summary['obs_mean']:10.3f}{summary['mod_mean']:10.3f}"
              f"{summary['bias']:10.3f}{summary['rmse']:10.3f}{corr}")

    print("\nFinal-profile error after each unnudged six-hour window\n")
    print(header)
    print("-" * len(header))
    units = {
        "temperature": "K",
        "specific_humidity": "g/kg",
        "u_wind": "m/s",
        "v_wind": "m/s",
    }
    for name, unit in units.items():
        summary = metrics(archive[f"final.model.{name}"], archive[f"final.obs.{name}"])
        corr = f"{summary['corr']:8.2f}" if np.isfinite(summary["corr"]) else f"{'n/a':>8s}"
        label = f"final {name} [{unit}]"
        print(f"{label:24s}{summary['obs_mean']:10.3f}{summary['mod_mean']:10.3f}"
              f"{summary['bias']:10.3f}{summary['rmse']:10.3f}{corr}")


def plot_comparison(archive: dict[str, np.ndarray], output: Path) -> None:
    """Plot daily means so radiation is comparable to the prescribed-state plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model_series(archive)
    pairs = [(label, model_key, obs_key) for label, (model_key, obs_key) in PAIRS.items()
             if obs_key in archive]
    times = archive["times"].astype("datetime64[s]")
    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 2.2 * len(pairs)), sharex=True)
    for ax, (label, model_key, obs_key) in zip(np.atleast_1d(axes), pairs):
        day, observed = to_daily(np.asarray(archive[obs_key]), times)
        _, predicted = to_daily(model[model_key], times)
        ax.plot(day, observed, "o-", label="ARMBE", color="#222")
        ax.plot(day, predicted, "s--", label="SPEEDY SCM", color="#c33")
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("SPEEDY unnudged six-hour physics-only hindcasts vs ARMBE, SGP (daily means)")
    fig.tight_layout()
    fig.savefig(output, dpi=130)


def plot_interval_comparison(archive: dict[str, np.ndarray], output: Path) -> None:
    """Plot raw six-hour window means without daily aggregation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model_series(archive)
    pairs = [(label, model_key, obs_key) for label, (model_key, obs_key) in PAIRS.items()
             if obs_key in archive]
    times = archive["times"].astype("datetime64[s]")
    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 2.2 * len(pairs)), sharex=True)
    for ax, (label, model_key, obs_key) in zip(np.atleast_1d(axes), pairs):
        ax.plot(times, archive[obs_key], "o-", label="ARMBE", color="#222")
        ax.plot(times, model[model_key], "s--", label="SPEEDY SCM", color="#c33")
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("SPEEDY unnudged six-hour physics-only hindcasts vs ARMBE, SGP (six-hour means)")
    fig.tight_layout()
    fig.savefig(output, dpi=130)


def _full_calendar_day_means(
    values: np.ndarray, starts: np.ndarray, window_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Duration-weight interval means into complete UTC calendar days only."""
    day_seconds = 24 * 60 * 60
    totals: dict[int, float] = {}
    coverage: dict[int, int] = {}
    starts_seconds = starts.astype("datetime64[s]").astype(np.int64)
    for value, start in zip(np.asarray(values, dtype=float), starts_seconds):
        if not np.isfinite(value):
            continue
        end = start + window_seconds
        day_start = start // day_seconds * day_seconds
        while day_start < end:
            overlap = min(end, day_start + day_seconds) - max(start, day_start)
            if overlap > 0:
                totals[day_start] = totals.get(day_start, 0.0) + value * overlap
                coverage[day_start] = coverage.get(day_start, 0) + overlap
            day_start += day_seconds
    days = sorted(day for day in totals if coverage[day] == day_seconds)
    return (
        np.asarray(days, dtype=np.int64).astype("datetime64[s]"),
        np.asarray([totals[day] / day_seconds for day in days]),
    )


def plot_complete_day_comparison(archive: dict[str, np.ndarray], output: Path) -> None:
    """Plot complete calendar-day means without assigning cross-midnight data twice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model_series(archive)
    pairs = [(label, model_key, obs_key) for label, (model_key, obs_key) in PAIRS.items()
             if obs_key in archive]
    times = archive["times"].astype("datetime64[s]")
    window_seconds = int(archive["window_seconds"])
    fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 2.2 * len(pairs)), sharex=True)
    for ax, (label, model_key, obs_key) in zip(np.atleast_1d(axes), pairs):
        observed_days, observed = _full_calendar_day_means(
            archive[obs_key], times, window_seconds)
        predicted_days, predicted = _full_calendar_day_means(
            model[model_key], times, window_seconds)
        ax.plot(observed_days, observed, "o-", label="ARMBE", color="#222")
        ax.plot(predicted_days, predicted, "s--", label="SPEEDY SCM", color="#c33")
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("SPEEDY unnudged six-hour physics-only hindcasts vs ARMBE, SGP (complete UTC days)")
    fig.tight_layout()
    fig.savefig(output, dpi=130)


def main(argv=None) -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atm", required=True)
    parser.add_argument("--cldrad", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--nlev", type=int, default=8)
    parser.add_argument("--window-seconds", type=int, default=21600)
    parser.add_argument("--dt", type=int, default=1800)
    parser.add_argument("--output", type=Path,
                        default=here / "outputs" / "six_hour_hindcasts.npz")
    parser.add_argument("--plot", action="store_true", help="write hindcast_compare.png")
    parser.add_argument("--complete-day-plot", action="store_true",
                        help="write hindcast_compare_complete_days.png")
    parser.add_argument("--interval-plot", action="store_true",
                        help="write hindcast_compare_6h.png")
    args = parser.parse_args(argv)
    if args.window_seconds % args.dt:
        parser.error("--window-seconds must be divisible by --dt")

    ds = load_armbe(args.atm, args.cldrad, args.start, args.end)
    states, times, meta = to_state_series(ds, nlev=args.nlev)
    states, times, meta = filter_regular_cadence(states, times, meta, args.window_seconds)
    validate_cadence(times, args.window_seconds)
    if len(states) < 2:
        parser.error("at least two contiguous observed profiles are required")

    # Use the full hourly surface-temperature record, rather than only profile
    # timestamps, while each atmospheric forecast window evolves at 30 minutes.
    forcing, _ = build_forcing(ds, ds.time.values)
    terrain = TerrainData.single_column(orog=SGP_OROG_M, fmask=1.0, lfluxland=True)
    steps_per_window = args.window_seconds // args.dt
    window_starts = np.asarray(times[:-1]).astype("datetime64[s]")
    window_ends = np.asarray(times[1:]).astype("datetime64[s]")

    diagnostics: dict[str, list[np.ndarray]] = {}
    final_model = {name: [] for name in ("temperature", "specific_humidity", "u_wind", "v_wind")}
    final_obs = {name: [] for name in final_model}
    for initial, target, start in zip(states[:-1], states[1:], window_starts):
        scm = SingleColumnModel(
            physics=speedy_physics(),
            vertical=SigmaCoordinates.equidistant(args.nlev),
            lat_deg=SGP_LAT_DEG,
            lon_deg=SGP_LON_DEG,
            terrain=terrain,
            forcing=forcing,
            dt_seconds=args.dt,
            start_date=start_date_from_timestamp(start),
            calendar="gregorian",
            prognostic_variables=("temperature", "specific_humidity", "u_wind", "v_wind"),
        )
        prediction = scm.run([initial] * steps_per_window)
        for name, values in extract(prediction.physics_data).items():
            diagnostics.setdefault(name, []).append(np.mean(values, axis=0))
        for name in final_model:
            final_model[name].append(np.asarray(prediction.relaxed_states[name][-1]))
            final_obs[name].append(np.asarray(getattr(target, name)))

    observations = _interval_observations(ds, window_starts, window_ends)
    archive = {
        "times": window_starts.astype("datetime64[s]").astype(np.int64),
        "window_end_times": window_ends.astype("datetime64[s]").astype(np.int64),
        "dt_seconds": np.asarray(args.dt),
        "window_seconds": np.asarray(args.window_seconds),
        **{f"model.{name}": np.stack(values) for name, values in diagnostics.items()},
        **{f"obs.{name}": values for name, values in observations.items()},
        **{f"final.model.{name}": np.stack(values) for name, values in final_model.items()},
        **{f"final.obs.{name}": np.stack(values) for name, values in final_obs.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **archive)

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "experiment": "independent unnudged six-hour physics-only hindcasts",
        "inputs": {"atm": str(Path(args.atm).resolve()), "cldrad": str(Path(args.cldrad).resolve())},
        "window": {"start": str(window_starts[0]), "end": str(window_ends[-1]), "n_windows": len(window_starts)},
        "configuration": {
            "nlev": args.nlev,
            "physics_dt_seconds": args.dt,
            "window_seconds": args.window_seconds,
            "prognostic_variables": ["temperature", "specific_humidity", "u_wind", "v_wind"],
            "surface_pressure_and_geopotential": "held at each window initial state",
            "physics_carry_and_tracers": "reset at each window",
        },
        "profile_filter": {"invalid_dropped": meta["n_dropped"], "off_cadence_dropped": meta.get("n_off_cadence_dropped", 0)},
        "git_revision": _git_revision(Path(__file__).resolve().parents[2]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")
    _print_comparison(archive)
    if args.plot:
        plot_path = args.output.parent / "hindcast_compare.png"
        plot_comparison(archive, plot_path)
        print(f"wrote {plot_path}")
    if args.complete_day_plot:
        plot_path = args.output.parent / "hindcast_compare_complete_days.png"
        plot_complete_day_comparison(archive, plot_path)
        print(f"wrote {plot_path}")
    if args.interval_plot:
        plot_path = args.output.parent / "hindcast_compare_6h.png"
        plot_interval_comparison(archive, plot_path)
        print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
