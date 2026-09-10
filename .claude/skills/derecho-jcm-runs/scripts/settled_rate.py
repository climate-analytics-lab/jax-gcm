#!/usr/bin/env python
"""Report a convergence-checked throughput from a finished jcm run log.

    python settled_rate.py <run.log> [--chunk-days 5] [--dt 15]

Post-hoc counterpart to ``tools/benchmark.py``: that one *drives* a run and
samples GPU telemetry alongside; this one reads a log that already exists,
which is what you want for a PBS job that has come back from the queue.

Both share ``tools/chunk_timing.py`` so the two cannot give different answers
about the same run -- see that module for why the cumulative ``sim days/hr``
line must not be used, and why a rate is only quoted once the run has settled.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# tools/ lives at the repo root; this script sits four levels below it
# (.claude/skills/derecho-jcm-runs/scripts/).
_REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "tools"))

from chunk_timing import DEFAULT_TOL, analyse, parse_walls  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--chunk-days", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=None,
                   help="timestep in minutes; adds a ms/step column")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   # %% : argparse %-expands help strings
                   help=f"convergence tolerance (default {DEFAULT_TOL:.0%})"
                        .replace("%", "%%"))
    a = p.parse_args()

    try:
        text = pathlib.Path(a.log).read_text()
    except OSError as exc:
        print(f"    cannot read {a.log}: {exc}")
        return 1

    walls = parse_walls(text)
    if not walls:
        print("    no chunk timings in log yet")
        return 1

    r = analyse(walls, a.chunk_days, tol=a.tol)
    print("    per-chunk walls: " + ", ".join(f"{w:.1f}s" for w in walls))
    if "s_per_sim_day" not in r:
        print(f"    {r['reason']}")
        return 1

    print(f"    compile chunk {r['compile_chunk_s']:.1f}s (discarded)")
    print(f"    {r['reason']}")
    if not r["converged"]:
        print("    NOT CONVERGED — do not quote this rate")
    extra = ""
    if a.dt:
        steps = a.chunk_days * 24 * 60 / a.dt
        extra = f", {r['steady_chunk_s'] / steps * 1000:.0f} ms/step"
    print(f"    {'SETTLED' if r['converged'] else 'PROVISIONAL'}: "
          f"{r['steady_chunk_s']:.1f}s per {a.chunk_days:g} days = "
          f"{r['sim_days_per_hour']:.1f} sim-days/hr "
          f"({r['s_per_sim_day']:.2f} s/sim-day, "
          f"{r['sim_years_per_day']:.2f} sim-years/day){extra}")
    return 0 if r["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
