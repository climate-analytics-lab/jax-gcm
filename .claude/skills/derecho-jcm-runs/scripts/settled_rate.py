#!/usr/bin/env python
"""Report a convergence-checked throughput from a finished jcm run log.

    python settled_rate.py <PBS stdout log> [--chunk-days 5] [--dt 15]

The log is the job's stdout (``runs/<tag>.log``), not Hydra's ``main.log``:
the per-chunk ``Wall:`` lines are prints and never reach the Hydra log.

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

import os


def _find_tools() -> pathlib.Path:
    """Locate the repo's ``tools/`` directory, wherever this script lives.

    A fixed parent depth only works from the in-repo copy: the same skill
    installed under ``~/.claude/skills/`` resolves to the home directory and
    the ``chunk_timing`` import fails. Search upward from the script and from
    the working directory for the module itself, honouring ``$JCM_REPO``
    first, so ``settled_rate.py`` needs no PYTHONPATH from either location.
    """
    roots = []
    if os.environ.get("JCM_REPO"):
        roots.append(pathlib.Path(os.environ["JCM_REPO"]).expanduser())
    roots += list(pathlib.Path(__file__).resolve().parents)
    roots += list(pathlib.Path.cwd().resolve().parents) + [pathlib.Path.cwd()]
    for root in roots:
        if (root / "tools" / "chunk_timing.py").is_file():
            return root / "tools"
    raise SystemExit(
        "cannot find the repo's tools/chunk_timing.py from "
        f"{pathlib.Path(__file__).resolve()} or {pathlib.Path.cwd()} — run "
        "this from inside a jax-gcm checkout, or set JCM_REPO to one.")


sys.path.insert(0, str(_find_tools()))

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
