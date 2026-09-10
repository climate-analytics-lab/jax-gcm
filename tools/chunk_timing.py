"""Convergence-checked throughput from jcm chunk wall times.

Shared by ``tools/benchmark.py`` (which drives a run) and the Derecho skill's
``settled_rate.py`` (which reads a finished log), so the two cannot drift to
different answers about what a run's throughput was.

The rules encoded here exist because each was got wrong at least once:

* **jcm's own ``N sim days/hr`` line is cumulative and includes compile.**
  Reading it as the throughput understates a run by several times early on and
  keeps drifting upward for the whole run. A 5.3x "regression" was filed
  against jax-rrtmgp on the strength of chunk 1 of a run that settled 22x
  faster, and had to be retracted. Use ``Wall: X s this chunk``.
* **Chunk 1 contains compilation** and is always discarded.
* **A rate is only quoted once the run has settled.** Chunk times fall for
  3-4 chunks (XLA autotuning, cache warming, host allocator). Convergence is
  judged on the *last two* chunks rather than the first agreeing pair: a
  noisy series can produce an early coincidental match while still drifting,
  and what matters is that the run had settled by the time it ended.
* **High GPU utilisation does not prove steady state.** XLA's autotuner keeps
  the device ~95% busy while it is still choosing kernels. Only
  chunk-to-chunk agreement is evidence.
"""

from __future__ import annotations

import re

WALL_RE = re.compile(r"Wall:\s*([0-9.]+)s this chunk")

# Agreement required between the last two chunks before a rate is quoted.
DEFAULT_TOL = 0.03


def parse_walls(text: str) -> list[float]:
    """Per-chunk wall times [s] from a jcm run log."""
    return [float(x) for x in WALL_RE.findall(text)]


def analyse(walls: list[float], days_per_chunk: float,
            tol: float = DEFAULT_TOL) -> dict:
    """Reduce per-chunk wall times to a converged throughput.

    Returns a dict that always carries ``converged`` and ``reason``; the rate
    fields are present whenever at least two chunks ran, so a caller can show
    a provisional number *as long as it also shows the flag*.

    The reported ``steady_chunk_s`` is the mean of the last two chunks rather
    than a single chunk -- same convergence criterion, slightly less noise.
    """
    if len(walls) < 2:
        return {"converged": False,
                "reason": "fewer than 2 chunks completed (chunk 1 is compile)",
                "chunk_walls_s": walls}

    post_compile = walls[1:]
    if len(post_compile) < 2:
        # Exactly one post-compile chunk: nothing to compare it against, so a
        # rate is reported but explicitly unconverged.
        return {
            "converged": False,
            "reason": ("only 1 post-compile chunk; need >=2 to judge "
                       "convergence (>=3 chunks in total)"),
            "chunk_walls_s": walls,
            "compile_chunk_s": walls[0],
            "steady_chunk_s": post_compile[-1],
            "first_settled_chunk": None,
            **_rates(post_compile[-1], days_per_chunk),
        }

    prev, last = post_compile[-2], post_compile[-1]
    converged = abs(last - prev) / max(last, prev) <= tol
    steady = (last + prev) / 2.0

    # Where it first settled, for reporting -- not the value used.
    first_settled = None
    for i in range(1, len(post_compile)):
        a, b = post_compile[i - 1], post_compile[i]
        if abs(a - b) / max(a, b) <= tol:
            first_settled = i + 1      # 1-based post-compile chunk index
            break

    reason = (f"last two post-compile chunks ({prev:.1f}s, {last:.1f}s) agree "
              f"within {tol:.0%}" if converged else
              f"last two post-compile chunks ({prev:.1f}s, {last:.1f}s) differ "
              f"by {abs(last - prev) / max(last, prev):.1%} > {tol:.0%} — "
              "still settling, do not quote")
    return {
        "converged": converged,
        "reason": reason,
        "chunk_walls_s": walls,
        "compile_chunk_s": walls[0],
        "steady_chunk_s": round(steady, 2),
        "first_settled_chunk": first_settled,
        **_rates(steady, days_per_chunk),
    }


def _rates(chunk_s: float, days_per_chunk: float) -> dict:
    per_day = chunk_s / days_per_chunk
    return {
        "s_per_sim_day": round(per_day, 2),
        "sim_days_per_hour": round(3600.0 / per_day, 1),
        "sim_years_per_day": round(86400.0 / per_day / 365.0, 2),
    }
