#!/usr/bin/env python
"""Report a convergence-checked throughput from a jcm run log.

    python settled_rate.py <run.log> [--chunk-days 5]

jcm's own ``N sim days/hr`` line is cumulative and includes compile time.
This reads ``Wall: X s this chunk`` instead and only quotes a rate when the
last two chunks agree within 5%.
"""
from __future__ import annotations

import argparse
import re
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--chunk-days", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=None,
                   help="timestep in minutes; adds a ms/step column")
    a = p.parse_args()

    try:
        text = open(a.log).read()
    except OSError as exc:
        print(f"    cannot read {a.log}: {exc}")
        return 1

    walls = [float(x) for x in
             re.findall(r"Wall: ([0-9.]+)s this chunk", text)]
    if not walls:
        print("    no chunk timings in log yet")
        return 1

    print("    per-chunk walls: " +
          ", ".join(f"{w:.1f}s" for w in walls))
    if len(walls) < 3:
        print("    NOT ENOUGH CHUNKS to quote a settled rate "
              "(need >=3; early chunks include compile)")
        return 1

    last, prev = walls[-1], walls[-2]
    converged = abs(last - prev) / max(last, prev) < 0.05
    mean = (last + prev) / 2.0
    rate = a.chunk_days / (mean / 3600.0)
    extra = ""
    if a.dt:
        steps = a.chunk_days * 24 * 60 / a.dt
        extra = f", {mean / steps * 1000:.0f} ms/step"
    print(f"    last two: {prev:.1f}s {last:.1f}s -> "
          f"{'CONVERGED' if converged else 'NOT CONVERGED (do not quote)'}")
    print(f"    SETTLED: {mean:.1f}s per {a.chunk_days:g} days = "
          f"{rate:.1f} sim-days/hr{extra}")
    print(f"    (chunk 1 was {walls[0]:.1f}s and includes compile; discarded)")
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
