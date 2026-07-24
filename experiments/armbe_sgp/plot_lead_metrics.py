"""Plot lead-time RMSE from an ARMBE free-forecast evaluation CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.metrics.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no lead metrics in {args.metrics}")
    lead_hours = [float(row["lead_time_seconds"]) / 3600 for row in rows]
    rmse = [float(row["rmse"]) for row in rows]
    counts = [int(row["count"]) for row in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lead_hours, rmse, "o-", color="#c33", label="SPEEDY SCM")
    for x, y, count in zip(lead_hours, rmse, counts, strict=True):
        ax.annotate(f"n={count}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set(xlabel="Forecast lead time (hours)", ylabel="Cloud-fraction RMSE", title="SGP ARMBE cloud cover, September 2018")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
