"""Plot QC-passed observed and predicted cloud-fraction pairs by forecast lead."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with xr.open_dataset(args.predictions) as raw:
        data = raw.load()
    prediction = np.asarray(data["prediction"].values)
    target = np.asarray(data["target"].values)
    mask = np.asarray(data["target_mask"].values, dtype=bool)
    lead_hours = np.asarray(data["lead_time_seconds"].values) / 3600

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for ax, lead, observed, forecast, good in zip(
        axes.flat, lead_hours, target.T, prediction.T, mask.T, strict=True
    ):
        ax.scatter(observed[good], forecast[good], s=20, alpha=0.7, color="#c33", edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color="#333", linewidth=1)
        ax.set(title=f"{lead:g} h lead (n={good.sum()})", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.grid(alpha=0.25)
    fig.supxlabel("Observed ARMBE cloud fraction")
    fig.supylabel("Forecast SPEEDY cloud fraction")
    fig.suptitle("SGP ARMBE cloud cover pairs, September 2018")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
