"""Plot same-time cloud-cover diagnostics against ARMBE observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="cloud_pairs.nc output")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    with xr.open_dataset(args.pairs) as raw:
        pairs = raw.load()
    prediction = np.asarray(pairs["prediction"].values)
    target = np.asarray(pairs["target"].values)
    mask = np.asarray(pairs["target_mask"].values, dtype=bool)
    time = np.asarray(pairs["time"].values)
    operator_name = pairs.attrs.get("operator", "cloudc")
    rmse = np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2))
    prediction_limit = max(1.0, float(np.nanmax(prediction[mask])))
    period = (
        f"{np.datetime_as_string(time[mask].min(), unit='D')} to "
        f"{np.datetime_as_string(time[mask].max(), unit='D')}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target[mask], prediction[mask], s=24, alpha=0.7, color="#c33", edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", color="#333", linewidth=1)
    ax.set(
        xlabel="Observed ARMBE total cloud fraction",
        ylabel=f"Diagnosed SPEEDY {operator_name}",
        xlim=(0, 1),
        ylim=(0, prediction_limit),
        title=f"SGP {operator_name} pairs (n={mask.sum()}, RMSE={rmse:.3f})",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "cloud_pairs.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        time[mask], target[mask], "o", markersize=4, color="#0072B2", label="ARMBE total cloud"
    )
    ax.plot(
        time[mask], prediction[mask], "x", markersize=5, color="#E69F00", label=f"SPEEDY {operator_name}"
    )
    ax.set(
        xlabel="Time",
        ylabel="Cloud fraction",
        ylim=(0, prediction_limit),
        title=f"SGP instantaneous {operator_name} diagnostics, {period}",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "cloud_time_series.png", dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
