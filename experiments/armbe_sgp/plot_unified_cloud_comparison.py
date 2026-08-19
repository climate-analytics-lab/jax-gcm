"""Plot held-out calibrated SPEEDY and capped-SR cloud-cover predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    residual = prediction - target
    return (
        float(np.sqrt(np.mean(residual**2))),
        float(np.mean(residual)),
        float(np.corrcoef(prediction, target)[0, 1]),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with xr.open_dataset(args.features) as raw:
        features = raw.load()
    test = np.asarray(features["split"].values).astype(str) == "test"
    target = np.asarray(features["target"].values)[test]
    symbolic = np.tanh(
        np.asarray(features["rh_cloudc_max"].values)[test] ** 4
        + np.sqrt(np.abs(np.asarray(features["gse"].values)[test]))
        * np.asarray(features["rh_lowest"].values)[test] ** 2
    )
    calibration = np.load(args.calibration)
    if not np.array_equal(np.asarray(calibration["split"]).astype(str) == "test", test):
        raise ValueError("calibration predictions do not align with the feature cache")
    if not np.allclose(calibration["target"][test], target):
        raise ValueError("calibration targets do not align with the feature cache")
    speedy = np.asarray(calibration["raw_fitted"])[test]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, layout="constrained")
    plots = (
        ("Calibrated SPEEDY raw sum", speedy),
        ("Capped SR equation", symbolic),
    )
    for axis, (title, prediction) in zip(axes, plots, strict=True):
        hexes = axis.hexbin(
            target,
            prediction,
            gridsize=55,
            mincnt=1,
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="viridis",
        )
        axis.plot((0, 1), (0, 1), color="white", linewidth=1.2, linestyle="--")
        rmse, bias, correlation = _metrics(prediction, target)
        axis.set_title(title)
        axis.text(
            0.04,
            0.96,
            f"test n = {len(target):,}\nRMSE = {rmse:.3f}\nbias = {bias:+.3f}\nr = {correlation:.3f}",
            transform=axis.transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none"},
        )
        figure.colorbar(hexes, ax=axis, label="Samples per hexagon")
        axis.set_aspect("equal")
        axis.set_xlabel("Observed ARMBE cloud cover")
    axes[0].set_ylabel("Predicted cloud cover")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
