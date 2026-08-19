"""Create compact held-out comparison plots for calibrated SPEEDY and capped SR."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


COLORS = {"Calibrated SPEEDY": "#c44e52", "Capped SR": "#4c72b0"}


def _load_test_data(features_path: Path, calibration_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(features_path) as raw:
        features = raw.load()
    test = np.asarray(features["split"].values).astype(str) == "test"
    target = np.asarray(features["target"].values)[test]
    symbolic = np.tanh(
        np.asarray(features["rh_cloudc_max"].values)[test] ** 4
        + np.sqrt(np.abs(np.asarray(features["gse"].values)[test]))
        * np.asarray(features["rh_lowest"].values)[test] ** 2
    )
    calibration = np.load(calibration_path)
    if not np.allclose(calibration["target"][test], target):
        raise ValueError("calibration targets do not align with the feature cache")
    return target, {
        "Calibrated SPEEDY": np.asarray(calibration["raw_fitted"])[test],
        "Capped SR": symbolic,
    }


def _binned(values: np.ndarray, target: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    means = []
    low = []
    high = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        rows = (target >= lower) & ((target < upper) if upper < edges[-1] else (target <= upper))
        if not np.any(rows):
            continue
        positions.append(float(np.mean(target[rows])))
        means.append(float(np.mean(values[rows])))
        low.append(float(np.quantile(values[rows], 0.1)))
        high.append(float(np.quantile(values[rows], 0.9)))
    return tuple(np.asarray(value) for value in (positions, means, low, high))


def _save_calibration(target: np.ndarray, predictions: dict[str, np.ndarray], output: Path) -> None:
    edges = np.linspace(0.0, 1.0, 21)
    figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    axis.plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=1, label="1:1")
    for name, prediction in predictions.items():
        x, mean, low, high = _binned(prediction, target, edges)
        axis.fill_between(x, low, high, color=COLORS[name], alpha=0.15)
        axis.plot(x, mean, marker="o", markersize=3, color=COLORS[name], label=name)
    axis.set(
        xlim=(0, 1), ylim=(0, 1), aspect="equal",
        xlabel="Mean observed cloud cover in bin",
        ylabel="Predicted cloud cover",
        title="Held-out conditional prediction curve",
    )
    axis.legend(loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_residuals(target: np.ndarray, predictions: dict[str, np.ndarray], output: Path) -> None:
    edges = np.linspace(0.0, 1.0, 21)
    figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    for name, prediction in predictions.items():
        x, mean, low, high = _binned(prediction - target, target, edges)
        axis.fill_between(x, low, high, color=COLORS[name], alpha=0.15)
        axis.plot(x, mean, marker="o", markersize=3, color=COLORS[name], label=name)
    axis.set(
        xlim=(0, 1), ylim=(-0.65, 0.65),
        xlabel="Mean observed cloud cover in bin",
        ylabel="Prediction minus observation",
        title="Held-out conditional residuals",
    )
    axis.legend(loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_contours(target: np.ndarray, predictions: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True, layout="constrained")
    for axis, (name, prediction) in zip(axes, predictions.items(), strict=True):
        counts, xedges, yedges = np.histogram2d(target, prediction, bins=45, range=((0, 1), (0, 1)))
        density = gaussian_filter(counts.T / counts.sum(), sigma=1.1)
        x = 0.5 * (xedges[1:] + xedges[:-1])
        y = 0.5 * (yedges[1:] + yedges[:-1])
        # Endpoint concentrations are orders of magnitude denser than the
        # interior. Log-spaced levels retain the weaker but informative interior.
        contour_levels = np.geomspace(density.max() * 1.0e-3, density.max() * 0.85, 6)
        relative_density = density / density.max()
        image = axis.pcolormesh(
            xedges,
            yedges,
            relative_density,
            cmap="magma",
            norm=LogNorm(vmin=1.0e-3, vmax=1.0),
            shading="flat",
        )
        axis.contour(x, y, density, levels=contour_levels, colors="white", linewidths=0.8)
        axis.plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=1)
        axis.set(title=name, xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Observed cloud cover")
        figure.colorbar(image, ax=axis, label="Relative density (log scale)")
    axes[0].set_ylabel("Predicted cloud cover")
    figure.suptitle("Held-out smoothed 2D density contours")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_ecdf(target: np.ndarray, predictions: dict[str, np.ndarray], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    series = {"Observed ARMBE": target} | predictions
    colors = {"Observed ARMBE": "#222222"} | COLORS
    for name, values in series.items():
        sorted_values = np.sort(values)
        probabilities = np.arange(1, len(values) + 1) / len(values)
        axis.step(sorted_values, probabilities, where="post", color=colors[name], label=name)
    axis.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Cloud cover", ylabel="Empirical cumulative probability",
        title="Held-out cloud-cover distributions",
    )
    axis.legend(loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    target, predictions = _load_test_data(args.features, args.calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_calibration(target, predictions, args.output_dir / "test_binned_calibration.png")
    _save_residuals(target, predictions, args.output_dir / "test_binned_residuals.png")
    _save_contours(target, predictions, args.output_dir / "test_density_contours.png")
    _save_ecdf(target, predictions, args.output_dir / "test_ecdf.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
