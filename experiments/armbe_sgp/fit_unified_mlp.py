"""Fit a small train-only MLP baseline for pooled ARMBE total cloud fraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    residual = prediction - target
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson_r": float(np.corrcoef(prediction, target)[0, 1]),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum((target - target.mean()) ** 2)),
    }


def _nested_rh_prediction(frame: pd.DataFrame) -> np.ndarray:
    """Evaluate the compact nested-RH equation on its four required features."""
    return np.tanh(
        (frame["rh_low_mean"] + frame["rh_mid_mean"])
        * (frame["rh_vertical_range"] ** 3 + frame["rh_high_mean"])
    ).to_numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 64])
    args = parser.parse_args(argv)
    if any(width < 1 for width in args.hidden_layers):
        raise ValueError("--hidden-layers widths must be positive")
    frames = {split: pd.read_csv(args.features_dir / f"{split}.csv") for split in ("train", "validation", "test")}
    feature_names = [name for name in frames["train"].columns if name != "target"]
    scaler = StandardScaler().fit(frames["train"][feature_names])
    model = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=1.0e-4,
        batch_size=256,
        learning_rate_init=1.0e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=args.seed,
    )
    model.fit(scaler.transform(frames["train"][feature_names]), frames["train"]["target"])
    parameter_count = sum(values.size for values in (*model.coefs_, *model.intercepts_))
    report = {
        "model": "MLPRegressor",
        "architecture": [len(feature_names), *args.hidden_layers, 1],
        "activation": "relu",
        "trainable_parameter_count": int(parameter_count),
        "feature_names": feature_names,
        "input_scaling": "StandardScaler fit on training rows only",
        "training": {
            "iterations": int(model.n_iter_),
            "best_internal_validation_score": float(model.best_validation_score_),
            "seed": args.seed,
        },
        "splits": {},
    }
    predictions = {}
    for split, frame in frames.items():
        raw = model.predict(scaler.transform(frame[feature_names]))
        clipped = np.clip(raw, 0.0, 1.0)
        compact = _nested_rh_prediction(frame)
        predictions[split] = clipped
        report["splits"][split] = {
            "raw": _metrics(raw, frame["target"].to_numpy()),
            "clipped_0_1": _metrics(clipped, frame["target"].to_numpy()),
            "fraction_clipped": float(np.mean(raw != clipped)),
            "compact_nested_rh": _metrics(compact, frame["target"].to_numpy()),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "predictions.npz", **predictions)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
