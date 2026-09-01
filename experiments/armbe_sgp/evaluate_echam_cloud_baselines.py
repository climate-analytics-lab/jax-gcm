"""Evaluate train-only baselines on the June ECHAM-layer validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "outputs/echam_layer_cloud_june_2018/echam_l47_june.nc"
DEFAULT_OUTPUT = HERE / "outputs/echam_layer_cloud_june_2018/baseline_comparison.json"
CORE_FEATURES = (
    "relative_humidity",
    "temperature",
    "rh_gradient_log_pressure",
    "qc",
    "qi",
)


def _metrics(target: np.ndarray, prediction: np.ndarray, profile: np.ndarray) -> dict[str, float | int]:
    prediction = np.clip(prediction, 0.0, 1.0)
    error = prediction - target
    profile_mse = []
    for value in np.unique(profile):
        selected = profile == value
        profile_mse.append(np.mean(error[selected] ** 2))
    return {
        "rows": int(target.size),
        "profiles": int(np.unique(profile).size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "equal_profile_rmse": float(np.sqrt(np.mean(profile_mse))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "correlation": float(np.corrcoef(target, prediction)[0, 1]),
    }


def _rows(dataset: xr.Dataset, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profile_selected = dataset.split.values == split
    row_selected = dataset.layer_valid.values & profile_selected[:, None]
    features = np.column_stack(
        [dataset[name].values[row_selected] for name in CORE_FEATURES]
    )
    target = dataset.cloud_fraction.values[row_selected]
    profile = np.broadcast_to(
        dataset.profile.values[:, None], dataset.layer_valid.shape
    )[row_selected]
    level = np.broadcast_to(
        dataset.level.values[None, :], dataset.layer_valid.shape
    )[row_selected]
    finite = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    return features[finite], target[finite], profile[finite], level[finite]


def evaluate(dataset: xr.Dataset) -> dict[str, object]:
    train_x, train_y, train_profile, train_level = _rows(dataset, "train")
    valid_x, valid_y, valid_profile, valid_level = _rows(dataset, "validation")
    train_counts = dict(zip(*np.unique(train_profile, return_counts=True)))
    train_weight = np.asarray([1.0 / train_counts[value] for value in train_profile])
    train_weight *= train_weight.size / train_weight.sum()

    level_mean = {
        int(level): float(np.mean(train_y[train_level == level]))
        for level in np.unique(train_level)
    }
    global_mean = float(np.average(train_y, weights=train_weight))
    climatology = np.asarray([level_mean.get(int(level), global_mean) for level in valid_level])

    rh = valid_x[:, 0]
    pressure_rows = dataset.pressure.values[
        dataset.layer_valid.values & (dataset.split.values == "validation")[:, None]
    ]
    finite_pressure = np.isfinite(dataset.rh_gradient_log_pressure.values[
        dataset.layer_valid.values & (dataset.split.values == "validation")[:, None]
    ])
    pressure_rows = pressure_rows[finite_pressure]
    surface_pressure = np.asarray([
        dataset.surface_pressure.values[int(value)] for value in valid_profile
    ])
    rh_critical = 0.75 + (0.975 - 0.75) * np.exp(
        1.0 - (surface_pressure / pressure_rows) ** 2.0
    )
    b0 = np.clip((rh - rh_critical) / (1.0 - rh_critical), 0.0, 1.0)
    sundqvist = np.where(pressure_rows < 1000.0, 0.0, 1.0 - np.sqrt(1.0 - b0))
    condensate_presence = ((valid_x[:, 3] + valid_x[:, 4]) > 0.0).astype(float)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(train_x, train_y, ridge__sample_weight=train_weight)
    gradient_boosting = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=20260731,
    )
    gradient_boosting.fit(train_x, train_y, sample_weight=train_weight)

    predictions = {
        "level_climatology": climatology,
        "sundqvist_no_inversion": sundqvist,
        "condensate_presence": condensate_presence,
        "ridge_core": ridge.predict(valid_x),
        "hist_gradient_boosting_core": gradient_boosting.predict(valid_x),
    }
    return {
        "features": list(CORE_FEATURES),
        "train_rows": int(train_y.size),
        "train_profiles": int(np.unique(train_profile).size),
        "validation_rows": int(valid_y.size),
        "validation_profiles": int(np.unique(valid_profile).size),
        "outer_holdout_evaluated": False,
        "target": {
            "train_mean": float(np.mean(train_y)),
            "validation_mean": float(np.mean(valid_y)),
            "train_zero_fraction": float(np.mean(train_y == 0.0)),
            "validation_zero_fraction": float(np.mean(valid_y == 0.0)),
        },
        "validation_metrics": {
            name: _metrics(valid_y, prediction, valid_profile)
            for name, prediction in predictions.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    with xr.open_dataset(args.input) as source:
        result = evaluate(source.load())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
