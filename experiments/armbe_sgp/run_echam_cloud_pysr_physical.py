"""Run repeatable PySR searches under the published EQ4 input contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler

from evaluate_echam_cloud_baselines import _metrics
from evaluate_grundner_eq4 import _height_gradient, liquid_relative_humidity


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "outputs/echam_layer_cloud_june_2018/echam_l47_june.nc"
DEFAULT_OUTPUT = HERE / "outputs/echam_layer_cloud_june_2018/pysr_physical"
FEATURES = ("rh_liquid", "temperature", "rh_gradient_height", "qc", "qi")


def _rows(
    dataset: xr.Dataset, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rh = liquid_relative_humidity(
        dataset.specific_humidity.values,
        dataset.pressure.values,
        dataset.temperature.values,
    )
    valid = dataset.layer_valid.values.astype(bool)
    rh_gradient = _height_gradient(rh, dataset.height.values, valid)
    selected = valid & (dataset.split.values == split)[:, None]
    features = np.column_stack(
        (
            rh[selected],
            dataset.temperature.values[selected],
            rh_gradient[selected],
            dataset.qc.values[selected],
            dataset.qi.values[selected],
        )
    )
    target = dataset.cloud_fraction.values[selected]
    profile = np.broadcast_to(
        dataset.profile.values[:, None], dataset.layer_valid.shape
    )[selected]
    finite = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    return features[finite], target[finite], profile[finite]


def _fit_seed(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_profile: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_profile: np.ndarray,
    valid_common_core: np.ndarray,
    output: Path,
    seed: int,
    iterations: int,
    timeout: int,
) -> dict[str, object]:
    train_condensate = (train_x[:, 3] + train_x[:, 4]) > 0.0
    fit_x = train_x[train_condensate]
    fit_y = train_y[train_condensate]
    fit_profile = train_profile[train_condensate]
    scaler = StandardScaler().fit(fit_x)
    fit_scaled = scaler.transform(fit_x)
    valid_scaled = scaler.transform(valid_x)
    counts = dict(zip(*np.unique(fit_profile, return_counts=True)))
    weights = np.asarray([1.0 / counts[value] for value in fit_profile])
    weights *= weights.size / weights.sum()

    model = PySRRegressor(
        niterations=iterations,
        populations=12,
        population_size=41,
        ncycles_per_iteration=200,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        complexity_of_operators={"/": 3, "square": 3},
        nested_constraints={"square": {"square": 0}},
        maxsize=17,
        maxdepth=6,
        model_selection="best",
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        timeout_in_seconds=timeout,
        output_directory=str(output),
        run_id=f"physical_seed{seed}",
        verbosity=1,
    )
    model.fit(fit_scaled, fit_y, weights=weights, variable_names=list(FEATURES))

    condensate_present = (valid_x[:, 3] + valid_x[:, 4]) > 0.0
    frontier = []
    best: dict[str, object] | None = None
    for index, equation in model.equations_.iterrows():
        try:
            raw = np.asarray(model.predict(valid_scaled, index=index), dtype=float)
        except Exception as error:  # PySR can retain unevaluable frontier members.
            frontier.append({"index": int(index), "error": repr(error)})
            continue
        prediction = np.where(condensate_present, np.clip(raw, 0.0, 1.0), 0.0)
        if not np.all(np.isfinite(prediction)):
            frontier.append({"index": int(index), "error": "nonfinite prediction"})
            continue
        metrics = _metrics(valid_y, prediction, valid_profile)
        candidate = {
            "index": int(index),
            "complexity": int(equation["complexity"]),
            "equation": str(equation["equation"]),
            "validation": metrics,
            "validation_common_core_rows": _metrics(
                valid_y[valid_common_core],
                prediction[valid_common_core],
                valid_profile[valid_common_core],
            ),
        }
        frontier.append(candidate)
        if best is None or metrics["equal_profile_rmse"] < best["validation"]["equal_profile_rmse"]:
            best = candidate
    return {
        "seed": seed,
        "fit_rows": int(fit_y.size),
        "fit_profiles": int(np.unique(fit_profile).size),
        "standardization": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "best": best,
        "frontier": frontier,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seeds", default="20260731,20260801,20260802")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]

    with xr.open_dataset(args.input) as source:
        dataset = source.load()
    train_x, train_y, train_profile = _rows(dataset, "train")
    valid_x, valid_y, valid_profile = _rows(dataset, "validation")
    selected = dataset.layer_valid.values & (dataset.split.values == "validation")[:, None]
    valid_common_core = np.isfinite(dataset.rh_gradient_log_pressure.values[selected])
    runs = [
        _fit_seed(
            train_x,
            train_y,
            train_profile,
            valid_x,
            valid_y,
            valid_profile,
            valid_common_core,
            args.output,
            seed,
            args.iterations,
            args.timeout,
        )
        for seed in seeds
    ]
    eligible = [
        run["best"] | {"seed": run["seed"], "standardization": run["standardization"]}
        for run in runs
        if run["best"]
    ]
    best = min(eligible, key=lambda candidate: candidate["validation"]["equal_profile_rmse"])
    result = {
        "features": list(FEATURES),
        "iterations_per_seed": args.iterations,
        "seeds": seeds,
        "train_rows_before_condensate_gate": int(train_y.size),
        "validation_rows": int(valid_y.size),
        "fit_regime": "qc + qi > 0; exact-zero gate applied after prediction",
        "selection": "minimum clipped, gated equal-profile validation RMSE across frontiers",
        "outer_holdout_evaluated": False,
        "best": best,
        "runs": runs,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
