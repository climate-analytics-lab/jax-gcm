"""Run a bounded five-feature PySR pilot on June ECHAM-layer observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler

from evaluate_echam_cloud_baselines import CORE_FEATURES, _metrics, _rows


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "outputs/echam_layer_cloud_june_2018/echam_l47_june.nc"
DEFAULT_OUTPUT = HERE / "outputs/echam_layer_cloud_june_2018/pysr_pilot"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input) as source:
        dataset = source.load()
    train_x, train_y, train_profile, _ = _rows(dataset, "train")
    valid_x, valid_y, valid_profile, _ = _rows(dataset, "validation")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    valid_scaled = scaler.transform(valid_x)
    counts = dict(zip(*np.unique(train_profile, return_counts=True)))
    weights = np.asarray([1.0 / counts[value] for value in train_profile])
    weights *= weights.size / weights.sum()

    model = PySRRegressor(
        niterations=args.iterations,
        populations=8,
        population_size=31,
        ncycles_per_iteration=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        complexity_of_operators={"/": 3, "square": 3},
        nested_constraints={"square": {"square": 0}},
        maxsize=15,
        maxdepth=5,
        model_selection="best",
        random_state=20260731,
        deterministic=True,
        parallelism="serial",
        timeout_in_seconds=600,
        output_directory=str(args.output),
        run_id="june_core_seed20260731",
        verbosity=1,
    )
    model.fit(
        train_scaled,
        train_y,
        weights=weights,
        variable_names=list(CORE_FEATURES),
    )

    frontier = []
    best: dict[str, object] | None = None
    condensate_present = (valid_x[:, 3] + valid_x[:, 4]) > 0.0
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
        }
        frontier.append(candidate)
        if best is None or metrics["equal_profile_rmse"] < best["validation"]["equal_profile_rmse"]:
            best = candidate

    result = {
        "features": list(CORE_FEATURES),
        "iterations": args.iterations,
        "seed": 20260731,
        "train_rows": int(train_y.size),
        "validation_rows": int(valid_y.size),
        "selection": "minimum clipped, zero-condensate-gated equal-profile validation RMSE",
        "outer_holdout_evaluated": False,
        "standardization": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "best": best,
        "frontier": frontier,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
