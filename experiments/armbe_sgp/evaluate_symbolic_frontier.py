"""Select a PySR Pareto equation on validation data and score it once on test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    centered_target = target - target.mean()
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson_r": float(np.corrcoef(prediction, target)[0, 1]),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum(centered_target**2)),
    }


def _evaluate(expression: str, frame: pd.DataFrame) -> np.ndarray:
    names = {name: sp.Symbol(name) for name in frame.columns if name != "target"}
    functions = {
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
        "sqrt_pos": lambda x: sp.sqrt(sp.Max(1e-9, x)),
        "log_abs": lambda x: sp.log(sp.Abs(x) + 1e-12),
        "relu": lambda x: sp.Max(x, 0),
        "clip01": lambda x: sp.Min(sp.Max(x, 0), 1),
        "min": sp.Min,
        "max": sp.Max,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "erf": sp.erf,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "gamma_safe": lambda x: sp.Piecewise((sp.gamma(x), x > 0), (sp.nan, True)),
    }
    symbolic = sp.sympify(expression, locals=names | functions)
    function = sp.lambdify(tuple(names.values()), symbolic, modules="numpy")
    prediction = np.asarray(function(*(frame[name].to_numpy() for name in names)), dtype=float)
    if prediction.ndim == 0:
        prediction = np.full(len(frame), prediction)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError(f"non-finite or incorrectly shaped prediction for {expression!r}")
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pysr-result", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clip-prediction",
        action="store_true",
        help="clip candidate predictions to [0, 1] before scoring",
    )
    args = parser.parse_args()

    result = json.loads(args.pysr_result.read_text())
    validation = pd.read_csv(args.validation)
    test = pd.read_csv(args.test)
    validation_target = validation["target"].to_numpy()

    candidates = []
    for candidate in result["pareto_frontier"]:
        try:
            prediction = _evaluate(candidate["expression"], validation)
            if args.clip_prediction:
                prediction = np.clip(prediction, 0.0, 1.0)
        except ValueError as error:
            candidates.append(candidate | {"validation_error": str(error)})
            continue
        candidates.append(candidate | {"validation": _metrics(prediction, validation_target)})

    valid_candidates = [candidate for candidate in candidates if "validation" in candidate]
    if not valid_candidates:
        raise ValueError("no Pareto candidate produced finite validation predictions")
    selected = min(valid_candidates, key=lambda candidate: candidate["validation"]["rmse"])
    test_prediction = _evaluate(selected["expression"], test)
    if args.clip_prediction:
        test_prediction = np.clip(test_prediction, 0.0, 1.0)
    report = {
        "selection_metric": "validation.rmse",
        "prediction_clip": [0.0, 1.0] if args.clip_prediction else None,
        "selected_equation": selected,
        "test": _metrics(test_prediction, test["target"].to_numpy()),
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
