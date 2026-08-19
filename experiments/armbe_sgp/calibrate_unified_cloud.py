"""Train-only calibration of pooled T30 SPEEDY and fixed symbolic cloud readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from scipy.optimize import minimize


RAW_BOUNDS = ((0.10, 0.60), (0.05, 0.60), (0.30, 0.90), (0.00, 0.40))
RAW_INITIAL = np.asarray((0.30, 0.20, 0.60, 0.15))
SYMBOLIC_BOUNDS = ((0.0, 5.0), (0.0, 5.0), (-5.0, 5.0))
SYMBOLIC_INITIAL = np.asarray((1.0, 1.0, 0.0))
NESTED_BOUNDS = ((0.0, 5.0), (0.0, 5.0), (-5.0, 5.0))
NESTED_INITIAL = np.asarray((1.0, 1.0, 0.0))


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    residual = prediction - target
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson_r": float(np.corrcoef(prediction, target)[0, 1]),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum((target - target.mean()) ** 2)),
        "prediction_std": prediction_std,
        "target_std": target_std,
        "fraction_above_one": float(np.mean(prediction > 1.0)),
        "max_prediction": float(np.max(prediction)),
    }


def speedy_raw_sum(features: dict[str, jax.Array], params: jax.Array) -> jax.Array:
    """Evaluate the default hard-hinge SPEEDY cloud diagnosis exactly from its inputs.

    In a single diagnostic step, these four parameters are read only after the
    humidity, convection, and condensation terms supplying this formula have
    been diagnosed. Reusing the exported inputs is consequently equivalent to
    rerunning that final diagnosis for any candidate parameter vector.
    """
    rhcl1, wpcl, clsmax, clsminl = params
    rh_term = jnp.clip((features["rh_cloudc_max"] - rhcl1) / (1.0 - rhcl1), 0.0, 1.0) ** 2
    precipitation = jnp.minimum(10.0, features["precip_mm_day"])
    cloudc = jnp.minimum(1.0, wpcl * jnp.sqrt(jnp.maximum(1.0e-9, precipitation)) + rh_term)
    stability = jnp.clip((features["gse"] - 0.25) / 0.15, 0.0, 1.0)
    cloudstr_sea = stability * jnp.maximum(clsmax - 1.2 * cloudc, 0.0)
    cloudstr_land = jnp.maximum(cloudstr_sea, clsminl) * features["rh_lowest"]
    cloudstr = jnp.minimum(
        1.0, cloudstr_sea + features["fmask"] * (cloudstr_land - cloudstr_sea)
    )
    return cloudc + cloudstr


def symbolic_cloud(features: dict[str, jax.Array], params: jax.Array) -> jax.Array:
    """Evaluate the selected SR structure with trainable structural coefficients."""
    a_rh, a_gse, bias = params
    return jnp.tanh(
        a_rh * features["rh_cloudc_max"] ** 4
        + a_gse * jnp.sqrt(jnp.abs(features["gse"])) * features["rh_lowest"] ** 2
        + bias
    )


def nested_symbolic_cloud(features: dict[str, jax.Array], params: jax.Array) -> jax.Array:
    """Evaluate the selected nested-RH structure with identifiable coefficients."""
    a_vertical, a_high, bias = params
    low_mid = features["rh_low_mean"] + features["rh_mid_mean"]
    return jnp.tanh(
        a_vertical * low_mid * features["rh_vertical_range"] ** 3
        + a_high * low_mid * features["rh_high_mean"]
        + bias
    )


def _fit(predictor, features, target, initial, bounds) -> tuple[np.ndarray, dict]:
    target = jnp.asarray(target)

    @jax.jit
    def objective(params):
        return jnp.mean((predictor(features, params) - target) ** 2)

    value_and_gradient = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(params):
        value, gradient = value_and_gradient(jnp.asarray(params))
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        scipy_objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1_000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not result.success:
        raise RuntimeError(f"calibration failed: {result.message}")
    return np.asarray(result.x), {
        "method": "L-BFGS-B",
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "final_train_mse": float(result.fun),
    }


def _features(data: xr.Dataset, fmask_by_site: dict[str, float]) -> dict[str, jax.Array]:
    sites = np.asarray(data["site_facility"].values).astype(str)
    return {
        name: jnp.asarray(data[name].values)
        for name in ("rh_cloudc_max", "precip_mm_day", "gse", "rh_lowest")
    } | {"fmask": jnp.asarray([fmask_by_site[site] for site in sites])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--nested-features",
        type=Path,
        help="optional pooled feature cache for calibrating the nested-RH equation",
    )
    parser.add_argument("--terrain-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    with xr.open_dataset(args.features) as raw:
        data = raw.load()
    terrain = json.loads(args.terrain_config.read_text())["sites"]
    fmask_by_site = {site: float(config["fmask"]) for site, config in terrain.items()}
    all_features = _features(data, fmask_by_site)
    labels = np.asarray(data["split"].values).astype(str)
    target = np.asarray(data["target"].values)
    raw_default = np.asarray(speedy_raw_sum(all_features, jnp.asarray(RAW_INITIAL)))
    archived_default = np.asarray(data["speedy_raw_sum"].values)
    if not np.allclose(raw_default, archived_default, rtol=2.0e-6, atol=2.0e-6):
        difference = float(np.max(np.abs(raw_default - archived_default)))
        raise ValueError(f"feature formula does not reproduce archived raw SPEEDY output: {difference}")

    train = labels == "train"
    train_features = {name: values[train] for name, values in all_features.items()}
    raw_params, raw_fit = _fit(speedy_raw_sum, train_features, target[train], RAW_INITIAL, RAW_BOUNDS)
    symbolic_params, symbolic_fit = _fit(
        symbolic_cloud, train_features, target[train], SYMBOLIC_INITIAL, SYMBOLIC_BOUNDS
    )
    predictions = {
        "raw_default": raw_default,
        "raw_fitted": np.asarray(speedy_raw_sum(all_features, jnp.asarray(raw_params))),
        "symbolic_initial": np.asarray(symbolic_cloud(all_features, jnp.asarray(SYMBOLIC_INITIAL))),
        "symbolic_fitted": np.asarray(symbolic_cloud(all_features, jnp.asarray(symbolic_params))),
    }
    report = {
        "training_split": "train",
        "raw_speedy": {
            "equation": "cloudc + cloudstr, default SPEEDY hard-hinge diagnosis",
            "parameters": ["rhcl1", "wpcl", "clsmax", "clsminl"],
            "bounds": RAW_BOUNDS,
            "initial_params": RAW_INITIAL.tolist(),
            "fitted_params": raw_params.tolist(),
            "optimizer": raw_fit,
        },
        "symbolic": {
            "equation": "tanh(a_rh * rh_cloudc_max^4 + a_gse * sqrt(abs(gse)) * rh_lowest^2 + bias)",
            "parameters": ["a_rh", "a_gse", "bias"],
            "bounds": SYMBOLIC_BOUNDS,
            "initial_params": SYMBOLIC_INITIAL.tolist(),
            "fitted_params": symbolic_params.tolist(),
            "optimizer": symbolic_fit,
        },
        "splits": {},
    }
    if args.nested_features is not None:
        with xr.open_dataset(args.nested_features) as raw:
            nested_data = raw.load()
        nested_labels = np.asarray(nested_data["split"].values).astype(str)
        nested_target = np.asarray(nested_data["target"].values)
        if not np.array_equal(nested_labels, labels) or not np.allclose(
            nested_target, target, rtol=0.0, atol=0.0
        ):
            raise ValueError("nested feature cache does not align with the calibration cache")
        nested_features = {
            name: jnp.asarray(nested_data[name].values)
            for name in ("rh_low_mean", "rh_mid_mean", "rh_vertical_range", "rh_high_mean")
        }
        nested_train_features = {name: values[train] for name, values in nested_features.items()}
        nested_params, nested_fit = _fit(
            nested_symbolic_cloud,
            nested_train_features,
            target[train],
            NESTED_INITIAL,
            NESTED_BOUNDS,
        )
        predictions.update({
            "nested_symbolic_initial": np.asarray(
                nested_symbolic_cloud(nested_features, jnp.asarray(NESTED_INITIAL))
            ),
            "nested_symbolic_fitted": np.asarray(
                nested_symbolic_cloud(nested_features, jnp.asarray(nested_params))
            ),
        })
        report["nested_symbolic"] = {
            "equation": (
                "tanh(a_vertical * (rh_low_mean + rh_mid_mean) * rh_vertical_range^3 "
                "+ a_high * (rh_low_mean + rh_mid_mean) * rh_high_mean + bias)"
            ),
            "parameters": ["a_vertical", "a_high", "bias"],
            "bounds": NESTED_BOUNDS,
            "initial_params": NESTED_INITIAL.tolist(),
            "fitted_params": nested_params.tolist(),
            "optimizer": nested_fit,
        }
    for split in ("train", "validation", "test"):
        rows = labels == split
        report["splits"][split] = {
            name: _metrics(prediction[rows], target[rows])
            for name, prediction in predictions.items()
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "predictions.npz", target=target, split=labels, **predictions)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
