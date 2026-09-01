"""Evaluate the published Grundner et al. EQ4 on the June L47 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from evaluate_echam_cloud_baselines import _metrics


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "outputs/echam_layer_cloud_june_2018/echam_l47_june.nc"
DEFAULT_OUTPUT = HERE / "outputs/echam_layer_cloud_june_2018/grundner_eq4.json"


def liquid_relative_humidity(
    specific_humidity: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Grundner liquid-water relative humidity definition, dimensionless."""
    return (
        0.00263
        * pressure
        * specific_humidity
        * np.exp(17.67 * (273.15 - temperature) / (temperature - 29.65))
    )


def grundner_eq4(
    relative_humidity: np.ndarray,
    temperature: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    rh_gradient_height: np.ndarray,
    enforce_rh_monotonicity: bool = True,
) -> np.ndarray:
    """Calculate published EQ4 cloud fraction with its gate and bounds."""
    rh = relative_humidity
    if enforce_rh_monotonicity:
        rh_floor = 0.317 - 1.623e-4 * (temperature - 257.06) ** 2
        rh = np.maximum(rh, rh_floor)
    centered_rh = rh - 0.6025
    centered_temperature = temperature - 257.06
    i1 = (
        0.4435
        + 1.1593 * centered_rh
        - 0.0145 * centered_temperature
        + 0.5 * 4.06 * centered_rh**2
        + 0.5
        * 1.3176e-3
        * centered_temperature**2
        * centered_rh
    )
    i2 = 584.8036**3 * (rh_gradient_height + 0.003) * rh_gradient_height**2
    i3 = -1.0 / (qc / 1.1573e-6 + qi / 3.073e-7 + 1.06)
    fraction = np.clip(i1 + i2 + i3, 0.0, 1.0)
    return np.where((qc + qi) == 0.0, 0.0, fraction)


def _height_gradient(values: np.ndarray, height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gradient = np.full_like(values, np.nan, dtype=float)
    for profile in range(values.shape[0]):
        selected = valid[profile] & np.isfinite(values[profile]) & np.isfinite(height[profile])
        indices = np.flatnonzero(selected)
        if indices.size >= 2:
            gradient[profile, indices] = np.gradient(
                values[profile, indices], height[profile, indices]
            )
    return gradient


def evaluate(dataset: xr.Dataset) -> dict[str, object]:
    valid = dataset.layer_valid.values.astype(bool)
    rh = liquid_relative_humidity(
        dataset.specific_humidity.values,
        dataset.pressure.values,
        dataset.temperature.values,
    )
    rh_gradient = _height_gradient(rh, dataset.height.values, valid)
    finite = (
        valid
        & np.isfinite(rh)
        & np.isfinite(rh_gradient)
        & np.isfinite(dataset.qc.values)
        & np.isfinite(dataset.qi.values)
        & np.isfinite(dataset.cloud_fraction.values)
    )
    results: dict[str, object] = {
        "equation": "Grundner et al. (2024) EQ4 / paper Eq. 10",
        "rh_definition": "liquid-water RH from paper",
        "gradient": "dRH/dz in m-1, geometric height increasing upward",
        "outer_holdout_evaluated": False,
        "splits": {},
    }
    for split in ("train", "validation"):
        selected = finite & (dataset.split.values == split)[:, None]
        profile = np.broadcast_to(
            dataset.profile.values[:, None], selected.shape
        )[selected]
        arguments = (
            rh[selected],
            dataset.temperature.values[selected],
            dataset.qc.values[selected],
            dataset.qi.values[selected],
            rh_gradient[selected],
        )
        target = dataset.cloud_fraction.values[selected]
        results["splits"][split] = {
            "eq4": _metrics(target, grundner_eq4(*arguments, False), profile),
            "eq4_rh_monotonic": _metrics(
                target, grundner_eq4(*arguments, True), profile
            ),
        }
        common = selected & np.isfinite(dataset.rh_gradient_log_pressure.values)
        common_profile = np.broadcast_to(
            dataset.profile.values[:, None], common.shape
        )[common]
        common_arguments = (
            rh[common],
            dataset.temperature.values[common],
            dataset.qc.values[common],
            dataset.qi.values[common],
            rh_gradient[common],
        )
        results["splits"][split]["eq4_rh_monotonic_common_core_rows"] = _metrics(
            dataset.cloud_fraction.values[common],
            grundner_eq4(*common_arguments, True),
            common_profile,
        )
    return results


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
