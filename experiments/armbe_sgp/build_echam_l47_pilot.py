"""Remap the June native-height pilot onto ECHAM L47 pressure layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

import jcm.constants as constants
from jcm.physics.echam.echam_levels import get_echam_levels


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "outputs/echam_layer_cloud_june_2018"
DEFAULT_OUTPUT = DEFAULT_INPUT / "echam_l47_june.nc"
MIN_COVERAGE = 0.90


def _weighted_mean(values: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    weight = overlap[:, valid]
    denominator = weight.sum(axis=1)
    numerator = weight @ values[valid]
    return np.divide(
        numerator,
        denominator,
        out=np.full(overlap.shape[0], np.nan),
        where=denominator > 0.0,
    )


def remap_profile(
    pressure: np.ndarray,
    density: np.ndarray,
    height: np.ndarray,
    temperature: np.ndarray,
    relative_humidity: np.ndarray,
    specific_humidity: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    cloud_fraction: np.ndarray,
    source_valid: np.ndarray,
    pressure_half: np.ndarray,
    dz: float = 30.0,
) -> dict[str, np.ndarray | float]:
    """Conservatively remap one observed profile onto pressure layers."""
    source_valid = source_valid & np.isfinite(pressure) & np.isfinite(density)
    source_dp = density * constants.grav * dz
    source_top = pressure - 0.5 * source_dp
    source_bottom = pressure + 0.5 * source_dp
    model_top = pressure_half[:-1]
    model_bottom = pressure_half[1:]
    model_dp = model_bottom - model_top
    if not np.all(model_dp > 0.0):
        raise ValueError("ECHAM pressure boundaries must increase top to bottom")
    overlap = np.maximum(
        0.0,
        np.minimum(model_bottom[:, None], source_bottom[None, :])
        - np.maximum(model_top[:, None], source_top[None, :]),
    )
    valid_overlap = np.where(source_valid[None, :], overlap, 0.0)
    observed_dp = valid_overlap.sum(axis=1)
    coverage = observed_dp / model_dp

    qc_numerator = valid_overlap @ np.where(np.isfinite(qc), qc, 0.0)
    qi_numerator = valid_overlap @ np.where(np.isfinite(qi), qi, 0.0)
    qc_layer = qc_numerator / model_dp
    qi_layer = qi_numerator / model_dp
    layer_valid = coverage >= MIN_COVERAGE
    qc_layer = np.where(layer_valid, qc_layer, np.nan)
    qi_layer = np.where(layer_valid, qi_layer, np.nan)

    result: dict[str, np.ndarray | float] = {
        "pressure": 0.5 * (model_top + model_bottom),
        "coverage_fraction": coverage,
        "layer_valid": layer_valid,
        "temperature": _weighted_mean(temperature, valid_overlap),
        "relative_humidity": _weighted_mean(relative_humidity, valid_overlap),
        "specific_humidity": _weighted_mean(specific_humidity, valid_overlap),
        "height": _weighted_mean(height, valid_overlap),
        "cloud_fraction": _weighted_mean(cloud_fraction, valid_overlap),
        "qc": qc_layer,
        "qi": qi_layer,
        "source_qc_mass": float(np.sum(np.where(source_valid, qc * source_dp, 0.0)) / constants.grav),
        "source_qi_mass": float(np.sum(np.where(source_valid, qi * source_dp, 0.0)) / constants.grav),
        "remapped_qc_mass": float(np.sum(qc_numerator) / constants.grav),
        "remapped_qi_mass": float(np.sum(qi_numerator) / constants.grav),
    }
    for name in (
        "temperature",
        "relative_humidity",
        "specific_humidity",
        "height",
        "cloud_fraction",
    ):
        result[name] = np.where(layer_valid, result[name], np.nan)
    return result


def _split(timestamp: np.datetime64) -> str:
    day = int(str(timestamp.astype("datetime64[D]"))[-2:])
    if day <= 16:
        return "train"
    if day <= 21:
        return "validation"
    return "outer_holdout"


def build_dataset(input_root: Path) -> tuple[xr.Dataset, dict[str, object]]:
    paths = sorted(input_root.glob("*/observed_atmosphere_paired.nc"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 daily paired caches, found {len(paths)}")
    levels = get_echam_levels(47)
    a_half = np.asarray(levels.a_boundaries, dtype=float)
    b_half = np.asarray(levels.b_boundaries, dtype=float)
    records: list[dict[str, np.ndarray | float]] = []
    times: list[np.datetime64] = []
    splits: list[str] = []
    surface_pressures: list[float] = []
    for path in paths:
        with xr.open_dataset(path) as source:
            data = source.load()
        for index, timestamp in enumerate(data.time.values):
            surface_pressure = float(data.surface_pressure.values[index])
            pressure_half = a_half + b_half * surface_pressure
            record = remap_profile(
                pressure=data.pressure.values[index],
                density=data.air_density.values[index],
                height=data.height.values,
                temperature=data.temperature.values[index],
                relative_humidity=data.relative_humidity.values[index],
                specific_humidity=data.specific_humidity.values[index],
                qc=data.qc.values[index],
                qi=data.qi.values[index],
                cloud_fraction=data.armbe_cloud_fraction.values[index],
                source_valid=data.model_sample_valid.values[index].astype(bool),
                pressure_half=pressure_half,
            )
            records.append(record)
            times.append(timestamp)
            splits.append(_split(timestamp))
            surface_pressures.append(surface_pressure)

    level_names = (
        "pressure",
        "coverage_fraction",
        "layer_valid",
        "temperature",
        "relative_humidity",
        "specific_humidity",
        "height",
        "cloud_fraction",
        "qc",
        "qi",
    )
    dataset = xr.Dataset(
        {
            name: (("profile", "level"), np.stack([record[name] for record in records]))
            for name in level_names
        },
        coords={
            "profile": np.arange(len(records)),
            "level": np.arange(47),
            "time": ("profile", np.asarray(times)),
            "split": ("profile", np.asarray(splits)),
        },
        attrs={
            "site": "sgpC1",
            "vertical_grid": "ECHAM6.3 L47 lmidatm",
            "minimum_coverage_fraction": MIN_COVERAGE,
            "outer_holdout_policy": "June 22-30; untouched by fitting and selection",
        },
    )
    dataset["surface_pressure"] = ("profile", surface_pressures)
    for name in ("source_qc_mass", "source_qi_mass", "remapped_qc_mass", "remapped_qi_mass"):
        dataset[name] = ("profile", [record[name] for record in records])
    dataset["rh_gradient_log_pressure"] = (
        ("profile", "level"),
        np.gradient(dataset.relative_humidity.values, axis=1)
        / np.gradient(np.log(dataset.pressure.values), axis=1),
    )
    dataset.pressure.attrs["units"] = "Pa"
    dataset.temperature.attrs["units"] = "K"
    dataset.relative_humidity.attrs["units"] = "1"
    dataset.specific_humidity.attrs["units"] = "kg kg-1"
    dataset.qc.attrs["units"] = "kg kg-1"
    dataset.qi.attrs["units"] = "kg kg-1"
    dataset.cloud_fraction.attrs["units"] = "1"
    dataset.height.attrs["units"] = "m above ground level"
    dataset.rh_gradient_log_pressure.attrs["units"] = "1"

    valid = dataset.layer_valid.values.astype(bool)
    qc_source = dataset.source_qc_mass.values
    qi_source = dataset.source_qi_mass.values
    qc_error = dataset.remapped_qc_mass.values - qc_source
    qi_error = dataset.remapped_qi_mass.values - qi_source
    report = {
        "profiles": dataset.sizes["profile"],
        "levels": dataset.sizes["level"],
        "valid_rows": int(valid.sum()),
        "split_profiles": {
            name: int(np.count_nonzero(dataset.split.values == name))
            for name in ("train", "validation", "outer_holdout")
        },
        "split_valid_rows": {
            name: int(valid[dataset.split.values == name].sum())
            for name in ("train", "validation", "outer_holdout")
        },
        "mass_closure": {
            "qc_max_abs_kg_m-2": float(np.max(np.abs(qc_error))),
            "qi_max_abs_kg_m-2": float(np.max(np.abs(qi_error))),
            "qc_max_relative": float(np.max(np.abs(qc_error) / np.maximum(np.abs(qc_source), 1e-15))),
            "qi_max_relative": float(np.max(np.abs(qi_error) / np.maximum(np.abs(qi_source), 1e-15))),
        },
    }
    return dataset, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    dataset, report = build_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(args.output)
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
