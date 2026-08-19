"""Evaluate independent same-time SPEEDY cloud and RSUT diagnostics against ERA5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax_datetime as jdt
import numpy as np
import pandas as pd
import xarray as xr

from evaluate_speedy_era5_smoke import (
    DEFAULT_ERA5_CLOUD_STORE,
    DEFAULT_ERA5_STORE,
    SCHEMES,
    area_weighted_bias,
    area_weighted_rmse,
    era5_2d_on_model_grid,
    era5_on_speedy_sigma,
    scheme_configuration,
)
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.prescribed_state_model import PrescribedStateModel
from jcm.terrain import TerrainData

DEFAULT_RSUT_STORE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)
SYNOPTIC_HOURS = (0, 6, 12, 18)
RSUT_DOWN = "mean_top_downward_short_wave_radiation_flux"
RSUT_NET = "mean_top_net_short_wave_radiation_flux"


def stratified_dates(start_year: int, end_year: int) -> list[pd.Timestamp]:
    """Return two fixed, evenly spaced windows per month over a year range."""
    if end_year < start_year:
        raise ValueError("end_year must not precede start_year")
    return [
        pd.Timestamp(year=year, month=month, day=day)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        for day in (7, 21)
    ]


def default_dates() -> list[pd.Timestamp]:
    """Return two monthly windows spanning 2016 through 2020."""
    return stratified_dates(2016, 2020)


def synoptic_times(date: pd.Timestamp) -> list[pd.Timestamp]:
    """Return the four six-hourly analysis times in one daily window."""
    day = date.normalize()
    return [day + pd.Timedelta(hours=hour) for hour in SYNOPTIC_HOURS]


def era5_daily_rsut(source: xr.Dataset, date: pd.Timestamp, coords) -> np.ndarray:
    """Return four-synoptic-time mean ERA5 outgoing TOA shortwave on the model grid."""
    times = synoptic_times(date)
    rsut = source[RSUT_DOWN].sel(time=times) - source[RSUT_NET].sel(time=times)
    rsut = rsut.mean("time").sortby("latitude")
    rsut = rsut.assign_coords(longitude=rsut.longitude % 360.0).sortby("longitude")
    lons = np.rad2deg(np.asarray(coords.horizontal.longitudes)) % 360.0
    lats = np.rad2deg(np.asarray(coords.horizontal.latitudes))
    return np.asarray(
        rsut.interp(longitude=lons, latitude=lats).transpose("longitude", "latitude").values
    )


def load_rsut_targets(
    dates: list[pd.Timestamp],
    coords,
    source_store: str,
    cache_path: Path | None,
) -> np.ndarray:
    """Load cached daily RSUT targets or derive and cache them from hourly ERA5."""
    lons = np.rad2deg(np.asarray(coords.horizontal.longitudes)) % 360.0
    lats = np.rad2deg(np.asarray(coords.horizontal.latitudes))
    requested_times = np.asarray(dates, dtype="datetime64[ns]")
    if cache_path is not None and cache_path.exists():
        with xr.open_dataset(cache_path) as cached:
            try:
                selected = cached["rsut_w_m2"].sel(time=requested_times)
            except KeyError as error:
                raise ValueError(
                    f"RSUT cache does not contain every requested date: {cache_path}"
                ) from error
            targets = np.asarray(selected.transpose("time", "longitude", "latitude").values)
        expected_shape = (len(dates), len(lons), len(lats))
        if targets.shape != expected_shape:
            raise ValueError(f"RSUT cache has shape {targets.shape}; expected {expected_shape}")
        return targets

    source = xr.open_zarr(
        source_store,
        consolidated=True,
        storage_options={"token": "anon"} if source_store.startswith("gs://") else None,
    )
    targets = np.stack([era5_daily_rsut(source, date, coords) for date in dates])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"rsut_w_m2": (("time", "longitude", "latitude"), targets)},
            coords={"time": requested_times, "longitude": lons, "latitude": lats},
            attrs={
                "source": source_store,
                "derivation": f"four-synoptic-time mean of {RSUT_DOWN} minus {RSUT_NET}",
            },
        ).to_netcdf(cache_path)
    return targets


def bootstrap_mean_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    """Return paired mean difference and a window-bootstrap confidence interval."""
    difference = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    bootstrap = np.mean(difference[indices], axis=1)
    return {
        "mean_difference": float(np.mean(difference)),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
    }


def area_weighted_distribution(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Return weighted mean and standard deviation over finite values."""
    values = np.asarray(values)
    weights = np.broadcast_to(weights, values.shape)
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError("distribution has no finite values")
    total_weight = np.sum(weights[valid])
    mean = np.sum(weights[valid] * values[valid]) / total_weight
    variance = np.sum(weights[valid] * (values[valid] - mean) ** 2) / total_weight
    return {"mean": float(mean), "standard_deviation": float(np.sqrt(variance))}


def run_scheme_predictions(
    scheme: str,
    states,
    times: list[pd.Timestamp],
    coords,
    terrain,
    forcing,
):
    """Evaluate one closure independently on every prescribed ERA5 state."""
    parameters, selector = scheme_configuration(scheme)
    model = PrescribedStateModel(
        physics=speedy_physics(parameters=parameters, cloud_cover_scheme=selector),
        coords=coords,
        terrain=terrain,
        dt_seconds=1800.0,
        start_date=jdt.to_datetime(times[0].isoformat()),
        calendar="gregorian",
    )
    offsets = np.asarray([(time - times[0]).total_seconds() / 86400.0 for time in times])
    return model.run(
        states,
        forcing=forcing,
        times=offsets,
    )


def run_scheme_diagnostics(
    scheme: str,
    states,
    times: list[pd.Timestamp],
    coords,
    terrain,
    forcing,
):
    """Return cloud cover and RSUT from independent prescribed-state physics."""
    predictions = run_scheme_predictions(
        scheme, states, times, coords, terrain, forcing
    )
    shortwave = predictions.physics_data["_shortwave_rad"]
    cloud_cover = np.asarray(shortwave.cloudc) + np.asarray(shortwave.cloudstr)
    rsut = np.asarray(shortwave.fsol) - np.asarray(shortwave.ftop)
    return cloud_cover, rsut


def main() -> None:
    overall_start = time.perf_counter()
    timings = {}

    def finish_stage(name: str, start: float) -> None:
        timings[name] = time.perf_counter() - start
        print(f"[timing] {name}: {timings[name]:.3f} s", flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        action="append",
        help="daily window (YYYY-MM-DD); repeat as needed (default: days 7/21 monthly, 2016-2020)",
    )
    parser.add_argument("--state-store", type=Path, default=Path(DEFAULT_ERA5_STORE))
    parser.add_argument("--cloud-store", type=Path, default=Path(DEFAULT_ERA5_CLOUD_STORE))
    parser.add_argument("--rsut-store", default=DEFAULT_RSUT_STORE)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument(
        "--rsut-target-cache",
        type=Path,
        help="optional NetCDF cache for processed daily RSUT on the model grid",
    )
    parser.add_argument(
        "--terrain",
        type=Path,
        default=Path("jcm/data/bc/t30/clim/terrain.nc"),
    )
    parser.add_argument(
        "--forcing",
        type=Path,
        default=Path("jcm/data/bc/t30/clim/forcing.nc"),
    )
    parser.add_argument("--scheme", action="append", choices=SCHEMES)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/armbe_sgp/outputs/era5_nested_rh_diagnostic.json"),
    )
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap-draws must be positive")

    dates = (
        [pd.Timestamp(value).normalize() for value in args.date]
        if args.date
        else stratified_dates(args.start_year, args.end_year)
    )
    if len(set(dates)) != len(dates):
        raise ValueError("--date values must be unique")
    schemes = args.scheme or list(SCHEMES[:3])
    if "calibrated_speedy" not in schemes:
        raise ValueError("calibrated_speedy is required as the paired baseline")

    print(
        f"[run] {len(dates)} daily windows, {4 * len(dates)} prescribed states, "
        f"schemes={','.join(schemes)}",
        flush=True,
    )
    stage_start = time.perf_counter()
    state_source = xr.open_zarr(args.state_store, consolidated=True)
    cloud_source = xr.open_zarr(args.cloud_store, consolidated=True)
    coords = get_speedy_coords(layers=8, spectral_truncation=31)
    terrain = TerrainData.from_file(args.terrain, coords=coords)
    forcing = ForcingData.from_file(args.forcing, coords=coords)
    weights = np.asarray(coords.horizontal.quadrature_weights)
    finish_stage("initialization", stage_start)

    all_times = [time for date in dates for time in synoptic_times(date)]
    stage_start = time.perf_counter()
    states = [era5_on_speedy_sigma(state_source, time, coords) for time in all_times]
    finish_stage("atmospheric_state_preparation", stage_start)
    stage_start = time.perf_counter()
    cloud_targets = np.stack([
        era5_2d_on_model_grid(cloud_source, time, coords, "total_cloud_cover")
        for time in all_times
    ])
    finish_stage("cloud_target_preparation", stage_start)
    stage_start = time.perf_counter()
    rsut_targets = load_rsut_targets(dates, coords, args.rsut_store, args.rsut_target_cache)
    finish_stage("rsut_target_preparation", stage_start)
    if not np.isfinite(cloud_targets).any() or not np.isfinite(rsut_targets).any():
        raise ValueError("ERA5 diagnostic targets contain no finite values")

    predictions = {}
    for scheme in schemes:
        stage_start = time.perf_counter()
        predictions[scheme] = run_scheme_diagnostics(
            scheme, states, all_times, coords, terrain, forcing
        )
        finish_stage(f"physics_{scheme}", stage_start)
    stage_start = time.perf_counter()
    daily_rsut_predictions = {
        scheme: np.stack([
            np.mean(rsut_prediction[4 * index : 4 * (index + 1)], axis=0)
            for index in range(len(dates))
        ])
        for scheme, (_, rsut_prediction) in predictions.items()
    }
    windows = []
    for index, date in enumerate(dates):
        rows = slice(4 * index, 4 * (index + 1))
        window = {"date": date.date().isoformat()}
        for scheme, (cloud_prediction, rsut_prediction) in predictions.items():
            daily_rsut_prediction = daily_rsut_predictions[scheme][index]
            window[scheme] = {
                "cloud_cover_rmse": area_weighted_rmse(
                    cloud_prediction[rows], cloud_targets[rows], weights
                ),
                "cloud_cover_bias": area_weighted_bias(
                    cloud_prediction[rows], cloud_targets[rows], weights
                ),
                "rsut_rmse_w_m2": area_weighted_rmse(
                    daily_rsut_prediction, rsut_targets[index], weights
                ),
                "rsut_bias_w_m2": area_weighted_bias(
                    daily_rsut_prediction, rsut_targets[index], weights
                ),
            }
        windows.append(window)

    metric_names = tuple(windows[0][schemes[0]])
    mean_metrics = {
        scheme: {
            metric: float(np.mean([window[scheme][metric] for window in windows]))
            for metric in metric_names
        }
        for scheme in schemes
    }
    distribution_statistics = {
        "era5": {
            "cloud_cover": area_weighted_distribution(cloud_targets, weights),
            "rsut_w_m2": area_weighted_distribution(rsut_targets, weights),
        },
        **{
            scheme: {
                "cloud_cover": area_weighted_distribution(predictions[scheme][0], weights),
                "rsut_w_m2": area_weighted_distribution(
                    daily_rsut_predictions[scheme], weights
                ),
            }
            for scheme in schemes
        },
    }
    normalized_mean_errors = {}
    for scheme in schemes:
        normalized_mean_errors[scheme] = {}
        for variable, rmse_metric, bias_metric in (
            ("cloud_cover", "cloud_cover_rmse", "cloud_cover_bias"),
            ("rsut_w_m2", "rsut_rmse_w_m2", "rsut_bias_w_m2"),
        ):
            target_stats = distribution_statistics["era5"][variable]
            normalized_mean_errors[scheme][variable] = {
                "rmse_percent_of_era5_mean": (
                    100.0 * mean_metrics[scheme][rmse_metric] / abs(target_stats["mean"])
                ),
                "rmse_percent_of_era5_standard_deviation": (
                    100.0
                    * mean_metrics[scheme][rmse_metric]
                    / target_stats["standard_deviation"]
                ),
                "bias_percent_of_era5_mean": (
                    100.0 * mean_metrics[scheme][bias_metric] / abs(target_stats["mean"])
                ),
            }
    paired_differences = {}
    for scheme_index, scheme in enumerate(schemes):
        if scheme == "calibrated_speedy":
            continue
        paired_differences[scheme] = {}
        for metric_index, metric in enumerate(("cloud_cover_rmse", "rsut_rmse_w_m2")):
            paired_differences[scheme][metric] = bootstrap_mean_difference(
                np.asarray([window[scheme][metric] for window in windows]),
                np.asarray([window["calibrated_speedy"][metric] for window in windows]),
                seed=args.seed + 10 * scheme_index + metric_index,
                draws=args.bootstrap_draws,
            )
    finish_stage("metric_and_bootstrap_evaluation", stage_start)
    timings["total_before_serialization"] = time.perf_counter() - overall_start

    report = {
        "method": {
            "sampling_unit": "independent daily window",
            "analysis_hours_utc": list(SYNOPTIC_HOURS),
            "cloud_target": "same-time ERA5 total_cloud_cover at four analyses",
            "rsut_target": (
                "four-synoptic-time mean of ERA5 mean_top_downward_short_wave_radiation_flux "
                "minus mean_top_net_short_wave_radiation_flux"
            ),
            "model_execution": "independent prescribed-state physics diagnostic; no forecast rollout",
            "area_weighting": "model Gaussian quadrature weights",
        },
        "stores": {
            "state": str(args.state_store),
            "cloud": str(args.cloud_store),
            "rsut": str(args.rsut_store),
            "rsut_target_cache": (
                str(args.rsut_target_cache) if args.rsut_target_cache else None
            ),
            "forcing": str(args.forcing),
            "terrain": str(args.terrain),
        },
        "schemes": schemes,
        "window_count": len(windows),
        "timings_seconds": timings,
        "evaluation_points_per_scheme": {
            "independent_daily_windows": len(windows),
            "prescribed_atmospheric_states": len(all_times),
            "cloud_cover_nominal": int(cloud_targets.size),
            "cloud_cover_finite": int(np.isfinite(cloud_targets).sum()),
            "rsut_nominal": int(rsut_targets.size),
            "rsut_finite": int(np.isfinite(rsut_targets).sum()),
        },
        "distribution_statistics": distribution_statistics,
        "mean_metrics": mean_metrics,
        "normalized_mean_errors": normalized_mean_errors,
        "paired_rmse_differences_vs_calibrated_speedy": paired_differences,
        "windows": windows,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
