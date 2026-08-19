"""Test whether SPEEDY cloud-component partition explains nested-RH RSUT error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pandas as pd
import xarray as xr
from jax.tree_util import tree_map

from evaluate_speedy_era5_diagnostic import (
    DEFAULT_RSUT_STORE,
    area_weighted_distribution,
    load_rsut_targets,
    run_scheme_predictions,
    stratified_dates,
    synoptic_times,
)
from evaluate_speedy_era5_smoke import (
    DEFAULT_ERA5_CLOUD_STORE,
    DEFAULT_ERA5_STORE,
    area_weighted_bias,
    area_weighted_rmse,
    era5_2d_on_model_grid,
    era5_on_speedy_sigma,
)
from jcm.date import DateData
from jcm.forcing import ForcingData
from jcm.physics.radiation.speedy_shortwave import get_shortwave_rad_fluxes
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords
from jcm.physics.speedy.speedy_terms import _data_from_diagnostics
from jcm.terrain import TerrainData


def repartition_cloud_cover(nested_sw, baseline_sw):
    """Preserve nested total cover but apply the baseline component ratio."""
    baseline_total = baseline_sw.cloudc + baseline_sw.cloudstr
    stratiform_fraction = jnp.where(
        baseline_total > 1.0e-8,
        baseline_sw.cloudstr / jnp.maximum(baseline_total, 1.0e-8),
        0.0,
    )
    stratiform_fraction = jnp.clip(stratiform_fraction, 0.0, 1.0)
    nested_total = nested_sw.cloudc + nested_sw.cloudstr
    return nested_sw.copy(
        cloudc=nested_total * (1.0 - stratiform_fraction),
        cloudstr=nested_total * stratiform_fraction,
    )


def counterfactual_rsut(
    states,
    times: list[pd.Timestamp],
    coords,
    terrain,
    forcing,
    cloud_targets: np.ndarray,
    nested_predictions,
    baseline_predictions,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Rerun shortwave after isolated component and cloud-top substitutions."""
    speedy_coords = SpeedyCoords.from_coordinate_system(coords)
    nodal_shape = tuple(states[0].normalized_surface_pressure.shape)
    parameters = Parameters.default()
    rsut = {
        "sr_nested_rh_speedy_partition": [],
        "sr_nested_rh_speedy_cloud_top": [],
        "sr_nested_rh_speedy_partition_and_cloud_top": [],
    }
    common_operator_rsut = {
        "era5_total_cloud": [],
        "calibrated_speedy_total_cloud": [],
    }
    primary_cover = []
    stratiform_cover = []
    for index, (state, timestamp) in enumerate(zip(states, times, strict=True)):
        diagnostics = tree_map(lambda value: value[index], nested_predictions.physics_data)
        data = _data_from_diagnostics(diagnostics, speedy_coords, nodal_shape, 8)
        baseline_sw = tree_map(
            lambda value: value[index],
            baseline_predictions.physics_data["_shortwave_rad"],
        )
        partitioned = repartition_cloud_cover(data.shortwave_rad, baseline_sw)
        date = DateData.set_date(
            jdt.to_datetime(timestamp.isoformat()),
            model_step=0,
            dt_seconds=1800.0,
            calendar="gregorian",
        )
        forcing_now = forcing.select(date, calendar="gregorian")
        variants = {
            "sr_nested_rh_speedy_partition": partitioned,
            "sr_nested_rh_speedy_cloud_top": data.shortwave_rad.copy(
                icltop=baseline_sw.icltop
            ),
            "sr_nested_rh_speedy_partition_and_cloud_top": partitioned.copy(
                icltop=baseline_sw.icltop
            ),
            "era5_total_cloud": data.shortwave_rad.copy(
                cloudc=jnp.clip(cloud_targets[index], 0.0, 1.0),
                cloudstr=jnp.zeros_like(data.shortwave_rad.cloudstr),
            ),
            "calibrated_speedy_total_cloud": data.shortwave_rad.copy(
                cloudc=jnp.clip(baseline_sw.cloudc + baseline_sw.cloudstr, 0.0, 1.0),
                cloudstr=jnp.zeros_like(data.shortwave_rad.cloudstr),
            ),
        }
        for name, shortwave in variants.items():
            variant_data = data.copy(shortwave_rad=shortwave)
            _, variant_data = get_shortwave_rad_fluxes(
                state, variant_data, parameters, forcing_now, terrain
            )
            outgoing = np.asarray(
                variant_data.shortwave_rad.fsol - variant_data.shortwave_rad.ftop
            )
            if name in rsut:
                rsut[name].append(outgoing)
            else:
                common_operator_rsut[name].append(outgoing)
        primary_cover.append(np.asarray(partitioned.cloudc))
        stratiform_cover.append(np.asarray(partitioned.cloudstr))
    return (
        {name: np.stack(values) for name, values in rsut.items()},
        {name: np.stack(values) for name, values in common_operator_rsut.items()},
        np.stack(primary_cover),
        np.stack(stratiform_cover),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", action="append")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--state-store", type=Path, default=Path(DEFAULT_ERA5_STORE))
    parser.add_argument(
        "--cloud-store", type=Path, default=Path(DEFAULT_ERA5_CLOUD_STORE)
    )
    parser.add_argument("--rsut-store", default=DEFAULT_RSUT_STORE)
    parser.add_argument("--rsut-target-cache", type=Path, required=True)
    parser.add_argument(
        "--terrain", type=Path, default=Path("jcm/data/bc/t30/clim/terrain.nc")
    )
    parser.add_argument(
        "--forcing", type=Path, default=Path("jcm/data/bc/t30/clim/forcing.nc")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/armbe_sgp/outputs/era5_partition_counterfactual_24window.json"
        ),
    )
    args = parser.parse_args()
    dates = (
        [pd.Timestamp(value).normalize() for value in args.date]
        if args.date
        else stratified_dates(args.start_year, args.end_year)
    )
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    if len(set(dates)) != len(dates):
        raise ValueError("--date values must be unique")
    times = [time for date in dates for time in synoptic_times(date)]

    coords = get_speedy_coords(layers=8, spectral_truncation=31)
    terrain = TerrainData.from_file(args.terrain, coords=coords)
    forcing = ForcingData.from_file(args.forcing, coords=coords)
    state_source = xr.open_zarr(args.state_store, consolidated=True)
    cloud_source = xr.open_zarr(args.cloud_store, consolidated=True)
    states = [era5_on_speedy_sigma(state_source, time, coords) for time in times]
    cloud_targets = np.stack([
        era5_2d_on_model_grid(cloud_source, time, coords, "total_cloud_cover")
        for time in times
    ])
    rsut_targets = load_rsut_targets(
        dates, coords, args.rsut_store, args.rsut_target_cache
    )
    baseline = run_scheme_predictions(
        "calibrated_speedy", states, times, coords, terrain, forcing
    )
    nested = run_scheme_predictions(
        "sr_nested_rh", states, times, coords, terrain, forcing
    )
    baseline_sw = baseline.physics_data["_shortwave_rad"]
    nested_sw = nested.physics_data["_shortwave_rad"]
    counterfactuals, common_operator_rsut, primary_cover, stratiform_cover = (
        counterfactual_rsut(
            states,
            times,
            coords,
            terrain,
            forcing,
            cloud_targets,
            nested,
            baseline,
        )
    )

    predictions = {
        "calibrated_speedy": (
            np.asarray(baseline_sw.cloudc) + np.asarray(baseline_sw.cloudstr),
            np.asarray(baseline_sw.fsol) - np.asarray(baseline_sw.ftop),
        ),
        "sr_nested_rh": (
            np.asarray(nested_sw.cloudc) + np.asarray(nested_sw.cloudstr),
            np.asarray(nested_sw.fsol) - np.asarray(nested_sw.ftop),
        ),
        "sr_nested_rh_speedy_partition": (
            primary_cover + stratiform_cover,
            counterfactuals["sr_nested_rh_speedy_partition"],
        ),
        "sr_nested_rh_speedy_cloud_top": (
            np.asarray(nested_sw.cloudc) + np.asarray(nested_sw.cloudstr),
            counterfactuals["sr_nested_rh_speedy_cloud_top"],
        ),
        "sr_nested_rh_speedy_partition_and_cloud_top": (
            primary_cover + stratiform_cover,
            counterfactuals["sr_nested_rh_speedy_partition_and_cloud_top"],
        ),
    }
    weights = np.asarray(coords.horizontal.quadrature_weights)
    incoming_solar = np.asarray(nested_sw.fsol)
    windows = []
    for index, date in enumerate(dates):
        rows = slice(4 * index, 4 * (index + 1))
        window = {"date": date.date().isoformat()}
        for name, (cloud, shortwave) in predictions.items():
            daily_rsut = np.mean(shortwave[rows], axis=0)
            candidate_common_operator = (
                common_operator_rsut["calibrated_speedy_total_cloud"]
                if name == "calibrated_speedy"
                else predictions["sr_nested_rh"][1]
            )
            daily_common_operator = np.mean(candidate_common_operator[rows], axis=0)
            daily_target_operator = np.mean(
                common_operator_rsut["era5_total_cloud"][rows], axis=0
            )
            window[name] = {
                "cloud_cover_rmse": area_weighted_rmse(
                    cloud[rows], cloud_targets[rows], weights
                ),
                "cloud_cover_insolation_weighted_rmse": area_weighted_rmse(
                    cloud[rows],
                    cloud_targets[rows],
                    weights * incoming_solar[rows],
                ),
                "cloud_amount_radiative_proxy_rmse_w_m2": area_weighted_rmse(
                    daily_common_operator,
                    daily_target_operator,
                    weights,
                ),
                "rsut_rmse_w_m2": area_weighted_rmse(
                    daily_rsut, rsut_targets[index], weights
                ),
                "rsut_bias_w_m2": area_weighted_bias(
                    daily_rsut, rsut_targets[index], weights
                ),
            }
        windows.append(window)

    metric_names = tuple(windows[0]["calibrated_speedy"])
    report = {
        "method": {
            "partition": (
                "nested total cloud and cloud top fixed; cloudc/cloudstr changed "
                "to calibrated-SPEEDY local component ratio"
            ),
            "cloud_top": (
                "nested total cloud and one-component treatment fixed; icltop "
                "changed to calibrated-SPEEDY diagnosis"
            ),
            "combined": "both substitutions applied to fixed nested total cloud",
            "insolation_weighted_cloud_rmse": (
                "cloud squared error weighted by Gaussian area weight times "
                "SPEEDY daily-mean incoming TOA solar flux"
            ),
            "cloud_amount_radiative_proxy": (
                "candidate and ERA5 total cloud passed as one-component cloudc "
                "through the same nested-state cloud top and SPEEDY shortwave operator"
            ),
        },
        "dates": [date.date().isoformat() for date in dates],
        "mean_metrics": {
            name: {
                metric: float(np.mean([window[name][metric] for window in windows]))
                for metric in metric_names
            }
            for name in predictions
        },
        "counterfactual_component_distributions": {
            "cloudc": area_weighted_distribution(primary_cover, weights),
            "cloudstr": area_weighted_distribution(stratiform_cover, weights),
        },
        "maximum_total_cloud_difference_from_nested": float(
            np.max(np.abs(primary_cover + stratiform_cover - predictions["sr_nested_rh"][0]))
        ),
        "windows": windows,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
