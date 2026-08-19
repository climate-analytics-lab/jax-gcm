"""Run ERA5-initialized SPEEDY default-versus-SR closure smoke tests.

This is not a balanced ERA5 analysis or a JEM-Cal calibration driver. It maps
ERA5 pressure-level profiles to each SPEEDY sigma column using local ERA5
surface pressure, then checks that both otherwise-identical short forecasts are
finite and reports state RMSE against the corresponding ERA5 analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax_datetime as jdt
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState


DEFAULT_ERA5_STORE = (
    "/public/wb2/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
DEFAULT_ERA5_CLOUD_STORE = (
    "/public/wb2/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
SCHEMES = (
    "calibrated_speedy",
    "sr_nested_rh",
    "sr_nested_rh_calibrated",
    "sr_total_cloudc",
)
PROFILE_FIELDS = (
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
)


def era5_on_speedy_sigma(source: xr.Dataset, time: pd.Timestamp, coords) -> PhysicsState:
    """Map one ERA5 analysis to SPEEDY's T31L8 nodal sigma grid.

    ERA5 levels are pressure surfaces, whereas SPEEDY's full levels are sigma
    surfaces. Horizontal interpolation therefore precedes vertical interpolation
    so the target pressure at every nodal column is ``sigma * ps``.
    """
    lons = np.rad2deg(np.asarray(coords.horizontal.longitudes)) % 360.0
    lats = np.rad2deg(np.asarray(coords.horizontal.latitudes))
    sigma = np.asarray(coords.vertical.centers)
    selected = source[list(PROFILE_FIELDS) + ["surface_pressure"]].sel(time=time)
    selected = selected.sortby("latitude")
    selected = selected.assign_coords(longitude=selected.longitude % 360.0).sortby("longitude")
    horizontal = selected.interp(longitude=lons, latitude=lats)
    surface_pressure = horizontal.surface_pressure
    # This local WeatherBench2 store begins at 50 hPa, but SPEEDY's top full
    # level is about 25 hPa. Edge clamping is preferable to unconstrained
    # extrapolation for this smoke test and makes the missing upper-air input
    # explicit in the experiment's scope.
    target_pressure_hpa = sigma[:, None, None] * surface_pressure.values[None, :, :] / 100.0
    target_pressure_hpa = np.clip(
        target_pressure_hpa,
        float(selected.level.min()),
        float(selected.level.max()),
    )
    sigma_pressure_hpa = xr.DataArray(
        target_pressure_hpa,
        dims=("sigma", "longitude", "latitude"),
        coords={"sigma": sigma, "longitude": lons, "latitude": lats},
    )

    def vertical(field: str) -> np.ndarray:
        values = horizontal[field].interp(level=sigma_pressure_hpa).transpose(
            "sigma", "longitude", "latitude"
        )
        return np.asarray(values.values)

    return PhysicsState(
        temperature=jnp.asarray(vertical("temperature")),
        # SPEEDY specific humidity is in g kg-1, unlike ERA5's kg kg-1.
        specific_humidity=jnp.asarray(np.maximum(vertical("specific_humidity") * 1000.0, 0.0)),
        u_wind=jnp.asarray(vertical("u_component_of_wind")),
        v_wind=jnp.asarray(vertical("v_component_of_wind")),
        geopotential=jnp.asarray(vertical("geopotential")),
        normalized_surface_pressure=jnp.asarray(surface_pressure.values / 100000.0),
        tracers={},
    )


def area_weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Return horizontal-area-weighted RMSE over optional leading dimensions."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    weights = np.broadcast_to(weights, prediction.shape)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        raise ValueError("metric has no finite prediction-target pairs")
    return float(np.sqrt(np.sum(weights[valid] * (prediction[valid] - target[valid]) ** 2) / np.sum(weights[valid])))


def area_weighted_bias(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Return horizontal-area-weighted bias over finite prediction-target pairs."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    weights = np.broadcast_to(weights, prediction.shape)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        raise ValueError("metric has no finite prediction-target pairs")
    return float(
        np.sum(weights[valid] * (prediction[valid] - target[valid])) / np.sum(weights[valid])
    )


def state_rmse(prediction, target: PhysicsState, weights: np.ndarray) -> dict[str, float]:
    """Return area-weighted RMSE for the prognostic ERA5-comparable fields."""
    predicted = prediction.dynamics
    fields = ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential")
    metrics = {}
    for field in fields:
        value = np.asarray(getattr(predicted, field))[-1]
        reference = np.asarray(getattr(target, field))
        metrics[f"{field}_rmse"] = area_weighted_rmse(value, reference, weights)
    return metrics


def era5_2d_on_model_grid(source: xr.Dataset, time: pd.Timestamp, coords, field: str) -> np.ndarray:
    """Interpolate one two-dimensional ERA5 field to the model's nodal grid."""
    lons = np.rad2deg(np.asarray(coords.horizontal.longitudes)) % 360.0
    lats = np.rad2deg(np.asarray(coords.horizontal.latitudes))
    selected = source[field].sel(time=time).sortby("latitude")
    selected = selected.assign_coords(longitude=selected.longitude % 360.0).sortby("longitude")
    selected = xr.concat(
        [
            selected.isel(longitude=-1).assign_coords(
                longitude=float(selected.longitude[-1]) - 360.0
            ),
            selected,
            selected.isel(longitude=0).assign_coords(
                longitude=float(selected.longitude[0]) + 360.0
            ),
        ],
        dim="longitude",
    )
    return np.asarray(
        selected.interp(longitude=lons, latitude=lats).transpose("longitude", "latitude").values
    )


def calibrated_speedy_parameters() -> Parameters:
    """Return SPEEDY parameters with the pooled train-only cloud fit."""
    parameters = Parameters.default()
    shortwave = dataclasses.replace(
        parameters.shortwave_radiation,
        rhcl1=jnp.asarray(0.32162740151353536),
        wpcl=jnp.asarray(0.05),
        clsmax=jnp.asarray(0.6399201885756207),
        clsminl=jnp.asarray(0.0),
    )
    return dataclasses.replace(parameters, shortwave_radiation=shortwave)


def scheme_configuration(scheme: str) -> tuple[Parameters, str]:
    """Resolve a benchmark name to SPEEDY parameters and cloud selector."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown benchmark scheme: {scheme!r}")
    if scheme == "calibrated_speedy":
        return calibrated_speedy_parameters(), "speedy"
    return Parameters.default(), scheme


def run_scheme(
    coords,
    initial_state: PhysicsState,
    target: PhysicsState,
    cloud_target: np.ndarray,
    scheme: str,
    lead_days: float,
    initial_time: pd.Timestamp,
) -> dict:
    """Run one closure with fixed initialization and report finite-state metrics."""
    parameters, cloud_cover_scheme = scheme_configuration(scheme)
    model = Model(
        coords=coords,
        physics=speedy_physics(
            parameters=parameters,
            cloud_cover_scheme=cloud_cover_scheme,
        ),
        time_step=30.0,
        start_date=jdt.to_datetime(initial_time.isoformat()),
        calendar="gregorian",
    )
    prediction = model.run(
        initial_state=initial_state,
        total_time=lead_days,
        save_interval=lead_days,
    )
    for field in ("temperature", "specific_humidity", "u_wind", "v_wind", "geopotential"):
        if not np.isfinite(np.asarray(getattr(prediction.dynamics, field))).all():
            raise RuntimeError(f"{scheme} produced non-finite {field} values.")
    weights = np.asarray(coords.horizontal.quadrature_weights)
    metrics = state_rmse(prediction, target, weights)
    shortwave = prediction.physics["_shortwave_rad"]
    model_cloud = (
        np.asarray(shortwave.cloudc)[-1] + np.asarray(shortwave.cloudstr)[-1]
    )
    metrics["total_cloud_cover_rmse"] = area_weighted_rmse(
        model_cloud, cloud_target, weights
    )
    metrics["total_cloud_cover_bias"] = area_weighted_bias(
        model_cloud, cloud_target, weights
    )
    return metrics


def mean_metrics(windows: list[dict], scheme: str) -> dict[str, float]:
    """Return arithmetic mean metrics over independently initialized windows."""
    keys = windows[0][scheme]
    return {key: float(np.mean([window[scheme][key] for window in windows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time",
        action="append",
        default=None,
        help="ERA5 initialization time, repeat for multiple independent windows",
    )
    parser.add_argument("--lead-hours", type=float, default=6.0)
    parser.add_argument("--store", type=Path, default=Path(DEFAULT_ERA5_STORE))
    parser.add_argument(
        "--cloud-store",
        type=Path,
        default=Path(DEFAULT_ERA5_CLOUD_STORE),
        help="six-hour WeatherBench2 store with finite total_cloud_cover",
    )
    parser.add_argument(
        "--scheme",
        action="append",
        choices=SCHEMES,
        help="scheme to evaluate; repeat as needed (default: first three schemes)",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/era5_speedy_smoke.json"))
    args = parser.parse_args()
    if args.lead_hours <= 0.0:
        raise ValueError("--lead-hours must be positive")

    initial_times = [pd.Timestamp(time) for time in args.time or ["2020-01-01T00:00:00"]]
    source = xr.open_zarr(args.store, consolidated=True)
    cloud_source = xr.open_zarr(args.cloud_store, consolidated=True)
    coords = get_speedy_coords(layers=8, spectral_truncation=31)
    schemes = args.scheme or list(SCHEMES[:3])
    lead_days = args.lead_hours / 24.0
    windows = []
    for initial_time in initial_times:
        target_time = initial_time + pd.Timedelta(hours=args.lead_hours)
        initial_state = era5_on_speedy_sigma(source, initial_time, coords)
        target_state = era5_on_speedy_sigma(source, target_time, coords)
        cloud_target = era5_2d_on_model_grid(
            cloud_source, target_time, coords, "total_cloud_cover"
        )
        if not np.isfinite(cloud_target).any():
            raise ValueError(f"cloud store has no finite total_cloud_cover at {target_time}")
        window = {
            "initial_time": initial_time.isoformat(),
            "target_time": target_time.isoformat(),
        }
        for scheme in schemes:
            window[scheme] = run_scheme(
                coords,
                initial_state,
                target_state,
                cloud_target,
                scheme,
                lead_days,
                initial_time,
            )
        windows.append(window)
    result = {
        "lead_hours": args.lead_hours,
        "state_store": str(args.store),
        "cloud_store": str(args.cloud_store),
        "schemes": schemes,
        "mean_metrics": {scheme: mean_metrics(windows, scheme) for scheme in schemes},
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
