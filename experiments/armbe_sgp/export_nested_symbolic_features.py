"""Export nested atmospheric-state feature groups for pooled ARMBE SR.

The existing five SPEEDY diagnostic features are preserved exactly from an
archived feature table. Additional features are deterministic summaries of the
same eight-level state stored in the unified ARMBE cache. All vertical
definitions use sigma values rather than level indices so their physical
meaning does not depend on array ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import jcm.constants as c
from jcm.physics.speedy.speedy_coords import compute_speedy_vertical_coords


BASE_FEATURES = (
    "rh_cloudc_max",
    "precip_mm_day",
    "gse",
    "rh_lowest",
    "fmask",
)
HUMIDITY_STABILITY_FEATURES = (
    "rh_cloudc_mean",
    "rh_low_mean",
    "rh_mid_mean",
    "rh_high_mean",
    "rh_low_mid_gradient",
    "rh_vertical_range",
    "low_level_stability",
    "maximum_inversion_strength",
    "low_level_lapse_rate",
)
MOISTURE_WIND_FEATURES = (
    "column_saturation_deficit",
    "precipitable_water",
    "low_level_wind_speed",
    "boundary_layer_wind_shear",
)
FEATURE_GROUPS = {
    "group_05_baseline": BASE_FEATURES,
    "group_14_humidity_stability": BASE_FEATURES + HUMIDITY_STABILITY_FEATURES,
    "group_18_moisture_wind": (
        BASE_FEATURES + HUMIDITY_STABILITY_FEATURES + MOISTURE_WIND_FEATURES
    ),
}

FEATURE_METADATA = {
    "rh_cloudc_mean": {
        "units": "1",
        "definition": "mean RH at sigma 0.34, 0.51, 0.685, and 0.835",
    },
    "rh_low_mean": {
        "units": "1",
        "definition": "mean RH at sigma 0.685, 0.835, and 0.95",
    },
    "rh_mid_mean": {
        "units": "1",
        "definition": "mean RH at sigma 0.34 and 0.51",
    },
    "rh_high_mean": {
        "units": "1",
        "definition": "mean RH at sigma 0.20 and 0.34",
    },
    "rh_low_mid_gradient": {
        "units": "1",
        "definition": "rh_low_mean minus rh_mid_mean",
    },
    "rh_vertical_range": {
        "units": "1",
        "definition": "RH maximum minus minimum over sigma 0.20 through 0.95",
    },
    "low_level_stability": {
        "units": "K",
        "definition": "potential temperature at sigma 0.685 minus surface potential temperature",
    },
    "maximum_inversion_strength": {
        "units": "K",
        "definition": "largest positive adjacent potential-temperature increase upward over sigma 0.51 through 0.95",
    },
    "low_level_lapse_rate": {
        "units": "K km-1",
        "definition": "temperature decrease upward from sigma 0.95 to 0.835 divided by geopotential-height difference",
    },
    "column_saturation_deficit": {
        "units": "1",
        "definition": "pressure-weighted mean positive RH deficit over tropospheric sigma levels",
    },
    "precipitable_water": {
        "units": "kg m-2",
        "definition": "pressure-integrated specific humidity",
    },
    "low_level_wind_speed": {
        "units": "m s-1",
        "definition": "mean wind speed at sigma 0.685, 0.835, and 0.95",
    },
    "boundary_layer_wind_shear": {
        "units": "m s-1",
        "definition": "vector wind difference between sigma 0.685 and 0.95",
    },
}

_CLOUD_SIGMAS = np.asarray((0.34, 0.51, 0.685, 0.835))
_LOW_SIGMAS = np.asarray((0.685, 0.835, 0.95))
_MID_SIGMAS = np.asarray((0.34, 0.51))
_HIGH_SIGMAS = np.asarray((0.20, 0.34))
_COLUMN_RH_SIGMAS = np.asarray((0.20, 0.34, 0.51, 0.685, 0.835, 0.95))
_INVERSION_SIGMAS = np.asarray((0.51, 0.685, 0.835, 0.95))


def _interp_profiles(
    values: np.ndarray, sigma: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Interpolate sample-by-level profiles to fixed sigma targets."""
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if values.ndim != 2 or values.shape[1] != sigma.size:
        raise ValueError("profiles must have shape (sample, sigma.size)")
    order = np.argsort(sigma)
    sigma = sigma[order]
    values = values[:, order]
    if np.any(np.diff(sigma) <= 0):
        raise ValueError("sigma coordinates must be unique")
    upper = np.searchsorted(sigma, targets)
    upper = np.clip(upper, 1, sigma.size - 1)
    lower = upper - 1
    weight = (targets - sigma[lower]) / (sigma[upper] - sigma[lower])
    return values[:, lower] * (1.0 - weight) + values[:, upper] * weight


def _saturation_specific_humidity(
    temperature: np.ndarray, normalized_surface_pressure: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Reproduce SPEEDY saturation specific humidity in g kg-1."""
    e0 = 6.108e-3
    warm = e0 * np.exp(17.269 * (temperature - 273.16) / (temperature - 35.86))
    cold = e0 * np.exp(21.875 * (temperature - 273.16) / (temperature - 7.66))
    vapor_pressure = np.where(temperature >= 273.16, warm, cold)
    pressure = normalized_surface_pressure[:, None] * sigma[None, :]
    return 622.0 * vapor_pressure / (pressure - 0.378 * vapor_pressure)


def derive_profile_features(samples: xr.Dataset) -> dict[str, np.ndarray]:
    """Derive the 13 added profile summaries from a labelled sample dataset."""
    nlev = int(samples.sizes["level"])
    _hsg, fsg, dhs, *_ = compute_speedy_vertical_coords(nlev)
    sigma = np.asarray(fsg, dtype=float)
    layer_thickness = np.asarray(dhs, dtype=float)
    order = np.argsort(sigma)
    sigma = sigma[order]
    layer_thickness = layer_thickness[order]

    temperature = np.asarray(samples["temperature"].values, dtype=float)[:, order]
    humidity = np.asarray(samples["specific_humidity"].values, dtype=float)[:, order]
    u_wind = np.asarray(samples["u_wind"].values, dtype=float)[:, order]
    v_wind = np.asarray(samples["v_wind"].values, dtype=float)[:, order]
    geopotential = np.asarray(samples["geopotential"].values, dtype=float)[:, order]
    normalized_ps = np.asarray(samples["normalized_surface_pressure"].values, dtype=float)
    surface_temperature = np.asarray(samples["surface_temperature"].values, dtype=float)

    qsat = _saturation_specific_humidity(temperature, normalized_ps, sigma)
    # A few extrapolated cold/low-pressure cache levels have near-zero or
    # negative SPEEDY qsat and diagnostic RH of O(10^2-10^4). They are finite
    # but not physically meaningful profile structure. Preserve the archived
    # baseline diagnostics exactly, while bounding only the added RH summaries
    # to the generous physical interval 0..1.2.
    rh = np.clip(humidity / qsat, 0.0, 1.2)
    rh_cloud = _interp_profiles(rh, sigma, _CLOUD_SIGMAS)
    rh_low = _interp_profiles(rh, sigma, _LOW_SIGMAS)
    rh_mid = _interp_profiles(rh, sigma, _MID_SIGMAS)
    rh_high = _interp_profiles(rh, sigma, _HIGH_SIGMAS)
    rh_column = _interp_profiles(rh, sigma, _COLUMN_RH_SIGMAS)

    pressure = normalized_ps[:, None] * sigma[None, :]
    potential_temperature = temperature * pressure ** (-float(c.akap))
    theta_inversion = _interp_profiles(
        potential_temperature, sigma, _INVERSION_SIGMAS
    )
    theta_685 = _interp_profiles(
        potential_temperature, sigma, np.asarray((0.685,))
    )[:, 0]
    surface_theta = surface_temperature * normalized_ps ** (-float(c.akap))

    temperature_low = _interp_profiles(
        temperature, sigma, np.asarray((0.835, 0.95))
    )
    height_low = _interp_profiles(
        geopotential / float(c.grav), sigma, np.asarray((0.835, 0.95))
    )
    height_difference_km = (height_low[:, 0] - height_low[:, 1]) / 1000.0

    u_low = _interp_profiles(u_wind, sigma, _LOW_SIGMAS)
    v_low = _interp_profiles(v_wind, sigma, _LOW_SIGMAS)
    u_shear = _interp_profiles(u_wind, sigma, np.asarray((0.685, 0.95)))
    v_shear = _interp_profiles(v_wind, sigma, np.asarray((0.685, 0.95)))

    column_mass = normalized_ps * float(c.p0) / float(c.grav)
    precipitable_water = column_mass * np.sum(
        humidity / 1000.0 * layer_thickness[None, :], axis=1
    )
    troposphere = sigma >= (0.2 - 1e-5)
    column_saturation_deficit = np.sum(
        np.maximum(1.0 - rh[:, troposphere], 0.0)
        * layer_thickness[None, troposphere],
        axis=1,
    ) / np.sum(layer_thickness[troposphere])

    return {
        "rh_cloudc_mean": np.mean(rh_cloud, axis=1),
        "rh_low_mean": np.mean(rh_low, axis=1),
        "rh_mid_mean": np.mean(rh_mid, axis=1),
        "rh_high_mean": np.mean(rh_high, axis=1),
        "rh_low_mid_gradient": np.mean(rh_low, axis=1) - np.mean(rh_mid, axis=1),
        "rh_vertical_range": np.ptp(rh_column, axis=1),
        "low_level_stability": theta_685 - surface_theta,
        "maximum_inversion_strength": np.maximum(
            np.max(theta_inversion[:, :-1] - theta_inversion[:, 1:], axis=1), 0.0
        ),
        "low_level_lapse_rate": (
            (temperature_low[:, 1] - temperature_low[:, 0]) / height_difference_km
        ),
        "column_saturation_deficit": column_saturation_deficit,
        "precipitable_water": precipitable_water,
        "low_level_wind_speed": np.mean(np.hypot(u_low, v_low), axis=1),
        "boundary_layer_wind_shear": np.hypot(
            u_shear[:, 0] - u_shear[:, 1], v_shear[:, 0] - v_shear[:, 1]
        ),
    }


def export_nested_features(
    cache: str | Path, baseline_features: str | Path, out_dir: str | Path
) -> Path:
    """Write one shared NetCDF table and CSV files for all nested groups."""
    cache = Path(cache)
    baseline_features = Path(baseline_features)
    out_dir = Path(out_dir)
    with xr.open_dataset(cache) as raw:
        samples = raw.load()
    with xr.open_dataset(baseline_features) as raw:
        baseline = raw.load()

    if not np.array_equal(samples["sample"].values, baseline["sample"].values):
        raise ValueError("cache and baseline feature sample coordinates differ")
    if not np.array_equal(samples["time"].values, baseline["time"].values):
        raise ValueError("cache and baseline feature time coordinates differ")

    derived = derive_profile_features(samples)
    for name, values in derived.items():
        if not np.all(np.isfinite(values)):
            bad = int(np.sum(~np.isfinite(values)))
            raise ValueError(f"derived feature {name!r} has {bad} non-finite values")

    data_vars = {
        "target": ("sample", np.asarray(baseline["target"].values)),
        "split": ("sample", np.asarray(baseline["split"].values)),
        "site_facility": ("sample", np.asarray(baseline["site_facility"].values)),
        **{
            name: ("sample", np.asarray(baseline[name].values))
            for name in BASE_FEATURES
        },
        **{name: ("sample", values) for name, values in derived.items()},
    }
    table = xr.Dataset(
        data_vars,
        coords={
            "sample": np.asarray(samples["sample"].values),
            "time": ("sample", np.asarray(samples["time"].values)),
        },
        attrs={
            "target": "ARMBECLDRAD tot_cld with qc_tot_cld == 0",
            "feature_design": "nested model-state features defined on physical sigma coordinates",
            "derived_rh_processing": "specific humidity divided by SPEEDY qsat, clipped to [0, 1.2]",
        },
    )
    for name, metadata in FEATURE_METADATA.items():
        table[name].attrs.update(metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_netcdf(out_dir / "features.nc")
    labels = np.asarray(table["split"].values).astype(str)
    split_counts = {}
    for group_name, feature_names in FEATURE_GROUPS.items():
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            mask = labels == split
            frame = pd.DataFrame({"target": table["target"].values[mask]})
            for name in feature_names:
                frame[name] = table[name].values[mask]
            frame.to_csv(group_dir / f"{split}.csv", index=False)
            split_counts[split] = int(np.sum(mask))
        group_manifest = {
            "group": group_name,
            "feature_columns": list(feature_names),
            "feature_count": len(feature_names),
            "split_counts": split_counts,
            "shared_features": str((out_dir / "features.nc").resolve()),
        }
        (group_dir / "manifest.json").write_text(
            json.dumps(group_manifest, indent=2, sort_keys=True) + "\n"
        )

    manifest = {
        "format": "armbe-nested-symbolic-features-v1",
        "cache": str(cache.resolve()),
        "baseline_features": str(baseline_features.resolve()),
        "samples": int(samples.sizes["sample"]),
        "groups": {name: list(features) for name, features in FEATURE_GROUPS.items()},
        "feature_metadata": FEATURE_METADATA,
        "split_counts": split_counts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out_dir = export_nested_features(args.cache, args.baseline_features, args.out_dir)
    print(f"wrote nested symbolic feature tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
