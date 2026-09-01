"""Build and audit one day of native-height MICROBASE/ARMBE data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

import microbase_physics as constants
from microbase_physics import saturation_vapor_pressure_hpa


HERE = Path(__file__).resolve().parent
DEFAULT_MICROBASE = (
    HERE / "data/microbase_probe/sgpmicrobaseC1.c1.20180601.000000.nc"
)
ORDER_ROOT = HERE / "data/order-267892/ftp.archive.arm.gov/fisherm1/267892"
DEFAULT_ATM = (
    ORDER_ROOT
    / "sgparmbeatmC1.c1/sgparmbeatmC1.c1.20180101.003000.nc"
)
DEFAULT_CLDRAD = (
    ORDER_ROOT
    / "sgparmbecldradC1.c1/sgparmbecldradC1.c1.20180101.003000.nc"
)
DEFAULT_OUTPUT = HERE / "outputs/echam_layer_cloud_pilot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_audit(values: np.ndarray) -> dict[str, object]:
    times = values.astype("datetime64[ns]")
    deltas = np.diff(times).astype("timedelta64[ns]").astype(np.int64) / 1e9
    return {
        "count": int(times.size),
        "first": str(times[0]),
        "last": str(times[-1]),
        "monotonic": bool(np.all(deltas > 0)),
        "duplicate_count": int(np.count_nonzero(deltas == 0)),
        "cadence_seconds": sorted(np.unique(deltas).tolist()),
    }


def _masked_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    count = valid.sum(axis=0)
    total = np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
    return np.divide(
        total,
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    error = prediction[valid] - target[valid]
    if not error.size:
        return {"count": 0, "rmse": float("nan"), "bias": float("nan")}
    return {
        "count": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
    }


def reconstruct_hydrostatic_state(
    height: np.ndarray,
    temperature: np.ndarray,
    vapor_pressure: np.ndarray,
    surface_pressure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct pressure, density, and specific humidity on a height grid."""
    height = np.asarray(height, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    vapor_pressure = np.asarray(vapor_pressure, dtype=float)
    if not (
        height.ndim == temperature.ndim == vapor_pressure.ndim == 1
        and height.size == temperature.size == vapor_pressure.size
    ):
        raise ValueError("height, temperature, and vapor pressure must be aligned 1-D arrays")
    if not np.all(np.diff(height) > 0):
        raise ValueError("height must be strictly increasing")
    if not (
        np.all(np.isfinite(temperature))
        and np.all(np.isfinite(vapor_pressure))
        and np.isfinite(surface_pressure)
    ):
        raise ValueError("hydrostatic reconstruction requires complete finite profiles")

    vapor_pressure = np.maximum(vapor_pressure, 0.0)
    pressure = np.empty_like(height)
    previous_pressure = float(surface_pressure)
    previous_height = 0.0
    previous_temperature = temperature[0]
    previous_vapor_pressure = vapor_pressure[0]
    for index, current_height in enumerate(height):
        layer_temperature = 0.5 * (previous_temperature + temperature[index])
        dz = current_height - previous_height
        dry_mid_pressure = previous_pressure * np.exp(
            -0.5 * constants.GRAV * dz / (constants.RD * layer_temperature)
        )
        layer_vapor_pressure = np.clip(
            0.5 * (previous_vapor_pressure + vapor_pressure[index]),
            0.0,
            0.99 * dry_mid_pressure,
        )
        virtual_temperature = layer_temperature / (
            1.0
            - (1.0 - constants.EPS) * layer_vapor_pressure / dry_mid_pressure
        )
        pressure[index] = previous_pressure * np.exp(
            -constants.GRAV * dz / (constants.RD * virtual_temperature)
        )
        previous_pressure = pressure[index]
        previous_height = current_height
        previous_temperature = temperature[index]
        previous_vapor_pressure = vapor_pressure[index]

    vapor_pressure = np.minimum(vapor_pressure, 0.99 * pressure)
    specific_humidity = (
        constants.EPS
        * vapor_pressure
        / (pressure - (1.0 - constants.EPS) * vapor_pressure)
    )
    density = (pressure - vapor_pressure) / (constants.RD * temperature)
    density += vapor_pressure / (constants.RV * temperature)
    return pressure, density, specific_humidity


def _canonicalize_atmosphere(atmosphere: xr.Dataset) -> xr.Dataset:
    """Normalize released ARMBEATM schema generations used by the pilot."""
    if "temperature_h" in atmosphere:
        return atmosphere
    legacy = {
        "z": "height",
        "T_z": "temperature_h",
        "Td_z": "dewpoint_h",
        "rh_z": "relative_humidity_h",
        "p_sfc": "pressure_sfc",
    }
    missing = [name for name in legacy if name not in atmosphere]
    if missing:
        raise ValueError(f"Unsupported ARMBEATM schema; missing {missing}")
    return atmosphere.rename(legacy)


def build_hourly_cache(
    microbase: xr.Dataset,
    atmosphere: xr.Dataset,
    cldrad: xr.Dataset,
    day: str,
) -> tuple[xr.Dataset, xr.Dataset, dict[str, object]]:
    """Build hourly cloud data and pair it with observed ARMBEATM profiles."""
    atmosphere = _canonicalize_atmosphere(atmosphere)
    start = np.datetime64(day)
    end = start + np.timedelta64(1, "D")
    cloud_day = cldrad.where(
        (cldrad.time >= start) & (cldrad.time < end), drop=True
    )
    atm_day = atmosphere.where(
        (atmosphere.time >= start) & (atmosphere.time < end), drop=True
    )
    micro_day = microbase.where(
        (microbase.time >= start) & (microbase.time < end), drop=True
    )
    if cloud_day.sizes.get("time") != 24 or atm_day.sizes.get("time") != 24:
        raise ValueError("Expected 24 hourly ARMBE samples on the pilot day")
    if not np.array_equal(micro_day.height.values, cloud_day.height.values):
        raise ValueError("MICROBASE and ARMBE cloud height grids do not align exactly")
    if micro_day.liquid_water_content.attrs.get("units") != "g m-3":
        raise ValueError("Unexpected MICROBASE liquid-water units")
    if micro_day.ice_water_content.attrs.get("units") != "g m-3":
        raise ValueError("Unexpected MICROBASE ice-water units")

    times = cloud_day.time.values
    height = cloud_day.height.values
    shape = (times.size, height.size)
    fields = {
        "cloud_fraction_strict": np.full(shape, np.nan),
        "cloud_fraction_inclusive": np.full(shape, np.nan),
        "strict_occurrence_count": np.zeros(shape, dtype=np.int32),
        "inclusive_occurrence_count": np.zeros(shape, dtype=np.int32),
        "liquid_water_concentration": np.full(shape, np.nan),
        "ice_water_concentration": np.full(shape, np.nan),
        "condensate_pair_count": np.zeros(shape, dtype=np.int32),
    }

    retrieval_counts: dict[str, int] = {}
    precip_missing = 0
    precip_total = 0
    half_window = np.timedelta64(30, "m")
    for index, center in enumerate(times):
        window = micro_day.where(
            (micro_day.time >= center - half_window)
            & (micro_day.time < center + half_window),
            drop=True,
        )
        retrieval = window.retrieval_flag.values
        liquid = window.liquid_water_content.values
        ice = window.ice_water_content.values
        liquid_qc = window.qc_liquid_water_content.values
        ice_qc = window.qc_ice_water_content.values

        finite_retrieval = np.isfinite(retrieval)
        for value, count in zip(
            *np.unique(retrieval[finite_retrieval].astype(int), return_counts=True)
        ):
            key = str(value)
            retrieval_counts[key] = retrieval_counts.get(key, 0) + int(count)

        strict_valid = finite_retrieval & np.isin(retrieval, (0, 1))
        inclusive_valid = finite_retrieval & np.isin(retrieval, (0, 1, 2))
        fields["cloud_fraction_strict"][index] = _masked_mean(
            retrieval == 1, strict_valid
        )
        fields["cloud_fraction_inclusive"][index] = _masked_mean(
            np.isin(retrieval, (1, 2)), inclusive_valid
        )
        fields["strict_occurrence_count"][index] = strict_valid.sum(axis=0)
        fields["inclusive_occurrence_count"][index] = inclusive_valid.sum(axis=0)

        clear = retrieval == 0
        good_cloud = (
            (retrieval == 1)
            & (liquid_qc == 0)
            & (ice_qc == 0)
            & np.isfinite(liquid)
            & np.isfinite(ice)
        )
        pair_valid = clear | good_cloud
        fields["liquid_water_concentration"][index] = _masked_mean(
            np.where(clear, 0.0, liquid), pair_valid
        )
        fields["ice_water_concentration"][index] = _masked_mean(
            np.where(clear, 0.0, ice), pair_valid
        )
        fields["condensate_pair_count"][index] = pair_valid.sum(axis=0)

        precip = window.precip_flag.values
        precip_missing += int(np.count_nonzero(~np.isfinite(precip)))
        precip_total += int(precip.size)

    hourly = xr.Dataset(
        data_vars={
            **{
                name: (("time", "height"), value)
                for name, value in fields.items()
            },
            "armbe_cloud_fraction": (
                ("time", "height"), cloud_day.cld_frac.values / 100.0
            ),
            "armbe_cloud_fraction_qc": (
                ("time", "height"), cloud_day.qc_cld_frac.values
            ),
        },
        coords={"time": times, "height": height},
        attrs={
            "site": "sgpC1",
            "window": "60 minutes centered on ARMBE hourly timestamps",
            "status": "hourly native-height cloud preparation; not ECHAM qc/qi",
            "condensate_semantics": (
                "instantaneous ARSCL-bin retrieval; density conversion still blocked"
            ),
        },
    )
    hourly.cloud_fraction_strict.attrs.update(units="1")
    hourly.cloud_fraction_inclusive.attrs.update(units="1")
    hourly.armbe_cloud_fraction.attrs.update(units="1")
    hourly.liquid_water_concentration.attrs.update(units="g m-3")
    hourly.ice_water_concentration.attrs.update(units="g m-3")

    # ARMBEATM keeps hourly records but populates complete height profiles only
    # every six hours. Select those observed profiles without time interpolation.
    profile_candidate = (
        atm_day.temperature_h.notnull().any("height")
        & atm_day.relative_humidity_h.notnull().any("height")
        & atm_day.dewpoint_h.notnull().any("height")
    )
    profile_valid = profile_candidate & atm_day.pressure_sfc.notnull()
    atmosphere_observed = atm_day.where(profile_valid, drop=True)
    observed_times = atmosphere_observed.time.values
    temperature = atmosphere_observed.temperature_h.interp(height=height)
    relative_humidity = atmosphere_observed.relative_humidity_h.interp(height=height)
    dewpoint = atmosphere_observed.dewpoint_h.interp(height=height)
    paired = hourly.sel(time=observed_times).copy()
    paired["temperature"] = (("time", "height"), temperature.values)
    paired["relative_humidity"] = (
        ("time", "height"), relative_humidity.values / 100.0
    )
    paired["dewpoint"] = (("time", "height"), dewpoint.values)
    paired["surface_pressure"] = (
        "time", atmosphere_observed.pressure_sfc.values * 100.0
    )
    source_height = atmosphere_observed.height.values
    pressure = np.full_like(temperature.values, np.nan, dtype=float)
    density = np.full_like(temperature.values, np.nan, dtype=float)
    specific_humidity = np.full_like(temperature.values, np.nan, dtype=float)
    for index in range(observed_times.size):
        source_valid = (
            np.isfinite(atmosphere_observed.temperature_h.values[index])
            & np.isfinite(atmosphere_observed.relative_humidity_h.values[index])
            & np.isfinite(atmosphere_observed.dewpoint_h.values[index])
        )
        profile_height = source_height[source_valid]
        source_vapor_pressure = np.asarray(
            100.0
            * saturation_vapor_pressure_hpa(
                atmosphere_observed.dewpoint_h.values[index, source_valid]
            )
        )
        source_pressure, _, _ = reconstruct_hydrostatic_state(
            profile_height,
            atmosphere_observed.temperature_h.values[index, source_valid],
            source_vapor_pressure,
            atmosphere_observed.pressure_sfc.values[index] * 100.0,
        )
        target_valid = (
            np.isfinite(temperature.values[index])
            & np.isfinite(relative_humidity.values[index])
            & np.isfinite(dewpoint.values[index])
            & (height >= profile_height.min())
            & (height <= profile_height.max())
        )
        pressure[index, target_valid] = np.exp(
            np.interp(
                height[target_valid], profile_height, np.log(source_pressure)
            )
        )
        target_vapor_pressure = np.asarray(
            100.0
            * saturation_vapor_pressure_hpa(dewpoint.values[index, target_valid])
        )
        vapor_pressure = np.minimum(
            target_vapor_pressure,
            0.99 * pressure[index, target_valid],
        )
        specific_humidity[index, target_valid] = (
            constants.EPS
            * vapor_pressure
            / (
                pressure[index, target_valid]
                - (1.0 - constants.EPS) * vapor_pressure
            )
        )
        density[index, target_valid] = (
            pressure[index, target_valid] - vapor_pressure
        ) / (constants.RD * temperature.values[index, target_valid])
        density[index, target_valid] += vapor_pressure / (
            constants.RV * temperature.values[index, target_valid]
        )
    paired["pressure"] = (("time", "height"), pressure)
    paired["air_density"] = (("time", "height"), density)
    paired["specific_humidity"] = (("time", "height"), specific_humidity)
    paired["atmospheric_valid"] = (
        ("time", "height"), np.isfinite(density)
    )
    paired["qc"] = (
        ("time", "height"),
        1.0e-3 * paired.liquid_water_concentration.values / density,
    )
    paired["qi"] = (
        ("time", "height"),
        1.0e-3 * paired.ice_water_concentration.values / density,
    )
    paired["model_sample_valid"] = (
        ("time", "height"),
        np.isfinite(density)
        & (paired.condensate_pair_count.values > 0)
        & np.isfinite(paired.liquid_water_concentration.values)
        & np.isfinite(paired.ice_water_concentration.values)
        & (paired.armbe_cloud_fraction_qc.values == 0)
        & np.isfinite(paired.armbe_cloud_fraction.values),
    )
    vertical_interpolation = "linear from labelled 45 m ARMBEATM height grid"
    paired.temperature.attrs.update(units="K", interpolation=vertical_interpolation)
    paired.relative_humidity.attrs.update(
        units="1", interpolation=vertical_interpolation
    )
    paired.dewpoint.attrs.update(units="K", interpolation=vertical_interpolation)
    paired.surface_pressure.attrs.update(units="Pa")
    paired.pressure.attrs.update(
        units="Pa",
        method=(
            "moist hydrostatic integration from observed surface pressure; "
            "vapor pressure from sounding dewpoint over liquid water"
        ),
    )
    paired.air_density.attrs.update(units="kg m-3", method="moist ideal gas law")
    paired.specific_humidity.attrs.update(units="kg kg-1")
    paired.atmospheric_valid.attrs.update(
        description="True where the observed sounding spans the cloud-grid height"
    )
    paired.model_sample_valid.attrs.update(
        description=(
            "True where atmosphere, condensate pair, and primary ARMBE target "
            "are all valid"
        )
    )
    paired.qc.attrs.update(
        units="kg kg-1",
        method="hourly mean concentration divided by ARMBEATM snapshot density",
    )
    paired.qi.attrs.update(
        units="kg kg-1",
        method="hourly mean concentration divided by ARMBEATM snapshot density",
    )
    paired.attrs.update(
        status="observed ARMBEATM profile times; no atmospheric time interpolation",
        atmospheric_profile_count=int(observed_times.size),
        pressure_profile_status="hydrostatically reconstructed from ARMBEATM",
    )

    primary_target = np.where(
        hourly.armbe_cloud_fraction_qc.values == 0,
        hourly.armbe_cloud_fraction.values,
        np.nan,
    )
    audit = {
        "day": day,
        "microbase_time": _time_audit(micro_day.time.values),
        "armbe_cloud_time": _time_audit(cloud_day.time.values),
        "height": {
            "cells": int(height.size),
            "minimum_m": float(height.min()),
            "maximum_m": float(height.max()),
            "spacing_m": sorted(np.unique(np.diff(height)).astype(float).tolist()),
            "microbase_armbe_cloud_exact_match": True,
            "armbeatm_exact_match": bool(
                np.array_equal(atm_day.height.values, height)
            ),
            "armbeatm_vertical_interpolation": vertical_interpolation,
            "armbeatm_time_interpolation": "none",
            "armbeatm_valid_profile_times_on_day": [
                str(value) for value in observed_times
            ],
            "armbeatm_profile_count": int(observed_times.size),
            "armbeatm_valid_cells": int(np.count_nonzero(np.isfinite(density))),
            "armbeatm_excluded_missing_surface_pressure": [
                str(value)
                for value in atm_day.time.values[
                    (profile_candidate & atm_day.pressure_sfc.isnull()).values
                ]
            ],
        },
        "retrieval_flag_counts": retrieval_counts,
        "precip_flag": {
            "missing_count": precip_missing,
            "total_count": precip_total,
            "status": "unknown" if precip_missing == precip_total else "partially observed",
        },
        "armbe_qc_counts": {
            str(value): int(count)
            for value, count in zip(
                *np.unique(hourly.armbe_cloud_fraction_qc.values, return_counts=True)
            )
        },
        "comparison_to_primary_armbe_target": {
            "strict": _metrics(
                hourly.cloud_fraction_strict.values, primary_target
            ),
            "inclusive": _metrics(
                hourly.cloud_fraction_inclusive.values, primary_target
            ),
        },
        "conversion_gates": {
            "condensate_convention_resolved": True,
            "pressure_profile_available": True,
            "precipitation_status_resolved": False,
        },
    }
    return hourly, paired, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbase", type=Path, default=DEFAULT_MICROBASE)
    parser.add_argument("--atmosphere", type=Path, default=DEFAULT_ATM)
    parser.add_argument("--cldrad", type=Path, default=DEFAULT_CLDRAD)
    parser.add_argument("--day", default="2018-06-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    with (
        xr.open_dataset(args.microbase) as microbase,
        xr.open_dataset(args.atmosphere) as atmosphere,
        xr.open_dataset(args.cldrad) as cldrad,
    ):
        hourly, paired, audit = build_hourly_cache(
            microbase, atmosphere, cldrad, args.day
        )
        audit["sources"] = {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in {
                "microbase": args.microbase,
                "armbeatm": args.atmosphere,
                "armbecldrad": args.cldrad,
            }.items()
        }
        hourly.to_netcdf(args.output / "june1_microbase_hourly.nc")
        paired.to_netcdf(args.output / "june1_observed_atmosphere_paired.nc")
    manifest = args.output / "microbase_sgp_pilot_manifest.json"
    manifest.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output / 'june1_microbase_hourly.nc'}")
    print(f"wrote {args.output / 'june1_observed_atmosphere_paired.nc'}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
