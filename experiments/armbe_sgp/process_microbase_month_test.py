"""Tests for month-level MICROBASE output validation."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from process_microbase_month import verify_day_outputs


def _write_outputs(root: Path) -> tuple[Path, Path]:
    day = np.datetime64("2018-06-01")
    time = day + np.arange(24) * np.timedelta64(1, "h") + np.timedelta64(30, "m")
    height = np.arange(596)
    base = xr.Dataset(
        {
            "armbe_cloud_fraction": (("time", "height"), np.zeros((24, 596))),
            "liquid_water_concentration": (
                ("time", "height"), np.zeros((24, 596))
            ),
            "ice_water_concentration": (
                ("time", "height"), np.zeros((24, 596))
            ),
            "condensate_pair_count": (
                ("time", "height"), np.ones((24, 596), dtype=np.int32)
            ),
        },
        coords={"time": time, "height": height},
    )
    paired = base.isel(time=[5, 11, 17, 23]).copy()
    paired["temperature"] = (
        ("time", "height"), np.full((4, 596), 280.0)
    )
    paired["relative_humidity"] = (
        ("time", "height"), np.full((4, 596), 0.5)
    )
    paired["dewpoint"] = (("time", "height"), np.full((4, 596), 270.0))
    pressure = np.broadcast_to(
        np.linspace(100000.0, 10000.0, 596), (4, 596)
    ).copy()
    paired["pressure"] = (("time", "height"), pressure)
    paired["air_density"] = (("time", "height"), pressure / (287.04 * 280.0))
    paired["specific_humidity"] = (
        ("time", "height"), np.full((4, 596), 0.005)
    )
    paired["qc"] = (("time", "height"), np.zeros((4, 596)))
    paired["qi"] = (("time", "height"), np.zeros((4, 596)))
    paired["atmospheric_valid"] = (
        ("time", "height"), np.ones((4, 596), dtype=bool)
    )
    paired["model_sample_valid"] = (
        ("time", "height"), np.ones((4, 596), dtype=bool)
    )
    hourly_path = root / "hourly.nc"
    paired_path = root / "paired.nc"
    base.to_netcdf(hourly_path)
    paired.to_netcdf(paired_path)
    return hourly_path, paired_path


def test_verify_day_outputs_accepts_complete_pair(tmp_path: Path):
    hourly_path, paired_path = _write_outputs(tmp_path)
    verify_day_outputs(hourly_path, paired_path, "2018-06-01")


def test_verify_day_outputs_rejects_incomplete_atmosphere(tmp_path: Path):
    hourly_path, paired_path = _write_outputs(tmp_path)
    with xr.open_dataset(paired_path) as paired:
        incomplete = paired.load()
    incomplete["temperature"][0, 0] = np.nan
    incomplete.to_netcdf(paired_path)
    with pytest.raises(ValueError, match="invalid"):
        verify_day_outputs(hourly_path, paired_path, "2018-06-01")


def test_verify_day_outputs_accepts_day_without_model_samples(tmp_path: Path):
    hourly_path, paired_path = _write_outputs(tmp_path)
    with xr.open_dataset(paired_path) as paired:
        empty = paired.load()
    empty["model_sample_valid"] = xr.zeros_like(
        empty.model_sample_valid, dtype=bool
    )
    empty[["qc", "qi", "armbe_cloud_fraction"]] = empty[
        ["qc", "qi", "armbe_cloud_fraction"]
    ].where(empty.model_sample_valid)
    empty.to_netcdf(paired_path)
    verify_day_outputs(hourly_path, paired_path, "2018-06-01")


def test_verify_day_outputs_accepts_day_without_retrievals(tmp_path: Path):
    hourly_path, paired_path = _write_outputs(tmp_path)
    with xr.open_dataset(hourly_path) as hourly:
        empty_hourly = hourly.load()
    empty_hourly[["liquid_water_concentration", "ice_water_concentration"]] = (
        empty_hourly[["liquid_water_concentration", "ice_water_concentration"]]
        .where(False)
    )
    empty_hourly["condensate_pair_count"] = xr.zeros_like(
        empty_hourly.condensate_pair_count
    )
    empty_hourly.to_netcdf(hourly_path)

    with xr.open_dataset(paired_path) as paired:
        empty_paired = paired.load()
    empty_paired["model_sample_valid"] = xr.zeros_like(
        empty_paired.model_sample_valid, dtype=bool
    )
    empty_paired[["qc", "qi", "armbe_cloud_fraction"]] = empty_paired[
        ["qc", "qi", "armbe_cloud_fraction"]
    ].where(empty_paired.model_sample_valid)
    empty_paired.to_netcdf(paired_path)

    verify_day_outputs(hourly_path, paired_path, "2018-06-01")


def test_verify_day_outputs_rejects_condensate_pair_mismatch(tmp_path: Path):
    hourly_path, paired_path = _write_outputs(tmp_path)
    with xr.open_dataset(hourly_path) as hourly:
        inconsistent = hourly.load()
    inconsistent[["liquid_water_concentration", "ice_water_concentration"]] = (
        inconsistent[["liquid_water_concentration", "ice_water_concentration"]]
        .where(False)
    )
    inconsistent.to_netcdf(hourly_path)

    with pytest.raises(ValueError, match="condensate"):
        verify_day_outputs(hourly_path, paired_path, "2018-06-01")
