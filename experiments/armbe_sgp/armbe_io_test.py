"""Regression tests for the ARMBE-to-SPEEDY adapter."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from armbe_io import (
    InvalidArmbeData,
    geopotential_on_sigma,
    load_armbe,
    to_obs_targets,
    to_state_series,
    validate_armbe_input,
)
from forecast_cache import build_cache, load_config, resolved_config
from diagnostic_cache import assign_year_splits, build_diagnostic_cache, discover_annual_files
from jcm.date import DateData
from evaluate import main as evaluate_main
from make_synthetic_armbe import build as build_synthetic_armbe
from run_scm import (
    build_forcing,
    filter_regular_cadence,
    start_date_from_timestamp,
    validate_cadence,
)
from run_scm import main as run_scm_main


def _dataset():
    time = np.array(["2018-06-01T00", "2018-06-01T01", "2018-06-01T02"],
                    dtype="datetime64[h]")
    lev = np.array([1000.0, 700.0, 300.0, 100.0])
    profile = np.array([[300.0, 280.0, 240.0, 220.0]] * len(time))
    return xr.Dataset(
        {
            "temp_p": (("time", "lev"), profile, {"units": "K"}),
            "q_p": (("time", "lev"), np.full_like(profile, 0.005), {"units": "kg/kg"}),
            "u_p": (("time", "lev"), np.zeros_like(profile), {"units": "m/s"}),
            "v_p": (("time", "lev"), np.zeros_like(profile), {"units": "m/s"}),
            "pressure_sfc": (("time",), np.array([975.0, np.nan, 975.0]), {"units": "hPa"}),
            "precip_rate_sfc": (("time",), np.array([1.0, 2.0, 3.0])),
        },
        coords={"time": time, "lev": ("lev", lev, {"units": "hPa"})},
    )


def test_geopotential_uses_speedy_sigma_order():
    sigma = np.array([0.025, 0.2, 0.7, 0.95])
    phi = geopotential_on_sigma(np.array([220.0, 240.0, 280.0, 300.0]), sigma)
    assert np.all(np.diff(phi) < 0)  # top-to-bottom: geopotential decreases
    assert np.isclose(phi[-1] / phi[0], np.log(.95) / np.log(.025))


def test_dropped_profile_keeps_matching_times_and_targets():
    ds = _dataset()
    states, times, meta = to_state_series(ds, nlev=7)
    assert len(states) == 2
    np.testing.assert_allclose(states[0].specific_humidity, 5.0)
    np.testing.assert_array_equal(meta["retained_indices"], [0, 2])
    np.testing.assert_array_equal(times, ds.time.values[[0, 2]])
    np.testing.assert_array_equal(
        to_obs_targets(ds, meta["retained_indices"])["precip"], [1.0, 3.0],
    )


def test_gkg_source_humidity_is_normalized_once():
    ds = _dataset()
    ds["q_p"].values *= 1000.0
    ds["q_p"].attrs["units"] = "g/kg"
    states, _, _ = to_state_series(ds, nlev=7)
    np.testing.assert_allclose(states[0].specific_humidity, 5.0)


def test_observation_targets_include_radiation_components():
    ds = _dataset().assign(
        swdn=("time", [100.0, 110.0, 120.0]),
        swup=("time", [20.0, 21.0, 22.0]),
        lwdn=("time", [300.0, 301.0, 302.0]),
        lwup=("time", [400.0, 401.0, 402.0]),
        sw_dn_TOA=("time", [500.0, 501.0, 502.0]),
        sw_net_TOA=("time", [250.0, 251.0, 252.0]),
        tot_cld=("time", [0.1, 0.2, 0.3]),
    )
    targets = to_obs_targets(ds)
    np.testing.assert_array_equal(targets["sw_net_sfc"], [80.0, 89.0, 98.0])
    np.testing.assert_array_equal(targets["sw_up_sfc"], [20.0, 21.0, 22.0])
    np.testing.assert_array_equal(targets["lw_up_sfc"], [400.0, 401.0, 402.0])
    np.testing.assert_array_equal(targets["sw_down_toa"], [500.0, 501.0, 502.0])
    np.testing.assert_array_equal(targets["sw_net_toa"], [250.0, 251.0, 252.0])
    np.testing.assert_array_equal(targets["cloud_fraction"], [0.1, 0.2, 0.3])


def test_subdaily_start_timestamp_selects_matching_forcing_step():
    times = np.array(["2018-06-01T00", "2018-06-01T06"], dtype="datetime64[h]")
    ds = xr.Dataset(
        {"temp_sfc": (("time",), np.array([290.0, 300.0]), {"units": "K"})},
        coords={"time": times},
    )
    forcing, _ = build_forcing(ds, times)
    date = DateData.set_date(
        model_time=start_date_from_timestamp(times[1]),
        model_step=0,
        dt_seconds=3600.0,
        calendar="gregorian",
    )
    selected = forcing.select(date, calendar="gregorian")
    np.testing.assert_allclose(selected.stl_am, [[300.0]])


def test_irregular_retained_cadence_is_rejected():
    times = np.array(["2018-06-01T00", "2018-06-01T02"], dtype="datetime64[h]")
    with pytest.raises(ValueError, match="regularly spaced"):
        validate_cadence(times, 3600)


def test_regular_cadence_filter_removes_extra_profile_times():
    times = np.array(
        ["2018-06-01T00", "2018-06-01T06", "2018-06-01T07", "2018-06-01T12"],
        dtype="datetime64[h]",
    )
    states, filtered_times, meta = filter_regular_cadence(
        list("abcd"), times, {"retained_indices": np.arange(4), "n_states": 4}, 21600,
    )
    assert states == ["a", "b", "d"]
    np.testing.assert_array_equal(filtered_times, times[[0, 1, 3]])
    np.testing.assert_array_equal(meta["retained_indices"], [0, 1, 3])
    assert meta["n_off_cadence_dropped"] == 1


def test_validation_rejects_duplicate_pressure_levels():
    ds = _dataset().assign_coords(lev=[1000.0, 700.0, 700.0, 100.0])
    with pytest.raises(InvalidArmbeData, match="strictly monotonic"):
        validate_armbe_input(ds)


def test_validation_rejects_non_profile_temperature():
    ds = _dataset().drop_vars("temp_p").assign(temp_p=("time", [290.0, 291.0, 292.0]))
    with pytest.raises(InvalidArmbeData, match="expected both"):
        validate_armbe_input(ds)


def test_loader_ignores_malformed_auxiliary_time_units(tmp_path):
    ds = _dataset()
    ds["time_frac"] = (
        ("time",),
        np.arange(ds.sizes["time"], dtype=float),
        {"units": "days since last day of the previous year"},
    )
    path = tmp_path / "sgparmbeatmC1.c1.20180101.000000.nc"
    ds.to_netcdf(path)

    loaded = load_armbe(path)

    np.testing.assert_array_equal(loaded.time.values, ds.time.values)


def test_directory_loader_accepts_legacy_cdf_files(tmp_path):
    ds = _dataset()
    path = tmp_path / "sgparmbeatmC1.c1.19960101.000000.cdf"
    ds.to_netcdf(path)

    loaded = load_armbe(tmp_path)

    np.testing.assert_array_equal(loaded.time.values, ds.time.values)


def test_annual_discovery_prefers_modern_duplicate_and_year_splits_are_disjoint(tmp_path):
    for suffix in (".cdf", ".nc"):
        (tmp_path / f"sgparmbeatmC1.c1.20110101.000000{suffix}").touch()
    for year in range(1996, 2002):
        (tmp_path / f"sgparmbeatmC1.c1.{year}0101.000000.cdf").touch()

    found = discover_annual_files(tmp_path)
    splits = assign_year_splits(found, validation_years=2, test_years=1, seed=20260731)

    assert found[2011].suffix == ".nc"
    assert set(splits.values()) == {"train", "validation", "test"}
    assert sum(value == "validation" for value in splits.values()) == 2
    assert sum(value == "test" for value in splits.values()) == 1


def test_diagnostic_cache_records_raw_sum_recipe_and_whole_year_splits(tmp_path):
    atm_dir, cldrad_dir = tmp_path / "atm", tmp_path / "cldrad"
    atm_dir.mkdir()
    cldrad_dir.mkdir()
    for year in (2016, 2017, 2018):
        atm, cldrad = build_synthetic_armbe(days=1)
        offset = np.datetime64(f"{year}-01-01") - np.datetime64("2018-01-01")
        atm = atm.assign_coords(time=atm.time.values + offset)
        cldrad = cldrad.assign_coords(time=cldrad.time.values + offset)
        atm.to_netcdf(atm_dir / f"sgparmbeatmC1.c1.{year}0101.000000.cdf")
        cldrad.to_netcdf(cldrad_dir / f"sgparmbecldradC1.c1.{year}0101.000000.cdf")

    cache = build_diagnostic_cache({
        "atm": str(atm_dir),
        "cldrad": str(cldrad_dir),
        "validation_years": 1,
        "test_years": 1,
    }, tmp_path / "diagnostic-cache")

    recipe = json.loads((cache / "recipe.json").read_text())
    samples = xr.open_dataset(cache / "samples.nc")
    splits = {str(value) for value in samples["split"].values}

    assert recipe["target"]["operator"] == "cloudc_plus_cloudstr_raw"
    assert "raw sum is unclipped" in recipe["semantics"]
    assert splits == {"train", "validation", "test"}
    assert all(np.unique(samples["split"].values[samples["year"].values == year]).size == 1
               for year in np.unique(samples["year"].values))


def test_synthetic_pipeline_runs_and_evaluates(tmp_path, capsys):
    """Exercise fixture creation, SCM diagnostics, archive writing, and scoring."""
    atm, cldrad = build_synthetic_armbe(days=2)
    atm_path = tmp_path / "sgparmbeatmC1.c1.synthetic.nc"
    cldrad_path = tmp_path / "sgparmbecldradC1.c1.synthetic.nc"
    output_path = tmp_path / "scm_run.npz"
    atm.to_netcdf(atm_path)
    cldrad.to_netcdf(cldrad_path)

    assert run_scm_main([
        "--atm", str(atm_path),
        "--cldrad", str(cldrad_path),
        "--output", str(output_path),
    ]) == 0

    with np.load(output_path) as run:
        assert run["times"].shape == (48,)
        for key in (
            "model.condensation.precls",
            "model.convection.precnv",
            "model.shortwave_rad.rsds",
            "model.surface_flux.rlds",
            "obs.precip",
            "obs.sw_down_sfc",
        ):
            assert key in run
            assert np.all(np.isfinite(run[key]))

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["archive"] == str(output_path.resolve())
    assert manifest["retained_times"]["n_states"] == 48
    assert manifest["inputs"]["resolved_variables"]["temperature"] == "temp_p"
    assert manifest["configuration"]["land_surface_temperature"] == "BY_DATE time series"

    assert evaluate_main(["--run", str(output_path)]) == 0
    assert "Daily-mean comparison" in capsys.readouterr().out


def test_forecast_cache_uses_profile_evaluations_and_cloud_qc(tmp_path):
    atm, cldrad = build_synthetic_armbe(days=1)
    atm_path, cldrad_path = tmp_path / "atm.nc", tmp_path / "cldrad.nc"
    atm.to_netcdf(atm_path)
    cldrad.to_netcdf(cldrad_path)

    cldrad["qc_tot_cld"][6] = 1
    cldrad.to_netcdf(cldrad_path)
    cache = build_cache({
        "atm": str(atm_path),
        "cldrad": str(cldrad_path),
        "horizon_minutes": 360,
        "stride_minutes": 360,
    }, tmp_path / "cache")

    config = json.loads((cache / "config.json").read_text())
    recipe = json.loads((cache / "recipe.json").read_text())
    manifest = json.loads((cache / "manifest.json").read_text())
    windows = xr.open_dataset(cache / "windows.nc")
    assert config["target"]["observation"] == "cloud_fraction"
    assert recipe["target"]["model"] == "shortwave_rad.cloudc"
    assert config["physics_dt_minutes"] == 30
    assert config["observation_cadence_minutes"] == 360
    assert config["physics_dt_seconds"] == 1800
    assert config["observation_cadence_seconds"] == 21600
    assert manifest["config"]["horizon_seconds"] == 21600
    assert manifest["recipe"]["horizon_seconds"] == 21600
    assert recipe["target_order"] == "window, profile evaluation lead"
    assert windows["target"].shape == (3, 1)
    assert windows["target_mask"].shape == (3, 1)
    assert windows["target_mask"][0, 0] == 0
    assert windows["lead_time_seconds"].values.tolist() == [21600]
    assert windows["temperature"].shape == (3, 8)
    assert windows["surface_temperature"].attrs["source_variable"] == "temp_sfc"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"horizon_minutes": 361}, "horizon_minutes must be divisible"),
        ({"horizon_minutes": 30, "observation_cadence_minutes": 60}, "include at least one"),
        ({"stride_minutes": 30, "observation_cadence_minutes": 60}, "multiple of"),
    ],
)
def test_forecast_cache_validates_public_minute_timing(config, message):
    with pytest.raises(ValueError, match=message):
        resolved_config({"atm": "atm.nc", **config})


def test_load_config_rejects_duplicate_yaml_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("atm: first.nc\natm: second.nc\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_config(config_path)
