"""Regression tests for the ARMBE-to-SPEEDY adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from armbe_io import geopotential_on_sigma, to_obs_targets, to_state_series
from jcm.date import DateData
from run_scm import build_forcing, start_date_from_timestamp, validate_cadence


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
    np.testing.assert_array_equal(meta["retained_indices"], [0, 2])
    np.testing.assert_array_equal(times, ds.time.values[[0, 2]])
    np.testing.assert_array_equal(
        to_obs_targets(ds, meta["retained_indices"])["precip"], [1.0, 3.0],
    )


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
