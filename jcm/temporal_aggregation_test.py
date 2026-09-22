"""Tests for bounds-aware output aggregation."""

import numpy as np
import pytest
import xarray as xr

from jcm import temporal_aggregation


def _daily(start, stop):
    edges = np.arange(np.datetime64(start),
                      np.datetime64(stop) + np.timedelta64(1, "D"),
                      np.timedelta64(1, "D")).astype("datetime64[ns]")
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    midpoints = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 2
    values = np.arange(len(midpoints), dtype=float)
    ds = xr.Dataset(
        {"air": ("time", values),
         "time_bounds": (("time", "bounds"), bounds),
         "constant": ("station", [3.0, 4.0])},
        coords={"time": midpoints, "station": [10, 11]},
        attrs={"source": "test"},
    )
    ds.time.attrs["bounds"] = "time_bounds"
    ds.air.attrs.update(cell_methods="time: mean", units="K")
    return ds


def test_january_and_leap_february_have_real_month_membership(tmp_path):
    daily = _daily("2000-01-01", "2000-03-01")
    result = temporal_aggregation.monthly_means(daily)

    np.testing.assert_array_equal(
        result.time_bounds.values[:, 0],
        np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(
        result.time_bounds.values[:, 1],
        np.array(["2000-02-01", "2000-03-01"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(result.air, [15.0, 45.0])
    np.testing.assert_allclose(result.time_coverage_fraction, [1.0, 1.0])
    assert result.time.values[0] == np.datetime64("2000-01-16T12:00")
    assert result.air.attrs["units"] == "K"
    np.testing.assert_array_equal(result.constant, [3.0, 4.0])
    path = tmp_path / "monthly.nc"
    result.to_netcdf(path)
    with xr.open_dataset(path) as decoded:
        np.testing.assert_array_equal(decoded.time, result.time)
        np.testing.assert_array_equal(decoded.time_bounds, result.time_bounds)


def test_nan_weighting_and_partial_month_coverage_are_per_variable():
    daily = _daily("2000-01-10", "2000-02-01")
    daily["other"] = daily.air.copy()
    daily["other"].attrs["cell_methods"] = "time: mean"
    daily["flag"] = ("time", np.arange(daily.sizes["time"]) % 2 == 0,
                     {"cell_methods": "time: mean"})
    daily["air"][5] = np.nan
    result = temporal_aggregation.monthly_means(daily)

    expected = np.delete(np.arange(22.0), 5).mean()
    assert result.air.item() == pytest.approx(expected)
    assert result.other.item() == pytest.approx(np.arange(22.0).mean())
    assert result.flag.item() == pytest.approx(0.5)
    assert result.time_coverage_fraction.item() == pytest.approx(22 / 31)
    np.testing.assert_array_equal(
        result.time_bounds.values[0],
        np.array(["2000-01-10", "2000-02-01"], dtype="datetime64[ns]"),
    )


def test_rejects_instantaneous_and_month_crossing_intervals():
    instantaneous = _daily("2000-01-01", "2000-01-03")
    instantaneous.air.attrs.pop("cell_methods")
    with pytest.raises(ValueError, match="interval means only"):
        temporal_aggregation.monthly_means(instantaneous)

    contradictory = _daily("2000-01-01", "2000-01-02")
    contradictory.air.attrs["cell_methods"] = "time: mean time: maximum"
    with pytest.raises(ValueError, match="interval means only"):
        temporal_aggregation.monthly_means(contradictory)

    crossing = _daily("2000-01-31", "2000-02-02").isel(time=[0])
    crossing.time_bounds.values[0] = np.array(
        ["2000-01-31", "2000-02-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="crosses a month boundary"):
        temporal_aggregation.monthly_means(crossing)

    with pytest.raises(ValueError, match="at least one"):
        temporal_aggregation.monthly_means(crossing.isel(time=slice(0, 0)))

    nat = _daily("2000-01-01", "2000-01-02")
    nat.time_bounds.values[0, 0] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="NaT"):
        temporal_aggregation.monthly_means(nat)


def test_streaming_chunk_resume_matches_batch_and_retains_only_statistics():
    daily = _daily("2000-01-01", "2000-03-01")
    daily.air.values[7] = np.nan
    daily["flag"] = ("time", np.arange(daily.sizes["time"]) % 2 == 0,
                     {"cell_methods": "time: mean"})
    batch = temporal_aggregation.monthly_means(daily)

    accumulator = temporal_aggregation.MonthlyMeanAccumulator()
    assert accumulator.update(daily.isel(time=slice(0, 17))) is None
    state = accumulator.state_dict()
    assert "air" in state["sums"]
    assert "time" not in state["sums"]["air"]["dims"]
    accumulator = temporal_aggregation.MonthlyMeanAccumulator.from_state_dict(state)
    closed = accumulator.update(daily.isel(time=slice(17, None)))
    final = accumulator.finish()
    streamed = xr.concat([closed, final], dim="time", data_vars="all")

    xr.testing.assert_allclose(streamed.air, batch.air)
    np.testing.assert_array_equal(streamed.time_bounds, batch.time_bounds)
    np.testing.assert_allclose(streamed.time_coverage_fraction,
                               batch.time_coverage_fraction)

    overlapping = temporal_aggregation.MonthlyMeanAccumulator()
    overlapping.update(daily.isel(time=slice(0, 3)))
    with pytest.raises(ValueError, match="across streaming chunks"):
        overlapping.update(daily.isel(time=slice(2, 4)))

    changed = temporal_aggregation.MonthlyMeanAccumulator()
    changed.update(daily.isel(time=slice(0, 2)))
    with pytest.raises(ValueError, match="same time-dependent variables"):
        changed.update(daily.drop_vars("air").isel(time=slice(2, 3)))

    shifted = daily.isel(time=slice(2, 3)).assign_coords(station=[12, 13])
    with pytest.raises(ValueError, match="spatial coordinates"):
        changed.update(shifted)


def test_half_second_midpoint_and_dates_beyond_nanosecond_range_are_exact():
    bounds = np.array([["2500-01-01T00:00:00.000",
                        "2500-01-01T00:00:01.000"]],
                      dtype="datetime64[ms]")
    ds = xr.Dataset(
        {"x": ("time", [2.0]),
         "time_bounds": (("time", "bounds"), bounds)},
        coords={"time": bounds[:, 0] + np.timedelta64(500, "ms")},
    )
    ds.time.attrs["bounds"] = "time_bounds"
    ds.x.attrs["cell_methods"] = "time: mean"

    result = temporal_aggregation.monthly_means(ds)

    assert str(result.time.values[0]) == "2500-01-01T00:00:00.500"
    np.testing.assert_array_equal(result.time_bounds, bounds)
