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


def _ten_minute_float32_means(n_days=31, seed=0):
    """Build a month of 10-minute float32 interval means (model output dtype)."""
    n = n_days * 144
    starts = (np.datetime64("2000-01-01T00:00:00", "ms")
              + np.arange(n) * np.timedelta64(600, "s"))
    values = (287.3 + np.random.default_rng(seed).normal(0.0, 5.0, (n, 4))
              ).astype(np.float32)
    ds = xr.Dataset(
        {"t": (("time", "x"), values, {"cell_methods": "time: mean"})},
        coords={"time": starts + np.timedelta64(300, "s")})
    ds["time_bounds"] = (("time", "bounds"),
                         np.stack([starts, starts + np.timedelta64(600, "s")], 1))
    ds.time.attrs["bounds"] = "time_bounds"
    return ds, values


def test_float32_stream_is_exact_and_resume_is_bit_identical():
    """Running sums are float64: exact for float32 inputs, restart-independent.

    A float32 running sum of 4464 ten-minute means drifts ~2e-4 K from the
    exact monthly mean, and ``state_dict`` restores the sums as float64, so a
    resumed stream would not reproduce an uninterrupted one. Both properties
    are part of the design's restart-independent-sums criterion.
    """
    ds, values = _ten_minute_float32_means()
    exact = values.astype(np.float64).mean(axis=0)

    whole = temporal_aggregation.MonthlyMeanAccumulator()
    assert whole.update(ds) is None
    uninterrupted = whole.finish()
    np.testing.assert_allclose(uninterrupted.t.values[0], exact,
                               rtol=1e-12, atol=0.0)
    batch = temporal_aggregation.monthly_means(ds)
    np.testing.assert_allclose(uninterrupted.t.values, batch.t.values,
                               rtol=1e-12, atol=0.0)

    # Resume through the checkpointable state at an arbitrary seam.
    split = 1000
    first = temporal_aggregation.MonthlyMeanAccumulator()
    assert first.update(ds.isel(time=slice(0, split))) is None
    resumed = temporal_aggregation.MonthlyMeanAccumulator.from_state_dict(
        first.state_dict())
    assert resumed.update(ds.isel(time=slice(split, None))) is None
    np.testing.assert_array_equal(resumed.finish().t.values,
                                  uninterrupted.t.values)


def test_rejected_streaming_chunk_does_not_mutate_pending_statistics():
    daily = _daily("2000-01-01", "2000-01-05")
    daily["zzz"] = ("time", np.arange(daily.sizes["time"], dtype=float),
                    {"cell_methods": "time: mean"})
    accumulator = temporal_aggregation.MonthlyMeanAccumulator()
    accumulator.update(daily.isel(time=slice(0, 2)))
    before = accumulator.state_dict()

    invalid = daily.isel(time=slice(2, 4)).copy()
    invalid["zzz"] = ("time", np.array(["bad", "chunk"]),
                      {"cell_methods": "time: mean"})
    with pytest.raises(TypeError, match="'zzz'.*not numeric"):
        accumulator.update(invalid)

    assert accumulator.state_dict() == before

    changed_dims = daily.isel(time=slice(2, 4)).copy()
    changed_dims["air"] = changed_dims.air.expand_dims(
        station=changed_dims.station
    ).transpose("time", "station")
    with pytest.raises(ValueError, match="changed dimensions"):
        accumulator.update(changed_dims)

    assert accumulator.state_dict() == before


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


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_millisecond_aligned_fine_bounds_are_accepted_unchanged(unit):
    daily = _daily("2000-01-01", "2000-02-01")
    ms = temporal_aggregation.monthly_means(daily)
    fine = daily.copy()
    fine["time_bounds"] = (("time", "bounds"),
                           daily.time_bounds.values.astype(f"datetime64[{unit}]"))
    xr.testing.assert_identical(temporal_aggregation.monthly_means(fine), ms)


@pytest.mark.parametrize("unit,offset", [("us", 1), ("ns", 500_000)])
def test_sub_millisecond_bounds_are_rejected_not_shifted(unit, offset):
    daily = _daily("2000-01-01", "2000-01-03")
    bounds = daily.time_bounds.values.astype(f"datetime64[{unit}]")
    bounds[0, 1] = bounds[0, 1] - np.timedelta64(offset, unit)
    daily["time_bounds"] = (("time", "bounds"), bounds)
    with pytest.raises(ValueError, match="millisecond precision"):
        temporal_aggregation.monthly_means(daily)
    with pytest.raises(ValueError, match="millisecond precision"):
        temporal_aggregation.MonthlyMeanAccumulator().update(daily)


def _gridded_daily(start, stop, nan_day=None):
    """Daily means on a small lat/lon grid with dotted names, as jcm writes."""
    ds = _daily(start, stop)
    n = ds.sizes["time"]
    rng = np.random.default_rng(0)
    field = rng.standard_normal((n, 3, 4))
    if nan_day is not None:
        field[nan_day, 1, 2] = np.nan
    ds["clouds.qc"] = (("time", "lat", "lon"), field)
    ds["clouds.qc"].attrs.update(cell_methods="time: mean", units="kg kg-1")
    ds = ds.assign_coords(lat=[-30.0, 0.0, 30.0], lon=[0.0, 90.0, 180.0, 270.0],
                          hybrid_a=("lat", [1.0, 2.0, 3.0]))
    ds.lat.attrs["units"] = "degrees_north"
    return ds


@pytest.mark.parametrize("nan_day", [None, 3])
def test_compact_restart_file_resumes_bit_identically(tmp_path, nan_day):
    """``save``/``load`` (the chunked CLI's restart file, #901) is exact.

    Covers a uniform valid duration (stored as one integer) and a missing
    value (stored as a full array), a non-dimension coordinate, a static
    variable and dotted field names.
    """
    ds = _gridded_daily("2000-02-20", "2000-03-05", nan_day=nan_day)
    whole = temporal_aggregation.MonthlyMeanAccumulator()
    closed = whole.update(ds)
    final = whole.finish()

    first = temporal_aggregation.MonthlyMeanAccumulator()
    assert first.update(ds.isel(time=slice(0, 5))) is None
    path = first.save(tmp_path / "state.monthly", clock="2000-02-25T00:00:00")
    resumed, meta = temporal_aggregation.MonthlyMeanAccumulator.load(path)
    assert meta == {"clock": "2000-02-25T00:00:00"}
    closed_resumed = resumed.update(ds.isel(time=slice(5, None)))
    xr.testing.assert_identical(closed_resumed, closed)
    xr.testing.assert_identical(resumed.finish(), final)
    assert not (tmp_path / "state.monthly.tmp").exists()


def test_compact_restart_file_of_an_empty_accumulator(tmp_path):
    empty = temporal_aggregation.MonthlyMeanAccumulator()
    restored, _ = temporal_aggregation.MonthlyMeanAccumulator.load(
        empty.save(tmp_path / "s"))
    assert restored.finish() is None
