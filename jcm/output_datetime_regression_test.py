"""Issue regressions for exact output labels and Gregorian month boundaries."""

from types import SimpleNamespace

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jcm.dycore.base import Predictions
from jcm.predictions import ModelPredictions, output_time_labels
from jcm.temporal_aggregation import MonthlyMeanAccumulator, monthly_means


def _exact_axis(start: str, count: int, step_seconds: int):
    origin = jdt.Datetime.from_datetime64(np.datetime64(start, "s"))
    return origin + jdt.Timedelta(
        seconds=jnp.arange(count, dtype=jnp.int32) * step_seconds)


@pytest.mark.parametrize("step_seconds", [600, 1200])
@pytest.mark.parametrize("start", ["2000-01-01T00:00:00",
                                    "2000-01-01T12:34:56"])
def test_issue_862_exact_labels_match_independent_timedelta_grid(
        step_seconds, start):
    count = 97
    actual = output_time_labels(_exact_axis(start, count, step_seconds))
    expected = (np.datetime64(start, "ms")
                + np.arange(count) * np.timedelta64(step_seconds, "s"))

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype("datetime64[ms]")


def test_issue_862_independent_axes_merge_without_union():
    count = 101
    atmosphere_time = output_time_labels(
        _exact_axis("2000-01-01T12:34:56", count, 600))
    downstream_time = (
        np.datetime64("2000-01-01T12:34:56", "ms")
        + np.arange(count) * np.timedelta64(600, "s"))
    atmosphere = xr.Dataset(
        {"atmosphere": ("time", np.arange(count))},
        coords={"time": atmosphere_time})
    downstream = xr.Dataset(
        {"downstream": ("time", np.arange(count))},
        coords={"time": downstream_time})

    merged = xr.merge([atmosphere, downstream])

    assert merged.sizes["time"] == count
    np.testing.assert_array_equal(merged.time, downstream_time)


def test_trajectory_serialization_calls_public_output_labeller(monkeypatch):
    import jcm.predictions as predictions_module

    calls = []
    real_labeller = predictions_module.output_time_labels

    def recording_labeller(values):
        calls.append(values)
        return real_labeller(values)

    monkeypatch.setattr(predictions_module, "output_time_labels",
                        recording_labeller)

    class FakeDycore:
        def to_xarray(self, predictions, times):
            return xr.Dataset(
                {"temperature": ("time", np.array([280.0, 281.0]))},
                coords={"time": times})

    coords = SimpleNamespace(horizontal=SimpleNamespace())
    physics = SimpleNamespace(output_attrs=lambda: {})
    raw = Predictions(
        dynamics=None,
        physics=None,
        times=_exact_axis("2000-01-01T12:34:56", 2, 600),
    )

    result = ModelPredictions(
        raw, coords, physics, dycore=FakeDycore()).to_xarray()

    assert len(calls) == 1
    np.testing.assert_array_equal(
        result.time,
        np.array(["2000-01-01T12:34:56.000",
                  "2000-01-01T12:44:56.000"], dtype="datetime64[ms]"))


def _bounded_daily(start: str, end: str) -> xr.Dataset:
    edges = np.arange(np.datetime64(start, "D"),
                      np.datetime64(end, "D") + np.timedelta64(1, "D"),
                      dtype="datetime64[D]").astype("datetime64[ms]")
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    midpoints = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 2
    ds = xr.Dataset(
        {"value": ("time", np.arange(len(midpoints), dtype=float)),
         "time_bounds": (("time", "bounds"), bounds)},
        coords={"time": midpoints})
    ds.time.attrs["bounds"] = "time_bounds"
    ds.value.attrs["cell_methods"] = "time: mean"
    return ds


@pytest.mark.parametrize(("year", "february_days"),
                         [(1900, 28), (2000, 29), (2100, 28)])
def test_century_leap_rules_and_month_edge_membership(year, february_days):
    daily = _bounded_daily(f"{year}-01-31", f"{year}-03-02")

    result = monthly_means(daily)

    expected_bounds = np.array([
        [f"{year}-01-31", f"{year}-02-01"],
        [f"{year}-02-01", f"{year}-03-01"],
        [f"{year}-03-01", f"{year}-03-02"],
    ], dtype="datetime64[ms]")
    np.testing.assert_array_equal(result.time_bounds, expected_bounds)
    np.testing.assert_allclose(
        result.time_coverage_fraction,
        [1 / 31, 1.0, 1 / 31])
    assert int(result.time_coverage.values[1] / np.timedelta64(1, "D")) \
        == february_days


def test_accumulator_checkpoint_on_month_boundary_matches_batch():
    daily = _bounded_daily("2000-01-01", "2000-03-01")
    batch = monthly_means(daily)
    january = daily.where(
        daily.time_bounds.isel(bounds=0) < np.datetime64("2000-02-01"),
        drop=True)
    february = daily.where(
        daily.time_bounds.isel(bounds=0) >= np.datetime64("2000-02-01"),
        drop=True)

    accumulator = MonthlyMeanAccumulator()
    assert accumulator.update(january) is None
    accumulator = MonthlyMeanAccumulator.from_state_dict(
        accumulator.state_dict())
    closed = accumulator.update(february)
    pending = accumulator.finish()
    streamed = xr.concat([closed, pending], dim="time", data_vars="all")

    xr.testing.assert_allclose(streamed, batch)


def test_output_time_labels_rejects_legacy_float_days():
    with pytest.raises(TypeError, match="jax_datetime.Datetime or datetime64"):
        output_time_labels(np.array([10957.0]))


def test_output_time_labels_rejects_nat_and_historical_128ns_offset():
    with pytest.raises(ValueError, match="NaT"):
        output_time_labels(np.array(["NaT"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="millisecond precision"):
        output_time_labels(np.array(
            ["2000-01-01T00:00:00.000000128"], dtype="datetime64[ns]"))


def test_output_time_labels_accepts_half_second_and_exact_pre_epoch_time():
    values = np.array(["1969-12-31T23:59:59.500000",
                       "2000-01-01T00:00:00.500000"],
                      dtype="datetime64[us]")

    result = output_time_labels(values)

    np.testing.assert_array_equal(
        result,
        np.array(["1969-12-31T23:59:59.500",
                  "2000-01-01T00:00:00.500"], dtype="datetime64[ms]"))


def test_output_time_labels_handles_scaled_numpy_datetime_units():
    exact_millisecond = np.array([1, -1], dtype="datetime64[1000us]")
    submillisecond = np.array([1], dtype="datetime64[500us]")

    np.testing.assert_array_equal(
        output_time_labels(exact_millisecond),
        np.array([1, -1], dtype="datetime64[ms]"))
    with pytest.raises(ValueError, match="millisecond precision"):
        output_time_labels(submillisecond)
