"""End-to-end real-date output and restart contract for the v3 interface."""

import numpy as np
import xarray as xr

from jcm.checkpoint import load_checkpoint, save_checkpoint
from jcm.model import Model
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords


def _model():
    return Model(coords=get_held_suarez_coords(layers=8, spectral_truncation=21),
                 physics=held_suarez_physics(), time_step=180,
                 start_time="2000-01-31")


def test_monthly_output_and_checkpoint_share_the_same_clock(tmp_path):
    """Jan 31 belongs to January, including when restarting on Feb 1."""
    model = _model()
    daily = model.run(end_time="2000-02-02", save_interval="1D",
                      output_averages=True)
    monthly = daily.monthly_means()
    np.testing.assert_array_equal(
        monthly.time_bounds.values,
        np.array([["2000-01-31", "2000-02-01"],
                  ["2000-02-01", "2000-02-02"]], dtype="datetime64[ns]"))
    np.testing.assert_allclose(monthly.time_coverage_fraction, [1 / 31, 1 / 29])

    first = _model()
    jan = first.run(end_time="2000-02-01", save_interval="1D", output_averages=True)
    path = tmp_path / "february.ckpt"
    save_checkpoint(first, path)
    second = _model()
    second.bootstrap_state()
    load_checkpoint(second, path)
    feb = second.resume(end_time="2000-02-02", save_interval="1D", output_averages=True)
    combined = xr.concat([jan.to_xarray(), feb.to_xarray()], dim="time")
    # Exact timestamps and physical fields agree across the restart seam.
    xr.testing.assert_allclose(daily.to_xarray(), combined)


def test_odd_length_mean_intervals_are_labelled_at_exact_half_seconds():
    """A 1 s step averaged over 3 s has a half-second midpoint, exactly.

    The traced clock holds whole seconds, so the label comes from the exact
    bounds in milliseconds, and monthly means weight the intervals by those
    bounds across the Jan/Feb seam.
    """
    model = Model(coords=get_held_suarez_coords(layers=8, spectral_truncation=21),
                  physics=held_suarez_physics(), time_step=1 / 60,
                  start_time="2000-01-31T23:59:54")
    preds = model.run(total_time="12 seconds", save_interval="3 seconds",
                      output_averages=True)
    starts = (np.datetime64("2000-01-31T23:59:54", "ms")
              + np.arange(4) * np.timedelta64(3, "s"))
    expected = starts + np.timedelta64(1500, "ms")
    np.testing.assert_array_equal(preds.time_labels(), expected)
    ds = preds.to_xarray()
    np.testing.assert_array_equal(ds.time.values.astype("datetime64[ms]"),
                                  expected)
    np.testing.assert_array_equal(
        ds.time_bounds.values.astype("datetime64[ms]"),
        np.stack([starts, starts + np.timedelta64(3, "s")], axis=1))
    # The datetime64[ms] round trip is exact.
    assert np.array_equal(
        expected.astype("datetime64[ns]").astype("datetime64[ms]"), expected)

    monthly = preds.monthly_means()
    temperature = ds.temperature.values.astype(np.float64)
    np.testing.assert_allclose(monthly.temperature.values[0],
                               temperature[:2].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(monthly.temperature.values[1],
                               temperature[2:].mean(axis=0), rtol=1e-6)
    np.testing.assert_array_equal(
        monthly.time_coverage.values.astype("timedelta64[ms]"),
        np.array([6000, 6000], dtype="timedelta64[ms]"))
