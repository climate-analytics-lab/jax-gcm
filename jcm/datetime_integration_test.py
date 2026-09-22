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
