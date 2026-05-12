"""Tests for ``jcm/data/bc/interpolate.py``.

Covers the helpers actually consumed by ``jcm.terrain`` and ``jcm.forcing``
(``interpolate_to_daily``, ``upsample_forcings_ds``, ``upsample_terrain_ds``).
The previous ``main()`` smoke test was dropped (issue #310) because it
wrote ``forcing_daily.nc`` / ``forcing_t31.nc`` / ``terrain_t31.nc`` to
the working directory on every pytest run while asserting nothing about
the output.
"""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from jcm.data.bc.interpolate import (
    interpolate_to_daily,
    upsample_forcings_ds,
    upsample_terrain_ds,
)
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES
from jcm.utils import get_coords


def _t21_grid():
    return get_coords(SIGMA_LAYER_BOUNDARIES[7], spectral_truncation=21).horizontal


def _monthly_source(n_lon: int = 8, n_lat: int = 4) -> xr.Dataset:
    times = pd.date_range("1981-01-01", periods=12, freq="MS")
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lat = np.linspace(-80.0, 80.0, n_lat)
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "sst": (("lon", "lat", "time"), rng.uniform(size=(n_lon, n_lat, 12))),
            "orog": (("lon", "lat"), rng.uniform(size=(n_lon, n_lat))),
        },
        coords={"lon": lon, "lat": lat, "time": times},
    )


class TestInterpolateToDaily(unittest.TestCase):

    def test_produces_365_days(self):
        ds = _monthly_source()
        daily = interpolate_to_daily(ds)
        self.assertEqual(len(daily["time"]), 365)
        # Static (non-time) variables pass through unchanged.
        np.testing.assert_array_equal(daily["orog"].values, ds["orog"].values)
        # Time-varying variables keep their non-time shape.
        self.assertEqual(daily["sst"].shape[:-1], ds["sst"].shape[:-1])

    def test_rejects_wrong_number_of_timestamps(self):
        ds = _monthly_source().isel(time=slice(0, 6))
        with self.assertRaises(ValueError):
            interpolate_to_daily(ds)

    def test_rejects_non_monthly_frequency(self):
        ds = _monthly_source().assign_coords(
            time=pd.date_range("1981-01-01", periods=12, freq="D"),
        )
        with self.assertRaises(ValueError):
            interpolate_to_daily(ds)


class TestUpsampleForcings(unittest.TestCase):

    def _source(self, n_lon: int = 8, n_lat: int = 4) -> xr.Dataset:
        lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = np.linspace(-80.0, 80.0, n_lat)
        rng = np.random.default_rng(1)
        # Fraction variables seeded above 1 to exercise the upper clip;
        # ``sst`` seeded with negatives to exercise the >=0 clip.
        return xr.Dataset(
            {
                "icec":     (("lon", "lat"), rng.uniform(0.5, 1.5, (n_lon, n_lat))),
                "soilw_am": (("lon", "lat"), rng.uniform(0.5, 1.5, (n_lon, n_lat))),
                "alb":      (("lon", "lat"), rng.uniform(0.5, 1.5, (n_lon, n_lat))),
                "sst":      (("lon", "lat"), rng.uniform(-5.0, 5.0, (n_lon, n_lat))),
            },
            coords={"lon": lon, "lat": lat},
        )

    def test_clips_fractions_to_unit_interval(self):
        out = upsample_forcings_ds(self._source(), _t21_grid())
        for v in ("icec", "soilw_am", "alb"):
            self.assertGreaterEqual(float(out[v].min()), 0.0, msg=v)
            self.assertLessEqual(float(out[v].max()), 1.0, msg=v)

    def test_clips_other_vars_to_nonnegative(self):
        out = upsample_forcings_ds(self._source(), _t21_grid())
        self.assertGreaterEqual(float(out["sst"].min()), 0.0)

    def test_output_lands_on_target_grid(self):
        out = upsample_forcings_ds(self._source(), _t21_grid())
        # T21 nodal grid is 64 lon × 32 lat.
        self.assertEqual(out.sizes["lon"], 64)
        self.assertEqual(out.sizes["lat"], 32)


class TestUpsampleTerrain(unittest.TestCase):

    def _source(self, n_lon: int = 8, n_lat: int = 4) -> xr.Dataset:
        lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
        lat = np.linspace(-80.0, 80.0, n_lat)
        rng = np.random.default_rng(2)
        return xr.Dataset(
            {
                "lsm":  (("lon", "lat"), rng.uniform(-0.5, 1.5, (n_lon, n_lat))),
                "orog": (("lon", "lat"), rng.uniform(-200.0, 3000.0, (n_lon, n_lat))),
            },
            coords={"lon": lon, "lat": lat},
        )

    def test_clips_lsm_to_unit_interval(self):
        out = upsample_terrain_ds(self._source(), _t21_grid())
        self.assertGreaterEqual(float(out["lsm"].min()), 0.0)
        self.assertLessEqual(float(out["lsm"].max()), 1.0)

    def test_preserves_negative_orography(self):
        # Real terrain has below-sea-level points (Dead Sea, etc.); the
        # upsampler intentionally does not clip orog.
        out = upsample_terrain_ds(self._source(), _t21_grid())
        self.assertLess(float(out["orog"].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
