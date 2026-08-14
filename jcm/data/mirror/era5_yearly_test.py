"""Unit tests for the pure construction logic in era5_yearly.

Everything here runs on synthetic data — the Glade I/O paths
(``_open_an_month``, ``_land_month_mean``) are monkeypatched, so these
tests verify the centred-window arithmetic, the month-start blend, the
ice-sheet-mask discipline and the GHG extrapolation without any mirror
source data.
"""

import unittest
from unittest import mock

import numpy as np
import xarray as xr

from jcm.data.mirror import era5_yearly
from jcm.data.mirror.bundles import translate_land
from jcm.physics.speedy.physical_constants import sd2sc

_EPOCH = np.datetime64("1990-01-01")


def _fake_an_month(code, year, month, every=6):
    """Synthetic 6-hourly month: value = hours since 1990 (+ per-code
    offset), constant in space except one always-NaN 'land' cell.
    """
    start = np.datetime64(f"{year:04d}-{month:02d}-01")
    y2, m2 = (year + 1, 1) if month == 12 else (year, month + 1)
    end = np.datetime64(f"{y2:04d}-{m2:02d}-01")
    times = np.arange(start, end, np.timedelta64(every, "h"))
    hours = (times - _EPOCH) / np.timedelta64(1, "h")
    offset = 1000.0 if code == "128_034_sstk" else 0.0
    vals = np.broadcast_to((offset + hours)[:, None, None],
                           (len(times), 2, 3)).copy()
    vals[:, 0, 0] = np.nan
    return xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                        coords={"time": times,
                                "latitude": [0.0, 1.0],
                                "longitude": [0.0, 1.0, 2.0]})


class MonthMidpointTest(unittest.TestCase):
    def test_exact_midpoints(self):
        self.assertEqual(era5_yearly._month_midpoint(2003, 1),
                         np.datetime64("2003-01-16T12"))
        self.assertEqual(era5_yearly._month_midpoint(2003, 2),
                         np.datetime64("2003-02-15T00"))     # 28 days
        self.assertEqual(era5_yearly._month_midpoint(2004, 2),
                         np.datetime64("2004-02-15T12"))     # leap
        self.assertEqual(era5_yearly._month_midpoint(2003, 12),
                         np.datetime64("2003-12-16T12"))


class SstIceConstructionTest(unittest.TestCase):
    @mock.patch.object(era5_yearly, "_open_an_month", _fake_an_month)
    def test_month_start_values_are_centred_window_means(self):
        ds = era5_yearly.build_sstice_year(1995)
        self.assertEqual(ds.sstk.shape, (12, 2, 3))
        np.testing.assert_array_equal(
            ds.time.values, era5_yearly._month_starts(1995))
        # Independently recompute each window mean from the synthetic
        # series: steps in [mid(m-1), mid(m)) drawn from both months.
        for m in (1, 6, 12):
            prev_y, prev_m = (1994, 12) if m == 1 else (1995, m - 1)
            steps = xr.concat([_fake_an_month("128_034_sstk", prev_y, prev_m),
                               _fake_an_month("128_034_sstk", 1995, m)],
                              dim="time")
            lo = era5_yearly._month_midpoint(prev_y, prev_m)
            hi = era5_yearly._month_midpoint(1995, m)
            window = steps.sel(time=(steps.time >= lo) & (steps.time < hi))
            expected = window.mean("time").values
            np.testing.assert_allclose(
                ds.sstk.isel(time=m - 1).values[1:], expected[1:],
                rtol=1e-6)
        # per-code offset keeps the two variables distinct
        np.testing.assert_allclose(
            (ds.sstk - ds.ci).values[:, 1:, :], 1000.0, rtol=1e-9)

    @mock.patch.object(era5_yearly, "_open_an_month", _fake_an_month)
    def test_static_land_mask_propagates_as_nan(self):
        ds = era5_yearly.build_sstice_year(1995)
        self.assertTrue(np.isnan(ds.sstk.values[:, 0, 0]).all())
        self.assertTrue(np.isfinite(ds.sstk.values[:, 1:, :]).all())

    def test_pre_1941_rejected(self):
        with self.assertRaises(ValueError):
            era5_yearly.build_sstice_year(1940)


class BlendTest(unittest.TestCase):
    def test_blend_is_adjacent_mean_on_month_starts(self):
        vals = np.arange(13, dtype=float)[:, None] * np.ones((13, 2))
        da = xr.DataArray(vals, dims=("time", "latitude"),
                          coords={"time": np.arange(13), "latitude": [0, 1]})
        times = era5_yearly._month_starts(2001)
        out = era5_yearly._blend_to_month_starts(da, times)
        np.testing.assert_allclose(out.values[:, 0],
                                   np.arange(12) + 0.5)
        np.testing.assert_array_equal(out.time.values, times)


class IceSheetMaskTest(unittest.TestCase):
    def test_snowc_zeroed_only_where_climatological_mask_says(self):
        # Two cells with identical deep transient snow; the mask (from a
        # fixed climatology, not this data) declares only cell 0 an ice
        # sheet — so a snowy year cannot make ice sheets flicker.
        sd = xr.DataArray(np.full((3, 2), 0.5),
                          dims=("time", "cell"))
        ones = xr.ones_like(sd)
        era5 = xr.Dataset({"sd": sd, "stl1": ones * 280.0,
                           "swvl1": ones * 0.2, "swvl2": ones * 0.2,
                           "cvh": ones.isel(time=0) * 0.5,
                           "cvl": ones.isel(time=0) * 0.5})
        mask = xr.DataArray([True, False], dims=("cell",))
        out = translate_land(era5, permanent_snow=mask)
        np.testing.assert_array_equal(out["snowc"].values[:, 0], 0.0)
        np.testing.assert_array_equal(out["snowc"].values[:, 1],
                                      np.minimum(500.0 / sd2sc, 1.0))
        self.assertTrue(((out["soilw_am"].values >= 0)
                         & (out["soilw_am"].values <= 1)).all())
        np.testing.assert_array_equal(out["stl"].values, 280.0)


class GhgExtrapolationTest(unittest.TestCase):
    def test_linear_series_continues_exactly(self):
        years = np.arange(2013, 2023)
        values = 400.0 + 2.5 * (years - 2013)
        self.assertAlmostEqual(
            era5_yearly._extrapolate_linear(years, values, 2024),
            400.0 + 2.5 * 11, places=9)


if __name__ == "__main__":
    unittest.main()
