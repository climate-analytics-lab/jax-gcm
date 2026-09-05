"""Tests for the ERA5 regrid helper (synthetic dataset, no cloud access)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from jcm.physics.bias_correction.era5_data import (
    cadence_slicing,
    drop_leap_day,
    era5_ds_to_model_grid,
    load_era5,
    load_era5_span,
    model_clock_seconds,
    times_to_seconds,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


def _synthetic_era5(ntime=2, nlev=5, nlat=8, nlon=16):
    """ERA5-convention dataset: (time, level, lat, lon), level in hPa, q in kg/kg."""
    times = pd.date_range("2001-01-01", periods=ntime, freq="6h")
    levels = np.array([100.0, 300.0, 500.0, 700.0, 925.0])[:nlev]
    lats = np.linspace(-80, 80, nlat)
    lons = np.linspace(0, 337.5, nlon)
    shape = (ntime, nlev, nlat, nlon)

    def var(value):
        return (("time", "level", "lat", "lon"),
                np.full(shape, value, dtype=np.float32))

    return xr.Dataset(
        {
            "u_wind": var(10.0),
            "v_wind": var(-3.0),
            "temperature": var(260.0),
            "specific_humidity": var(5e-3),   # kg/kg, raw ERA5 convention
        },
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )


class TestCadenceSlicing(unittest.TestCase):
    """The slot arithmetic notebook 08's target indexing relies on."""

    def test_six_hourly_takes_every_slot(self):
        stride, n_slices = cadence_slicing(6, 363)
        self.assertEqual(stride, 1)
        self.assertEqual(n_slices, 1452)   # covers the 96-step lead boundary

    def test_daily_reproduces_m2_slices(self):
        stride, n_slices = cadence_slicing(24, 360)
        self.assertEqual(stride, 4)
        self.assertEqual(n_slices, 1440)
        # slice(0, 1440, 4) yields exactly 360 daily 00Z slots
        self.assertEqual(len(range(0, n_slices, stride)), 360)

    def test_rejects_off_grid_cadence(self):
        with self.assertRaises(ValueError):
            cadence_slicing(5, 10)


class TestEra5Regrid(unittest.TestCase):
    """Shapes, ordering, and the kg/kg -> g/kg conversion."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.ds = _synthetic_era5()

    def test_output_shapes_match_model_grid(self):
        fields = era5_ds_to_model_grid(self.ds, self.coords)
        nlev = self.coords.vertical.layers
        nlon, nlat = self.coords.horizontal.nodal_shape
        for name, a in fields.items():
            self.assertEqual(a.shape, (2, nlev, nlon, nlat), name)

    def test_humidity_converted_to_g_per_kg(self):
        fields = era5_ds_to_model_grid(self.ds, self.coords)
        # Constant 5e-3 kg/kg must arrive as 5.0 g/kg everywhere.
        np.testing.assert_allclose(fields["specific_humidity"], 5.0,
                                   rtol=1e-5)
        # And the non-humidity fields are untouched.
        np.testing.assert_allclose(fields["temperature"], 260.0, rtol=1e-5)

    def test_times_to_seconds_matches_epoch(self):
        seconds = times_to_seconds(self.ds.time.values)
        start_2001 = (pd.Timestamp("2001-01-01")
                      - pd.Timestamp("1970-01-01")).total_seconds()
        self.assertEqual(seconds[0], start_2001)
        self.assertEqual(seconds[1] - seconds[0], 6 * 3600.0)

    def test_longitude_seam_wraps_not_extrapolates(self):
        # Longitude is periodic; the regrid must wrap past the last source
        # longitude, not linearly extrapolate. Constant fields cannot detect
        # this (extrapolating a constant is accidentally correct), so use a
        # smooth lon-dependent field: T = 260 + 10*cos(lon). Model longitudes
        # beyond the source's last column (337.5 deg here) must interpolate
        # between lon=337.5 and lon=360(=0), staying within the true wrapped
        # values -- extrapolation instead continues the local slope and
        # overshoots badly near the seam.
        ds = self.ds.copy(deep=True)
        lon_rad = np.deg2rad(np.asarray(ds.lon.values))
        pattern = 10.0 * np.cos(lon_rad)                     # (nlon,)
        t = np.full((2, 5, 8, 16), 260.0) + pattern[None, None, None, :]
        ds["temperature"] = (("time", "level", "lat", "lon"),
                             t.astype(np.float32))
        fields = era5_ds_to_model_grid(ds, self.coords)
        lon_model = np.degrees(
            np.asarray(self.coords.horizontal.longitudes))    # (nlon,)
        truth = 260.0 + 10.0 * np.cos(np.deg2rad(lon_model))
        got = fields["temperature"][0, 0, :, 0]               # (nlon,) one lat
        # Linear interp of a smooth cosine on a 22.5-deg grid is accurate to
        # well under 1 K; the pre-fix extrapolation error at the seam was
        # several K and growing with distance past the last source column.
        seam = lon_model > float(ds.lon.values.max())
        self.assertTrue(seam.any(), "test needs model lons past the source max")
        np.testing.assert_allclose(got[seam], truth[seam], atol=1.0)
        np.testing.assert_allclose(got, truth, atol=1.0)


class TestLeapDayHandling(unittest.TestCase):
    """29 February must not reach a model running SPEEDY's 365_day calendar."""

    @staticmethod
    def _axis(start, periods):
        times = pd.date_range(start, periods=periods, freq="6h")
        return xr.Dataset({"x": ("time", np.zeros(periods))},
                          coords={"time": times})

    def test_noop_on_non_leap_year(self):
        # 2001 has no 29 February, so this must return the axis untouched --
        # the regression guard for every result produced before this change.
        ds = self._axis("2001-01-01", 4 * 365)
        kept = drop_leap_day(ds)
        self.assertEqual(kept.sizes["time"], 4 * 365)
        np.testing.assert_array_equal(np.asarray(kept.time.values),
                                      np.asarray(ds.time.values))

    def test_removes_only_the_leap_day(self):
        ds = self._axis("2004-01-01", 4 * 366)
        kept = drop_leap_day(ds)
        self.assertEqual(kept.sizes["time"], 4 * 365)   # exactly 4 slots gone
        t = pd.DatetimeIndex(np.asarray(kept.time.values))
        self.assertFalse(((t.month == 2) & (t.day == 29)).any())


class TestModelClock(unittest.TestCase):
    """The 365-day clock the BY_DATE nudging lookup is aligned against."""

    def test_matches_real_timestamps_in_a_non_leap_year(self):
        # `absolute_seconds_since_epoch` is calendar-agnostic real seconds, so
        # in a non-leap year the synthetic clock IS the true axis. This is what
        # makes the change a no-op for the 2001/2002 runs.
        real = times_to_seconds(
            pd.date_range("2001-01-01", periods=4 * 365, freq="6h").values)
        clock = model_clock_seconds(2001, 4 * 365, 6)
        np.testing.assert_array_equal(clock, real)

    def test_is_uniform_regardless_of_leap_years(self):
        # Spanning 2004 (leap): the real calendar would step an extra day at
        # the year break, which is exactly the drift a 365_day model must not
        # inherit.
        clock = model_clock_seconds(2004, 4 * 365 * 2, 6)
        np.testing.assert_array_equal(np.diff(clock), 6 * 3600.0)


def _write_synthetic_store(path):
    """Write an ERA5-convention zarr spanning a leap year and into the next.

    Temperature encodes the source day-of-year, so a regridded sample
    identifies the exact calendar day it came from, which is what makes the
    pairing assertions below meaningful rather than shape checks.
    """
    times = pd.date_range("2004-01-01", "2005-01-10 18:00", freq="6h")
    nlev, nlat, nlon = 3, 6, 8
    doy = np.asarray(times.dayofyear, dtype=np.float32)
    shape = (len(times), nlev, nlat, nlon)
    dims = ("time", "level", "latitude", "longitude")

    def const(value):
        return dims, np.full(shape, value, dtype=np.float32)

    ds = xr.Dataset(
        {
            "u_component_of_wind": const(10.0),
            "v_component_of_wind": const(-3.0),
            # 260 + day-of-year, uniform in space and height.
            "temperature": (dims, np.broadcast_to(
                260.0 + doy[:, None, None, None], shape).astype(np.float32)),
            "specific_humidity": const(5e-3),      # kg/kg, raw ERA5
        },
        coords={"time": times,
                "level": np.array([300.0, 500.0, 850.0])[:nlev],
                "latitude": np.linspace(-80, 80, nlat),
                "longitude": np.linspace(0, 315, nlon)},
    )
    ds.to_zarr(path)


class TestLoadEra5Pairing(unittest.TestCase):
    """End-to-end load: which calendar day lands in which sample slot."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.url = str(Path(cls._tmp.name) / "era5.zarr")
        _write_synthetic_store(cls.url)
        cls.coords = get_speedy_coords(layers=8, spectral_truncation=21)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _doy_of(fields, k):
        """Day-of-year encoded in sample ``k`` (uniform, so take any point)."""
        return float(np.asarray(fields["temperature"])[k].flat[0]) - 260.0

    def test_leap_day_is_skipped_in_the_sample_sequence(self):
        # 2004 day-of-year: Feb 28 = 59, Feb 29 = 60, Mar 1 = 61. After the
        # drop, slot 58 is Feb 28 and slot 59 must be MAR 1 -- matching the
        # model's own day 59 on a calendar with no 29 February.
        fields, _ = load_era5(self.coords, 2004, 70, cadence_hours=24,
                              url=self.url)
        self.assertEqual(self._doy_of(fields, 58), 59.0)   # 28 Feb
        self.assertEqual(self._doy_of(fields, 59), 61.0)   # 1 Mar, not 29 Feb
        seen = [self._doy_of(fields, k) for k in range(70)]
        self.assertNotIn(60.0, seen, "29 February reached the model grid")

    def test_clock_is_the_model_clock_not_the_gregorian_one(self):
        _, seconds = load_era5(self.coords, 2004, 70, cadence_hours=24,
                               url=self.url)
        start = (pd.Timestamp("2004-01-01")
                 - pd.Timestamp("1970-01-01")).total_seconds()
        self.assertEqual(seconds[0], start)
        np.testing.assert_array_equal(np.diff(seconds), 86400.0)
        # Slot 59 is 1 March, whose true timestamp is 60 days after 1 January;
        # on the model clock it sits at 59 days, one day earlier. That offset
        # IS the correction -- the model calls its own day 59 "1 March" too.
        self.assertEqual(seconds[59], start + 59 * 86400.0)

    def test_span_stitches_years_without_a_step(self):
        # 365 days of 2004 (leap day dropped) + 5 days of 2005.
        fields, seconds = load_era5_span(self.coords, 2004, 370,
                                         cadence_hours=24, url=self.url)
        self.assertEqual(np.asarray(fields["temperature"]).shape[0], 370)
        np.testing.assert_array_equal(np.diff(seconds), 86400.0)
        # Model day 365 must be 1 January of the following year: that is the
        # whole point of a 365-day year, and where a real Gregorian axis would
        # instead still be sitting on 31 December 2004.
        self.assertEqual(self._doy_of(fields, 365), 1.0)
        # Dropping 29 February leaves days-of-year 1-59 then 61-366, so the
        # last model day of a leap year is still 31 December (doy 366): the
        # year stays whole, it just contains 365 slots.
        self.assertEqual(self._doy_of(fields, 364), 366.0)   # 31 Dec 2004

    def test_humidity_units_survive_the_span_path(self):
        # The kg/kg -> g/kg conversion is the units bug; assert it holds on
        # the multi-year path too, not just the single-year one.
        fields, _ = load_era5_span(self.coords, 2004, 370, cadence_hours=24,
                                   url=self.url)
        np.testing.assert_allclose(
            np.asarray(fields["specific_humidity"]), 5.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
