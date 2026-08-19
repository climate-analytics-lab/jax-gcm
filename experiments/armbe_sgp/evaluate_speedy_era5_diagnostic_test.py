import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from evaluate_speedy_era5_diagnostic import (
    area_weighted_distribution,
    bootstrap_mean_difference,
    default_dates,
    era5_daily_rsut,
    load_rsut_targets,
    stratified_dates,
    synoptic_times,
)


class Era5DiagnosticTest(unittest.TestCase):
    def test_default_dates_are_unique_stratified_windows(self):
        dates = default_dates()

        self.assertEqual(len(dates), 120)
        self.assertEqual(len(set(dates)), 120)
        self.assertEqual(dates[0], pd.Timestamp("2016-01-07"))
        self.assertEqual(dates[-1], pd.Timestamp("2020-12-21"))
        self.assertEqual({date.day for date in dates}, {7, 21})

    def test_ten_year_stratification_has_240_windows(self):
        dates = stratified_dates(2011, 2020)

        self.assertEqual(len(dates), 240)
        self.assertEqual(dates[0], pd.Timestamp("2011-01-07"))
        self.assertEqual(dates[-1], pd.Timestamp("2020-12-21"))

    def test_stratification_rejects_reversed_years(self):
        with self.assertRaisesRegex(ValueError, "end_year"):
            stratified_dates(2020, 2019)

    def test_synoptic_times_cover_one_day(self):
        times = synoptic_times(pd.Timestamp("2020-02-15"))

        self.assertEqual([time.hour for time in times], [0, 6, 12, 18])
        self.assertTrue(all(time.date() == pd.Timestamp("2020-02-15").date() for time in times))

    def test_bootstrap_reports_paired_candidate_minus_baseline(self):
        result = bootstrap_mean_difference(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([2.0, 3.0, 4.0]),
            seed=1,
            draws=1_000,
        )

        self.assertEqual(result["mean_difference"], -1.0)
        self.assertEqual(result["ci_95_low"], -1.0)
        self.assertEqual(result["ci_95_high"], -1.0)

    def test_daily_rsut_is_downward_minus_net_shortwave(self):
        times = synoptic_times(pd.Timestamp("2020-02-15"))
        latitude = np.asarray([-30.0, 30.0])
        longitude = np.asarray([0.0, 180.0])
        shape = (len(times), len(latitude), len(longitude))
        source = xr.Dataset(
            {
                "mean_top_downward_short_wave_radiation_flux": (
                    ("time", "latitude", "longitude"),
                    np.full(shape, 300.0),
                ),
                "mean_top_net_short_wave_radiation_flux": (
                    ("time", "latitude", "longitude"),
                    np.full(shape, 200.0),
                ),
            },
            coords={"time": times, "latitude": latitude, "longitude": longitude},
        )
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(
                longitudes=np.deg2rad(longitude),
                latitudes=np.deg2rad(latitude),
            )
        )

        actual = era5_daily_rsut(source, pd.Timestamp("2020-02-15"), coords)

        np.testing.assert_allclose(actual, 100.0)

    def test_area_weighted_distribution_uses_finite_values(self):
        values = np.asarray([[1.0, 3.0, np.nan]])
        weights = np.asarray([[1.0, 1.0, 10.0]])

        result = area_weighted_distribution(values, weights)

        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["standard_deviation"], 1.0)

    def test_load_rsut_targets_uses_processed_cache(self):
        dates = [pd.Timestamp("2020-01-07"), pd.Timestamp("2020-01-21")]
        longitude = np.asarray([0.0, 180.0])
        latitude = np.asarray([-30.0, 30.0])
        expected = np.arange(8.0).reshape(2, 2, 2)
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(
                longitudes=np.deg2rad(longitude),
                latitudes=np.deg2rad(latitude),
            )
        )
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "rsut.nc"
            xr.Dataset(
                {"rsut_w_m2": (("time", "longitude", "latitude"), expected)},
                coords={"time": dates, "longitude": longitude, "latitude": latitude},
            ).to_netcdf(cache_path)

            actual = load_rsut_targets(dates, coords, "unused", cache_path)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
