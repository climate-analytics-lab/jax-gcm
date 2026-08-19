import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from evaluate_speedy_era5_smoke import (
    area_weighted_bias,
    area_weighted_rmse,
    calibrated_speedy_parameters,
    era5_2d_on_model_grid,
    scheme_configuration,
)


class Era5SpeedySmokeTest(unittest.TestCase):
    def test_area_weighted_rmse_handles_leading_levels(self):
        prediction = np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]])
        target = np.zeros_like(prediction)
        weights = np.asarray([[1.0, 2.0]])

        expected = np.sqrt((1.0 + 8.0 + 9.0 + 32.0) / 6.0)

        self.assertAlmostEqual(area_weighted_rmse(prediction, target, weights), expected)

    def test_area_weighted_metrics_ignore_nonfinite_pairs(self):
        prediction = np.asarray([[1.0, 3.0, 100.0]])
        target = np.asarray([[0.0, 1.0, np.nan]])
        weights = np.asarray([[1.0, 2.0, 10.0]])

        self.assertAlmostEqual(area_weighted_rmse(prediction, target, weights), np.sqrt(3.0))
        self.assertAlmostEqual(area_weighted_bias(prediction, target, weights), 5.0 / 3.0)

    def test_calibrated_speedy_parameters_match_train_fit(self):
        shortwave = calibrated_speedy_parameters().shortwave_radiation

        self.assertAlmostEqual(float(shortwave.rhcl1), 0.32162740151353536)
        self.assertAlmostEqual(float(shortwave.wpcl), 0.05)
        self.assertAlmostEqual(float(shortwave.clsmax), 0.6399201885756207)
        self.assertAlmostEqual(float(shortwave.clsminl), 0.0)

    def test_nested_scheme_configuration_preserves_selector(self):
        _, selector = scheme_configuration("sr_nested_rh_calibrated")

        self.assertEqual(selector, "sr_nested_rh_calibrated")

    def test_era5_interpolation_wraps_periodic_longitude(self):
        time = pd.Timestamp("2020-01-01")
        source = xr.Dataset(
            {
                "cloud": (
                    ("time", "latitude", "longitude"),
                    np.asarray([[[1.0, 0.0, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]]]),
                )
            },
            coords={
                "time": [time],
                "latitude": [-10.0, 10.0],
                "longitude": [0.0, 90.0, 180.0, 270.0],
            },
        )
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(
                longitudes=np.deg2rad(np.asarray([315.0])),
                latitudes=np.deg2rad(np.asarray([0.0])),
            )
        )

        actual = era5_2d_on_model_grid(source, time, coords, "cloud")

        np.testing.assert_allclose(actual, 0.5)


if __name__ == "__main__":
    unittest.main()
