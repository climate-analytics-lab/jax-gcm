import unittest

import numpy as np

from export_nested_symbolic_features import (
    BASE_FEATURES,
    FEATURE_GROUPS,
    HUMIDITY_STABILITY_FEATURES,
    MOISTURE_WIND_FEATURES,
    _interp_profiles,
)


class NestedSymbolicFeaturesTest(unittest.TestCase):
    def test_groups_are_strictly_nested(self):
        baseline = FEATURE_GROUPS["group_05_baseline"]
        humidity = FEATURE_GROUPS["group_14_humidity_stability"]
        moisture = FEATURE_GROUPS["group_18_moisture_wind"]

        self.assertEqual(baseline, BASE_FEATURES)
        self.assertEqual(humidity, BASE_FEATURES + HUMIDITY_STABILITY_FEATURES)
        self.assertEqual(
            moisture,
            BASE_FEATURES + HUMIDITY_STABILITY_FEATURES + MOISTURE_WIND_FEATURES,
        )
        self.assertEqual((len(baseline), len(humidity), len(moisture)), (5, 14, 18))

    def test_interpolation_uses_sigma_not_array_order(self):
        sigma = np.asarray((0.1, 0.4, 0.7, 1.0))
        profiles = np.asarray((sigma, 2.0 * sigma))
        targets = np.asarray((0.25, 0.85))

        expected = np.asarray(((0.25, 0.85), (0.5, 1.7)))
        np.testing.assert_allclose(_interp_profiles(profiles, sigma, targets), expected)
        np.testing.assert_allclose(
            _interp_profiles(profiles[:, ::-1], sigma[::-1], targets), expected
        )

    def test_interpolation_rejects_duplicate_sigma(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            _interp_profiles(
                np.ones((2, 3)), np.asarray((0.2, 0.2, 0.8)), np.asarray((0.5,))
            )


if __name__ == "__main__":
    unittest.main()
