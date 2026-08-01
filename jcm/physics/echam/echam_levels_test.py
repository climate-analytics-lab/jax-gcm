"""Tests for the built-in ECHAM/ICON hybrid level tables."""

import unittest

import numpy as np

from jcm.physics.echam.echam_levels import get_echam_levels


class EchamLevelsTest(unittest.TestCase):
    def _check_table(self, nlev, p_top_hpa_max):
        hc = get_echam_levels(nlev)
        a = np.asarray(hc.a_boundaries)
        b = np.asarray(hc.b_boundaries)
        self.assertEqual(a.shape, (nlev + 1,))
        self.assertEqual(b.shape, (nlev + 1,))
        # Half-pressures strictly increase from TOA to a 1013.25 hPa surface.
        ph = a + b * 101325.0
        self.assertTrue(np.all(np.diff(ph) > 0.0))
        self.assertAlmostEqual(float(ph[-1]), 101325.0)
        self.assertLessEqual(float(ph[0]) / 100.0, p_top_hpa_max)
        # b spans [0, 1] exactly at the ends.
        self.assertEqual(float(b[0]), 0.0)
        self.assertEqual(float(b[-1]), 1.0)

    def test_l47_table(self):
        self._check_table(47, p_top_hpa_max=0.01)

    def test_l95_ma_table(self):
        # ECHAM6 middle-atmosphere grid: lid at ~0.01 hPa, first full
        # level below ~0.02 hPa half-pressure.
        self._check_table(95, p_top_hpa_max=1e-6)
        hc = get_echam_levels(95)
        ph1_hpa = float(np.asarray(hc.a_boundaries)[1]) / 100.0
        self.assertLess(ph1_hpa, 0.03)
        self.assertGreater(ph1_hpa, 0.01)

    def test_unsupported_count_raises(self):
        with self.assertRaises(ValueError):
            get_echam_levels(63)


if __name__ == "__main__":
    unittest.main()
