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

    def test_l40_now_unsupported(self):
        # The old L40 "grid" was the bottom 40 rows of L47 — a 274 hPa model
        # top, not a real grid (issue #680 item 1). It is gone; 40 now raises
        # the unknown-count error like any other unsupported count.
        with self.assertRaises(ValueError) as ctx:
            get_echam_levels(40)
        self.assertIn("47, 95", str(ctx.exception))

    def test_truncated_table_guard_fires(self):
        # A table whose top interface still sits at a high pressure (a bottom
        # slice of a longer vct) is rejected by the construction-time guard.
        import jax.numpy as jnp

        from jcm.physics.echam.echam_levels import _checked_hybrid
        # 3-level toy with a 274 hPa "top" (a[0]=27400 Pa) and b spanning [0,1].
        a = jnp.array([27400.0, 10000.0, 2000.0, 0.0])
        b = jnp.array([0.0, 0.3, 0.6, 1.0])
        with self.assertRaisesRegex(ValueError, "truncated vct"):
            _checked_hybrid(a, b)

    def test_full_depth_table_passes_guard(self):
        import jax.numpy as jnp

        from jcm.physics.echam.echam_levels import _checked_hybrid
        a = jnp.array([0.0, 5000.0, 2000.0, 0.0])
        b = jnp.array([0.0, 0.3, 0.6, 1.0])
        hc = _checked_hybrid(a, b)
        self.assertEqual(np.asarray(hc.a_boundaries).shape, (4,))


if __name__ == "__main__":
    unittest.main()
