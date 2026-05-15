"""Tests for ``jcm.diffusion`` scaling helpers."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.diffusion import DiffusionFilter, level_dependent_scaling, uniform_scaling


# Realistic dimensional Laplacian eigenvalues for a triangular T63 grid in
# JCM's nondimensional units. ``laplacian_eigenvalues`` from the dinosaur
# Grid is O(1e-10) here — small enough that ``|eig|^p`` for p>=4 underflows
# in float32 if computed naively (``1e-10 ** 4 = 1e-40``), which is the
# regression these tests pin down.
_T63_NLAT_MODES = 65
_EIG_T63 = -jnp.linspace(0.0, 1.0248212624113382e-10, _T63_NLAT_MODES) ** 1


class LevelDependentScalingTest(unittest.TestCase):
    """Pin behaviour of :func:`level_dependent_scaling`."""

    def test_no_nan_at_high_orders(self):
        """Regression: order=4 (del⁸) used to NaN due to float32 underflow.

        Original implementation computed ``|eig_max|^p`` directly: for the
        T63 nondimensional eigenvalues (max |eig| ~1e-10) and p=4 this is
        1e-40, which underflows in float32 → ``dt / 0 = inf`` → ``inf · 0 =
        NaN``. The fix normalises ``|eig| / |eig_max|`` *before* raising
        to ``p`` so the intermediate stays in [0, 1]. NaN here means the
        diffusion step silently zeroed out the bottom of the atmosphere
        every timestep — caught only when the model NaN'd within one
        save_interval (run_logs/probe_diffEchams_only_260514_230440.log).
        """
        nlev = 47
        # Profile from the ECHAM lmidatm sudif L47 table: del² at top 4
        # levels, then del⁴, del⁶, del⁸ — exactly what
        # ``DiffusionFilter.echam_t63_l47()`` ships with.
        orders = jnp.asarray([1] * 4 + [2] * 3 + [3] * 2 + [4] * 38, dtype=jnp.int32)
        s = level_dependent_scaling(_EIG_T63, timescale=17.5 * 3600.0, orders_per_level=orders, time_step=720.0)
        self.assertEqual(s.shape, (nlev, 1, _T63_NLAT_MODES))
        self.assertFalse(bool(jnp.isnan(s).any()))

    def test_largest_mode_damping_matches_dt_over_tau(self):
        """At |eig| = |eig_max|, per-step factor must equal ``exp(-dt/τ)``."""
        orders = jnp.asarray([2] * 5, dtype=jnp.int32)
        s = level_dependent_scaling(_EIG_T63, timescale=24 * 3600.0, orders_per_level=orders, time_step=720.0)
        expected = float(np.exp(-720.0 / (24 * 3600.0)))
        for k in range(5):
            self.assertAlmostEqual(float(s[k, 0, -1]), expected, places=6)

    def test_smallest_mode_no_damping(self):
        """At |eig| = 0 the per-step factor must be 1 (no damping)."""
        orders = jnp.asarray([1, 4], dtype=jnp.int32)
        s = level_dependent_scaling(_EIG_T63, timescale=24 * 3600.0, orders_per_level=orders, time_step=720.0)
        for k in range(2):
            self.assertAlmostEqual(float(s[k, 0, 0]), 1.0, places=6)


class UniformScalingTest(unittest.TestCase):

    def test_no_nan_at_order_8(self):
        """Same float32-underflow regression as ``level_dependent_scaling``."""
        s = uniform_scaling(_EIG_T63, timescale=24 * 3600.0, order=4, time_step=720.0)
        self.assertFalse(bool(jnp.isnan(s).any()))

    def test_largest_mode_damping_matches_dt_over_tau(self):
        s = uniform_scaling(_EIG_T63, timescale=24 * 3600.0, order=2, time_step=720.0)
        expected = float(np.exp(-720.0 / (24 * 3600.0)))
        self.assertAlmostEqual(float(s[-1]), expected, places=6)


class EchamL47FactoryTest(unittest.TestCase):

    def test_t63_uses_7h_base_timescale(self):
        d = DiffusionFilter.echam_t63_l47()
        self.assertAlmostEqual(d.vor_q_timescale, 7 * 3600.0, places=3)
        self.assertAlmostEqual(d.div_timescale, 7 * 3600.0 / 5.0, places=3)
        self.assertAlmostEqual(d.temp_timescale, 7 * 3600.0 / 0.4, places=3)
        # ECHAM lmidatm L47 sudif profile
        np.testing.assert_array_equal(np.asarray(d.level_orders_temp), [1] * 4 + [2] * 3 + [3] * 2 + [4] * 38)

    def test_t85_uses_3h_base_timescale(self):
        d = DiffusionFilter.echam_t85_l47()
        self.assertAlmostEqual(d.vor_q_timescale, 3 * 3600.0, places=3)
        np.testing.assert_array_equal(d.level_orders_temp, d.level_orders_div)


if __name__ == "__main__":
    unittest.main()
