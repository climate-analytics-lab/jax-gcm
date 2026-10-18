"""Tests for the Tetens saturation of Betts-Miller and the JAM modules.

ECHAM physics does not use this module (it takes Sonntag 1990 from
``jcm.physics.thermodynamics``); these tests pin the Tetens formula the
out-of-scope schemes keep, value for value.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection import saturation as sat


class TestSaturationVaporPressure(unittest.TestCase):
    """Tetens ``es(T)`` and its phase selection."""

    def test_equals_es0_at_melting_point(self):
        # tc = 0 at the melting point, so every phase collapses to ES0.
        for phase in ("auto", "water", "ice"):
            es = sat.saturation_vapor_pressure(jnp.array(c.tmelt), phase=phase)
            self.assertAlmostEqual(float(es), sat.ES0, places=4)

    def test_monotonic_increasing_in_temperature(self):
        T = jnp.linspace(200.0, 320.0, 50)
        es = sat.saturation_vapor_pressure(T, phase="water")
        self.assertTrue(bool(jnp.all(jnp.diff(es) > 0.0)))

    def test_auto_selects_water_above_ice_below(self):
        warm = jnp.array(c.tmelt + 10.0)
        cold = jnp.array(c.tmelt - 10.0)
        self.assertAlmostEqual(
            float(sat.saturation_vapor_pressure(warm, phase="auto")),
            float(sat.saturation_vapor_pressure(warm, phase="water")), places=6)
        self.assertAlmostEqual(
            float(sat.saturation_vapor_pressure(cold, phase="auto")),
            float(sat.saturation_vapor_pressure(cold, phase="ice")), places=6)

    def test_ice_coefficients_pin(self):
        # Regression guard for the historical broken ice coefficient
        # (A_ICE = 35.86, the *water* c4, in place of ECHAM's c3ies =
        # 21.875): with the correct ECHAM ice pair,
        # es_ice(253.15 K) ≈ 102.8 Pa (reference tables: ≈ 103.2 Pa). The
        # broken coefficient gave ≈ 33 Pa here (~3× low), so a tight
        # window pins the fix.
        es = sat.saturation_vapor_pressure(jnp.array(253.15), phase="ice")
        self.assertGreater(float(es), 102.0)
        self.assertLess(float(es), 103.0)

    def test_ice_below_water_below_freezing(self):
        # Below freezing, saturation over ice is below that over water.
        cold = jnp.array(c.tmelt - 20.0)
        es_ice = sat.saturation_vapor_pressure(cold, phase="ice")
        es_water = sat.saturation_vapor_pressure(cold, phase="water")
        self.assertLess(float(es_ice), float(es_water))


class TestSaturationSpecificHumidity(unittest.TestCase):
    """``qs(T, p)`` behaviour, clipping, and the mixing-ratio alias."""

    def test_positive_and_increases_with_temperature(self):
        p = jnp.array(8.0e4)
        T = jnp.linspace(220.0, 310.0, 40)
        qs = sat.saturation_specific_humidity(T, p, phase="water")
        self.assertTrue(bool(jnp.all(qs > 0.0)))
        self.assertTrue(bool(jnp.all(jnp.diff(qs) > 0.0)))

    def test_decreases_with_pressure(self):
        T = jnp.array(290.0)
        p = jnp.linspace(3.0e4, 1.0e5, 30)
        qs = sat.saturation_specific_humidity(T, p, phase="water")
        self.assertTrue(bool(jnp.all(jnp.diff(qs) < 0.0)))

    def test_matches_definition(self):
        # qs = eps*es / (p - es*(1-eps)), with es capped below p.
        T, p = jnp.array(295.0), jnp.array(9.0e4)
        es = sat.saturation_vapor_pressure(T, phase="water")
        expected = c.eps * es / (p - es * (1.0 - c.eps))
        qs = sat.saturation_specific_humidity(T, p, phase="water")
        self.assertAlmostEqual(float(qs), float(expected), places=8)

    def test_clip_bounds_result(self):
        T, p = jnp.array(305.0), jnp.array(8.0e4)
        qs = sat.saturation_specific_humidity(T, p, phase="water",
                                              clip=(0.0, 1e-3))
        # Unclipped qs at 305 K / 800 hPa is ~0.03; the clip pins it to the
        # ceiling (float32 rounding allows a sub-epsilon overshoot).
        self.assertLessEqual(float(qs), 1e-3 + 1e-9)

    def test_broadcasts_over_shapes(self):
        # Column temperature (kx,) against a (kx, ncols) pressure broadcasts.
        T = jnp.linspace(240.0, 300.0, 6)[:, None]
        p = jnp.linspace(5.0e4, 1.0e5, 4)[None, :]
        qs = sat.saturation_specific_humidity(T, p, phase="auto")
        self.assertEqual(qs.shape, (6, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(qs))))


class TestTetensValuesPinned(unittest.TestCase):
    """The formula, value for value, so Betts-Miller and JAM stay bit-identical."""

    def test_hand_computed_tetens(self):
        with jax.enable_x64(True):
            for T0, phase, c3, c4 in ((300.0, "water", 17.269, 35.86),
                                      (250.0, "ice", 21.875, 7.66),
                                      (250.0, "auto", 21.875, 7.66),
                                      (c.tmelt, "auto", 17.269, 35.86)):
                es = sat.saturation_vapor_pressure(jnp.asarray(T0), phase=phase)
                want = 610.78 * np.exp(c3 * (T0 - c.tmelt) / (T0 - c4))
                np.testing.assert_allclose(float(es), want, rtol=1e-14)

    def test_specific_humidity_uses_eps(self):
        with jax.enable_x64(True):
            T, p = jnp.asarray(290.0), jnp.asarray(8.5e4)
            es = float(sat.saturation_vapor_pressure(T, phase="water"))
            want = 0.622 * es / (8.5e4 - 0.378 * es)
            np.testing.assert_allclose(
                float(sat.saturation_specific_humidity(T, p, phase="water")),
                want, rtol=1e-13)


class TestDewpointTemperature(unittest.TestCase):
    """``dewpoint_temperature`` is the exact inverse of ``qs(T, p)`` (#1140)."""

    def test_round_trip_water(self):
        with jax.enable_x64(True):
            p = jnp.array(98000.0)
            for T in (280.0, 290.0, 300.0, 310.0):
                q_sat = sat.saturation_specific_humidity(
                    jnp.asarray(T), p, phase="water")
                t_d = sat.dewpoint_temperature(q_sat, p, phase="water")
                np.testing.assert_allclose(float(t_d), T, rtol=1e-10)

    def test_round_trip_ice(self):
        with jax.enable_x64(True):
            p = jnp.array(98000.0)
            for T in (230.0, 250.0, 265.0):
                q_sat = sat.saturation_specific_humidity(
                    jnp.asarray(T), p, phase="ice")
                t_d = sat.dewpoint_temperature(q_sat, p, phase="ice")
                np.testing.assert_allclose(float(t_d), T, rtol=1e-10)

    def test_round_trip_auto_matches_branch_on_both_legs(self):
        """``phase="auto"`` picks water above ``tmelt``, ice below, on BOTH
        the forward (``saturation_specific_humidity``) and inverse
        (``dewpoint_temperature``) legs, so a round trip agrees for a
        column straddling the melting point.
        """
        with jax.enable_x64(True):
            p = jnp.array(95000.0)
            T = jnp.linspace(c.tmelt - 30.0, c.tmelt + 30.0, 13)
            q_sat = sat.saturation_specific_humidity(T, p, phase="auto")
            t_d = sat.dewpoint_temperature(q_sat, p, phase="auto")
            np.testing.assert_allclose(np.asarray(t_d), np.asarray(T), rtol=1e-9)

    def test_subsaturated_air_has_dewpoint_below_air_temperature(self):
        # q below qsat(T) must invert to a dew point colder than T.
        with jax.enable_x64(True):
            T, p = jnp.asarray(295.0), jnp.asarray(98000.0)
            q_sat = sat.saturation_specific_humidity(T, p, phase="water")
            q_sub = 0.7 * q_sat
            t_d = sat.dewpoint_temperature(q_sub, p, phase="water")
            self.assertLess(float(t_d), float(T))

    def test_increases_monotonically_with_humidity(self):
        with jax.enable_x64(True):
            p = jnp.array(98000.0)
            q = jnp.linspace(0.002, 0.02, 20)
            t_d = sat.dewpoint_temperature(q, p, phase="water")
            self.assertTrue(bool(jnp.all(jnp.diff(t_d) > 0.0)))

    def test_gradient_is_finite(self):
        def f(q):
            return jnp.sum(sat.dewpoint_temperature(q, jnp.asarray(98000.0),
                                                     phase="auto"))
        g = jax.grad(f)(jnp.linspace(0.002, 0.02, 10))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


class TestSaturationConstantsOverride(unittest.TestCase):
    """Saturation reads eps/tmelt by attribute access, so overrides are honoured."""

    def test_set_constants_override_changes_qs(self):
        T, p = jnp.array(290.0), jnp.array(9.0e4)
        base = float(sat.saturation_specific_humidity(T, p, phase="water"))
        original_eps = c.eps
        try:
            c.set_constants(eps=original_eps * 0.5)
            scaled = float(sat.saturation_specific_humidity(T, p, phase="water"))
        finally:
            c.set_constants(eps=original_eps)
        # qs scales roughly with eps; halving eps must lower qs.
        self.assertLess(scaled, base)
        # And the override is cleanly reverted.
        self.assertAlmostEqual(
            float(sat.saturation_specific_humidity(T, p, phase="water")),
            base, places=10)


if __name__ == "__main__":
    unittest.main()
