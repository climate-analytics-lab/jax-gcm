"""Tests for the shared Tetens saturation thermodynamics.

These cover the formulas that ``tiedtke_nordeng`` and ``betts_miller`` both
depend on: the saturation vapour pressure, specific humidity / mixing ratio, and
the analytic ``dqs/dT`` used by the Newton adjustment steps.
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

    def test_mixing_ratio_is_clipped_alias(self):
        # saturation_mixing_ratio swaps the arg order and clips to [0, 0.5].
        T, p = jnp.array(295.0), jnp.array(9.0e4)
        r = sat.saturation_mixing_ratio(p, T, phase="auto")
        qs = sat.saturation_specific_humidity(T, p, phase="auto",
                                              clip=(0.0, 0.5))
        self.assertAlmostEqual(float(r), float(qs), places=10)

    def test_broadcasts_over_shapes(self):
        # Column temperature (kx,) against a (kx, ncols) pressure broadcasts.
        T = jnp.linspace(240.0, 300.0, 6)[:, None]
        p = jnp.linspace(5.0e4, 1.0e5, 4)[None, :]
        qs = sat.saturation_specific_humidity(T, p, phase="auto")
        self.assertEqual(qs.shape, (6, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(qs))))


class TestSaturationDerivative(unittest.TestCase):
    """The analytic ``dqs/dT`` used by the Newton adjustment steps."""

    def test_qs_matches_plain_call(self):
        T, p = jnp.array(285.0), jnp.array(8.5e4)
        qs, _ = sat.saturation_specific_humidity_and_derivative(
            T, p, phase="auto")
        qs_plain = sat.saturation_specific_humidity(T, p, phase="auto")
        self.assertAlmostEqual(float(qs), float(qs_plain), places=8)

    def test_derivative_matches_autodiff(self):
        # The closed-form dqs/dT should match JAX's autodiff of qs(T) (away from
        # the es<0.99p cap, where the two branches diverge by construction).
        for T0 in (250.0, 285.0, 305.0):
            T, p = jnp.array(T0), jnp.array(9.0e4)
            _, dqs_dT = sat.saturation_specific_humidity_and_derivative(
                T, p, phase="auto")
            ad = jax.grad(
                lambda t: sat.saturation_specific_humidity(t, p, phase="auto"))(T)
            self.assertTrue(np.isclose(float(dqs_dT), float(ad), rtol=1e-3),
                            msg=f"T={T0}: {float(dqs_dT)} vs {float(ad)}")


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
