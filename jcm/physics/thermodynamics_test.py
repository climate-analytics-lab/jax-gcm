"""Tests for the shared saturation thermodynamics module.

Value pins are against standard references (Murphy & Koop 2005 for ice,
standard meteorological tables for water) with tolerances that reflect the
inherent accuracy of the ECHAM Tetens/Magnus fits (a few tenths of a
percent), plus finite-difference verification of the analytic derivative
and gradient-finiteness checks per the repo testing conventions.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics import thermodynamics as thermo
from jcm.testing import check_gradients


class TestSaturationVaporPressure(unittest.TestCase):
    """es(T) values, phase selection, and reference pins."""

    def test_melting_point_anchor_both_phases(self):
        # es(tmelt) = c1es for water, ice and auto alike.
        for phase in ("auto", "water", "ice"):
            es = thermo.saturation_vapor_pressure(
                jnp.array(273.15), phase=phase)
            self.assertAlmostEqual(float(es), 610.78, places=2)

    def test_water_at_300K(self):
        # Reference (Guide to Meteorological Instruments / Magnus fits):
        # es_water(300 K) ≈ 3537 Pa.
        es = thermo.saturation_vapor_pressure(jnp.array(300.0), phase="water")
        self.assertGreater(float(es), 3530.0)
        self.assertLess(float(es), 3570.0)

    def test_ice_at_250K_murphy_koop(self):
        # Murphy & Koop (2005): es_ice(250 K) ≈ 76.0 Pa. The ECHAM Magnus
        # fit gives ≈ 75.6 Pa (0.6 % low) — pin within 1 %. This is the
        # regression guard for the broken historical ice coefficient
        # (A_ICE = 35.86), which gave ≈ 6 Pa here (~12× low).
        es = thermo.saturation_vapor_pressure(jnp.array(250.0), phase="ice")
        self.assertTrue(np.isclose(float(es), 76.0, rtol=0.01),
                        msg=f"es_ice(250K) = {float(es)}")

    def test_water_ice_ratio_at_250K(self):
        # The supersaturation-over-ice ratio drives the Bergeron process;
        # at 250 K es_water/es_ice ≈ 1.25.
        es_w = thermo.saturation_vapor_pressure(jnp.array(250.0), phase="water")
        es_i = thermo.saturation_vapor_pressure(jnp.array(250.0), phase="ice")
        self.assertTrue(np.isclose(float(es_w) / float(es_i), 1.25, rtol=0.01),
                        msg=f"ratio = {float(es_w) / float(es_i)}")

    def test_auto_switches_at_tmelt(self):
        warm, cold = jnp.array(283.15), jnp.array(263.15)
        self.assertAlmostEqual(
            float(thermo.saturation_vapor_pressure(warm, phase="auto")),
            float(thermo.saturation_vapor_pressure(warm, phase="water")),
            places=6)
        self.assertAlmostEqual(
            float(thermo.saturation_vapor_pressure(cold, phase="auto")),
            float(thermo.saturation_vapor_pressure(cold, phase="ice")),
            places=6)

    def test_invalid_phase_raises(self):
        with self.assertRaises(ValueError):
            thermo.saturation_vapor_pressure(jnp.array(280.0), phase="mixed")

    def test_extreme_temperatures_finite(self):
        # The [50, 500] K clip keeps the math finite for garbage inputs.
        T = jnp.array([0.0, 10.0, 50.0, 500.0, 1000.0])
        for phase in ("auto", "water", "ice"):
            es = thermo.saturation_vapor_pressure(T, phase=phase)
            self.assertTrue(bool(jnp.all(jnp.isfinite(es))))


class TestSaturationSpecificHumidity(unittest.TestCase):
    """qs(T, p) values, guards, and the 0.5 cap."""

    def test_value_at_300K_1000hPa(self):
        qs = thermo.saturation_specific_humidity(
            jnp.array(300.0), jnp.array(1.0e5))
        self.assertTrue(np.isclose(float(qs), 0.0223, rtol=0.03),
                        msg=f"qs = {float(qs)}")

    def test_matches_definition(self):
        T, p = jnp.array(295.0), jnp.array(9.0e4)
        es = thermo.saturation_vapor_pressure(T, phase="water")
        expected = c.eps * es / (p - (1.0 - c.eps) * es)
        qs = thermo.saturation_specific_humidity(T, p, phase="water")
        self.assertAlmostEqual(float(qs), float(expected), places=8)

    def test_capped_at_half(self):
        # Very hot / very low pressure would give qs >> 1; the ECHAM lookup
        # guard caps it at 0.5.
        qs = thermo.saturation_specific_humidity(
            jnp.array(400.0), jnp.array(5.0e3), phase="water")
        self.assertAlmostEqual(float(qs), 0.5, places=6)

    def test_broadcasting_column_vs_block(self):
        # Broadcasting-native per CLAUDE.md: a (kx,) column and a
        # (kx, ncols) block must agree per column.
        kx, ncols = 6, 4
        T_col = jnp.linspace(230.0, 300.0, kx)
        p_col = jnp.linspace(3.0e4, 1.0e5, kx)
        T_blk = jnp.tile(T_col[:, None], (1, ncols))
        p_blk = jnp.tile(p_col[:, None], (1, ncols))
        for phase in ("auto", "water", "ice"):
            qs_col = thermo.saturation_specific_humidity(T_col, p_col,
                                                         phase=phase)
            qs_blk = thermo.saturation_specific_humidity(T_blk, p_blk,
                                                         phase=phase)
            self.assertEqual(qs_blk.shape, (kx, ncols))
            for j in range(ncols):
                # Same math, but XLA may fuse the block differently than the
                # column, so allow float32 ULP-level differences.
                np.testing.assert_allclose(np.asarray(qs_blk[:, j]),
                                           np.asarray(qs_col), rtol=1e-6)


class TestDerivative(unittest.TestCase):
    """Analytic dqs/dT against central finite differences."""

    def test_derivative_matches_finite_difference(self):
        # Verified in float64 so the FD reference is accurate to ~1e-8;
        # temperatures straddle the auto water/ice switch at tmelt (none
        # within h of the switch itself, where a one-sided kink is real).
        with jax.enable_x64():
            p = jnp.array(9.0e4, dtype=jnp.float64)
            h = 1e-3
            for phase in ("water", "ice", "auto"):
                for T0 in (230.0, 250.0, 270.0, 272.5, 274.0, 285.0, 300.0):
                    T = jnp.array(T0, dtype=jnp.float64)
                    _, dqs = thermo.saturation_specific_humidity_and_derivative(
                        T, p, phase=phase)
                    qp = thermo.saturation_specific_humidity(T + h, p, phase=phase)
                    qm = thermo.saturation_specific_humidity(T - h, p, phase=phase)
                    fd = (float(qp) - float(qm)) / (2.0 * h)
                    self.assertTrue(
                        np.isclose(float(dqs), fd, rtol=1e-4),
                        msg=f"phase={phase} T={T0}: analytic {float(dqs)} vs FD {fd}")

    def test_qs_matches_plain_call(self):
        T, p = jnp.array(265.0), jnp.array(7.0e4)
        qs, _ = thermo.saturation_specific_humidity_and_derivative(T, p)
        qs_plain = thermo.saturation_specific_humidity(T, p)
        self.assertAlmostEqual(float(qs), float(qs_plain), places=10)


class TestGradients(unittest.TestCase):
    """All functions must be differentiable with finite gradients."""

    def test_grad_finiteness(self):
        p = jnp.array(8.0e4)
        for phase in ("auto", "water", "ice"):
            for T0 in (230.0, 273.15, 300.0):
                T = jnp.array(T0)
                g_es = jax.grad(
                    lambda t: thermo.saturation_vapor_pressure(t, phase=phase))(T)
                g_qs = jax.grad(
                    lambda t: thermo.saturation_specific_humidity(
                        t, p, phase=phase))(T)
                g_dq = jax.grad(
                    lambda t: thermo.saturation_specific_humidity_and_derivative(
                        t, p, phase=phase)[1])(T)
                for g in (g_es, g_qs, g_dq):
                    self.assertTrue(bool(jnp.isfinite(g)),
                                    msg=f"phase={phase} T={T0}: {g}")
        g_w = jax.grad(thermo.mixed_phase_weight)(jnp.array(255.0))
        self.assertTrue(bool(jnp.isfinite(g_w)))
        g_ic = jax.grad(
            lambda x: thermo.grid_mean_to_in_cloud(x, jnp.array(0.0)))(
            jnp.array(1e-4))
        self.assertTrue(bool(jnp.isfinite(g_ic)))


class TestMixedPhaseWeight(unittest.TestCase):
    """Linear liquid-fraction ramp."""

    def test_endpoints_and_midpoint(self):
        self.assertEqual(float(thermo.mixed_phase_weight(jnp.array(238.15))), 0.0)
        self.assertEqual(float(thermo.mixed_phase_weight(jnp.array(220.0))), 0.0)
        self.assertEqual(float(thermo.mixed_phase_weight(jnp.array(273.15))), 1.0)
        self.assertEqual(float(thermo.mixed_phase_weight(jnp.array(290.0))), 1.0)
        mid = 0.5 * (238.15 + 273.15)
        self.assertAlmostEqual(
            float(thermo.mixed_phase_weight(jnp.array(mid))), 0.5, places=5)

    def test_custom_bounds(self):
        w = thermo.mixed_phase_weight(jnp.array(250.0), t_min=240.0, t_max=260.0)
        self.assertAlmostEqual(float(w), 0.5, places=6)


class TestGridMeanToInCloud(unittest.TestCase):
    """Grid-mean → in-cloud conversion and its masked-region behaviour."""

    def test_divides_where_cloudy_zero_elsewhere(self):
        x = jnp.array([1e-4, 1e-4, 1e-4])
        cf = jnp.array([0.5, 1.0, 0.0])
        out = thermo.grid_mean_to_in_cloud(x, cf)
        np.testing.assert_allclose(np.asarray(out), [2e-4, 1e-4, 0.0])


if __name__ == "__main__":
    unittest.main()


class TestThermodynamicsGradients(unittest.TestCase):
    """AD against a central difference (#820).

    These are regression fences on functions every ECHAM scheme calls: all
    four are green, and pinning them means a later guard added upstream
    cannot silently break them.

    The operating points sit strictly inside the mixed-phase ramp rather than
    on ``t_min = 238.15`` or on ``c.tmelt``. ``mixed_phase_weight`` is a
    ``clip``, so its derivative at either end of the ramp is one-sided; a
    fixture placed exactly there would be testing the kink, which is a
    property of the point and not a defect in the formula.
    """

    def _profile(self):
        """Return (temperature, pressure) spanning the mixed-phase range."""
        return (jnp.linspace(235.0, 300.0, 12),
                jnp.linspace(2.0e4, 1.0e5, 12))

    def test_saturation_specific_humidity_column_and_block(self):
        """Broadcasting-native: a column and a 3-column block both check out.

        ``rtol=1e-2`` on the block. The consistency search settles on a coarse
        rung there (5e-4) because the block's three columns project with
        opposite signs and cancel, and at that rung the secant's own
        truncation leaves it ~0.6% from the AD value; the single column
        reaches 1e-3 at a finer rung.
        """
        temperature, pressure = self._profile()
        check_gradients(thermo.saturation_specific_humidity,
                        (temperature, pressure), rtol=1e-3)

        stack = lambda a, s: jnp.stack(  # noqa: E731
            [a * (1.0 + s * k) for k in range(3)], axis=1)
        check_gradients(
            thermo.saturation_specific_humidity,
            (stack(temperature, 0.01), stack(pressure, 0.0)), rtol=1e-2)

    def test_saturation_specific_humidity_and_derivative(self):
        """The paired value/derivative form agrees with a secant too."""
        temperature, pressure = self._profile()
        check_gradients(thermo.saturation_specific_humidity_and_derivative,
                        (temperature, pressure), rtol=1e-3)

    def test_mixed_phase_weight_inside_the_ramp(self):
        """Strictly between t_min and tmelt, where the clip is inactive."""
        check_gradients(thermo.mixed_phase_weight,
                        (jnp.linspace(241.0, 270.0, 12),), rtol=1e-3)

    def test_grid_mean_to_in_cloud(self):
        """Cloud fractions well above the eps guard."""
        check_gradients(
            thermo.grid_mean_to_in_cloud,
            (jnp.full(8, 1.0e-4), jnp.linspace(0.12, 0.9, 8)), rtol=1e-3)


class TestMoistIsobaricHeatCapacity:
    """ECHAM ``zcpq = cpd·(1 + vtmpc2·MAX(pqm1, 0))`` (mo_cumastr.f90:229)."""

    def test_dry_air_is_cpd(self):
        cp = thermo.moist_isobaric_heat_capacity(jnp.asarray(0.0))
        np.testing.assert_allclose(float(cp), c.cpd, rtol=1e-7)

    def test_moist_extremes_hand_computed(self):
        # vtmpc2 = cpv/cpd − 1, so cp = cpd + (cpv − cpd)·q exactly.
        for q in (0.018, 0.035):
            expected = c.cpd + (c.cpv - c.cpd) * q
            cp = thermo.moist_isobaric_heat_capacity(jnp.asarray(q))
            np.testing.assert_allclose(float(cp), expected, rtol=1e-6)
        # The shift the dry-cpd form missed: ~1.5 % at 18 g/kg, ~3 % at 35.
        ratio_18 = float(thermo.moist_isobaric_heat_capacity(
            jnp.asarray(0.018))) / c.cpd
        ratio_35 = float(thermo.moist_isobaric_heat_capacity(
            jnp.asarray(0.035))) / c.cpd
        assert 1.014 < ratio_18 < 1.017
        assert 1.028 < ratio_35 < 1.032

    def test_negative_humidity_clamped(self):
        cp = thermo.moist_isobaric_heat_capacity(jnp.asarray(-1e-3))
        np.testing.assert_allclose(float(cp), c.cpd, rtol=1e-7)

    def test_broadcasting_native(self):
        q = jnp.linspace(0.0, 0.02, 12).reshape(3, 4)
        cp = thermo.moist_isobaric_heat_capacity(q)
        assert cp.shape == q.shape
        np.testing.assert_allclose(
            np.asarray(cp), c.cpd * (1.0 + c.vtmpc2 * np.asarray(q)),
            rtol=1e-6)

    def test_gradient_finite_at_clamp(self):
        g = jax.grad(lambda q: thermo.moist_isobaric_heat_capacity(q))
        assert np.isfinite(float(g(jnp.asarray(0.0))))
        np.testing.assert_allclose(
            float(g(jnp.asarray(0.01))), c.cpd * c.vtmpc2, rtol=1e-6)
