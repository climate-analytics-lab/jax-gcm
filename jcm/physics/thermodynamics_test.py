"""Tests for jcm's ECHAM saturation thermodynamics (Sonntag 1990).

The reference is ECHAM itself: ``jcm/data/test/echam_saturation_tables/``
holds ECHAM6.3's ``ua`` and ``uaw`` lookup tables, and their slopes, evaluated
by ECHAM's own compiled ``mo_echam_convect_tables.f90`` (provenance in the
README there). The analytic fit jcm evaluates differs from those spline tables
by their interpolation error, at most 4.0e-12 in ``es`` and 1.6e-9 in
``d ln es/dT`` on the file's temperatures; the float64 tolerances sit 2.5x
above that. In float32 the input temperature itself rounds by up to 1.5e-5 K
(a relative ``es`` change of up to 4e-6 at the 0.27/K slope of 150 K) and the
five-term sum, with terms up to ~40, carries an absolute error of a few 1e-6
in ``ln es``; the float32 tolerances are 2e-5 in ``es`` and 2e-6 in the slope.
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics import thermodynamics as thermo
from jcm.testing import check_gradients

_REFERENCE = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "test",
    "echam_saturation_tables", "echam_saturation_tables.npz")

# float64: 2.5x ECHAM's spline interpolation error on the file's grid.
_F64_RTOL_ES = 1.0e-11
_F64_RTOL_SLOPE = 4.0e-9
# float32: rounding of T and of the five-term sum (module docstring).
_F32_RTOL_ES = 2.0e-5
_F32_RTOL_SLOPE = 2.0e-6


def _reference():
    return dict(np.load(_REFERENCE))


class TestAgainstEchamTables(unittest.TestCase):
    """jcm's Sonntag functions against ECHAM's own evaluated lookup tables."""

    def _compare(self, dtype, rtol_es, rtol_slope):
        ref = _reference()
        t = jnp.asarray(ref["temperature"], dtype=dtype)
        pairs = (
            (thermo.es_ua(t), ref["es_ua"], rtol_es, "es_ua"),
            (thermo.saturation_vapor_pressure(t, phase="auto"), ref["es_ua"],
             rtol_es, "saturation_vapor_pressure(auto)"),
            (thermo.es_water(t), ref["es_uaw"], rtol_es, "es_water"),
            (thermo.saturation_vapor_pressure(t, phase="water"), ref["es_uaw"],
             rtol_es, "saturation_vapor_pressure(water)"),
            (thermo.dlnes_dT_ua(t), ref["dlnes_dT_ua"], rtol_slope,
             "dlnes_dT_ua"),
            (thermo.dlnes_dT_water(t), ref["dlnes_dT_uaw"], rtol_slope,
             "dlnes_dT_water"),
        )
        for got, want, rtol, name in pairs:
            self.assertEqual(got.dtype, dtype, msg=name)
            np.testing.assert_allclose(np.asarray(got, np.float64), want,
                                       rtol=rtol, atol=0.0, err_msg=name)

    def test_float64(self):
        with jax.enable_x64(True):
            self._compare(jnp.float64, _F64_RTOL_ES, _F64_RTOL_SLOPE)

    def test_float32(self):
        with jax.enable_x64(False):
            self._compare(jnp.float32, _F32_RTOL_ES, _F32_RTOL_SLOPE)

    def test_ice_fit_matches_ua_below_melting(self):
        """``es_ice`` is the ``ua`` table where that table holds ice."""
        ref = _reference()
        cold = ref["temperature"] <= ref["echam_tmelt"]
        with jax.enable_x64(True):
            t = jnp.asarray(ref["temperature"][cold])
            np.testing.assert_allclose(np.asarray(thermo.es_ice(t)),
                                       ref["es_ua"][cold],
                                       rtol=_F64_RTOL_ES, atol=0.0)
            np.testing.assert_allclose(np.asarray(thermo.dlnes_dT_ice(t)),
                                       ref["dlnes_dT_ua"][cold],
                                       rtol=_F64_RTOL_SLOPE, atol=0.0)

    def test_reference_covers_the_phase_switch(self):
        """The file pins ``tmelt`` itself (ice) and both sides of it."""
        ref = _reference()
        t = ref["temperature"]
        self.assertIn(273.15, t)
        self.assertEqual(float(ref["echam_tmelt"]), c.tmelt)
        self.assertTrue(np.any((t > 273.10) & (t < 273.15)))
        self.assertTrue(np.any((t > 273.15) & (t < 273.20)))
        self.assertLessEqual(t.min(), 150.1)
        self.assertGreaterEqual(t.max(), 330.0)


class TestPhaseRule(unittest.TestCase):
    """ECHAM's ``ua`` table: ice at and below tmelt, water above."""

    def test_ua_ice_phase_is_inclusive_at_tmelt(self):
        t = jnp.array([c.tmelt - 1e-3, c.tmelt, c.tmelt + 1e-3])
        np.testing.assert_array_equal(np.asarray(thermo.ua_ice_phase(t)),
                                      [True, True, False])

    def test_es_ua_selects_the_fit(self):
        for t0, fit in ((250.0, thermo.es_ice), (c.tmelt, thermo.es_ice),
                        (283.15, thermo.es_water)):
            t = jnp.array(t0)
            self.assertEqual(float(thermo.es_ua(t)), float(fit(t)))

    def test_phases_of_the_public_interface(self):
        t = jnp.linspace(200.0, 320.0, 13)
        for phase, fit in (("water", thermo.es_water), ("ice", thermo.es_ice),
                           ("auto", thermo.es_ua)):
            np.testing.assert_array_equal(
                np.asarray(thermo.saturation_vapor_pressure(t, phase=phase)),
                np.asarray(fit(t)))

    def test_invalid_phase_raises(self):
        with self.assertRaises(ValueError):
            thermo.saturation_vapor_pressure(jnp.array(280.0), phase="mixed")
        with self.assertRaises(ValueError):
            thermo.saturation_specific_humidity_and_derivative(
                jnp.array(280.0), jnp.array(9e4), phase="mixed")

    def test_jump_at_tmelt(self):
        """The fits meet at the triple point, so switching at tmelt steps.

        ``es`` steps by -9.7e-5 of its value (0.059 Pa) and ``d ln es/dT``
        by +13 %: the numbers the module docstring states, and the reason
        the switch needs no surrogate gradient.
        """
        with jax.enable_x64(True):
            t = jnp.asarray(c.tmelt)
            jump = float(thermo.es_ice(t) / thermo.es_water(t) - 1.0)
            slope = float(thermo.dlnes_dT_ice(t) / thermo.dlnes_dT_water(t)
                          - 1.0)
            self.assertAlmostEqual(jump, -9.70e-5, delta=0.02e-5)
            self.assertAlmostEqual(slope, 0.1333, delta=0.0005)
            # The curves cross at the triple point, 273.16 K.
            tp = jnp.asarray(273.16)
            self.assertLess(abs(float(thermo.es_ice(tp) / thermo.es_water(tp))
                                - 1.0), 2e-6)
            self.assertAlmostEqual(float(thermo.es_water(tp)), 611.657,
                                   delta=0.002)

    def test_gradient_on_each_side_is_the_fit_slope(self):
        """AD at the switch returns the selected fit's own slope."""
        with jax.enable_x64(True):
            for t0, dln in ((c.tmelt, thermo.dlnes_dT_ice),
                            (c.tmelt + 1e-6, thermo.dlnes_dT_water)):
                t = jnp.asarray(t0)
                g = jax.grad(thermo.es_ua)(t)
                np.testing.assert_allclose(
                    float(g), float(thermo.es_ua(t) * dln(t)), rtol=1e-12)

    def test_extreme_temperatures_finite(self):
        # The clip to ECHAM's table range keeps the fit finite for garbage.
        T = jnp.array([0.0, 10.0, 50.0, 400.0, 1000.0])
        for phase in ("auto", "water", "ice"):
            es = thermo.saturation_vapor_pressure(T, phase=phase)
            self.assertTrue(bool(jnp.all(jnp.isfinite(es))))
            self.assertTrue(bool(jnp.all(es >= 0.0)))


class TestSaturationSpecificHumidity(unittest.TestCase):
    """ECHAM's ``x = MIN(es·rd/rv/p, 0.5)``, ``qs = x/(1 − vtmpc1·x)``."""

    def test_echam_form(self):
        with jax.enable_x64(True):
            T, p = jnp.asarray(295.0), jnp.asarray(9.0e4)
            es = thermo.saturation_vapor_pressure(T, phase="water")
            x = es * c.rd / c.rv / p
            np.testing.assert_allclose(
                float(thermo.saturation_specific_humidity(T, p, phase="water")),
                float(x / (1.0 - c.vtmpc1 * x)), rtol=1e-14)

    def test_equals_mixing_ratio_form_with_rd_over_rv(self):
        """Below the cap it is ``eps·es/(p − (1 − eps)·es)``, eps = rd/rv."""
        with jax.enable_x64(True):
            es, p = jnp.asarray(2500.0), jnp.asarray(8.0e4)
            eps = c.rd / c.rv
            np.testing.assert_allclose(
                float(thermo.qsat_from_es(es, p)),
                float(eps * es / (p - (1.0 - eps) * es)), rtol=1e-13)

    def test_value_at_300K_1000hPa(self):
        qs = thermo.saturation_specific_humidity(
            jnp.array(300.0), jnp.array(1.0e5))
        self.assertTrue(np.isclose(float(qs), 0.0222, rtol=0.01),
                        msg=f"qs = {float(qs)}")

    def test_capped(self):
        # es·rd/rv/p capped at 0.5 (hot, near-vacuum): qs = 0.5/(1 − vtmpc1/2).
        qs = thermo.saturation_specific_humidity(
            jnp.array(390.0), jnp.array(5.0e3), phase="water")
        self.assertAlmostEqual(float(qs), 0.5 / (1.0 - 0.5 * c.vtmpc1),
                               places=6)

    def test_reads_constants_at_call_time(self):
        T, p = jnp.array(290.0), jnp.array(9.0e4)
        base = float(thermo.saturation_specific_humidity(T, p))
        original_rv = c.rv
        try:
            c.set_constants(rv=461.51)            # ECHAM's value
            echam = float(thermo.saturation_specific_humidity(T, p))
        finally:
            c.set_constants(rv=original_rv)
        self.assertLess(echam, base)              # larger rv, smaller rd/rv
        self.assertAlmostEqual(
            float(thermo.saturation_specific_humidity(T, p)), base, places=10)

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
                np.testing.assert_allclose(np.asarray(qs_blk[:, j]),
                                           np.asarray(qs_col), rtol=1e-6)


class TestDerivative(unittest.TestCase):
    """ECHAM's ``dqs/dT`` against the derivative of ``qs(T)``."""

    def test_derivative_matches_finite_difference(self):
        # float64 so the central difference is accurate to ~1e-8; no point
        # within h of the ``ua`` switch at tmelt, where the step is real.
        with jax.enable_x64(True):
            p = jnp.asarray(9.0e4)
            h = 1e-3
            for phase in ("water", "ice", "auto"):
                for T0 in (180.0, 230.0, 250.0, 270.0, 272.5, 274.0, 285.0,
                           300.0):
                    T = jnp.asarray(T0)
                    _, dqs = thermo.saturation_specific_humidity_and_derivative(
                        T, p, phase=phase)
                    qp = thermo.saturation_specific_humidity(T + h, p,
                                                             phase=phase)
                    qm = thermo.saturation_specific_humidity(T - h, p,
                                                             phase=phase)
                    fd = (float(qp) - float(qm)) / (2.0 * h)
                    self.assertTrue(
                        np.isclose(float(dqs), fd, rtol=1e-7),
                        msg=f"phase={phase} T={T0}: {float(dqs)} vs FD {fd}")

    def test_derivative_matches_autodiff(self):
        with jax.enable_x64(True):
            T = jnp.linspace(200.0, 310.0, 23)
            p = jnp.linspace(2.0e4, 1.0e5, 23)
            for phase in ("water", "ice", "auto"):
                _, dqs = thermo.saturation_specific_humidity_and_derivative(
                    T, p, phase=phase)
                ad = jax.vmap(jax.grad(
                    lambda t, pp: thermo.saturation_specific_humidity(
                        t, pp, phase=phase)))(T, p)
                np.testing.assert_allclose(np.asarray(dqs), np.asarray(ad),
                                           rtol=1e-12)

    def test_ub_branch_at_high_es_over_p(self):
        """Where es·rd/rv/p >= 0.4 ECHAM uses ``qs·zcor·d ln es/dT`` (``ub``).

        Below the cap that is still the analytic slope; above it the slope
        carries the capped ``x``, as ECHAM's does, not zero.
        """
        with jax.enable_x64(True):
            T = jnp.asarray(350.0)
            es = thermo.es_water(T)
            des = es * thermo.dlnes_dT_water(T)
            for x_target in (0.45, 0.8):
                p = es * c.rd / c.rv / x_target
                x = min(x_target, 0.5)
                zcor = 1.0 / (1.0 - c.vtmpc1 * x)
                want = x * zcor * zcor * float(thermo.dlnes_dT_water(T))
                got = float(thermo.dqsat_dT_from_es(es, des, p))
                np.testing.assert_allclose(got, want, rtol=1e-13)
            # Continuous across the 0.4 regime boundary.
            p04 = es * c.rd / c.rv / 0.4
            lo = float(thermo.dqsat_dT_from_es(es, des, p04 * (1 + 1e-9)))
            hi = float(thermo.dqsat_dT_from_es(es, des, p04 * (1 - 1e-9)))
            np.testing.assert_allclose(lo, hi, rtol=1e-7)

    def test_gradients_finite_at_zero_pressure(self):
        """Zero pressure (the model-top interface) is capped and differentiable."""
        T, p = jnp.array([200.0, 300.0]), jnp.array([0.0, 0.0])
        for phase in ("water", "ice", "auto"):
            for i in (0, 1):
                g = jax.grad(
                    lambda t, pp: jnp.sum(
                        thermo.saturation_specific_humidity_and_derivative(
                            t, pp, phase=phase)[i]),
                    argnums=(0, 1))(T, p)
                for x in g:
                    self.assertTrue(bool(jnp.all(jnp.isfinite(x))))
        self.assertAlmostEqual(
            float(thermo.saturation_specific_humidity(T[1], p[1])),
            0.5 / (1.0 - 0.5 * c.vtmpc1), places=6)

    def test_qs_matches_plain_call(self):
        T, p = jnp.array(265.0), jnp.array(7.0e4)
        qs, _ = thermo.saturation_specific_humidity_and_derivative(T, p)
        qs_plain = thermo.saturation_specific_humidity(T, p)
        self.assertEqual(float(qs), float(qs_plain))


class TestGradients(unittest.TestCase):
    """All functions must be differentiable with finite gradients."""

    def test_grad_finiteness(self):
        p = jnp.array(8.0e4)
        for phase in ("auto", "water", "ice"):
            for T0 in (52.0, 180.0, 230.0, 273.15, 300.0):
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
