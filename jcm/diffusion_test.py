"""Tests for ``jcm.diffusion`` scaling helpers."""

import unittest

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jcm.diffusion import (
    DiffusionFilter,
    echam_dampth_hours,
    echam_lmidatm_orders,
    level_dependent_scaling,
    uniform_scaling,
)


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


class SpmdPaddedEigenvaluesTest(unittest.TestCase):
    """Modal-axis zero padding under SPMD must not break the normalisation.

    ``FastSphericalHarmonics`` (used whenever an ``spmd_mesh`` is set) pads the
    modal axis with zeros so it divides evenly across devices, and those zeros
    land on the last index. The scaling helpers normalise by ``max(|eig|)``
    rather than ``|eig[-1]|`` precisely so this padding cannot zero the
    normaliser (``|eig[-1]| = 0`` → ``dt/0 = inf`` → NaN, which NaN'd every
    diffused field in multi-device runs).
    """

    _N_PAD = 7

    def _padded(self):
        return jnp.concatenate([_EIG_T63, jnp.zeros(self._N_PAD)])

    def test_uniform_scaling_ignores_trailing_padding(self):
        s_pad = uniform_scaling(self._padded(), timescale=24 * 3600.0, order=4, time_step=720.0)
        s_ref = uniform_scaling(_EIG_T63, timescale=24 * 3600.0, order=4, time_step=720.0)
        self.assertFalse(bool(jnp.isnan(s_pad).any()))
        # Real modes unchanged; padding modes are undamped (factor 1).
        npt.assert_allclose(np.asarray(s_pad[:_T63_NLAT_MODES]), np.asarray(s_ref), rtol=1e-6)
        npt.assert_allclose(np.asarray(s_pad[_T63_NLAT_MODES:]), 1.0, rtol=1e-6)

    def test_level_dependent_scaling_ignores_trailing_padding(self):
        orders = jnp.asarray([1, 2, 4], dtype=jnp.int32)
        s_pad = level_dependent_scaling(self._padded(), 24 * 3600.0, orders, 720.0)
        s_ref = level_dependent_scaling(_EIG_T63, 24 * 3600.0, orders, 720.0)
        self.assertFalse(bool(jnp.isnan(s_pad).any()))
        npt.assert_allclose(
            np.asarray(s_pad[:, :, :_T63_NLAT_MODES]), np.asarray(s_ref), rtol=1e-6,
        )


class EchamReferenceTableTest(unittest.TestCase):
    """Pin the ported ECHAM6.3 ``lmidatm`` tables against the Fortran source.

    These are transcriptions, so the test is a transcription check: every
    ``(nn, nlev)`` pair ``mo_hdiff.f90::sudif`` defines must come back
    verbatim, and ``setdyn.f90``'s ``dampth`` values must be exact.
    """

    # mo_hdiff.f90::sudif, lmidatm branch. Top-first, 1 = del², 4 = del⁸.
    _SUDIF = {
        (31, 47): [1] * 4 + [2] * 3 + [3] * 2 + [4] * 2 + [5] * 36,
        (63, 47): [1] * 4 + [2] * 3 + [3] * 2 + [4] * 38,
        (63, 95): [1] * 10 + [2] * 10 + [3] * 5 + [4] * 70,
        (127, 95): [1] * 10 + [2] * 15 + [3] * 70,
        (255, 95): [1] * 10 + [2] * 15 + [3] * 70,
    }

    def test_order_tables_match_sudif(self):
        for (nn, nlev), expected in self._SUDIF.items():
            with self.subTest(truncation=nn, layers=nlev):
                np.testing.assert_array_equal(
                    np.asarray(echam_lmidatm_orders(nn, nlev)), expected,
                )

    def test_dampth_matches_setdyn(self):
        # setdyn.f90 section 1.5: dampth by truncation, lmidatm.
        for nn, hours in ((31, 12.0), (63, 7.0), (127, 1.5), (255, 0.5)):
            with self.subTest(truncation=nn):
                self.assertAlmostEqual(echam_dampth_hours(nn), hours, places=9)

    def test_grid_levels_sit_at_the_pressures_sudif_annotates(self):
        """The index tables are transferable only because our grids ARE ECHAM's.

        ``sudif`` places the order transitions by level *index* and annotates
        each with a pressure. Porting the indices verbatim is therefore only
        correct while jcm's hybrid tables put those indices at the same
        pressures — so assert it, and fail loudly if the level definitions
        ever drift (#579).
        """
        from jcm.physics.echam.echam_levels import get_echam_levels

        # (layers, 1-based ECHAM level, hPa quoted in the sudif comment)
        annotated = [
            (47, 4, 0.23), (47, 7, 1.22), (47, 9, 2.96),
            (95, 10, 0.15), (95, 20, 0.77), (95, 25, 1.50),
        ]
        for layers, jk, p_ref in annotated:
            with self.subTest(layers=layers, level=jk):
                h = get_echam_levels(layers)
                p_half = (np.asarray(h.a_boundaries)
                          + np.asarray(h.b_boundaries) * 101325.0)
                p_full_hpa = 0.5 * (p_half[1:] + p_half[:-1])[jk - 1] / 100.0
                # sudif quotes two decimals, so match to its rounding.
                self.assertAlmostEqual(p_full_hpa, p_ref, places=2)


class EchamLmidatmFactoryTest(unittest.TestCase):

    def test_t63_l47_is_the_exact_reference_configuration(self):
        """T63L47 is tabulated on both axes — the fidelity anchor."""
        d = DiffusionFilter.echam_t63_l47()
        self.assertAlmostEqual(d.vor_q_timescale, 7 * 3600.0, places=3)
        # mo_hdiff.f90: difd = 5*difvo, dift = 0.4*difvo.
        self.assertAlmostEqual(d.div_timescale, 7 * 3600.0 / 5.0, places=3)
        self.assertAlmostEqual(d.temp_timescale, 7 * 3600.0 / 0.4, places=3)
        np.testing.assert_array_equal(
            np.asarray(d.level_orders_temp), [1] * 4 + [2] * 3 + [3] * 2 + [4] * 38,
        )

    def test_profile_length_matches_layers(self):
        for truncation, layers in ((63, 47), (85, 47), (106, 95), (119, 95)):
            with self.subTest(truncation=truncation, layers=layers):
                d = DiffusionFilter.echam_lmidatm(truncation, layers)
                for orders in (d.level_orders_div, d.level_orders_vor_q,
                               d.level_orders_temp):
                    self.assertEqual(len(orders), layers)

    def test_untabulated_truncation_borrows_nearest_in_log_space(self):
        """T85 sits nearer T63 than T127 in log space; T106/T119 nearer T127.

        This matters on L95, where the two reference profiles genuinely
        differ: T63L95 grades down to del⁸, T127L95 stops at del⁶.
        """
        np.testing.assert_array_equal(
            np.asarray(echam_lmidatm_orders(85, 47)),
            np.asarray(echam_lmidatm_orders(63, 47)),
        )
        for truncation in (106, 119):
            with self.subTest(truncation=truncation):
                np.testing.assert_array_equal(
                    np.asarray(echam_lmidatm_orders(truncation, 95)),
                    np.asarray(echam_lmidatm_orders(127, 95)),
                )
                # ...and specifically NOT the del⁸ T63L95 profile.
                self.assertEqual(
                    int(np.max(np.asarray(echam_lmidatm_orders(truncation, 95)))), 3,
                )

    def test_dampth_is_monotone_and_bracketed(self):
        """Derived timescales must decrease with truncation and interpolate."""
        truncations = [31, 63, 85, 106, 119, 127, 255]
        taus = [echam_dampth_hours(nn) for nn in truncations]
        self.assertEqual(taus, sorted(taus, reverse=True))
        # Each derived value lies strictly between its ECHAM neighbours.
        for nn in (85, 106, 119):
            with self.subTest(truncation=nn):
                self.assertLess(echam_dampth_hours(nn), 7.0)
                self.assertGreater(echam_dampth_hours(nn), 1.5)

    def test_unsupported_layer_count_raises_a_named_error(self):
        with self.assertRaisesRegex(ValueError, r"No ECHAM lmidatm .*31 levels"):
            DiffusionFilter.echam_lmidatm(truncation=63, layers=31)

    def test_missing_truncation_raises_before_reaching_math_log(self):
        """A zero/absent ``grid.spectral_truncation`` must name the config key.

        Both selection rules take ``log(truncation)``, so without this guard
        an unset key surfaces as a bare ``math domain error``.
        """
        # Label with a string, not the callable: under pytest-xdist the
        # subTest parameters are serialised across the execnet gateway, which
        # cannot pickle a function ("DumpError: can't serialize <class
        # 'function'>") — the test then fails only in the parallel run.
        entry_points = {
            "echam_dampth_hours": lambda: echam_dampth_hours(0),
            "echam_lmidatm_orders": lambda: echam_lmidatm_orders(0, 47),
            "DiffusionFilter.echam_lmidatm": lambda: DiffusionFilter.echam_lmidatm(0, 47),
        }
        for name, call in entry_points.items():
            with self.subTest(entry_point=name):
                with self.assertRaisesRegex(ValueError, "grid.spectral_truncation"):
                    call()

    def test_all_three_slots_share_one_profile(self):
        d = DiffusionFilter.echam_lmidatm(63, 95)
        np.testing.assert_array_equal(d.level_orders_temp, d.level_orders_div)
        np.testing.assert_array_equal(d.level_orders_temp, d.level_orders_vor_q)


class DiffusionFilterAutoTest(unittest.TestCase):
    """`DiffusionFilter.auto` resolution-aware selection (#579)."""

    def test_hybrid_tabulated_layers_pick_lmidatm(self):
        # An ECHAM-family hybrid grid at a tabulated level count gets the
        # level-dependent lmidatm profile matching that (truncation, layers).
        auto = DiffusionFilter.auto(63, 47, "hybrid")
        expected = DiffusionFilter.echam_lmidatm(63, 47)
        npt.assert_array_equal(auto.level_orders_temp,
                               expected.level_orders_temp)

    def test_hybrid_untabulated_layers_fall_back_and_warn(self):
        # A hybrid grid at an untabulated level count falls back to the
        # uniform SPEEDY profile, and must warn since that is unlikely
        # intended.
        with self.assertLogs("jcm.diffusion", level="WARNING") as captured:
            auto = DiffusionFilter.auto(63, 31, "hybrid")
        self.assertIsNone(auto.level_orders_temp)
        self.assertEqual(float(auto.temp_timescale),
                         float(DiffusionFilter.default().temp_timescale))
        self.assertIn("no ECHAM lmidatm profile", "\n".join(captured.output))

    def test_sigma_grid_is_silent_uniform_default(self):
        # SPEEDY/Held-Suarez sigma grids are tuned for the uniform profile
        # and must not warn.
        with self.assertNoLogs("jcm.diffusion", level="WARNING"):
            auto = DiffusionFilter.auto(31, 8, "sigma")
        self.assertIsNone(auto.level_orders_temp)
        self.assertEqual(float(auto.temp_timescale),
                         float(DiffusionFilter.default().temp_timescale))


class DiffusionFilterValidateLayersTest(unittest.TestCase):

    def test_mismatched_profile_length_raises(self):
        # An L47 level-dependent profile pinned on an L95 grid must fail
        # with a named error, not an opaque broadcast error downstream.
        profile = DiffusionFilter.echam_lmidatm(85, 47)
        with self.assertRaisesRegex(
                ValueError, r"47 levels .* grid has 95 levels"):
            profile.validate_layers(95)

    def test_matching_length_passes(self):
        DiffusionFilter.echam_lmidatm(63, 47).validate_layers(47)

    def test_uniform_profile_has_no_level_constraint(self):
        # A uniform profile (no per-level orders) matches any level count.
        DiffusionFilter.default().validate_layers(95)


class DiffusionFilterScaledTest(unittest.TestCase):

    def test_scale_one_is_identity(self):
        base = DiffusionFilter.default()
        self.assertIs(base.scaled(1.0), base)

    def test_scale_multiplies_all_timescales(self):
        base = DiffusionFilter.default()
        scaled = base.scaled(3.0)
        self.assertEqual(float(scaled.div_timescale),
                         float(base.div_timescale) * 3.0)
        self.assertEqual(float(scaled.vor_q_timescale),
                         float(base.vor_q_timescale) * 3.0)
        self.assertEqual(float(scaled.temp_timescale),
                         float(base.temp_timescale) * 3.0)
        # Orders are unchanged by scaling.
        self.assertEqual(int(scaled.temp_order), int(base.temp_order))

    def test_scale_preserves_level_orders(self):
        base = DiffusionFilter.echam_lmidatm(63, 47)
        scaled = base.scaled(2.0)
        npt.assert_array_equal(scaled.level_orders_temp,
                               base.level_orders_temp)


if __name__ == "__main__":
    unittest.main()
