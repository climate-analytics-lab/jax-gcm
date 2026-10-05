"""Fast unit tests for ``ham_activation.py`` (#1017).

Faithfulness against the compiled ECHAM6.3-HAM2.3 routines is
``ham_activation_reference_test.py``'s job; this file covers shapes,
branches, basic physical sanity and the ``HamActivation`` term's own
contract (bad-scheme rejection, broadcasting) without the reference data.
"""
from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.activation.ham_activation import (
    ham_arg,
    ham_logtail,
    ham_updraft,
    koehler_ab,
    lin_leaitch,
)
from jcm.physics.aerosol.jam.activation.ham_activation_term import HamActivation
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.population import AerosolMode, AerosolSpecies, ModalAerosolSpec
from jcm.physics_interface import PhysicsState


def _two_mode_spec():
    """Build a minimal 2-mode (one activatable, one not) population for unit tests."""
    species = (
        AerosolSpecies("so4", 96.0631e-3, 1841.0, 0.6),
        AerosolSpecies("du", 250.0e-3, 2650.0, 0.0),
    )
    modes = (
        AerosolMode("accum", "acc", 1.59, 1.0e-7, 1.0e-8, 1.0e-6, ("so4", "du"),
                    soluble=True, can_activate=True, sediments=True),
        AerosolMode("insol", "ins", 1.59, 1.0e-7, 1.0e-8, 1.0e-6, ("du",),
                    soluble=False, can_activate=False, sediments=True),
    )
    return ModalAerosolSpec(modes=modes, species=species)


class KoehlerAbTest(unittest.TestCase):
    def test_zero_outside_can_activate(self):
        spec = _two_mode_spec()
        mass = {("so4", "acc"): jnp.asarray([1.0e-9]), ("du", "ins"): jnp.asarray([1.0e-9])}
        a, b = koehler_ab(spec, mass, jnp.asarray([280.0]))
        self.assertGreater(float(a[0, 0]), 0.0)
        self.assertGreater(float(b[0, 0]), 0.0)
        self.assertEqual(float(a[1, 0]), 0.0)   # "ins" cannot activate
        self.assertEqual(float(b[1, 0]), 0.0)

    def test_empty_mode_is_zero(self):
        spec = _two_mode_spec()
        a, b = koehler_ab(spec, {}, jnp.asarray([280.0]))
        self.assertEqual(float(a[0, 0]), 0.0)
        self.assertEqual(float(b[0, 0]), 0.0)

    def test_a_increases_as_temperature_falls(self):
        # A = _A_FACTOR / T: colder air has a LARGER curvature parameter.
        spec = _two_mode_spec()
        mass = {("so4", "acc"): jnp.asarray([1.0e-9, 1.0e-9])}
        a, _ = koehler_ab(spec, mass, jnp.asarray([250.0, 290.0]))
        self.assertGreater(float(a[0, 0]), float(a[0, 1]))

    def test_grad_through_mass_and_temperature_finite(self):
        spec = _two_mode_spec()

        def loss(m, t):
            a, b = koehler_ab(spec, {("so4", "acc"): m}, t)
            return jnp.sum(a) + jnp.sum(b)

        ga, gt = jax.grad(loss, argnums=(0, 1))(jnp.asarray([1.0e-9]), jnp.asarray([280.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(ga))))
        self.assertTrue(np.all(np.isfinite(np.asarray(gt))))


class HamLogtailTest(unittest.TestCase):
    def test_in_unit_interval(self):
        frac = ham_logtail(jnp.asarray(1.0e-7), jnp.asarray(1.2e-7), jnp.log(1.59))
        self.assertTrue(0.0 <= float(frac) <= 1.0)

    def test_smaller_cutoff_gives_larger_fraction(self):
        lo = ham_logtail(jnp.asarray(1.0e-7), jnp.asarray(0.5e-7), jnp.log(1.59))
        hi = ham_logtail(jnp.asarray(1.0e-7), jnp.asarray(2.0e-7), jnp.log(1.59))
        self.assertGreater(float(lo), float(hi))

    def test_mass_factor_shifts_the_distribution(self):
        number_frac = ham_logtail(jnp.asarray(1.0e-7), jnp.asarray(1.0e-7), jnp.log(1.8))
        mass_factor = float(jnp.exp(3.0 * jnp.log(1.8) ** 2))
        mass_frac = ham_logtail(jnp.asarray(1.0e-7), jnp.asarray(1.0e-7), jnp.log(1.8),
                                 mass_factor=mass_factor)
        # The mass median sits above the number median for sigma_g > 1, so
        # at the SAME cutoff the mass fraction above it is larger.
        self.assertGreater(float(mass_frac), float(number_frac))

    def test_grad_finite_including_at_the_branch_boundaries(self):
        def loss(r):
            return jnp.sum(ham_logtail(jnp.full((3,), 1.0e-7), r, jnp.log(1.59)))

        g = jax.grad(loss)(jnp.asarray([1.0e-7, 0.0, 1.0e-30]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))


class HamUpdraftTest(unittest.TestCase):
    def test_single_updraft_shape_and_w_min_floor(self):
        w, pwpdf = ham_updraft(jnp.asarray([0.0, 1.0]), jnp.asarray([10.0, -10.0]),
                                jnp.asarray([1.0, 1.0]), w_min=0.2, fact_tke=0.7,
                                n_pdf_bins=None)
        self.assertEqual(w.shape, (1, 2))
        np.testing.assert_allclose(np.asarray(pwpdf), 1.0)
        # Strongly descending + zero TKE: w_large + w_turb < 0, w_min binds.
        self.assertAlmostEqual(float(w[0, 0]), 0.2, places=6)

    def test_pdf_shape_and_bins_span_zero_to_4sigma(self):
        w, pwpdf = ham_updraft(jnp.asarray([1.0]), jnp.asarray([0.0]), jnp.asarray([1.0]),
                                w_min=0.0, fact_tke=0.7, n_pdf_bins=5)
        self.assertEqual(w.shape, (5, 1))
        self.assertEqual(pwpdf.shape, (5, 1))
        self.assertTrue(np.all(np.asarray(w) >= 0.0))
        self.assertTrue(np.all(np.asarray(pwpdf) >= 0.0))

    def test_grad_at_zero_tke_finite(self):
        def loss(tke):
            w, _ = ham_updraft(tke, jnp.asarray([0.0]), jnp.asarray([1.0]),
                                w_min=0.0, fact_tke=0.7, n_pdf_bins=None)
            return jnp.sum(w)

        g = jax.grad(loss)(jnp.asarray([0.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))

    def test_broadcast_1d_vs_2d(self):
        tke = jnp.asarray([0.3, 0.5])
        omega = jnp.asarray([-5.0, 3.0])
        rho = jnp.asarray([1.0, 1.1])
        w1, p1 = ham_updraft(tke, omega, rho, 0.0, 0.7, n_pdf_bins=4)
        tke2 = jnp.stack([tke, tke], axis=-1)
        omega2 = jnp.stack([omega, omega], axis=-1)
        rho2 = jnp.stack([rho, rho], axis=-1)
        w2, p2 = ham_updraft(tke2, omega2, rho2, 0.0, 0.7, n_pdf_bins=4)
        for c in range(2):
            np.testing.assert_allclose(np.asarray(w2[:, :, c]), np.asarray(w1))
            np.testing.assert_allclose(np.asarray(p2[:, :, c]), np.asarray(p1))


def _arg_inputs(n_modes=2, n=1):
    r_dry = jnp.full((n_modes, n), 0.1e-6)
    number_vol = jnp.full((n_modes, n), 1.0e8)
    a = jnp.full((n_modes, n), 1.0e-9)
    b = jnp.full((n_modes, n), 0.5)
    can_activate = jnp.asarray([True] + [False] * (n_modes - 1))
    sigma_g = jnp.full((n_modes,), 1.6)
    return r_dry, number_vol, a, b, can_activate, sigma_g


class HamArgTest(unittest.TestCase):
    def test_fraction_in_unit_interval_and_cdnc_bounded(self):
        r_dry, number_vol, a, b, can_activate, sigma_g = _arg_inputs()
        w, pwpdf = ham_updraft(jnp.asarray([0.3]), jnp.asarray([-5.0]), jnp.asarray([1.0]),
                                0.0, 0.7, n_pdf_bins=None)
        cdncact, nfrac, nact, sm, smax, rc = ham_arg(
            r_dry, number_vol, a, b, can_activate, sigma_g, w, pwpdf,
            jnp.asarray([280.0]), jnp.asarray([9.0e4]), jnp.asarray([5.0e-3]),
            jnp.asarray([1000.0]),
        )
        self.assertTrue(0.0 <= float(nfrac[0, 0]) <= 1.0)
        self.assertEqual(float(nfrac[1, 0]), 0.0)   # not can_activate
        self.assertLessEqual(float(cdncact[0]), 1.0e8 + 1.0)
        self.assertTrue(np.isfinite(float(smax[0, 0])))

    def test_cold_cell_gives_zero_activation(self):
        """T below cthomi (238.15 K) gates ARG off entirely."""
        r_dry, number_vol, a, b, can_activate, sigma_g = _arg_inputs()
        w, pwpdf = ham_updraft(jnp.asarray([0.3]), jnp.asarray([-5.0]), jnp.asarray([1.0]),
                                0.0, 0.7, n_pdf_bins=None)
        cdncact, *_ = ham_arg(
            r_dry, number_vol, a, b, can_activate, sigma_g, w, pwpdf,
            jnp.asarray([230.0]), jnp.asarray([9.0e4]), jnp.asarray([5.0e-3]),
            jnp.asarray([1000.0]),
        )
        self.assertEqual(float(cdncact[0]), 0.0)

    def test_grad_through_number_and_updraft_finite(self):
        r_dry, number_vol, a, b, can_activate, sigma_g = _arg_inputs()

        def loss(n_scale, w_val):
            w = jnp.full((1, 1), w_val)
            cdncact, *_ = ham_arg(
                r_dry, number_vol * n_scale, a, b, can_activate, sigma_g, w,
                jnp.ones_like(w), jnp.asarray([280.0]), jnp.asarray([9.0e4]),
                jnp.asarray([5.0e-3]), jnp.asarray([1000.0]),
            )
            return jnp.sum(cdncact)

        gn, gw = jax.grad(loss, argnums=(0, 1))(1.0, 0.3)
        self.assertTrue(np.isfinite(gn) and np.isfinite(gw))

    def test_broadcast_1col_vs_3col(self):
        r_dry, number_vol, a, b, can_activate, sigma_g = _arg_inputs(n=1)
        w, pwpdf = ham_updraft(jnp.asarray([0.3]), jnp.asarray([-5.0]), jnp.asarray([1.0]),
                                0.0, 0.7, n_pdf_bins=None)
        out1 = ham_arg(r_dry, number_vol, a, b, can_activate, sigma_g, w, pwpdf,
                        jnp.asarray([280.0]), jnp.asarray([9.0e4]), jnp.asarray([5.0e-3]),
                        jnp.asarray([1000.0]))

        def tile(x, k):
            return jnp.concatenate([x] * k, axis=-1)

        out3 = ham_arg(tile(r_dry, 3), tile(number_vol, 3), tile(a, 3), tile(b, 3),
                        can_activate, sigma_g, tile(w, 3), tile(pwpdf, 3),
                        tile(jnp.asarray([280.0]), 3), tile(jnp.asarray([9.0e4]), 3),
                        tile(jnp.asarray([5.0e-3]), 3), tile(jnp.asarray([1000.0]), 3))
        for i in range(3):
            np.testing.assert_allclose(np.asarray(out3[0][i:i + 1]), np.asarray(out1[0]))


class LinLeaitchTest(unittest.TestCase):
    def test_fraction_and_bounds(self):
        number_vol = jnp.full((2, 1), 1.0e8)
        can_activate = jnp.asarray([True, False])
        wet_radius = jnp.full((2, 1), 0.1e-6)
        sigma_g = jnp.full((2,), 1.6)
        na, na_cv, cdncact, cdncact_cv = lin_leaitch(
            number_vol, can_activate, wet_radius, sigma_g, jnp.asarray([0.3]))
        self.assertGreater(float(na[0]), 0.0)
        self.assertLessEqual(float(cdncact[0]), float(na[0]) + 1.0)
        self.assertGreaterEqual(float(cdncact_cv[0]), float(cdncact[0]))  # smaller cut, more available

    def test_zero_updraft_gives_zero_activation(self):
        number_vol = jnp.full((1, 1), 1.0e8)
        can_activate = jnp.asarray([True])
        wet_radius = jnp.full((1, 1), 0.1e-6)
        sigma_g = jnp.full((1,), 1.6)
        _, _, cdncact, _ = lin_leaitch(
            number_vol, can_activate, wet_radius, sigma_g, jnp.asarray([0.0]))
        self.assertEqual(float(cdncact[0]), 0.0)


class HamActivationTermTest(unittest.TestCase):
    def _spec(self):
        return _two_mode_spec()

    def _state_and_diag(self, nlev=3, ncols=2, scheme="arg"):
        spec = self._spec()
        n_modes = len(spec.modes)
        shape = (n_modes, nlev, ncols)
        jam = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6), r_wet=jnp.full(shape, 0.15e-6),
            rho=jnp.full(shape, 1700.0), kappa=jnp.full(shape, 0.5),
            mass=jnp.zeros(shape), number=jnp.full(shape, 1.0e8 / 1.1),
        )
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 283.0),
            specific_humidity=jnp.full((nlev, ncols), 5.0e-3),
            tracers={"m_so4_acc": jnp.full((nlev, ncols), 1.0e-9)},
        )
        diagnostics = {
            "_jam_state": jam,
            "pressure_full": jnp.full((nlev, ncols), 9.0e4),
            "air_density": jnp.full((nlev, ncols), 1.1),
            "_dycore_fields": {"omega": jnp.full((nlev, ncols), -5.0)},
            "vertical_diffusion": type("V", (), {"tke": jnp.full((nlev, ncols), 0.3)})(),
        }
        return state, diagnostics

    def test_arg_writes_activated_cdnc(self):
        state, diag = self._state_and_diag()
        term = HamActivation(self._spec(), scheme="arg")
        tend, out = term(state, diag, None, None)
        self.assertTrue(bool(jnp.all(tend.temperature == 0.0)))
        self.assertIn("activated_cdnc", out)
        self.assertEqual(out["activated_cdnc"].shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(np.asarray(out["activated_cdnc"]))))
        self.assertTrue(bool(jnp.all(out["activated_fraction"] <= 1.0 + 1e-6)))

    def test_lin_leaitch_writes_activated_cdnc(self):
        state, diag = self._state_and_diag(scheme="lin_leaitch")
        term = HamActivation(self._spec(), scheme="lin_leaitch")
        _, out = term(state, diag, None, None)
        self.assertTrue(np.all(np.isfinite(np.asarray(out["activated_cdnc"]))))

    def test_bad_scheme_rejected(self):
        with self.assertRaises(ValueError):
            HamActivation(self._spec(), scheme="nope")

    def test_broadcast_modes_nlev_vs_modes_nlev_ncols(self):
        state2, diag2 = self._state_and_diag(nlev=3, ncols=1)
        state3, diag3 = self._state_and_diag(nlev=3, ncols=3)
        term = HamActivation(self._spec(), scheme="arg")
        _, out2 = term(state2, diag2, None, None)
        _, out3 = term(state3, diag3, None, None)
        for c in range(3):
            np.testing.assert_allclose(
                np.asarray(out3["activated_cdnc"][:, c]),
                np.asarray(out2["activated_cdnc"][:, 0]), rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
