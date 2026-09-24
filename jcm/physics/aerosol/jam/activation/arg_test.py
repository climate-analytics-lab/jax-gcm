"""Phase 1 tests: ARG activation core + term, incl. Ghosh-2025 Table-3 oracle."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from unittest import mock

from jcm.physics.aerosol.jam.activation import arg as arg_module
from jcm.physics.aerosol.jam.activation.arg import (
    _shape_coefficients,
    air_thermal_conductivity,
    arg_activation,
    vapor_diffusivity,
)
from jcm.physics.aerosol.jam.activation.arg_term import ArgActivation


# Ghosh et al. (2025) Table 3: σ_acc -> (f, g, p).
_GHOSH_TABLE3 = {
    1.4: (0.0109, 0.6608, 0.0462),
    1.6: (0.0124, 0.5968, 0.3198),
    1.8: (0.0141, 0.5328, 0.5221),
    2.0: (0.0160, 0.4688, 0.6659),
    2.1: (0.0172, 0.4368, 0.7226),
}


class ShapeCoefficientTest(unittest.TestCase):
    def test_ghosh2025_reproduces_table3(self):
        # ζ/η ≤ 1 selects the p_lim polynomial branch.
        zeta_over_eta = jnp.asarray(0.5)
        ln_sigma = jnp.asarray(0.0)  # unused by the ghosh branch
        for sigma_acc, (f_ref, g_ref, p_ref) in _GHOSH_TABLE3.items():
            f, g, p = _shape_coefficients(
                ln_sigma, zeta_over_eta, sigma_acc, "ghosh2025",
            )
            self.assertAlmostEqual(float(f), f_ref, delta=2e-4)
            self.assertAlmostEqual(float(g), g_ref, delta=2e-3)
            self.assertAlmostEqual(float(p), p_ref, delta=1.2e-2)

    def test_ghosh2025_kinetic_branch_sets_p_1p5(self):
        f, g, p = _shape_coefficients(
            jnp.asarray(0.0), jnp.asarray(2.0), 1.8, "ghosh2025",
        )
        self.assertAlmostEqual(float(p), 1.5)

    def test_arg2000_coefficients(self):
        ln_sigma = jnp.log(jnp.asarray(2.0))
        f, g, p = _shape_coefficients(
            ln_sigma, jnp.asarray(0.5), 1.8, "arg2000",
        )
        self.assertAlmostEqual(float(f), 0.5 * np.exp(2.5 * np.log(2.0) ** 2),
                               places=5)
        self.assertAlmostEqual(float(g), 1.0 + 0.25 * np.log(2.0), places=5)
        self.assertAlmostEqual(float(p), 1.5)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            _shape_coefficients(jnp.asarray(0.0), jnp.asarray(0.5), 1.8, "x")


class PerModeFractionTest(unittest.TestCase):
    def test_per_mode_fractions_physical(self):
        # Two modes, the second masked non-activatable. Number and mass
        # fractions must be in [0, 1], the mass fraction at least the number
        # fraction (large particles activate preferentially), and masked
        # modes exactly zero in both.
        two = lambda a, b: jnp.asarray([a, b]).reshape(2, 1, 1)
        kw = dict(
            r_dry=two(0.05e-6, 0.05e-6),
            kappa=two(0.6, 0.6),
            number_vol=two(1.0e8, 1.0e8),
            sigma_g=two(1.8, 1.8),
            can_activate=two(1.0, 0.0),
        )
        _, _, _, fn, fm = arg_activation(
            updraft=jnp.full((1, 1), 0.5),
            temperature=jnp.full((1, 1), 283.0),
            pressure=jnp.full((1, 1), 9.0e4),
            sigma_acc=1.8,
            variant="arg2000",
            **kw,
        )
        fn0, fm0 = float(fn[0, 0, 0]), float(fm[0, 0, 0])
        self.assertGreater(fn0, 0.0)
        self.assertLessEqual(fn0, 1.0)
        self.assertGreaterEqual(fm0, fn0)
        self.assertLessEqual(fm0, 1.0)
        self.assertEqual(float(fn[1, 0, 0]), 0.0)
        self.assertEqual(float(fm[1, 0, 0]), 0.0)

    def test_mass_fraction_is_the_shifted_number_erf(self):
        # Pin the exact ARG (2000) relation between the two fractions:
        # recover u from the returned number fraction (fn = erfc(u)/2) and
        # require fm == erfc(u − 3·lnσ/√2)/2. A plausible-wrong shift
        # (3·lnσ/2, 3√2·lnσ, √2·lnσ) fails this; the inequality checks
        # above would not catch it.
        from scipy.special import erf, erfinv

        sigma = 1.8
        _, _, _, fn, fm = arg_activation(
            updraft=jnp.full((1, 1), 0.5),
            temperature=jnp.full((1, 1), 283.0),
            pressure=jnp.full((1, 1), 9.0e4),
            sigma_acc=sigma,
            variant="arg2000",
            **_single_mode(sigma=sigma),
        )
        fn0, fm0 = float(fn[0, 0, 0]), float(fm[0, 0, 0])
        u = erfinv(1.0 - 2.0 * fn0)
        expected = 0.5 * (
            1.0 - erf(u - 3.0 * np.log(sigma) / np.sqrt(2.0))
        )
        self.assertAlmostEqual(fm0, float(expected), places=5)


def _single_mode(r_dry=0.05e-6, kappa=0.6, n_percc=100.0, sigma=1.8):
    """One-mode (M=1, 1 level, 1 col) ARG input set."""
    one = lambda v: jnp.full((1, 1, 1), v)
    return dict(
        r_dry=one(r_dry),
        kappa=one(kappa),
        number_vol=one(n_percc * 1.0e6),   # cm^-3 -> m^-3
        sigma_g=one(sigma),
        can_activate=one(1.0),
    )


def _scalar(x):
    return float(jnp.squeeze(x))


class ArgActivationCoreTest(unittest.TestCase):
    def _run(self, w=0.5, T=283.0, p=9.0e4, variant="arg2000", **over):
        kw = _single_mode(**over)
        n_act, frac, smax, _, _ = arg_activation(
            updraft=jnp.full((1, 1), w),
            temperature=jnp.full((1, 1), T),
            pressure=jnp.full((1, 1), p),
            sigma_acc=1.8,
            variant=variant,
            **kw,
        )
        return _scalar(n_act), _scalar(frac), _scalar(smax)

    def test_fraction_in_unit_interval(self):
        n_act, frac, smax = self._run()
        self.assertTrue(0.0 <= frac <= 1.0)
        self.assertTrue(smax > 0.0)
        # activated number cannot exceed available number
        self.assertLessEqual(n_act, 100.0 * 1.0e6 + 1.0)

    def test_more_updraft_activates_more(self):
        _, lo, _ = self._run(w=0.05)
        _, hi, _ = self._run(w=2.0)
        self.assertGreaterEqual(hi, lo)

    def test_more_aerosol_lowers_fraction(self):
        _, dilute, _ = self._run(n_percc=10.0)
        _, polluted, _ = self._run(n_percc=2000.0)
        self.assertLessEqual(polluted, dilute)

    def test_empty_population_no_activation(self):
        n_act, frac, _ = self._run(n_percc=0.0)
        self.assertAlmostEqual(n_act, 0.0)
        self.assertAlmostEqual(frac, 0.0)

    def test_negative_number_ringing_stays_bounded(self):
        """Negative modal number (spectral Gibbs ringing on the growing
        aerosol field) must not blow the activated fraction. Without flooring
        the number at 0, ``n_total`` goes negative, ``activated_fraction =
        n_act / n_total`` diverges to +/-huge, and that garbage fraction feeds
        wet scavenging. Assert a floored, physical result instead.
        """
        n_act, frac, smax = self._run(n_percc=-100.0)
        self.assertTrue(np.isfinite(n_act) and np.isfinite(frac))
        self.assertGreaterEqual(n_act, 0.0)          # no negative droplets
        self.assertTrue(0.0 <= frac <= 1.0)          # not +/-1e35 / -inf

    def test_ghosh_variant_runs_and_differs(self):
        _, arg, _ = self._run(variant="arg2000", n_percc=1500.0, w=0.2)
        _, gho, _ = self._run(variant="ghosh2025", n_percc=1500.0, w=0.2)
        self.assertTrue(np.isfinite(gho))
        # In the polluted/low-w regime the two should not be identical.
        self.assertNotAlmostEqual(arg, gho, places=6)

    def test_grad_through_updraft_finite(self):
        def loss(w):
            n_act, *_ = arg_activation(
                updraft=jnp.full((1, 1), w),
                temperature=jnp.full((1, 1), 283.0),
                pressure=jnp.full((1, 1), 9.0e4),
                sigma_acc=1.8, variant="arg2000", **_single_mode(),
            )
            return jnp.sum(n_act)

        g = jax.jit(jax.grad(loss))(jnp.asarray(0.5))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreaterEqual(float(g), 0.0)  # more updraft -> more droplets


def _mam4_like():
    """Aitken / accumulation / coarse modes, MAM4-like sizes and kappas."""
    col = lambda *v: jnp.asarray(v).reshape(3, 1, 1)
    return dict(
        r_dry=col(0.013e-6, 0.06e-6, 1.0e-6),
        kappa=col(0.5, 0.5, 1.1),
        number_vol=col(500.0e6, 200.0e6, 1.0e6),
        sigma_g=col(1.6, 1.8, 1.8),
        can_activate=col(1.0, 1.0, 1.0),
    )


class ArgTransportCoefficientTest(unittest.TestCase):
    """T/p-dependent Dv and Ka, CAM ``ndrop.F90`` ``diff0`` / ``conduct0`` (#679)."""

    def test_match_cam_ndrop_expressions(self):
        for t, p in ((273.0, 101325.0), (280.0, 9.0e4), (250.0, 5.0e4)):
            with self.subTest(t=t, p=p):
                diff0 = 0.211e-4 * (1013.25e2 / p) * (t / 273.0) ** 1.94
                conduct0 = (5.69 + 0.017 * (t - 273.0)) * 4.186e2 * 1.0e-5
                self.assertAlmostEqual(
                    float(vapor_diffusivity(jnp.asarray(t), jnp.asarray(p))),
                    diff0, delta=1e-6 * diff0)
                self.assertAlmostEqual(
                    float(air_thermal_conductivity(jnp.asarray(t))),
                    conduct0, delta=1e-6 * conduct0)

    def _n_act(self, t, p, constant):
        """Activated number [m^-3]; ``constant`` pins the old sea-level Dv/Ka."""
        def run():
            n_act, *_ = arg_activation(
                updraft=jnp.full((1, 1), 0.3),
                temperature=jnp.full((1, 1), t),
                pressure=jnp.full((1, 1), p),
                sigma_acc=1.8, variant="arg2000", **_mam4_like(),
            )
            return _scalar(n_act)

        if not constant:
            return run()
        with mock.patch.object(arg_module, "vapor_diffusivity",
                               lambda t, p: 2.11e-5), \
             mock.patch.object(arg_module, "air_thermal_conductivity",
                               lambda t: 0.024):
            return run()

    def test_activation_drops_aloft_by_the_expected_magnitude(self):
        """Constant Dv/Ka over-activate, increasingly with altitude.

        At 500 hPa / 250 K Dv is ~1.7x its sea-level value, so the growth
        resistance falls, the maximum supersaturation falls, and a MAM4-like
        population activates ~10-25 % fewer droplets than with the constants
        (the #679 audit measured -4 % at 900 hPa and -19 % at 500 hPa/260 K on
        a similar population). Near the reference state the two agree to a
        few percent.
        """
        drop = {}
        for t, p in ((280.0, 9.0e4), (270.0, 7.0e4), (250.0, 5.0e4)):
            local = self._n_act(t, p, constant=False)
            const = self._n_act(t, p, constant=True)
            self.assertGreater(local, 0.0)
            drop[p] = 1.0 - local / const
        self.assertGreater(drop[5.0e4], 0.10)
        self.assertLess(drop[5.0e4], 0.25)
        self.assertLess(abs(drop[9.0e4]), 0.08)
        self.assertGreater(drop[5.0e4], drop[7.0e4])
        self.assertGreater(drop[7.0e4], drop[9.0e4])

    def test_grad_through_temperature_and_pressure_finite(self):
        def loss(t, p):
            n_act, *_ = arg_activation(
                updraft=jnp.full((1, 1), 0.3),
                temperature=jnp.full((1, 1), t),
                pressure=jnp.full((1, 1), p),
                sigma_acc=1.8, variant="arg2000", **_mam4_like(),
            )
            return jnp.sum(n_act)

        gt, gp = jax.grad(loss, argnums=(0, 1))(
            jnp.asarray(250.0), jnp.asarray(5.0e4))
        self.assertTrue(np.isfinite(float(gt)) and np.isfinite(float(gp)))
        self.assertNotEqual(float(gp), 0.0)


class ArgTermTest(unittest.TestCase):
    def _jam_state(self, nlev=3, ncols=2):
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.jam_state import JamAerosolState

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        return JamAerosolState(
            r_dry=jnp.full(shape, 0.05e-6),
            r_wet=jnp.full(shape, 0.1e-6),
            rho=jnp.full(shape, 1700.0),
            kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9),
            number=jnp.full(shape, 1.0e8),  # kg^-1
        )

    def test_term_writes_activated_cdnc(self):
        from jcm.physics_interface import PhysicsState

        nlev, ncols = 3, 2
        term = ArgActivation()
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 283.0),
            specific_humidity=jnp.full((nlev, ncols), 0.004),
        )
        diagnostics = {
            "_jam_state": self._jam_state(nlev, ncols),
            "pressure_full": jnp.full((nlev, ncols), 9.0e4),
            "air_density": jnp.full((nlev, ncols), 1.1),
        }
        tend, diag = term(state, diagnostics, None, None)
        self.assertTrue(bool(jnp.all(tend.temperature == 0.0)))
        self.assertIn("activated_cdnc", diag)
        self.assertEqual(diag["activated_cdnc"].shape, (nlev, ncols))
        self.assertTrue(np.all(np.isfinite(np.asarray(diag["activated_cdnc"]))))
        self.assertTrue(bool(jnp.all(diag["activated_cdnc"] >= 0.0)))
        self.assertTrue(bool(jnp.all(diag["activated_fraction"] <= 1.0 + 1e-6)))

    def test_bad_variant_rejected(self):
        with self.assertRaises(ValueError):
            ArgActivation(variant="nope")


if __name__ == "__main__":
    unittest.main()
