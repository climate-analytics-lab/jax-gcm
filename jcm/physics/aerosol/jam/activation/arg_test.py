"""Phase 1 tests: ARG activation core + term, incl. Ghosh-2025 Table-3 oracle."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.activation.arg import (
    _shape_coefficients,
    arg_activation,
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
        n_act, frac, smax = arg_activation(
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

    def test_ghosh_variant_runs_and_differs(self):
        _, arg, _ = self._run(variant="arg2000", n_percc=1500.0, w=0.2)
        _, gho, _ = self._run(variant="ghosh2025", n_percc=1500.0, w=0.2)
        self.assertTrue(np.isfinite(gho))
        # In the polluted/low-w regime the two should not be identical.
        self.assertNotAlmostEqual(arg, gho, places=6)

    def test_grad_through_updraft_finite(self):
        def loss(w):
            n_act, _, _ = arg_activation(
                updraft=jnp.full((1, 1), w),
                temperature=jnp.full((1, 1), 283.0),
                pressure=jnp.full((1, 1), 9.0e4),
                sigma_acc=1.8, variant="arg2000", **_single_mode(),
            )
            return jnp.sum(n_act)

        g = jax.jit(jax.grad(loss))(jnp.asarray(0.5))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreaterEqual(float(g), 0.0)  # more updraft -> more droplets


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
