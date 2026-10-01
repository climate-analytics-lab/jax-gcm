"""κ-Köhler diagnostics: differentiable at an empty mode, in float32.

The forward values are covered by ``placeholder_test`` and
``jam_phase0_test``; what those never checked is the reverse pass at a mode
with no mass or no particles. Measured 2026-10-01 before the fix: in
float32 every ``_jam_state`` field had a NaN gradient there — ``r_dry``
with respect to number, ``rho``/``kappa`` with respect to mass, ``r_wet``
with respect to both — while in float64 all were finite. The cause was a
floored divisor whose square underflows in float32, so a NaN cotangent
reached every upstream parameter through ``where``'s unselected branch.

These tests pin the fix in the dtype the default configuration and CI
actually run.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics import koehler
from jcm.physics.aerosol.jam.microphysics.koehler import equilibrium_modal_state
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

ACC = MAM4_SPEC.mode("accum")
SHAPE = (2, 1)
FIELDS = ("r_dry", "r_wet", "rho", "kappa", "mass", "number")


def _state(acc_so4_mass, acc_number, saturation=0.8):
    """Every mode populated except as overridden on the accumulation mode."""
    masses = {mass_name(sp, m.short): jnp.full(SHAPE, 1.0e-10)
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.full(SHAPE, 1.0e8)
               for m in MAM4_SPEC.modes}
    for sp in ACC.species:
        masses[mass_name(sp, ACC.short)] = jnp.zeros(SHAPE)
    masses[mass_name("so4", ACC.short)] = acc_so4_mass
    numbers[number_name(ACC.short)] = acc_number
    return equilibrium_modal_state(masses, numbers, MAM4_SPEC,
                                   jnp.full(SHAPE, saturation))


class EmptyModeGradientTest(unittest.TestCase):

    def test_gradients_finite_with_no_mass_in_the_mode(self):
        for field in FIELDS:
            grad = jax.grad(lambda m: jnp.sum(
                getattr(_state(m, jnp.full(SHAPE, 1.0e8)), field)))(
                jnp.zeros(SHAPE))
            self.assertTrue(np.all(np.isfinite(np.asarray(grad))),
                            msg=f"d{field}/dmass is not finite at mass=0")

    def test_gradients_finite_with_no_particles_in_the_mode(self):
        for field in FIELDS:
            grad = jax.grad(lambda n: jnp.sum(
                getattr(_state(jnp.full(SHAPE, 1.0e-10), n), field)))(
                jnp.zeros(SHAPE))
            self.assertTrue(np.all(np.isfinite(np.asarray(grad))),
                            msg=f"d{field}/dnumber is not finite at number=0")

    def test_gradients_finite_with_neither(self):
        for field in FIELDS:
            grad = jax.grad(lambda m, n: jnp.sum(
                getattr(_state(m, n), field)), argnums=(0, 1))(
                jnp.zeros(SHAPE), jnp.zeros(SHAPE))
            self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in grad),
                            msg=f"{field} gradient not finite at (0, 0)")


class EmptyModeForwardTest(unittest.TestCase):

    def test_empty_mode_reports_reference_diameter_and_default_density(self):
        state = _state(jnp.zeros(SHAPE), jnp.zeros(SHAPE))
        acc = MAM4_SPEC.mode_index("accum")
        np.testing.assert_allclose(np.asarray(state.r_dry[acc]),
                                   0.5 * ACC.dgnum, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(state.rho[acc]),
            MAM4_SPEC.species_props(ACC.species[0]).density, rtol=1e-6)
        self.assertTrue(np.all(np.asarray(state.kappa[acc]) == 0.0))

    def test_populated_modes_are_untouched_by_the_empty_one(self):
        # The guards act per mode: an empty accumulation mode must not
        # perturb the Aitken diagnostics beside it.
        empty = _state(jnp.zeros(SHAPE), jnp.zeros(SHAPE))
        full = _state(jnp.full(SHAPE, 1.0e-10), jnp.full(SHAPE, 1.0e8))
        ait = MAM4_SPEC.mode_index("aitken")
        for field in FIELDS:
            np.testing.assert_array_equal(
                np.asarray(getattr(empty, field)[ait]),
                np.asarray(getattr(full, field)[ait]),
                err_msg=f"aitken {field} changed when accum was emptied")


if __name__ == "__main__":
    unittest.main()


class MamWaterUptakeTest(unittest.TestCase):
    """D-A6: MAM's Köhler solve, RH cap and hysteresis, pinned on closed forms."""

    def test_kelvin_coefficient_is_mams(self):
        # MAM: 2e4 · 18 · 76 / (8.3e7 · 273 · 1) µm = 1.2066e-3 µm; jcm's
        # constants (R = 8.3145, M_w = 18.015) give 1.2064 nm.
        self.assertAlmostEqual(koehler.kelvin_coefficient() * 1e9, 1.2064, places=3)

    def test_solve_satisfies_mams_quartic(self):
        """The returned radius is a root of r⁴ − (A/ln s) r³ + (κ r_d³/ln s − r_d³) r
        + A r_d³/ln s, and the smallest one at or above r_d (numpy's roots).
        """
        a = koehler.kelvin_coefficient()
        for r_d in (5e-9, 30e-9, 200e-9, 2e-6):
            for kappa in (0.05, 0.5, 1.16):
                for s in (0.4, 0.8, 0.95, 0.98):
                    x = float(koehler.kohler_growth_factor(jnp.asarray(r_d), jnp.asarray(kappa),
                                                           jnp.asarray(s)))
                    ln_s = math.log(s)
                    roots = np.roots([1.0, -a / ln_s, 0.0, kappa * r_d ** 3 / ln_s - r_d ** 3,
                                      a * r_d ** 3 / ln_s])
                    real = sorted(r.real for r in roots
                                  if abs(r.imag) < 1e-6 * abs(r) and r.real >= r_d * (1 - 1e-6))
                    self.assertTrue(real, (r_d, kappa, s))
                    self.assertAlmostEqual(x * r_d / real[0], 1.0, places=5, msg=(r_d, kappa, s))

    def test_kelvin_term_suppresses_small_particles_only(self):
        s, kappa = jnp.asarray(0.95), jnp.asarray(0.5)
        no_kelvin = (1.0 + 0.5 / -math.log(0.95)) ** (1 / 3)
        small = float(koehler.kohler_growth_factor(jnp.asarray(10e-9), kappa, s))
        large = float(koehler.kohler_growth_factor(jnp.asarray(5e-6), kappa, s))
        self.assertLess(small, no_kelvin * 0.97)
        self.assertAlmostEqual(large / no_kelvin, 1.0, places=2)   # A/r = 2e-4 at 5 µm

    def test_insoluble_and_empty_do_not_grow(self):
        for kappa in (0.0, 1e-10):
            g = float(koehler.kohler_growth_factor(jnp.asarray(50e-9), jnp.asarray(kappa),
                                                   jnp.asarray(0.9)))
            self.assertAlmostEqual(g, 1.0, places=6)

    def test_hysteresis_dry_below_crystallization_and_continuous(self):
        acc = MAM4_SPEC.mode("accum")
        r_v, kappa = jnp.asarray(100e-9), jnp.asarray(0.5)
        g = lambda rh: float(koehler.uptake_growth_factor(r_v, kappa, jnp.asarray(rh), acc))  # noqa: E731
        for rh in (0.0, 0.1, 0.3, 0.3499):
            self.assertEqual(g(rh), 1.0, rh)
        self.assertAlmostEqual(g(0.3501), 1.0, places=3)                  # continuous at 0.35
        self.assertAlmostEqual(g(0.7999) / g(0.8001), 1.0, delta=2e-3)    # continuous at 0.80
        self.assertGreater(g(0.6), g(0.5))
        # Between the switch points the WATER VOLUME is linear in RH.
        full = koehler.kohler_growth_factor(r_v, kappa, jnp.asarray(0.6)) ** 3 - 1.0
        self.assertAlmostEqual((g(0.6) ** 3 - 1.0) / float(full), (0.6 - 0.35) / 0.45, places=4)

    def test_humidity_is_capped_at_mams_value(self):
        acc = MAM4_SPEC.mode("accum")
        r_v, kappa = jnp.asarray(100e-9), jnp.asarray(0.5)
        g98 = float(koehler.uptake_growth_factor(r_v, kappa, jnp.asarray(0.98), acc))
        g99 = float(koehler.uptake_growth_factor(r_v, kappa, jnp.asarray(0.999), acc))
        self.assertEqual(g98, g99)

    def test_state_uses_the_volume_mean_radius_and_applies_the_ratio(self):
        """``dgncur_awet = dgncur_a · wetrad/dryrad`` with the solve at the
        volume-mean particle: the state's growth equals the growth of
        ``r_dry · exp(1.5 ln²σ)``, not of ``r_dry`` itself.
        """
        acc = MAM4_SPEC.mode("accum")
        sat = jnp.full((1, 1), 0.9)
        masses = {mass_name(sp, m.short): jnp.zeros((1, 1)) for m in MAM4_SPEC.modes for sp in m.species}
        numbers = {number_name(m.short): jnp.full((1, 1), 1e8) for m in MAM4_SPEC.modes}
        masses[mass_name("so4", "acc")] = jnp.full((1, 1), 1e-9)
        state = equilibrium_modal_state(masses, numbers, MAM4_SPEC, sat)
        i = [m.short for m in MAM4_SPEC.modes].index("acc")
        r_dry = state.r_dry[i]
        r_v = r_dry * math.exp(1.5 * math.log(acc.geom_std_dev) ** 2)
        expected = koehler.uptake_growth_factor(r_v, state.kappa[i], sat, acc)
        np.testing.assert_allclose(np.asarray(state.r_wet[i] / r_dry), np.asarray(expected), rtol=1e-6)
        naive = koehler.uptake_growth_factor(r_dry, state.kappa[i], sat, acc)
        self.assertFalse(np.allclose(np.asarray(state.r_wet[i] / r_dry), np.asarray(naive), rtol=1e-4))

    def test_gradients_finite_in_float32_across_the_switches(self):
        acc = MAM4_SPEC.mode("accum")
        for rh in (0.0, 0.2, 0.35, 0.5, 0.8, 0.9, 0.98, 0.999):
            grads = jax.grad(
                lambda r, k, s: koehler.uptake_growth_factor(r, k, s, acc), argnums=(0, 1, 2))(
                jnp.asarray(50e-9), jnp.asarray(0.5), jnp.asarray(rh))
            self.assertTrue(all(bool(jnp.isfinite(g)) for g in grads), rh)
        grads = jax.grad(lambda r, k, s: koehler.uptake_growth_factor(r, k, s, acc),
                         argnums=(0, 1, 2))(jnp.asarray(50e-9), jnp.asarray(0.0), jnp.asarray(0.6))
        self.assertTrue(all(bool(jnp.isfinite(g)) for g in grads))
