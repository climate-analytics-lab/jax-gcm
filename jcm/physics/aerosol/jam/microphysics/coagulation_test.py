"""AER-05: coagulation, pinned on closed forms, quadrature and conservation.

The CMAQ "fast" coefficients are analytic moment integrals with fitted
corrections, so they can be checked two ways without a reference run: in
the monodisperse limit (σ_g → 1) every correction table reduces to a
known constant and the expressions must reproduce the classical Brownian
kernels exactly; and at MAM4 geometry they must agree with a brute-force
double quadrature of the same kernels over the two log-normals to the
accuracy Binkowski and Roselle (2003) claim for the method. The number
solution is checked against fine explicit integration in each of MAM's
three regimes and for continuity across the switches. The process is
checked for conservation, monotone number loss, the coating record and
finite float32 gradients at empty modes.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c

from jcm.physics.aerosol.jam.microphysics import coagulation as cg
from jcm.physics.aerosol.jam.microphysics.coagulation_tables import BM0, BM0IJ, BM3I
from jcm.physics.aerosol.jam.microphysics.koehler import (
    equilibrium_modal_state,
    saturation_ratio,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.modal import (
    MicrophysicsEnvironment,
    ModalPopulation,
)
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

SHAPE = (3, 2)
T, P, Q, DT = 285.0, 9.0e4, 0.005, 1800.0
#: MAM4 reference geometry: (wet diameter [m], wet density [kg/m³], σ_g).
AITKEN = (26.0e-9, 1770.0, 1.6)
ACCUM = (110.0e-9, 1770.0, 1.8)
PCARBON = (50.0e-9, 1500.0, 1.6)


def _population(seed=0, *, empty_modes=False, empty=()):
    rng = np.random.default_rng(seed)
    masses = {mass_name(sp, m.short): jnp.asarray(rng.uniform(1e-12, 1e-9, SHAPE))
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.asarray(rng.uniform(1e6, 1e9, SHAPE))
               for m in MAM4_SPEC.modes}
    for short in MAM4_SPEC.modes if empty_modes else ():
        numbers[number_name(short.short)] = jnp.zeros(SHAPE)
    for short in empty:
        numbers[number_name(short)] = jnp.zeros(SHAPE)
        mode = next(m for m in MAM4_SPEC.modes if m.short == short)
        for sp in mode.species:
            masses[mass_name(sp, short)] = jnp.zeros(SHAPE)
    gases = {gas_name("h2so4"): jnp.full(SHAPE, 1.0e-10),
             gas_name("soag"): jnp.full(SHAPE, 1.0e-10)}
    return ModalPopulation(masses=masses, numbers=numbers, gases=gases)


def _env(population):
    temp = jnp.full(SHAPE, T)
    pressure = jnp.full(SHAPE, P)
    sat = saturation_ratio(temp, jnp.full(SHAPE, Q), pressure)
    return MicrophysicsEnvironment(
        temperature=temp, pressure=pressure, saturation=sat,
        air_density=pressure / (c.rd * temp),
        state_start=equilibrium_modal_state(
            population.masses, population.numbers, MAM4_SPEC, sat),
        h2so4_production=jnp.zeros(SHAPE),
        height_above_surface=jnp.full(SHAPE, 500.0),
        pbl_height=jnp.zeros(SHAPE[1:]), dt=DT)


# --- reference kernels, per particle pair (Binkowski and Shankar 1995, A5/A6) ---

def _kernel(d1, d2, rho1, rho2, temperature, mean_free_path, viscosity):
    """Harmonic mean of the free-molecular and near-continuum kernels."""
    knc = (2.0 / 3.0) * c.ak * temperature / viscosity
    kfm = math.sqrt(6.0 * c.ak * temperature / (rho1 + rho2))
    a = cg.SLIP_CORRECTION
    k_nc = knc * (d1 + d2) * (1.0 / d1 + 1.0 / d2
                              + 2.0 * a * mean_free_path * (1.0 / d1 ** 2 + 1.0 / d2 ** 2))
    k_fm = kfm * (d1 + d2) ** 2 * np.sqrt(1.0 / d1 ** 3 + 1.0 / d2 ** 3)
    return k_fm * k_nc / (k_fm + k_nc)


def _lognormal_grid(diameter, sigma, n=600, width=7.0):
    ln_sigma = math.log(sigma)
    x = np.linspace(-width * ln_sigma, width * ln_sigma, n)
    d = diameter * np.exp(x)
    pdf = np.exp(-0.5 * (x / ln_sigma) ** 2) / (ln_sigma * math.sqrt(2.0 * math.pi))
    w = pdf * (x[1] - x[0])
    return d, w / w.sum()


def _quadrature(donor, receiver, temperature=T, pressure=P):
    """``(β_ij0, β_ij3)`` by double quadrature over the two log-normals."""
    lam, mu = (float(v) for v in cg.air_properties(jnp.asarray(temperature),
                                                   jnp.asarray(pressure)))
    d1, w1 = _lognormal_grid(donor[0], donor[2])
    d2, w2 = _lognormal_grid(receiver[0], receiver[2])
    k = _kernel(d1[:, None], d2[None, :], donor[1], receiver[1], temperature, lam, mu)
    beta_0 = (w1[:, None] * w2[None, :] * k).sum()
    volume = d1 ** 3
    beta_3 = (w1[:, None] * w2[None, :] * k * volume[:, None]).sum() / (w1 * volume).sum()
    return beta_0, beta_3


def _intra_quadrature(mode, temperature=T, pressure=P):
    beta_0, _ = _quadrature(mode, mode, temperature, pressure)
    return 0.5 * beta_0


def _coefficients(donor, receiver, temperature=T, pressure=P):
    lam, mu = cg.air_properties(jnp.asarray(temperature), jnp.asarray(pressure))
    b0, b3 = cg.intermodal_coefficients(
        jnp.asarray(donor[0]), jnp.asarray(donor[1]), donor[2],
        jnp.asarray(receiver[0]), jnp.asarray(receiver[1]), receiver[2],
        jnp.asarray(temperature), lam, mu)
    return float(b0), float(b3)


def _self_coefficient(mode, temperature=T, pressure=P):
    lam, mu = cg.air_properties(jnp.asarray(temperature), jnp.asarray(pressure))
    return float(cg.intramodal_coefficient(
        jnp.asarray(mode[0]), jnp.asarray(mode[1]), mode[2],
        jnp.asarray(temperature), lam, mu))


class TableTest(unittest.TestCase):

    def test_tables_have_cmaq_shapes_and_ranges(self):
        self.assertEqual(BM0.shape, (10,))
        self.assertEqual(BM0IJ.shape, (10, 10, 10))
        self.assertEqual(BM3I.shape, (10, 10, 10))
        for table in (BM0, BM0IJ, BM3I):
            self.assertTrue(np.all((table > 0.6) & (table <= 1.0)))

    def test_monodisperse_corners_are_the_classical_constants(self):
        """At σ_g = 1 Whitby's expansions carry 8 (intra) and 9 (inter) unit
        terms against the exact ``4√2``, so the fitted corrections must be
        ``4√2/8`` and ``4√2/9`` there. The latter is the evidence that the
        tables were fitted WITH the nine-term 0th-moment form (the module
        docstring's transcription note) and absorb it.
        """
        self.assertAlmostEqual(BM0[0], 4.0 * math.sqrt(2.0) / 8.0, places=8)
        self.assertAlmostEqual(BM0IJ[0, 0, 0], 4.0 * math.sqrt(2.0) / 9.0, places=5)
        # 3rd moment at σ = 1, equal diameters: 1+1+2+1+1+2 = 8 unit terms.
        self.assertAlmostEqual(BM3I[0, 0, 0], 4.0 * math.sqrt(2.0) / 8.0, places=3)


class KernelTest(unittest.TestCase):

    def test_air_properties_at_standard_conditions(self):
        lam, mu = cg.air_properties(jnp.asarray(c.tmelt + 15.0), jnp.asarray(101325.0))
        self.assertAlmostEqual(float(lam) / 6.6328e-8, 1.0, places=6)
        self.assertAlmostEqual(float(mu) / 1.789e-5, 1.0, places=2)

    def test_monodisperse_limit_reproduces_the_classical_kernel(self):
        """σ_g → 1: every ``es`` is 1, the σ class is 1, and the analytic
        forms must equal the pairwise kernel at the mean diameter exactly
        (intra: half of it, since dN/dt = −½ K N²).
        """
        lam, mu = (float(v) for v in cg.air_properties(jnp.asarray(T), jnp.asarray(P)))
        for d in (10e-9, 100e-9, 1e-6):
            beta = _self_coefficient((d, 1770.0, 1.0 + 1e-9))
            expected = 0.5 * _kernel(d, d, 1770.0, 1770.0, T, lam, mu)
            # The fm expansion has 8 unit terms × 4√2/8 → exact; the
            # harmonic mean then matches the pairwise form identically.
            self.assertAlmostEqual(beta / expected, 1.0, places=5)

    def test_coefficients_match_quadrature_at_mam4_geometry(self):
        """The fast method against brute-force double quadrature of the
        same kernels. Binkowski and Roselle (2003) report the approximate
        0th and 3rd moment rates within ~10 % of the quadrature over the
        range of modal parameters; that is the bar here.
        """
        for donor, receiver in ((AITKEN, ACCUM), (AITKEN, PCARBON), (PCARBON, ACCUM)):
            b0, b3 = _coefficients(donor, receiver)
            q0, q3 = _quadrature(donor, receiver)
            self.assertLess(abs(b0 / q0 - 1.0), 0.10, (donor, receiver, b0, q0))
            self.assertLess(abs(b3 / q3 - 1.0), 0.10, (donor, receiver, b3, q3))
        for mode in (AITKEN, ACCUM, PCARBON):
            self.assertLess(abs(_self_coefficient(mode) / _intra_quadrature(mode) - 1.0),
                            0.10, mode)

    def test_third_moment_coefficient_survives_float32(self):
        """Regression: the d³-scaled limits underflowed float32 to zero."""
        _, b3 = _coefficients(AITKEN, ACCUM)
        self.assertGreater(b3, 1e-16)
        self.assertLess(b3, 1e-13)

    def test_coefficients_positive_and_finite_across_the_size_range(self):
        for d_i in (5e-9, 50e-9, 500e-9):
            for d_j in (5e-9, 50e-9, 500e-9, 5e-6):
                b0, b3 = _coefficients((d_i, 1770.0, 1.6), (d_j, 1770.0, 1.8))
                self.assertTrue(math.isfinite(b0) and math.isfinite(b3))
                self.assertGreater(b0, 0.0)
                self.assertGreater(b3, 0.0)

    def test_larger_receiver_coagulates_faster_with_small_donor(self):
        b_small, _ = _coefficients(AITKEN, (50e-9, 1770.0, 1.8))
        b_large, _ = _coefficients(AITKEN, (500e-9, 1770.0, 1.8))
        self.assertGreater(b_large, b_small)


def _integrate(number, inter_rate, self_coefficient, dt, steps=20000):
    """Explicit fine-step reference for dN/dt = −(a N + b N²)."""
    n = number
    h = dt / steps
    for _ in range(steps):
        n = n - h * (inter_rate * n + self_coefficient * n * n)
    return n


class NumberSolutionTest(unittest.TestCase):

    def test_receiver_solution_is_exact(self):
        n0, beta = 1e9, 1e-15
        exact = n0 / (1.0 + DT * beta * n0)
        self.assertAlmostEqual(float(cg.receiver_number(jnp.asarray(n0), beta, DT)) / exact,
                               1.0, places=6)
        self.assertAlmostEqual(_integrate(n0, 0.0, beta, DT) / exact, 1.0, places=4)

    def test_donor_solution_in_each_regime(self):
        cases = {
            "small_total": (1e8, 1e-6, 1e-15),       # a + bN ≈ 0.002
            "small_inter": (1e10, 1e-7, 1e-15),      # a = 1.8e-4, bN = 0.018
            "general": (1e10, 1e-4, 1e-15),          # a = 0.18, bN = 0.018
            "strong": (1e11, 1e-3, 1e-14),           # a = 1.8, bN = 1.8
        }
        for name, (n0, inter, beta) in cases.items():
            got = float(cg.donor_number(jnp.asarray(n0), inter, beta, DT))
            ref = _integrate(n0, inter, beta, DT)
            self.assertLess(abs(got / ref - 1.0), 2e-4, (name, got, ref))

    def test_donor_solution_is_continuous_across_the_switches(self):
        n0 = 1e9
        for a_target, b_target in ((0.01, 0.0), (0.0, 0.01), (0.001, 0.05)):
            for sign in (-1.0, 1.0):
                lo = cg.donor_number(jnp.asarray(n0), (a_target + sign * 1e-6) / DT,
                                     b_target / DT / n0, DT)
                hi = cg.donor_number(jnp.asarray(n0), (a_target - sign * 1e-6) / DT,
                                     b_target / DT / n0, DT)
                self.assertAlmostEqual(float(lo) / float(hi), 1.0, places=4)

    def test_donor_solution_never_exceeds_the_start_or_goes_negative(self):
        n0 = jnp.asarray([0.0, 1e3, 1e9, 1e12])
        out = cg.donor_number(n0, 5e-4, 1e-14, DT)
        self.assertTrue(bool(jnp.all(out <= n0)))
        self.assertTrue(bool(jnp.all(out >= 0.0)))


class ProcessTest(unittest.TestCase):

    def _run(self, population):
        return cg.coagulate(population, _env(population), MAM4_SPEC, None)

    def test_mass_per_species_is_conserved_and_coarse_untouched(self):
        before = _population()
        after = self._run(before)
        for sp in {s for m in MAM4_SPEC.modes for s in m.species}:
            total_before = sum(before.masses[mass_name(sp, m.short)]
                               for m in MAM4_SPEC.modes if sp in m.species)
            total_after = sum(after.masses[mass_name(sp, m.short)]
                              for m in MAM4_SPEC.modes if sp in m.species)
            np.testing.assert_allclose(np.asarray(total_after), np.asarray(total_before),
                                       rtol=1e-5)
        for sp in MAM4_SPEC.mode("coarse").species:
            np.testing.assert_array_equal(np.asarray(after.masses[mass_name(sp, "cor")]),
                                          np.asarray(before.masses[mass_name(sp, "cor")]))
        np.testing.assert_array_equal(np.asarray(after.numbers[number_name("cor")]),
                                      np.asarray(before.numbers[number_name("cor")]))
        for key in before.gases:
            np.testing.assert_array_equal(np.asarray(after.gases[key]),
                                          np.asarray(before.gases[key]))

    def test_number_only_ever_decreases(self):
        before = _population()
        after = self._run(before)
        for short in ("ait", "acc", "pcm"):
            self.assertTrue(bool(jnp.all(after.numbers[number_name(short)]
                                         < before.numbers[number_name(short)])))

    def test_aitken_mass_moves_to_accumulation_only(self):
        before = _population()
        after = self._run(before)
        for sp in MAM4_SPEC.mode("aitken").species:
            self.assertTrue(bool(jnp.all(after.masses[mass_name(sp, "ait")]
                                         < before.masses[mass_name(sp, "ait")])))
            self.assertTrue(bool(jnp.all(after.masses[mass_name(sp, "acc")]
                                         > before.masses[mass_name(sp, "acc")])))
        for sp in MAM4_SPEC.mode("primary_carbon").species:
            self.assertTrue(bool(jnp.all(after.masses[mass_name(sp, "pcm")]
                                         < before.masses[mass_name(sp, "pcm")])))

    def test_coating_records_the_primary_carbon_share_of_aitken_sulfate_and_soa(self):
        before = _population()
        after = self._run(before)
        share = after.diagnostics["coagulation_aitken_share_to_pcm"]
        fraction = after.diagnostics["coagulation_aitken_mass_fraction"]
        self.assertTrue(bool(jnp.all((share > 0.0) & (share < 1.0))))
        for sp in ("so4", "soa"):
            expected = before.masses[mass_name(sp, "ait")] * fraction * share
            np.testing.assert_allclose(np.asarray(after.coating[sp]), np.asarray(expected),
                                       rtol=1e-5)
        self.assertEqual(set(after.coating), {"so4", "soa"})

    def test_no_primary_carbon_means_no_coating_and_no_pcm_share(self):
        before = _population(empty=("pcm",))
        after = self._run(before)
        self.assertTrue(bool(jnp.all(after.diagnostics["coagulation_aitken_share_to_pcm"] == 0.0)))
        self.assertTrue(bool(jnp.all(after.coating["so4"] == 0.0)))
        # Aitken still coagulates with accumulation.
        self.assertTrue(bool(jnp.all(after.numbers[number_name("ait")]
                                     < before.numbers[number_name("ait")])))

    def test_empty_population_is_a_fixed_point(self):
        before = _population(empty_modes=True)
        after = self._run(before)
        for key in before.numbers:
            np.testing.assert_array_equal(np.asarray(after.numbers[key]),
                                          np.asarray(before.numbers[key]))
        for key in before.masses:
            np.testing.assert_array_equal(np.asarray(after.masses[key]),
                                          np.asarray(before.masses[key]))

    def test_step_fractions_are_bounded(self):
        after = self._run(_population())
        for key in ("coagulation_aitken_mass_fraction", "coagulation_pcm_mass_fraction"):
            value = after.diagnostics[key]
            self.assertTrue(bool(jnp.all((value >= 0.0) & (value < 1.0))))

    def test_gradient_finite_in_float32_at_empty_modes(self):
        population = _population(empty_modes=True)
        env = _env(population)
        target = number_name("ait")

        def accumulation_number(n_ait):
            pop = ModalPopulation(masses=population.masses,
                                  numbers={**population.numbers, target: n_ait},
                                  gases=population.gases)
            out = cg.coagulate(pop, env, MAM4_SPEC, None)
            return jnp.sum(out.numbers[number_name("acc")] + out.numbers[target])

        grad = jax.grad(accumulation_number)(population.numbers[target])
        self.assertTrue(np.all(np.isfinite(np.asarray(grad))))

    def test_gradient_through_a_populated_step_is_finite_and_nontrivial(self):
        population = _population()
        env = _env(population)
        target = number_name("acc")

        def aitken_number(n_acc):
            pop = ModalPopulation(masses=population.masses,
                                  numbers={**population.numbers, target: n_acc},
                                  gases=population.gases)
            return jnp.sum(cg.coagulate(pop, env, MAM4_SPEC, None).numbers[number_name("ait")])

        grad = np.asarray(jax.grad(aitken_number)(population.numbers[target]))
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertTrue(np.all(grad < 0.0))      # more accumulation, fewer Aitken


if __name__ == "__main__":
    unittest.main()
