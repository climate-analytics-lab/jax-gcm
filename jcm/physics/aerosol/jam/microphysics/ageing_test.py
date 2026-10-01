"""AER-06: primary-carbon ageing, pinned on the criterion and conservation.

The criterion is a closed form, so it is checked against a hand-built
monodisperse case where the shell thickness is known; the process is
checked for conservation of every species and of number, for the
direction of the transfer, for the coating being consumed, for the empty
and uncoated cases, and for finite float32 gradients — including through
the monolayer count, which is the one tunable.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c

from jcm.physics.aerosol.jam.microphysics import ageing
from jcm.physics.aerosol.jam.microphysics import coagulation as cg
from jcm.physics.aerosol.jam.microphysics import condensation as cond
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
from jcm.testing import check_gradients

SHAPE = (3, 2)
T, P, Q, DT = 285.0, 9.0e4, 0.005, 1800.0
PCM = MAM4_SPEC.mode("primary_carbon")


def _population(seed=0, *, coating=None, empty_pcm=False):
    rng = np.random.default_rng(seed)
    masses = {mass_name(sp, m.short): jnp.asarray(rng.uniform(1e-12, 1e-9, SHAPE))
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.asarray(rng.uniform(1e6, 1e9, SHAPE))
               for m in MAM4_SPEC.modes}
    if empty_pcm:
        numbers[number_name("pcm")] = jnp.zeros(SHAPE)
        for sp in PCM.species:
            masses[mass_name(sp, "pcm")] = jnp.zeros(SHAPE)
    gases = {gas_name("h2so4"): jnp.full(SHAPE, 1.0e-10),
             gas_name("soag"): jnp.full(SHAPE, 1.0e-10)}
    return ModalPopulation(masses=masses, numbers=numbers, gases=gases,
                           coating={} if coating is None else coating)


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


def _age(population, params=None):
    return ageing.age(population, _env(population), MAM4_SPEC,
                      ageing.AgeingParameters() if params is None else params)


class CriterionTest(unittest.TestCase):

    def test_shell_equivalence_is_the_hygroscopicity_ratio(self):
        factors = ageing.shell_equivalence(MAM4_SPEC)
        self.assertEqual(set(factors), {"so4", "soa"})
        self.assertEqual(factors["so4"], 1.0)
        self.assertAlmostEqual(factors["soa"], 0.140 / 0.507, places=9)

    def test_fraction_is_shell_thickness_over_n_monolayers(self):
        """Monodisperse check: for σ_g → 1 a shell of volume V on N
        spheres of diameter D has thickness V/(N π D²), and the fraction
        is that over n δ. The log-normal factor exp(2.5 ln²σ) is 1 there.
        """
        n, d, rho = 1.0e9, 50.0e-9, 1500.0
        core_volume = n * math.pi / 6.0 * d ** 3
        for n_layers, thickness_in_layers in ((8.0, 2.0), (3.0, 0.5), (8.0, 20.0)):
            shell_volume = thickness_in_layers * n_layers * ageing.MONOLAYER_THICKNESS \
                * n * math.pi * d ** 2
            fraction = float(ageing.aged_fraction(
                jnp.asarray(shell_volume), jnp.asarray(core_volume), jnp.asarray(d),
                1.0, n_layers))
            expected = min(thickness_in_layers, 1.0)
            self.assertAlmostEqual(fraction, expected, places=5)
        _ = rho

    def test_lognormal_surface_factor_matches_the_moments(self):
        """6 e^{−2.5 ln²σ}/D_g is the mode's surface-to-volume ratio, so the
        fraction must equal shell / (surface · n δ) computed from M₂ and M₃.
        """
        n, d, sigma, n_layers = 1.0e9, 50.0e-9, 1.6, 8.0
        ln2 = math.log(sigma) ** 2
        surface = math.pi * n * d ** 2 * math.exp(2.0 * ln2)
        volume = math.pi / 6.0 * n * d ** 3 * math.exp(4.5 * ln2)
        shell = 0.3 * n_layers * ageing.MONOLAYER_THICKNESS * surface
        fraction = float(ageing.aged_fraction(
            jnp.asarray(shell), jnp.asarray(volume), jnp.asarray(d), sigma, n_layers))
        self.assertAlmostEqual(fraction, 0.3, places=5)

    def test_fraction_is_capped_just_below_one_and_zero_without_core(self):
        capped = float(ageing.aged_fraction(jnp.asarray(1e-12), jnp.asarray(1e-15),
                                            jnp.asarray(50e-9), 1.6, 8.0))
        self.assertLess(capped, 1.0)
        self.assertGreater(capped, 1.0 - 20.0 * jnp.finfo(jnp.float32).eps)
        # Below the core floor the mode is empty for the criterion.
        self.assertEqual(float(ageing.aged_fraction(
            jnp.asarray(1.0), jnp.asarray(1e-30), jnp.asarray(50e-9), 1.6, 8.0)), 0.0)
        self.assertEqual(float(ageing.aged_fraction(
            jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(50e-9), 1.6, 8.0)), 0.0)

    def test_gradient_through_the_criterion(self):
        shell = jnp.asarray(2.0e-15)
        core = jnp.asarray(1.0e-13)
        check_gradients(
            lambda s, k, n: ageing.aged_fraction(s, k, jnp.asarray(50e-9), 1.6, n),
            (shell, core, jnp.asarray(8.0)), rtol=1e-3)


class ProcessTest(unittest.TestCase):

    def _coated(self, so4=2.0e-12, soa=1.0e-12):
        return _population(coating={"so4": jnp.full(SHAPE, so4),
                                    "soa": jnp.full(SHAPE, soa)})

    def test_conserves_every_species_and_number(self):
        before = self._coated()
        after = _age(before)
        for sp in {s for m in MAM4_SPEC.modes for s in m.species}:
            total_before = sum(before.masses[mass_name(sp, m.short)]
                               for m in MAM4_SPEC.modes if sp in m.species)
            total_after = sum(after.masses[mass_name(sp, m.short)]
                              for m in MAM4_SPEC.modes if sp in m.species)
            np.testing.assert_allclose(np.asarray(total_after), np.asarray(total_before),
                                       rtol=1e-6)
        total_n = sum(before.numbers.values())
        np.testing.assert_allclose(np.asarray(sum(after.numbers.values())),
                                   np.asarray(total_n), rtol=1e-6)

    def test_moves_primary_carbon_to_accumulation_by_one_fraction(self):
        before = self._coated()
        after = _age(before)
        fraction = after.diagnostics["ageing_fraction"]
        self.assertTrue(bool(jnp.all((fraction > 0.0) & (fraction < 1.0))))
        for sp in PCM.species:
            np.testing.assert_allclose(
                np.asarray(after.masses[mass_name(sp, "pcm")]),
                np.asarray(before.masses[mass_name(sp, "pcm")] * (1.0 - fraction)),
                rtol=1e-6)
            # The gain is ~1e-3 of the accumulation mass, so compare it at
            # the receiver's float32 resolution, not its own.
            np.testing.assert_allclose(
                np.asarray(after.masses[mass_name(sp, "acc")]
                           - before.masses[mass_name(sp, "acc")]),
                np.asarray(before.masses[mass_name(sp, "pcm")] * fraction),
                atol=float(jnp.max(before.masses[mass_name(sp, "acc")])) * 1e-6)
        np.testing.assert_allclose(
            np.asarray(after.numbers[number_name("pcm")]),
            np.asarray(before.numbers[number_name("pcm")] * (1.0 - fraction)), rtol=1e-6)
        for short in ("ait", "cor"):
            np.testing.assert_array_equal(np.asarray(after.numbers[number_name(short)]),
                                          np.asarray(before.numbers[number_name(short)]))

    def test_consumes_the_coating(self):
        after = _age(self._coated())
        self.assertEqual(after.coating, {})

    def test_no_coating_means_no_ageing(self):
        before = _population()
        after = _age(before)
        self.assertTrue(bool(jnp.all(after.diagnostics["ageing_fraction"] == 0.0)))
        for key in before.masses:
            np.testing.assert_array_equal(np.asarray(after.masses[key]),
                                          np.asarray(before.masses[key]))

    def test_thicker_shell_and_fewer_monolayers_age_more(self):
        thin = _age(self._coated(so4=1e-12, soa=0.0)).diagnostics["ageing_fraction"]
        thick = _age(self._coated(so4=4e-12, soa=0.0)).diagnostics["ageing_fraction"]
        self.assertTrue(bool(jnp.all(thick > thin)))
        np.testing.assert_allclose(np.asarray(thick / thin), 4.0, rtol=1e-5)
        three = _age(self._coated(so4=1e-12, soa=0.0),
                     ageing.AgeingParameters(n_so4_monolayers=3.0)).diagnostics["ageing_fraction"]
        np.testing.assert_allclose(np.asarray(three / thin), 8.0 / 3.0, rtol=1e-5)

    def test_soa_counts_at_its_hygroscopicity_equivalent(self):
        so4_only = _age(self._coated(so4=1e-12, soa=0.0)).diagnostics["ageing_fraction"]
        soa_only = _age(self._coated(so4=0.0, soa=1e-12)).diagnostics["ageing_fraction"]
        # Same mass, same density (1770 vs 1000 kg/m³ differ, so compare volumes):
        expected = (0.140 / 0.507) * (1770.0 / 1000.0)
        np.testing.assert_allclose(np.asarray(soa_only / so4_only), expected, rtol=1e-5)

    def test_empty_primary_carbon_with_a_deposit_is_finite_and_moves_nothing(self):
        before = _population(empty_pcm=True,
                             coating={"so4": jnp.full(SHAPE, 1e-12)})
        after = _age(before)
        self.assertTrue(bool(jnp.all(after.diagnostics["ageing_fraction"] == 0.0)))
        self.assertTrue(bool(jnp.all(after.numbers[number_name("acc")]
                                     == before.numbers[number_name("acc")])))

    def test_end_to_end_with_the_depositing_processes(self):
        """Condensation and coagulation write the coating; ageing reads it."""
        population = _population()
        env = _env(population)
        population = cond.condense(population, env, MAM4_SPEC, cond.CondensationParameters())
        population = cg.coagulate(population, env, MAM4_SPEC, None)
        self.assertTrue(bool(jnp.all(population.coating["so4"] > 0.0)))
        aged = ageing.age(population, env, MAM4_SPEC, ageing.AgeingParameters())
        self.assertTrue(bool(jnp.all(aged.diagnostics["ageing_fraction"] > 0.0)))
        self.assertTrue(bool(jnp.all(aged.numbers[number_name("pcm")]
                                     < population.numbers[number_name("pcm")])))

    def test_gradients_finite_in_float32_including_the_monolayer_count(self):
        for empty in (False, True):
            population = _population(empty_pcm=empty,
                                     coating={"so4": jnp.full(SHAPE, 1e-12)})
            env = _env(population)

            def accumulation_number(n_layers, pop=population, env=env):
                out = ageing.age(pop, env, MAM4_SPEC,
                                 ageing.AgeingParameters(n_so4_monolayers=n_layers))
                return jnp.sum(out.numbers[number_name("acc")])

            grad = jax.grad(accumulation_number)(jnp.asarray(8.0))
            self.assertTrue(bool(jnp.isfinite(grad)))
            if not empty:
                self.assertLess(float(grad), 0.0)     # more layers needed, less ageing


if __name__ == "__main__":
    unittest.main()
