"""AER-03: gas-aerosol exchange, checked against the physics it claims.

Each MAM formula is pinned against something independent of itself: the
Fuchs-Sutugin limits against the free-molecular and continuum laws, the
two-point quadrature against brute-force integration of the same integrand
over the log-normal density, the H2SO4 solve against its own analytic
form and against sulfur-mole conservation, the SOAG solve against
equilibrium (no net flux) and against total-SOA conservation. The
sub-step scan is checked for independence from ``max_substeps`` once the
adaptive count fits. Gradients are checked where a float32 NaN was found
and fixed.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
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
M_H2SO4 = GAS_SPECIES["h2so4"].molar_mass
M_SOAG = GAS_SPECIES["soag"].molar_mass
M_SO4 = MAM4_SPEC.species_props("so4").molar_mass
ACC = MAM4_SPEC.mode("accum")


def _population(seed=0, *, gases=None, empty=()):
    rng = np.random.default_rng(seed)
    masses = {mass_name(sp, m.short): jnp.asarray(rng.uniform(1e-12, 1e-9, SHAPE))
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.asarray(rng.uniform(1e6, 1e9, SHAPE))
               for m in MAM4_SPEC.modes}
    by_short = {m.short: m for m in MAM4_SPEC.modes}
    for short in empty:
        for sp in by_short[short].species:
            masses[mass_name(sp, short)] = jnp.zeros(SHAPE)
        numbers[number_name(short)] = jnp.zeros(SHAPE)
    gases = dict(gases or {})
    gases.setdefault(gas_name("h2so4"), jnp.full(SHAPE, 1.0e-12))
    gases.setdefault(gas_name("soag"), jnp.full(SHAPE, 1.0e-10))
    return ModalPopulation(masses=masses, numbers=numbers, gases=gases)


def _env(population, *, production=0.0, temperature=T):
    temp = jnp.full(SHAPE, temperature)
    pressure = jnp.full(SHAPE, P)
    sat = saturation_ratio(temp, jnp.full(SHAPE, Q), pressure)
    return MicrophysicsEnvironment(
        temperature=temp, pressure=pressure, saturation=sat,
        air_density=pressure / (c.rd * temp),
        state_start=equilibrium_modal_state(
            population.masses, population.numbers, MAM4_SPEC, sat),
        h2so4_production=jnp.full(SHAPE, production),
        height_above_surface=jnp.full(SHAPE, 500.0),
        pbl_height=jnp.zeros(SHAPE[1:]), dt=DT)


def _sulfur_moles(population):
    """Moles of S per kg air: gas H2SO4 plus every sulfate tracer."""
    total = population.gases[gas_name("h2so4")] / M_H2SO4
    for mode in MAM4_SPEC.modes:
        key = mass_name("so4", mode.short)
        if key in population.masses:
            total = total + population.masses[key] / M_SO4
    return total


def _organic_mass(population):
    """Sum SOAG gas and every SOA tracer [kg/kg]; conserved 1:1 by mass."""
    total = population.gases[gas_name("soag")]
    for mode in MAM4_SPEC.modes:
        key = mass_name("soa", mode.short)
        if key in population.masses:
            total = total + population.masses[key]
    return total


class FuchsSutuginTest(unittest.TestCase):

    def test_continuum_limit_is_one(self):
        self.assertAlmostEqual(float(cond.fuchs_sutugin(jnp.asarray(0.0), 0.65)),
                               1.0, places=6)

    def test_free_molecular_limit(self):
        # f -> 0.75 alpha / Kn as Kn -> inf.
        kn = jnp.asarray(1.0e4)
        self.assertAlmostEqual(
            float(cond.fuchs_sutugin(kn, 0.65) * kn / (0.75 * 0.65)), 1.0, places=3)

    def test_beta_spans_one_to_two(self):
        self.assertAlmostEqual(float(cond._log_slope_beta(jnp.asarray(1e-6), 0.65)),
                               1.0, places=5)
        self.assertAlmostEqual(float(cond._log_slope_beta(jnp.asarray(1e6), 0.65)),
                               2.0, places=3)

    def test_fuller_diffusivity_matches_the_laboratory_order(self):
        # Poschl et al. (1998): 0.088 cm^2/s at 303 K; Hanson & Eisele
        # (2000): 0.094 at 298 K. Fuller must land in that neighbourhood.
        d = float(cond.gas_diffusivity(jnp.asarray(298.0), jnp.asarray(101325.0),
                                       M_H2SO4, cond.DIFFUSION_VOLUME_H2SO4))
        self.assertGreater(d, 0.07e-4)
        self.assertLess(d, 0.12e-4)


class QuadratureTest(unittest.TestCase):
    """Two-point Gauss-Hermite against brute-force integration."""

    def _brute(self, number_density, dg, sigma, diffusivity, mfp, alpha, n=40001):
        ln_sigma = math.log(sigma)
        x = np.linspace(math.log(dg) - 8 * ln_sigma, math.log(dg) + 8 * ln_sigma, n)
        density = (number_density / (math.sqrt(2 * math.pi) * ln_sigma)
                   * np.exp(-0.5 * ((x - math.log(dg)) / ln_sigma) ** 2))
        d = np.exp(x)
        f = np.asarray(cond.fuchs_sutugin(jnp.asarray(2 * mfp / d), alpha))
        return 2 * math.pi * diffusivity * np.trapezoid(f * d * density, x)

    def test_matches_brute_force_across_regimes(self):
        diffusivity = float(cond.gas_diffusivity(jnp.asarray(T), jnp.asarray(P),
                                                 M_H2SO4, cond.DIFFUSION_VOLUME_H2SO4))
        mfp = 3 * diffusivity / float(cond.mean_molecular_speed(jnp.asarray(T), M_H2SO4))
        for mode in MAM4_SPEC.modes:
            for dg in (mode.dgnum_lo, mode.dgnum, mode.dgnum_hi):
                coefficient = float(cond.mass_transfer_coefficient(
                    jnp.asarray(1.0e9), jnp.asarray(dg), math.log(mode.geom_std_dev),
                    jnp.asarray(diffusivity), jnp.asarray(mfp), 0.65))
                brute = self._brute(1.0e9, dg, mode.geom_std_dev, diffusivity, mfp, 0.65)
                # MAM's own verification page accepts percent-level error
                # for the two-point rule; 3 % is a generous bound.
                self.assertAlmostEqual(coefficient / brute, 1.0, delta=0.03,
                                       msg=f"{mode.name} dg={dg:.2e}")

    def test_continuum_limit_recovers_first_moment(self):
        # Huge particles (Kn -> 0): C = 2 pi D M_1 with f == 1 exactly.
        mode = ACC
        number, dg = 1.0e9, 50.0e-6
        diffusivity, mfp = 1.0e-5, 1.0e-7
        coefficient = float(cond.mass_transfer_coefficient(
            jnp.asarray(number), jnp.asarray(dg), math.log(mode.geom_std_dev),
            jnp.asarray(diffusivity), jnp.asarray(mfp), 0.65))
        m1 = number * dg * math.exp(0.5 * math.log(mode.geom_std_dev) ** 2)
        self.assertAlmostEqual(coefficient / (2 * math.pi * diffusivity * m1),
                               1.0, places=2)


class SulfuricAcidTest(unittest.TestCase):

    def test_analytic_regime_matches_closed_form(self):
        q, p, dt = jnp.asarray(5e-12), jnp.asarray(0.0), 1800.0
        coefficients = jnp.asarray([2e-3, 1e-3])        # two modes
        condensed, fraction, mean, _ = cond.solve_h2so4(q, p, coefficients, dt)
        total = 3e-3
        # Exact time-mean of q0 e^{-total t} over the step.
        self.assertAlmostEqual(
            float(mean) / (5e-12 * (1 - math.exp(-total * dt)) / (total * dt)),
            1.0, places=4)
        self.assertAlmostEqual(float(condensed) / (5e-12 * (1 - math.exp(-total * dt))),
                               1.0, places=5)
        np.testing.assert_allclose(np.asarray(fraction), [2 / 3, 1 / 3], rtol=1e-6)

    def test_production_is_coupled_not_split(self):
        # With no initial gas, everything that is produced and condenses
        # must come from P*: the sequential split would give zero.
        q, p, dt = jnp.asarray(0.0), jnp.asarray(1e-15), 1800.0
        condensed, _, _, _ = cond.solve_h2so4(q, p, jnp.asarray([2e-3]), dt)
        expected = p * dt - (p / 2e-3) * (1 - math.exp(-2e-3 * dt))
        self.assertAlmostEqual(float(condensed) / float(expected), 1.0, places=5)

    def test_regimes_are_continuous_at_their_thresholds(self):
        q, p, dt = jnp.asarray(1e-11), jnp.asarray(1e-16), 1.0
        for tau in (cond._ANALYTIC_ABOVE, cond._NEGLECT_BELOW):
            below = cond.solve_h2so4(q, p, jnp.asarray([tau * 0.999]), dt)[0]
            above = cond.solve_h2so4(q, p, jnp.asarray([tau * 1.001]), dt)[0]
            # Both sides of each switch agree to the Taylor truncation.
            np.testing.assert_allclose(float(below), float(above),
                                       rtol=1e-2, atol=1e-30)

    def test_sulfur_moles_are_conserved(self):
        population = _population()
        out = cond.condense(population, _env(population), MAM4_SPEC,
                            cond.CondensationParameters())
        np.testing.assert_allclose(np.asarray(_sulfur_moles(out)),
                                   np.asarray(_sulfur_moles(population)),
                                   rtol=1e-5)

    def test_gas_only_decreases_and_sulfate_only_grows(self):
        population = _population()
        out = cond.condense(population, _env(population), MAM4_SPEC,
                            cond.CondensationParameters())
        self.assertTrue(bool(jnp.all(out.gases[gas_name("h2so4")]
                                     <= population.gases[gas_name("h2so4")])))
        # Never negative, even at full depletion: a -1.6e-17 residual from
        # a 1e-10 input is what the subtractive update used to leave.
        self.assertTrue(bool(jnp.all(out.gases[gas_name("h2so4")] >= 0.0)))
        for mode in MAM4_SPEC.modes:
            key = mass_name("so4", mode.short)
            if key in out.masses:
                self.assertTrue(bool(jnp.all(out.masses[key] >= population.masses[key])),
                                msg=key)


class SoagTest(unittest.TestCase):

    def test_no_net_flux_at_equilibrium(self):
        # Set the gas to each mode's Raoult equilibrium with one mode only.
        population = _population()
        env = _env(population)
        params = cond.CondensationParameters()
        q_equ = cond.soag_equilibrium(env.temperature, env.pressure, params)
        # Single mode, so the surface equilibrium is the one value.
        soa = jnp.full(SHAPE, 1e-10) * cond.M_AIR / M_SOAG
        opoa = jnp.full(SHAPE, 2e-10) * cond.M_AIR / M_SOAG * 0.1
        sat = soa / (opoa + soa) * q_equ
        gas_new, soa_new, remaining = cond.solve_soag(
            sat, soa[jnp.newaxis], opoa[jnp.newaxis],
            jnp.full((1,) + SHAPE, 1e-3), q_equ, DT, params)
        self.assertTrue(bool(jnp.all(remaining == 0.0)))
        np.testing.assert_allclose(np.asarray(gas_new), np.asarray(sat), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(soa_new[0]), np.asarray(soa), rtol=1e-5)

    def test_total_organic_mass_is_conserved(self):
        population = _population()
        out = cond.condense(population, _env(population), MAM4_SPEC,
                            cond.CondensationParameters())
        np.testing.assert_allclose(np.asarray(_organic_mass(out)),
                                   np.asarray(_organic_mass(population)),
                                   rtol=1e-5)

    def test_evaporation_cannot_overdraw(self):
        # Zero gas, SOA present: net evaporation into the gas. A mode may
        # still GAIN SOA from another whose equilibrium is higher (more
        # oxygenated POA solvent lowers a mode's saturation), so the
        # invariants are non-negativity and a non-increasing total, not a
        # per-mode decrease.
        population = _population(gases={gas_name("soag"): jnp.zeros(SHAPE)})
        out = cond.condense(population, _env(population), MAM4_SPEC,
                            cond.CondensationParameters())
        total_before = total_after = jnp.zeros(SHAPE)
        for mode in MAM4_SPEC.modes:
            key = mass_name("soa", mode.short)
            if key in out.masses:
                self.assertTrue(bool(jnp.all(out.masses[key] >= 0.0)), msg=key)
                total_before = total_before + population.masses[key]
                total_after = total_after + out.masses[key]
        self.assertTrue(bool(jnp.all(total_after <= total_before * (1 + 1e-6))))
        self.assertTrue(bool(jnp.all(out.gases[gas_name("soag")] >= 0.0)))

    def test_result_is_independent_of_the_scan_length_once_it_fits(self):
        """Measured 2026-10-01: this fixture converges between 200 and 400.

        At the default the shortfall must be zero and doubling the scan
        must change nothing; at 10 the cap binds and the result is
        materially different, which is the trap the diagnostic exists for.
        """
        population = _population()
        env = _env(population)
        default = cond.condense(population, env, MAM4_SPEC,
                                cond.CondensationParameters())
        self.assertTrue(bool(jnp.all(
            default.diagnostics["condensation_soag_substep_shortfall"] == 0.0)))
        doubled = cond.condense(population, env, MAM4_SPEC,
                                cond.CondensationParameters(max_substeps=800))
        for key in default.masses:
            np.testing.assert_array_equal(np.asarray(default.masses[key]),
                                          np.asarray(doubled.masses[key]),
                                          err_msg=key)
        capped = cond.condense(population, env, MAM4_SPEC,
                               cond.CondensationParameters(max_substeps=10))
        self.assertTrue(bool(jnp.all(
            capped.diagnostics["condensation_soag_substep_shortfall"] > 0.0)))

    def test_equilibrium_mixing_ratio_has_the_documented_reference(self):
        params = cond.CondensationParameters()
        value = float(cond.soag_equilibrium(jnp.asarray(298.0),
                                            jnp.asarray(101325.0), params))
        self.assertAlmostEqual(value / 1.0e-10, 1.0, places=6)


class PrimaryCarbonRoutingTest(unittest.TestCase):

    def test_pcm_condensate_lands_on_accumulation_and_is_recorded(self):
        population = _population()
        out = cond.condense(population, _env(population), MAM4_SPEC,
                            cond.CondensationParameters())
        # pcm has no so4/soa slot, so nothing should have been created there.
        self.assertNotIn(mass_name("so4", "pcm"), out.masses)
        # The coating is what the primary-carbon mode took up.
        self.assertIn("so4", out.coating)
        self.assertTrue(bool(jnp.all(out.coating["so4"] > 0.0)))
        # Accumulation gained at least its own share plus pcm's.
        own_and_pcm = out.masses[mass_name("so4", "acc")] - population.masses[mass_name("so4", "acc")]
        self.assertTrue(bool(jnp.all(own_and_pcm >= out.coating["so4"])))

    def test_depletion_diagnostic_is_a_fraction(self):
        population = _population()
        out = cond.condense(population, _env(population), MAM4_SPEC,
                            cond.CondensationParameters())
        d = out.diagnostics["condensation_h2so4_depletion"]
        self.assertTrue(bool(jnp.all((d >= 0.0) & (d <= 1.0))))


class DifferentiabilityTest(unittest.TestCase):

    def test_gradient_finite_with_no_gas_and_an_empty_mode_in_float32(self):
        """The float32 NaN found 2026-10-01, pinned fixed."""
        population = _population(
            gases={gas_name("h2so4"): jnp.zeros(SHAPE),
                   gas_name("soag"): jnp.zeros(SHAPE)},
            empty=("acc",))
        env = _env(population)
        target = mass_name("so4", "ait")

        def sulfate_on_coarse(mass):
            masses = {**population.masses, target: mass}
            out = cond.condense(
                ModalPopulation(masses=masses, numbers=population.numbers,
                                gases=population.gases),
                env, MAM4_SPEC, cond.CondensationParameters())
            return jnp.sum(out.masses[mass_name("so4", "cor")])

        grad = jax.grad(sulfate_on_coarse)(population.masses[target])
        self.assertTrue(np.all(np.isfinite(np.asarray(grad))))

    def test_gradients_match_a_central_difference(self):
        population = _population()
        env = _env(population)
        params = cond.CondensationParameters()

        # Differentiate the condensed INCREMENT with respect to an O(1)
        # multiplier on the gas. Two float32 traps otherwise: a central
        # difference on the 1e-12 kg/kg mixing ratio itself cannot resolve a
        # relative step of 1e-6 (the input is unchanged after rounding), and
        # the pre-existing ~1e-9 sulfate swamps a ~1e-14 increment at
        # float32 resolution. Both report a zero reference against a
        # correct, nonzero AD value.
        # The cancellation happens INSIDE condense: a ~1e-14 increment added
        # to a ~1e-9 base is lost below the base's float32 ulp before any
        # test can subtract it back out. So the fixture starts the
        # accumulation mode with no sulfate, and the output IS the increment.
        masses = {**population.masses, mass_name("so4", "acc"): jnp.zeros(SHAPE)}
        base = population.gases[gas_name("h2so4")]

        def condensed_increment(multiplier):
            gases = {**population.gases, gas_name("h2so4"): base * multiplier}
            out = cond.condense(
                ModalPopulation(masses=masses, numbers=population.numbers,
                                gases=gases), env, MAM4_SPEC, params)
            return out.masses[mass_name("so4", "acc")] * 1.0e13

        check_gradients(condensed_increment, (jnp.asarray(1.0),), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
