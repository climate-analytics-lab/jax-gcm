"""AER-04: new-particle formation, pinned on structure and conservation.

No numeric benchmark for the Vehkamäki et al. (2002) rates is published
outside the paper's figures (which MAM reproduces in its own tests), so the
binary pathway is pinned on what can be checked without them: the
documented validity clipping, monotonicity in vapour and humidity, and
critical-cluster sizes in the physical range. The empirical boundary-layer
pathway and the 1 nm cluster are exact formulae and are checked exactly.
The survival step is checked at its two limits (no sink: everything
survives; a strong sink: almost nothing does) and for the dimensional
consistency the module docstring relies on. The update is checked for
sulfur conservation, non-negativity of the vapour and the clipping rule.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics import condensation as cond
from jcm.physics.aerosol.jam.microphysics import new_particle_formation as npf
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
M_H2SO4 = GAS_SPECIES["h2so4"].molar_mass
M_SO4 = MAM4_SPEC.species_props("so4").molar_mass


def _population(seed=0, *, gas=1.0e-10, empty_modes=False):
    rng = np.random.default_rng(seed)
    masses = {mass_name(sp, m.short): jnp.asarray(rng.uniform(1e-12, 1e-9, SHAPE))
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.asarray(rng.uniform(1e6, 1e9, SHAPE))
               for m in MAM4_SPEC.modes}
    if empty_modes:
        numbers = {k: jnp.zeros(SHAPE) for k in numbers}
    gases = {gas_name("h2so4"): jnp.full(SHAPE, gas),
             gas_name("soag"): jnp.full(SHAPE, 1.0e-10)}
    return ModalPopulation(masses=masses, numbers=numbers, gases=gases)


def _env(population, *, height=50.0, pbl=800.0, temperature=T):
    temp = jnp.full(SHAPE, temperature)
    pressure = jnp.full(SHAPE, P)
    sat = saturation_ratio(temp, jnp.full(SHAPE, Q), pressure)
    return MicrophysicsEnvironment(
        temperature=temp, pressure=pressure, saturation=sat,
        air_density=pressure / (c.rd * temp),
        state_start=equilibrium_modal_state(
            population.masses, population.numbers, MAM4_SPEC, sat),
        h2so4_production=jnp.zeros(SHAPE),
        height_above_surface=jnp.full(SHAPE, height),
        pbl_height=jnp.full(SHAPE[1:], pbl), dt=DT)


def _after_exchange(population, env):
    """NPF reads the exchange step's coupling diagnostics, so run it first."""
    return cond.condense(population, env, MAM4_SPEC, cond.CondensationParameters())


def _sulfur_moles(population):
    total = population.gases[gas_name("h2so4")] / M_H2SO4
    for mode in MAM4_SPEC.modes:
        key = mass_name("so4", mode.short)
        if key in population.masses:
            total = total + population.masses[key] / M_SO4
    return total


class BinaryFormationTest(unittest.TestCase):

    def test_zero_below_the_concentration_bound(self):
        rate, _, _ = npf.binary_formation(jnp.asarray(250.0), jnp.asarray(0.5),
                                          jnp.asarray(5.0e3))
        self.assertEqual(float(rate), 0.0)

    def test_monotone_in_vapour_and_humidity(self):
        conc = jnp.asarray([1e6, 1e7, 1e8, 1e9, 1e10])
        rates = np.asarray(npf.binary_formation(jnp.asarray(250.0),
                                                jnp.asarray(0.5), conc)[0])
        self.assertTrue(np.all(np.diff(rates) > 0))
        rh = jnp.asarray([0.2, 0.4, 0.6, 0.8])
        rates = np.asarray(npf.binary_formation(jnp.asarray(250.0), rh,
                                                jnp.asarray(1e8))[0])
        self.assertTrue(np.all(np.diff(rates) > 0))

    def test_cold_formation_dominates_warm(self):
        # Binary H2SO4-H2O formation is an upper-troposphere process.
        cold = npf.binary_formation(jnp.asarray(236.0), jnp.asarray(0.5), jnp.asarray(1e8))[0]
        warm = npf.binary_formation(jnp.asarray(285.0), jnp.asarray(0.5), jnp.asarray(1e8))[0]
        self.assertGreater(float(cold) / float(warm), 1.0e10)

    def test_critical_cluster_is_molecular_scale(self):
        for temp, rh, conc in ((236.0, 0.5, 1e8), (285.0, 0.5, 1e8), (298.0, 0.3, 1e9)):
            _, n, r = npf.binary_formation(jnp.asarray(temp), jnp.asarray(rh),
                                           jnp.asarray(conc))
            self.assertGreater(float(n), 1.0)
            self.assertLess(float(n), 200.0)
            self.assertGreater(float(r), 0.2e-9)
            self.assertLess(float(r), 3.0e-9)

    def test_out_of_range_inputs_are_clipped_not_extrapolated(self):
        inside = npf.binary_formation(jnp.asarray(230.15), jnp.asarray(0.5), jnp.asarray(1e8))[0]
        below = npf.binary_formation(jnp.asarray(200.0), jnp.asarray(0.5), jnp.asarray(1e8))[0]
        np.testing.assert_allclose(float(below), float(inside), rtol=1e-6)


class BoundaryLayerFormationTest(unittest.TestCase):

    def test_rate_is_first_order_with_the_documented_coefficient(self):
        rate, _, _ = npf.boundary_layer_formation(
            jnp.asarray(1.0e7), npf.NewParticleFormationParameters())
        self.assertAlmostEqual(float(rate), 10.0, places=5)

    def test_one_nanometre_pure_acid_cluster(self):
        _, n, r = npf.boundary_layer_formation(
            jnp.asarray(1.0e7), npf.NewParticleFormationParameters())
        expected = (npf.AVOGADRO * math.pi / 6.0 * 1e-27 * npf.RHO_H2SO4 / M_H2SO4)
        self.assertAlmostEqual(float(n) / expected, 1.0, places=5)
        self.assertAlmostEqual(float(n), 5.7, delta=0.2)
        self.assertAlmostEqual(float(r) / 0.5e-9, 1.0, places=6)


class SurvivalTest(unittest.TestCase):

    def _survive(self, uptake_sum, concentration=1e7):
        rate = jnp.asarray(10.0)
        _, n, r = npf.boundary_layer_formation(
            jnp.asarray(concentration), npf.NewParticleFormationParameters())
        air_moles = jnp.asarray(P / (c.r_universal * T))
        apparent, diameter = npf.survival_to_aitken(
            rate, n, r, jnp.asarray(T), jnp.asarray(0.5),
            jnp.asarray(concentration), jnp.asarray(uptake_sum), air_moles)
        return float(apparent) / 10.0, float(diameter)

    def test_everything_survives_with_no_sink(self):
        fraction, diameter = self._survive(0.0)
        self.assertAlmostEqual(fraction, 1.0, places=6)
        self.assertAlmostEqual(diameter / npf._AITKEN_LO, 1.0, places=6)

    def test_a_strong_sink_removes_nearly_everything(self):
        fraction, _ = self._survive(0.08)
        self.assertLess(fraction, 1.0e-6)
        self.assertGreaterEqual(fraction, 0.0)

    def test_survival_decreases_with_sink_and_increases_with_vapour(self):
        f_weak, _ = self._survive(1e-4)
        f_strong, _ = self._survive(1e-2)
        self.assertGreater(f_weak, f_strong)
        f_low, _ = self._survive(1e-3, concentration=1e6)
        f_high, _ = self._survive(1e-3, concentration=1e8)
        self.assertGreater(f_high, f_low)

    def test_eta_is_of_order_nanometres(self):
        """The unit reading the module docstring states, checked numerically.

        With GR from the ``3.0e-9`` constant read as nm/h, eta = gamma CS'/GR
        is in nm and eta/D is dimensionless. At a typical sink and vapour it
        must be O(1-100) nm — a wrong unit reading would be off by 3600 or
        1e9.
        """
        _, n, r = npf.boundary_layer_formation(
            jnp.asarray(1e7), npf.NewParticleFormationParameters())
        air_moles = P / (c.r_universal * T)
        f_v = 1.0 - 0.56 / math.log(0.5)
        rho_nuc = npf.RHO_H2SO4 / f_v
        d_ini, d_fin = 1.0, 1e9 * npf._AITKEN_LO * f_v ** (1 / 3)
        gamma = (0.23 * d_ini ** 0.2 * (d_fin / 3) ** 0.075
                 * (rho_nuc / 1000) ** -0.33 * (T / 293) ** -0.75)
        growth = 3e-9 * 14.7 * math.sqrt(T) * M_H2SO4 * 1e3 * 1e7 / rho_nuc
        diffusivity = 6.7037e-6 * T ** 0.75 / air_moles
        sink = 1e-3 / (4 * math.pi * diffusivity * 0.65)
        eta = gamma * sink / growth
        self.assertGreater(eta, 0.1)
        self.assertLess(eta, 1000.0)


class ProcessTest(unittest.TestCase):

    def test_sulfur_moles_conserved_and_vapour_stays_non_negative(self):
        for empty in (False, True):
            population = _population(empty_modes=empty)
            env = _env(population)
            before = _after_exchange(population, env)
            after = npf.form_new_particles(
                before, env, MAM4_SPEC, npf.NewParticleFormationParameters())
            np.testing.assert_allclose(np.asarray(_sulfur_moles(after)),
                                       np.asarray(_sulfur_moles(before)),
                                       rtol=1e-5)
            self.assertTrue(bool(jnp.all(after.gases[gas_name("h2so4")] >= 0.0)))

    def test_new_particles_enter_the_aitken_mode_only(self):
        population = _population(empty_modes=True)      # no sink: survival 1
        env = _env(population)
        before = _after_exchange(population, env)
        after = npf.form_new_particles(
            before, env, MAM4_SPEC, npf.NewParticleFormationParameters())
        self.assertTrue(bool(jnp.all(after.numbers[number_name("ait")]
                                     > before.numbers[number_name("ait")])))
        self.assertTrue(bool(jnp.all(after.masses[mass_name("so4", "ait")]
                                     > before.masses[mass_name("so4", "ait")])))
        for mode in MAM4_SPEC.modes:
            if mode.short == "ait":
                continue
            np.testing.assert_array_equal(
                np.asarray(after.numbers[number_name(mode.short)]),
                np.asarray(before.numbers[number_name(mode.short)]))

    def test_clipping_never_overdraws_the_vapour(self):
        # No sink and lots of formation: the rule must book exactly what
        # is available and no more.
        population = _population(empty_modes=True)
        env = _env(population)
        before = _after_exchange(population, env)
        after = npf.form_new_particles(
            before, env, MAM4_SPEC, npf.NewParticleFormationParameters())
        booked = after.masses[mass_name("so4", "ait")] - before.masses[mass_name("so4", "ait")]
        available = before.gases[gas_name("h2so4")] * (M_SO4 / M_H2SO4)
        self.assertTrue(bool(jnp.all(booked <= available * (1 + 1e-6))))
        self.assertTrue(bool(jnp.all(after.gases[gas_name("h2so4")] >= 0.0)))

    def test_boundary_layer_pathway_switches_with_height(self):
        population = _population(empty_modes=True)
        inside = _env(population, height=50.0, pbl=800.0)
        above = _env(population, height=2000.0, pbl=800.0)
        for env, expected in ((inside, 1.0), (above, 0.0)):
            out = npf.form_new_particles(
                _after_exchange(population, env), env, MAM4_SPEC,
                npf.NewParticleFormationParameters())
            self.assertTrue(bool(jnp.all(
                out.diagnostics["npf_boundary_layer_pathway"] == expected)))

    def test_hundred_metre_floor_applies_when_pbl_height_is_absent(self):
        """The D-A2 sub-decision: a lagged or missing PBL height is zero."""
        population = _population(empty_modes=True)
        low = _env(population, height=80.0, pbl=0.0)
        out = npf.form_new_particles(
            _after_exchange(population, low), low, MAM4_SPEC,
            npf.NewParticleFormationParameters())
        self.assertTrue(bool(jnp.all(out.diagnostics["npf_boundary_layer_pathway"] == 1.0)))

    def test_disabling_the_pathway_leaves_binary_formation(self):
        population = _population(empty_modes=True)
        env = _env(population)
        params = npf.NewParticleFormationParameters(pbl_enabled=0.0)
        out = npf.form_new_particles(_after_exchange(population, env), env,
                                     MAM4_SPEC, params)
        self.assertTrue(bool(jnp.all(out.diagnostics["npf_boundary_layer_pathway"] == 0.0)))

    def test_gradient_finite_in_float32_with_no_gas_and_an_empty_mode(self):
        """The growth-rate floor trap, pinned fixed."""
        population = _population(gas=0.0)
        masses = dict(population.masses)
        for sp in MAM4_SPEC.mode("accum").species:
            masses[mass_name(sp, "acc")] = jnp.zeros(SHAPE)
        population = ModalPopulation(masses=masses, numbers=population.numbers,
                                     gases=population.gases)
        env = _env(population)
        target = mass_name("so4", "ait")

        def aitken_number(mass):
            pop = ModalPopulation(masses={**masses, target: mass},
                                  numbers=population.numbers, gases=population.gases)
            out = npf.form_new_particles(_after_exchange(pop, env), env, MAM4_SPEC,
                                         npf.NewParticleFormationParameters())
            return jnp.sum(out.numbers[number_name("ait")])

        grad = jax.grad(aitken_number)(masses[target])
        self.assertTrue(np.all(np.isfinite(np.asarray(grad))))


class ReferenceBookkeepingTest(unittest.TestCase):
    """What the AER-12 A/B against MAM4-JAX changed, pinned."""

    def test_rate_below_mam_floor_forms_nothing_and_stays_finite(self):
        """Below Vehkamäki's concentration bound the fit returns a cluster
        above the Aitken bound; MAM returns before survival for rates
        under 1e-6 cm⁻³ s⁻¹, and so must we (this was a float64 NaN).
        """
        population = _population(empty_modes=True, gas=1.8e-13)
        # Free troposphere: the fixture's default 50 m / 800 m PBL would
        # select the empirical pathway instead of the binary one at issue.
        env = _env(population, temperature=291.3, height=2000.0, pbl=0.0)
        out = npf.form_new_particles(_after_exchange(population, env), env, MAM4_SPEC,
                                     npf.NewParticleFormationParameters())
        self.assertTrue(bool(jnp.all(jnp.isfinite(out.diagnostics["npf_rate_apparent"]))))
        self.assertTrue(bool(jnp.all(out.diagnostics["npf_rate_apparent"] == 0.0)))
        np.testing.assert_array_equal(np.asarray(out.numbers[number_name("ait")]),
                                      np.asarray(population.numbers[number_name("ait")]))

    def test_number_is_counted_from_the_pure_sulfate_mass(self):
        """MAM: ``qnuma_del = qso4a_del · mw_so4a(96) / mass_part`` while the
        host mass is booked at the aerosol species' 115 — one particle of
        the booked size holds 115/96 of its pure mass.
        """
        population = _population(empty_modes=True, gas=1.0e-9)      # vapour-limited
        env = _env(population, temperature=240.0)
        before = _after_exchange(population, env)
        after = npf.form_new_particles(before, env, MAM4_SPEC,
                                       npf.NewParticleFormationParameters())
        d_number = after.numbers[number_name("ait")] - before.numbers[number_name("ait")]
        d_mass = after.masses[mass_name("so4", "ait")] - before.masses[mass_name("so4", "ait")]
        particle_mass = (math.pi / 6.0) * npf._AITKEN_LO ** 3 * npf.RHO_H2SO4
        np.testing.assert_allclose(np.asarray(d_number * particle_mass / d_mass),
                                   npf._M_SO4_PURE / M_SO4, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
