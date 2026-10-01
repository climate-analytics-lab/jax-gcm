"""AER-07: renaming, pinned on the tail integrals, MAM's guards and conservation.

The transfer is a closed form in the log-normal tail fractions, so those
are checked against numeric integration and the whole fraction against a
numpy transcription of MAM's per-cell arithmetic. Each of MAM's guards is
exercised: no growth, shrinkage, empty mode, growth that leaves the
diameter below the reference, and the guaranteed transfer once the new
diameter is past the cut. The process is checked for conservation, for
the receiver, for the required environment field and for finite float32
gradients.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erfc

import jcm.constants as c

from jcm.physics.aerosol.jam.microphysics import condensation as cond
from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.microphysics import rename
from jcm.physics.aerosol.jam.microphysics.koehler import (
    equilibrium_modal_state,
    saturation_ratio,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.modal import (
    JamModalMicrophysics,
    MicrophysicsEnvironment,
    ModalPopulation,
    PROCESS_ORDER,
)
from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY, mirror_names
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name
from jcm.physics_interface import PhysicsState
from jcm.testing import check_gradients

SHAPE = (3, 2)
T, P, Q, DT = 285.0, 9.0e4, 0.005, 1800.0
AIT = MAM4_SPEC.mode("aitken")
ACC = MAM4_SPEC.mode("accum")


def _population(seed=0, *, gas=1.0e-10):
    rng = np.random.default_rng(seed)
    masses = {mass_name(sp, m.short): jnp.asarray(rng.uniform(1e-12, 1e-9, SHAPE))
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.asarray(rng.uniform(1e6, 1e9, SHAPE))
               for m in MAM4_SPEC.modes}
    gases = {gas_name("h2so4"): jnp.full(SHAPE, gas),
             gas_name("soag"): jnp.full(SHAPE, 1.0e-10)}
    return ModalPopulation(masses=masses, numbers=numbers, gases=gases)


def _env(population, *, volume_start=None):
    temp = jnp.full(SHAPE, T)
    pressure = jnp.full(SHAPE, P)
    sat = saturation_ratio(temp, jnp.full(SHAPE, Q), pressure)
    if volume_start is None:
        volume_start = {m.short: moments.mode_volume(population.masses, m, MAM4_SPEC)
                        for m in MAM4_SPEC.modes}
    return MicrophysicsEnvironment(
        temperature=temp, pressure=pressure, saturation=sat,
        air_density=pressure / (c.rd * temp),
        state_start=equilibrium_modal_state(
            population.masses, population.numbers, MAM4_SPEC, sat),
        h2so4_production=jnp.zeros(SHAPE),
        height_above_surface=jnp.full(SHAPE, 500.0),
        pbl_height=jnp.zeros(SHAPE[1:]), dt=DT, dry_volume_start=volume_start)


def _grown(population, factor):
    """Scale every Aitken mass of ``population`` by ``factor``."""
    masses = dict(population.masses)
    for sp in AIT.species:
        masses[mass_name(sp, "ait")] = masses[mass_name(sp, "ait")] * factor
    return ModalPopulation(masses=masses, numbers=population.numbers,
                           gases=population.gases)


def _aitken_population(number, volume_fraction_of_reference=1.0):
    """Build an Aitken mode of sulfate only, at ``number`` and a chosen volume."""
    population = _population()
    masses = {k: jnp.zeros(SHAPE) if k.endswith("_ait") else v
              for k, v in population.masses.items()}
    volume = moments.volume_from_diameter(number, AIT.dgnum, AIT) * volume_fraction_of_reference
    masses[mass_name("so4", "ait")] = jnp.full(SHAPE, volume * 1770.0)
    numbers = {**population.numbers, number_name("ait"): jnp.full(SHAPE, number)}
    return ModalPopulation(masses=masses, numbers=numbers, gases=population.gases)


def _mam_fractions(volume_old, volume_new, number):
    """Transcribe ``modal_aero_rename_sub`` for one cell, in numpy."""
    ln_sigma = math.log(AIT.geom_std_dev)
    f = (math.pi / 6.0) * math.exp(4.5 * ln_sigma ** 2)
    yy = math.sqrt(0.5) / ln_sigma
    cut = rename.cut_diameter(AIT, ACC)
    smallest = rename._SMALLEST_VOLUME
    v2nlo = rename.RELAXATION / (f * AIT.dgnum_lo ** 3)
    v2nhi = 1.0 / (f * AIT.dgnum_hi ** 3) / rename.RELAXATION
    old_bnd = max(volume_old, smallest)
    delta = volume_new - volume_old
    if volume_new <= smallest or delta <= 1e-6 * old_bnd:
        return 0.0, 0.0
    n = max(0.0, number)
    n = min(old_bnd * v2nlo, n)
    n = max(old_bnd * v2nhi, n)
    dg_new = (volume_new / (n * f)) ** (1 / 3)
    if dg_new <= AIT.dgnum:
        return 0.0, 0.0
    tail = lambda ln_d: 0.5 * erfc((math.log(cut) - ln_d) * yy)   # noqa: E731
    tn_new, tv_new = tail(math.log(dg_new)), tail(math.log(dg_new) + 3 * ln_sigma ** 2)
    dg_old = (old_bnd / (n * f)) ** (1 / 3)
    if dg_new >= cut:
        dg_old = min(dg_old, 0.99 * cut)
    tn_old, tv_old = tail(math.log(dg_old)), tail(math.log(dg_old) + 3 * ln_sigma ** 2)
    dum = tv_new * volume_new - tv_old * volume_old
    if dum <= 0.0:
        return 0.0, 0.0
    xv = min(min(dum, volume_new) / volume_new, 1.0 - 10 * 2.2e-16)
    xn = max(0.0, min(tn_new - tn_old, xv))
    return xv, xn


class TailTest(unittest.TestCase):

    def test_tail_fractions_against_numeric_integration(self):
        dg, sigma = 40e-9, 1.6
        ln_sigma = math.log(sigma)
        cut = rename.cut_diameter(AIT, ACC)
        x = np.linspace(-8 * ln_sigma, 8 * ln_sigma, 20001)
        d = dg * np.exp(x)
        pdf = np.exp(-0.5 * (x / ln_sigma) ** 2)
        above = d > cut
        number_ref = pdf[above].sum() / pdf.sum()
        volume_ref = (pdf * d ** 3)[above].sum() / (pdf * d ** 3).sum()
        number = float(rename.tail_fraction(math.log(cut), jnp.asarray(math.log(dg)), ln_sigma))
        volume = float(rename.tail_fraction(
            math.log(cut), jnp.asarray(math.log(dg) + 3 * ln_sigma ** 2), ln_sigma))
        # float32 erfc against a 20001-point sum: 6e-4 measured.
        self.assertAlmostEqual(number / number_ref, 1.0, delta=2e-3)
        self.assertAlmostEqual(volume / volume_ref, 1.0, delta=2e-3)

    def test_cut_lies_between_the_modes(self):
        cut = rename.cut_diameter(AIT, ACC)
        self.assertGreater(cut, AIT.dgnum)
        self.assertLess(cut, ACC.dgnum)
        expected = math.sqrt(26e-9 * math.exp(1.5 * math.log(1.6) ** 2)
                             * 110e-9 * math.exp(1.5 * math.log(1.8) ** 2))
        self.assertAlmostEqual(cut / expected, 1.0, places=9)


class FractionTest(unittest.TestCase):

    def _fractions(self, volume_old, volume_new, number):
        xv, xn = rename.transfer_fractions(
            jnp.asarray(volume_old), jnp.asarray(volume_new), jnp.asarray(number), AIT, ACC)
        return float(xv), float(xn)

    def test_matches_a_transcription_of_mam_per_cell(self):
        number = 1.0e9
        reference = moments.volume_from_diameter(number, AIT.dgnum, AIT)
        cases = [
            (reference, reference * 1.0),          # no growth
            (reference, reference * 0.8),          # shrinkage
            (reference * 0.5, reference * 0.9),    # growth, still below dgnum
            (reference, reference * 1.5),          # modest growth past dgnum
            (reference, reference * 5.0),          # strong growth
            (reference * 8.0, reference * 12.0),   # already past the cut
            (reference, reference * 1e3),          # absurd growth
            (0.0, reference),                      # from nothing
            (reference, reference * 1.5),
        ]
        numbers = [number] * 8 + [number * 1e4]    # the last: number clipped by the bounds
        for (old, new), n in zip(cases, numbers):
            got = self._fractions(old, new, n)
            want = _mam_fractions(old, new, n)
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-7, err_msg=str((old, new, n)))

    def test_guards(self):
        number = 1.0e9
        reference = moments.volume_from_diameter(number, AIT.dgnum, AIT)
        self.assertEqual(self._fractions(reference, reference, number), (0.0, 0.0))
        self.assertEqual(self._fractions(reference, 0.5 * reference, number), (0.0, 0.0))
        self.assertEqual(self._fractions(0.0, 0.0, 0.0), (0.0, 0.0))
        self.assertEqual(self._fractions(0.3 * reference, 0.9 * reference, number), (0.0, 0.0))

    def test_growth_transfers_and_number_never_exceeds_volume(self):
        number = 1.0e9
        reference = moments.volume_from_diameter(number, AIT.dgnum, AIT)
        for factor in (1.2, 2.0, 5.0, 20.0):
            xv, xn = self._fractions(reference, reference * factor, number)
            self.assertGreater(xv, 0.0)
            self.assertLessEqual(xn, xv)
            self.assertLess(xv, 1.0)

    def test_past_the_cut_transfer_is_guaranteed(self):
        number = 1.0e9
        cut_volume = moments.volume_from_diameter(number, rename.cut_diameter(AIT, ACC), AIT)
        xv, xn = self._fractions(1.5 * cut_volume, 1.6 * cut_volume, number)
        self.assertGreater(xv, 0.0)
        self.assertGreater(xn, 0.0)

    def test_gradients_through_a_transfer(self):
        """AD against a central difference, in float64.

        In float32 the function is smooth to the eye (secants agree to 1e-3
        per direction) but ``check_gradients``' two-rung consistency test
        finds no usable step: erfc's float32 seams put ~1e-6 roughness on a
        fraction of 0.2, which a relative-step difference amplifies to 1 %.
        The precedent is ``physics_interface_test``: flip x64 for the check
        and restore it.
        """
        number = 1.0e9
        reference = moments.volume_from_diameter(number, AIT.dgnum, AIT)
        was_x64 = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            def fractions(old_scale, new_scale, number_scale):
                return jnp.stack(rename.transfer_fractions(
                    old_scale * reference, new_scale * reference,
                    number_scale * number, AIT, ACC))

            check_gradients(fractions, (jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(1.0)),
                            rtol=1e-3)
        finally:
            jax.config.update("jax_enable_x64", was_x64)

    def test_gradients_finite_in_float32_where_nothing_moves(self):
        for old, new, n in ((0.0, 0.0, 0.0), (1e-15, 1e-15, 1e9), (1e-15, 0.5e-15, 1e9)):
            grads = jax.grad(
                lambda o, v, k: jnp.sum(jnp.stack(rename.transfer_fractions(o, v, k, AIT, ACC))),
                argnums=(0, 1, 2))(jnp.asarray(old), jnp.asarray(new), jnp.asarray(n))
            self.assertTrue(all(bool(jnp.isfinite(g)) for g in grads), (old, new, n))


class ProcessTest(unittest.TestCase):

    def test_requires_the_start_volume(self):
        population = _population()
        env = _env(population, volume_start=None)
        env = MicrophysicsEnvironment(**{**env.__dict__, "dry_volume_start": None})
        with self.assertRaises(ValueError):
            rename.rename_aitken(population, env, MAM4_SPEC, None)

    def test_no_growth_no_transfer(self):
        population = _population()
        out = rename.rename_aitken(population, _env(population), MAM4_SPEC, None)
        for key in population.masses:
            np.testing.assert_array_equal(np.asarray(out.masses[key]),
                                          np.asarray(population.masses[key]))
        self.assertTrue(bool(jnp.all(out.diagnostics["rename_volume_fraction"] == 0.0)))

    def test_growth_moves_aitken_to_accumulation_and_conserves(self):
        before = _aitken_population(1.0e9)
        env = _env(before)
        grown = _grown(before, 3.0)
        after = rename.rename_aitken(grown, env, MAM4_SPEC, None)
        xv = after.diagnostics["rename_volume_fraction"]
        xn = after.diagnostics["rename_number_fraction"]
        self.assertTrue(bool(jnp.all((xv > 0.0) & (xn > 0.0) & (xn <= xv))))
        for sp in AIT.species:
            total_before = grown.masses[mass_name(sp, "ait")] + grown.masses[mass_name(sp, "acc")]
            total_after = after.masses[mass_name(sp, "ait")] + after.masses[mass_name(sp, "acc")]
            np.testing.assert_allclose(np.asarray(total_after), np.asarray(total_before), rtol=1e-6)
            np.testing.assert_allclose(
                np.asarray(after.masses[mass_name(sp, "ait")]),
                np.asarray(grown.masses[mass_name(sp, "ait")] * (1.0 - xv)), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(after.numbers[number_name("ait")] + after.numbers[number_name("acc")]),
            np.asarray(grown.numbers[number_name("ait")] + grown.numbers[number_name("acc")]),
            rtol=1e-6)
        for short in ("cor", "pcm"):
            np.testing.assert_array_equal(np.asarray(after.numbers[number_name(short)]),
                                          np.asarray(grown.numbers[number_name(short)]))

    def test_end_to_end_condensation_then_rename(self):
        """Vapour grows the Aitken mode past its reference; renaming moves it on."""
        population = _aitken_population(1.0e8)
        population = ModalPopulation(
            masses=population.masses, numbers=population.numbers,
            gases={**population.gases, gas_name("h2so4"): jnp.full(SHAPE, 5.0e-9)})
        env = _env(population)
        condensed = cond.condense(population, env, MAM4_SPEC, cond.CondensationParameters())
        renamed = rename.rename_aitken(condensed, env, MAM4_SPEC, None)
        self.assertTrue(bool(jnp.all(renamed.diagnostics["rename_volume_fraction"] > 0.0)))
        self.assertTrue(bool(jnp.all(renamed.numbers[number_name("acc")]
                                     > condensed.numbers[number_name("acc")])))

    def test_driver_supplies_the_start_volume(self):
        population = _aitken_population(1.0e8)
        tracers = {**population.masses, **population.numbers,
                   gas_name("h2so4"): jnp.full(SHAPE, 5.0e-9)}
        state = PhysicsState.zeros(SHAPE).copy(
            temperature=jnp.full(SHAPE, T), specific_humidity=jnp.full(SHAPE, Q),
            tracers=tracers)
        diagnostics = {"pressure_full": jnp.full(SHAPE, P), "_dt_seconds": DT}
        core = JamModalMicrophysics(processes=PROCESS_ORDER[:2])   # exchange, rename
        tendency, _ = core(state, diagnostics, None, None)
        self.assertTrue(bool(jnp.all(tendency.tracers[number_name("acc")] > 0.0)))
        self.assertTrue(bool(jnp.all(tendency.tracers[number_name("ait")] < 0.0)))


class CloudBorneTest(unittest.TestCase):
    """D-A5: both phases renamed by fractions from the combined volume."""

    def _with_cloud_borne(self, population, mass=2.0e-11, number=2.0e8):
        cloud_borne = {name: jnp.zeros(SHAPE) for name in mirror_names(MAM4_SPEC)}
        for sp in AIT.species:
            cloud_borne[mass_name(sp, "ait", cloud_borne=True)] = jnp.full(SHAPE, mass)
        cloud_borne[number_name("ait", cloud_borne=True)] = jnp.full(SHAPE, number)
        return ModalPopulation(masses=population.masses, numbers=population.numbers,
                               gases=population.gases, cloud_borne=cloud_borne)

    def test_fractions_use_the_combined_volume_and_number(self):
        before = _aitken_population(1.0e9)
        env = _env(before)
        grown = self._with_cloud_borne(_grown(before, 3.0))
        after = rename.rename_aitken(grown, env, MAM4_SPEC, None)
        cb_volume = sum(float(grown.cloud_borne[mass_name(sp, "ait", cloud_borne=True)][0, 0])
                        / MAM4_SPEC.species_props(sp).density for sp in AIT.species)
        old = float(env.dry_volume_start["ait"][0, 0]) + cb_volume
        new = float(moments.mode_volume(grown.masses, AIT, MAM4_SPEC)[0, 0]) + cb_volume
        n = float(grown.numbers[number_name("ait")][0, 0]) + 2.0e8
        xv, xn = _mam_fractions(old, new, n)
        np.testing.assert_allclose(
            float(after.diagnostics["rename_volume_fraction"][0, 0]), xv, rtol=2e-5)
        np.testing.assert_allclose(
            float(after.diagnostics["rename_number_fraction"][0, 0]), xn, rtol=2e-5)
        # And the cloud-borne presence changed the answer.
        alone = rename.rename_aitken(_grown(before, 3.0), env, MAM4_SPEC, None)
        self.assertFalse(np.allclose(np.asarray(alone.diagnostics["rename_volume_fraction"]),
                                     np.asarray(after.diagnostics["rename_volume_fraction"])))

    def test_cloud_borne_phase_moves_by_the_same_fractions_and_conserves(self):
        before = _aitken_population(1.0e9)
        grown = self._with_cloud_borne(_grown(before, 3.0))
        after = rename.rename_aitken(grown, _env(before), MAM4_SPEC, None)
        xv = after.diagnostics["rename_volume_fraction"]
        xn = after.diagnostics["rename_number_fraction"]
        for sp in AIT.species:
            src = mass_name(sp, "ait", cloud_borne=True)
            dst = mass_name(sp, "acc", cloud_borne=True)
            np.testing.assert_allclose(np.asarray(after.cloud_borne[src]),
                                       np.asarray(grown.cloud_borne[src] * (1.0 - xv)), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(after.cloud_borne[src] + after.cloud_borne[dst]),
                                       np.asarray(grown.cloud_borne[src] + grown.cloud_borne[dst]),
                                       rtol=1e-6)
        src, dst = number_name("ait", cloud_borne=True), number_name("acc", cloud_borne=True)
        np.testing.assert_allclose(np.asarray(after.cloud_borne[src]),
                                   np.asarray(grown.cloud_borne[src] * (1.0 - xn)), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(after.cloud_borne[dst]),
                                   np.asarray(grown.cloud_borne[src] * xn), rtol=1e-6)
        for name in mirror_names(MAM4_SPEC):
            if "_ait" not in name and "_acc" not in name:
                np.testing.assert_array_equal(np.asarray(after.cloud_borne[name]),
                                              np.asarray(grown.cloud_borne[name]))

    def test_driver_books_the_cloud_borne_change_to_the_carry_store(self):
        population = _aitken_population(1.0e8)
        tracers = {**population.masses, **population.numbers,
                   gas_name("h2so4"): jnp.full(SHAPE, 5.0e-9)}
        state = PhysicsState.zeros(SHAPE).copy(
            temperature=jnp.full(SHAPE, T), specific_humidity=jnp.full(SHAPE, Q),
            tracers=tracers)
        carry = {name: jnp.zeros(SHAPE) for name in mirror_names(MAM4_SPEC)}
        carry[number_name("ait", cloud_borne=True)] = jnp.full(SHAPE, 2.0e7)
        carry[mass_name("so4", "ait", cloud_borne=True)] = jnp.full(SHAPE, 1.0e-11)
        diagnostics = {"pressure_full": jnp.full(SHAPE, P), "_dt_seconds": DT,
                       CARRY_KEY: carry}
        core = JamModalMicrophysics(processes=PROCESS_ORDER[:2])
        self.assertIn(CARRY_KEY, core.requires)
        tendency, out = core(state, diagnostics, None, None)
        new_carry = out[CARRY_KEY]
        self.assertLess(float(new_carry[number_name("ait", cloud_borne=True)][0, 0]), 2.0e7)
        self.assertGreater(float(new_carry[number_name("acc", cloud_borne=True)][0, 0]), 0.0)
        np.testing.assert_allclose(
            np.asarray(new_carry[number_name("ait", cloud_borne=True)]
                       + new_carry[number_name("acc", cloud_borne=True)]), 2.0e7, rtol=1e-6)
        # Cloud-borne names never appear in the tracer tendency.
        self.assertFalse(any(k.startswith(("mc_", "nc_")) for k in tendency.tracers))

    def test_without_the_store_the_probe_still_runs(self):
        state = PhysicsState.zeros(SHAPE).copy(
            temperature=jnp.full(SHAPE, T), specific_humidity=jnp.full(SHAPE, Q), tracers={})
        diagnostics = {"pressure_full": jnp.full(SHAPE, P), "_dt_seconds": DT}
        _, out = JamModalMicrophysics()(state, diagnostics, None, None)
        self.assertNotIn(CARRY_KEY, out)   # no store to update, none invented


if __name__ == "__main__":
    unittest.main()
