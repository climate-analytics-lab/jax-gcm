"""Tests for the population-owned distribution API (#498)."""

import math
import unittest

from jcm.physics.aerosol.jam.emissions.distributors import particle_mean_mass
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC


class ClassesForTest(unittest.TestCase):
    def test_classes_for_returns_carrying_modes(self):
        shorts = [m.short for m in MAM4_SPEC.classes_for("ss")]
        # ss is carried by acc, ait and cor in MAM4 (not the primary-carbon mode).
        self.assertEqual(set(shorts), {"acc", "ait", "cor"})
        self.assertNotIn("pcm", shorts)

    def test_classes_for_preserves_spec_order(self):
        shorts = [m.short for m in MAM4_SPEC.classes_for("so4")]
        spec_order = [m.short for m in MAM4_SPEC.modes if "so4" in m.species]
        self.assertEqual(shorts, spec_order)


class PrimarySplitTest(unittest.TestCase):
    def test_so4_splits_aitken_accum_half(self):
        split = MAM4_SPEC.primary_split("so4")
        self.assertEqual({m.short for m, _ in split}, {"ait", "acc"})
        for _, frac in split:
            self.assertAlmostEqual(frac, 0.5)

    def test_primary_carbon_to_pcm_only(self):
        for species in ("bc", "poa"):
            split = MAM4_SPEC.primary_split(species)
            self.assertEqual(len(split), 1)
            self.assertEqual(split[0][0].short, "pcm")
            self.assertAlmostEqual(split[0][1], 1.0)

    def test_fractions_sum_to_one(self):
        for species in ("so4", "bc", "poa", "du"):
            total = sum(frac for _, frac in MAM4_SPEC.primary_split(species))
            self.assertAlmostEqual(total, 1.0)

    def test_unknown_species_raises(self):
        with self.assertRaises(KeyError):
            MAM4_SPEC.primary_split("not_a_species")


class NumberFactorTest(unittest.TestCase):
    def test_particle_mean_mass_matches_lognormal_moment(self):
        # particle_mean_mass now delegates to mode.number_factor; check it still
        # equals the closed-form log-normal third-moment mean particle mass.
        mode = MAM4_SPEC.mode("acc")
        rho = 1770.0
        ln_sig = math.log(mode.geom_std_dev)
        expected = rho * (math.pi / 6.0) * mode.dgnum ** 3 * math.exp(4.5 * ln_sig ** 2)
        self.assertAlmostEqual(particle_mean_mass(mode, rho) / expected, 1.0,
                               places=10)


if __name__ == "__main__":
    unittest.main()
