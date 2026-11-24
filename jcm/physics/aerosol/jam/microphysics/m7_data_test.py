"""Every M7 population number against its Fortran source (jax-gcm#1017).

Transcribed from ECHAM6.3-HAM2.3 r7492 (see ``m7_data.py``'s module
docstring for the file:line provenance of each table); this test re-checks
those same numbers so a future edit cannot silently drift from the source.
"""

import math
import unittest

from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import HamFreezingClasses
from jcm.physics.aerosol.jam.microphysics.m7_data import (
    M7_CACCSO4,
    M7_CBCR,
    M7_CBCS,
    M7_CSR_STRAT_ICE,
    M7_CSR_STRAT_MIX,
    M7_CSR_STRAT_WAT,
    M7_FREEZING_ROLES,
    M7_MODES,
    M7_NION,
    M7_SPEC,
    M7_SPECIES_BY_NAME,
)

_SHORTS = ("ns", "ks", "as", "cs", "ki", "ai", "ci")


class M7SpeciesTest(unittest.TestCase):
    """mo_ham_species.f90 (SO4 molar mass: mo_ham.f90:311)."""

    # token -> (molar_mass [kg/mol], density [kg/m3], kappa, nion)
    REFERENCE = {
        "so4": (0.0960631, 1841.0, 0.60, 2),
        "bc": (0.01201, 2000.0, 0.0, 0),
        "oc": (0.180, 2000.0, 0.06, 0),
        "ss": (0.058443, 2165.0, 1.0, 2),
        "du": (0.250, 2650.0, 0.0, 0),
        "h2o": (0.018, 1000.0, 0.0, None),
    }

    def test_species_table(self):
        for token, (mw, rho, kappa, nion) in self.REFERENCE.items():
            sp = M7_SPECIES_BY_NAME[token]
            self.assertAlmostEqual(sp.molar_mass, mw, places=7, msg=token)
            self.assertAlmostEqual(sp.density, rho, msg=token)
            self.assertAlmostEqual(sp.hygroscopicity, kappa, msg=token)
            if nion is not None:
                self.assertEqual(M7_NION[token], nion, msg=token)

    def test_only_five_dry_species_plus_water(self):
        self.assertEqual(
            {s.name for s in M7_SPEC.species},
            {"so4", "bc", "oc", "ss", "du", "h2o"},
        )

    def test_global_mam4_species_table_untouched(self):
        from jcm.physics.aerosol.jam.species import SPECIES
        self.assertEqual(SPECIES["so4"].molar_mass, 0.115)
        self.assertNotIn("oc", SPECIES)


class M7ModesTest(unittest.TestCase):
    """mo_ham_m7ctl.f90: sizeclass(1..7), crdiv, sigma, csr_conv."""

    # short -> (name, sigma_g, soluble, can_activate, sediments, csr_conv)
    REFERENCE = {
        "ns": ("nucleation_soluble", 1.59, True, False, False, 0.20),
        "ks": ("aitken_soluble", 1.59, True, True, False, 0.60),
        "as": ("accumulation_soluble", 1.59, True, True, True, 0.99),
        "cs": ("coarse_soluble", 2.0, True, True, True, 0.99),
        "ki": ("aitken_insoluble", 1.59, False, False, False, 0.20),
        "ai": ("accumulation_insoluble", 1.59, False, False, True, 0.40),
        "ci": ("coarse_insoluble", 2.0, False, False, True, 0.40),
    }

    # short -> species tuple (mo_ham_m7ctl.f90 mass-index comment, lines ~104-145)
    SPECIES = {
        "ns": ("so4",), "ks": ("so4", "bc", "oc"),
        "as": ("so4", "bc", "oc", "ss", "du"),
        "cs": ("so4", "bc", "oc", "ss", "du"),
        "ki": ("bc", "oc"), "ai": ("du",), "ci": ("du",),
    }

    # short -> (dgnum_lo, dgnum_hi) [m], from crdiv (mo_ham_m7ctl.f90:164)
    # dry radii 0.0005, 0.005, 0.05, 0.5 um -> diameters 1nm, 10nm, 100nm, 1um;
    # M7 has no coarse upper bound, so CS/CI take 10um (the design note).
    BOUNDS = {
        "ns": (1e-9, 1e-8), "ks": (1e-8, 1e-7), "as": (1e-7, 1e-6),
        "cs": (1e-6, 1e-5), "ki": (1e-8, 1e-7), "ai": (1e-7, 1e-6),
        "ci": (1e-6, 1e-5),
    }

    def test_mode_order_and_shorts(self):
        self.assertEqual(tuple(m.short for m in M7_MODES), _SHORTS)

    def test_mode_table(self):
        for m in M7_MODES:
            name, sigma, sol, act, sed, csr = self.REFERENCE[m.short]
            self.assertEqual(m.name, name, msg=m.short)
            self.assertAlmostEqual(m.geom_std_dev, sigma, msg=m.short)
            self.assertEqual(m.soluble, sol, msg=m.short)
            self.assertEqual(m.can_activate, act, msg=m.short)
            self.assertEqual(m.sediments, sed, msg=m.short)
            self.assertAlmostEqual(m.csr_conv, csr, msg=m.short)
            self.assertEqual(m.species, self.SPECIES[m.short], msg=m.short)

    def test_size_bounds_and_geometric_midpoint(self):
        for m in M7_MODES:
            lo, hi = self.BOUNDS[m.short]
            self.assertAlmostEqual(m.dgnum_lo, lo, msg=m.short)
            self.assertAlmostEqual(m.dgnum_hi, hi, msg=m.short)
            self.assertAlmostEqual(m.dgnum, math.sqrt(lo * hi), msg=m.short)

    def test_mass_tracer_count_is_18_number_is_7(self):
        n_mass = sum(len(m.species) for m in M7_MODES)
        self.assertEqual(n_mass, 18)
        self.assertEqual(len(M7_MODES), 7)


class M7WetRemovalTablesTest(unittest.TestCase):
    """mo_ham_m7ctl.f90:213 (caccso4), 515-526 (the rest)."""

    def _check(self, table, values):
        for short, v in zip(_SHORTS, values):
            self.assertAlmostEqual(table[short], v, msg=short)

    def test_csr_strat_wat(self):
        self._check(M7_CSR_STRAT_WAT, (0.10, 0.25, 0.85, 0.99, 0.20, 0.40, 0.40))

    def test_csr_strat_mix(self):
        self._check(M7_CSR_STRAT_MIX, (0.10, 0.40, 0.75, 0.75, 0.10, 0.40, 0.40))

    def test_csr_strat_ice(self):
        self._check(M7_CSR_STRAT_ICE, (0.10,) * 7)

    def test_cbcr(self):
        self._check(M7_CBCR, (5.0e-4, 1.0e-4, 1.0e-3, 1.0e-1, 1.0e-4, 1.0e-3, 1.0e-1))

    def test_cbcs(self):
        self._check(M7_CBCS, (5.0e-3,) * 7)

    def test_caccso4(self):
        self._check(M7_CACCSO4, (1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3))


class M7SpecTest(unittest.TestCase):
    def test_no_explicit_cloud_borne_phase(self):
        self.assertFalse(M7_SPEC.cloud_borne)

    def test_accumulation_mode_is_as(self):
        self.assertEqual(M7_SPEC.accumulation_mode, "as")
        self.assertEqual(M7_SPEC.mode(M7_SPEC.accumulation_mode).short, "as")

    def test_freezing_roles(self):
        self.assertIsInstance(M7_SPEC.freezing_roles, HamFreezingClasses)
        self.assertEqual(M7_SPEC.freezing_roles, M7_FREEZING_ROLES)
        self.assertEqual(M7_FREEZING_ROLES.soluble, ("as", "cs"))
        self.assertEqual(M7_FREEZING_ROLES.insoluble_aitken, "ki")
        self.assertEqual(M7_FREEZING_ROLES.insoluble_accumulation, "ai")
        self.assertEqual(M7_FREEZING_ROLES.insoluble_coarse, "ci")
        M7_FREEZING_ROLES.validate(M7_SPEC)

    def test_aqueous_sulfate_modes_is_as_cs(self):
        self.assertEqual(M7_SPEC.aqueous_sulfate_modes, ("as", "cs"))

    def test_dust_emission_policy_set(self):
        policy = M7_SPEC.dust_emission
        self.assertIsNotNone(policy)
        self.assertEqual(policy.accum_mode, "ai")
        self.assertEqual(policy.coarse_mode, "ci")

    def test_primary_emission_policy(self):
        self.assertEqual(
            M7_SPEC.primary_split("so4"),
            ((M7_SPEC.mode("ks"), 0.5), (M7_SPEC.mode("as"), 0.5)),
        )
        self.assertEqual(
            M7_SPEC.primary_split("bc"), ((M7_SPEC.mode("ki"), 1.0),))
        self.assertEqual(
            M7_SPEC.primary_split("oc"), ((M7_SPEC.mode("ki"), 1.0),))
        du = dict(M7_SPEC.primary_split("du"))
        self.assertAlmostEqual(sum(du.values()), 1.0)

    def test_mam4_spec_unaffected(self):
        from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
        self.assertIsNone(MAM4_SPEC.dust_emission)
        self.assertIsNone(MAM4_SPEC.freezing_roles)
        self.assertIsNone(MAM4_SPEC.aqueous_sulfate_modes)
        self.assertEqual(MAM4_SPEC.accumulation_mode, "accum")
        self.assertTrue(MAM4_SPEC.cloud_borne)
        self.assertTrue(all(m.csr_conv is None for m in MAM4_SPEC.modes))


if __name__ == "__main__":
    unittest.main()
