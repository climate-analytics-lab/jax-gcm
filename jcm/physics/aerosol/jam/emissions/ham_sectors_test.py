"""Tests for HAM's per-sector emission policy (jax-gcm#1017).

Every factor is checked against a hand transcription of the Fortran
constants/formulas (``mo_ham_m7_emissions.f90``, cited inline), and the
``emission_diameter`` mechanism is checked to reproduce HAM's own
mass->number conversion ``zm2n`` exactly (algebraic identity).
"""

import math
import unittest

from jcm.physics.aerosol.jam.emissions.distributors import particle_mean_mass
from jcm.physics.aerosol.jam.emissions.ham_sectors import (
    BB_WSOC_FRACTION,
    CMR_BB,
    CMR_FF,
    CMR_SA,
    CMR_SC,
    CMR_SK,
    cmr_to_emission_diameter,
    m7_sector_policy,
)
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.population import AerosolMode


def _zm2n_fortran(cmr_m: float, sigma_g: float, rho: float) -> float:
    """Direct transcription of ``mo_ham_m7_emissions.f90``'s ``zm2n`` formula.

    ``zm2n = 3/(4·pi·rho·(cmr·cmr2ram(sigma))³)``,
    ``cmr2ram(sigma) = exp(1.5·ln(sigma)²)`` (mo_ham_m7ctl.f90:427).
    """
    cmr2ram = math.exp(1.5 * math.log(sigma_g) ** 2)
    r_ram = cmr_m * cmr2ram
    return 3.0 / (4.0 * math.pi * rho * r_ram ** 3)


class CmrToEmissionDiameterTest(unittest.TestCase):
    def test_reproduces_ham_zm2n_for_every_cmr_sigma_combination(self):
        rho = 1841.0  # arbitrary (so4 density); the identity is rho-agnostic
        for cmr_m in (CMR_FF, CMR_BB, CMR_SK, CMR_SA, CMR_SC):
            for sigma_g in (1.59, 2.0):
                d = cmr_to_emission_diameter(cmr_m, sigma_g)
                mode = AerosolMode(
                    name="t", short="t", geom_std_dev=sigma_g,
                    dgnum=d, dgnum_lo=d, dgnum_hi=d, species=(),
                    soluble=True, can_activate=False, sediments=False,
                )
                m_p = particle_mean_mass(mode, rho, emission_diameter=d)
                zm2n_from_distributor = 1.0 / m_p
                zm2n_expected = _zm2n_fortran(cmr_m, sigma_g, rho)
                self.assertAlmostEqual(
                    zm2n_from_distributor / zm2n_expected, 1.0, places=10,
                    msg=(cmr_m, sigma_g))


class M7SectorPolicyTest(unittest.TestCase):
    """Every (class, species) target against mo_ham_m7_emissions.f90."""

    def setUp(self):
        self.policy = m7_sector_policy(om_oc=1.4)

    def _targets(self, ham_class, species):
        return {t.mode: (t.mass_fraction, t.cmr_m)
                for t in self.policy.targets[ham_class][species]}

    def test_om_oc_is_zom2oc(self):
        self.assertAlmostEqual(self.policy.om_oc, 1.4)  # line 93

    def test_fossil_bc(self):
        # line 567: pfactor(mbcki)=1; line 574: zm2n_bcki_ff (cmr_ff).
        self.assertEqual(self._targets("fossil", "bc"), {"ki": (1.0, CMR_FF)})

    def test_fossil_oc(self):
        # lines 582-583: all to KI at cmr_ff.
        self.assertEqual(self._targets("fossil", "oc"), {"ki": (1.0, CMR_FF)})

    def test_fossil_so4(self):
        # lines 621-624: 50/50 KS(cmr_sk)/AS(cmr_sa).
        self.assertEqual(
            self._targets("fossil", "so4"),
            {"ks": (0.5, CMR_SK), "as": (0.5, CMR_SA)})

    def test_energy_ships_bc_oc_same_as_fossil(self):
        # No ENE/SHIPS branch for BC/OC in the Fortran.
        self.assertEqual(
            self._targets("energy_ships", "bc"), self._targets("fossil", "bc"))
        self.assertEqual(
            self._targets("energy_ships", "oc"), self._targets("fossil", "oc"))

    def test_energy_ships_so4(self):
        # lines 632-635: ms4ks->0, ms4as UNCHANGED (still cmr_sa), ms4cs set
        # (cmr_sc) -> net 50% AS / 50% CS.
        self.assertEqual(
            self._targets("energy_ships", "so4"),
            {"as": (0.5, CMR_SA), "cs": (0.5, CMR_SC)})

    def test_biomass_like_bc(self):
        # line 572: zm2n_bcki_bb (cmr_bb); mass fraction still all-KI.
        self.assertEqual(
            self._targets("biomass_like", "bc"), {"ki": (1.0, CMR_BB)})

    def test_biomass_like_oc(self):
        # lines 588-591: (1-0.65) -> KI, 0.65 -> KS, both at cmr_bb.
        self.assertEqual(
            self._targets("biomass_like", "oc"),
            {"ki": (1.0 - BB_WSOC_FRACTION, CMR_BB),
             "ks": (BB_WSOC_FRACTION, CMR_BB)})

    def test_biomass_like_so4_is_bit_identical_to_fossil(self):
        # lines 253-254 vs 250-251: zm2n_s4ks_bb/zm2n_s4as_bb are the SAME
        # Fortran expression as the _ff ones (cmr_sk/cmr_sa, not a biomass
        # radius) -- a faithful port, not a simplification.
        self.assertEqual(
            self._targets("biomass_like", "so4"),
            self._targets("fossil", "so4"))

    def test_m7_spec_carries_the_same_policy(self):
        self.assertEqual(M7_SPEC.sector_emission.om_oc, 1.4)
        self.assertEqual(
            M7_SPEC.sector_emission.targets, self.policy.targets)


if __name__ == "__main__":
    unittest.main()
