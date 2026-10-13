"""Phase 0 tests: gas-phase precursor species table + g_* tracer layout (#496)."""

import unittest

from jcm.physics.aerosol.jam.gas_species import (
    GAS_SPECIES,
    MAM4_GAS,
    SULFUR_GASES,
)
from jcm.physics.aerosol.jam.tracer_layout import gas_name, gas_tracer_specs


class GasSpeciesTest(unittest.TestCase):
    def test_gas_name(self):
        self.assertEqual(gas_name("so2"), "g_so2")
        self.assertEqual(gas_name("h2so4"), "g_h2so4")

    def test_tracer_specs_cover_the_chain(self):
        specs = gas_tracer_specs(SULFUR_GASES)
        self.assertEqual(
            {s.name for s in specs},
            {"g_dms", "g_so2", "g_h2so4", "g_soag"},
        )
        self.assertTrue(all(s.units == "kg/kg" for s in specs))

    def test_mam4_gas_is_the_consumed_subset(self):
        self.assertEqual(MAM4_GAS, ("h2so4", "soag"))
        self.assertTrue(set(MAM4_GAS).issubset(GAS_SPECIES))
        self.assertTrue(set(SULFUR_GASES).issubset(GAS_SPECIES))

    def test_molar_masses_match_mam4_constants(self):
        # MAM4-JAX MW_GAS / ADV_MASS values (g/mol) — the adapter hand-off
        # must be unit-consistent with the core.
        expected = {
            "h2so4": 98.0784,
            "soag": 150.0,
            "so2": 64.0648,
            "dms": 62.1324,
        }
        for sp, mw in expected.items():
            self.assertAlmostEqual(
                GAS_SPECIES[sp].molar_mass * 1000.0, mw, places=3
            )


if __name__ == "__main__":
    unittest.main()
