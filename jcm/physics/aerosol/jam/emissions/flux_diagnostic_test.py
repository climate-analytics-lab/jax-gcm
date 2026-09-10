"""Tests for the shared per-species emission-flux diagnostic."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
    EMITTED_SPECIES,
    accumulate_emission_fluxes,
    emission_flux_keys,
    _species_of,
)

NLEV, NCOLS = 6, 4


RHO = 1.0     # kg/m^3
DZ = 100.0    # m  -> layer mass 100 kg/m^2, column mass 600 kg/m^2


def _mass_fields():
    return (jnp.full((NLEV, NCOLS), RHO), jnp.full((NLEV, NCOLS), DZ))


class SpeciesParsingTest(unittest.TestCase):

    def test_recognises_interstitial_cloud_borne_and_gas(self):
        self.assertEqual(_species_of("m_so4_acc"), "so4")
        self.assertEqual(_species_of("mc_so4_acc"), "so4")
        self.assertEqual(_species_of("g_dms"), "dms")
        self.assertEqual(_species_of("m_ss_cor"), "ss")

    def test_ignores_number_tracers_and_unknown_species(self):
        # Number tracers carry no mass and must not enter a mass flux.
        self.assertIsNone(_species_of("n_acc"))
        self.assertIsNone(_species_of("nc_ait"))
        self.assertIsNone(_species_of("m_unobtainium_acc"))


class FluxAccumulationTest(unittest.TestCase):

    def test_column_integral_gives_kg_per_m2_per_s(self):
        """Emi = sum(dq/dt * dp/g), i.e. the mass the term adds per area."""
        tend = {"m_ss_acc": jnp.full((NLEV, NCOLS), 1e-12)}
        out = accumulate_emission_fluxes({}, tend, *_mass_fields())
        want = 1e-12 * RHO * DZ * NLEV
        np.testing.assert_allclose(np.asarray(out["emi_ss"]), want, rtol=1e-6)

    def test_modes_of_one_species_are_summed(self):
        tend = {"m_ss_acc": jnp.full((NLEV, NCOLS), 1e-12),
                "m_ss_cor": jnp.full((NLEV, NCOLS), 3e-12)}
        out = accumulate_emission_fluxes({}, tend, *_mass_fields())
        want = 4e-12 * RHO * DZ * NLEV
        np.testing.assert_allclose(np.asarray(out["emi_ss"]), want, rtol=1e-6)

    def test_accumulates_across_terms(self):
        """Several terms emit the same species; AeroCom wants the total."""
        d = accumulate_emission_fluxes(
            {}, {"g_so2": jnp.full((NLEV, NCOLS), 1e-12)}, *_mass_fields())
        d = accumulate_emission_fluxes(
            d, {"g_so2": jnp.full((NLEV, NCOLS), 2e-12)}, *_mass_fields())
        want = 3e-12 * RHO * DZ * NLEV
        np.testing.assert_allclose(np.asarray(d["emi_so2"]), want, rtol=1e-6)

    def test_emits_a_static_key_set(self):
        """All species keys appear even when nothing emits them.

        The diagnostics dict is part of the scan carry, so a key set that
        depends on which tracers a term happens to carry would change the
        carry pytree between the initial probe and the real step.
        """
        out = accumulate_emission_fluxes({}, {}, *_mass_fields())
        self.assertEqual(set(emission_flux_keys()), set(out))
        for key in emission_flux_keys():
            np.testing.assert_allclose(np.asarray(out[key]), 0.0)
        self.assertEqual(len(EMITTED_SPECIES), len(set(EMITTED_SPECIES)))

    def test_elevated_injection_is_counted(self):
        """A term injecting above the surface must still register its flux."""
        tend = jnp.zeros((NLEV, NCOLS)).at[2].set(5e-12)  # mid-troposphere
        out = accumulate_emission_fluxes({}, {"m_bc_acc": tend}, *_mass_fields())
        self.assertGreater(float(jnp.min(out["emi_bc"])), 0.0)


if __name__ == "__main__":
    unittest.main()


class EndToEndEmissionFluxTest(unittest.TestCase):
    """The fluxes must appear in a real JAM run and close the mass budget."""

    def test_sea_salt_flux_matches_the_mass_it_adds(self):
        """emi_ss must equal the sea-salt mass the term actually injects.

        Derived from the term's own tendencies rather than by re-deriving
        its flux formula, so this catches the diagnostic drifting from the
        physics whatever the scheme does internally.
        """
        from jcm.physics.aerosol.jam.emissions.seasalt_test import _inputs
        from jcm.physics.aerosol.jam.emissions.seasalt import SeaSaltEmissions

        state, diagnostics, forcing, terrain = _inputs()
        tend, diag = SeaSaltEmissions()(state, diagnostics, forcing, terrain)

        dm = diagnostics["air_density"] * diagnostics["layer_thickness"]
        want = sum(jnp.sum(t * dm, axis=0)
                   for n, t in tend.tracers.items() if n.startswith("m_ss_"))
        got = np.asarray(diag["emi_ss"])
        np.testing.assert_allclose(
            got, np.asarray(want), rtol=1e-6,
            err_msg="emi_ss must equal the mass the term injects")
        self.assertGreater(got.max(), 0.0, "fixture should emit sea salt")
        # Species no term emits are present and zero (static key set).
        np.testing.assert_allclose(np.asarray(diag["emi_du"]), 0.0)
