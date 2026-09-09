"""κ-Köhler placeholder core: the wet density that pairs with the wet radius."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    equilibrium_modal_state,
)
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

RHO_WATER = 1000.0


def _pure_species_state(species, saturation, mass=1.0e-9, number=1.0e8):
    """Diagnose the population with one species present in every mode."""
    shape = jnp.shape(saturation)
    masses = {
        mass_name(sp, mode.short): jnp.full(shape, mass if sp == species else 0.0)
        for mode in MAM4_SPEC.modes for sp in mode.species
    }
    numbers = {
        number_name(mode.short): jnp.full(shape, number)
        for mode in MAM4_SPEC.modes
    }
    return equilibrium_modal_state(masses, numbers, MAM4_SPEC, saturation)


class WetDensityTest(unittest.TestCase):

    def _sea_salt(self, rh):
        sat = jnp.full((1, 1), rh)
        aer = _pure_species_state("ss", sat)
        cor = [i for i, m in enumerate(MAM4_SPEC.modes) if m.short == "cor"][0]
        return aer, cor

    def test_matches_the_mass_weighted_mixture(self):
        # ρ_wet = (ρ_dry + (g³-1)ρ_w)/g³ with g the κ-Köhler growth factor:
        # the density of the same particle the wet radius describes.
        for rh in (0.5, 0.8, 0.95, 0.99):
            aer, cor = self._sea_salt(rh)
            kappa = float(aer.kappa[cor, 0, 0])
            growth3 = 1.0 + kappa * rh / (1.0 - rh)
            rho_dry = MAM4_SPEC.species_props("ss").density
            expected = (rho_dry + (growth3 - 1.0) * RHO_WATER) / growth3
            self.assertAlmostEqual(
                float(aer.rho[cor, 0, 0]) / expected, 1.0, places=5,
                msg=f"RH = {rh}")

    def test_sea_salt_dilutes_towards_water(self):
        # Documented magnitudes: the dry density overstates the settled mass
        # by 1.64x at 80 % RH and 1.89x at 99 % for coarse sea salt.
        rho_dry = MAM4_SPEC.species_props("ss").density
        for rh, ratio in ((0.80, 1.64), (0.99, 1.89)):
            aer, cor = self._sea_salt(rh)
            self.assertAlmostEqual(
                rho_dry / float(aer.rho[cor, 0, 0]), ratio, delta=0.02,
                msg=f"RH = {rh}")

    def test_bounded_between_water_and_dry_material(self):
        # A wet particle is a mixture, so its density lies between the dry
        # material's and water's, for every mode carrying the species.
        sat = jnp.linspace(0.0, 0.99, 8).reshape((8, 1))
        for species in ("ss", "du", "so4", "bc"):
            aer = _pure_species_state(species, sat)
            rho_dry = MAM4_SPEC.species_props(species).density
            for i, mode in enumerate(MAM4_SPEC.modes):
                if species not in mode.species:
                    continue
                rho = np.asarray(aer.rho[i])
                label = f"{species}/{mode.short}"
                self.assertTrue(np.all(np.isfinite(rho)), label)
                self.assertTrue(np.all(rho <= rho_dry * (1.0 + 1e-5)), label)
                self.assertTrue(
                    np.all(rho >= min(RHO_WATER, rho_dry) * (1.0 - 1e-5)),
                    label)

    def test_empty_mode_falls_back_to_its_first_species(self):
        # An empty mode has no mixture to weight; CAM's convention is the
        # mode's first species density (``specdens_amode(1,mode)``).
        aer = _pure_species_state("ss", jnp.full((1, 1), 0.5))
        pcm = [i for i, m in enumerate(MAM4_SPEC.modes)
               if m.short == "pcm"][0]
        self.assertAlmostEqual(
            float(aer.rho[pcm, 0, 0]),
            MAM4_SPEC.species_props(
                MAM4_SPEC.modes[pcm].species[0]).density, places=3)

    def test_dry_air_leaves_the_material_density(self):
        # κ-Köhler growth is 1 at zero water activity, so ρ_wet = ρ_dry.
        aer = _pure_species_state("du", jnp.zeros((1, 1)))
        cor = [i for i, m in enumerate(MAM4_SPEC.modes) if m.short == "cor"][0]
        self.assertAlmostEqual(
            float(aer.rho[cor, 0, 0]),
            MAM4_SPEC.species_props("du").density, places=3)

    def test_column_and_block_agree(self):
        col = jnp.linspace(0.1, 0.9, 5).reshape((5, 1))
        block = jnp.tile(col, (1, 3))
        rho_col = np.asarray(_pure_species_state("ss", col).rho)
        rho_block = np.asarray(_pure_species_state("ss", block).rho)
        np.testing.assert_allclose(
            rho_block, np.broadcast_to(rho_col, rho_block.shape), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
