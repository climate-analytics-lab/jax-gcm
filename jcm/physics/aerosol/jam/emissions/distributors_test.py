"""Tests for the modal emission distributor."""

import unittest

import jax.numpy as jnp

from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
from jcm.physics.aerosol.jam.emissions.distributors import (
    distribute_surface_flux,
    particle_mean_mass,
)


class DistributorTest(unittest.TestCase):
    def test_particle_mean_mass_positive(self):
        m = particle_mean_mass(MAM4_SPEC.mode("coarse"), 1900.0)
        self.assertGreater(m, 0.0)

    def test_distribute_adds_mass_and_number_at_surface(self):
        air_density = jnp.full((3, 2), 1.2)
        dz = jnp.full((3, 2), 100.0)
        flux = jnp.full((2,), 1.0e-10)  # kg/m²/s
        tends = distribute_surface_flux(
            MAM4_SPEC, [("ss", "acc", flux)], air_density, dz,
        )
        mname = mass_name("ss", "acc")
        nname = number_name("acc")
        self.assertIn(mname, tends)
        self.assertIn(nname, tends)
        # Only the surface layer is sourced.
        self.assertTrue(bool(jnp.all(tends[mname][:-1] == 0.0)))
        self.assertGreater(float(tends[mname][-1, 0]), 0.0)
        self.assertGreater(float(tends[nname][-1, 0]), 0.0)

    def test_emission_diameter_sets_the_emitted_number(self):
        """m_p = rho (pi/6) D^3 when a scheme supplies its emitted size."""
        import math
        rho = MAM4_SPEC.species_props("ss").density
        d = 0.3e-6
        m = particle_mean_mass(MAM4_SPEC.mode("accum"), rho, d)
        self.assertAlmostEqual(m, rho * math.pi / 6.0 * d ** 3, places=24)

    def test_emission_diameter_changes_number_only(self):
        air_density = jnp.full((3, 2), 1.2)
        dz = jnp.full((3, 2), 100.0)
        flux = jnp.full((2,), 1.0e-10)
        default = distribute_surface_flux(
            MAM4_SPEC, [("ss", "acc", flux)], air_density, dz)
        sized = distribute_surface_flux(
            MAM4_SPEC, [("ss", "acc", flux, 0.3e-6)], air_density, dz)
        mname, nname = mass_name("ss", "acc"), number_name("acc")
        self.assertEqual(float(sized[mname][-1, 0]), float(default[mname][-1, 0]))
        # Number scales as D^-3 against the class's equilibrium geometry.
        mode = MAM4_SPEC.mode("accum")
        rho = MAM4_SPEC.species_props("ss").density
        expect = particle_mean_mass(mode, rho) / particle_mean_mass(
            mode, rho, 0.3e-6)
        self.assertAlmostEqual(
            float(sized[nname][-1, 0]) / float(default[nname][-1, 0]),
            expect, places=3)


if __name__ == "__main__":
    unittest.main()
