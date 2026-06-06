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


if __name__ == "__main__":
    unittest.main()
