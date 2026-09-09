"""Tests for the modal emission distributor."""

import math
import unittest

import jax.numpy as jnp
import numpy as np

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

    def test_emission_diameter_gives_cams_mass_to_number(self):
        """m_p = rho (pi/6) D^3, i.e. CESM's x_mton = 6/(pi rho D^3) (#768)."""
        rho = 1700.0
        m = particle_mean_mass(MAM4_SPEC.mode("primary_carbon"), rho, 0.134e-6)
        self.assertAlmostEqual(1.0 / m / 4.669e17, 1.0, places=3)
        self.assertAlmostEqual(m, rho * math.pi / 6.0 * 0.134e-6 ** 3, places=24)

    def test_emission_diameter_changes_the_emitted_number_only(self):
        air_density = jnp.full((3, 2), 1.2)
        dz = jnp.full((3, 2), 100.0)
        flux = jnp.full((2,), 1.0e-10)
        default = distribute_surface_flux(
            MAM4_SPEC, [("du", "acc", flux)], air_density, dz)
        sized = distribute_surface_flux(
            MAM4_SPEC, [("du", "acc", flux, 0.7806e-6)], air_density, dz)
        mname, nname = mass_name("du", "acc"), number_name("acc")
        np.testing.assert_allclose(np.asarray(sized[mname]),
                                   np.asarray(default[mname]))
        # The mode's 0.11 um equilibrium size emits ~75x too many particles.
        ratio = float(default[nname][-1, 0]) / float(sized[nname][-1, 0])
        self.assertAlmostEqual(ratio / 75.5, 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
