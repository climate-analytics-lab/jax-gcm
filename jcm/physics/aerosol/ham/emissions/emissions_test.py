"""Phase 4 tests: emission sources, distributor, and the term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.ham import MAM4_SPEC, mass_name, number_name
from jcm.physics.aerosol.ham.emissions.distributors import (
    distribute_surface_flux,
    particle_mean_mass,
)
from jcm.physics.aerosol.ham.emissions.emissions_term import (
    EmissionParameters,
    HamEmissions,
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


class EmissionTermTest(unittest.TestCase):
    def _state(self, nlev=4, ncols=2, wind=8.0):
        from jcm.physics_interface import PhysicsState

        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0),
            u_wind=jnp.full((nlev, ncols), wind),
        )
        diagnostics = {
            "air_density": jnp.full((nlev, ncols), 1.2),
            "layer_thickness": jnp.full((nlev, ncols), 100.0),
        }
        return state, diagnostics

    def test_seasalt_emitted_over_ocean(self):
        state, diagnostics = self._state()
        term = HamEmissions()
        tend, _ = term(state, diagnostics, None, None)  # terrain None -> ocean
        ss = tend.tracers[mass_name("ss", "cor")]
        self.assertGreater(float(ss[-1, 0]), 0.0)

    def test_seasalt_grows_with_wind(self):
        term = HamEmissions()
        s_calm, _ = term(*self._state(wind=2.0), None, None)
        s_windy, _ = term(*self._state(wind=15.0), None, None)
        key = mass_name("ss", "cor")
        self.assertGreater(
            float(s_windy.tracers[key][-1, 0]),
            float(s_calm.tracers[key][-1, 0]),
        )

    def test_dust_zero_on_aquaplanet(self):
        state, diagnostics = self._state(wind=20.0)
        term = HamEmissions()
        tend, _ = term(state, diagnostics, None, None)
        du = tend.tracers[mass_name("du", "cor")]
        self.assertTrue(bool(jnp.all(du == 0.0)))  # land fraction 0

    def test_grad_through_seasalt_coeff(self):
        state, diagnostics = self._state()

        def loss(coeff):
            params = EmissionParameters.default()
            params = EmissionParameters(
                seasalt_coeff=coeff,
                seasalt_wind_exp=params.seasalt_wind_exp,
                seasalt_accum_frac=params.seasalt_accum_frac,
                dust_coeff=params.dust_coeff,
                dust_u_threshold=params.dust_u_threshold,
                dms_coeff=params.dms_coeff,
                volcanic_so4=params.volcanic_so4,
                biogenic_soa=params.biogenic_soa,
            )
            term = HamEmissions(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers[mass_name("ss", "cor")])

        g = jax.grad(loss)(jnp.asarray(1.0e-13))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)


if __name__ == "__main__":
    unittest.main()
