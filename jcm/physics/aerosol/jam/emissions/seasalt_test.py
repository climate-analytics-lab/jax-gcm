"""Tests for the Gong (2003) sea-salt emission scheme."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import mass_name, number_name
from jcm.physics.aerosol.jam.emissions.seasalt import (
    SeaSaltEmissions,
    SeaSaltParameters,
    gong_mode_factors,
)


class GongFactorTest(unittest.TestCase):
    def test_factors_positive(self):
        f = gong_mode_factors(1900.0)
        for v in f.values():
            self.assertGreater(v, 0.0)

    def test_coarse_dominates_mass_accum_dominates_number(self):
        f = gong_mode_factors(1900.0)
        # Coarse particles carry far more mass per particle…
        self.assertGreater(f["mass_cor"], f["mass_acc"])
        # …while the accumulation range holds more particles (number).
        self.assertGreater(f["numb_acc"], f["numb_cor"])


def _inputs(nlev=3, ncols=2, wind=10.0, land=0.0, sice=0.0):
    state = __import__(
        "jcm.physics_interface", fromlist=["PhysicsState"]
    ).PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 285.0),
        u_wind=jnp.full((nlev, ncols), wind),
    )
    diagnostics = {
        "air_density": jnp.full((nlev, ncols), 1.2),
        "layer_thickness": jnp.full((nlev, ncols), 100.0),
    }
    terrain = types.SimpleNamespace(fmask=jnp.full((ncols,), land))
    forcing = types.SimpleNamespace(sice_am=jnp.full((ncols,), sice))
    return state, diagnostics, forcing, terrain


class SeaSaltTermTest(unittest.TestCase):
    def test_emits_over_ocean(self):
        term = SeaSaltEmissions()
        tend, _ = term(*_inputs(land=0.0))
        for key in (mass_name("ss", "cor"), number_name("acc")):
            self.assertGreater(float(tend.tracers[key][-1, 0]), 0.0)
            self.assertTrue(bool(jnp.all(tend.tracers[key][:-1] == 0.0)))

    def test_zero_over_land_and_ice(self):
        term = SeaSaltEmissions()
        land_tend, _ = term(*_inputs(land=1.0))
        ice_tend, _ = term(*_inputs(land=0.0, sice=1.0))
        key = mass_name("ss", "cor")
        self.assertAlmostEqual(float(land_tend.tracers[key][-1, 0]), 0.0)
        self.assertAlmostEqual(float(ice_tend.tracers[key][-1, 0]), 0.0)

    def test_grows_with_wind(self):
        term = SeaSaltEmissions()
        key = mass_name("ss", "cor")
        calm, _ = term(*_inputs(wind=3.0))
        windy, _ = term(*_inputs(wind=15.0))
        self.assertGreater(
            float(windy.tracers[key][-1, 0]), float(calm.tracers[key][-1, 0])
        )

    def test_magnitude_plausible(self):
        # Coarse sea-salt mass flux at u10=10 m/s, full ocean: O(1e-10..1e-8).
        term = SeaSaltEmissions()
        tend, _ = term(*_inputs(wind=10.0))
        # back out the surface mass flux [kg/m²/s] = dq*rho*dz
        dq = float(tend.tracers[mass_name("ss", "cor")][-1, 0])
        flux = dq * 1.2 * 100.0
        self.assertTrue(1e-11 < flux < 1e-7)

    def test_grad_through_scale(self):
        state, diagnostics, forcing, terrain = _inputs()

        def loss(scale):
            term = SeaSaltEmissions(
                params=SeaSaltParameters(
                    scale=scale, wind_exponent=jnp.asarray(3.41)
                )
            )
            tend, _ = term(state, diagnostics, forcing, terrain)
            return jnp.sum(tend.tracers[mass_name("ss", "cor")])

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)


if __name__ == "__main__":
    unittest.main()
