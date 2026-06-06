"""Tests for the Tegen-physics dust emission scheme."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import mass_name
from jcm.physics.aerosol.jam.emissions.dust import (
    DustEmissions,
    DustParameters,
    horizontal_flux,
)


class HorizontalFluxTest(unittest.TestCase):
    def test_zero_below_threshold(self):
        g = horizontal_flux(
            jnp.asarray(0.1), jnp.asarray(0.3), jnp.asarray(1.2), jnp.asarray(1.0)
        )
        self.assertAlmostEqual(float(g), 0.0)

    def test_grows_above_threshold(self):
        lo = horizontal_flux(jnp.asarray(0.4), jnp.asarray(0.2), jnp.asarray(1.2), jnp.asarray(1.0))
        hi = horizontal_flux(jnp.asarray(0.8), jnp.asarray(0.2), jnp.asarray(1.2), jnp.asarray(1.0))
        self.assertGreater(float(hi), float(lo))
        self.assertGreater(float(lo), 0.0)


def _inputs(nlev=3, ncols=2, u_star=0.6, source=0.5):
    from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
        VerticalDiffusionData,
    )

    state = __import__(
        "jcm.physics_interface", fromlist=["PhysicsState"]
    ).PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 295.0),
    )
    vd = VerticalDiffusionData.zeros((ncols,), nlev).copy(
        surface_friction_velocity=jnp.full((ncols,), u_star),
    )
    diagnostics = {
        "air_density": jnp.full((nlev, ncols), 1.2),
        "layer_thickness": jnp.full((nlev, ncols), 100.0),
        "vertical_diffusion": vd,
    }
    forcing = types.SimpleNamespace(dust_source=jnp.full((ncols,), source))
    return state, diagnostics, forcing, None


class DustTermTest(unittest.TestCase):
    def test_emits_with_source_and_wind(self):
        term = DustEmissions()
        tend, _ = term(*_inputs(u_star=0.6, source=0.8))
        key = mass_name("du", "cor")
        self.assertGreater(float(tend.tracers[key][-1, 0]), 0.0)

    def test_zero_without_source(self):
        term = DustEmissions()
        tend, _ = term(*_inputs(source=0.0))
        key = mass_name("du", "cor")
        self.assertAlmostEqual(float(tend.tracers[key][-1, 0]), 0.0)

    def test_zero_below_threshold(self):
        term = DustEmissions()  # default u_threshold 0.2
        tend, _ = term(*_inputs(u_star=0.1, source=0.8))
        key = mass_name("du", "cor")
        self.assertAlmostEqual(float(tend.tracers[key][-1, 0]), 0.0)

    def test_grad_through_alpha(self):
        state, diagnostics, forcing, terrain = _inputs()

        def loss(alpha):
            term = DustEmissions(
                params=DustParameters(
                    scale=jnp.asarray(1.0), alpha=alpha,
                    u_threshold=jnp.asarray(0.2),
                    accum_fraction=jnp.asarray(0.1),
                    u_star_default=jnp.asarray(0.3),
                )
            )
            tend, _ = term(state, diagnostics, forcing, terrain)
            return jnp.sum(tend.tracers[mass_name("du", "cor")])

        g = jax.grad(loss)(jnp.asarray(1.0e-5))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)


if __name__ == "__main__":
    unittest.main()
