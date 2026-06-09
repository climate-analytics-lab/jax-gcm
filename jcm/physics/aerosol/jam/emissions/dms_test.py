"""Tests for the Nightingale (2000) DMS emission scheme."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import mass_name
from jcm.physics.aerosol.jam.emissions.dms import (
    DmsEmissions,
    DmsParameters,
    dms_schmidt_number,
    piston_velocity,
)


class DmsFunctionTest(unittest.TestCase):
    def test_schmidt_positive_and_decreasing_with_temp(self):
        cold = float(dms_schmidt_number(jnp.asarray(5.0)))
        warm = float(dms_schmidt_number(jnp.asarray(30.0)))
        self.assertGreater(cold, 0.0)
        self.assertGreater(warm, 0.0)
        self.assertGreater(cold, warm)

    def test_piston_velocity_grows_with_wind(self):
        sc = dms_schmidt_number(jnp.asarray(20.0))
        lo = float(piston_velocity(jnp.asarray(2.0), sc))
        hi = float(piston_velocity(jnp.asarray(12.0), sc))
        self.assertGreaterEqual(hi, lo)
        self.assertGreater(hi, 0.0)


def _inputs(nlev=3, ncols=2, wind=8.0, dms=1.0e-6):
    state = __import__(
        "jcm.physics_interface", fromlist=["PhysicsState"]
    ).PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 290.0),
        u_wind=jnp.full((nlev, ncols), wind),
    )
    diagnostics = {
        "air_density": jnp.full((nlev, ncols), 1.2),
        "layer_thickness": jnp.full((nlev, ncols), 100.0),
    }
    forcing = types.SimpleNamespace(
        sea_surface_temperature=jnp.full((ncols,), 293.0),
        dms_seawater=jnp.full((ncols,), dms),
    )
    terrain = types.SimpleNamespace(fmask=jnp.zeros((ncols,)))
    return state, diagnostics, forcing, terrain


class DmsTermTest(unittest.TestCase):
    def test_emits_sulfate_with_seawater_field(self):
        term = DmsEmissions()
        tend, _ = term(*_inputs(dms=2.0e-6))
        key = mass_name("so4", "acc")
        self.assertGreater(float(tend.tracers[key][-1, 0]), 0.0)

    def test_zero_without_seawater_field(self):
        term = DmsEmissions()
        tend, _ = term(*_inputs(dms=0.0))
        key = mass_name("so4", "acc")
        self.assertAlmostEqual(float(tend.tracers[key][-1, 0]), 0.0)

    def test_grad_through_flux_scale(self):
        state, diagnostics, forcing, terrain = _inputs()

        def loss(scale):
            term = DmsEmissions(
                params=DmsParameters(
                    flux_scale=scale, aitken_fraction=jnp.asarray(0.5)
                )
            )
            tend, _ = term(state, diagnostics, forcing, terrain)
            return jnp.sum(tend.tracers[mass_name("so4", "acc")])

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))


if __name__ == "__main__":
    unittest.main()
