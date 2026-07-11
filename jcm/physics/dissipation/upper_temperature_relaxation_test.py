"""Tests for the upper-atmosphere temperature/wind relaxation term."""

import unittest

import numpy as np
import jax.numpy as jnp

from jcm.forcing import ForcingData
from jcm.physics.dissipation.upper_temperature_relaxation import (
    UpperTemperatureRelaxation,
)
from jcm.physics_interface import PhysicsState


def _state(nlev=12, ncols=3, t0=150.0, u0=80.0):
    shape = (nlev, ncols)
    return PhysicsState.zeros(
        shape,
        temperature=jnp.full(shape, t0),
        u_wind=jnp.full(shape, u0),
        v_wind=jnp.full(shape, -u0 / 2.0),
        normalized_surface_pressure=jnp.ones((ncols,)),
    )


class UpperTemperatureRelaxationTest(unittest.TestCase):
    def test_temperature_relaxation_rates(self):
        nlev = 12
        t_ref = np.full(nlev, 190.0)
        term = UpperTemperatureRelaxation(
            t_ref, n_levels=8, timescale_s=21600.0, ramp=2.5)
        tend, diags = term(_state(nlev), {}, ForcingData.zeros((3,)), None)
        dtdt = np.asarray(tend.temperature)
        # Top level: exactly (190-150)/21600.
        np.testing.assert_allclose(dtdt[0], (190.0 - 150.0) / 21600.0,
                                   rtol=1e-6)
        # Ramp: each level down is 2.5x slower.
        np.testing.assert_allclose(dtdt[1], dtdt[0] / 2.5, rtol=1e-5)
        # Below the sponge: untouched.
        np.testing.assert_array_equal(dtdt[8:], 0.0)
        # Winds untouched by default.
        np.testing.assert_array_equal(np.asarray(tend.u_wind), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.v_wind), 0.0)
        self.assertIn("upper_t_relaxation", diags)

    def test_wind_rayleigh_damping(self):
        nlev = 12
        t_ref = np.full(nlev, 190.0)
        term = UpperTemperatureRelaxation(
            t_ref, n_levels=8, timescale_s=21600.0, ramp=2.5,
            wind_timescale_s=43200.0)
        state = _state(nlev, u0=80.0)
        tend, _ = term(state, {}, ForcingData.zeros((3,)), None)
        dudt = np.asarray(tend.u_wind)
        dvdt = np.asarray(tend.v_wind)
        # Top level: -u / tau exactly; v scales with v.
        np.testing.assert_allclose(dudt[0], -80.0 / 43200.0, rtol=1e-6)
        np.testing.assert_allclose(dvdt[0], 40.0 / 43200.0, rtol=1e-6)
        np.testing.assert_allclose(dudt[1], dudt[0] / 2.5, rtol=1e-5)
        # Below the sponge: untouched.
        np.testing.assert_array_equal(dudt[8:], 0.0)
        # Temperature relaxation unchanged by enabling the wind damping.
        np.testing.assert_allclose(
            np.asarray(tend.temperature)[0], (190.0 - 150.0) / 21600.0,
            rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
