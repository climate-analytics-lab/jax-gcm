"""Tests for the upper-atmosphere temperature/wind relaxation term."""

import unittest

import numpy as np
import jax.numpy as jnp

import jcm.constants as c
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

    def test_wind_rayleigh_damping_implicit_with_heating(self):
        nlev = 12
        dt = 1800.0
        tau0 = 43200.0
        t_ref = np.full(nlev, 190.0)
        term = UpperTemperatureRelaxation(
            t_ref, n_levels=8, timescale_s=21600.0,
            wind_timescale_s=tau0, wind_center_level=3.0,
            wind_range_levels=2.0)
        state = _state(nlev, u0=80.0)
        tend, _ = term(state, {"_dt_seconds": dt},
                       ForcingData.zeros((3,)), None)
        dudt = np.asarray(tend.u_wind)
        dvdt = np.asarray(tend.v_wind)

        # CAM rayleigh_friction form: k(level) from the tanh profile,
        # Euler-backward du/dt = -k u / (1 + k dt).
        k_prof = (1.0 / tau0) * 0.5 * (
            1.0 + np.tanh((3.0 - np.arange(nlev)) / 2.0))
        c2 = 1.0 / (1.0 + k_prof * dt)
        np.testing.assert_allclose(
            dudt, np.broadcast_to((-k_prof * c2 * 80.0)[:, None], dudt.shape),
            rtol=1e-4)
        np.testing.assert_allclose(
            dvdt, np.broadcast_to((k_prof * c2 * 40.0)[:, None], dvdt.shape),
            rtol=1e-4)
        # Smooth profile: half strength at the center level.
        np.testing.assert_allclose(k_prof[3], 0.5 / tau0, rtol=1e-6)

        # Energy return: the heating equals the discrete-exact KE loss of
        # the implicit update, dT = 0.5 (1 - c2^2)(u^2 + v^2)/(cp dt).
        dtdt = np.asarray(tend.temperature)
        t_relax = (190.0 - 150.0) / (21600.0 * 2.5 ** np.arange(nlev))
        t_relax[8:] = 0.0
        expected_heat = (0.5 * (1.0 - c2 ** 2) * (80.0 ** 2 + 40.0 ** 2)
                         / (float(c.cpd) * dt))
        np.testing.assert_allclose(
            dtdt, np.broadcast_to((t_relax + expected_heat)[:, None], dtdt.shape),
            rtol=1e-4, atol=1e-9)
        self.assertTrue((expected_heat >= 0).all())

        # Unconditional stability: even a 1-second timescale cannot
        # overshoot (|u + dt du/dt| = |c2 u| <= |u|).
        strong = UpperTemperatureRelaxation(
            t_ref, n_levels=8, timescale_s=21600.0, wind_timescale_s=1.0)
        tend_s, _ = strong(state, {"_dt_seconds": dt},
                           ForcingData.zeros((3,)), None)
        u_next = 80.0 + dt * np.asarray(tend_s.u_wind)
        self.assertTrue((np.abs(u_next) <= 80.0 + 1e-6).all())
        self.assertTrue((u_next >= -1e-6).all())  # no sign flip


if __name__ == "__main__":
    unittest.main()
