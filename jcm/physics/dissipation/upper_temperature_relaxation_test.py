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
from jcm.testing import check_gradients


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


class UpperTemperatureRelaxationFromUssaTest(unittest.TestCase):
    """`from_ussa` builds a USSA reference profile on a hybrid grid."""

    def _boundaries(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        v = get_echam_levels(47)
        return (np.asarray(v.a_boundaries, dtype=float),
                np.asarray(v.b_boundaries, dtype=float))

    def test_t_ref_matches_direct_ussa_at_mid_pressures(self):
        from jcm.initial_states.ussa1976 import (
            ussa_pressure, ussa_temperature,
        )

        a, b = self._boundaries()
        term = UpperTemperatureRelaxation.from_ussa(
            a, b, n_levels=8, timescale_s=6.0 * 3600.0)

        # Reproduce the level mid-pressures and the log-p → z → T mapping.
        p0 = 101325.0
        p_mid = 0.5 * (a[:-1] + a[1:]) + 0.5 * (b[:-1] + b[1:]) * p0
        zs = np.linspace(0.0, 84000.0, 4000)
        ps = np.asarray(ussa_pressure(zs))
        z_of_p = np.interp(np.log(p_mid), np.log(ps[::-1]), zs[::-1])
        expected = np.asarray(ussa_temperature(z_of_p))

        t_ref = np.asarray(term._t_ref.get_value())
        np.testing.assert_allclose(t_ref, expected, rtol=1e-5)

    def test_surface_level_is_near_288K_and_troposphere_decreases(self):
        a, b = self._boundaries()
        term = UpperTemperatureRelaxation.from_ussa(
            a, b, n_levels=8, timescale_s=6.0 * 3600.0)
        p0 = 101325.0
        p_mid = 0.5 * (a[:-1] + a[1:]) + 0.5 * (b[:-1] + b[1:]) * p0
        t_ref = np.asarray(term._t_ref.get_value())

        # These grids are surface-last in the hybrid boundary order: the
        # highest-pressure mid-level sits near 1000 hPa at ~USSA surface T.
        i_sfc = int(np.argmax(p_mid))
        self.assertGreater(p_mid[i_sfc], 90000.0)
        self.assertAlmostEqual(float(t_ref[i_sfc]), 288.0, delta=3.0)

        # Through the troposphere (down to the tropopause ~ 200 hPa) T falls
        # monotonically toward the surface as pressure rises.
        order = np.argsort(p_mid)
        trop = order[p_mid[order] >= 20000.0]
        t_trop = t_ref[trop]
        self.assertTrue(np.all(np.diff(t_trop) >= -1e-6))


class UpperTemperatureRelaxationGradientTest(unittest.TestCase):
    """AD against a central difference for the relaxation (#820).

    Green. The temperature branch is linear in T and the Rayleigh branch is
    linear in u and v, so the only non-trivial term is the CAM kinetic-energy
    return ``0.5 (1 - c2^2)(u^2 + v^2) / (cp dt)``, which is a smooth
    quadratic. This is a fence, not a hunt.

    The state is given horizontal structure rather than the uniform fields
    the other tests here use: a relative finite-difference step on a uniform
    field displaces every column identically, and the per-column
    contributions to the projection would then be indistinguishable.
    """

    @staticmethod
    def _fields(shape, seed=0):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(60.0 + 15.0 * rng.standard_normal(shape),
                            jnp.float32),
                jnp.asarray(-25.0 + 10.0 * rng.standard_normal(shape),
                            jnp.float32),
                jnp.asarray(180.0 + 12.0 * rng.standard_normal(shape),
                            jnp.float32))

    @staticmethod
    def _tendencies(term, shape, ncols):
        def f(u_wind, v_wind, temperature):
            state = PhysicsState.zeros(
                shape, temperature=temperature, u_wind=u_wind, v_wind=v_wind,
                normalized_surface_pressure=jnp.ones((ncols,)))
            tendency, diagnostics = term(
                state, {"_dt_seconds": 1800.0}, ForcingData.zeros((ncols,)),
                None)
            return (tendency.u_wind, tendency.v_wind, tendency.temperature,
                    diagnostics["upper_t_relaxation"])

        return f

    def test_gradients_with_wind_damping(self):
        """Temperature relaxation, Rayleigh friction and the KE return."""
        nlev, ncols = 12, 4
        term = UpperTemperatureRelaxation(
            np.linspace(260.0, 190.0, nlev), n_levels=8,
            timescale_s=21600.0, wind_timescale_s=43200.0,
            wind_center_level=3.0, wind_range_levels=2.0)
        for seed in (0, 4):
            with self.subTest(seed=seed):
                check_gradients(
                    self._tendencies(term, (nlev, ncols), ncols),
                    self._fields((nlev, ncols), seed=seed), rtol=1e-3,
                    seed=seed)

    def test_gradients_without_wind_damping(self):
        """``wind_timescale_s=None``: the temperature-only default."""
        nlev, ncols = 12, 4
        term = UpperTemperatureRelaxation(
            np.linspace(260.0, 190.0, nlev), n_levels=8, timescale_s=21600.0)
        u_wind, v_wind, temperature = self._fields((nlev, ncols))
        f = self._tendencies(term, (nlev, ncols), ncols)
        # Only T is an input on this branch; u and v are passed through to a
        # zero tendency, so a direction along them would carry no signal and
        # the liveness of T is what there is to check.
        check_gradients(lambda t: f(u_wind, v_wind, t)[2:], (temperature,),
                        rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
