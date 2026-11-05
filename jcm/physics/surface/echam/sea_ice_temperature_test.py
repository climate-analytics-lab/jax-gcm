"""Tests for the prognostic sea-ice surface temperature (ECHAM ``s_sicetemp``)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.surface.echam.sea_ice_temperature import (
    SeaIceTemperatureParameters,
    carried_ice_temperature,
    ice_thickness,
    melt_cap,
    update_ice_surface_temperature,
)
from jcm.testing import check_gradients

P = SeaIceTemperatureParameters.default()
DT = 720.0


def _fortran_step(tsi, zsflx, siced, dt):
    """``s_sicetemp`` (uncoupled branch, psni = 0) written out in plain floats."""
    alice, rhoice, cpice, dice, ctfreez = 2.1656, 917.0, 2106.0, 0.05, 271.38
    tmelt = 273.15
    zcpdt = rhoice * cpice * dice / dt
    zsniced = siced
    zicefl = alice * ctfreez / zsniced
    ptsi = (zcpdt * tsi + zsflx + zicefl) / (zcpdt + alice / zsniced)
    if ptsi > tmelt:
        pqres = (zcpdt + alice / zsniced) * (ptsi - tmelt)
        ptsi = tmelt
    else:
        pqres = 0.0
    pahfice = alice * (ptsi - ctfreez) / zsniced
    return ptsi, pahfice, pqres


class TestSicetemp(unittest.TestCase):

    def setUp(self):
        # Compare against the plain-float Fortran formula to round-off.
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        self.addCleanup(jax.config.update, "jax_enable_x64", previous)

    def test_matches_hand_computed_fortran_step(self):
        for tsi, flux, h in [(255.0, -40.0, 2.0), (268.0, 15.0, 1.0), (271.38, 0.0, 2.0)]:
            want = _fortran_step(tsi, flux, h, DT)
            got = update_ice_surface_temperature(
                jnp.float64(tsi), jnp.float64(flux), jnp.float64(h), DT, P)
            np.testing.assert_allclose([float(x) for x in got], want, rtol=1e-12, atol=1e-10)

    def test_hand_number(self):
        # C/dt = 917*2106*0.05/720 = 134.116..., k = 2.1656/2 = 1.0828.
        t, cond, melt = update_ice_surface_temperature(
            jnp.float64(260.0), jnp.float64(-20.0), jnp.float64(2.0), DT, P)
        cdt, k = 917.0 * 2106.0 * 0.05 / 720.0, 2.1656 / 2.0
        np.testing.assert_allclose(float(t), (cdt * 260.0 - 20.0 + k * 271.38) / (cdt + k))
        np.testing.assert_allclose(float(cond), k * (float(t) - 271.38))
        self.assertEqual(float(melt), 0.0)

    def test_equilibrium_is_the_conduction_balance(self):
        # A steady net flux F (positive down) into the surface balances the
        # conduction k·(T − T_freeze) from below: T = T_freeze + F/k.
        for flux, h in [(-5.0, 2.0), (-2.0, 1.0), (0.5, 1.0)]:
            t = jnp.float64(270.0)
            for _ in range(4000):
                t, _, _ = update_ice_surface_temperature(
                    t, jnp.float64(flux), jnp.float64(h), DT, P)
            want = 271.38 + flux * h / 2.1656
            np.testing.assert_allclose(float(t), want, atol=1e-6)

    def test_melt_cap(self):
        t, cond, melt = update_ice_surface_temperature(
            jnp.float64(273.0), jnp.float64(300.0), jnp.float64(1.0), DT, P)
        self.assertEqual(float(t), c.tmelt)
        self.assertGreater(float(melt), 0.0)
        np.testing.assert_allclose(float(cond), 2.1656 * (c.tmelt - 271.38))
        # The cap passes a cold temperature untouched.
        np.testing.assert_allclose(float(melt_cap(jnp.float64(260.0), P)), 260.0)

    def test_thickness_by_hemisphere(self):
        h = ice_thickness(jnp.array([True, False]), P)
        np.testing.assert_allclose(h, [2.0, 1.0])

    def test_carried_temperature_rules(self):
        sst = jnp.array([275.0, 270.0, 280.0, 272.0])
        frac = jnp.array([0.5, 0.5, 0.0, 0.5])
        carried = jnp.array([260.0, 0.0, 250.0, 265.0])
        np.testing.assert_allclose(
            carried_ice_temperature(carried, sst, frac), [260.0, 270.0, 280.0, 265.0])
        np.testing.assert_allclose(
            carried_ice_temperature(None, sst, frac), [271.38, 270.0, 280.0, 271.38])

    def test_gradient_through_step(self):
        # Away from the melt cap the derivative is the exact one.
        def step(temperature, flux, thickness):
            return update_ice_surface_temperature(temperature, flux, thickness, DT, P)[:2]

        check_gradients(step, (jnp.full(3, 260.0), jnp.array([-30.0, -5.0, 10.0]),
                               jnp.array([2.0, 1.0, 2.0])), rtol=1e-3)

    def test_melt_cap_surrogate_gradient_is_live_and_value_exact(self):
        t = jnp.float64(274.0)
        self.assertEqual(float(melt_cap(t, P)), c.tmelt)
        g = jax.grad(lambda x: melt_cap(x, P))(jnp.float64(c.tmelt + 0.2))
        self.assertTrue(np.isfinite(float(g)))
        self.assertGreater(float(g), 0.0)
        # A surface above the melting point sees a smooth, non-zero response
        # to the energy it is given, where the exact min has slope 0.
        g_step = jax.grad(lambda f: update_ice_surface_temperature(
            jnp.float64(273.1), f, jnp.float64(1.0), DT, P)[0])(jnp.float64(50.0))
        self.assertGreater(float(g_step), 0.0)

    def test_float32_dtype_preserved(self):
        t, cond, melt = update_ice_surface_temperature(
            jnp.float32(260.0), jnp.float32(-10.0), jnp.float32(2.0), 720.0, P)
        self.assertEqual(t.dtype, jnp.float32)


if __name__ == "__main__":
    unittest.main()
