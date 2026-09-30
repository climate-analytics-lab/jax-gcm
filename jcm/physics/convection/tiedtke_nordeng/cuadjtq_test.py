"""Tests for the Tiedtke-Nordeng saturation: ECHAM's ``ua`` table and ``cuadjtq``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics import thermodynamics
from jcm.physics.convection.tiedtke_nordeng.cuadjtq import (
    lcp_ua,
    cuadjtq_newton,
    cuadjtq_newton_evap,
    saturation_mixing_ratio,
)
from jcm.testing import check_gradients


class TestUaSaturation:
    """Convection reads ECHAM's ``ua`` table: ice at and below tmelt."""

    def test_mixing_ratio_is_ua_qsat(self):
        with jax.enable_x64(True):
            T = jnp.array([230.0, c.tmelt, 273.2, 300.0])
            p = jnp.array([3.0e4, 7.0e4, 8.0e4, 1.0e5])
            es = jnp.where(T <= c.tmelt, thermodynamics.es_ice(T),
                           thermodynamics.es_water(T))
            x = es * c.rd / c.rv / p
            np.testing.assert_allclose(
                np.asarray(saturation_mixing_ratio(p, T)),
                np.asarray(x / (1.0 - c.vtmpc1 * x)), rtol=1e-14)

    def test_latent_heat_switches_with_the_table(self):
        T = jnp.array([c.tmelt - 1.0, c.tmelt, c.tmelt + 1.0])
        np.testing.assert_allclose(
            np.asarray(lcp_ua(T)),
            np.array([c.alhs, c.alhs, c.alhc]) / c.cpd, rtol=1e-6)


class TestCuadjtqNewton:
    """The damped Newton adjustment on the ``ua`` saturation."""

    def test_converges_to_saturation_and_conserves_water(self):
        with jax.enable_x64(True):
            # None of the parcels warms across tmelt, where each Newton
            # pass takes L from its own temperature.
            T = jnp.array([250.0, 265.0, 290.0])
            p = jnp.array([4.0e4, 7.0e4, 9.0e4])
            q = 1.3 * saturation_mixing_ratio(p, T)
            T_adj, vap, liq = cuadjtq_newton(T, q, p)
            np.testing.assert_allclose(np.asarray(vap + liq), np.asarray(q),
                                       rtol=1e-14)
            np.testing.assert_allclose(
                np.asarray(vap), np.asarray(saturation_mixing_ratio(p, T_adj)),
                rtol=1e-6)
            # The released heat is L/cp of the condensate, with the ua phase.
            np.testing.assert_allclose(
                np.asarray(T_adj - T),
                np.asarray(lcp_ua(T_adj) * liq), rtol=1e-3)

    def test_wet_bulb_conserves_moist_static_energy(self):
        with jax.enable_x64(True):
            T = jnp.array([260.0, 285.0])
            p = jnp.array([6.0e4, 9.0e4])
            q = 0.5 * saturation_mixing_ratio(p, T)
            T_wb, q_wb = cuadjtq_newton_evap(T, q, p)
            L_cp = lcp_ua(T)
            np.testing.assert_allclose(
                np.asarray(T_wb - T), np.asarray(-L_cp * (q_wb - q)),
                rtol=1e-12)
            assert bool(jnp.all(q_wb > q))


class TestCuadjtqNewtonGradients:
    """AD against a central difference for the saturation adjustment (#820).

    ``cuadjtq_newton`` is a fixed-length Newton solve — no convergence
    branch, no ``lax.while_loop`` — which makes it one of the better
    finite-difference candidates in the package. Both the supersaturated
    branch (condensation) and the subsaturated one (re-evaporation, bounded
    by the available liquid) are checked; the function is elementwise, so a
    parcel vector is the natural shape.
    """

    @staticmethod
    def _parcels():
        """Return (temperature, pressure) for four warm parcels."""
        return (jnp.array([283.0, 288.0, 292.0, 297.0]),
                jnp.array([9.5e4, 9.0e4, 8.0e4, 7.0e4]))

    @pytest.mark.parametrize("saturation_ratio", [1.25, 0.6])
    @pytest.mark.parametrize("seed", [0, 4])
    def test_gradients_match_a_central_difference(self, saturation_ratio,
                                                  seed):
        """Supersaturated and subsaturated parcels alike.

        The ratios are 1.25 and 0.6 rather than anything near 1.0: at exactly
        saturation the first pass's ``max(condensate, 0)`` sits on its hinge,
        which is a one-sided point rather than a defect in the solve.
        """
        temperature, pressure = self._parcels()
        total_water = saturation_ratio * saturation_mixing_ratio(
            pressure, temperature)
        check_gradients(cuadjtq_newton,
                        (temperature, total_water, pressure),
                        rtol=1e-3, seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    def test_ice_branch_gradients(self, seed):
        """Cold parcels, on the ``ua`` table's ice fit and ``als/cpd``.

        250-266 K and supersaturated by 25 %, so no parcel warms across
        tmelt, where the phase of both switches. Checked in float64: with
        ~1 g/kg of water the step on q moves T by ~1e-3 K, a few float32
        ulps of a 250 K temperature, so the float32 secant is round-off; in
        float64 AD and a centred difference agree to 1e-9 here.
        """
        with jax.enable_x64(True):
            temperature = jnp.array([250.0, 255.0, 260.0, 266.0],
                                    dtype=jnp.float64)
            pressure = jnp.array([5.0e4, 6.0e4, 7.0e4, 8.0e4],
                                 dtype=jnp.float64)
            total_water = 1.25 * saturation_mixing_ratio(pressure, temperature)
            check_gradients(cuadjtq_newton,
                            (temperature, total_water, pressure),
                            rtol=1e-4, seed=seed)

    def test_gradients_finite_at_the_zero_pressure_top(self):
        """The model-top interface has p = 0, which callers pass in."""
        temperature = jnp.array([200.0, 220.0])
        pressure = jnp.array([0.0, 0.0])
        total_water = jnp.array([1.0e-6, 2.0e-6])
        grads = jax.grad(
            lambda t, q, p: sum(jnp.sum(x) for x in cuadjtq_newton(t, q, p)),
            argnums=(0, 1, 2))(temperature, total_water, pressure)
        for g in grads:
            assert bool(jnp.all(jnp.isfinite(g)))
