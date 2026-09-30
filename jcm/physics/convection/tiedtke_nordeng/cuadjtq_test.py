"""Tests for the Tiedtke-Nordeng saturation: ECHAM's ``ua`` table and ``cuadjtq``.

The reference for :func:`cuadjtq` is ECHAM itself:
``jcm/data/test/echam_cuadjtq_reference/`` holds what ECHAM6.3's compiled
``mo_cuadjust.f90::cuadjtq`` returns for 55 designed parcels in each of its
three ``kcall`` modes (provenance in the README there), once with ECHAM's
spline lookup tables and once with the tables replaced by the Sonntag fit
they tabulate. jcm runs with ECHAM's constants for the comparison.

Tolerances:

* float64 against the analytic-table build: the two evaluate the same
  arithmetic, so they differ by rounding alone (measured: 5.7e-14 K, one
  ulp of a 300 K temperature, and 4.8e-15 of the humidity). The bounds are
  1e-12 K and 1e-13 of the humidity, ~20 ulps.
* float64 against ECHAM as it runs: ECHAM's own spline tables move its
  result from the analytic one by up to 2.4e-11 K and 4.5e-13 of the
  humidity (the tables' interpolation error carried through the two steps);
  the bounds are 1e-10 K and 2e-12, 4x that.
* float32: the inputs round to float32 (1.5e-5 K on a 300 K parcel) and
  ``q − qs`` cancels about one decimal of the humidity. Measured: 3.9e-5 K
  and 1.6e-6 of the humidity; the bounds are 1e-4 K (~3 ulps of 300 K) and
  1e-5.

Returning the first Newton step alone moves the temperature by up to 2.7 K
from ECHAM's result, and the refinement with other clips or step counts by
6e-9 to 2 K on these parcels, so the float64 tolerances resolve ECHAM's
second step.
"""

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics import thermodynamics
from jcm.physics.convection.tiedtke_nordeng.cuadjtq import (
    cuadjtq,
    cuadjtq_newton,
    cuadjtq_newton_evap,
    lcp_ua,
    saturation_mixing_ratio,
)
from jcm.testing import check_gradients

_REFERENCE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "data",
    "test", "echam_cuadjtq_reference", "echam_cuadjtq.npz")

# (T atol [K], q rtol) per comparison; see the module docstring.
_TOL = {
    ("float64", "analytic_"): (1.0e-12, 1.0e-13),
    ("float64", ""): (1.0e-10, 2.0e-12),
    ("float32", "analytic_"): (1.0e-4, 1.0e-5),
    ("float32", ""): (1.0e-4, 1.0e-5),
}


def _reference():
    with np.load(_REFERENCE) as z:
        return {k: z[k] for k in z.files}


@contextlib.contextmanager
def _echam_constants(ref):
    """ECHAM's ``rv``, ``alv``, ``als`` for the duration (``rd``, ``cpd`` and
    ``tmelt`` already agree), then jcm's own again.
    """
    saved = c.physical_constants
    c.set_constants(rv=float(ref["echam_rv"]), alhc=float(ref["echam_alv"]),
                    alhs=float(ref["echam_als"]), cpd=float(ref["echam_cpd"]),
                    tmelt=float(ref["echam_tmelt"]))
    try:
        yield
    finally:
        c.set_constants(saved)


def _qsat_and_dqsat_dt(temperature, pressure):
    return thermodynamics.saturation_specific_humidity_and_derivative(
        temperature, pressure)


class TestAgainstEchamCuadjtq:
    """jcm's ``cuadjtq`` against ECHAM's compiled ``mo_cuadjust.f90``."""

    def test_constants_match_echam(self):
        """Under the override jcm's derived rd and vtmpc1 are ECHAM's."""
        ref = _reference()
        with _echam_constants(ref):
            np.testing.assert_allclose(c.rd, ref["echam_rd"], rtol=1e-15)
            np.testing.assert_allclose(c.vtmpc1, ref["echam_vtmpc1"],
                                       rtol=1e-14)

    @pytest.mark.parametrize("variant", ["analytic_", ""],
                             ids=["analytic_tables", "echam_tables"])
    @pytest.mark.parametrize("precision", ["float64", "float32"])
    @pytest.mark.parametrize("kcall", [0, 1, 2])
    def test_matches_echam(self, kcall, precision, variant):
        ref = _reference()
        t_atol, q_rtol = _TOL[(precision, variant)]
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        with _echam_constants(ref), jax.enable_x64(precision == "float64"):
            t = jnp.asarray(ref["temperature_in"], dtype)
            q = jnp.asarray(ref["humidity_in"], dtype)
            p = jnp.asarray(ref["pressure"], dtype)
            t_out, q_out, cond = cuadjtq(t, q, p, kcall=kcall)
            wrapped = {1: cuadjtq_newton, 2: cuadjtq_newton_evap}.get(kcall)
            wrapped_out = None if wrapped is None else wrapped(t, q, p)
            np.testing.assert_array_equal(np.asarray(cond),
                                          np.asarray(q - q_out))
        assert t_out.dtype == dtype and q_out.dtype == dtype
        want_t = ref[f"{variant}kcall{kcall}_temperature_out"]
        want_q = ref[f"{variant}kcall{kcall}_humidity_out"]
        cases = ref["case"]
        self._assert_close(t_out, want_t, cases, f"kcall={kcall} T",
                           dict(atol=t_atol, rtol=0.0))
        self._assert_close(q_out, want_q, cases, f"kcall={kcall} q",
                           dict(atol=0.0, rtol=q_rtol))
        if wrapped_out is not None:
            # The wrappers are cuadjtq itself, bit for bit.
            np.testing.assert_array_equal(np.asarray(wrapped_out[0]),
                                          np.asarray(t_out))
            np.testing.assert_array_equal(np.asarray(wrapped_out[1]),
                                          np.asarray(q_out))
            if kcall == 1:
                np.testing.assert_array_equal(np.asarray(wrapped_out[2]),
                                              np.asarray(cond))

    @staticmethod
    def _assert_close(got, want, cases, label, kw):
        got = np.asarray(got, np.float64)
        bad = ~np.isclose(got, want, **kw)
        assert not bad.any(), (
            f"{label}: {int(bad.sum())} parcels off ECHAM, e.g. "
            + "; ".join(f"{cases[i]}: jcm {got[i]!r} ECHAM {want[i]!r}"
                        for i in np.flatnonzero(bad)[:3]))

    def test_first_step_alone_is_not_echam(self):
        """The reference resolves the refinement: one step is kelvins off."""
        ref = _reference()
        with _echam_constants(ref), jax.enable_x64(True):
            t, q, p = (jnp.asarray(ref[k]) for k in
                       ("temperature_in", "humidity_in", "pressure"))
            t1, _, _ = cuadjtq(t, q, p, kcall=1, refine=False)
            miss = np.max(np.abs(np.asarray(t1) - ref["kcall1_temperature_out"]))
        assert miss > 1.0

    def test_refinement_is_unclipped_across_tmelt(self):
        """``kcall = 2``: evaporation that cools a parcel below tmelt moves
        the refinement to the ice table and ``als``, and its step there is a
        CONDENSATION; ECHAM takes it, the refinement being unclipped.
        """
        ref = _reference()
        crossing = np.char.startswith(ref["case"], "cools across tmelt")
        with _echam_constants(ref), jax.enable_x64(True):
            t, q, p = (jnp.asarray(ref[k][crossing]) for k in
                       ("temperature_in", "humidity_in", "pressure"))
            _, q1, _ = cuadjtq(t, q, p, kcall=2, refine=False)
            _, q2, _ = cuadjtq(t, q, p, kcall=2)
            refinement = np.asarray(q1 - q2)       # the second step's zcond
        assert np.any(refinement > 0.0)
        np.testing.assert_allclose(np.asarray(q2),
                                   ref["kcall2_humidity_out"][crossing],
                                   rtol=2e-12, atol=0.0)


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


class TestCuadjtqModes:
    """The sign clips of the first step and what the refinement does."""

    def test_invalid_kcall_raises(self):
        with pytest.raises(ValueError):
            cuadjtq(jnp.array(280.0), jnp.array(1e-3), jnp.array(9e4), kcall=3)

    def test_no_adjustment_for_subsaturated_input(self):
        """``kcall = 1`` leaves subsaturated air exactly as it was: the clip
        zeroes the first step, so the refinement does not run.
        """
        T, p = jnp.array(280.0), jnp.array(90000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 0.5 * qs
        T_adj, q_adj, cond = cuadjtq(T, q, p, kcall=1)
        assert float(T_adj) == float(T)
        assert float(q_adj) == float(q)
        assert float(cond) == 0.0

    def test_kcall_2_leaves_supersaturated_air(self):
        """``kcall = 2`` leaves supersaturated air exactly as it was."""
        T, p = jnp.array(290.0), jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.3 * qs
        T_adj, q_adj, cond = cuadjtq(T, q, p, kcall=2)
        assert float(T_adj) == float(T)
        assert float(q_adj) == float(q)
        assert float(cond) == 0.0

    def test_kcall_0_allows_both_directions(self):
        """``kcall = 0`` (cuini) adjusts toward saturation either way."""
        T, p = jnp.array(290.0), jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        _, _, cond_dry = cuadjtq(T, 0.7 * qs, p, kcall=0)
        _, _, cond_wet = cuadjtq(T, 1.3 * qs, p, kcall=0)
        assert float(cond_dry) < 0.0
        assert float(cond_wet) > 0.0

    def test_modest_supersaturation_lands_at_saturation(self):
        """At 10 % supersaturation the two steps land within 1e-5 of qs."""
        T, p = jnp.array(290.0), jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        T_adj, q_adj, _ = cuadjtq(T, 1.10 * qs, p, kcall=1)
        qs_adj, _ = _qsat_and_dqsat_dt(T_adj, p)
        assert abs(float(q_adj / qs_adj) - 1.0) < 1e-5

    def test_refinement_reevaporates_the_overshoot(self):
        """At 50 % supersaturation the first step over-condenses (qs is
        convex in T); the unclipped refinement re-evaporates part of it and
        leaves the parcel closer to, and still just below, saturation.
        """
        T, p = jnp.array(290.0), jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.5 * qs
        T1, q1, cond1 = cuadjtq(T, q, p, kcall=1, refine=False)
        T2, q2, cond2 = cuadjtq(T, q, p, kcall=1)
        rh1 = float(q1 / _qsat_and_dqsat_dt(T1, p)[0])
        rh2 = float(q2 / _qsat_and_dqsat_dt(T2, p)[0])
        assert float(cond2) < float(cond1)
        assert rh1 < rh2 <= 1.0

    def test_moist_static_energy_conserved(self):
        """Each step moves ``(L/cp)·zcond`` of heat for ``zcond`` of water,
        so ``cpd·T + L·q`` is conserved where L does not switch.
        """
        T, p = jnp.array(290.0), jnp.array(80000.0)
        qs, _ = _qsat_and_dqsat_dt(T, p)
        q = 1.5 * qs
        T_adj, q_adj, _ = cuadjtq(T, q, p, kcall=1)
        h_before = c.cpd * T + c.alhc * q
        h_after = c.cpd * T_adj + c.alhc * q_adj
        assert float(jnp.abs(h_after - h_before) / h_before) < 1e-6


class TestCuadjtqNewton:
    """The updraft and wet-bulb wrappers on the ``ua`` saturation."""

    def test_converges_to_saturation_and_conserves_water(self):
        with jax.enable_x64(True):
            # None of the parcels warms across tmelt, where each Newton
            # step takes L from its own temperature.
            T = jnp.array([250.0, 265.0, 290.0])
            p = jnp.array([4.0e4, 7.0e4, 9.0e4])
            q = 1.3 * saturation_mixing_ratio(p, T)
            T_adj, vap, liq = cuadjtq_newton(T, q, p)
            np.testing.assert_allclose(np.asarray(vap + liq), np.asarray(q),
                                       rtol=1e-14)
            # ECHAM's two steps leave a second-order residual below
            # saturation: 1.8e-7 at 250 K to 5.9e-5 at 290 K here.
            residual = np.asarray(vap / saturation_mixing_ratio(p, T_adj) - 1)
            assert np.all(residual <= 0.0)
            assert np.all(residual > -1e-4)
            # The released heat is L/cp of the condensate, with the ua phase.
            np.testing.assert_allclose(
                np.asarray(T_adj - T),
                np.asarray(lcp_ua(T_adj) * liq), rtol=1e-12)

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

    ``cuadjtq`` is a fixed two-step Newton solve — no convergence branch, no
    ``lax.while_loop`` — which makes it one of the better finite-difference
    candidates in the package. Supersaturated (condensing) and subsaturated
    (clipped to no change) parcels are checked; the function is elementwise,
    so a parcel vector is the natural shape.
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
        saturation the first step's ``max(zcond, 0)`` sits on its hinge,
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

    @pytest.mark.parametrize("kcall", [0, 2])
    def test_other_modes_gradients(self, kcall):
        """The environment (``kcall = 0``) and wet-bulb (``kcall = 2``)
        modes, on subsaturated warm parcels where both take two steps.
        """
        with jax.enable_x64(True):
            temperature, pressure = self._parcels()
            humidity = 0.6 * saturation_mixing_ratio(pressure, temperature)
            check_gradients(
                lambda t, q, p: cuadjtq(t, q, p, kcall=kcall)[:2],
                (temperature, humidity, pressure), rtol=1e-4, seed=0)

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
