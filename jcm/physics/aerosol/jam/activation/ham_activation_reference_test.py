"""jcm's HAM M7 activation against the compiled ECHAM6.3-HAM2.3 routines (#1017).

``jcm/data/test/echam_cloud_reference/ham_activ_M7.npz`` holds the outputs of
the UNMODIFIED (extracted, see the harness provenance) r7492 activation
routines (``ham_activ_koehler_ab``, ``activ_updraft``, both the single-updraft
and 20-bin West et al. 2013 PDF runs, ``ham_activ_abdulrazzak_ghan``,
``ham_avail_activ_lin_leaitch`` + ``activ_lin_leaitch``) on 20 designed
single-level M7 cells (``ham_activ_README.md``, ``ham_activ_provenance.json``).
This module builds the same M7-shaped population locally (the pattern of
``ice_nucleation/ham_freezing_reference_test.py::m7_spec``) and compares the
pure functions AND the :class:`HamActivation` term against those numbers.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.activation.ham_activation import (
    PDF_DEFAULT_BINS,
    ham_arg,
    ham_logtail,
    ham_updraft,
    koehler_ab,
    lin_leaitch,
)
from jcm.physics.aerosol.jam.activation.ham_activation_term import (
    HamActivation,
)
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.population import AerosolMode, AerosolSpecies, ModalAerosolSpec
from jcm.physics_interface import PhysicsState

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "ham_activ_M7.npz")

MODES = ("nucs", "aits", "accs", "coas", "aiti", "acci", "coai")
SPECIES = ("so4", "bc", "oc", "ss", "du")
MEMBER = {"nucs": ("so4",), "aits": ("so4", "bc", "oc"),
          "accs": ("so4", "bc", "oc", "ss", "du"), "coas": ("so4", "bc", "oc", "ss", "du"),
          "aiti": ("bc", "oc"), "acci": ("du",), "coai": ("du",)}
# M7 can-activate (ARG lactivation) and sigma_g -- the #1017 design doc table.
CAN_ACTIVATE = {"nucs": False, "aits": True, "accs": True, "coas": True,
                "aiti": False, "acci": False, "coai": False}
SIGMA_G = {"nucs": 1.59, "aits": 1.59, "accs": 1.59, "coas": 2.0,
           "aiti": 1.59, "acci": 1.59, "coai": 2.0}


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def m7_spec():
    """Local M7 population: only species membership and can_activate enter
    the activation routines; size-bound fields are harness placeholders (the
    same documented limitation as ``ham_freezing_reference_test.py::m7_spec``).
    """
    z = load()
    species = tuple(
        AerosolSpecies(s, float(z[f"in/moleweight/{s}"]) * 1.0e-3,
                        float(z[f"in/density/{s}"]), 0.1)
        for s in SPECIES
    )
    modes = tuple(
        AerosolMode(m, m, SIGMA_G[m], 1.0e-7, 1.0e-8, 1.0e-6, MEMBER[m],
                    soluble=m not in ("aiti", "acci", "coai"),
                    can_activate=CAN_ACTIVATE[m], sediments=True)
        for m in MODES
    )
    return ModalAerosolSpec(modes=modes, species=species)


def _g(z, key, n, dtype):
    return jnp.asarray(z[key], dtype) if key in z else jnp.zeros(n, dtype)


def _inputs(prec):
    z = load()
    dtype = jnp.float64 if prec == "float64" else jnp.float32
    n = len(z["meta/names"])
    rdry = jnp.stack([_g(z, f"in/rdry/{m}", n, dtype) for m in MODES])
    rwet = jnp.stack([_g(z, f"in/rwet/{m}", n, dtype) for m in MODES])
    number = jnp.stack([_g(z, f"in/number/{m}", n, dtype) for m in MODES])
    rho = _g(z, "in/rho", n, dtype)
    number_vol = number * rho[jnp.newaxis, :]
    mass = {
        (sp, m): _g(z, f"in/mass/{m}/{sp}", n, dtype)
        for m in MODES for sp in MEMBER[m]
    }
    sigma_g = jnp.asarray([SIGMA_G[m] for m in MODES], dtype)
    can_activate = jnp.asarray([CAN_ACTIVATE[m] for m in MODES])
    return dict(
        z=z, n=n, dtype=dtype, rdry=rdry, rwet=rwet, number_vol=number_vol, mass=mass,
        sigma_g=sigma_g, can_activate=can_activate,
        t=_g(z, "in/t", n, dtype), p=_g(z, "in/p", n, dtype), q=_g(z, "in/q", n, dtype),
        esw=_g(z, "in/esw", n, dtype), tke=_g(z, "in/tke", n, dtype),
        omega=_g(z, "in/omega", n, dtype), rho=rho,
    )


# Every field here is closed-form against the harness once ``ham_logtail``
# calls HAM's own m7_cumulative_normal (not erf) and Lin & Leaitch's crcut
# constants carry the reference source's single-precision-literal rounding
# (see ham_activation.py's LL_CRCUT_STRAT/_CONV and
# _m7_cumulative_normal docstrings) -- every field below is measured at
# float64 round-off (see the "Precision" section at the bottom of this
# file for the achieved max per field).
# float32's own eps is ~1.19e-7; the deepest chain here (ham_arg's Cody
# rational-fraction evaluation inside ham_logtail, several multiply-adds
# then a power) measures up to ~4.8e-6 relative on one mode/cell -- 2e-6 was
# too tight for that specific chain once it was no longer masking a real
# (f64) discrepancy; 1e-5 clears the measured max with headroom.
RTOL = {"float64": 1e-12, "float32": 1e-5}
# atol=0 throughout except where the harness's own reference value is
# exactly 0 (a masked/non-activating mode, a gated-off cell): there a tiny
# floor (not a precision concession -- the compared values are themselves
# exact zeros) avoids a spurious "0 != 1e-300"-style failure from a
# subnormal residual.
ATOL_ZERO = 1e-30


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_koehler_ab_matches_reference(prec):
    with jax.enable_x64(prec == "float64"):
        d = _inputs(prec)
        a, b = koehler_ab(m7_spec(), d["mass"], d["t"])
        for i, m in enumerate(MODES):
            np.testing.assert_allclose(
                np.asarray(a[i], np.float64), d["z"][f"out/a/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"A[{m}]")
            np.testing.assert_allclose(
                np.asarray(b[i], np.float64), d["z"][f"out/b/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"B[{m}]")


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_ham_updraft_nactivpdf0_matches_reference(prec):
    with jax.enable_x64(prec == "float64"):
        d = _inputs(prec)
        w, pwpdf = ham_updraft(d["tke"], d["omega"], d["rho"], 0.0, 0.7, n_pdf_bins=None)
        np.testing.assert_allclose(
            np.asarray(w[0], np.float64), d["z"]["out/w0"],
            rtol=RTOL[prec], atol=ATOL_ZERO)
        assert bool(jnp.all(pwpdf == 1.0))


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_ham_arg_nactivpdf0_matches_reference(prec):
    with jax.enable_x64(prec == "float64"):
        d = _inputs(prec)
        w, pwpdf = ham_updraft(d["tke"], d["omega"], d["rho"], 0.0, 0.7, n_pdf_bins=None)
        a, b = koehler_ab(m7_spec(), d["mass"], d["t"])
        cdncact, nfrac, nact, sm, smax, rc = ham_arg(
            d["rdry"], d["number_vol"], a, b, d["can_activate"], d["sigma_g"],
            w, pwpdf, d["t"], d["p"], d["q"], d["esw"],
        )
        np.testing.assert_allclose(
            np.asarray(cdncact, np.float64), d["z"]["out/cdncact0"],
            rtol=RTOL[prec], atol=ATOL_ZERO, err_msg="cdncact0")
        for i, m in enumerate(MODES):
            np.testing.assert_allclose(
                np.asarray(sm[i], np.float64), d["z"][f"out/sc/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"sc[{m}]")
            np.testing.assert_allclose(
                np.asarray(nact[i], np.float64), d["z"][f"out/nact0/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"nact0[{m}]")
            np.testing.assert_allclose(
                np.asarray(nfrac[i], np.float64), d["z"][f"out/fracn0/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"fracn0[{m}]")
            np.testing.assert_allclose(
                np.asarray(rc[i, 0], np.float64), d["z"][f"out/rc0/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"rc0[{m}]")
        np.testing.assert_allclose(
            np.asarray(smax[0], np.float64), d["z"]["out/smax0"],
            rtol=RTOL[prec], atol=ATOL_ZERO, err_msg="smax0")


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_ham_arg_pdf_matches_reference(prec):
    """Nactivpdf = 1 -> the 20-bin West et al. 2013 PDF."""
    with jax.enable_x64(prec == "float64"):
        d = _inputs(prec)
        w, pwpdf = ham_updraft(
            d["tke"], d["omega"], d["rho"], 0.0, 0.7, n_pdf_bins=PDF_DEFAULT_BINS)
        a, b = koehler_ab(m7_spec(), d["mass"], d["t"])
        cdncact, nfrac, nact, sm, smax, rc = ham_arg(
            d["rdry"], d["number_vol"], a, b, d["can_activate"], d["sigma_g"],
            w, pwpdf, d["t"], d["p"], d["q"], d["esw"],
        )
        np.testing.assert_allclose(
            np.asarray(cdncact, np.float64), d["z"]["out/cdncact1"],
            rtol=RTOL[prec], atol=ATOL_ZERO, err_msg="cdncact1")
        # smax/rc carry the bin axis FIRST (n_w, ncol); the harness npz
        # records it last (ncol, n_w) -- transpose before comparing.
        np.testing.assert_allclose(
            np.asarray(smax, np.float64).T, d["z"]["out/smax1"],
            rtol=RTOL[prec], atol=ATOL_ZERO, err_msg="smax1")
        for i, m in enumerate(MODES):
            np.testing.assert_allclose(
                np.asarray(rc[i], np.float64).T, d["z"][f"out/rc1/{m}"],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=f"rc1[{m}]")


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_lin_leaitch_matches_reference(prec):
    """Closed-form: HAM's own ``m7_cumulative_normal`` and its crcut
    constants' single-precision-literal rounding, both ported exactly
    (see ``ham_activation.py``), reach float64 round-off here.
    """
    with jax.enable_x64(prec == "float64"):
        d = _inputs(prec)
        w, _ = ham_updraft(d["tke"], d["omega"], d["rho"], 0.0, 1.33, n_pdf_bins=None)
        na, na_cv, cdncact, cdncact_cv = lin_leaitch(
            d["number_vol"], d["can_activate"], d["rwet"], d["sigma_g"], w[0],
        )
        for name, val, key in (
            ("na", na, "out/na"), ("na_cv", na_cv, "out/na_cv"),
            ("cdncact", cdncact, "out/cdncact_ll"), ("cdncact_cv", cdncact_cv, "out/cdncact_cv"),
        ):
            np.testing.assert_allclose(
                np.asarray(val, np.float64), d["z"][key],
                rtol=RTOL[prec], atol=ATOL_ZERO, err_msg=name)


def test_ham_logtail_branches():
    """The three mo_ham_tools.f90:295-315 branches (non-empty/above,
    non-empty/below (=1), empty (=0)).
    """
    sigmaln = jnp.log(1.59)
    cmr = jnp.asarray([1.0e-7, 1.0e-7, 0.0])
    r = jnp.asarray([1.0e-7, 1.0e-30, 1.0e-7])
    frac = ham_logtail(cmr, r, sigmaln)
    assert 0.0 < float(frac[0]) < 1.0
    np.testing.assert_allclose(float(frac[1]), 1.0)
    np.testing.assert_allclose(float(frac[2]), 0.0)


class TestHamActivationTerm:
    """:class:`HamActivation` on the local M7 spec reproduces the harness CDNC."""

    def _jam_state(self, d):
        n = d["n"]
        return JamAerosolState(
            r_dry=d["rdry"], r_wet=d["rwet"],
            rho=jnp.full((7, n), 1700.0, d["dtype"]),
            kappa=jnp.full((7, n), 0.5, d["dtype"]),
            mass=jnp.zeros((7, n), d["dtype"]),
            number=d["number_vol"] / d["rho"][jnp.newaxis, :],
        )

    def _state(self, d, mass):
        n = d["n"]
        tracers = {f"m_{sp}_{m}": mass[(sp, m)] for (sp, m) in mass}
        return PhysicsState.zeros((n,)).copy(
            temperature=d["t"], specific_humidity=d["q"], tracers=tracers)

    def _diagnostics(self, d, jam_state):
        return {
            "_jam_state": jam_state,
            "pressure_full": d["p"],
            "air_density": d["rho"],
            "_dycore_fields": {"omega": d["omega"]},
            "vertical_diffusion": type("V", (), {"tke": d["tke"]})(),
        }

    def test_arg_scheme_matches_reference_cdnc(self):
        with jax.enable_x64(True):
            d = _inputs("float64")
            term = HamActivation(m7_spec(), scheme="arg", nactivpdf=0)
            state = self._state(d, d["mass"])
            diag = self._diagnostics(d, self._jam_state(d))
            tend, out = term(state, diag, None, None)
            np.testing.assert_allclose(
                np.asarray(out["activated_cdnc"], np.float64), d["z"]["out/cdncact0"],
                rtol=RTOL["float64"], atol=ATOL_ZERO)
            assert bool(jnp.all(tend.temperature == 0.0))
            assert jnp.all(out["activated_fraction"] <= 1.0 + 1e-6)

    def test_lin_leaitch_scheme_matches_reference_cdnc(self):
        with jax.enable_x64(True):
            d = _inputs("float64")
            term = HamActivation(m7_spec(), scheme="lin_leaitch")
            state = self._state(d, d["mass"])
            diag = self._diagnostics(d, self._jam_state(d))
            _, out = term(state, diag, None, None)
            np.testing.assert_allclose(
                np.asarray(out["activated_cdnc"], np.float64), d["z"]["out/cdncact_ll"],
                rtol=RTOL["float64"], atol=ATOL_ZERO)

    def test_grad_through_number_and_updraft_finite(self):
        with jax.enable_x64(True):
            d = _inputs("float64")

            def loss(number_scale, tke):
                term = HamActivation(m7_spec(), scheme="arg")
                jam = self._jam_state(d).copy(number=self._jam_state(d).number * number_scale)
                state = self._state(d, d["mass"])
                diag = self._diagnostics(d, jam)
                diag["vertical_diffusion"] = type(
                    "V", (), {"tke": tke})()
                _, out = term(state, diag, None, None)
                return jnp.sum(out["activated_cdnc"])

            g_n, g_tke = jax.grad(loss, argnums=(0, 1))(1.0, d["tke"])
            assert np.all(np.isfinite(np.asarray(g_n)))
            assert np.all(np.isfinite(np.asarray(g_tke)))

    def test_broadcast_modes_nlev_vs_modes_nlev_ncols(self):
        """(n_modes, nlev) vs (n_modes, nlev, ncols) agree per column."""
        with jax.enable_x64(True):
            d = _inputs("float64")
            spec = m7_spec()
            ncols = 3
            rdry = jnp.stack([d["rdry"]] * ncols, axis=-1)
            number_vol = jnp.stack([d["number_vol"]] * ncols, axis=-1)
            t = jnp.stack([d["t"]] * ncols, axis=-1)
            p = jnp.stack([d["p"]] * ncols, axis=-1)
            q = jnp.stack([d["q"]] * ncols, axis=-1)
            esw = jnp.stack([d["esw"]] * ncols, axis=-1)
            tke = jnp.stack([d["tke"]] * ncols, axis=-1)
            omega = jnp.stack([d["omega"]] * ncols, axis=-1)
            rho = jnp.stack([d["rho"]] * ncols, axis=-1)
            mass = {k: jnp.stack([v] * ncols, axis=-1) for k, v in d["mass"].items()}

            a1, b1 = koehler_ab(spec, d["mass"], d["t"])
            a3, b3 = koehler_ab(spec, mass, t)
            for c in range(ncols):
                np.testing.assert_allclose(np.asarray(a3[:, :, c]), np.asarray(a1))
                np.testing.assert_allclose(np.asarray(b3[:, :, c]), np.asarray(b1))

            w1, pwpdf1 = ham_updraft(d["tke"], d["omega"], d["rho"], 0.0, 0.7, n_pdf_bins=None)
            w3, pwpdf3 = ham_updraft(tke, omega, rho, 0.0, 0.7, n_pdf_bins=None)
            for c in range(ncols):
                np.testing.assert_allclose(np.asarray(w3[:, :, c]), np.asarray(w1))

            cdncact1, *_ = ham_arg(
                d["rdry"], d["number_vol"], a1, b1, d["can_activate"], d["sigma_g"],
                w1, pwpdf1, d["t"], d["p"], d["q"], d["esw"])
            cdncact3, *_ = ham_arg(
                rdry, number_vol, a3, b3, d["can_activate"], d["sigma_g"],
                w3, pwpdf3, t, p, q, esw)
            for c in range(ncols):
                np.testing.assert_allclose(
                    np.asarray(cdncact3[:, c]), np.asarray(cdncact1), rtol=1e-12)


def test_reference_cells_exercise_every_branch():
    """The fixture is not vacuous: w_min binds and the cthomi/empty-mode
    gates are each hit by some cell.
    """
    z = load()
    names = [str(s) for s in z["meta/names"]]
    col = {n: i for i, n in enumerate(names)}
    assert z["out/w0"][col["descending_weak_tke_wmin_binds"]] == 0.0
    assert z["out/cdncact0"][col["cold_high_alt"]] == 0.0
    assert z["out/a/accs"][col["ss_rich_coarse"]] == 0.0   # AS mode empty there
    # More aerosol suppresses the ACTIVATED FRACTION, not necessarily the
    # absolute CDNC (the polluted cell's much larger population still wins
    # in absolute terms): compare fracn, not cdncact.
    assert (z["out/fracn0/accs"][col["pure_so4_large_n"]]
            < z["out/fracn0/aits"][col["low_number_pristine"]])


# -----------------------------------------------------------------------
# Precision (measured, from this test file's run against ham_activ_M7.npz,
# float64; all at RTOL["float64"] = 1e-12 with no atol floor beyond
# ATOL_ZERO's protection of exact-zero reference entries):
#
#   koehler_ab (A, B):                      max rel err 3.3e-16 (round-off)
#   ham_updraft (nactivpdf=0, w):            max rel err 0.0    (identical formula)
#   ham_arg (nactivpdf=0: cdncact, sc,
#            nact, fracn, rc, smax):         max rel err 5.3e-15 (round-off)
#   ham_arg (nactivpdf=1, PDF):              max rel err 6.3e-16 (round-off)
#   lin_leaitch (na, na_cv,
#                cdncact, cdncact_cv):       EXACT (0.0 max abs/rel diff)
#
# Every field reaches float64 round-off (or exact equality). Getting there
# needed two fixes to this module beyond the obvious "port the formula":
#
# 1. ham_logtail must call HAM's OWN normal CDF (_m7_cumulative_normal,
#    a port of m7_cumulative_normal, mo_ham_m7.f90:75-326 -- the Cody/
#    DCDFLIB rational-Chebyshev approximation PLUS its EPSILON(1) tail
#    cutoff), not jax.scipy.special.erf. erf is a different, also highly
#    accurate, approximation of the same mathematical function; before this
#    fix ARG's fields still matched (dominated by near-0/near-1 fractions,
#    where erf and the Cody fit agree to many more digits) but ONE cell's
#    deep-tail mode (high_number_polluted_ks_as_cs) and ALL of Lin & Leaitch
#    (summing several mid-range, evaluator-sensitive fractions) sat at
#    ~1e-8-1e-9, not 1e-12 -- erroneously attributed, in an earlier version
#    of this module, to "two independent accurate implementations of the
#    same function" rather than tracked down further.
# 2. LL_CRCUT_STRAT/LL_CRCUT_CONV must reproduce
#    mo_ham_activ.f90:622,626's ``crcut=0.03*1E-6_dp`` /
#    ``crcut_cv=0.02*1E-6_dp`` bit-for-bit: the undecorated ``0.03``/
#    ``0.02`` literals are DEFAULT (single) precision in Fortran, so the
#    compiled constants are measurably not 3.0e-8/2.0e-8 -- fixing #1 above
#    and STILL comparing against the "intended" 0.03e-6/0.02e-6 left
#    Lin & Leaitch's own fields at ~1e-8, which is what exposed this one.
# -----------------------------------------------------------------------
