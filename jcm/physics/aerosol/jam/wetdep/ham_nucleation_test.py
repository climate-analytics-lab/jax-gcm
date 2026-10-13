"""jcm's HAM nucleation-scavenging math against the compiled ECHAM reference (#1017).

``jcm/data/test/echam_cloud_reference/haminvertlogtail.npz`` holds the
outputs of the UNMODIFIED r7492 ``mo_ham_tools.f90::ham_m7_invertlogtail``
on 33 designed (count-median-radius, xie, sigma) test points spanning the
small/mid/huge-tail branches and both M7 geometric standard deviations.
This is the one genuinely new piece of math follow-up A needs (a closed-
form inverse-erf approximation); ``ham_logtail``/the normal CDF it also
calls on the forward side are already ported and held to a compiled
reference in ``ham_activation_reference_test.py``.

``icscavnuc.npz`` is the second, broader reference: it holds the outputs of
the UNMODIFIED ``ic_scav -> get_icscavfrac -> ic_scav_nuc`` chain itself
(not just ``ham_m7_invertlogtail`` in isolation) on 19 designed M7 columns
-- the review round that required this (house rule: "faithful means
compiled numbers," a Known-Gaps note is not a resting place for an
unvalidated pathway) is why both the per-primitive AND the full-chain
compiled comparisons now exist side by side.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import numpy as np
import pytest

from jcm.physics.aerosol.jam.wetdep.ham_below_cloud import cmedr2mmedr
from jcm.physics.aerosol.jam.wetdep.ham_nucleation import (
    ham_m7_invertlogtail,
    ice_phase_xie,
    nucleation_scavenged_fraction,
    water_phase_xie,
)

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "haminvertlogtail.npz")
REF_CHAIN = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
             / "icscavnuc.npz")
# Measured max relative error over the 33 cases: float64 1.6e-16 (round-
# off); float32 1.8e-3, all at the single most extreme case (xie=-0.999999,
# the small/huge-tail boundary, where log(1-x2) loses precision near its
# singularity) -- the tolerance below is that measured max, not a guess.
RTOL = {"float64": 1e-12, "float32": 2e-3}

# M7 mode index (icscavnuc.npz's 1-based Fortran kmod) -> jcm mode_short,
# for the three modes ic_scav_nuc actually computes (mo_ham_wetdep.f90:727:
# "IF (kmod < 2 .OR. kmod > 4) THEN ... RETURN"); every other kmod's row is
# a non-relevant-mode case, checked separately below.
_KMOD_TO_SHORT = {2: "ks", 3: "as", 4: "cs"}


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


@functools.lru_cache(maxsize=None)
def load_chain():
    with np.load(REF_CHAIN) as z:
        return {k: z[k] for k in z.files}


def test_full_chain_matches_compiled_icscavnuc_reference():
    """Every row of ``icscavnuc.npz`` -- the compiled, verbatim
    ``ic_scav -> get_icscavfrac -> ic_scav_nuc`` chain, not just
    ``ham_m7_invertlogtail`` standalone -- reproduced from jcm's
    ``water_phase_xie``/``ice_phase_xie``/``nucleation_scavenged_fraction``
    at float64. Covers: liquid-only/ice-only/mixed-phase columns, cdnc/icnc
    and na both sides of their gates, the ice phase's CS-then-AS-then-KS
    size-ordered depletion with KS/AS/CS emptied in turn, ARG vs
    Lin & Leaitch radius selection, the xie>=1 "huge tail" clip, and
    non-relevant modes (checked separately, see below).
    """
    z = load_chain()
    n = len(z["case_name"])
    with jax.enable_x64(True):
        for i in range(n):
            kwat_phase = int(z["kwat_phase"][i])
            kmod = int(z["kmod"][i])
            ktrac_phase = int(z["ktrac_phase"][i])
            sigmaln = np.log(z["sigma"][i])
            radius = z["radius"][i]
            ref_sfnuc = z["sfnuc"][i]
            ref_rcritrad = z["rcritrad"][i]

            if kmod not in _KMOD_TO_SHORT:
                # ic_scav_nuc's own early RETURN zeroes these outright; jcm
                # never calls the nucleation math for them at all (zeroed
                # directly in wetdep_term.py's per-mode loop instead) -- a
                # different code path reaching the same answer, so there is
                # no Python function call to make here. Confirm the
                # reference itself recorded exactly that.
                assert ref_sfnuc == 0.0, (i, z["case_name"][i], "expected 0 for non-relevant mode")
                continue

            mode_short = _KMOD_TO_SHORT[kmod]
            if kwat_phase == 1:
                xie = water_phase_xie(z["cdnc"][i], z["prho"][i], z["na"][i], z["frac"][i])
            else:
                xie = ice_phase_xie(z["icnc"][i], z["nks"][i], z["nas"][i], z["ncs"][i], mode_short)

            mass_factor = cmedr2mmedr(z["sigma"][i])
            frac_number, frac_mass = nucleation_scavenged_fraction(
                xie, radius, sigmaln, mass_factor)
            got_sfnuc = float(frac_number if ktrac_phase == 1 else frac_mass)
            got_rcritrad = float(ham_m7_invertlogtail(radius, xie, sigmaln))

            np.testing.assert_allclose(got_rcritrad, ref_rcritrad, rtol=1e-12, atol=1e-30,
                                        err_msg=f"rcritrad mismatch, row {i} ({z['case_name'][i]})")
            np.testing.assert_allclose(got_sfnuc, ref_sfnuc, rtol=1e-12, atol=1e-30,
                                        err_msg=f"sfnuc mismatch, row {i} ({z['case_name'][i]})")


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_invertlogtail_matches_compiled_reference(prec):
    z = load()
    with jax.enable_x64(prec == "float64"):
        dtype = np.float64 if prec == "float64" else np.float32
        for cmr, xie, sigma, ref in zip(z["cmr"], z["xie"], z["sigma"], z["critrad"]):
            got = float(ham_m7_invertlogtail(
                np.asarray(cmr, dtype), np.asarray(xie, dtype),
                np.asarray(np.log(sigma), dtype)))
            np.testing.assert_allclose(got, ref, rtol=RTOL[prec], atol=1e-30)


def test_invertlogtail_then_logtail_round_trips():
    """A sanity check independent of the compiled reference: inverting at a
    target tail fraction and reading the SAME tail forward must reproduce
    that fraction (ic_scav_nuc's own two-call pattern,
    mo_ham_wetdep.f90:779-794).
    """
    from jcm.physics.aerosol.jam.activation.ham_activation import ham_logtail
    with jax.enable_x64(True):
        cmr = np.asarray(0.05e-6)
        sigmaln = np.asarray(np.log(1.59))
        for target in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            xie = 1.0 - 2.0 * target
            rcrit = ham_m7_invertlogtail(cmr, np.asarray(xie), sigmaln)
            back = float(ham_logtail(cmr, rcrit, sigmaln))
            # The Winitzki inverse-erf approximation's own round-trip
            # residual (not round-off): matches the harness-measured
            # magnitude, tightest at the median and loosest at the tails.
            assert abs(back - target) < 2e-3, (target, back)


def test_water_phase_xie_gate_and_bounds():
    cdnc = np.asarray([0.0, 1e-9, 1e-9])
    rho = np.asarray([1.0, 1.0, 1.0])
    na = np.asarray([1e8, 0.0, 1e8])
    frac = np.asarray([0.5, 0.5, 0.5])
    xie = np.asarray(water_phase_xie(cdnc, rho, na, frac))
    # No condensate, or no available aerosol number: the "no detectable
    # signal" default (xie=1, mo_ham_wetdep.f90:751-755,777-778).
    assert xie[0] == 1.0
    assert xie[1] == 1.0
    assert -1.0 <= xie[2] <= 1.0


def test_ice_phase_xie_size_ordered_depletion():
    # CS (coarse) depletes first: icnc entirely attributable to CS when
    # icnc <= n_cs.
    icnc = np.asarray(5e5)
    n_ks = np.asarray(1e6)
    n_as = np.asarray(1e6)
    n_cs = np.asarray(1e6)
    xie_cs = float(ice_phase_xie(icnc, n_ks, n_as, n_cs, "cs"))
    xie_as = float(ice_phase_xie(icnc, n_ks, n_as, n_cs, "as"))
    xie_ks = float(ice_phase_xie(icnc, n_ks, n_as, n_cs, "ks"))
    # ratio_cs = icnc/n_cs = 0.5 -> xie = 1-2*0.5 = 0
    assert abs(xie_cs - 0.0) < 1e-12
    # AS/KS see icnc already "used up" by CS (icnc <= n_cs): remainder 0.
    assert xie_as == 1.0
    assert xie_ks == 1.0

    def _raises():
        ice_phase_xie(icnc, n_ks, n_as, n_cs, "ns")
    with pytest.raises(ValueError):
        _raises()


def test_nucleation_scavenged_fraction_is_clipped_and_mass_differs():
    with jax.enable_x64(True):
        xie = np.asarray(-0.5)
        radius = np.asarray(0.05e-6)
        sigmaln = np.asarray(np.log(1.59))
        number_frac, mass_frac = nucleation_scavenged_fraction(
            xie, radius, sigmaln, mass_factor=4.0)
        assert 0.0 <= float(number_frac) <= 1.0
        assert 0.0 <= float(mass_frac) <= 1.0
        # A larger effective radius (mass_factor > 1) reads further up the
        # SAME tail at the SAME critical radius, so the mass fraction must
        # differ from the number fraction (mo_ham_wetdep.f90:793's distinct
        # ll_trac_phase calls would not be needed otherwise).
        assert float(mass_frac) != float(number_frac)
