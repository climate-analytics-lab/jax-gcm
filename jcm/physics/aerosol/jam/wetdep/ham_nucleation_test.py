"""jcm's HAM nucleation-scavenging math against the compiled ECHAM reference (#1017).

``jcm/data/test/echam_cloud_reference/haminvertlogtail.npz`` holds the
outputs of the UNMODIFIED r7492 ``mo_ham_tools.f90::ham_m7_invertlogtail``
on 33 designed (count-median-radius, xie, sigma) test points spanning the
small/mid/huge-tail branches and both M7 geometric standard deviations.
This is the one genuinely new piece of math follow-up A needs (a closed-
form inverse-erf approximation); ``ham_logtail``/the normal CDF it also
calls on the forward side are already ported and held to a compiled
reference in ``ham_activation_reference_test.py``.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import numpy as np
import pytest

from jcm.physics.aerosol.jam.wetdep.ham_nucleation import (
    ham_m7_invertlogtail,
    ice_phase_xie,
    nucleation_scavenged_fraction,
    water_phase_xie,
)

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "haminvertlogtail.npz")
# Measured max relative error over the 33 cases: float64 1.6e-16 (round-
# off); float32 1.8e-3, all at the single most extreme case (xie=-0.999999,
# the small/huge-tail boundary, where log(1-x2) loses precision near its
# singularity) -- the tolerance below is that measured max, not a guess.
RTOL = {"float64": 1e-12, "float32": 2e-3}


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


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
