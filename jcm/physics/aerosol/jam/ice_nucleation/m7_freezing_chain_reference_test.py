"""jcm's HAM M7 mixed-phase freezing CHAIN against compiled ECHAM6.3-HAM2.3 (jax-gcm#1017 task 2 part 1).

``ham_freezing_reference_test.py`` and ``lohmann_2m_freezing_reference_test.py``
each verify ONE stage of ECHAM-HAM's mixed-phase freezing against the compiled
Fortran, fed that stage's OWN independently-designed inputs: the partition
(``ham_IN_setup`` / :func:`ham_freezing_aerosol`) against designed M7 aerosol
mass/number cells, and the freezing rate (``het_mxphase_freezing``) against
hand-picked fraction/radius values. Neither, on its own, confirms that a REAL
M7 aerosol state's partition output is what then drives nonzero contact
freezing -- the two reference datasets were never chained on one consistent
state.

``jcm/data/test/echam_cloud_reference/m7_freezing_chain.npz`` does: it holds
the compiled, UNMODIFIED ``mo_cloud_micro_2m.f90::het_mxphase_freezing``
(F 2675-2840) run with the HAM freezing inputs set to the EXACT fractions/radii
``hamfrz_M7.npz``'s ``mixed_all`` cell produces from the compiled, UNMODIFIED
``mo_ham_freezing.f90::ham_IN_setup`` -- one M7 aerosol state, through both
compiled routines in sequence. This module reproduces that chain in jcm
(:func:`ham_freezing_aerosol` on the ``mixed_all`` masses/numbers, then
:func:`het_mxphase_freezing` on the result) and compares both the partition
and the end-to-end freezing rate to the compiled numbers.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing_reference_test import (
    M7_CLASSES,
    load as load_hamfrz,
    m7_spec,
)
from jcm.physics.clouds.lohmann_2m.deposition_freezing import het_mxphase_freezing
from jcm.physics.clouds.lohmann_2m_fortran_reference_test import echam_constants, echam_params, precision

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "m7_freezing_chain.npz")
M7_MODES = ("nucs", "aits", "accs", "coas", "aiti", "acci", "coai")


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def _mixed_all_inputs():
    """Return the ``mixed_all`` cell's masses/numbers/wet-radii/activated-numbers,
    straight off ``hamfrz_M7.npz`` (the same fixture
    ``ham_freezing_reference_test.py`` loads), so this module never
    re-transcribes the cell by hand.

    Returns plain Python ``float``s (not jnp arrays): this fixture is built
    once at module scope, before any test has chosen a precision, and
    ``jnp.asarray`` on a Python float WITHOUT an explicit dtype follows
    whatever ``jax_enable_x64`` happens to be at call time -- silently
    rounding to float32 outside an ``enable_x64(True)`` block, a rounding
    casting back UP to float64 later can never undo. Plain floats defer
    that choice to each call site's own explicit ``jnp.asarray(v, dtype)``.
    """
    z = load_hamfrz()
    names = [str(s) for s in z["meta/names"]]
    i = names.index("mixed_all")

    def g(k):
        return float(z[k][i]) if k in z else 0.0

    masses = {(sp, m): g(f"in/mass/{m}/{sp}")
              for m in M7_MODES for sp in m7_spec().mode(m).species}
    number = {m: g(f"in/number/{m}") for m in M7_MODES}
    nact = {m: g(f"in/nact/{m}") for m in M7_MODES}
    rwet = {m: g(f"in/rwet/{m}") for m in M7_MODES}
    rho = g("in/rho")
    cdncact = g("in/cdncact")
    return masses, number, nact, rwet, rho, cdncact


# ===========================================================================
# Stage 1: the partition, on the mixed_all cell, as a sanity link to the
# fraction/radius constants ``columns_frz.py``'s M7_MIXED hard-codes into
# this module's Fortran reference (both come from the SAME hamfrz_M7.npz
# cell; faithfulness at 1e-12 is already ``ham_freezing_reference_test.py``'s
# parametrized coverage, including this cell).
# ===========================================================================
def test_mixed_all_partition_matches_the_chain_harness_constants():
    from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import ham_freezing_aerosol

    with jax.enable_x64(True):
        masses, number, nact, rwet, rho, cdncact = _mixed_all_inputs()
        fa = ham_freezing_aerosol(m7_spec(), M7_CLASSES, masses, number, nact, rwet, rho, cdncact)
    z = load()
    expected = {
        "dust_soluble": "fracdusol", "bc_soluble": "fracbcsol",
        "dust_insoluble_accumulation": "fracduai", "dust_insoluble_coarse": "fracduci",
        "bc_insoluble": "fracbcinsol", "wet_radius_insoluble_aitken": "rwetki",
        "wet_radius_insoluble_accumulation": "rwetai", "wet_radius_insoluble_coarse": "rwetci",
    }
    j = [str(x) for x in z["meta/names"]].index("frz_m7_mixed")
    for field, key in expected.items():
        # in/<key> is (nlev, ncol): the frz_m7_mixed column's designed levels
        # all carry the SAME M7_MIXED value, 0 elsewhere -- the max over the
        # column recovers it without needing the row index.
        np.testing.assert_allclose(
            float(getattr(fa, field)), float(np.max(z[f"in/{key}"][:, j])),
            rtol=1e-12, atol=0.0, err_msg=key)
    # The contact-freezing dust inputs are genuinely non-zero (not merely
    # finite) -- the thing #1017 task 2 part 1 asks to verify.
    assert float(fa.dust_insoluble_accumulation) > 0.0
    assert float(fa.dust_insoluble_coarse) > 0.0


# ===========================================================================
# Stage 2 + chain: jcm's own partition output (NOT the recorded fractions),
# fed through jcm's own het_mxphase_freezing, against the compiled Fortran
# CHAIN's ``frz_m7_mixed`` column. The het_mxphase_freezing INPUTS other
# than the eight HAM fractions/radii (pressure, TKE, cover, CDNC/ICNC
# in-state, ...) are read off the Fortran's own recorded ``hf_*_in``
# diagnostics -- the same mapping ``lohmann_2m_freezing_reference_test.py::
# run_jcm_block`` uses -- so this test isolates exactly the two things it
# means to check (the partition and the rate), not a hand-built column.
# ===========================================================================
_J = lambda x, dtype: jnp.asarray(np.asarray(x, np.float64), dtype)  # noqa: E731


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_full_chain_matches_compiled_fortran(prec):
    """``ham_freezing_aerosol`` on the mixed_all masses, then
    ``het_mxphase_freezing`` on its own output -- end to end in jcm --
    against the compiled Fortran chain's ``frz_m7_mixed`` column
    (``m7_freezing_chain.npz``), at every level ECHAM called the routine.
    """
    masses, number, nact, rwet, rho_cell, cdncact = _mixed_all_inputs()
    dtype = jnp.float64 if prec == "float64" else jnp.float32
    z = load()
    names = [str(x) for x in z["meta/names"]]
    j = names.index("frz_m7_mixed")
    g = {k: z[f"diag/dt1200/{k}"][:, j] for k in (
        "hf_mask", "hf_tp1tmp", "hf_cdncmin", "hf_icnc_in", "hf_cdnc_in",
        "frl_hom", "hf_xib_in", "hf_xlb_in", "rho", "uicw_aclc")}
    din = {k: z[f"in/{k}"][:, j] for k in ("pvervel", "ptkem1", "papm1")}

    with jax.enable_x64(prec == "float64"), echam_constants(), precision(prec):
        from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import ham_freezing_aerosol

        fa = ham_freezing_aerosol(
            m7_spec(), M7_CLASSES,
            {k: jnp.asarray(v, dtype) for k, v in masses.items()},
            {k: jnp.asarray(v, dtype) for k, v in number.items()},
            {k: jnp.asarray(v, dtype) for k, v in nact.items()},
            {k: jnp.asarray(v, dtype) for k, v in rwet.items()},
            jnp.asarray(rho_cell, dtype), jnp.asarray(cdncact, dtype))

        n = g["rho"].shape[0]
        broadcast = lambda x: jnp.full((n,), x, dtype)  # noqa: E731
        J = functools.partial(_J, dtype=dtype)
        mask = jnp.asarray(g["hf_mask"] > 0.5)
        out = het_mxphase_freezing(
            mask, J(din["papm1"]), J(din["ptkem1"]), J(din["pvervel"]), J(g["uicw_aclc"]),
            broadcast(fa.bc_soluble), broadcast(fa.bc_insoluble), broadcast(fa.dust_soluble),
            broadcast(fa.dust_insoluble_accumulation), broadcast(fa.dust_insoluble_coarse),
            J(g["rho"]), 1.0 / J(g["rho"]), broadcast(fa.wet_radius_insoluble_aitken),
            broadcast(fa.wet_radius_insoluble_accumulation), broadcast(fa.wet_radius_insoluble_coarse),
            J(g["hf_tp1tmp"]), J(g["hf_cdncmin"]), J(g["hf_icnc_in"]), J(g["hf_cdnc_in"]),
            J(g["frl_hom"]), J(g["hf_xib_in"]), J(g["hf_xlb_in"]),
            jnp.asarray(1200.0, dtype), 1.0e-12, echam_params())
        icnc, cdnc, frl, xib, xlb, frln = out

    gate = g["hf_mask"] > 0.5
    assert int(gate.sum()) >= 4   # the harness README: contact freezing is active on this column

    # Measured (not guessed), and attributed, not just margined: max relative
    # error over the gate-passing levels is float64 2.0e-10, float32 5.9e-6,
    # both concentrated at the SINGLE warmest designed level (268 K -- the
    # catalogue's own "negligible" edge of the immersion-rate range,
    # columns_frz.py's ``MIXED_LEVELS`` docstring, not a representative
    # state). Isolated by feeding ``het_mxphase_freezing`` the compiled
    # ``ham_IN_setup``'s OWN fractions/radii directly (bypassing
    # :func:`ham_freezing_aerosol` entirely): the error is UNCHANGED, so it
    # is not stage-1 round-off amplified by stage 2 -- partition accuracy at
    # this cell is exactly 0.0 relative
    # (test_mixed_all_partition_matches_the_chain_harness_constants), ruling
    # that out. Decomposed further via the ``hf_dfduai``/``hf_dfduci``/
    # ``hf_ztte``/``hf_frzcnt``/``hf_frzimm`` intermediates the harness
    # records: the contact branch's Brownian diffusivities and the cooling
    # rate ``ztte`` are bit-identical (0.0 relative) at EVERY level; the
    # residual lives entirely in the immersion branch's
    # ``exp(tmelt - T)``-driven rate, which itself agrees to float64
    # round-off (~1e-16) at the three coldest levels and only degrades at
    # 258 K (7e-10) and 268 K (4.2e-6) -- yet ``jnp.exp`` was checked against
    # ``math.exp`` (glibc) for this exact argument (``exp(5.15)``) and
    # matches to 1 ULP, so jcm's own evaluation is not implicated; the
    # remaining few-ULP difference is consistent with the compiled
    # reference's OWN ``EXP`` intrinsic (gfortran -O2, no ``-ffast-math``
    # but auto-vectorization can still substitute a less-precise
    # transcendental at -O2) landing a handful of ULP away from glibc's,
    # which is outside jcm's control and not a port defect. This is NOT a
    # derivative/conditioning amplification -- no step here subtracts two
    # comparable O(1) quantities or divides by a near-zero denominator, and
    # the ABSOLUTE error is in fact roughly CONSTANT across all 5 levels
    # (1.1e-20 to 1.4e-20, consistent with ordinary ULP accumulation through
    # a ~15-20-op float64 chain with two transcendentals). The measured
    # RELATIVE error only grows at 268 K because ``frl`` itself has shrunk to
    # 3.5e-11 there (seven decades below the coldest level's 1e-4) --
    # dividing a near-constant absolute floor by a shrinking denominator,
    # not amplification of a small input perturbation by a steep derivative.
    # rtol carries a >=15x safety margin over the measured value at each
    # precision (not reproducing it exactly) so the test does not flake on
    # an unrelated, harmless change to operation order elsewhere in the
    # chain.
    rtol = {"float64": 1e-8, "float32": 1e-4}[prec]
    atol = {"float64": (1e-14, 1e-14, 1e-14), "float32": (1e-6, 1e2, 1e-12)}[prec]

    def check(name, jcm_v, ref, i):
        jcm_v = np.asarray(jcm_v, np.float64)[gate]
        ref = ref[gate]
        scale = max(float(np.max(np.abs(ref))), 1e-30)
        np.testing.assert_allclose(jcm_v, ref, rtol=rtol, atol=atol[i] + rtol * scale, err_msg=name)

    check("hf_frl chain", frl, z["diag/dt1200/hf_frl"][:, j], 0)
    check("hf_frln chain", frln, z["diag/dt1200/hf_frln"][:, j], 1)
    check("hf_icnc chain", icnc, z["diag/dt1200/hf_icnc"][:, j], 2)
    # The headline claim: contact freezing is active, not just finite.
    assert float(np.max(np.asarray(frl)[gate])) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
