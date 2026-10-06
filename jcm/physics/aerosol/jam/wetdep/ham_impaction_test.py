"""jcm's HAM in-cloud impaction math against the compiled ECHAM reference (#1017).

``jcm/data/test/echam_cloud_reference/icscavimp.npz`` holds the outputs of
the UNMODIFIED r7492 ``ic_scav -> get_icscavfrac -> {ic_scav_nuc,
ic_scav_imp}`` chain -- the REAL ``ic_scav_imp`` (not follow-up A's
zeroing stub) -- on a 27-case matrix covering the water/ice collector-
radius bins (including the disputed ``cdroprad(6)=0.0`` node) and
``get_icscavfrac``'s own combination, ``pfrac = clip(pfrac_nuc +
pfrac_imp, 0, 1)``. See ``icscavimp_README.md`` for the full case list.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import jax
import numpy as np
import pytest

from jcm.physics.aerosol.jam.wetdep.ham_below_cloud import (
    cmedr2mmedr,
    default_croft_tables,
)
from jcm.physics.aerosol.jam.wetdep.ham_impaction import (
    CDROPRAD_UM_AS_COMPILED,
    default_impaction_tables,
    drop_radius_bin,
    ice_impaction_fraction,
    measure_cdroprad_bug_6_effect,
    plate_radius_bin,
    water_impaction_fraction,
)
from jcm.physics.aerosol.jam.wetdep.ham_nucleation import (
    ice_phase_xie,
    nucleation_scavenged_fraction,
    water_phase_xie,
)

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "icscavimp.npz")

_KMOD_TO_SHORT = {2: "ks", 3: "as", 4: "cs"}


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def test_full_chain_matches_compiled_icscavimp_reference():
    """Every row of ``icscavimp.npz`` reproduced from jcm's water_phase_xie/
    ice_phase_xie/nucleation_scavenged_fraction (follow-up A) PLUS
    water_impaction_fraction/ice_impaction_fraction (follow-up B), combined
    exactly as ``get_icscavfrac`` does: ``pfrac = clip(pfrac_nuc +
    pfrac_imp, 0, 1)``.
    """
    z = load()
    n = len(z["case_name"])
    with jax.enable_x64(True):
        caerorad = default_croft_tables().caerorad
        imp_tables = default_impaction_tables()
        for i in range(n):
            kwat_phase = int(z["kwat_phase"][i])
            kmod = int(z["kmod"][i])
            ktrac_phase = int(z["ktrac_phase"][i])
            sigma = float(z["sigma"][i])
            sigmaln = math.log(sigma)
            radius = z["radius"][i]
            wetrad = z["wetrad"][i]
            prho = z["prho"][i]
            ref_pfrac_nuc = z["pfrac_nuc"][i]
            ref_pfrac_imp = z["pfrac_imp"][i]
            ref_pfrac = z["pfrac"][i]

            # --- Nucleation (0 for a non-relevant mode: ic_scav_nuc's own
            # early RETURN; jcm never calls the math for these either,
            # mirroring wetdep_term.py's per-mode gating).
            if kmod in _KMOD_TO_SHORT:
                mode_short = _KMOD_TO_SHORT[kmod]
                if kwat_phase == 1:
                    xie = water_phase_xie(z["cdnc"][i], prho, z["na"][i], z["frac"][i])
                else:
                    xie = ice_phase_xie(z["icnc"][i], z["nks"][i], z["nas"][i],
                                         z["ncs"][i], mode_short)
                mass_factor = cmedr2mmedr(sigma)
                fn, fm = nucleation_scavenged_fraction(xie, radius, sigmaln, mass_factor)
                got_pfrac_nuc = float(fn if ktrac_phase == 1 else fm)
            else:
                got_pfrac_nuc = 0.0

            # --- Impaction (no activating-mode gate: every mode gets this).
            # mr/indexy1/indexy2's own zrad_fac (mo_ham_wetdep.f90:209-220):
            # 1.0 for the number phase, cmedr2mmedr(mode) for mass -- the
            # SAME scaling ham_below_cloud.py's mr_num/mr_mass apply; the
            # Fortran driver records the RAW wetrad, not the scaled mr, so
            # this test must scale it itself (a test-script detail, not a
            # port one -- wetdep_term.py already builds mr_num/mr_mass this
            # way for the below-cloud pathway and reuses them here).
            phase_str = "number" if ktrac_phase == 1 else "mass"
            mr_m = wetrad if ktrac_phase == 1 else wetrad * cmedr2mmedr(sigma)
            if kwat_phase == 1:
                got_pfrac_imp = float(water_impaction_fraction(
                    z["reffl"][i], mr_m, phase=phase_str, caerorad=caerorad,
                    tables=imp_tables, cdroprad_um=CDROPRAD_UM_AS_COMPILED))
            else:
                # ic_scav_imp's OWN zicnc = pxtp1c(idt_icnc)*prhop1
                # (mo_ham_wetdep.f90:852) -- icnc here (like ice_phase_xie's
                # input above) is the MIXING ratio; multiply by density to
                # get the number concentration ic_scav_imp's exponential
                # transform actually needs.
                icnc_numconc = z["icnc"][i] * prho
                got_pfrac_imp = float(ice_impaction_fraction(
                    z["reffi"][i], icnc_numconc, mr_m, 1800.0, caerorad,
                    tables=imp_tables))

            got_pfrac = float(np.clip(got_pfrac_nuc + got_pfrac_imp, 0.0, 1.0))

            np.testing.assert_allclose(got_pfrac_nuc, ref_pfrac_nuc, rtol=1e-12, atol=1e-30,
                                        err_msg=f"pfrac_nuc mismatch row {i} ({z['case_name'][i]})")
            np.testing.assert_allclose(got_pfrac_imp, ref_pfrac_imp, rtol=1e-12, atol=1e-30,
                                        err_msg=f"pfrac_imp mismatch row {i} ({z['case_name'][i]})")
            np.testing.assert_allclose(got_pfrac, ref_pfrac, rtol=1e-12, atol=1e-30,
                                        err_msg=f"pfrac mismatch row {i} ({z['case_name'][i]})")


def test_drop_radius_bin_matches_cdroprad_node_6():
    # reffl=30 lands exactly on the disputed node (idx1=floor(30/5)=6 ->
    # cdroprad=0.0 as compiled; idx2=7 -> cdroprad=35).
    idx1, idx2 = drop_radius_bin(np.asarray(30.0))
    assert int(idx1) == 6
    assert int(idx2) == 7
    # reffl=27 straddles the node from below: idx1=5 (cdroprad=25),
    # idx2=6 (cdroprad=0.0 as compiled).
    idx1, idx2 = drop_radius_bin(np.asarray(27.0))
    assert int(idx1) == 5
    assert int(idx2) == 6


def test_plate_radius_bin_three_regimes():
    # Dead (reffi<1 or icnc<eps): index 0.
    idx1, idx2 = plate_radius_bin(np.asarray(0.5), np.asarray(1.0e6))
    assert int(idx1) == 0 and int(idx2) == 0
    idx1, idx2 = plate_radius_bin(np.asarray(22.0), np.asarray(0.0))
    assert int(idx1) == 0 and int(idx2) == 0
    # Fine (1<=reffi<50): floor(reffi/5).
    idx1, idx2 = plate_radius_bin(np.asarray(22.0), np.asarray(1.0e6))
    assert int(idx1) == 4 and int(idx2) == 5
    # Coarse (reffi>=50): 8+floor(reffi/50).
    idx1, idx2 = plate_radius_bin(np.asarray(120.0), np.asarray(1.0e6))
    assert int(idx1) == 10 and int(idx2) == 11


def test_water_impaction_fraction_is_zero_without_droplets():
    caerorad = default_croft_tables().caerorad
    out = water_impaction_fraction(
        np.asarray(0.0), np.asarray(6.0e-8), phase="number", caerorad=caerorad)
    assert float(out) == 0.0


def test_ice_impaction_fraction_is_zero_without_icnc():
    caerorad = default_croft_tables().caerorad
    out = ice_impaction_fraction(
        np.asarray(22.0), np.asarray(0.0), np.asarray(6.0e-8), 1800.0, caerorad)
    assert float(out) == 0.0


def test_measure_cdroprad_bug_6_effect_is_nonzero_near_the_node():
    caerorad = default_croft_tables().caerorad
    reffl = np.array([10.0, 30.0, 50.0])
    mr_m = np.full_like(reffl, 0.06e-6)
    out = measure_cdroprad_bug_6_effect(reffl, mr_m, phase="number", caerorad=caerorad)
    assert not out["hits_node_6"][0]
    assert out["hits_node_6"][1]
    assert out["relative_difference"][1] > 0.0


def test_rejects_unknown_phase():
    caerorad = default_croft_tables().caerorad
    with pytest.raises(ValueError):
        water_impaction_fraction(
            np.asarray(10.0), np.asarray(6.0e-8), phase="bogus", caerorad=caerorad)
