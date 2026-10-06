"""jcm's full nic_cirrus=2 column against the compiled ECHAM6.3-HAM2.3 chain
(jax-gcm#1017 task 2 part 2b, the lead's end-to-end verification request).

``cloud2m_cirrus_T63L47.npz`` holds the compiled, UNMODIFIED
``mo_cloud_micro_2m.f90::cloud_micro_interface`` run WITH the UNMODIFIED
``mo_cirrus.f90::xfrzmstr`` linked in (``nic_cirrus=2``, not the harness's
abort-stub), feeding ``pascs``/``papnx``/``paprx``/``papsigx`` through the
same ``cloud_subm_1`` interface ``ham_IN_setup`` uses, on 5 designed columns:
a pure-cirrus supersaturated cell, the same cell with pre-existing ice, two
updraft variants, and a no-aerosol floor control. See
``cloud2m_cirrus_README.md``/``_provenance.json``.

This verifies the two equivalence claims the wiring in
``cloud_microphysics_2m`` rests on against compiled numbers, per the lead's
explicit request, rather than a reading of the Fortran:

(a) jcm's depletion reference at the ``xfrzmstr`` call is ECHAM's own
    ``zicncq`` there (``icnc0 + znidetr``, BEFORE sedimentation or melting
    -- not a later-stage quantity);
(b) the step-start ice supersaturation ``q_m1/qsi - 1`` is what ECHAM
    passes, not section 5's adjusted humidity.

**Both were initially wrong, and this harness is what caught it.** The
first wiring attempt used ``icnc_melt`` (a POST-sedimentation, POST-melt
quantity that does not exist yet at the point ECHAM calls ``xfrzmstr``) as
the depletion reference, AND fed the nucleated number only as
``update_in_cloud_water``'s own fallback ``pnicex`` argument -- never into
the EARLY, PRIMARY addition ECHAM itself performs
(``zicncq += znidetr + zninucl``, mo_cloud_micro_2m.f90:982,1050-1058,
"",BEFORE sedimentation, melting, section 4 or section 5). Since
``update_in_cloud_water``'s fallback only fires when the primary path left
ICNC at or below ``icemin`` (which section 1's own addition almost always
prevents), every non-degenerate cirrus column was silently pinned at the
``icemin`` floor regardless of the aerosol number -- a 100% error (``rel``
column below was exactly 1.0 for every aerosol-driven cell before the fix).
Fixed by moving the ``xfrzmstr`` call to the point matching ECHAM's own
section 1 (before sedimentation) and using ``icnc0 + znidetr`` as the
depletion reference there; see ``scheme.py``'s own comments at that call
site for the full attribution.

**Residual after the fix (isolated, not further investigated here).**
Bisecting the compiled reference's own intermediate diagnostics
(``icnc_add`` through ``icnc_6``, all identical to float64 round-off)
against the final ``icnc_7`` shows the ENTIRE remaining 6e-4 to 3.1e-2
relative discrepancy enters during ``precip_formation_cold`` (section 7) --
a function this task's cirrus wiring does not touch, called identically
regardless of ``nic_cirrus``, and not independently verified against the
compiled reference anywhere in this test family before (the existing
``lohmann_2m_fortran_reference_test.py`` covers ``zrid``/``lo2``/
``sedimentation_ice``/``update_in_cloud_water``/``znidetr`` and end-to-end
number-tendency columns, but never names ``precip_formation_cold``
directly). ``xfrzmstr``, the depletion inputs (``sice``, the turbulent
updraft, ``icnc0``, ``znidetr``) and the capped nucleated number were each
checked directly against the compiled reference's own recorded
intermediates and match to float64 round-off (0.0-2e-16 relative) --
confirmed in this investigation, not asserted here (re-asserting a single
level's intermediate would duplicate ``cirrus_reference_test.py``'s own
coverage of ``xfrzmstr`` itself). Per the house rule (a shared-function
finding outside the ``nic_cirrus=2``-only path is a STOP, not something
this task fixes), this residual is reported to the lead rather than
patched here; the tolerance below is set to the MEASURED worst case with a
margin, not tightened further by altering shared code.

**Root cause, isolated and tracked as #1039.** Feeding the compiled
reference's own recorded section-7 inputs directly into
``precip_formation_cold`` (bypassing the rest of the column sweep) shows
``xib_7``/``spr``/``sacl`` match to float64 round-off but ``icnc_7`` does
not (1.07e-8 to 4.75e-5 relative at that isolated call -- smaller than the
6e-4 to 3.1e-2 seen end-to-end above because the error compounds through
sections 8's tendency formation). Bisected to one line: ``precip.py``'s
``zsprn1 = ice_number * (zsaci + zsaut) / (zxibold_sec + params.eps)``
uses ``params.eps`` = ``np.finfo(np.float32).eps`` (``lohmann_2m_params.py``)
unconditionally, versus the Fortran's ``EPSILON(1.0_dp)`` (``eps``,
``mo_cloud_utils.f90``, used at ``mo_cloud_micro_2m.f90:3405``) -- ~9
orders of magnitude smaller. Substituting the float64 epsilon for
``params.eps`` alone closes the gap to exactly 0.0 relative on all 5
columns. This is shared code (identical regardless of ``nic_cirrus``,
reachable by any 2M preset wherever in-cloud ice is comparable to or
below ~1e-7 kg/kg -- an ordinary thin-cirrus regime), so it is tracked in
#1039 rather than fixed on this branch.

**zqinucl/zdep (#552, closed by #1017 w6).** ``scheme.py`` now computes
ECHAM's section-1 ``zqinucl`` (``cloud_utils.karcher_lohmann_deposition_rate``,
mo_cloud_micro_2m.f90:1046-1102) and feeds it into section 5's ``zdep`` the
way the Fortran does (1449-1458), instead of hard-zeroing ``zdep`` at
``nic_cirrus=2``. Checked directly against the harness's own recorded
``zqinucl``/``dep0`` intermediates (added to ``cloud2m_cirrus_T63L47.npz``
for this fix; see the harness README) in
:func:`test_karcher_lohmann_deposition_rate_matches_compiled_fortran` and
:func:`test_nic_cirrus_2_zdep_formula_matches_compiled_fortran`: both match
to float64 round-off (0.0 on 4/5 designed columns; 6.2e-4 relative / 9.3e-15
absolute on the fifth, a deliberately degenerate zero-aerosol control cell
where ``apn_cm3 = max(1e-6*(papnx-icncq), 1e-6)`` subtracts two nearly-equal
numbers).

**Final q/qi are NOT a clean end-to-end check of this fix (measured, not
guessed) -- final ICNC remains the headline metric.** Comparing
``out/pqte``/``out/pxite`` (final humidity/ice) end-to-end the same way the
ICNC check below does: 3 of 5 columns match ECHAM to ~1e-8-1e-16 relative;
the other 2 (``cirrus_low_updraft``, ``cirrus_no_aerosol_floor_control`` --
the columns DESIGNED to have negligible homogeneous nucleation) diverge by
up to 42%. Root cause: ``deposition_freezing.py``'s Koop homogeneous-
freezing floor (section 12b, "interim toward #552") has NO ``nic_cirrus``
gate at all -- it fires identically whether ``nic_cirrus`` is 1 or 2,
whenever ``lo2 & T < cthomi``, which this design's T=210 K and S_ice=1.8
satisfy in EVERY one of the 5 columns. It is reachable on, but not
exclusive to, the ``nic_cirrus=2`` path (the #1017 w6 STOP condition for
touching it was "unless it is on the nic_cirrus=2 path"; it is reachable
there, but identically so on nic_cirrus=1, so it is left untouched here and
reported instead). In the 3 "normal" columns the proper deposition
branches (now including ``zqinucl``) already consume most of the
supersaturation before the Koop check runs, so its excess is negligible;
in the 2 designed-near-zero-nucleation columns, deposition barely touches
the supersaturation, so Koop's floor -- which ECHAM's own
``mo_cloud_micro_2m.f90`` has NO equivalent of at all -- deposits most of
it instead, a mechanism this test's reference never exercises. This is a
pre-existing divergence (the Koop floor predates this fix and does not
depend on ``zqinucl``), not a defect in the ``zqinucl``/``zdep`` port, so
no q/qi end-to-end check is added here.

**The eps-patch (#1039) experiment, with this fix in place.** Re-running
the ICNC comparison below with ``params.eps`` patched to
``np.finfo(np.float64).eps`` (a LOCAL, uncommitted experiment -- #1039
fixes this for real on its own branch): the 3 "normal" columns close from
~4e-5/4e-5/4e-5 relative to ~3e-7/5e-7/3e-7 (round-off, confirming #1039 is
their entire residual, unchanged by this fix). The other 2 columns stay at
3.11e-2 (not 3.10e-2 -- a negligible shift) regardless of the eps patch:
their residual is the SAME Koop-floor interaction described above, not
#1039, so patching #1039 alone cannot close it. Since the residual does
NOT uniformly close to round-off, ``RTOL_FLOAT64`` below is NOT tightened
(per the house rule: tighten only when it closes); it already comfortably
covers both measured mechanisms.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.clouds.cloud_utils import karcher_lohmann_deposition_rate
from jcm.physics.clouds.lohmann_2m.cirrus import xfrzmstr
from jcm.physics.clouds.lohmann_2m.scheme import cloud_microphysics_2m
from jcm.physics.clouds.lohmann_2m_fortran_reference_test import (
    echam_constants, echam_params, precision,
)

REF = (Path(__file__).resolve().parents[3] / "data" / "test"
       / "echam_cloud_reference" / "cloud2m_cirrus_T63L47.npz")
# Measured (not guessed) max relative error across the 5 designed columns,
# all attributable to precip_formation_cold (see module docstring): 6e-4
# (the 3 large-ICNC columns) to 3.1e-2 (the 2 columns whose final ICNC sits
# within ~2x of icemin=10, where the SAME few-percent absolute perturbation
# from precip_formation_cold is a larger fraction of a small number).
# Root cause tracked in #1039 (precip_formation_cold's ice-number-loss
# division guard uses a float32 eps unconditionally); not tightened here
# per the lead -- re-check this tolerance once #1039 is fixed.
RTOL_FLOAT64 = 0.05
ATOL_FLOAT64 = 1.0  # 1/m3, comfortably above icemin=10's own scale


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def test_karcher_lohmann_deposition_rate_matches_compiled_fortran():
    """``zqinucl`` itself, at every level of every designed column.

    Feeds the harness's own recorded section-1 intermediates (the SAME
    ones ``scheme.py`` reads: ``icncq_detr`` = zicncq before += zninucl,
    ``vervx``, ``rid``, ``rho``/``aaa``/``viscos``/``esi``/``qsi``) through
    ``xfrzmstr`` then :func:`karcher_lohmann_deposition_rate`, exactly as
    the production call site does, and compares to ``diag/zqinucl``.
    Matches to EXACT float64 equality on 4 of 5 columns; the 5th
    (``cirrus_no_aerosol_floor_control``, ``papnx=0``) differs by 9.3e-15
    absolute (6.2e-4 relative, amplified only because the value itself is
    ~1.5e-11 -- ordinary round-off through ``apn_cm3 = max(1e-6*(papnx -
    icncq), 1e-6)``'s near-cancelling subtraction, not a formula error).
    """
    z = load()
    names = [str(x) for x in z["meta/names"]]
    dt = float(z["config/time_step_len"])
    papm1 = z["in/papm1"]

    with jax.enable_x64(True), echam_constants(), precision("float64"):
        p = echam_params().replace(nic_cirrus=2)

        def one_level(sice, verv_ms, apn_cm3, t, pr, zrid, icncq_early,
                      qi_m1, cf, rho, adc, visc, esi, q_m1, qsi):
            ri_raw, pnicex = xfrzmstr(sice, verv_ms, apn_cm3, t, pr,
                                       jnp.asarray(dt), p)
            zninucl = jnp.minimum(pnicex, apn_cm3 * 1.0e6)
            _icncq_floored, zqinucl = karcher_lohmann_deposition_rate(
                ri_raw, zrid, icncq_early + zninucl, qi_m1, cf, rho, adc,
                visc, sice, t, pr, esi, q_m1, qsi, jnp.asarray(dt), p)
            return zqinucl

        sice = jnp.maximum(z["in/pqm1"] / jnp.maximum(z["diag/qsi"], 1e-300)
                            - 1.0, 0.0)
        verv_ms = z["diag/vervx"] / 100.0
        icncq_early = z["diag/icncq_detr"]
        apn_cm3 = jnp.maximum(1.0e-6 * (z["in/papnx"] - icncq_early), 1.0e-6)

        vmapped = jax.vmap(jax.vmap(one_level))
        zqinucl_jcm = vmapped(
            sice, verv_ms, apn_cm3, jnp.asarray(z["in/ptm1"]),
            jnp.asarray(papm1), jnp.asarray(z["diag/rid"]),
            jnp.asarray(icncq_early), jnp.asarray(z["in/pxim1"]),
            jnp.asarray(z["in/paclc"]), jnp.asarray(z["diag/rho"]),
            jnp.asarray(z["diag/aaa"]), jnp.asarray(z["diag/viscos"]),
            jnp.asarray(z["diag/esi"]), jnp.asarray(z["in/pqm1"]),
            jnp.asarray(z["diag/qsi"]))

    np.testing.assert_allclose(
        np.asarray(zqinucl_jcm), z["diag/zqinucl"], rtol=1e-9, atol=1e-13,
        err_msg="zqinucl vs diag/zqinucl (all levels, all columns)")

    # The headline claim: at least the designed cirrus level shows genuine
    # nonzero nucleated-vapour deposition, not a degenerate all-zero match.
    papm1_arr = np.asarray(papm1)
    ks = [int(np.argmin(np.abs(papm1_arr[:, j] - 23000.0)))
          for j in range(len(names))]
    assert np.max([float(zqinucl_jcm[k, j]) for j, k in enumerate(ks)]) > 1e-8


def test_nic_cirrus_2_zdep_formula_matches_compiled_fortran():
    """``zdep`` before 5.4 (``diag/dep0``), rebuilt from ``diag/zqinucl``.

    Isolates the dispatch ``scheme.py`` added at section 5 (``zdep =
    zqinucl*zifrac`` under dissipation, ``lo2*zqinucl`` under growth, 0
    under plain condensation) from the ``zqinucl`` computation itself
    (checked separately above): feeds the compiled reference's OWN
    ``zqinucl``/in-cloud ice-liquid/``lo2``/``zqcdif`` into that formula
    and compares to the compiled reference's OWN ``dep0``. Matches to
    EXACTLY 0.0 on every level of every column (measured).
    """
    z = load()
    zqcdif = z["diag/qcdif"]
    xib4, xlb4 = z["diag/xib_4"], z["diag/xlb_4"]
    lo2 = z["diag/lo2"]
    zqinucl = z["diag/zqinucl"]

    dissipation = zqcdif < 0.0
    zifrac = np.clip(xib4 / np.maximum(xib4 + xlb4, 1e-12), 0.0, 1.0)
    zdep0 = np.where(dissipation, zqinucl * zifrac, lo2 * zqinucl)

    np.testing.assert_allclose(zdep0, z["diag/dep0"], rtol=0.0, atol=0.0,
                                err_msg="zdep0 formula vs diag/dep0")


def test_nic_cirrus_2_end_to_end_matches_compiled_fortran():
    z = load()
    names = [str(x) for x in z["meta/names"]]
    dt = float(z["config/time_step_len"])

    with jax.enable_x64(True), echam_constants(), precision("float64"):
        p = echam_params().replace(nic_cirrus=2)

        def one(t1, q1, pr, qi1, qni1, cf, rho, dz_, tke, papnx):
            zero = jnp.zeros_like(t1)
            out = cloud_microphysics_2m(
                t1, q1, pr, zero, qi1, zero, qni1, cf, rho, dz_, tke,
                zero, zero, zero, jnp.asarray(dt, t1.dtype), p,
                cirrus_aerosol_number=papnx,
            )
            return out[0]

        rho = z["diag/rho"]
        dz = z["diag/dp"] / (rho * c.grav)
        args = [z["in/ptm1"], z["in/pqm1"], z["in/papm1"], z["in/pxim1"],
                z["in/xtm1_icnc"], z["in/paclc"], rho, dz, z["in/ptkem1"],
                z["in/papnx"]]
        tend = jax.vmap(one, in_axes=1, out_axes=1)(*[jnp.asarray(a) for a in args])

    qni_end = z["in/xtm1_icnc"] + dt * np.asarray(tend.dqnidt)
    icnc_end = qni_end * rho
    ref = z["out/picnc"]

    papm1 = z["in/papm1"]
    for j, n in enumerate(names):
        k = int(np.argmin(np.abs(papm1[:, j] - 23000.0)))
        np.testing.assert_allclose(
            icnc_end[k, j], ref[k, j], rtol=RTOL_FLOAT64, atol=ATOL_FLOAT64,
            err_msg=f"{n} (icnc_end vs picnc)")

    # The headline claim: non-degenerate cirrus columns reach ICNC well
    # above the floor (the #552 regression this whole task guards) -- not
    # just "close to the compiled reference's own floor-pinned value",
    # which the pre-fix wiring would also have passed trivially.
    assert icnc_end[int(np.argmin(np.abs(papm1[:, 0] - 23000.0))), 0] > 100.0 * 10.0


def test_karcher_lohmann_deposition_rate_gradients_finite_including_gated_cells():
    """No NaN/inf gradient through a gated-out (zero-ICNC) cell.

    ``cirrus_no_aerosol_floor_control`` has ``papnx=0`` everywhere, so most
    levels have ``icnc_before_floor == 0`` with ``ll_ice`` false (no
    pre-existing ice) -- exactly the discarded-branch division
    ``jnp.where`` still evaluates (JAX_gotchas.md's class; see the floor
    added in ``karcher_lohmann_deposition_rate`` next to ``zmmean``).
    """
    z = load()
    dt = float(z["config/time_step_len"])
    j = [str(x) for x in z["meta/names"]].index("cirrus_no_aerosol_floor_control")

    with jax.enable_x64(True), echam_constants(), precision("float64"):
        p = echam_params().replace(nic_cirrus=2)

        def total(air_density, ice_mmr_previous, cloud_fraction,
                   icnc_before_floor):
            _icnc, zqinucl = karcher_lohmann_deposition_rate(
                jnp.asarray(z["diag/zri_cirrus"][:, j]),
                jnp.asarray(z["diag/rid"][:, j]),
                icnc_before_floor,
                ice_mmr_previous,
                cloud_fraction,
                air_density,
                jnp.asarray(z["diag/aaa"][:, j]),
                jnp.asarray(z["diag/viscos"][:, j]),
                jnp.asarray(z["diag/sice"][:, j]),
                jnp.asarray(z["in/ptm1"][:, j]),
                jnp.asarray(z["in/papm1"][:, j]),
                jnp.asarray(z["diag/esi"][:, j]),
                jnp.asarray(z["in/pqm1"][:, j]),
                jnp.asarray(z["diag/qsi"][:, j]),
                jnp.asarray(dt), p)
            return jnp.sum(zqinucl)

        # ``icnc_before_floor`` is the PRE-floor sum (zicncq_detr + zninucl)
        # the production call site passes, not the already-floored
        # ``icncq_qd`` -- this column's ``papnx=0`` makes ``ninucl`` 0 too,
        # so most levels really do hit ``icnc_before_floor == 0``.
        grads = jax.grad(total, argnums=(0, 1, 2, 3))(
            jnp.asarray(z["diag/rho"][:, j]),
            jnp.asarray(z["in/pxim1"][:, j]),
            jnp.asarray(z["in/paclc"][:, j]),
            jnp.asarray(z["diag/icncq_detr"][:, j] + z["diag/ninucl"][:, j]))

    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g))), g


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
