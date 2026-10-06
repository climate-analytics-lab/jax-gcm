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
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
