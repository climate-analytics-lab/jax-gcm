"""jcm's full ``nic_cirrus=2`` column against the compiled ECHAM6.3-HAM2.3 chain (#1017).

``cloud2m_cirrus_T63L47.npz`` holds the compiled, UNMODIFIED
``mo_cloud_micro_2m.f90::cloud_micro_interface`` run with the UNMODIFIED
``mo_cirrus.f90::xfrzmstr`` linked in (``nic_cirrus=2``), fed
``pascs``/``papnx``/``paprx``/``papsigx`` through the same ``cloud_subm_1``
interface ``ham_IN_setup`` uses, on 5 designed columns: a pure-cirrus
supersaturated cell, the same cell with pre-existing ice, two updraft
variants and a no-aerosol floor control (``cloud2m_cirrus_README.md``,
``_provenance.json``). Numbers only; no ECHAM source is in the repository.

What the column must reproduce, each checked here against the compiled
numbers:

- the ``xfrzmstr`` call sits in section 1, before sedimentation and melting,
  with ECHAM's own depletion reference ``zicncq = icnc0 + znidetr`` and the
  step-start ice supersaturation ``q_m1/qsi - 1`` (mo_cloud_micro_2m.f90:982,
  1050-1058), and its nucleated number joins ``zicncq`` there;
- the depositional growth of the nucleated crystals, ``zqinucl``
  (``cloud_utils.karcher_lohmann_deposition_rate``, :1046-1102), is section
  5's deposition leg ``zdep`` (:1449-1458);
- at ``nic_cirrus=2`` jcm's Koop homogeneous-freezing floor (a stand-in for
  the sink ``nic_cirrus=1`` lacks; r7492 has no such floor) is off, and
  ``ll_het`` is ECHAM's ``.false.`` (``ld_het = lhetfreeze``,
  mo_ham_freezing.f90:156).

The ICNC comparison is jcm's tracer tendency ``dqnidt`` against the compiled
tracer tendency ``out/pxtte_icnc``. The same-step ``out/picnc`` is a
different quantity: in a near-zero-ice-mass cell the ccwmin mass-consistency
repair (mo_cloud_micro_2m.f90:3640-3660) zeroes the tracer tendency but not
that diagnostic.

Measured (float64): ``zqinucl`` and the ``zdep`` dispatch match to round-off
(0.0 on 4 of 5 columns; 9.3e-15 absolute on the zero-aerosol control, a
near-cancelling subtraction). End to end, ``dqnidt`` matches ``pxtte_icnc``
to <= 3.23e-7 relative on every column (exactly 0.0 on the two
near-zero-nucleation columns), with ``precip_formation_cold``'s guards at the
working dtype's epsilon as in r7492 (``EPSILON(1._dp)``).
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
# Measured max relative error of ``dqnidt`` against ``out/pxtte_icnc`` across
# the 5 designed columns: 3.23e-7 (see the module docstring); ~3x margin.
RTOL_FLOAT64 = 1.0e-6
ATOL_FLOAT64 = 1.0e-3  # 1/kg/s; both sides are exactly 0.0 on 2 of 5 columns


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

    # Compare the ICNC TRACER TENDENCY (jcm's ``dqnidt``) against the
    # compiled reference's own tracer tendency ``out/pxtte_icnc`` -- the
    # true Fortran counterpart of ``dqnidt`` (both are per-kg-of-air
    # tendencies the host adds to the advected tracer). ``out/picnc`` is a
    # DIFFERENT quantity (this-step's internal number, read by radiation/
    # precip but not itself subject to the ccwmin mass-consistency repair
    # the tracer tendency gets) -- see the module docstring; comparing
    # against it was this test's own pre-existing bug, not a code defect.
    ref = z["out/pxtte_icnc"]

    papm1 = z["in/papm1"]
    for j, n in enumerate(names):
        k = int(np.argmin(np.abs(papm1[:, j] - 23000.0)))
        np.testing.assert_allclose(
            tend.dqnidt[k, j], ref[k, j], rtol=RTOL_FLOAT64, atol=ATOL_FLOAT64,
            err_msg=f"{n} (dqnidt vs pxtte_icnc)")

    # The headline claim: non-degenerate cirrus columns reach ICNC well
    # above the floor (the #552 regression this whole task guards) -- not
    # just "close to the compiled reference's own floor-pinned value",
    # which the pre-fix wiring would also have passed trivially. This is
    # jcm's own internal reconstruction (qni anchor + dt*dqnidt, converted
    # to 1/m3), a physical sanity check, not a second comparison against
    # the Fortran -- the cross-reference check is the one above.
    qni_end = z["in/xtm1_icnc"] + dt * np.asarray(tend.dqnidt)
    icnc_end = qni_end * rho
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
