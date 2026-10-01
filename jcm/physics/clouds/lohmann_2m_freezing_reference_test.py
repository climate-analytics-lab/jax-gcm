"""jcm's heterogeneous mixed-phase freezing against the ECHAM6.3-HAM2.3 Fortran (#953).

The reference ``jcm/data/test/echam_cloud_reference/cloud2m_frz_T63L47.npz``
holds the inputs, every INTENT(INOUT)/(OUT) argument and the freezing
intermediates of the UNMODIFIED ECHAM6.3-HAM2.3 r7492 routine
``mo_cloud_micro_2m.f90::cloud_micro_interface`` run on designed supercooled
cloud columns with the HAM freezing inputs of ``het_mxphase_freezing``
(F 2675-2840) set per level; ``cloud2m_frz_README.md`` documents every array
and ``cloud2m_frz_provenance.json`` how they were produced. It is the #941
harness with the aerosol inputs switched on (the harness reproduces the #941
fixture bit for bit with them off).

What is compared
----------------
* ``het_mxphase_freezing`` itself, fed ECHAM's own inputs to the routine
  (``hf_*_in`` and the section-5.5 cover), against ECHAM's state after it:
  the exact comparison, every freezing column including ``frz_omega`` (the
  function takes the large-scale omega; only the scheme does not plumb it).
* The section-6.2 wiring, through ``cloud_microphysics_2m(...,
  freezing_aerosol=)``: a spy pins the arguments section 6.2 hands the routine
  (ECHAM's gate on every cell, each HAM input in its slot, omega = 0); and the
  freezing EFFECT on the end-of-step crystal number and condensate (each
  column minus the aerosol-free ``frz_none``) against ECHAM's, at a stated
  tolerance. The ``frz_omega`` column is a strict xfail: the scheme has no
  large-scale omega (#705).

Constants and parameters are ECHAM's for the comparison (as in
``lohmann_2m_fortran_reference_test.py``, whose helpers are reused).
Tolerances: float64 ``rtol = 1e-9`` of the column scale (the block agrees to
round-off, ~1e-16), float32 ``rtol = 2e-3`` (the #941 test's; the float32
self-difference of these closed forms is ~1e-6 of scale).
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.clouds.lohmann_2m_fortran_reference_test import (
    RTOL,
    echam_constants,
    echam_params,
    precision,
)

REF = (Path(__file__).resolve().parents[2] / "data" / "test" / "echam_cloud_reference"
       / "cloud2m_frz_T63L47.npz")
STEPS = ("dt1200", "dt720")
PRECISIONS = ("float64", "float32")
FRZ_KEYS = ("fracdusol", "fracduai", "fracduci", "fracbcsol", "fracbcinsol",
            "rwetki", "rwetai", "rwetci")
ATOL = {"number": (1e-6, 1e-2), "mass": (1e-18, 1e-12)}


@functools.lru_cache(maxsize=None)
def load() -> dict:
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def inp() -> dict:
    return {k[3:]: v for k, v in load().items() if k.startswith("in/")}


def out(step) -> dict:
    return {k.split("/", 2)[2]: v for k, v in load().items() if k.startswith(f"out/{step}/")}


def diag(step) -> dict:
    return {k.split("/", 2)[2]: v for k, v in load().items() if k.startswith(f"diag/{step}/")}


def ztmst(step) -> float:
    return float(load()[f"timestep/{step}/time_step_len"])


def names() -> list[str]:
    return [str(s) for s in load()["meta/names"]]


def called_levels(step) -> np.ndarray:
    """Levels on which ECHAM called het_mxphase_freezing (some column passes the gate, F 1547)."""
    g = diag(step)
    return np.any(g["hf_mask"] > 0.5, axis=1, keepdims=True) & np.ones_like(g["hf_mask"], bool)


def _j(x, prec):
    return jnp.asarray(np.asarray(x, np.float64), jnp.float64 if prec == "float64" else jnp.float32)


def assert_close(field, family, jcm_v, ref, prec, where=None):
    jcm_v, ref = np.asarray(jcm_v, np.float64), np.asarray(ref, np.float64)
    if where is not None:
        jcm_v, ref = np.where(where, jcm_v, 0.0), np.where(where, ref, 0.0)
    atol = ATOL[family][0 if prec == "float64" else 1]
    tol = atol + RTOL[prec] * np.max(np.abs(ref), axis=0, keepdims=True)
    bad = np.abs(jcm_v - ref) > tol
    if np.any(bad):
        lines = [f"  {names()[j]} level {k}: jcm {jcm_v[k, j]:.12e} echam {ref[k, j]:.12e}"
                 for k, j in list(zip(*np.where(bad)))[:12]]
        raise AssertionError(f"{field}: {int(bad.sum())} element(s) outside tolerance\n" + "\n".join(lines))


# ===========================================================================
# the reference itself (ECHAM's behaviour; no jcm involved)
# ===========================================================================
@pytest.mark.parametrize("step", STEPS)
def test_reference_cooling_rate_is_vertical_motion(step):
    """ECHAM's immersion cooling rate is ztte = (omega - fact_tke*sqrt(TKE)*rho*g)/(cpd*rho)
    (F 2800-2802): the adiabatic cooling of the large-scale plus the turbulent updraft,
    not the model's temperature tendency (the columns carry none).
    """
    d, g = inp(), diag(step)
    lev = called_levels(step) & (g["hf_mask"] > 0.5)
    ztte = (d["pvervel"] - float(load()["param/fact_tke"]) * np.sqrt(d["ptkem1"]) * g["rho"] * 9.80665) \
        / (1004.64 * g["rho"])
    np.testing.assert_allclose(g["hf_ztte"][lev], ztte[lev], rtol=1e-13, atol=1e-20)
    assert np.all(d["ptte"] == 0.0)


@pytest.mark.parametrize("step", STEPS)
def test_reference_process_selectivity(step):
    """What each column isolates: no contact freezing without insoluble dust (BC contact is
    disabled, F 2784), no immersion without cooling (F 2804), complete immersion freezing at
    239 K, and a frozen number of 0 where CDNC sits at its floor (F 2819-2821) while mass freezes.
    """
    d, g, n = inp(), diag(step), names()
    gate = g["hf_mask"] > 0.5
    col = {name: j for j, name in enumerate(n)}
    j = col["frz_none"]
    assert np.all(g["hf_frl_ic"][gate[:, j], j] == 0.0)
    j = col["frz_bc"]
    assert gate[:, j].sum() == 5 and np.all(g["hf_frzcnt"][gate[:, j], j] == 0.0)
    assert np.all(g["hf_frzimm"][gate[:, j], j] > 0.0)
    j = col["frz_dust_nocool"]
    assert np.all(g["hf_frzimm"][gate[:, j], j] == 0.0) and np.all(g["hf_frzcnt"][gate[:, j], j] > 0.0)
    j = col["frz_dust"]
    k239 = int(np.argmin(np.abs(d["ptm1"][:, j] - 239.0)))
    assert gate[k239, j] and g["hf_xlb"][k239, j] == 0.0
    j = col["frz_outside_window"]
    k = int(np.argmin(np.abs(d["papm1"][:, j] - 60000.0)))
    assert gate[k, j] and g["hf_frl_ic"][k, j] > 0.0 and g["hf_frln"][k, j] == 0.0
    assert g["hf_cdnc_in"][k, j] == g["hf_cdncmin"][k, j]
    assert gate[:, j].sum() == 1


# ===========================================================================
# jcm's het_mxphase_freezing on ECHAM's inputs: the exact comparison
# ===========================================================================
def run_jcm_block(step, prec):
    from jcm.physics.clouds.lohmann_2m.deposition_freezing import het_mxphase_freezing
    d, g = inp(), diag(step)
    J = functools.partial(_j, prec=prec)
    o = het_mxphase_freezing(
        jnp.asarray(g["hf_mask"] > 0.5), J(d["papm1"]), J(d["ptkem1"]), J(d["pvervel"]),
        J(g["uicw_aclc"]), J(d["fracbcsol"]), J(d["fracbcinsol"]), J(d["fracdusol"]),
        J(d["fracduai"]), J(d["fracduci"]), J(g["rho"]), 1.0 / J(g["rho"]), J(d["rwetki"]),
        J(d["rwetai"]), J(d["rwetci"]), J(g["hf_tp1tmp"]), J(g["hf_cdncmin"]),
        J(g["hf_icnc_in"]), J(g["hf_cdnc_in"]), J(g["frl_hom"]), J(g["hf_xib_in"]),
        J(g["hf_xlb_in"]), jnp.asarray(ztmst(step), J(0.0).dtype),
        float(load()["param/cqtmin"]), echam_params())
    return dict(zip(("hf_icnc", "hf_cdnc", "hf_frl", "hf_xib", "hf_xlb", "hf_frln"),
                    (np.asarray(x) for x in o)))


@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_het_mxphase_freezing_matches_echam(step, prec):
    """ICNC, CDNC, grid-mean freezing (pfrl*paclc), in-cloud ice and liquid, and the frozen
    number, on every level ECHAM called the routine, in every freezing column.
    """
    with echam_constants(), precision(prec):
        g = diag(step)
        r = run_jcm_block(step, prec)
        lev = called_levels(step)
        assert int(lev[:, 0].sum()) >= 5
        for k, fam in (("hf_icnc", "number"), ("hf_cdnc", "number"), ("hf_frln", "number"),
                       ("hf_frl", "mass"), ("hf_xib", "mass"), ("hf_xlb", "mass")):
            ref = np.where(g["hf_mask"] > 0.5, g[k], 0.0) if k == "hf_frln" else g[k]
            assert_close(k, fam, r[k], ref, prec, where=lev)


# ===========================================================================
# section 6.2 wiring, end to end
# ===========================================================================
def run_jcm_column(step, prec, columns):
    """cloud_microphysics_2m with the HAM freezing inputs; the #941 test's mapping of ECHAM's
    inputs to jcm's anchor and increments (``ptm1`` etc. as the anchor, ``ztmst`` times the
    accumulated tendencies as the increments; no detrainment here).
    """
    from jcm.physics.clouds.lohmann_2m.scheme import cloud_microphysics_2m
    from jcm.physics.clouds.lohmann_2m.types import HeterogeneousFreezingAerosol
    p = echam_params()
    d, g, dt = inp(), diag(step), ztmst(step)
    sel = lambda a: np.asarray(a)[:, columns]  # noqa: E731
    prov = dict(T=d["ptm1"] + dt * d["ptte"], q=d["pqm1"] + dt * d["pqte"],
                qc=d["pxlm1"] + dt * d["pxlte"], qi=d["pxim1"] + dt * d["pxite"],
                qnc=d["xtm1_cdnc"] + dt * d["xtte_cdnc"], qni=d["xtm1_icnc"] + dt * d["xtte_icnc"])
    dz = g["rho"] * 0.0 + (np.diff(d["paphm1"], axis=0) / (g["rho"] * c.grav))

    def one(t1, q1, qc1, qi1, qnc1, qni1, cf, rho, dz_, tke, pr,
            dT, dq, dqc, dqi, dqnc, dqni, *frz):
        zero = jnp.zeros_like(t1)
        fa = HeterogeneousFreezingAerosol(
            dust_soluble=frz[0], dust_insoluble_accumulation=frz[1], dust_insoluble_coarse=frz[2],
            bc_soluble=frz[3], bc_insoluble=frz[4], wet_radius_insoluble_aitken=frz[5],
            wet_radius_insoluble_accumulation=frz[6], wet_radius_insoluble_coarse=frz[7])
        o = cloud_microphysics_2m(
            t1, q1, pr, qc1, qi1, qnc1, qni1, cf, rho, dz_, tke, zero, zero, zero,
            jnp.asarray(dt, t1.dtype), p, temperature_increment=dT, humidity_increment=dq,
            qc_increment=dqc, qi_increment=dqi, qnc_increment=dqnc, qni_increment=dqni,
            freezing_aerosol=fa)
        return o[0]

    args = [d["ptm1"], d["pqm1"], d["pxlm1"], d["pxim1"], d["xtm1_cdnc"], d["xtm1_icnc"],
            d["paclc"], g["rho"], dz, d["ptkem1"], d["papm1"],
            dt * d["ptte"], dt * d["pqte"], dt * d["pxlte"], dt * d["pxite"],
            dt * d["xtte_cdnc"], dt * d["xtte_icnc"]]
    args += [d[k] for k in FRZ_KEYS]
    tend = jax.vmap(one, in_axes=1, out_axes=1)(*[_j(sel(a), prec) for a in args])
    return dict(qc=sel(prov["qc"]) + dt * np.asarray(tend.dqcdt),
                qi=sel(prov["qi"]) + dt * np.asarray(tend.dqidt),
                qni=sel(prov["qni"]) + dt * np.asarray(tend.dqnidt))


def _freezing_effect_error(step, which):
    """Jcm's and ECHAM's freezing EFFECT on the end-of-step state in the gate cells: each
    column minus ``frz_none`` (the same thermodynamics without aerosol), so the parts of
    the step that do not involve freezing cancel. Returns, per column and field, the
    largest |jcm - ECHAM| over the column's largest |ECHAM effect|.
    """
    d, o, g, dt = inp(), out(step), diag(step), ztmst(step)
    n = names()
    cols = [n.index("frz_none")] + [n.index(c_) for c_ in which]
    with echam_constants(), precision("float64"):
        jc = run_jcm_column(step, "float64", cols)
    ref = dict(qc=d["pxlm1"] + dt * o["pxlte"], qi=d["pxim1"] + dt * o["pxite"],
               qni=d["xtm1_icnc"] + dt * o["pxtte_icnc"])
    err = {}
    for i, name in enumerate(which, start=1):
        j = cols[i]
        gate = g["hf_mask"][:, j] > 0.5
        assert gate.sum() >= 1
        for fld in ("qni", "qi", "qc"):
            de = ref[fld][gate, j] - ref[fld][gate, cols[0]]
            dj = jc[fld][gate, i] - jc[fld][gate, 0]
            err[(name, fld)] = float(np.max(np.abs(dj - de)) / max(np.max(np.abs(de)), 1e-30))
    return err


FREEZING_COLUMNS = ("frz_dust", "frz_bc", "frz_both", "frz_dust_nocool")

# Without any aerosol (frz_none) jcm's end-of-step liquid is ECHAM's to 7e-5 of the
# column maximum, and the measured worst freezing effect over these columns and both
# steps is 2.4e-3 (frz_bc, liquid); the tolerance leaves a factor of four. A mis-wired
# input misses by far more: frz_omega, whose only difference is the large-scale omega
# jcm lacks, misses by 18-100 %.
EFFECT_TOL = 1e-2


@pytest.mark.parametrize("step", STEPS)
def test_section62_freezing_effect_end_to_end(step):
    """The rates act in the scheme as in ECHAM: the change the HAM freezing inputs make to
    the end-of-step ICNC, cloud ice and liquid in the gate cells.
    """
    err = _freezing_effect_error(step, FREEZING_COLUMNS)
    bad = {k: v for k, v in err.items() if v > EFFECT_TOL}
    assert not bad, bad


@pytest.mark.xfail(strict=True, reason="the 2M scheme has no large-scale omega (#705): "
                   "ECHAM's immersion cooling in frz_omega includes it, jcm's is TKE-only")
@pytest.mark.parametrize("step", STEPS)
def test_section62_with_large_scale_omega(step):
    """frz_omega: ascent and subsidence change ECHAM's immersion freezing (F 2800)."""
    err = _freezing_effect_error(step, ("frz_omega",))
    assert all(v <= EFFECT_TOL for v in err.values()), err


def test_section62_hands_the_ham_inputs_to_het_mxphase_freezing(monkeypatch):
    """The wiring, exactly: section 6.2 calls het_mxphase_freezing with ECHAM's gate on
    every gate cell, the TKE, omega = 0 (#705), and each HAM input in its own slot.
    """
    import jcm.physics.clouds.lohmann_2m.scheme as scheme
    real = scheme.het_mxphase_freezing
    seen = []

    def spy(*a):
        jax.debug.callback(lambda *v: seen.append(tuple(np.asarray(x) for x in v)),
                           a[1], a[0], a[2], a[3], a[5], a[6], a[7], a[8], a[9], a[12], a[13], a[14])
        return real(*a)

    monkeypatch.setattr(scheme, "het_mxphase_freezing", spy)
    step = "dt720"
    d, g = inp(), diag(step)
    j = names().index("frz_both")
    with echam_constants(), precision("float64"):
        run_jcm_column(step, "float64", [j])
    assert len(seen) == d["ptm1"].shape[0]
    for rec in seen:
        k = int(np.argmin(np.abs(d["papm1"][:, j] - rec[0])))
        assert rec[0] == d["papm1"][k, j]
        assert bool(rec[1]) == bool(g["hf_mask"][k, j] > 0.5), k
        assert rec[2] == d["ptkem1"][k, j] and rec[3] == 0.0
        for got, key in zip(rec[4:], ("fracbcsol", "fracbcinsol", "fracdusol", "fracduai", "fracduci",
                                      "rwetki", "rwetai", "rwetci")):
            assert got == d[key][k, j], (key, k)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
