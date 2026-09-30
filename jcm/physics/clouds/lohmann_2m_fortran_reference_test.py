"""jcm's Lohmann 2M ice sources (#941) against the ECHAM6.3-HAM2.3 Fortran, number for number.

The reference ``jcm/data/test/echam_cloud_reference/cloud2m_T63L47.npz`` holds
the inputs, every INTENT(INOUT)/(OUT) argument and 111 intermediates of the
UNMODIFIED ECHAM6.3-HAM2.3 r7492 routine
``mo_cloud_micro_2m.f90::cloud_micro_interface`` (the whole routine, compiled
standalone without HAMMOZ, aerosol-free, ``nic_cirrus = 1``), run on designed
T63L47 columns at two time steps (``dt1200``: ECHAM's T63 leapfrog step;
``dt720``: jcm's default T63 step). ``cloud2m_README.md`` documents every
array; ``cloud2m_provenance.json`` records how they were produced. Line
numbers ``F <n>`` below refer to that ``mo_cloud_micro_2m.f90``.

What is compared
----------------
Each test feeds a jcm building block ECHAM's own inputs to that block (read
from the intermediates) and compares its outputs, so a failure names the
block and is not contaminated by differences elsewhere in the column:

* ``zrid`` (F 945-956): the temperature-parameterised crystal radius.
* section-1 criterion ``lo2_2d`` (F 812-885): updraft ``zvervx``, Schumann
  radius ``zrice``, threshold ``zvervmax``.
* section-4 criterion ``lo2`` (F 1276-1298) on the post-sedimentation ice.
* ``sedimentation_ice`` on ECHAM's sedimentation input, which excludes this
  step's detrained condensate (F 1227-1248; the fixture itself shows it).
* ``update_in_cloud_water`` with ``prid = zrid`` (F 1511, 2610-2624): the ICNC
  diagnosis.
* ``znidetr`` (F 958-983) -- needs the #941 core helper.
* End to end (needs the #941 ``cloud_microphysics_2m(..., detrained_qc=,
  detrained_qi=)`` signature): the number tendencies against the RAW tracer
  (F 1781, 3625-3652) where ECHAM pins the end-of-step number, and the end
  state of clear cells that receive detrained condensate (the section-4
  split and the ``ptte`` fix, F 1300-1317).

Tests that need the merged #941 core skip on a jcm without it, keyed on the
``detrained_qc`` keyword of ``cloud_microphysics_2m``; once that keyword
exists they run, and a missing helper is then a FAILURE (the adapter must be
updated), never a silent skip.

Physical constants
------------------
jcm's global constants differ from ECHAM's (grav, rv, alv, als, eps). Every
comparison runs with them set to ECHAM's values and restores them afterwards,
so the formulation is tested rather than the constants (the maintainer keeps
jcm's set; ``test_cloud_params_relevant_to_941_match_echam`` checks the
scheme parameters the #941 pieces use and pins the known differences).

Tolerances
----------
An element passes if ``|jcm - ref| <= atol[field] + rtol * scale`` with
``scale`` the column's largest ``|ref|`` of that field, as in the 1M
reference test (echam_fortran_reference_test.py on feat/m1-reference):
float64 ``rtol = 1e-9`` (a faithful port differs only by operation-order
roundoff, ~1e-15; the closed-form pieces here agree exactly), float32
``rtol = 2e-3`` (jcm's float32-vs-float64 self-difference is ~2e-4 of
scale). ``atol`` floors sit far below any physical signal but above ECHAM's
own numerical floors (``cqtmin = 1e-12`` numbers, the ``EPSILON(1d0)``
sedimentation floor of F 1228). Boolean decisions must agree exactly.

Run: ``JAX_PLATFORMS=cpu pytest jcm/physics/clouds/lohmann_2m_fortran_reference_test.py``
"""
from __future__ import annotations

import contextlib
import functools
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c

REF = Path(__file__).resolve().parents[2] / "data" / "test" / "echam_cloud_reference" / "cloud2m_T63L47.npz"

# ECHAM6.3 mo_physical_constants.f90:96-147 (rd = 287.04 = akap*cpd already agrees).
ECHAM_CONSTANTS = dict(grav=9.80665, rv=461.51, alhc=2.5008e6, alhs=2.8345e6,
                       eps=287.04 / 461.51, cpd=1004.64, cpv=1869.46, tmelt=273.15,
                       rhow=1000.0)
STEPS = ("dt1200", "dt720")
PRECISIONS = ("float64", "float32")
RTOL = {"float64": 1e-9, "float32": 2e-3}
# atol floors per field family (float64 / float32)
ATOL = {
    "radius": (1e-20, 1e-12),      # m
    "velocity": (1e-20, 1e-9),     # m/s, cm/s
    "number": (1e-6, 1e-3),        # 1/m3 (ECHAM floors are cqtmin = 1e-12, icemin = 10)
    "number_flux": (1e-9, 1e-3),   # 1/m2/s
    "mass": (1e-15, 1e-10),        # kg/kg (ECHAM's EPSILON(1d0) sedimentation floor is 2.2e-16)
    "mass_flux": (1e-15, 1e-10),   # kg/m2/s
    "fraction": (1e-12, 1e-6),
    "temperature": (1e-9, 1e-4),   # K (end state)
    "tracer": (1e-6, 1e-2),        # 1/kg
}


# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def load() -> dict:
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def echam_in() -> dict:
    z = load()
    return {k[3:]: z[k] for k in z if k.startswith("in/")}


def echam_out(step: str) -> dict:
    z = load()
    pre = f"out/{step}/"
    return {k[len(pre):]: z[k] for k in z if k.startswith(pre)}


def echam_diag(step: str) -> dict:
    z = load()
    pre = f"diag/{step}/"
    return {k[len(pre):]: z[k] for k in z if k.startswith(pre)}


def ztmst(step: str) -> float:
    return float(load()[f"timestep/{step}/time_step_len"])


def column_names() -> list[str]:
    return [str(s) for s in load()["meta/names"]]


@contextlib.contextmanager
def echam_constants():
    """Set jcm's global constants to ECHAM's for the duration, then restore them."""
    saved = c.physical_constants
    c.set_constants(**ECHAM_CONSTANTS)
    try:
        yield
    finally:
        c.set_constants(saved)


@contextlib.contextmanager
def precision(name: str):
    with jax.enable_x64(name == "float64"):
        yield


def _dtype(prec):
    return jnp.float64 if prec == "float64" else jnp.float32


def assert_close(field: str, family: str, jcm, ref, prec: str, what: str = ""):
    """Per-column tolerance ``atol + rtol * column scale``; report failing columns/levels."""
    jcm = np.asarray(jcm, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert jcm.shape == ref.shape, (field, jcm.shape, ref.shape)
    atol = ATOL[family][0 if prec == "float64" else 1]
    scale = np.max(np.abs(ref), axis=0, keepdims=True) if ref.ndim == 2 else np.max(np.abs(ref))
    tol = atol + RTOL[prec] * scale
    bad = np.abs(jcm - ref) > tol
    if np.any(bad):
        names = column_names()
        lines = []
        for k, j in zip(*np.where(bad)) if ref.ndim == 2 else [(i, None) for i in np.where(bad)[0]]:
            col = names[j] if j is not None else "-"
            lines.append(f"  {col} level {k}: jcm {jcm[k] if j is None else jcm[k, j]:.12e} "
                         f"echam {ref[k] if j is None else ref[k, j]:.12e}")
            if len(lines) >= 12:
                break
        raise AssertionError(f"{what}{field}: {int(bad.sum())} element(s) outside tolerance\n"
                             + "\n".join(lines))


def has_merged_core() -> bool:
    """Return whether jcm's cloud_microphysics_2m takes the #941 detrainment arguments."""
    from jcm.physics.clouds.lohmann_2m.scheme import cloud_microphysics_2m
    return "detrained_qc" in inspect.signature(cloud_microphysics_2m).parameters


needs_merged_core = pytest.mark.skipif(
    not has_merged_core(),
    reason="jcm's cloud_microphysics_2m has no detrained_qc/detrained_qi arguments yet: the "
           "#941 core (znidetr, ICE-1/3/4/5) is not merged into this checkout")


# jcm tunables that differ from ECHAM6.3-HAM2.3 at T63 (the maintainer's tuning, kept in
# jcm). The comparisons use ECHAM's values so the formulation is tested, not the tuning;
# ``test_cloud_params_relevant_to_941_match_echam`` pins the differences.
KNOWN_PARAM_DIFFERENCES = {"ccsaut": (900.0, 95.0), "ccraut": (10.6, 15.0)}


def echam_params():
    """CloudParams2M for the comparisons: jcm's defaults built under ECHAM's constants
    (tmelt, grav, cthomi follow them), ECHAM's aggregation/autoconversion rates
    ``ccsaut``/``ccraut`` (KNOWN_PARAM_DIFFERENCES), and ``activation_smoothing = 0``
    (jcm's smooth max on the activation increment; ECHAM's is a hard MAX, F 2599-2600).
    ``n_aer_coarse`` stays: DeMott runs only in jcm's mixed-phase freezing substitute,
    which acts only on supercooled liquid in cloud, and no compared cell holds any.
    """
    from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
    return CloudParams2M.default(
        activation_smoothing=0.0,
        **{k: echam for k, (_jcm, echam) in KNOWN_PARAM_DIFFERENCES.items()})


def _j(x, prec):
    return jnp.asarray(np.asarray(x, dtype=np.float64), dtype=_dtype(prec))


# ===========================================================================
# ADAPTERS -- the ONLY code in this module that knows jcm's interfaces.
# Each takes ECHAM's names (arrays (nlev, ncol), TOP-FIRST, as the Fortran
# received them) and returns ECHAM's names. When jcm's signatures change,
# edit these (and nothing else).
# ===========================================================================
def run_jcm_zrid(t_m1, prec):
    """ECHAM zrid [m] (F 945-956) from the step-start temperature ptm1."""
    from jcm.physics.clouds import cloud_utils as cu
    p = echam_params()
    helper = getattr(cu, "ice_volume_mean_radius_from_temperature", None)
    t = _j(t_m1, prec)
    if helper is not None:                      # the #941 core helper
        return np.asarray(helper(t, p))
    if has_merged_core():
        raise AssertionError("merged #941 core but no zrid helper: update run_jcm_zrid")
    # dev: the effective radius parameterisation is one line; the conversion is jcm's helper.
    reff = jnp.maximum(23.2 * jnp.exp(0.015 * jnp.minimum(t - p.tmelt, 0.0)), 1.0)
    return np.asarray(cu.effective_2_volmean_radius_param_Schuman_2011(reff, p))


def run_jcm_wbf_criterion(xip1, picnc, paclc, rho, esw, esi, eta, tkem1, prec):
    """Return zvervx [cm/s], zrice [m], zvervmax [m/s] and 0.01*zvervx < zvervmax from jcm's
    helpers, as ECHAM computes them in section 1 (F 814-885) and section 4 (F 1281-1298).
    """
    from jcm.physics.clouds import cloud_utils as cu
    p = echam_params()
    ice_gm3 = 1000.0 * _j(xip1, prec) * _j(rho, prec) / jnp.maximum(_j(paclc, prec), p.clc_min)
    zrice = cu.ice_volume_mean_radius_schumann(ice_gm3, _j(picnc, prec), p)
    zvervmax = cu.threshold_vert_vel(sat_vap_pres_water=_j(esw, prec), sat_vap_pres_ice=_j(esi, prec),
                                     icnc=_j(picnc, prec), ice_radius=zrice, eta=_j(eta, prec),
                                     params=p)
    zvervx = cu.turbulent_updraft_velocity(_j(tkem1, prec), p)   # pvervel = 0 in every column
    return dict(vervx=np.asarray(zvervx), rice=np.asarray(zrice), vervmax=np.asarray(zvervmax),
                crit=np.asarray(0.01 * zvervx < zvervmax))


def run_jcm_sedimentation(g, dt, prec):
    """Run jcm's sedimentation_ice on ECHAM's per-level section-4 inputs."""
    from jcm.physics.clouds.lohmann_2m.sedimentation_melt import sedimentation_ice
    p = echam_params()
    inp = echam_in()
    out = sedimentation_ice(
        _j(inp["paclc"], prec), _j(g["aaa"], prec), _j(g["dp"], prec), _j(g["rho"], prec),
        1.0 / _j(g["rho"], prec), _j(g["sed_xip1_in"], prec), _j(g["sed_icnc_in"], prec),
        _j(g["sed_xiflux_in"], prec), _j(g["sed_xifluxn_in"], prec), _j(g["sed_clcfi_in"], prec),
        jnp.asarray(dt, _dtype(prec)), p)
    names = ("sed_xip1", "sed_icnc", "sed_xiflux", "sed_xifluxn", "sed_clcfi", "sed_mrateps")
    return {k: np.asarray(v) for k, v in zip(names, out)}


def run_jcm_update_in_cloud_water(g, dt, prec):
    """Run jcm's update_in_cloud_water on ECHAM's section-5.5 inputs, prid = zrid (F 1504-1516)."""
    from jcm.physics.clouds.lohmann_2m.assembly import update_in_cloud_water
    p = echam_params()
    inp = echam_in()
    zero = jnp.zeros_like(_j(g["cnd"], prec))
    out = update_in_cloud_water(
        _j(inp["papm1"], prec), zero, _j(g["cnd"], prec), _j(g["dep"], prec), zero, zero, zero,
        _j(g["qp1tmp"], prec), _j(g["qsp1tmp"], prec), _j(g["rho"], prec),
        _j(run_jcm_zrid(inp["ptm1"], prec), prec), _j(inp["ptm1"], prec),
        jnp.asarray(g["ll_cc"] > 0.5), _j(g["uicw_icnc_in"], prec), zero,
        _j(g["uicw_cdnc_in"], prec), _j(g["uicw_aclc_in"], prec), _j(g["uicw_xib_in"], prec),
        _j(g["uicw_xlb_in"], prec), jnp.asarray(dt, _dtype(prec)), p)
    names = ("ll_cc_out", "uicw_icnc", "qnuc", "uicw_cdnc", "uicw_aclc", "uicw_xib", "uicw_xlb",
             "cdnc_min")
    return {k: np.asarray(v) for k, v in zip(names, out)}


def run_jcm_znidetr(g, dt, prec):
    """ECHAM znidetr [1/m3] (F 958-983) from jcm's #941 helper, on ECHAM's zxtec,
    ptm1, lo2_2d, paclc, rho and zrid.
    """
    from jcm.physics.clouds import cloud_utils as cu
    helper = getattr(cu, "detrained_ice_crystal_number", None)
    if helper is None:
        raise AssertionError("merged #941 core but no znidetr helper "
                             "(cloud_utils.detrained_ice_crystal_number): update run_jcm_znidetr")
    inp = echam_in()
    return np.asarray(helper(
        _j(dt * g["xtec"], prec), _j(inp["ptm1"], prec), jnp.asarray(g["lo2_2d"] > 0.5),
        _j(inp["paclc"], prec), _j(g["rho"], prec), _j(g["rid"], prec), echam_params()))


def run_jcm_column(inp: dict, g: dict, dt: float, prec: str) -> dict:
    """ECHAM cloud_micro_interface -> jcm cloud_microphysics_2m, column by column.

    Mapping (ECHAM's leapfrog tendencies -> jcm's operator-split provisional state):
    T = ptm1 + ztmst*ptte, q = pqm1 + ztmst*pqte, qc = pxlm1 + ztmst*pxlte +
    detrained_qc, qi = pxim1 + ztmst*pxite + detrained_qi, the ``*_m1`` = ECHAM's m1
    fields; the detrained condensate ztmst*zxtec is split into liquid and ice at
    ztconv > tmelt, exactly as cudtdq split it (mo_cufluxdts.f90:645-664, pten = the
    convection temperature ztconv); the raw number tracers qnc/qni = pxtm1 +
    ztmst*pxtte (per kg of air); air density = ECHAM's papm1/(rd*ptvm1); layer
    thickness dp/(rho*grav) so jcm's rho*g*dz is ECHAM's dp; TKE = ptkem1; no aerosol
    (activated_cdnc = ice_nuclei = 0). Returns the end-of-step state X + dt*dX/dt.
    """
    from jcm.physics.clouds.lohmann_2m.scheme import cloud_microphysics_2m
    p = echam_params()
    nlev, ncol = inp["ptm1"].shape
    xt = dt * inp["zxtec"]
    ice = inp["ztconv"] <= c.tmelt
    det_qi, det_qc = np.where(ice, xt, 0.0), np.where(ice, 0.0, xt)
    prov = dict(T=inp["ptm1"] + dt * inp["ptte"], q=inp["pqm1"] + dt * inp["pqte"],
                qc=inp["pxlm1"] + dt * inp["pxlte"] + det_qc,
                qi=inp["pxim1"] + dt * inp["pxite"] + det_qi,
                qnc=inp["xtm1_cdnc"] + dt * inp["xtte_cdnc"],
                qni=inp["xtm1_icnc"] + dt * inp["xtte_icnc"])
    dz = g["dp"] / (g["rho"] * c.grav)

    def one(T, q, qc, qi, qnc, qni, cf, rho, dz_, tke, pr, t1, q1, qc1, qi1, dqc, dqi):
        zero = jnp.zeros_like(T)
        out = cloud_microphysics_2m(
            T, q, pr, qc, qi, qnc, qni, cf, rho, dz_, tke, zero, zero, zero,
            jnp.asarray(dt, T.dtype), p, temperature_m1=t1, specific_humidity_m1=q1,
            qc_m1=qc1, qi_m1=qi1, detrained_qc=dqc, detrained_qi=dqi)
        return out[0]

    args = [prov["T"], prov["q"], prov["qc"], prov["qi"], prov["qnc"], prov["qni"],
            inp["paclc"], g["rho"], dz, inp["ptkem1"], inp["papm1"], inp["ptm1"], inp["pqm1"],
            inp["pxlm1"], inp["pxim1"], det_qc, det_qi]
    tend = jax.vmap(one, in_axes=1, out_axes=1)(*[_j(a, prec) for a in args])
    return dict(T=prov["T"] + dt * np.asarray(tend.dtedt), q=prov["q"] + dt * np.asarray(tend.dqdt),
                qc=prov["qc"] + dt * np.asarray(tend.dqcdt), qi=prov["qi"] + dt * np.asarray(tend.dqidt),
                qnc=prov["qnc"] + dt * np.asarray(tend.dqncdt),
                qni=prov["qni"] + dt * np.asarray(tend.dqnidt))


def echam_end_state(step: str) -> dict:
    """ECHAM's end-of-step state X_m1 + ztmst*pXte (its tendencies are relative to m1)."""
    inp, o, dt = echam_in(), echam_out(step), ztmst(step)
    return dict(T=inp["ptm1"] + dt * o["ptte"], q=inp["pqm1"] + dt * o["pqte"],
                qc=inp["pxlm1"] + dt * o["pxlte"], qi=inp["pxim1"] + dt * o["pxite"],
                qnc=inp["xtm1_cdnc"] + dt * o["pxtte_cdnc"],
                qni=inp["xtm1_icnc"] + dt * o["pxtte_icnc"])


# ===========================================================================
# the reference itself (documents ECHAM's behaviour; no jcm involved)
# ===========================================================================
@pytest.mark.parametrize("step", STEPS)
def test_reference_sedimentation_excludes_detrainment(step):
    """ECHAM sediments max(pxim1 + ztmst*pxite, EPSILON) with pxite the upstream (+melt)
    tendency only; the detrained condensate zxtec joins through zxidt afterwards
    (F 1227-1228, 1316). Checked on every level of every column.
    """
    inp, g, dt = echam_in(), echam_diag(step), ztmst(step)
    expect = np.maximum(inp["pxim1"] + dt * g["mlt_xite"], np.finfo(float).eps)
    np.testing.assert_array_equal(g["sed_xip1_in"], expect)
    j = column_names().index("sediment_then_detrain")
    k = int(np.argmax(inp["zxtec"][:, j]))
    assert g["sed_xip1_in"][k, j] == inp["pxim1"][k, j] + dt * inp["pxite"][k, j]
    assert g["xidt"][k, j] == pytest.approx(dt * (g["sed_xite"][k, j] + inp["zxtec"][k, j]), rel=1e-15)


@pytest.mark.parametrize("step", STEPS)
def test_reference_number_tendency_uses_raw_tracer(step):
    """The number tendency is (n/rho - pxtm1)/ztmst with the UNCLAMPED pxtm1 (F 1781, 3625, 3628) and the
    ccwmin repair zeroing the end-of-step tracer where the condensate is repaired
    (F 3641-3652): in number_clamp's clear cells the end tracer is 0 whatever the raw input.
    """
    inp, o, g = echam_in(), echam_out(step), echam_diag(step)
    j = column_names().index("number_clamp")
    end = echam_end_state(step)
    for fld, xtm1 in (("qni", "xtm1_icnc"), ("qnc", "xtm1_cdnc")):
        raw = inp[xtm1][:, j]
        clear = (inp["paclc"][:, j] == 0) & (raw != 0)
        assert clear.sum() == 2
        np.testing.assert_allclose(end[fld][clear, j], 0.0, atol=1e-15 * np.max(np.abs(raw)))
    k = int(np.argmax(inp["xtm1_icnc"][:, j] * (inp["paclc"][:, j] > 0)))   # cold cloudy cell
    assert g["icnc_add"][k, j] == float(load()["param/icemax"])          # capped at F 1252
    np.testing.assert_allclose(end["qni"][k, j], o["picnc"][k, j] / g["rho"][k, j], rtol=1e-13)


# ===========================================================================
# jcm building blocks on ECHAM's inputs (run on dev today)
# ===========================================================================
@pytest.mark.parametrize("prec", PRECISIONS)
def test_zrid_matches_echam(prec):
    """The radius zrid = max(1e-6, 0.9e-6*max(23.2*exp(0.015*min(T-tmelt,0)), 1)) at ptm1."""
    with echam_constants(), precision(prec):
        inp, g = echam_in(), echam_diag("dt1200")
        assert_close("zrid", "radius", run_jcm_zrid(inp["ptm1"], prec), g["rid"], prec)
        np.testing.assert_array_equal(echam_diag("dt720")["rid"], g["rid"])


@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_section1_wbf_criterion_matches_echam(step, prec):
    """lo2_2d (F 885) and its pieces zvervx (F 814-816), zrice (F 866-877), zvervmax
    (F 880-882) on the section-1 ice max(pxim1 + ztmst*pxite, 0) and ICNC.
    """
    with echam_constants(), precision(prec):
        inp, g = echam_in(), echam_diag(step)
        r = run_jcm_wbf_criterion(g["s1_xip1"], g["s1_picnc"], inp["paclc"], g["rho"], g["esw"],
                                  g["esi"], g["eta"], inp["ptkem1"], prec)
        assert_close("zvervx", "velocity", r["vervx"], g["vervx"], prec)
        assert_close("zrice", "radius", r["rice"], g["s1_rice"], prec)
        assert_close("zvervmax", "velocity", r["vervmax"], g["s1_vervmax"], prec)
        np.testing.assert_array_equal(r["crit"], g["lo2_2d"] > 0.5)


@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_section4_phase_criterion_matches_echam(step, prec):
    """lo2 (F 1295-1298): T < cthomi, or T < tmelt and 0.01*zvervx < zvervmax on the
    post-sedimentation ice and the ICNC after += znidetr (F 1251-1263).
    """
    with echam_constants(), precision(prec):
        inp, g = echam_in(), echam_diag(step)
        p = echam_params()
        r = run_jcm_wbf_criterion(g["sed_xip1"], g["icnc_s4"], inp["paclc"], g["rho"], g["esw"],
                                  g["esi"], g["eta"], inp["ptkem1"], prec)
        assert_close("zrice", "radius", r["rice"], g["s4_rice"], prec)
        assert_close("zvervmax", "velocity", r["vervmax"], g["s4_vervmax"], prec)
        t = inp["ptm1"]
        lo2 = (t < float(p.cthomi)) | ((t < float(p.tmelt)) & r["crit"])
        np.testing.assert_array_equal(lo2, g["lo2"] > 0.5)


@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_sedimentation_ice_matches_echam(step, prec):
    """sedimentation_ice (F 2152-2285) on ECHAM's section-4 input, which excludes the
    detrained condensate (ICE-1).
    """
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        r = run_jcm_sedimentation(g, ztmst(step), prec)
        assert_close("zxip1", "mass", r["sed_xip1"], g["sed_xip1"], prec)
        assert_close("picnc", "number", r["sed_icnc"], g["sed_icnc"], prec)
        assert_close("zxiflux", "mass_flux", r["sed_xiflux"], g["sed_xiflux"], prec)
        assert_close("zxifluxn", "number_flux", r["sed_xifluxn"], g["sed_xifluxn"], prec)


@pytest.mark.xfail(strict=True, reason=(
    "jcm's sedimentation_ice clamps the flux a level ABSORBS from above at 0 when it updates "
    "the falling-ice cover and the in-cloud sedimentation ledger; ECHAM passes the negative "
    "zxiflx_from_level to gridbox_frac_falling_hydrometeor (F 2275-2277, cover 1.0 instead of "
    "0.6 in sediment_then_detrain at 350 hPa) and keeps pmrateps negative (F 2264-2265, "
    "-1.5e-4 instead of 0). Pre-existing, not part of #941; mass, number and fluxes agree."))
@pytest.mark.parametrize("step", STEPS)
def test_sedimentation_falling_ice_cover_matches_echam(step):
    """The falling-ice cover zclcfi and the sedimentation ledger zmrateps (F 2264-2277)."""
    prec = "float64"
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        r = run_jcm_sedimentation(g, ztmst(step), prec)
        assert_close("zclcfi", "fraction", r["sed_clcfi"], g["sed_clcfi"], prec)
        assert_close("zmrateps", "mass", r["sed_mrateps"], g["sed_mrateps"], prec)


@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_update_in_cloud_water_icnc_diagnosis_at_zrid(step, prec):
    """update_in_cloud_water with prid = zrid (F 1511): the diagnosed ICNC
    0.75*rho*zxib/(pi*rhoice*zrid**3) where the cell holds ice at picnc <= icemin
    (F 2611-2624). ECHAM's diagnosis is uncapped; jcm caps it at icemax, which no
    column reaches (largest diagnosed value 3e5 m-3).
    """
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        assert np.max(g["icnc_cand"] * g["icnc_dmask"]) < 1e7
        r = run_jcm_update_in_cloud_water(g, ztmst(step), prec)
        assert_close("picnc", "number", r["uicw_icnc"], g["uicw_icnc"], prec)
        assert_close("zxib", "mass", r["uicw_xib"], g["uicw_xib"], prec)
        assert_close("zxlb", "mass", r["uicw_xlb"], g["uicw_xlb"], prec)
        assert_close("paclc", "fraction", r["uicw_aclc"], g["uicw_aclc"], prec)
        assert_close("zcdnc", "number", r["uicw_cdnc"], g["uicw_cdnc"], prec)
        # the diagnosis actually fired (and at ECHAM's radius) in the zrid column
        j = column_names().index("icnc_diagnosis_zrid")
        assert int(g["icnc_dmask"][:, j].sum()) == 4


# ===========================================================================
# the #941 core (skipped until it is merged)
# ===========================================================================
@needs_merged_core
@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_znidetr_matches_echam(step, prec):
    """The detrained-ice number znidetr (F 958-983): ECHAM's prefactor, the whole zxtec gated by ll_cv, zero at
    cf <= clc_min, floored at cqtmin.
    """
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        assert_close("znidetr", "number", run_jcm_znidetr(g, ztmst(step), prec), g["nidetr"], prec)


@needs_merged_core
@pytest.mark.parametrize("step", STEPS)
def test_number_tendencies_against_raw_tracer(step):
    """ICE-5 end to end: where ECHAM pins the end-of-step number (the ccwmin repair
    zeroes it in condensate-free cells, F 3641-3652), jcm's raw + dt*tendency must
    land on it for raw inputs that are negative or above icemax/rho. The warm liquid
    cloud's CDNC is excluded: jcm clips raw CDNC at 1e11 m-3 on entry
    (_cdnc_max_phys_per_m3), ECHAM carries 1e12 (no upper bound, F 600-601).
    """
    prec = "float64"
    with echam_constants(), precision(prec):
        inp, g, dt = echam_in(), echam_diag(step), ztmst(step)
        j = column_names().index("number_clamp")
        sl = {k: v[:, j:j + 1] for k, v in inp.items() if getattr(v, "ndim", 0) == 2}
        gl = {k: v[:, j:j + 1] for k, v in g.items()}
        jc = run_jcm_column(sl, gl, dt, prec)
        ref = {k: v[:, j:j + 1] for k, v in echam_end_state(step).items()}
        pinned_i = (g["end_xip1"][:, j] < float(load()["param/ccwmin"]))
        pinned_c = (g["end_xlp1"][:, j] < float(load()["param/ccwmin"]))
        assert pinned_i.sum() >= 3 and pinned_c.sum() >= 3
        for fld, pin in (("qni", pinned_i), ("qnc", pinned_c)):
            raw = np.abs(sl["xtm1_icnc" if fld == "qni" else "xtm1_cdnc"][:, 0])
            tol = 1e-12 * np.maximum(raw, 1.0) + 1e-6
            err = np.abs(jc[fld][pin, 0] - ref[fld][pin, 0])
            assert np.all(err <= tol[pin]), (fld, jc[fld][pin, 0], ref[fld][pin, 0])


@needs_merged_core
@pytest.mark.parametrize("step", STEPS)
def test_clear_cell_detrainment_end_state(step):
    """ICE-4 end to end in clear cells (cf = 0), where nothing but the detrained
    condensate acts: it is split by lo2, all of it evaporates (zxlevap/zxievap,
    F 1399-1402) and ECHAM corrects ptte by -(als-alv)*zxtec/cpd where ztconv <= tmelt
    and lo2 is false (F 1300-1307). jcm applies the same correction with its moist cp
    (and in both directions; the reverse case does not occur here), so ECHAM's
    reference temperature is shifted by fix*(cpd/cp_moist - 1), the one documented
    deviation (a 1e-4 relative effect), before the comparison.
    """
    prec = "float64"
    with echam_constants(), precision(prec):
        inp, g, dt = echam_in(), echam_diag(step), ztmst(step)
        ref = echam_end_state(step)
        cols = [column_names().index(n) for n in
                ("detrainment_cold", "detrainment_mixed_lo2_false", "detrainment_warm")]
        for j in cols:
            sl = {k: v[:, j:j + 1] for k, v in inp.items() if getattr(v, "ndim", 0) == 2}
            gl = {k: v[:, j:j + 1] for k, v in g.items()}
            jc = run_jcm_column(sl, gl, dt, prec)
            ks = np.where((inp["paclc"][:, j] == 0) & (inp["zxtec"][:, j] > 0))[0]
            assert ks.size == 1
            k = int(ks[0])
            fix = dt * (g["ptte_fix"][k, j] - g["ptte_pre"][k, j])          # ECHAM, /cpd
            cp_moist = c.cpd + (c.cpv - c.cpd) * max(inp["pqm1"][k, j], 0.0)
            t_ref = ref["T"][k, j] + fix * (c.cpd / cp_moist - 1.0)
            assert abs(jc["T"][k, 0] - t_ref) <= 1e-9 + 1e-9 * abs(t_ref - inp["ptm1"][k, j])
            for fld in ("q", "qc", "qi"):
                sc = dt * inp["zxtec"][k, j]
                assert abs(jc[fld][k, 0] - ref[fld][k, j]) <= 1e-9 * sc, (fld, jc[fld][k, 0], ref[fld][k, j])


@needs_merged_core
@pytest.mark.parametrize("step", STEPS)
def test_detrained_ice_end_to_end(step):
    """ICE-1 + ICE-3 wiring, end to end, in the cloudy cells that receive detrained
    condensate as ice (lo2 true): the end-of-step ICNC (znidetr added after
    sedimentation and capped at icemax, F 1251-1252, then ECHAM's own sinks) and, below
    cthomi, the ice mass that stays in the cell (the detrained ice is not sedimented this
    step, F 1227-1236). jcm without #941 misses these by factors of 10 to 1e4 (ICNC
    ~1e3-1e5 m-3 against ECHAM's ~1e7; 5 % of the ice kept).

    These are WIRING checks, not the exact comparison (that is test_znidetr_matches_echam
    and the block tests above): the end state also passes through the section-5
    deposition and the aggregation number sink, which depend on the saturation formula
    (jcm's Tetens differs from ECHAM's Sonntag tables by 1-8 % below 273 K). Measured
    against the #941 core: ICNC within 3.4e-3, cold-cell ice within 7.4e-4; the
    tolerances are 1e-2 and 5e-3.
    """
    prec = "float64"
    with echam_constants(), precision(prec):
        inp, g, dt = echam_in(), echam_diag(step), ztmst(step)
        ref = echam_end_state(step)
        jc = run_jcm_column(inp, g, dt, prec)
        cells = (inp["paclc"] > 0.01) & (inp["zxtec"] > 0) & (g["lo2"] > 0.5)
        assert cells.sum() == 7
        n_jcm, n_ref = jc["qni"] * g["rho"], ref["qni"] * g["rho"]
        rel = np.abs(n_jcm - n_ref)[cells] / n_ref[cells]
        assert np.all(rel <= 1e-2), (rel, n_jcm[cells], n_ref[cells])
        cold = cells & (inp["ptm1"] < float(load()["param/cthomi"]))
        assert cold.sum() == 4
        rel = np.abs(jc["qi"] - ref["qi"])[cold] / ref["qi"][cold]
        assert np.all(rel <= 5e-3), (rel, jc["qi"][cold], ref["qi"][cold])


# ===========================================================================
# parameters
# ===========================================================================
def test_cloud_params_relevant_to_941_match_echam():
    """The CloudParams2M DEFAULTS equal ECHAM's (mo_cloud_utils.f90,
    mo_echam_cloud_params.f90 at T63) for every constant the #941 pieces read, and
    differ exactly by KNOWN_PARAM_DIFFERENCES elsewhere.
    """
    from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
    with echam_constants(), precision("float64"):
        p = CloudParams2M.default()
    z = load()
    for name in ("cqtmin", "cthomi", "ceffmin", "ceffmax", "crhoi", "cvtfall", "ccwmin",
                 "crhosno", "cn0s", "icemin", "icemax", "conv_effr2mvr", "clc_min", "fact_PK",
                 "pow_PK", "rhoice", "fact_tke", "epsec"):
        assert float(getattr(p, name)) == pytest.approx(float(z[f"param/{name}"]), rel=1e-12), name
    for name, (jcm_v, echam_v) in KNOWN_PARAM_DIFFERENCES.items():
        assert float(getattr(p, name)) == pytest.approx(jcm_v, rel=1e-12), name
        assert float(z[f"param/{name}"]) == pytest.approx(echam_v, rel=1e-12), name


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
