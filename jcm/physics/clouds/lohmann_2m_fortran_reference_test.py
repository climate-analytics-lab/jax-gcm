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
* ``znidetr`` (F 958-983).
* ``precip_formation_warm``/``_cold`` (F 1662-1783, #1087): the per-level
  rain/riming/snow formation mass (``zrpr``/``zsacl``/``zspr``) on ECHAM's own
  section-6-exit state, chained exactly as ``scheme.py`` sequences the two
  calls, plus the end-of-section-7 in-cloud condensate and ICNC/CDNC.
* End to end, through ``cloud_microphysics_2m`` with ECHAM's anchor,
  increments and ``detrained_qc``/``detrained_qi``: the number tendencies against the RAW tracer
  (F 1781, 3625-3652) where ECHAM pins the end-of-step number, and the end
  state of clear cells that receive detrained condensate (the section-4
  split and the ``ptte`` fix, F 1300-1317).

Nothing here skips: a renamed jcm helper or argument is a FAILURE of the
adapter that calls it, which must then be updated.

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


# jcm tunables that differ from ECHAM6.3-HAM2.3 at T63 (the maintainer's tuning, kept in
# jcm). The comparisons use ECHAM's values so the formulation is tested, not the tuning;
# ``test_cloud_params_relevant_to_941_match_echam`` pins the differences.
KNOWN_PARAM_DIFFERENCES = {"ccraut": (10.6, 15.0)}


def echam_params():
    """CloudParams2M for the comparisons: jcm's defaults built under ECHAM's constants
    (tmelt, grav, cthomi follow them), ECHAM's autoconversion rate ``ccraut``
    (KNOWN_PARAM_DIFFERENCES), and ``activation_smoothing = 0``
    (jcm's smooth max on the activation increment; ECHAM's is a hard MAX, F 2599-2600).

    ``n_aer_coarse = 0``: this reference's ECHAM build has no dust/BC at all
    (``cloud2m_README.md``'s "Aerosol-free" configuration), so its own
    heterogeneous mixed-phase freezing is a hard no-op everywhere (verified:
    ``diag/<step>/frl`` and ``frl_hom`` are exactly 0 on every column, level
    and step of the fixture). jcm's aerosol-free closure is a deliberate,
    documented SUBSTITUTE for ECHAM's unimplemented ``lccnclim`` mode
    (scheme.py's ``het_mxphase_freezing`` else-branch, #705) with its own
    background coarse-aerosol default (``n_aer_coarse = 0.5`` /cm3) — a
    tuning choice, not part of the formulation this harness tests, so it is
    zeroed here exactly like ``ccraut`` above. Earlier revisions of this
    docstring claimed "no compared cell holds any [supercooled liquid]" for
    the DeMott substitute; that was wrong — jax-gcm#1087's
    ``detrainment_mixed_lo2_false`` column holds supercooled liquid exactly
    at the CDNC activation floor (238-264 K is DeMott's active range), and
    with the default ``n_aer_coarse`` the substitute nudged that cell's CDNC
    a few hundred m^-3 below the floor, flipping the downstream warm-rain
    gate off where ECHAM's (truly aerosol-free) run keeps it on.
    """
    from jcm.physics.clouds.lohmann_2m_params import CloudParams2M
    return CloudParams2M.default(
        activation_smoothing=0.0,
        n_aer_coarse=0.0,
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
    from jcm.physics.clouds.cloud_utils import ice_volume_mean_radius_from_temperature
    return np.asarray(ice_volume_mean_radius_from_temperature(_j(t_m1, prec), echam_params()))


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


def _entering_from_above(x: np.ndarray) -> np.ndarray:
    """Shift a (nlev, ncol) LEAVING-this-level array to ENTERING-this-level.

    The fixture's vertical ordering is TOP-FIRST (level 0 = model top): what
    enters level ``k`` is what left level ``k-1``; nothing enters the model
    top, so row 0 is 0.
    """
    out = np.zeros_like(x)
    out[1:] = x[:-1]
    return out


def run_jcm_precip_formation(g, dt, prec):
    """ECHAM section 7.1 (F 1662-1683) then 7.2 (F 1686-1783) on ECHAM's own
    section-6-exit per-level state, chained exactly as ``scheme.py`` sequences
    them: ``precip_formation_cold``'s CDNC/in-cloud-liquid INOUT is
    ``precip_formation_warm``'s OUTPUT, matching the Fortran's single
    ``pcdnc``/``pxlb`` array threaded through both calls (F 1680-1681,
    1780-1783).

    Inputs are ``xlb_6``/``xib_6``/``icnc_6``/``uicw_cdnc`` -- confirmed
    IDENTICAL on every column, level and step of this fixture (sections
    6.1/6.2/WBF are a hard no-op here: ``diag/<step>/frl`` and ``frl_hom``
    are exactly 0 everywhere, so nothing moves mass or number between the
    5.5 state and section 7's). The rain/snow mixing ratio "from above"
    (``zxrp1``/``zxsp1``, F 1636-1654) is recovered from the ENTERING
    ``clcpre``/``rsfl_lev``/``ssfl_lev`` -- the level above's LEAVING values,
    0 at the model top -- through :func:`rain_mixing_ratio_from_flux` /
    :func:`snow_mixing_ratio_from_flux`, the same helpers ``scheme.py`` itself
    calls (jax-gcm#1087 extracted them from what was inline arithmetic so this
    test cannot drift from what jcm actually runs).
    """
    from jcm.physics.clouds.cloud_utils import (
        rain_mixing_ratio_from_flux,
        snow_mixing_ratio_from_flux,
    )
    from jcm.physics.clouds.lohmann_2m.precip import precip_formation_cold, precip_formation_warm
    p = echam_params()

    cloud_mask = jnp.asarray(g["ll_cc"] > 0.5)
    paclc = _j(g["uicw_aclc"], prec)
    rho = _j(g["rho"], prec)
    dz = _j(g["dz"], prec)
    cdnc_min = _j(g["cdnc_min_in"], prec)
    cdnc0 = _j(g["uicw_cdnc"], prec)
    xlb0 = _j(g["xlb_6"], prec)
    xib0 = _j(g["xib_6"], prec)
    icnc0 = _j(g["icnc_6"], prec)
    temp = _j(g["tp1tmp"], prec)
    visc = _j(g["viscos"], prec)
    mrateps_seed = _j(g["sed_mrateps"], prec)

    precip_cover_in = _j(_entering_from_above(g["clcpre"]), prec)
    rain_flux_in = _j(_entering_from_above(g["rsfl_lev"]), prec)
    snow_flux_in = _j(_entering_from_above(g["ssfl_lev"]), prec)

    zqrho = 1.3 / rho
    inv_rho = 1.0 / rho
    zxrp1 = rain_mixing_ratio_from_flux(rain_flux_in, precip_cover_in, zqrho, p)
    zxsp1 = snow_mixing_ratio_from_flux(snow_flux_in, precip_cover_in, p)
    zauloc = jnp.clip(p.cauloc / 5000.0 * dz, p.clmin, p.clmax)
    zclcstar = jnp.minimum(paclc, precip_cover_in)

    ll_warm = cloud_mask & (xlb0 > p.cqtmin) & (cdnc0 >= cdnc_min)
    (cdnc_p, xlb_p, _auto_ic, zrpr, _rprn, _auto_only,
     _accr_only) = precip_formation_warm(
        ll_warm, zauloc, paclc, zclcstar, rho, zxrp1, cdnc_min, cdnc0, xlb0, dt, p)

    (icnc_c, cdnc_c, _mrateps_c, xib_c, xlb_c, _psprn, zsacl,
     _psacln, _pmsnowacl, zspr) = precip_formation_cold(
        cloud_mask, zauloc, paclc, zclcstar, zqrho, inv_rho, temp, visc,
        zxsp1, rho, cdnc_min, icnc0, cdnc_p, mrateps_seed, xib0, xlb_p, dt, p)

    return dict(rpr=np.asarray(zrpr), sacl=np.asarray(zsacl), spr=np.asarray(zspr),
                xlb_7=np.asarray(xlb_c), xib_7=np.asarray(xib_c),
                icnc_7=np.asarray(icnc_c), cdnc_7=np.asarray(cdnc_c))


def run_jcm_znidetr(g, dt, prec):
    """ECHAM znidetr [1/m3] (F 958-983) from jcm's helper, on ECHAM's zxtec,
    ptm1, lo2_2d, paclc, rho and zrid.
    """
    from jcm.physics.clouds.cloud_utils import detrained_ice_crystal_number
    inp = echam_in()
    return np.asarray(detrained_ice_crystal_number(
        _j(dt * g["xtec"], prec), _j(inp["ptm1"], prec), jnp.asarray(g["lo2_2d"] > 0.5),
        _j(inp["paclc"], prec), _j(g["rho"], prec), _j(g["rid"], prec), echam_params()))


def run_jcm_column(inp: dict, g: dict, dt: float, prec: str) -> dict:
    """ECHAM cloud_micro_interface -> jcm cloud_microphysics_2m, column by column.

    Mapping: the anchor = ECHAM's m1 fields (ptm1, pqm1, pxlm1, pxim1 and the
    number tracers' pxtm1, per kg of air); the increments = ztmst times ECHAM's
    accumulated tendencies (ptte, pqte, pxlte, pxite, pxtte); the detrained
    condensate ztmst*zxtec split into liquid and ice at ztconv > tmelt, exactly
    as cudtdq split it (mo_cufluxdts.f90:645-664, pten = the convection
    temperature ztconv), as ``detrained_qc``/``detrained_qi``; air density =
    ECHAM's papm1/(rd*ptvm1); layer thickness dp/(rho*grav) so jcm's rho*g*dz is
    ECHAM's dp; TKE = ptkem1; no aerosol (activated_cdnc = ice_nuclei = 0).
    jcm's tendencies are relative to the provisional state anchor + increment
    (+ detrainment for the condensate). Returns the end-of-step state
    X + dt*dX/dt.
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

    def one(t1, q1, qc1, qi1, qnc1, qni1, cf, rho, dz_, tke, pr,
            dT, dq, dqc_, dqi_, dqnc, dqni, dqc, dqi):
        zero = jnp.zeros_like(t1)
        out = cloud_microphysics_2m(
            t1, q1, pr, qc1, qi1, qnc1, qni1, cf, rho, dz_, tke, zero, zero, zero,
            jnp.asarray(dt, t1.dtype), p,
            temperature_increment=dT, humidity_increment=dq,
            qc_increment=dqc_, qi_increment=dqi_,
            qnc_increment=dqnc, qni_increment=dqni,
            detrained_qc=dqc, detrained_qi=dqi)
        return out[0]

    args = [inp["ptm1"], inp["pqm1"], inp["pxlm1"], inp["pxim1"],
            inp["xtm1_cdnc"], inp["xtm1_icnc"],
            inp["paclc"], g["rho"], dz, inp["ptkem1"], inp["papm1"],
            dt * inp["ptte"], dt * inp["pqte"], dt * inp["pxlte"], dt * inp["pxite"],
            dt * inp["xtte_cdnc"], dt * inp["xtte_icnc"], det_qc, det_qi]
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


@pytest.mark.parametrize("step", STEPS)
def test_sedimentation_falling_ice_cover_matches_echam(step):
    """The falling-ice cover zclcfi and the sedimentation ledger zmrateps (F 2264-2277).

    Includes the absorbing level of sediment_then_detrain at 350 hPa, where the
    flux from the level is negative: ECHAM passes it signed to
    gridbox_frac_falling_hydrometeor (cover 1.0) and keeps pmrateps negative.

    One deviation remains, in jcm's gridbox_frac_falling_hydrometeor itself:
    it reports no cover where the total falling flux is below 1e-9 kg/m2/s
    (ECHAM: cqtmin = 1e-12), a guard against the 1/flux**2 reverse-mode
    amplifier of the cover weighting (cloud_utils). The cover is compared
    where the falling-ice flux leaving the level reaches that threshold, and
    below it jcm must report exactly no cover (icnc_diagnosis_zrid level 28:
    1.9e-10 / 5.4e-11 kg/m2/s, cover 1.0 in ECHAM).
    """
    prec = "float64"
    jcm_min_cover_flux = 1.0e-9     # cloud_utils.gridbox_frac_falling_hydrometeor
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        r = run_jcm_sedimentation(g, ztmst(step), prec)
        resolved = g["sed_xiflux"] > jcm_min_cover_flux
        assert_close("zclcfi", "fraction", np.where(resolved, r["sed_clcfi"], 0.0),
                     np.where(resolved, g["sed_clcfi"], 0.0), prec)
        assert np.all(r["sed_clcfi"][~resolved] == 0.0)
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
# section 7: precipitation formation (#1087)
# ===========================================================================
@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_precip_formation_matches_echam(step, prec):
    """``precip_formation_warm``/``_cold`` (F 1662-1783): the per-level rain
    (``zrpr``), riming (``zsacl``) and snow (``zspr``) formation mass that
    feeds ``update_precip_fluxes`` -- the two functions #1087 found had no
    direct comparison against the compiled reference, on every column and
    level of both time steps.

    Before jax-gcm#1087's ``echam_params()``/``demott2010_inp`` fix, the
    ``detrainment_mixed_lo2_false`` column's level 35 cell (supercooled
    liquid exactly at the CDNC activation floor) failed outright: jcm's
    aerosol-free closure used a nonzero background INP by default even
    where ECHAM's own (truly aerosol-free) run has none, nudging CDNC a
    few hundred m^-3 below the floor and turning off the whole level's
    rain formation (ECHAM's ``rpr`` there is 1.375e-7; jcm's was 0). With
    ``n_aer_coarse`` zeroed to match this reference's actual aerosol-free
    build, this level (and every other) matches to float64 round-off --
    that gap was never in ``precip_formation_warm`` itself.
    """
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        r = run_jcm_precip_formation(g, ztmst(step), prec)
        assert_close("rpr", "mass", r["rpr"], g["rpr"], prec)
        assert_close("sacl", "mass", r["sacl"], g["sacl"], prec)
        assert_close("spr", "mass", r["spr"], g["spr"], prec)
        # the cell #1087 named explicitly: rain forms where ECHAM's does.
        j = column_names().index("detrainment_mixed_lo2_false")
        assert r["rpr"][35, j] > 0.0
        assert g["rpr"][35, j] > 0.0


@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_precip_formation_end_of_section7_state_matches_echam(step, prec):
    """Fixture fields section 7 carries but no existing test reads (#1087
    point 3): the in-cloud ice/liquid and ICNC/CDNC that ``precip_formation_
    cold`` leaves behind (``xib_7``/``xlb_7``/``icnc_7``, and the grid-level
    ``out/<step>/picnc``/``pacdnc`` -- CDNC/ICNC are never split in-cloud vs
    grid-mean the way condensate is, so the SAME array is section 7's final
    ``picnc``/``pcdnc``).
    """
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        o = echam_out(step)
        r = run_jcm_precip_formation(g, ztmst(step), prec)
        assert_close("xib_7", "mass", r["xib_7"], g["xib_7"], prec)
        assert_close("xlb_7", "mass", r["xlb_7"], g["xlb_7"], prec)
        assert_close("icnc_7", "number", r["icnc_7"], g["icnc_7"], prec)
        assert_close("picnc", "number", r["icnc_7"], o["picnc"], prec)
        assert_close("pacdnc", "number", r["cdnc_7"], o["pacdnc"], prec)


def test_zauloc_constants_match_echam_literal():
    """``zauloc`` (F 1618-1619): ``mo_cloud_micro_2m.f90`` hardcodes ``3./
    5000.`` inline rather than reading the module-level ``cauloc`` parameter
    (which is 0, a DIFFERENT 1-moment-scheme constant mo_cloud.f90 reads --
    checking jcm's ``cauloc`` against it would be wrong). ``clmax``/``clmin``
    (0.5/0.0) DO come from ``mo_echam_cloud_params.f90:60-61`` and apply here.
    None of the three are in this fixture's ``param/*`` dump (not a `cloud_
    micro_interface` argument or local the harness patches), so -- like
    ``ECHAM_CONSTANTS`` above -- they are pinned here as literals read
    directly from the reference Fortran, not from compiled output.
    """
    with echam_constants(), precision("float64"):
        p = echam_params()
    assert float(p.cauloc) == pytest.approx(3.0, rel=1e-12)
    assert float(p.clmin) == pytest.approx(0.0, abs=0.0)
    assert float(p.clmax) == pytest.approx(0.5, rel=1e-12)


# ===========================================================================
# the #941 core
# ===========================================================================
@pytest.mark.parametrize("prec", PRECISIONS)
@pytest.mark.parametrize("step", STEPS)
def test_znidetr_matches_echam(step, prec):
    """The detrained-ice number znidetr (F 958-983): ECHAM's prefactor, the whole zxtec gated by ll_cv, zero at
    cf <= clc_min, floored at cqtmin.
    """
    with echam_constants(), precision(prec):
        g = echam_diag(step)
        assert_close("znidetr", "number", run_jcm_znidetr(g, ztmst(step), prec), g["nidetr"], prec)


@pytest.mark.parametrize("step", STEPS)
def test_number_tendencies_against_raw_tracer(step):
    """ICE-5 end to end: where ECHAM pins the end-of-step number (the ccwmin repair
    zeroes it in condensate-free cells, F 3641-3652), jcm's raw + dt*tendency must
    land on it for raw inputs that are negative or above icemax/rho.

    jcm, like ECHAM, floors the working numbers at cqtmin only (F 600-605), caps ICNC
    at icemax only after the detrained number joins it (F 1252), and never caps CDNC.
    So the warm liquid cloud's raw 1e12 m-3 survives the step in both; its end number
    is not pinned (the cloud keeps its water) and differs from ECHAM's only through the
    autoconversion number sink (~2e3 m-3, which depends on the saturation formula), so
    it is compared at 1e-9 relative.
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
        warm = ~pinned_c & (sl["xtm1_cdnc"][:, 0] * g["rho"][:, j] > 1e11)
        assert warm.sum() == 1
        np.testing.assert_allclose(jc["qnc"][warm, 0], ref["qnc"][warm, 0], rtol=1e-9)


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


# A level counts as touched by the ice-sourcing chain if it detrains condensate
# itself OR receives a physically meaningful falling-ice flux from above; the
# same floor as test_sedimentation_falling_ice_cover_matches_echam's
# jcm_min_cover_flux, well above the ~1e-18 kg/m2/s EPSILON(1d0)-floor noise
# every quiescent column's sedimentation input carries (F 1228) and far below
# any real flux in the fixture (>= 1e-5 kg/m2/s).
_MEANINGFUL_FLUX = 1.0e-9


@pytest.mark.parametrize("step", STEPS)
def test_detrained_ice_end_to_end(step):
    """ICE-1 + ICE-3 wiring, end to end: the end-of-step ICNC (znidetr added
    after sedimentation and capped at icemax, F 1251-1252, then ECHAM's own
    sinks) and, below cthomi, the ice mass that stays in the cell (the
    detrained ice is not sedimented this step, F 1227-1236), in every cloudy
    ice cell (lo2 true) touched by the chain: cells that detrain condensate
    themselves, AND — #1110 — the cell(s) that only *receive* sedimented ice
    from a detraining level above (``sediment_then_detrain`` level 29, no
    detrainment of its own). jcm without #941 misses the detraining cells by
    factors of 10 to 1e4 (ICNC ~1e3-1e5 m-3 against ECHAM's ~1e7; 5 % of the
    ice kept).

    These are WIRING checks, not the exact comparison (that is
    test_znidetr_matches_echam and the block tests above): the end state also
    passes through the section-5 deposition, section 6 freezing/WBF and the
    aggregation number sink, which this module does not compare block by
    block.

    #972/#1110 traced the residual this test measured (ICNC within 4.1e-3,
    cold-cell ice within 9.2e-5, against the fixture before this fix) to the
    column sweep's 3.2/3.3 sublimation call reading the just-sedimented
    OUTGOING flux of its own level instead of the flux ENTERING it: a level
    with no incoming ice (every level that is the first to sediment any, the
    ordinary case) spuriously sublimated part of the ice it had itself only
    just created. ``cloud_microphysics_2m`` now runs sublimation before
    sedimentation on the incoming carry (see the comment there). With that
    fixed, ``icnc_add`` (ECHAM's end of section 4, before sections 5-7) is
    exact to float64 round-off at every level of every fixture column,
    including the three stacked detraining+sedimenting levels of
    `detrainment_mixed_lo2_true` that previously grew a compounding 4.6e-4 ->
    1.75e-3 -> 3.6e-3 ICNC residual down the column.

    The residual THIS test still measures (ICNC within 6.6e-3, cold-cell ice
    within 8.4e-5) is therefore entirely downstream of section 4, in sections
    5-7 this module does not block-test. Its single largest contributor
    (`detrainment_mixed_lo2_true` level 37) is explained, not open: that
    level's convective detrainment was counted as liquid by convection
    (``ztconv`` 274 K > tmelt) but reassigned to ice here by ``lo2``, and this
    scheme's re-split (1301-1317) corrects the latent-heat mismatch in BOTH
    directions, including the one ECHAM's own ``ztconv <= tmelt`` gate (F
    1302-1303) leaves uncorrected — a deliberate, already-decided deviation
    (see the "Re-split of the detrained condensate" comment in scheme.py) that
    trades an exact match against ECHAM for an exact column enthalpy budget,
    and shows up here as a small temperature (hence saturation/nucleation)
    difference feeding sections 5-7. Not the saturation formula either: jcm's
    is the Sonntag (1990) fit ECHAM's tables hold, and evaluating it on the
    tables' 1 mK knots (NINT(1000*T), F 3932-3933) with their
    forward-difference slope changes neither number by more than 0.2 %.

    #1116 (the same reorder) also fixed a second, independent defect this
    fixture cannot exercise: a warm level (T_m1 > tmelt) sedimenting its own
    in-cloud ice before melting it. No column here holds both T_m1 > tmelt
    and incoming falling ice at the same level, so that fix changes none of
    the numbers checked above -- see
    ``TestWarmLevelMeltsBeforeSedimenting2M`` in ``lohmann_2m_test.py`` for
    its fixture-free, differential check instead.
    """
    prec = "float64"
    with echam_constants(), precision(prec):
        inp, g, dt = echam_in(), echam_diag(step), ztmst(step)
        ref = echam_end_state(step)
        jc = run_jcm_column(inp, g, dt, prec)
        cells = (inp["paclc"] > 0.01) & (g["lo2"] > 0.5) & (
            (inp["zxtec"] > 0) | (g["sed_xiflux_in"] > _MEANINGFUL_FLUX))
        assert cells.sum() == 8
        n_jcm, n_ref = jc["qni"] * g["rho"], ref["qni"] * g["rho"]
        rel = np.abs(n_jcm - n_ref)[cells] / n_ref[cells]
        assert np.all(rel <= 7e-3), (rel, n_jcm[cells], n_ref[cells])
        cold = cells & (inp["ptm1"] < float(load()["param/cthomi"]))
        assert cold.sum() == 5
        rel = np.abs(jc["qi"] - ref["qi"])[cold] / ref["qi"][cold]
        assert np.all(rel <= 1e-4), (rel, jc["qi"][cold], ref["qi"][cold])


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
                 "pow_PK", "rhoice", "fact_tke", "epsec", "ccsaut"):
        assert float(getattr(p, name)) == pytest.approx(float(z[f"param/{name}"]), rel=1e-12), name
    for name, (jcm_v, echam_v) in KNOWN_PARAM_DIFFERENCES.items():
        assert float(getattr(p, name)) == pytest.approx(jcm_v, rel=1e-12), name
        assert float(z[f"param/{name}"]) == pytest.approx(echam_v, rel=1e-12), name


# ===========================================================================
# precip_cover (pclc / zclcpre) -- jax-gcm#1017
# ===========================================================================
def run_jcm_precip_cover(inp: dict, g: dict, dt: float, prec: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the per-level ``precip_cover`` the 2M scheme would publish via
    ``configure_wetdep_hydro_diagnostics``, end to end from ECHAM's raw step
    inputs (same adapter as ``run_jcm_column``, just different elements of
    :func:`cloud_microphysics_2m`'s return tuple -- the POST-update
    precipitating-area fraction, ECHAM's ``zclcpre_2d``
    (mo_cloud_micro_2m.f90:1742), compared against the reference's own
    ``diag/<step>/clcpre``). Also returns ``rain_flux_profile +
    snow_flux_profile``. That is not the precipitation flux the recurrence
    sees: ``snow_flux_profile`` is the COSP-facing frozen flux, which adds the
    sedimenting ice at interior levels.
    """
    from jcm.physics.clouds.lohmann_2m.scheme import cloud_microphysics_2m
    p = echam_params()
    xt = dt * inp["zxtec"]
    ice = inp["ztconv"] <= c.tmelt
    det_qi, det_qc = np.where(ice, xt, 0.0), np.where(ice, 0.0, xt)
    dz = g["dp"] / (g["rho"] * c.grav)

    def one(t1, q1, qc1, qi1, qnc1, qni1, cf, rho, dz_, tke, pr,
            dT, dq, dqc_, dqi_, dqnc, dqni, dqc, dqi):
        zero = jnp.zeros_like(t1)
        out = cloud_microphysics_2m(
            t1, q1, pr, qc1, qi1, qnc1, qni1, cf, rho, dz_, tke, zero, zero, zero,
            jnp.asarray(dt, t1.dtype), p,
            temperature_increment=dT, humidity_increment=dq,
            qc_increment=dqc_, qi_increment=dqi_,
            qnc_increment=dqnc, qni_increment=dqni,
            detrained_qc=dqc, detrained_qi=dqi)
        # The tuple ends precip_cover, pfrain, pfsnow, rain_flux, snow_flux
        # (see the return statement of cloud_microphysics_2m).
        return out[-5], out[-2] + out[-1]

    args = [inp["ptm1"], inp["pqm1"], inp["pxlm1"], inp["pxim1"],
            inp["xtm1_cdnc"], inp["xtm1_icnc"],
            inp["paclc"], g["rho"], dz, inp["ptkem1"], inp["papm1"],
            dt * inp["ptte"], dt * inp["pqte"], dt * inp["pxlte"], dt * inp["pxite"],
            dt * inp["xtte_cdnc"], dt * inp["xtte_icnc"], det_qc, det_qi]
    cover, flux = jax.vmap(one, in_axes=1, out_axes=1)(*[_j(a, prec) for a in args])
    return np.asarray(cover), np.asarray(flux)


def echam_precip_flux_into_cover(g: dict, dt: float) -> np.ndarray:
    """ECHAM's total precipitation flux in ``gridbox_frac_falling_hydrometeor``.

    The flux arriving from above (the level above's ``rsfl``/``ssfl``) plus
    the rain and snow formed at the level, ``(rpr + spr + sacl)·dp/(dt·g)``
    (``update_precip_fluxes``' ``zpretot + zpredel``) [kg/m2/s].
    """
    above = np.zeros_like(g["rsfl_lev"])
    above[1:] = g["rsfl_lev"][:-1] + g["ssfl_lev"][:-1]
    formed = (g["rpr"] + g["spr"] + g["sacl"]) * g["dp"] / (dt * c.grav)
    return above + formed


#: jcm's precip-cover flux floor (``cloud_utils.gridbox_frac_falling_hydrometeor``,
#: a documented differentiability choice) and ECHAM's ``cqtmin``.
_COVER_FLUX_FLOOR, _CQTMIN = 1.0e-9, 1.0e-12


def _ice_sourcing_residual_mask() -> np.ndarray:
    """Cells left out pending jax-gcm#1110, the #941 ice-sourcing residual.

    ``sediment_then_detrain`` levels 29-31 (~0.2 % ``clcpre``, ~3.5 % ``spr``)
    and ``detrainment_cold`` levels 32-33 (~2e-6 relative, float64 only): both
    are downstream of a small (~1.8 %) ICNC/ice-mass residual the existing
    ``zrid``/``znidetr``/``sedimentation_ice`` chain (#941) leaves on a level
    that only RECEIVES falling ice (no detrainment of its own, so outside
    ``test_detrained_ice_end_to_end``'s own ``cells`` mask). Confirmed
    upstream of THIS routine: #1087's own direct test of
    ``precip_formation_warm``/``_cold``, fed ECHAM's exact section-6-exit
    state, matches every column, level and step to float64 round-off, so
    nothing here is a defect in rain/snow formation itself.
    """
    mask = np.zeros((47, len(column_names())), dtype=bool)
    mask[29:32, column_names().index("sediment_then_detrain")] = True
    mask[32:34, column_names().index("detrainment_cold")] = True
    return mask


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("prec", PRECISIONS)
def test_precip_cover_matches_echam_clcpre(step, prec):
    """``precip_cover`` (jax-gcm#1017's HAM-below-cloud ``pclc`` input)
    against ECHAM's own ``clcpre`` diagnostic -- the ONLY input that slice
    could not otherwise verify against a compiled number. The reference
    DOES cover ``pclcpre`` (``diag/<step>/clcpre``, confirmed here).

    Two bands are left out, both already-documented, already-tracked
    deviations rather than failures of this recurrence:

    * cells whose ECHAM flux lies between ``cqtmin`` and jcm's 1e-9 floor,
      where jcm reads no cover by its documented differentiability choice
      (``icnc_diagnosis_zrid``'s ~6e-11 kg/m2/s snow tail is the case);
    * :func:`_ice_sourcing_residual_mask`'s cells, jax-gcm#1110.

    ``detrainment_mixed_lo2_false`` (jax-gcm#1087's own named cell -- jcm
    formed no rain at level 35 where ECHAM forms ``rpr`` ~1.4e-7 kg/kg from
    detrained supercooled liquid with CDNC exactly on the activation floor,
    because the aerosol-free closure's background INP was nonzero even for
    an explicitly aerosol-free comparison) is asserted directly: it is NOT
    masked, so a regression here fails loudly.
    """
    with echam_constants(), precision(prec):
        inp, g, dt = echam_in(), echam_diag(step), ztmst(step)
        got, _flux = run_jcm_precip_cover(inp, g, dt, prec)
        flux = echam_precip_flux_into_cover(g, dt)
    floor_band = (flux > _CQTMIN) & (flux <= _COVER_FLUX_FLOOR)
    excluded = floor_band | _ice_sourcing_residual_mask()
    assert_close("precip_cover", "fraction",
                 np.where(excluded, g["clcpre"], got), g["clcpre"], prec)
    j = column_names().index("detrainment_mixed_lo2_false")
    assert got[35, j] == pytest.approx(float(g["clcpre"][35, j]), rel=RTOL[prec])


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
