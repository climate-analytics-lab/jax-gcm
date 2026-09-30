"""jcm's cover and 1M cloud scheme against the ECHAM6.3 Fortran, number for number.

The reference data in ``jcm/data/test/echam_cloud_reference/`` are the inputs,
outputs and intermediates of the UNMODIFIED ECHAM6.3 r7492 routines
``mo_cover.f90::cover`` and ``mo_cloud.f90::cloud`` (the 1M branch), run on
designed and sampled columns by the standalone harness (see that directory's
README and provenance.json). This module feeds jcm the same inputs and compares
every output, per column. Each synthetic column isolates one process of
mo_cloud.f90 / mo_cover.f90, so a failing column names the process.

What is compared
----------------
* ``cover``: ``paclc``.
* ``cloud``: the routine's increments -- ``ptte``/``pqte`` (out - in),
  ``pxlte``/``pxite`` (out - in - detrainment), the surface fluxes
  ``prsfl``/``pssfl``, the written-back cover ``paclc``, and ``ktype``.

Reference variant
-----------------
The primary reference is the ``sonntag`` variant: ECHAM with its saturation
tables replaced by the Sonntag (1990) formula they tabulate (the tables
reproduce it to 4e-12 in e_s and 2e-9 in de_s/dT; the table variant's outputs
differ from it by < 3e-11 of each column's scale, far inside the tolerance).
The ``tetens`` variant (NOT ECHAM: jcm's former Tetens e_s inside the
otherwise unmodified routines) is compared too, with jcm's cloud schemes
pointed at the same Tetens pair for the duration (``saturation_formula``), so
a disagreement can be localised to the formulation or to the saturation
formula.

Physical constants
------------------
jcm's global constants differ from ECHAM's (grav, rv, alv, als, eps). Every
comparison runs with jcm's constants set to ECHAM's values, restored
afterwards, so the scheme's formulation is tested rather than the constants;
the constant differences themselves are asserted separately in
``test_physical_constants_match_echam``.

Tolerances
----------
An element passes if ``|jcm - ref| <= atol[field] + rtol * scale`` where
``scale`` is the column's largest ``|ref|`` for that field.

* float64: ``rtol = 1e-9``. Perturbing every input of the analytic Fortran by
  1e-13 (relative, 8 seeds) moves its outputs by at most 2.5e-10 of the column
  scale on physical columns (5.4e-10 on the unphysical ``supersat_ub_branch``);
  a faithful port differs from it only by operation-order roundoff
  (<= 1e-15 relative), so it lands within ~1e-11. 1e-9 keeps a 100x margin and
  sits 1e5x or more below every formulation difference found (constants 2e-4,
  Tetens vs Sonntag 1e-3 to 1e-1). The measurement is
  ``fortran_harness/echam_cloud/py/conditioning.py`` on the harness branch.
* float32: ``rtol = 2e-3``. jcm run in float32 differs from itself in float64
  by at most 2.2e-4 of the column scale on these inputs (fields above their
  ``atol`` floors), in line with the Fortran conditioning above
  (float32 epsilon 6e-8 times the measured amplification 2.5e3 = 1.5e-4);
  2e-3 is 10x that.
* ``atol`` floors (``ATOL``) sit below any physical signal but above ECHAM's
  own numerical floors: its ice-sedimentation input is floored at
  ``EPSILON(1.0) = 2.2e-16`` kg/kg (mo_cloud.f90:583) and its in-cloud
  condensate at 1e-20 (:911-912), which leave fluxes of O(1e-18) kg/m2/s and
  heating of O(1e-17) K/s in cloud-free columns.
* ``paclc`` near 1: ``cc = 1 - sqrt(1 - b0)`` has an unbounded slope at
  ``b0 = 1``; an error ``d`` in ``b0`` moves the cover by
  ``min(sqrt(d), d / (2 (1 - cc)))``, which is added to the tolerance with
  ``d = 1e-12`` (float64) / ``1e-5`` (float32, ~10 ulp of ``zqr`` over
  ``1 - rhc = 0.25``). It only matters within ~1e-3 of full cover.

Known gaps
----------
Comparisons that fail against today's jcm are marked ``xfail(strict=True)``
from ``known_gaps.json`` (in the data directory), each with the fields and
magnitudes that fail, so it becomes a hard failure (XPASS) the moment it is
fixed. Remove fixed entries with
``python jcm/physics/clouds/echam_fortran_reference_test.py --prune``, which
only ever deletes entries that now pass; it cannot add one. Tolerances are
never loosened to make a case pass.
"""
from __future__ import annotations

import contextlib
import functools
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c

REF_DIR = Path(__file__).resolve().parents[2] / "data" / "test" / "echam_cloud_reference"

# ECHAM6.3 mo_physical_constants.f90:96-147. rd = 287.04 and cpd = 1004.64
# already agree with jcm (rd = akap*cpd); eps is rd/rv.
ECHAM_CONSTANTS = dict(grav=9.80665, rv=461.51, alhc=2.5008e6, alhs=2.8345e6,
                       eps=287.04 / 461.51, cpd=1004.64, cpv=1869.46, tmelt=273.15,
                       rhow=1000.0)

RTOL_F64 = 1e-9
RTOL_F32 = 2e-3        # see the module docstring for both measurements
ATOL = {
    "ptte": 1e-13,     # K/s
    "pqte": 1e-16,     # kg/kg/s
    "pxlte": 1e-16,
    "pxite": 1e-16,
    "prsfl": 1e-16,    # kg/m2/s
    "pssfl": 1e-16,
    "paclc": 1e-12,    # 1
}
ATOL_F32 = {"ptte": 1e-9, "pqte": 1e-12, "pxlte": 1e-12, "pxite": 1e-12,
            "prsfl": 1e-12, "pssfl": 1e-12, "paclc": 1e-6}
B0_ERROR = {"float64": 1e-12, "float32": 1e-5}   # see module docstring


# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def load(kind: str) -> dict:
    """Arrays of ``cover_T63L47.npz`` / ``cloud_T63L47.npz`` (see README)."""
    with np.load(REF_DIR / f"{kind}_T63L47.npz") as z:
        return {k: z[k] for k in z.files}


@functools.lru_cache(maxsize=None)
def load_resolution() -> dict:
    with np.load(REF_DIR / "resolution_T31_T127_T255.npz") as z:
        return {k: z[k] for k in z.files}


def echam_inputs(kind: str, variant: str = "sonntag", nn: int | None = None) -> dict:
    """ECHAM input arguments of every column: ``{name: array}``, 2-D arrays
    shaped ``(nlev, ncol)`` top-first (index 0 = model top), exactly as the
    Fortran received them. For the realistic cloud columns ``paclc``/``knvb``
    came from that variant's (and truncation's) ``cover`` output.
    """
    z = load(kind)
    inp = {k[3:]: z[k] for k in z if k.startswith("in/")}
    if kind == "cloud":
        inp["paclc"] = z[f"in_variant/{variant}/paclc"]
        inp["knvb"] = z[f"in_variant/{variant}/knvb"]
        if nn is not None and nn != 63:
            r = load_resolution()
            inp["paclc"] = r[f"T{nn}/cloud/in/paclc"]
            inp["knvb"] = r[f"T{nn}/cloud/in/knvb"]
    inp["nn"] = int(inp["nn"]) if nn is None else nn
    return inp


def echam_outputs(kind: str, variant: str = "sonntag", nn: int | None = None) -> dict:
    if nn is not None and nn != 63:
        r = load_resolution()
        pre = f"T{nn}/{kind}/out/"
        return {k[len(pre):]: r[k] for k in r if k.startswith(pre)}
    z = load(kind)
    pre = f"out/{variant}/"
    return {k[len(pre):]: z[k] for k in z if k.startswith(pre)}


def column_names(kind: str) -> list[str]:
    return [str(s) for s in load(kind)["meta/names"]]


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
def saturation_formula(variant: str):
    """Run jcm with the vapour pressure of the reference variant.

    jcm's ECHAM cloud schemes use ECHAM's formula (Sonntag), the ``sonntag``
    and resolution references. For the ``tetens`` variant, a localisation
    aid, ``echam_saturation.SATURATION_FORMULA`` is pointed at the same
    Tetens pair for the duration.
    """
    from jcm.physics.clouds import echam_saturation
    saved = echam_saturation.SATURATION_FORMULA
    echam_saturation.SATURATION_FORMULA = (
        "tetens" if variant == "tetens" else "sonntag")
    try:
        yield
    finally:
        echam_saturation.SATURATION_FORMULA = saved


@contextlib.contextmanager
def precision(name: str):
    with jax.enable_x64(name == "float64"):
        yield


# ===========================================================================
# ADAPTERS -- the ONLY code in this module that knows jcm's interfaces.
# When jcm's cover / 1M signatures change, edit these two functions (and
# nothing else). Each takes ECHAM's input arguments (names, units and
# top-first layout of the Fortran argument list, see echam_inputs) and must
# return ECHAM's output arguments (same names, units and layout, float64
# numpy). ``nn`` is the spectral truncation whose resolution-dependent
# constants ECHAM used (mo_echam_cloud_params.f90:198-240).
# ===========================================================================
def run_jcm_cover(inp: dict, nn: int = 63) -> dict:
    """ECHAM ``cover`` -> jcm ``SundqvistCloudFraction``. Returns ``paclc``.

    Mapping: ptm1/pqm1/pxim1 -> state T/q/tracers["qi"] (step-start state,
    as ECHAM's cover reads the m1 fields, physc.f90:543-549); pgeo -> state
    geopotential (only level differences enter); papm1 -> pressure_full;
    paphm1[-1] -> surface_pressure; land fraction ``1 - pfrw - pfri`` ->
    terrain.fmask; sea-ice fraction of the water part ``pfri / (pfrw + pfri)``
    -> forcing.sice_am; ktype -> convection.ktype. ``vct`` is the vertical
    grid ``cache_coords`` derives ECHAM's jbmin/jbmax from, and ``nn`` selects
    ECHAM's parameter row (``CloudParameters.default(truncation=nn)``).
    """
    from jcm.physics.clouds.sundqvist import (
        CloudParameters,
        SundqvistCloudFraction,
    )

    t = jnp.asarray(inp["ptm1"])
    nlev, ncol = t.shape
    frl = 1.0 - inp["pfrw"] - inp["pfri"]
    water = inp["pfrw"] + inp["pfri"]
    sice = np.where(water > 0.0, inp["pfri"] / np.where(water > 0.0, water, 1.0), 0.0)
    state = SimpleNamespace(
        temperature=t, specific_humidity=jnp.asarray(inp["pqm1"]),
        geopotential=jnp.asarray(inp["pgeo"]),
        tracers={"qi": jnp.asarray(inp["pxim1"]), "qc": jnp.zeros_like(t)},
        u_wind=jnp.zeros_like(t), v_wind=jnp.zeros_like(t))
    diagnostics = {
        "pressure_full": jnp.asarray(inp["papm1"]),
        "surface_pressure": jnp.asarray(inp["paphm1"][-1]),
        "convection": SimpleNamespace(ktype=jnp.asarray(inp["ktype"])),
    }
    terrain = SimpleNamespace(fmask=jnp.asarray(frl))
    forcing = SimpleNamespace(sice_am=jnp.asarray(sice))
    vct = np.asarray(inp["vct"])
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(longitude_wavenumbers=nn + 1,
                                   nodal_shape=(ncol,)),
        vertical=SimpleNamespace(a_boundaries=vct[:nlev + 1],
                                 b_boundaries=vct[nlev + 1:]))
    term = SundqvistCloudFraction(CloudParameters.default(truncation=nn))
    term.cache_coords(coords)
    _, out = term(state, diagnostics, forcing, terrain)
    return {"paclc": np.asarray(out["clouds"].cloud_fraction, np.float64)}


class _Convection(SimpleNamespace):
    def replace(self, **kw):
        return _Convection(**{**vars(self), **kw})


def run_jcm_cloud(inp: dict, nn: int = 63) -> dict:
    """ECHAM ``cloud`` (1M) -> jcm ``Echam1MMicrophysics``.

    Mapping (ECHAM passes m1 state plus accumulated tendencies; jcm passes the
    step-start state, the provisional ``thermo_run`` view and interim
    condensate):
      state T, q, qc, qi       <- ptm1, pqm1, pxlm1, pxim1
      thermo_run T, q          <- ptm1 + ptte*dt, pqm1 + pqte*dt
      clouds.qc / clouds.qi    <- pxlm1 + (pxlte + pxtecl)*dt, pxim1 + (pxite + pxteci)*dt
      clouds.cloud_fraction    <- paclc
      air_density              <- papm1 / (rd*ptvm1)            (mo_cloud.f90:382)
      layer_thickness          <- dp / (g*air_density), dp from paphm1, so that
                                  rho*dz = dp/g exactly as ECHAM's layer mass
      droplet number           <- jcm's prescribed profile; equals pacdnc
                                  (ECHAM physc.f90 3.12 acdnc) for the land flag
      convection ktype/top     <- ktype, kctop - 1 (0-based)
      dt                       <- ptime_step_len
    Returns ECHAM's INOUT arrays after the call: tendencies = input +
    jcm's increment (+ detrainment for pxlte/pxite, which ECHAM adds in 8.3),
    prsfl/pssfl, paclc after the ccwmin write-back, and ktype.
    jcm has no resolution-dependent 1M constants, so ``nn`` is unused.
    """
    from jcm.physics.clouds.cloud_data import CloudData
    from jcm.physics.clouds.echam_1m import Echam1MMicrophysics

    del nn
    dt = float(inp["ptime_step_len"])
    ptm1 = jnp.asarray(inp["ptm1"])
    nlev, ncol = ptm1.shape
    dp = jnp.asarray(np.diff(inp["paphm1"], axis=0))
    rho = jnp.asarray(inp["papm1"] / (c.rd * inp["ptvm1"]))
    qc_int = jnp.asarray(inp["pxlm1"] + (inp["pxlte"] + inp["pxtecl"]) * dt)
    qi_int = jnp.asarray(inp["pxim1"] + (inp["pxite"] + inp["pxteci"]) * dt)
    state = SimpleNamespace(
        temperature=ptm1, specific_humidity=jnp.asarray(inp["pqm1"]),
        tracers={"qc": jnp.asarray(inp["pxlm1"]), "qi": jnp.asarray(inp["pxim1"])},
        u_wind=jnp.zeros_like(ptm1), v_wind=jnp.zeros_like(ptm1))
    clouds = CloudData.zeros((ncol,), nlev).copy(
        cloud_fraction=jnp.asarray(inp["paclc"]), qc=qc_int, qi=qi_int)
    diagnostics = {
        "_dt_seconds": dt,
        "pressure_full": jnp.asarray(inp["papm1"]),
        "air_density": rho,
        "layer_thickness": dp / (c.grav * rho),
        "pressure_thickness": dp,
        "clouds": clouds,
        "aerosol": SimpleNamespace(cdnc_factor=jnp.ones(ncol)),
        "thermo_run": {
            "temperature": jnp.asarray(inp["ptm1"] + inp["ptte"] * dt),
            "specific_humidity": jnp.asarray(inp["pqm1"] + inp["pqte"] * dt),
            "qc": qc_int, "qi": qi_int,
        },
        "convection": _Convection(ktype=jnp.asarray(inp["ktype"]),
                                  cloud_top=jnp.asarray(inp["kctop"] - 1)),
    }
    terrain = SimpleNamespace(fmask=jnp.asarray(inp["land"].astype(float)))
    forcing = SimpleNamespace()
    tend, out = Echam1MMicrophysics()(state, diagnostics, forcing, terrain)
    f = functools.partial(np.asarray, dtype=np.float64)
    return {
        "ptte": inp["ptte"] + f(tend.temperature),
        "pqte": inp["pqte"] + f(tend.specific_humidity),
        "pxlte": inp["pxlte"] + inp["pxtecl"] + f(tend.tracers["qc"]),
        "pxite": inp["pxite"] + inp["pxteci"] + f(tend.tracers["qi"]),
        "prsfl": f(out["clouds"].precip_rain),
        "pssfl": f(out["clouds"].precip_snow),
        "paclc": f(out["clouds"].cloud_fraction),
        "ktype": np.asarray(out["convection"].ktype).astype(np.int64),
    }
# ===========================================================================
# end of adapters
# ===========================================================================


def increments(kind: str, inp: dict, out: dict) -> dict:
    """Return the routine's own contribution, computed identically for ECHAM and jcm."""
    if kind == "cover":
        return {"paclc": out["paclc"]}
    return {
        "ptte": out["ptte"] - inp["ptte"],
        "pqte": out["pqte"] - inp["pqte"],
        "pxlte": out["pxlte"] - inp["pxlte"] - inp["pxtecl"],
        "pxite": out["pxite"] - inp["pxite"] - inp["pxteci"],
        "prsfl": out["prsfl"][None, :],
        "pssfl": out["pssfl"][None, :],
        "paclc": out["paclc"],
    }


@functools.lru_cache(maxsize=None)
def comparison(kind: str, variant: str, prec: str, nn: int = 63) -> dict:
    """Per column: {field: (max_abs_err, max_err/tol, scale)} plus ktype.

    ``max_err/tol`` <= 1 means the field passes.
    """
    inp = echam_inputs(kind, variant, None if nn == 63 else nn)
    ref = echam_outputs(kind, variant, None if nn == 63 else nn)
    run = run_jcm_cover if kind == "cover" else run_jcm_cloud
    with echam_constants(), saturation_formula(variant), precision(prec):
        got = run(inp, nn=nn)
    ri, gi = increments(kind, inp, ref), increments(kind, inp, got)
    rtol = RTOL_F64 if prec == "float64" else RTOL_F32
    atol = ATOL if prec == "float64" else ATOL_F32
    res = {}
    for j, name in enumerate(column_names(kind)):
        fields = {}
        for k in ri:
            r, g = ri[k][:, j], gi[k][:, j]
            scale = float(np.max(np.abs(r)))
            tol = atol[k] + rtol * scale
            if k == "paclc":
                d = B0_ERROR[prec]
                tol = tol + np.minimum(np.sqrt(d), d / (2.0 * np.maximum(1.0 - r, 1e-300)))
            err = np.abs(g - r)
            fields[k] = (float(err.max()), float(np.max(err / tol)), scale)
        if kind == "cloud":
            fields["ktype"] = (float(abs(int(got["ktype"][j]) - int(ref["ktype"][j]))),
                               float(int(got["ktype"][j]) != int(ref["ktype"][j])) * 2.0, 0.0)
        res[name] = fields
    return res


def _failure_message(fields: dict) -> str:
    bad = {k: v for k, v in fields.items() if v[1] > 1.0}
    return "; ".join(f"{k}: max|err| {e:.3g} ({ratio:.3g} x tol, scale {s:.3g})"
                     for k, (e, ratio, s) in sorted(bad.items()))


def _check(kind, variant, prec, column, nn=63):
    fields = comparison(kind, variant, prec, nn)[column]
    msg = _failure_message(fields)
    assert not msg, f"{kind} [{variant}, {prec}, T{nn}] column {column}: {msg}"


# ---------------------------------------------------------------------------
# known gaps of today's jcm: "kind|variant|precision|column" -> reason
# (variant is "sonntag", "tetens" or "T31"/"T127"/"T255" for the
# resolution tests)
# ---------------------------------------------------------------------------
KNOWN_GAPS_FILE = REF_DIR / "known_gaps.json"


def _load_known_gaps() -> dict[tuple, str]:
    import json
    if not KNOWN_GAPS_FILE.exists():
        return {}
    raw = json.loads(KNOWN_GAPS_FILE.read_text())
    return {tuple(k.split("|")): v for k, v in raw["gaps"].items()}


KNOWN_GAPS = _load_known_gaps()


def _cases(kind, variants=("sonntag", "tetens"), precs=("float64", "float32"), nn=63):
    out = []
    for variant in variants:
        for prec in precs:
            for col in column_names(kind):
                tags = ""
                if kind == "cloud":
                    tags = str(load("cloud")["meta/tags"][column_names("cloud").index(col)])
                if prec == "float32" and "f64_only" in tags:
                    continue
                key = (kind, variant, prec, col) if nn == 63 else (kind, f"T{nn}", prec, col)
                marks = []
                if key in KNOWN_GAPS:
                    marks.append(pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[key]))
                out.append(pytest.param(variant, prec, col, marks=marks,
                                        id=f"{variant}-{prec}-{col}"))
    return out


@pytest.mark.parametrize("variant,prec,column", _cases("cover"))
def test_cover_matches_echam(variant, prec, column):
    """Check jcm's cover against ECHAM mo_cover.f90::cover on one column."""
    _check("cover", variant, prec, column)


@pytest.mark.parametrize("variant,prec,column", _cases("cloud"))
def test_cloud_matches_echam(variant, prec, column):
    """Check jcm's 1M scheme against ECHAM mo_cloud.f90::cloud on one column."""
    _check("cloud", variant, prec, column)


def _res_cases(kind):
    out = []
    for nn in (31, 127, 255):
        for col in column_names(kind):
            key = (kind, f"T{nn}", "float64", col)
            marks = [pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[key])] if key in KNOWN_GAPS else []
            out.append(pytest.param(nn, col, marks=marks, id=f"T{nn}-{col}"))
    return out


@pytest.mark.parametrize("nn,column", _res_cases("cover"))
def test_cover_resolution_constants(nn, column):
    """Cover with ECHAM's T31/T127/T255 constants (crs, crt, nex, nadd, csatsc,
    cinv; mo_echam_cloud_params.f90:198-237), analytic saturation.
    """
    _check("cover", "sonntag", "float64", column, nn)


@pytest.mark.parametrize("nn,column", _res_cases("cloud"))
def test_cloud_resolution_constants(nn, column):
    """1M with ECHAM's T31/T127/T255 constants (cvtfall, csecfrl, clwprat)."""
    _check("cloud", "sonntag", "float64", column, nn)


def test_reference_data_integrity():
    """The committed data are internally consistent: the table variant agrees
    with the analytic one far inside the float64 tolerance, the stored
    parameters are ECHAM's T63 values, and inputs are finite.
    """
    for kind in ("cover", "cloud"):
        z = load(kind)
        for k, v in z.items():
            if v.dtype.kind == "f":
                assert np.all(np.isfinite(v)), k
        inp = echam_inputs(kind)
        rt = increments(kind, inp, echam_outputs(kind, "table"))
        ra = increments(kind, inp, echam_outputs(kind, "sonntag"))
        for k in rt:
            scale = np.max(np.abs(ra[k]), axis=0)
            assert np.all(np.abs(rt[k] - ra[k]) <= ATOL[k] + 1e-2 * RTOL_F64 * scale + 3e-11 * scale), k
        assert float(z["param/sonntag/crs"]) == 0.975
        assert float(z["param/sonntag/cvtfall"]) == 2.5
        assert int(z["param/sonntag/jbmin"]) == 40 and int(z["param/sonntag/jbmax"]) == 45


@pytest.mark.xfail(strict=True, reason=(
    "jcm constants differ from ECHAM6.3 mo_physical_constants.f90: grav 9.81 vs "
    "9.80665, rv 461.0 vs 461.51, alhc 2.501e6 vs 2.5008e6, alhs 2.834e6 vs "
    "2.8345e6, eps 0.622 vs rd/rv = 0.621958 (relative 1e-4 to 1e-3)."))
def test_physical_constants_match_echam():
    """Check jcm's global constants against ECHAM6.3's (the comparisons above run
    with ECHAM's substituted, so this is the only place their difference
    shows).
    """
    pc = c.PhysicalConstants.default()
    diffs = {k: (getattr(pc, k), v) for k, v in ECHAM_CONSTANTS.items()
             if not np.isclose(getattr(pc, k), v, rtol=1e-12, atol=0.0)}
    assert not diffs, diffs


# ---------------------------------------------------------------------------
# known_gaps.json maintenance
# ---------------------------------------------------------------------------
def _all_keys():
    for kind in ("cover", "cloud"):
        for variant in ("sonntag", "tetens"):
            for prec in ("float64", "float32"):
                for col in column_names(kind):
                    yield (kind, variant, prec, col), (kind, variant, prec, 63)
        for nn in (31, 127, 255):
            for col in column_names(kind):
                yield (kind, f"T{nn}", "float64", col), (kind, "sonntag", "float64", nn)


def _gap_reason(kind, col, fields):
    """Compact reason: process, then each failing field's max |error| and its
    ratio to the tolerance.
    """
    proc = str(load(kind)["meta/process"][column_names(kind).index(col)])
    bad = sorted((k, v) for k, v in fields.items() if v[1] > 1.0)
    return f"[{proc}] " + ", ".join(f"{k} {e:.2g} ({r:.1g}x tol)" for k, (e, r, _s) in bad)


def _write_known_gaps(mode: str) -> None:
    """``init``: record every failing comparison of the current jcm (used once
    to create the file). ``prune``: drop entries whose comparison now passes;
    never adds one.
    """
    import json
    old = {"|".join(k): v for k, v in _load_known_gaps().items()}
    gaps = {}
    for key, (kind, variant, prec, nn) in _all_keys():
        col = key[3]
        tags = ""
        if kind == "cloud":
            tags = str(load("cloud")["meta/tags"][column_names("cloud").index(col)])
        if prec == "float32" and "f64_only" in tags:
            continue
        fields = comparison(kind, variant, prec, nn)[col]
        failing = bool(_failure_message(fields))
        k = "|".join(key)
        if mode == "init" and failing:
            gaps[k] = _gap_reason(kind, col, fields)
        elif mode == "prune" and k in old and failing:
            gaps[k] = old[k]
    doc = {
        "about": ("Comparisons of jcm against the ECHAM6.3 reference that fail today, "
                  "run as xfail(strict=True) by echam_fortran_reference_test.py. Remove an "
                  "entry when it is fixed (the --prune option does this and never adds)."),
        "gaps": dict(sorted(gaps.items())),
    }
    KNOWN_GAPS_FILE.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(gaps)} known gaps written ({len(old)} before)")


def show(kind: str, column: str, variant: str = "sonntag", prec: str = "float64") -> None:
    """Print one column level by level: pressure, ECHAM and jcm increments of
    every field that fails, and (cloud, sonntag) the ECHAM intermediates that
    are non-zero at that level -- to localise a failure to one process.
    """
    inp = echam_inputs(kind, variant)
    ref = echam_outputs(kind, variant)
    run = run_jcm_cover if kind == "cover" else run_jcm_cloud
    with echam_constants(), saturation_formula(variant), precision(prec):
        got = run(inp)
    j = column_names(kind).index(column)
    ri, gi = increments(kind, inp, ref), increments(kind, inp, got)
    fields = comparison(kind, variant, prec)[column]
    print(f"{kind} {column} [{variant}, {prec}]: {_failure_message(fields) or 'passes'}")
    z = load(kind)
    diag = {k.split("/")[-1]: z[k][:, j] for k in z if k.startswith(f"diag/{variant}/")}
    for k in ri:
        if fields[k][1] <= 1.0 or ri[k].shape[0] == 1:
            if fields[k][1] > 1.0:
                print(f"  {k}: ECHAM {ri[k][0, j]:.6e}  jcm {gi[k][0, j]:.6e}")
            continue
        print(f"  {k} (level, p[hPa], ECHAM, jcm):")
        for lev in range(ri[k].shape[0]):
            r, g = ri[k][lev, j], gi[k][lev, j]
            if max(abs(r), abs(g)) <= ATOL[k]:
                continue
            active = [n for n, v in diag.items()
                      if n.startswith("z") and abs(v[lev]) > 1e-14 and n not in ("zqsm1", "ztp1", "zqp1",
                                                                             "ztp1tmp", "ztp1tmp_pre54",
                                                                             "zqp1tmp_pre54", "zqsp1tmp",
                                                                             "zrieff", "zcolleffi",
                                                                             "zdqsat1", "zxlb_7", "zxib_7")]
            print(f"    {lev:2d} {inp['papm1'][lev, j] / 100:7.1f} {r: .6e} {g: .6e}"
                  + (f"   ECHAM active: {' '.join(active)}" if active else ""))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args[:1] == ["--show"] and len(args) >= 3:
        show(*args[1:])
    elif len(args) == 1 and args[0] in ("--prune", "--init"):
        _write_known_gaps(args[0][2:])
    else:
        raise SystemExit("usage: echam_fortran_reference_test.py --prune | --init | "
                         "--show <cover|cloud> <column> [variant] [precision]")
