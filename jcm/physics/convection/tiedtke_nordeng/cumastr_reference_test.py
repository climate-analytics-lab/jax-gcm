"""The Tiedtke-Nordeng scheme against ECHAM6.3's compiled convection.

``jcm/data/test/echam_cumastr_reference/`` holds what ECHAM6.3's compiled
``cucall``/``cumastr`` returns for 600 columns (provenance in the README
there): 400 states of jcm's whole-model radiative-convective column, where
ECHAM's shallow plume at the first interface above cloud base decides whether
the column convects at all, 100 of them with a synthetic resolved ascent (the
mid-level trigger) and 100 with a synthetic moisture convergence (deep
convection, the Nordeng closure and downdrafts).

jcm runs in float64 with ECHAM's physical constants for the comparison, as
``cuadjtq_test.py`` does: ECHAM's ``grav``, ``rv``, ``alv`` and ``als``
(``rd``, ``cpd``, ``cpv`` and ``tmelt`` already agree). Under them:

* every decision agrees on every column: whether it convects (``ldcum``),
  its type, its cloud base and its cloud top;
* the cloud-base mass flux, the surface precipitation and the temperature and
  humidity tendency of every level agree to rounding. Measured, relative to
  ECHAM's value (per level: to the column's largest tendency): 7.8e-13,
  6.0e-13, 2.2e-12 and 1.3e-12. The bounds are 1e-10, a hundred times the
  largest.

jcm's own constants differ from ECHAM's in ``rv`` (461.0 against 461.51),
which moves ``qsat`` and the virtual-temperature coefficient by 0.11 %, and in
the latent heats (2.501e6 and 2.834e6 against 2.5008e6 and 2.8345e6 J/kg).
The model keeps its unified constants; with them 53 of the 600 columns take a
different decision (``_JCM_CONSTANT_STEPS``). ECHAM's ``rv`` alone brings all
but two back (``_JCM_CONSTANT_STEPS_AFTER_RV``) and the latent heats those
two. Every one of these columns sits within hundredths of a kelvin of an
ascent-test threshold.
"""

import contextlib
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    tiedtke_nordeng_convection,
)

_REFERENCE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "data",
    "test", "echam_cumastr_reference", "echam_cumastr.npz")

_INPUTS = ("temperature", "humidity", "pressure", "layer_thickness", "rho",
           "u_wind", "v_wind", "qc", "qi", "land_fraction", "moisture_supply",
           "moisture_tend_profile", "thvsig", "omega", "qte_dynamics",
           "layer_mass", "humidity_m1", "pressure_half")

#: Relative tolerance of every continuous output (see the module docstring).
_RTOL = 1.0e-10

#: The columns whose decision differs from ECHAM's under jcm's constants.
_JCM_CONSTANT_STEPS = (
    9, 11, 12, 13, 14, 15, 17, 18, 19, 28, 30, 34, 35, 37, 51, 52, 53, 60, 61,
    64, 66, 67, 68, 223, 311, 320, 322, 323, 324, 330, 331, 332, 334, 341,
    342, 343, 344, 345, 348, 349, 350, 353, 354, 355, 356, 357, 362, 504, 510,
    524, 525, 558, 591)
#: ... and those that still differ with ECHAM's ``rv`` alone.
_JCM_CONSTANT_STEPS_AFTER_RV = (53, 354)


@functools.lru_cache(maxsize=1)
def _reference():
    with np.load(_REFERENCE) as z:
        return {k: z[k] for k in z.files}


@contextlib.contextmanager
def _constants(**overrides):
    saved = c.physical_constants
    c.set_constants(**overrides)
    try:
        yield
    finally:
        c.set_constants(saved)


def _echam_constants(ref):
    return dict(grav=float(ref["echam_grav"]), rv=float(ref["echam_rv"]),
                alhc=float(ref["echam_alv"]), alhs=float(ref["echam_als"]),
                cpd=float(ref["echam_cpd"]), tmelt=float(ref["echam_tmelt"]))


@functools.lru_cache(maxsize=None)
def _port(constants):
    """Run the scheme in float64 on every column under ``constants``."""
    ref = _reference()
    with _constants(**dict(constants)), jax.enable_x64(True):
        args = [jnp.asarray(ref[f"input_{k}"], jnp.float64) for k in _INPUTS]
        dt = float(ref["dt"])
        config = ConvectionParameters.default()

        def column(*a):
            (temperature, humidity, pressure, layer_thickness, rho, u_wind,
             v_wind, qc, qi, land_fraction, moisture_supply,
             moisture_tend_profile, thvsig, omega, qte_dynamics, layer_mass,
             humidity_m1, pressure_half) = a
            return tiedtke_nordeng_convection(
                temperature, humidity, pressure, layer_thickness, rho, u_wind,
                v_wind, qc, qi, dt, config, land_fraction, moisture_supply,
                moisture_tend_profile, thvsig, omega, qte_dynamics,
                layer_mass, humidity_m1, False, pressure_half)

        tend, state = jax.jit(jax.vmap(column))(*args)
        kbase = np.asarray(state.kbase)
        return dict(
            ktype=np.asarray(state.ktype), kcbot=kbase + 1,
            kctop=np.asarray(state.ktop) + 1,
            mfub=np.take_along_axis(
                np.asarray(state.mfu), kbase[:, None], axis=1)[:, 0],
            precip=np.asarray(tend.precip_conv),
            dtedt=np.asarray(tend.dtedt), dqdt=np.asarray(tend.dqdt),
        )


def _echam(ref=None):
    ref = _reference() if ref is None else ref
    return _port(tuple(sorted(_echam_constants(ref).items())))


def _decision_mismatches(port, ref):
    """Columns where any decision differs, with the first that does."""
    on_port, on_echam = port["ktype"] > 0, ref["echam_ktype"] > 0
    out = {}
    for i in range(len(on_port)):
        if on_port[i] != on_echam[i]:
            out[i] = f"ldcum jcm {bool(on_port[i])} ECHAM {bool(on_echam[i])}"
        elif on_port[i]:
            for name, key in (("ktype", "echam_ktype"), ("kcbot", "echam_kcbot"),
                              ("kctop", "echam_kctop")):
                if port[name][i] != ref[key][i]:
                    out[i] = f"{name} jcm {port[name][i]} ECHAM {ref[key][i]}"
                    break
    return out


def _describe(mismatches, ref):
    return "; ".join(
        f"column {i} ({ref['group'][i]}, step {ref['source_step'][i]}): {what}"
        for i, what in sorted(mismatches.items()))


class TestReferenceData:
    """The reference exercises each path it is meant to."""

    def test_every_decision_is_represented(self):
        ref = _reference()
        kt, group = ref["echam_ktype"], ref["group"]
        for name in ("rce_fogged", "rce_warm"):
            m = group == name
            assert np.any(kt[m] == 0) and np.any(kt[m] == 2), name
        assert np.any(kt[np.char.startswith(group, "midlevel")] == 3)
        assert np.any(kt[np.char.startswith(group, "deep")] == 1)
        # jcm's earlier scheme convected in these columns and ECHAM does not.
        prev = ref["sample_class"] == "previously_port_only"
        assert prev.sum() == 140 and np.all(kt[prev] == 0)

    def test_constants_match_echam(self):
        ref = _reference()
        with _constants(**_echam_constants(ref)):
            np.testing.assert_allclose(c.rd, ref["echam_rd"], rtol=1e-15)
            np.testing.assert_allclose(c.cpv, ref["echam_cpv"], rtol=1e-15)
            np.testing.assert_allclose(
                c.vtmpc1, ref["echam_rv"] / ref["echam_rd"] - 1.0, rtol=1e-14)


class TestAgainstEcham:
    """jcm's scheme against ECHAM6.3, float64, ECHAM's constants."""

    def test_decisions_match(self):
        ref = _reference()
        mismatches = _decision_mismatches(_echam(), ref)
        assert not mismatches, _describe(mismatches, ref)

    @pytest.mark.parametrize("name,key", [
        ("mfub", "echam_cloud_base_mass_flux"),
        ("precip", None),
    ])
    def test_column_outputs_match(self, name, key):
        ref = _reference()
        port = _echam()
        on = ref["echam_ktype"] > 0
        want = (ref["echam_rain"] + ref["echam_snow"] if key is None
                else ref[key])
        np.testing.assert_allclose(port[name][on], want[on], rtol=_RTOL,
                                   atol=0.0, err_msg=name)

    @pytest.mark.parametrize("name,key", [
        ("dtedt", "echam_temperature_tendency"),
        ("dqdt", "echam_humidity_tendency"),
    ])
    def test_tendencies_match(self, name, key):
        ref = _reference()
        port = _echam()
        want = ref[key]
        scale = np.max(np.abs(want), axis=1, keepdims=True)
        err = np.abs(port[name] - want)
        on = ref["echam_ktype"] > 0
        # Where ECHAM does not convect the scheme returns exactly nothing.
        assert np.all(port[name][~on] == 0.0), name
        worst = np.max(err[on] / scale[on])
        assert worst <= _RTOL, f"{name}: worst per-level error {worst:.2e}"


class TestJcmConstants:
    """The decisions under jcm's unified constants, pinned column by column."""

    def test_columns_that_differ(self):
        ref = _reference()
        mismatches = _decision_mismatches(_port(()), ref)
        assert tuple(sorted(mismatches)) == _JCM_CONSTANT_STEPS, (
            _describe(mismatches, ref))

    def test_echam_rv_restores_all_but_two(self):
        ref = _reference()
        mismatches = _decision_mismatches(
            _port((("rv", float(ref["echam_rv"])),)), ref)
        assert tuple(sorted(mismatches)) == _JCM_CONSTANT_STEPS_AFTER_RV, (
            _describe(mismatches, ref))
