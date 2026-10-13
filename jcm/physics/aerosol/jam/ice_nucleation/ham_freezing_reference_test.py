"""jcm's HAM freezing-input partition against the compiled ECHAM6.3-HAM2.3 routine (#953).

``jcm/data/test/echam_cloud_reference/hamfrz_M7.npz`` holds the outputs of the
UNMODIFIED r7492 ``mo_ham_freezing.f90::ham_IN_setup`` (with
``get_aerofreez_nc``) on designed single-level M7 aerosol cells
(``hamfrz_README.md``, ``hamfrz_provenance.json``). The same cells are fed to
:func:`ham_freezing_aerosol` on an M7-shaped jcm population (HAM's seven modes,
its species membership and densities, with HAM's roles for the classes), so the
comparison tests the partition, not the MAM4 mapping (which is documented and
tested in ``ice_nucleation_test.py``). The fractions are closed forms of the
inputs, so float64 agrees to round-off and float32 to its precision.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import (
    HamFreezingClasses,
    ham_freezing_aerosol,
)
from jcm.physics.aerosol.jam.population import AerosolMode, AerosolSpecies, ModalAerosolSpec

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "hamfrz_M7.npz")
# M7 membership (mo_ham_m7ctl / mo_ham_init): SO4 BC OC SS DU.
M7_MEMBER = {"nucs": ("so4",), "aits": ("so4", "bc", "oc"),
             "accs": ("so4", "bc", "oc", "ss", "du"), "coas": ("so4", "bc", "oc", "ss", "du"),
             "aiti": ("bc", "oc"), "acci": ("du",), "coai": ("du",)}
M7_CLASSES = HamFreezingClasses(soluble=("accs", "coas"), insoluble_aitken="aiti",
                                insoluble_accumulation="acci", insoluble_coarse="coai")
FIELDS = (("dust_soluble", "fracdusol"), ("bc_soluble", "fracbcsol"),
          ("dust_insoluble_accumulation", "fracduai"), ("dust_insoluble_coarse", "fracduci"),
          ("bc_insoluble", "fracbcinsol"), ("wet_radius_insoluble_aitken", "rwetki"),
          ("wet_radius_insoluble_accumulation", "rwetai"),
          ("wet_radius_insoluble_coarse", "rwetci"))
RTOL = {"float64": 1e-12, "float32": 2e-6}


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def m7_spec():
    z = load()
    species = tuple(AerosolSpecies(s, 0.1, float(z[f"in/density/{s}"]), 0.1)
                    for s in ("so4", "bc", "oc", "ss", "du"))
    # Only species membership, solubility and density enter the partition; the
    # geometry fields are placeholders.
    modes = tuple(AerosolMode(m, m, 1.6, 1e-7, 1e-8, 1e-6, M7_MEMBER[m],
                              soluble=m.endswith("s"), can_activate=m.endswith("s"),
                              sediments=True) for m in M7_MEMBER)
    return ModalAerosolSpec(modes=modes, species=species)


def run_jcm(prec):
    z = load()
    dtype = jnp.float64 if prec == "float64" else jnp.float32
    n = len(z["meta/names"])

    def g(k):
        return jnp.asarray(z[k] if k in z else np.zeros(n), dtype)

    masses = {(s, m): g(f"in/mass/{m}/{s}") for m in M7_MEMBER for s in M7_MEMBER[m]}
    number = {m: g(f"in/number/{m}") for m in M7_MEMBER}
    nact = {m: g(f"in/nact/{m}") for m in M7_MEMBER}
    rwet = {m: g(f"in/rwet/{m}") for m in M7_MEMBER}
    return ham_freezing_aerosol(m7_spec(), M7_CLASSES, masses, number, nact, rwet,
                                g("in/rho"), g("in/cdncact"))


@pytest.mark.parametrize("prec", ("float64", "float32"))
def test_ham_freezing_aerosol_matches_ham_in_setup(prec):
    """Every fraction and radius of every cell, including the empty-class and tiny-ratio
    thresholds (F 248-257) and the MIN(., 1) clip (F 116-120).
    """
    with jax.enable_x64(prec == "float64"):
        fa = run_jcm(prec)
        z = load()
        for field, key in FIELDS:
            np.testing.assert_allclose(np.asarray(getattr(fa, field), np.float64), z[f"out/{key}"],
                                       rtol=RTOL[prec], atol=0.0, err_msg=key)


def test_reference_cells_exercise_every_branch():
    """The fixture is not vacuous: each branch the partition has is hit by some cell."""
    z = load()
    names = [str(s) for s in z["meta/names"]]
    col = {n: i for i, n in enumerate(names)}
    assert z["out/fracdusol"][col["fraction_clipped"]] == 1.0
    assert 0.0 < z["out/fracdusol"][col["dust_soluble"]] < 1.0
    assert 0.0 < z["out/fracbcsol"][col["bc_soluble"]] < 1.0
    for key in ("fracduai", "fracduci", "fracbcinsol"):
        assert 0.0 < z[f"out/{key}"][col["insoluble"]] < 1.0
    for cell in ("empty", "no_activation", "tiny_ratio", "tiny_class"):
        assert all(z[f"out/{k}"][col[cell]] == 0.0 for _, k in FIELDS[:5]), cell
    # the soluble Aitken BC is excluded (DN #295): bc_soluble counts only accs + coas
    assert z["out/nbcsol_strat"][col["bc_soluble"]] > 0.0
