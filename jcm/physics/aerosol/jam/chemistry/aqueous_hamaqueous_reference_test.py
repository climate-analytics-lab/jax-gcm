"""``_aqueous_so4`` against the compiled ECHAM6.3-HAM2.3 routine (#1031).

``jcm/data/test/echam_cloud_reference/hamaqueous_M7.npz`` holds the outputs
of the UNMODIFIED r7492 ``mo_ham_chemistry.f90::ham_wet_chemistry`` on
designed single-level M7 cells (``hamaqueous_README.md``,
``hamaqueous_provenance.json``). Reproducing it required moving five HAM
literals ``_aqueous_so4`` otherwise hardcoded differently (the SO2
Henry's-law pair, the gas constant, Avogadro's number and its separate
``xtoc``/``ctox`` rounding, SO2's molar mass) to r7492's own values — the
SO2 Henry pair specifically was a genuine defect (#1031), the dominant
driver of a -9% to -52% disagreement in produced sulfate. This change is
on the shared kernel, so it moves the ``echam-jam``/MAM4 default path too
(a deliberate, measured change — see the PR, not this test).

The reference fixture's SO4 species is M7's own (96.0631 g/mol), not jcm's
MAM4 value (115 g/mol, kept as :data:`aqueous._MW_SO4` — jcm's own species
choice, not a HAM literal) — ``_aqueous_so4``'s ``mw_so4``/``conv_so2_so4``
parameters let this test supply M7's value directly without wiring up an
M7 population on this branch (no ``AqueousConstants``/``ModalAerosolSpec``
override exists here; see jax-gcm#1017 for that fuller M7 adapter).

This test calls the bare kernel, not the full ``AqueousSulfur`` term: the
reference records GRID-MEAN tendencies after HAM's own number-fraction
mode split (``pxtte_ms4as``/``pxtte_ms4cs``), but that split only
redistributes the kernel's own undivided in-cloud production, so
``(pxtte_ms4as + pxtte_ms4cs)·dt/paclc`` recovers exactly the mass
``_aqueous_so4`` itself returns (``dso4``) — the comparison this test
makes. Cell 0 (``no_cloud``, ``paclc=0``) produces no sulfate on either
side and is compared directly rather than divided by a zero cloud
fraction.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.chemistry.aqueous import _aqueous_so4

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "hamaqueous_M7.npz")
RTOL = 1e-12
#: r7492's own M7 SO4 molar mass and SO2 molar mass (``hamaqueous_
#: provenance.json``'s ``configuration``), giving the mass-conversion
#: ratio the fixture's cells were produced with.
_MW_SO4_M7 = 96.0631
_MW_SO2_REF = 64.0643


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def _kernel_dso4(z, dtype=jnp.float64):
    """Run ``_aqueous_so4`` on every recorded cell at M7's own SO4 mass."""
    def g(k):
        return jnp.asarray(z[k], dtype)

    so4_total = g("in/ms4as") + g("in/ms4cs")
    dt = float(z["in/time_step_len"])
    return _aqueous_so4(
        so2=g("in/so2"), so4=so4_total, h2o2=g("in/h2o2_density"),
        o3=g("in/o3_density"), lwc=g("in/pmlwc"), rho=g("in/rhop1"),
        temperature=g("in/tm1"), dt=dt,
        mw_so4=_MW_SO4_M7, conv_so2_so4=_MW_SO4_M7 / _MW_SO2_REF,
    )


def test_aqueous_so4_kernel_matches_ham_wet_chemistry():
    """Every cell, at float64 round-off, against the undivided production.

    Covers the full branch structure the 16 designed cells exercise (the
    LWC gate, warm/cold, oxidant-limited regimes, and every HAM
    number-fraction split branch — identical to the kernel regardless of
    which branch, since the split only redistributes its output).
    """
    with jax.enable_x64(True):
        z = load()
        got = np.asarray(_kernel_dso4(z), np.float64)

        paclc = z["in/paclc"]
        rate = z["out/pxtte_ms4as"] + z["out/pxtte_ms4cs"]
        dt = float(z["in/time_step_len"])
        # no_cloud (paclc=0) produces no sulfate on either side; every
        # other cell's rate -> mass is rate*dt/paclc (the grid-mean ->
        # in-cloud undo, see the module docstring above). The floor avoids
        # a 0/0 at paclc=0, where the `where` below discards it anyway.
        safe_paclc = np.where(paclc > 0.0, paclc, 1.0)
        expected = np.where(paclc > 0.0, rate * dt / safe_paclc, 0.0)

        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=0.0)


def test_reference_cells_exercise_every_branch():
    """The fixture is not vacuous: each of HAM's split branches is hit."""
    z = load()
    names = [str(s) for s in z["meta/names"]]
    col = {n: i for i, n in enumerate(names)}
    assert z["out/pxtte_so2"][col["no_cloud"]] == 0.0
    assert z["out/pxtte_so2"][col["thick_cloud"]] < 0.0
    assert z["out/pxtte_ncs"][col["as_below_cs_below"]] > 0.0
    assert z["out/pxtte_ncs"][col["both_empty"]] > 0.0
    assert all(z["out/pxtte_nas"] == 0.0)
    for cell in ("as_above_cs_above", "as_above_cs_below", "as_below_cs_above"):
        assert z["out/pxtte_ncs"][col[cell]] == 0.0, cell
    # The mode split redistributes, it does not change, the kernel's own
    # undivided production: every "neither/either present" cell (9-13)
    # sums to the same total.
    split_cells = ["as_below_cs_below", "as_above_cs_above",
                   "as_above_cs_below", "as_below_cs_above", "both_empty"]
    totals = [z["out/pxtte_ms4as"][col[c]] + z["out/pxtte_ms4cs"][col[c]]
              for c in split_cells]
    assert np.allclose(totals, totals[0], rtol=1e-12)
