"""``AqueousSulfur`` (M7 path) against the compiled ECHAM6.3-HAM2.3 routine.

``jcm/data/test/echam_cloud_reference/hamaqueous_M7.npz`` holds the outputs
of the UNMODIFIED r7492 ``mo_ham_chemistry.f90::ham_wet_chemistry`` on
designed single-level M7 cells (``hamaqueous_README.md``,
``hamaqueous_provenance.json``). jax-gcm#1017 task 3 found that reproducing
it required :data:`AqueousConstants.HAM_AQUEOUS_CONSTANTS` — every HAM
literal ``_aqueous_so4`` otherwise reads from jcm's own constants/species
tables (the SO2 Henry's-law pair, the gas constant, Avogadro's number and
its separate ``xtoc``/``ctox`` rounding, SO2's molar mass) differs from
r7492's own value; the SO2 Henry pair specifically is a genuine MAM4-shared
defect (jax-gcm#1031) that stays unfixed on the MAM4 default path (``spec=
M7_SPEC`` opts in to the HAM-correct values instead; MAM4 is unaffected —
see ``aqueous_test.py``/the bitid probe).

``pxtte_*`` in the reference is already a GRID-MEAN tendency [kg/kg/s] (the
Fortran receives ``paclc`` and weights internally), the same convention
``AqueousSulfur.__call__`` produces (``so4_rate = cloud_fraction*dso4/dt``) —
so no extra ``paclc`` weighting is needed on jcm's side here, unlike the
scratch comparison that first found the SO2-Henry discrepancy (which called
the bare :func:`_aqueous_so4` kernel, an IN-CLOUD rate).

``in/pmlwc``/``in/paclc`` are fed to the Fortran directly as the in-cloud
liquid water and cloud fraction; ``AqueousSulfur`` instead derives in-cloud
LWC from a GRID-MEAN ``clouds.qc`` (``qc/cloud_fraction``), so the cells
here set ``qc = pmlwc*paclc`` to recover ``pmlwc`` through that division
(exact for every cell but ``no_cloud``, where ``paclc=pmlwc=0`` and the
cell is gated off before LWC matters; the float64 round-trip elsewhere is
within the 1e-12 tolerance below).
"""
from __future__ import annotations

import functools
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.chemistry.aqueous import AqueousSulfur
from jcm.physics.aerosol.jam.chemistry.oxidants import OxidantField
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics_interface import PhysicsState

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "hamaqueous_M7.npz")
RTOL = 1e-12


@functools.lru_cache(maxsize=None)
def load():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


def run_jcm():
    z = load()
    n = len(z["meta/names"])
    dtype = jnp.float64

    def g(k):
        return jnp.asarray(z[k], dtype)

    tracers = {
        "g_so2": g("in/so2"),
        mass_name("so4", "as"): g("in/ms4as"),
        mass_name("so4", "cs"): g("in/ms4cs"),
        number_name("as"): g("in/nas"),
        number_name("cs"): g("in/ncs"),
    }
    state = PhysicsState.zeros((n,)).copy(
        temperature=g("in/tm1"), tracers=tracers,
    )
    ox = OxidantField(
        oh=jnp.zeros(n, dtype), no3=jnp.zeros(n, dtype),
        o3=g("in/o3_density"), h2o2=g("in/h2o2_density"),
    )
    paclc = g("in/paclc")
    diagnostics = {
        "oxidants": ox,
        "clouds": types.SimpleNamespace(
            cloud_fraction=paclc, qc=g("in/pmlwc") * paclc,
        ),
        "air_density": g("in/rhop1"),
        "_dt_seconds": float(z["in/time_step_len"]),
    }
    term = AqueousSulfur(spec=M7_SPEC)
    tend, _ = term(state, diagnostics, None, None)
    return tend


def test_aqueous_sulfur_m7_matches_ham_wet_chemistry():
    """Every recorded field, every cell, at float64 round-off.

    Covers the full branch structure HAM's number-fraction split has
    (both-present/only-AS/only-CS/neither-present, the ``no_cloud`` LWC
    gate, warm/cold, and every oxidant-limited regime) — see
    ``hamaqueous_README.md``'s cell table.
    """
    with jax.enable_x64(True):
        tend = run_jcm()
        z = load()
        n = len(z["meta/names"])
        zeros = np.zeros(n)

        np.testing.assert_allclose(
            np.asarray(tend.tracers["g_so2"], np.float64),
            z["out/pxtte_so2"], rtol=RTOL, atol=0.0, err_msg="g_so2",
        )
        np.testing.assert_allclose(
            np.asarray(tend.tracers[mass_name("so4", "as")], np.float64),
            z["out/pxtte_ms4as"], rtol=RTOL, atol=0.0, err_msg="m_so4_as",
        )
        np.testing.assert_allclose(
            np.asarray(tend.tracers[mass_name("so4", "cs")], np.float64),
            z["out/pxtte_ms4cs"], rtol=RTOL, atol=0.0, err_msg="m_so4_cs",
        )
        # HAM never adjusts the AS number tracer in ham_wet_chemistry; jcm
        # correspondingly never emits an "n_as" tendency key.
        np.testing.assert_allclose(
            np.asarray(tend.tracers.get(number_name("as"), zeros), np.float64),
            z["out/pxtte_nas"], rtol=RTOL, atol=0.0, err_msg="n_as",
        )
        # Nonzero only in the two "neither present" cells (as_below_cs_below,
        # both_empty), where HAM seeds new CS number from the produced mass.
        np.testing.assert_allclose(
            np.asarray(tend.tracers.get(number_name("cs"), zeros), np.float64),
            z["out/pxtte_ncs"], rtol=RTOL, atol=0.0, err_msg="n_cs",
        )


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
