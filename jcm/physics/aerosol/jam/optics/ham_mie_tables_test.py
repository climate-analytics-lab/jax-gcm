"""HAM Mie table parity against the compiled ``ham_rad_fitplus`` reference,

plus a direct ``mie.py`` spot check of the table content itself.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

from jcm.physics.aerosol.jam.optics.mie import mie_efficiencies
from jcm.physics.aerosol.jam.optics.ham_mie_tables import (
    HAM_TABLE_AXES,
    _GH_NODES,
    _GH_WEIGHTS,
    default_ham_mie_tables,
    ham_rad_fitplus,
)

REF = (Path(__file__).resolve().parents[4] / "data" / "test" / "echam_cloud_reference"
       / "hamrad_lookup.npz")
_NAME_OF_KTABLE = {1: "sw_fine", 2: "sw_coarse", 3: "lw_fine", 4: "lw_coarse"}


@pytest.fixture(scope="module")
def reference():
    with np.load(REF) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def tables():
    return default_ham_mie_tables()


@pytest.mark.parametrize("x64", [False, True])
def test_lookup_matches_compiled_ham_rad_fitplus(reference, tables, x64):
    """Every designed case (axis edges, out-of-range, half-step ties,

    interior, both sigma classes, SW and LW) must match the compiled
    ``ham_rad_fitplus`` run on these exact tables, index for index.
    """
    if x64:
        jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp
        n = len(reference["labels"])
        q = np.empty(n)
        s = np.empty(n)
        g = np.empty(n)
        for i in range(n):
            lut = tables[_NAME_OF_KTABLE[int(reference["ktable"][i])]]
            qi, si, gi = ham_rad_fitplus(
                lut, jnp.asarray(reference["x"][i]), jnp.asarray(reference["nr"][i]),
                jnp.asarray(reference["ni"][i]))
            q[i], s[i], g[i] = float(qi), float(si), float(gi)
        if x64:
            # Exact (to float64 rounding): the designed cases include exact
            # half-step ties in the NINT bucketing, which only agree with
            # the compiled routine's round-half-away-from-zero at full
            # precision.
            np.testing.assert_allclose(q, reference["q_ext"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(s, reference["ssa"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(g, reference["g"], rtol=0, atol=1e-12)
        else:
            # float32 (jcm's default physics precision): exact except at
            # the DESIGNED exact-half-step ties, where float32 rounding of
            # the scaled coordinate can cross the 0.5 threshold differently
            # than float64 and land on the adjacent table bin -- a bounded,
            # expected precision artifact of a measure-zero input, not a
            # correctness gap (see hamrad_provenance.json's own measured
            # 2.7% at exactly these cases vs 0 everywhere else).
            tie = np.array(["tie" in lbl for lbl in reference["labels"]])
            for actual, expected in ((q, reference["q_ext"]), (s, reference["ssa"]), (g, reference["g"])):
                np.testing.assert_allclose(actual[~tie], expected[~tie], rtol=1e-6, atol=1e-9)
                assert np.all(np.isfinite(actual[tie]))
    finally:
        if x64:
            jax.config.update("jax_enable_x64", False)


def test_out_of_range_cases_are_exactly_zero(reference):
    """The designed below-/above-range cases must be bit-exact zero, the

    native routine's "no answer" convention, not a clamp to the nearest
    table edge (mo_ham_rad.f90:1387-1389,1406-1410).
    """
    out_of_range = np.array(["below_range" in lbl or "above_range" in lbl
                             for lbl in reference["labels"]])
    assert out_of_range.sum() >= 12  # 3 axes x 4 tables (x/nr; x/ni covered)
    np.testing.assert_array_equal(reference["q_ext"][out_of_range], 0.0)
    np.testing.assert_array_equal(reference["ssa"][out_of_range], 0.0)
    np.testing.assert_array_equal(reference["g"][out_of_range], 0.0)


def test_lw_tables_carry_no_ssa_or_asymmetry(tables):
    for name, axes in HAM_TABLE_AXES.items():
        if not axes.sw:
            assert tables[name].ssa is None
            assert tables[name].g is None
        else:
            assert tables[name].ssa is not None
            assert tables[name].g is not None


@pytest.mark.parametrize("name", list(HAM_TABLE_AXES))
def test_table_matches_direct_lognormal_quadrature_at_spot_nodes(tables, name):
    """Independent spot check of the table CONTENT: at a handful of (x, nr,

    ni) nodes exactly on the table's own grid, the stored value must equal
    a fresh Gauss-Hermite lognormal quadrature over the scalar
    ``mie_efficiencies`` (not the batched grid evaluator the table builder
    itself uses) -- an end-to-end check from the physics kernel, independent
    of ``ham_mie_tables.py``'s internal batching.
    """
    import math

    axes = HAM_TABLE_AXES[name]
    lut = tables[name]
    rng = np.random.default_rng(0)
    for _ in range(4):
        ix = rng.integers(0, 101)
        ir = rng.integers(0, 101)
        ii = rng.integers(0, 201)
        lx = math.log(axes.x_min) + (math.log(axes.x_max) - math.log(axes.x_min)) * ix / 100.0
        x = math.exp(lx)
        nr = axes.nr_min + (axes.nr_max - axes.nr_min) * ir / 100.0
        lni = math.log(axes.ni_min) + (math.log(axes.ni_max) - math.log(axes.ni_min)) * ii / 200.0
        ni = math.exp(lni)

        ln_sigma = math.log(axes.sigma)
        sec = sec_scat = sec_gscat = 0.0
        for t_k, w_k in zip(_GH_NODES, _GH_WEIGHTS):
            growth = math.exp(math.sqrt(2.0) * ln_sigma * t_k)
            q_k, ssa_k, g_k = mie_efficiencies(x * growth, nr, ni)
            wgt = (w_k / math.sqrt(math.pi)) * growth ** 2
            sec += wgt * q_k
            sec_scat += wgt * q_k * ssa_k
            sec_gscat += wgt * q_k * ssa_k * g_k
        expected_q = sec * x * x / (4.0 * math.pi)

        np.testing.assert_allclose(float(lut.q_ext[ix, ir, ii]), expected_q, rtol=1e-4)
        if axes.sw:
            expected_ssa = sec_scat / sec if sec > 1e-300 else 0.0
            expected_g = sec_gscat / sec_scat if sec_scat > 1e-300 else 0.0
            np.testing.assert_allclose(float(lut.ssa[ix, ir, ii]), expected_ssa, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(float(lut.g[ix, ir, ii]), expected_g, rtol=1e-4, atol=1e-6)
