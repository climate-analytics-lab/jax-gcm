"""HAM's Mie tables must be built outside any trace (#1017 regression).

The tables used to be memoised on first use inside ``_mode_optics``; when that
first use happened inside the compiled model step, the process-global memo kept
a tracer and the next trace failed with ``UnexpectedTracerError``.
"""
import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.optics import ham_mie_tables


def test_memo_holds_numpy_and_survives_two_traces(monkeypatch):
    monkeypatch.setattr(ham_mie_tables, "_DEFAULT_TABLES", None)

    @jax.jit
    def first(x):
        tables = ham_mie_tables.default_ham_mie_tables()
        return x + jnp.sum(tables["lw_fine"].q_ext[0, 0, :2])

    first(jnp.float32(0.0))
    for lut in ham_mie_tables.default_ham_mie_tables().values():
        assert isinstance(lut.q_ext, np.ndarray)

    @jax.jit
    def second(x):
        return x + jnp.sum(ham_mie_tables.default_ham_mie_tables()["sw_fine"].q_ext[0, 0, :2])

    assert np.isfinite(float(second(jnp.float32(1.0))))


def test_lut_is_a_pytree_with_static_axes():
    lut = ham_mie_tables.default_ham_mie_tables()["sw_coarse"]
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.asarray, lut))
    assert len(leaves) == 3          # q_ext, ssa, g
    assert all(leaf.shape == lut.q_ext.shape for leaf in leaves)
