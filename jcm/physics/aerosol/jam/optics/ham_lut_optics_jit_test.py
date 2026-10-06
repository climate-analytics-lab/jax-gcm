"""``HamLutOpticsTerm`` must build its tables outside any trace (#1017 regression).

Before #1017 W1, the tables were memoised by a process-global
``default_ham_mie_tables()`` called lazily on first use inside
``_mode_optics``; when that first use happened inside the compiled model
step, the process-global memo kept a tracer and the next trace failed with
``UnexpectedTracerError``. Tables now load (from HAM's authentic files, or
jcm's own built fallback -- ``default_ham_mie_tables()``, used explicitly
here via ``tables=`` so this test is deterministic regardless of
``HAM_INPUT_DIR``'s ambient state) once at ``HamLutOpticsTerm.__init__``,
which is what this module actually guards: construct outside any trace,
then call (and re-call under a second, independent trace) inside one.
"""
import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.optics.ham_lut_optics_term import HamLutOpticsTerm
from jcm.physics.aerosol.jam.optics.ham_mie_tables import default_ham_mie_tables


def test_tables_survive_two_independent_traces():
    term = HamLutOpticsTerm(tables=default_ham_mie_tables())

    @jax.jit
    def first(x):
        return x + jnp.sum(term._ham_tables["lw_fine"].q_ext[0, 0, :2])

    first(jnp.float32(0.0))
    for lut in term._ham_tables.values():
        assert isinstance(lut.q_ext, jnp.ndarray)

    @jax.jit
    def second(x):
        return x + jnp.sum(term._ham_tables["sw_fine"].q_ext[0, 0, :2])

    assert np.isfinite(float(second(jnp.float32(1.0))))


def test_lut_is_a_pytree_with_static_axes():
    term = HamLutOpticsTerm(tables=default_ham_mie_tables())
    lut = term._ham_tables["sw_coarse"]
    leaves = jax.tree_util.tree_leaves(lut)
    assert len(leaves) == 3          # q_ext, ssa, g
    assert all(leaf.shape == lut.q_ext.shape for leaf in leaves)
