"""Tests for the profiler scope labels.

The load-bearing property is not that ``scope()`` returns a context manager,
but that the label it opens still exists in the *optimized* HLO — after
inlining, fusion and DCE have run. If XLA ever stopped propagating ``op_name``
metadata, ``tools/profile_terms.py`` would keep producing a report, with every
kernel silently falling into the ``unattributed`` bucket. That is what
``test_scope_survives_compilation`` is here to catch.
"""

import jax
import jax.numpy as jnp

from jcm import profiling


def test_scope_prefixes_the_label():
    with profiling.scope("my_term"):
        pass  # the context manager is a no-op outside tracing
    assert profiling.SCOPE_PREFIX == "jcm:"


def test_label_from_op_name_finds_a_scope():
    assert profiling.label_from_op_name(
        "jit(step)/jcm:dynamics/add"
    ) == "dynamics"


def test_label_from_op_name_ignores_foreign_scopes():
    """Scopes JAX and flax open themselves must not be mistaken for labels."""
    assert profiling.label_from_op_name("jit(step)/while/body/mul") is None
    assert profiling.label_from_op_name("") is None


def test_label_from_op_name_takes_the_innermost():
    """A term nested in the enclosing physics scope is charged to the term."""
    assert profiling.label_from_op_name(
        "jit(step)/jcm:bridge_to_dynamics/checkpoint/jcm:hines_gwd/mul"
    ) == "hines_gwd"


def test_component_labels_are_distinct():
    labels = {profiling.DYNAMICS, profiling.BRIDGE_TO_PHYSICS,
              profiling.BRIDGE_TO_DYNAMICS}
    assert len(labels) == 3


def test_scoped_labels_a_callable():
    """``scoped`` must accept callables without ``__name__``, e.g. partials."""
    import functools

    fn = functools.partial(lambda x, k: jnp.sin(x) * k, k=2.0)
    labelled = profiling.scoped(fn, "gamma")

    text = jax.jit(labelled).lower(jnp.ones(4)).compile().as_text()
    assert "jcm:gamma" in text
    # And it is transparent: same values as the unwrapped callable.
    assert jnp.allclose(labelled(jnp.ones(4)), fn(jnp.ones(4)))


def test_scope_survives_compilation():
    """A scope opened while tracing is still readable in the optimized HLO."""
    def f(x):
        with profiling.scope("alpha"):
            y = jnp.sin(x) * 2.0
        with profiling.scope("beta"):
            return jnp.cos(y) + 1.0

    text = jax.jit(f).lower(jnp.ones((8, 8))).compile().as_text()
    labels = set()
    for line in text.splitlines():
        if 'op_name="' in line:
            op_name = line.split('op_name="')[1].split('"')[0]
            label = profiling.label_from_op_name(op_name)
            if label is not None:
                labels.add(label)
    assert {"alpha", "beta"} <= labels
