"""Trace-time scope labels that make a compiled step attributable per component.

A jitted JCM step is one XLA module: dynamics, the spectral/gridpoint bridge and
every physics term are fused into a single stream of kernels with no runtime
boundary between them, so a wall-clock timer cannot say where the time went.
What survives compilation is instruction *metadata*: a
:func:`jax.named_scope` entered while tracing is recorded as an ``op_name``
prefix on every HLO instruction produced inside it, and it stays there through
optimization. A profiler trace reports the HLO instruction behind each kernel,
so joining the two attributes measured device time back to the component that
emitted it. ``tools/profile_terms.py`` is that join.

The labels are unconditional rather than gated behind a debug flag: a named
scope is trace-time-only metadata that XLA does not optimize on, so it costs
nothing at runtime, and leaving it always on means the profiled build is the
production build rather than a near-copy of it.

Scopes are emitted at exactly two places, both of them drivers rather than
physics:

- :mod:`jcm.model`'s operator-split step, for the dynamical core and the two
  bridge directions;
- :class:`jcm.physics.composable_physics.ComposablePhysics`'s term loops, one
  scope per :class:`~jcm.physics.physics_term.PhysicsTerm`.

Labelling the loops rather than the terms is deliberate. ``PhysicsTerm.__call__``
is overridden by every one of the ~60 concrete terms, so a base-class wrapper
would not run; the loop is the one place that sees all of them, and it keeps the
schemes themselves free of profiling code.
"""

from __future__ import annotations

import jax

#: Marks a scope as one of ours. Compiled ``op_name`` metadata interleaves JCM
#: scopes with names JAX and flax introduce on their own (``jit(...)``,
#: transpose/remat wrappers, module paths), so the parser needs to recognise a
#: label rather than trust its position in the path. The colon cannot occur in
#: a Python identifier, hence never in a term name.
SCOPE_PREFIX = "jcm:"

#: Label for the dynamical core's advance, ``DynamicalCore.step`` — which also
#: carries the forward-Euler tendency add and the spectral filters.
DYNAMICS = "dynamics"

#: Label for the dynamics-to-physics projection, ``to_physics_state`` plus any
#: dycore-supplied diagnostic fields: the spectral-to-gridpoint transforms and
#: tracer cleaning that precede the physics.
BRIDGE_TO_PHYSICS = "bridge_to_physics"

#: Label for the physics call's own overhead: forcing time selection,
#: state/tendency verification, the gridpoint-to-column reshapes and the
#: tendency accumulation that wrap the term loop. This scope ENCLOSES the
#: per-term scopes, and the innermost-wins rule in :func:`label_from_op_name`
#: is what leaves it holding the coupling overhead alone.
BRIDGE_TO_DYNAMICS = "bridge_to_dynamics"


def scope(label: str):
    """Return a :func:`jax.named_scope` context manager tagged for the profiler.

    Parameters
    ----------
    label
        Component name, e.g. :data:`DYNAMICS` or a
        :class:`~jcm.physics.physics_term.PhysicsTerm`'s ``name``.

    Returns
    -------
    context manager
        Entering it prefixes ``jcm:<label>`` onto the ``op_name`` metadata of
        every HLO instruction traced inside.

    """
    return jax.named_scope(f"{SCOPE_PREFIX}{label}")


def scoped(fn, label: str):
    """Wrap a callable so every trace of it runs inside :func:`scope`.

    Parameters
    ----------
    fn
        Any traceable callable, including a ``functools.partial`` or a
        ``jax.checkpoint``-wrapped term (which have no ``__name__``, so this
        deliberately does not use ``functools.wraps``).
    label
        Component name, as for :func:`scope`.

    Returns
    -------
    callable
        ``fn`` with the scope applied.

    Notes
    -----
    Preferred over an inline ``with`` block at the call site when the callable
    is invoked from more than one branch: labelling the function once keeps the
    branches themselves unindented and unmodified.

    """
    def wrapper(*args, **kwargs):
        with scope(label):
            return fn(*args, **kwargs)
    return wrapper


def label_from_op_name(op_name: str) -> str | None:
    """Extract the JCM component label from an HLO ``op_name``, if any.

    Parameters
    ----------
    op_name
        The ``metadata={op_name="..."}`` string of an HLO instruction, e.g.
        ``jit(_run_from_state)/scan/jcm:tiedtke_convection/mul``.

    Returns
    -------
    str or None
        The label of the INNERMOST JCM scope on the path, or ``None`` if the
        instruction was not traced inside one.

    Notes
    -----
    Innermost wins because the scopes nest: :data:`BRIDGE_TO_DYNAMICS` encloses
    the whole physics call, and each term scope sits inside it. Charging an
    instruction to the deepest scope containing it therefore bills a term's work
    to the term and leaves the enclosing scope holding only the glue between
    terms, which is the split we want. Scopes that JAX and flax open on their
    own lack the prefix and are skipped.

    """
    label = None
    for part in op_name.split("/"):
        if part.startswith(SCOPE_PREFIX):
            label = part[len(SCOPE_PREFIX):]
    return label
