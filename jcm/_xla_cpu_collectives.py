"""Unique per-collective channel ids for multi-device CPU runs.

jax lowers every collective (``ppermute``, ``psum``, ``all_gather``,
``all_to_all``, ...) with the single constant ``mlir.COLLECTIVE_CHANNEL_ID``
(== 1; see each use in ``jax/_src/lax/parallel.py``). XLA:CPU's in-process
communicator keys its cross-device rendezvous on ``(run_id, devices, op_id)``
where ``op_id`` is that channel id, and its thunk executor runs independent
thunks concurrently. Two collective thunks that happen to be scheduled at the
same time therefore share one rendezvous and XLA aborts with
``rendezvous.h: Check failed: id < num_threads`` (or ``state.use_count() ==
num_threads``), or stalls until its collective timeout. The spectral
transforms emit hundreds of independent collectives per step, so this fires on
most ECHAM runs at >= 8 CPU devices. Serialising the scheduler
(``--xla_cpu_enable_concurrency_optimized_scheduler=false``) does not prevent
it.

Two pieces make every collective *equation* carry its own id:

* the ``mlir`` name inside ``jax._src.lax.parallel`` is a proxy whose
  ``COLLECTIVE_CHANNEL_ID`` draws a fresh integer on each read;
* every collective primitive is marked uncacheable. jax otherwise caches a
  lowering by ``(primitive, input avals, params)`` and reuses it, so the id
  would be drawn once per *distinct* collective: the ECHAM T63 dynamics step
  has 492 ``collective_permute`` ops and only 12 distinct cache keys.

The counter is process-wide and never restarts. XLA:CPU gives every program
on the same devices the same ``run_id``, and two programs do run at once (a
chunk's output conversion on the I/O thread overlaps the next chunk's
integration), so ids must not repeat across modules either. The price is that
a module's text, the key of jax's persistent compilation cache, depends on how
many collectives the process lowered before it: a sharded CPU program is
recompiled when that history differs (a resumed run against a fresh one).

What remains shared: a function jax lowers once and calls from several sites (a
nested ``jit``) repeats its ids at each site. The sharded dinosaur step has two
such functions with collectives inside (``uv_nodal_to_vor_div_modal`` and
``vor_div_to_uv_nodal``, two call sites each); the 528 collective ops in its
lowered text at ECHAM T63 ``[4,4,1]`` all have distinct ids.
"""

from __future__ import annotations

import itertools
import logging

logger = logging.getLogger(__name__)

# jax's host-callback channels count up from 2 and must stay below 65536, so
# starting here keeps the two id ranges disjoint by construction.
_FIRST_COLLECTIVE_CHANNEL_ID = 1 << 16


class _UniqueChannelMlir:
    """Proxy for ``jax._src.interpreters.mlir`` that hands out fresh channel ids.

    Only ``COLLECTIVE_CHANNEL_ID`` is intercepted; every other attribute is
    forwarded to the real module.
    """

    def __init__(self, real):
        self._real = real
        self._ids = itertools.count(_FIRST_COLLECTIVE_CHANNEL_ID)

    def __getattr__(self, name):
        if name == "COLLECTIVE_CHANNEL_ID":
            return next(self._ids)
        return getattr(self._real, name)


def ensure_unique_collective_channel_ids() -> bool:
    """Make every collective lowered from now on carry its own channel id.

    Installs the two pieces described in the module docstring. The change
    takes effect at lowering time, so it only has to run before the model is
    first jit-compiled. Idempotent.

    Degrades safely if jax internals move: when the expected attributes are
    missing nothing is changed and one warning is logged. Never raises.

    Returns
    -------
    bool
        True if unique ids are in force (now or from an earlier call).

    """
    try:
        import jax._src.lax.parallel as parallel
        from jax._src import core

        mlir = parallel.mlir
        if isinstance(mlir, _UniqueChannelMlir):
            return True
        mlir.COLLECTIVE_CHANNEL_ID  # noqa: B018 - existence probe
        uncacheable = mlir._uncacheable_primitives
        collectives = [
            v for v in vars(parallel).values() if isinstance(v, core.Primitive)
        ]
        if not collectives:
            raise AttributeError("no collective primitives found")
    except Exception:  # AttributeError/ImportError if jax internals moved
        logger.warning(
            "Could not install unique collective channel ids (jax internals "
            "changed); multi-CPU-device runs may abort with an XLA "
            "'rendezvous.h: Check failed' error."
        )
        return False

    uncacheable.update(collectives)
    parallel.mlir = _UniqueChannelMlir(mlir)
    return True
