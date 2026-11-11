"""Unique per-collective channel ids for multi-device CPU runs.

jax lowers every collective (``ppermute``, ``psum``, ``all_gather``,
``all_to_all``) with the single constant ``mlir.COLLECTIVE_CHANNEL_ID`` (== 1;
see each use in ``jax/_src/lax/parallel.py``). XLA:CPU's in-process
communicator keys its cross-device rendezvous on ``(run_id, devices, op_id)``
where ``op_id`` is that channel id, and its thunk executor runs independent
thunks concurrently. Two collective thunks that happen to be scheduled at the
same time therefore share one rendezvous and XLA aborts with
``rendezvous.h: Check failed: id < num_threads`` (or ``state.use_count() ==
num_threads``). The spectral transforms emit many independent collectives, so
this fires on most runs at >= 8 CPU devices. Serialising the scheduler
(``--xla_cpu_enable_concurrency_optimized_scheduler=false``) does not prevent
it. Giving each lowered collective its own channel id does.
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

    Replaces the ``mlir`` name inside ``jax._src.lax.parallel`` with a proxy
    whose ``COLLECTIVE_CHANNEL_ID`` returns a fresh integer on each access.
    The change takes effect at lowering time, so it only has to run before the
    model is first jit-compiled. Idempotent.

    Degrades safely if jax internals move: when the expected attributes are
    missing nothing is changed and one warning is logged. Never raises.

    Returns
    -------
    bool
        True if unique ids are in force (now or from an earlier call).

    """
    try:
        import jax._src.lax.parallel as parallel

        mlir = parallel.mlir
        if isinstance(mlir, _UniqueChannelMlir):
            return True
        mlir.COLLECTIVE_CHANNEL_ID  # noqa: B018 - existence probe
        parallel.mlir = _UniqueChannelMlir(mlir)
        return True
    except Exception:  # AttributeError/ImportError if jax internals moved
        logger.warning(
            "Could not install unique collective channel ids (jax internals "
            "changed); multi-CPU-device runs may abort with an XLA "
            "'rendezvous.h: Check failed' error."
        )
        return False
