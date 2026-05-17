"""Dinosaur-backed implementation of the :class:`DynamicalCore` protocol.

This subpackage wraps the spectral primitive-equations dycore from the external
``dinosaur`` package. Modal↔nodal transforms, hyperdiffusion filters, and the
IMEX-RK SIL3 step all live here — outside this subpackage, the rest of
jax-gcm only sees the gridpoint :class:`PhysicsState` projection.
"""

from jcm.dycore.dinosaur.dycore import DinosaurDycore
from jcm.dycore.registry import register_dycore


@register_dycore("dinosaur")
def _build_dinosaur_dycore(**kwargs):
    """Factory registered under the ``"dinosaur"`` name in the dycore registry.

    Keyword arguments are forwarded straight through to
    :class:`DinosaurDycore` — the Hydra runner (Phase 4) is responsible for
    mapping ``cfg.dycore.<...>`` keys onto these kwargs.
    """
    return DinosaurDycore(**kwargs)


__all__ = ["DinosaurDycore"]
