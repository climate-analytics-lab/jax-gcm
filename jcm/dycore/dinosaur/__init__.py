"""Dinosaur-backed implementation of the :class:`DynamicalCore` protocol.

This subpackage wraps the spectral primitive-equations dycore from the external
``dinosaur`` package. Modal↔nodal transforms, hyperdiffusion filters, and the
time step — by default the two-time-level semi-Lagrangian semi-implicit
Crank–Nicolson RK2 step (``semi_lagrangian_crank_nicolson_rk2``, off-centred by
:data:`~jcm.dycore.dinosaur.dycore.DEFAULT_OFF_CENTERING`), or the Eulerian
IMEX-RK SIL3 step for tracer-free physics that asks for it (SPEEDY) — all live here —
outside this subpackage, the rest of jax-gcm only sees the gridpoint
:class:`PhysicsState` projection.
"""

from jcm.dycore.dinosaur.dot_precision import apply_dot_precision_workaround
from jcm.dycore.dinosaur.dycore import DinosaurDycore
from jcm.dycore.registry import register_dycore

# Applied at import, before anything dinosaur can be traced (it reads the dot
# algorithm at trace time); see jcm.dycore.dinosaur.dot_precision.
apply_dot_precision_workaround()


@register_dycore("dinosaur")
def _build_dinosaur_dycore(**kwargs):
    """Build a :class:`DinosaurDycore` from registry kwargs.

    Registered under the ``"dinosaur"`` name. Keyword arguments are forwarded
    straight through; the Hydra runner (Phase 4) maps ``cfg.dycore.<...>``
    keys onto these kwargs.
    """
    return DinosaurDycore(**kwargs)


__all__ = ["DinosaurDycore"]
