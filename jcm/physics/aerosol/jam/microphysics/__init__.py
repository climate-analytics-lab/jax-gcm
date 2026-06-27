"""Interchangeable aerosol microphysics cores for the JAM harness.

Cores whose adapter module imports an **optional/heavy dependency at import
time** (e.g. ``Mam4JaxMicrophysics``, which pulls the GPL-3.0 ``mam4-jax``) are
resolved **lazily** via module ``__getattr__``, so importing *this* package — or
the ``jam`` package above it — never triggers that import. Plain jcm imports
therefore stay Apache-only, and CI without the ``jcm[mam4]`` extra can still
collect every non-mam4 JAM test. The dependency is pulled in only when the core
is actually accessed (``from ...microphysics import Mam4JaxMicrophysics``,
construction, or an explicit import of its adapter module).

To add a microphysics core:

* **pure jcm** (no optional dep): import it eagerly below, like
  :class:`PlaceholderMicrophysics`.
* **wraps an optional dependency**: add one ``name -> module`` entry to
  ``_LAZY_CORES``. The ``__getattr__`` mechanism below never needs to change.
"""

import importlib

from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.mam4_data import (
    MAM4_SPEC,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)

#: Cores deferred to first access: attribute name -> module that defines it.
#: Listed here precisely because importing their module pulls an optional
#: dependency we don't want on the plain-import path.
_LAZY_CORES = {
    "Mam4JaxMicrophysics": "jcm.physics.aerosol.jam.microphysics.mam4_jax",
}

__all__ = [
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "MAM4_SPEC",
    *_LAZY_CORES,
]


def __getattr__(name):
    """Resolve a deferred core (PEP 562) on first access, keeping its import lazy."""
    module = _LAZY_CORES.get(name)
    if module is not None:
        return getattr(importlib.import_module(module), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
