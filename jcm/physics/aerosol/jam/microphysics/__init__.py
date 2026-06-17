"""Interchangeable aerosol microphysics cores for the JAM harness.

``Mam4JaxMicrophysics`` is importable here (``from ...microphysics import
Mam4JaxMicrophysics``), but it is resolved **lazily** via module ``__getattr__``:
its adapter module imports the optional GPL-3.0 ``mam4-jax`` dependency at its
top, so importing *this* package — or the ``jam`` package above it — must not
trigger that. Plain jcm imports therefore stay Apache-only, and CI without the
``jcm[mam4]`` extra can still collect every non-mam4 JAM test. The GPL dependency
is pulled in only when ``Mam4JaxMicrophysics`` is actually accessed (construction
or an explicit import of the adapter module, e.g. from ``jam_terms``).
"""

from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.mam4_data import (
    MAM4_SPEC,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)

__all__ = [
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "Mam4JaxMicrophysics",
    "MAM4_SPEC",
]


def __getattr__(name):
    """Lazily resolve ``Mam4JaxMicrophysics`` (PEP 562) to defer the GPL import."""
    if name == "Mam4JaxMicrophysics":
        from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
            Mam4JaxMicrophysics,
        )
        return Mam4JaxMicrophysics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
