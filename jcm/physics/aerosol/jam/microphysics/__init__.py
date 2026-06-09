"""Interchangeable aerosol microphysics cores for the JAM harness.

``Mam4JaxMicrophysics`` is importable here, but it only pulls in the optional
GPL-3.0 ``mam4-jax`` dependency when an instance is *constructed*, not at
import — so a plain jcm import stays Apache-only.
"""

from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.mam4_data import (
    MAM4_SPEC,
)
from jcm.physics.aerosol.jam.microphysics.mam4_jax import Mam4JaxMicrophysics
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)

__all__ = [
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "Mam4JaxMicrophysics",
    "MAM4_SPEC",
]
