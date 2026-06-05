"""Interchangeable aerosol microphysics cores for the JAM harness."""

from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.mam4_data import (
    MAM4_JAX_COMMIT,
    MAM4_SPEC,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)

__all__ = [
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "MAM4_SPEC",
    "MAM4_JAX_COMMIT",
]
