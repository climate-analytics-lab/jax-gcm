"""Interchangeable aerosol microphysics cores for the HAM harness."""

from jcm.physics.aerosol.ham.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.ham.microphysics.mam4_data import (
    MAM4_JAX_COMMIT,
    MAM4_SPEC,
)
from jcm.physics.aerosol.ham.microphysics.placeholder import (
    PlaceholderMicrophysics,
)

__all__ = [
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "MAM4_SPEC",
    "MAM4_JAX_COMMIT",
]
