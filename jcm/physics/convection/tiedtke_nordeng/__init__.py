"""Tiedtke-Nordeng mass-flux convection scheme (deep/shallow/mid-level).

Only the API consumed via this package is re-exported (the composable
physics term and its parameter struct); import scheme internals from
their submodules (``.tiedtke_nordeng``, ``.updraft``, ``.downdraft``,
``.adjustment``, ``.flux_tendencies``) directly so grep-for-callers
stays meaningful.
"""

from .tiedtke_nordeng import (
    ConvectionParameters,
    TiedtkeConvection,
)

__all__ = [
    "ConvectionParameters",
    "TiedtkeConvection",
]
