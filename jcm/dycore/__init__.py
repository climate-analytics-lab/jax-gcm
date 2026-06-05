"""Pluggable dynamical-core backends for jax-gcm.

The :class:`DynamicalCore` protocol in :mod:`jcm.dycore.base` defines the contract
between :class:`jcm.model.Model` and a specific dycore implementation. Backends live
under this package (``dinosaur``, ``pyses``, ...) and register themselves with the
factory registry in :mod:`jcm.dycore.registry`.

The boundary between physics and dynamics is **gridpoint on both sides** on the
dycore's own native horizontal layout. Physics arrays are shaped
``(nlev, *horizontal_shape)`` where ``horizontal_shape`` comes from the dycore — a
regular ``(nlon, nlat)`` for the dinosaur spectral backend, ``(nelem, gll, gll)`` for a
cubed-sphere spectral-element backend, etc. There is no horizontal regrid at the
physics-dynamics seam; lat/lon regridding (when it happens) is a dycore-internal step
on the way to xarray output.

See ``docs/source/design.rst`` for the architecture overview.
"""

from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.registry import register_dycore, build_dycore, list_dycores

__all__ = [
    "DynamicalCore",
    "Predictions",
    "register_dycore",
    "build_dycore",
    "list_dycores",
]
