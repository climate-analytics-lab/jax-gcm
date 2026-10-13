"""pySES CAM-SE spectral-element dynamical-core backend.

Registers :class:`PysesCamSEDycore` under the dycore-registry name
``"pyses_cam_se"``. The heavy ``pyses`` dependency (``pip install
jcm[pyses]``) is imported lazily at dycore *construction* time, so importing
this package (and hence populating the registry) is always safe; a missing
pyses surfaces as a clear :class:`ImportError` from the constructor.

See :mod:`jcm.dycore.pyses.dycore` for the architecture and precision
contract, :mod:`jcm.dycore.pyses.physics_grid` for the pg2 physics grid, and
:mod:`jcm.dycore.pyses.forcing` for the prescribed-forcing builder.
"""

from jcm.dycore.pyses.coords import (
    EchamSECoords,
    SEHorizontalGrid,
    full_echam_hybrid,
)
from jcm.dycore.pyses.dycore import PysesCamSEDycore, cast_to_physics_dtype
from jcm.dycore.pyses.forcing import build_forcing
from jcm.dycore.pyses.physics_grid import FVPhysicsGrid
from jcm.dycore.registry import register_dycore


@register_dycore("pyses_cam_se")
def _build_pyses_cam_se(**kwargs) -> PysesCamSEDycore:
    """Registry factory for the pyses CAM-SE backend."""
    return PysesCamSEDycore(**kwargs)


__all__ = [
    "PysesCamSEDycore",
    "cast_to_physics_dtype",
    "FVPhysicsGrid",
    "EchamSECoords",
    "SEHorizontalGrid",
    "full_echam_hybrid",
    "build_forcing",
]
