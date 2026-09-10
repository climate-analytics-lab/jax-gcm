"""CAM spectral non-orographic gravity-wave drag (frontal source).

Faithful JAX port of CAM's ``gw_common.F90`` solver and ``gw_front.F90``
frontogenesis-triggered source (ESCOMP/CAM, ref ``cam_cesm2_2_rel``),
plus a lat-lon frontogenesis-function provider. See
``docs/source/design/frontal_gravity_wave_drag.md``.
"""

from jcm.physics.gravity_waves.spectral.frontal import (
    CMSource,
    flat_spectrum,
    gaussian_spectrum,
    gw_cm_src,
)
from jcm.physics.gravity_waves.spectral.frontogenesis import (
    frontogenesis_function,
)
from jcm.physics.gravity_waves.spectral.params import (
    FrontalGWParameters,
    SpectrumShape,
)
from jcm.physics.gravity_waves.spectral.solver import (
    GWBand,
    GWDragResult,
    gw_drag_prof,
    gw_prof,
)
from jcm.physics.gravity_waves.spectral.term import FrontalGravityWaveDrag

__all__ = [
    "CMSource",
    "FrontalGWParameters",
    "FrontalGravityWaveDrag",
    "GWBand",
    "GWDragResult",
    "SpectrumShape",
    "flat_spectrum",
    "frontogenesis_function",
    "gaussian_spectrum",
    "gw_cm_src",
    "gw_drag_prof",
    "gw_prof",
]
