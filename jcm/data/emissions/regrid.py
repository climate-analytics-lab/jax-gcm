"""Backward-compatible alias — the implementation lives in
:mod:`jcm.data.regridding`, the shared host-side regridding module.
"""

from jcm.data.regridding import (Regridder, build_regridder,  # noqa: F401
                                 model_grid)
