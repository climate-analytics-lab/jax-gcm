"""ICON-derived grey two-stream radiation scheme.

A drastic simplification of the original ICON radiation using grey-band
two-stream shortwave and longwave transfer. See `radiation_scheme.py`
for the main entry point.
"""

from .radiation_scheme import (
    radiation_scheme,
    prepare_radiation_state,
)

__all__ = [
    'radiation_scheme',
    'prepare_radiation_state',
]
