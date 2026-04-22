"""Aerosol parameterizations.

Currently provides MACv2-SP (Stevens et al. 2017) simple plume aerosol,
as used in ICON.
"""

from .macv2_sp_params import AerosolParameters
from .macv2_sp import get_simple_aerosol

__all__ = [
    'AerosolParameters',
    'get_simple_aerosol',
]
