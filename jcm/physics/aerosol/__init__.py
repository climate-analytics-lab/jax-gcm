"""Aerosol parameterizations.

Currently provides MACv2-SP (Stevens et al. 2017) simple plume aerosol,
as used in ICON.
"""

from .macv2_sp_params import AerosolParameters
from .macv2_sp import Macv2SpAerosol, get_simple_aerosol
from .macv2_natural import Macv2NaturalAerosol

__all__ = [
    'AerosolParameters',
    'Macv2SpAerosol',
    'Macv2NaturalAerosol',
    'get_simple_aerosol',
]
