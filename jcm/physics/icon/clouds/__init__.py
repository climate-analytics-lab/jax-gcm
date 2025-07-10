"""
Cloud physics parameterizations for ICON

This module contains cloud microphysics and cloud cover schemes including:
- Shallow cloud scheme with diagnostic cloud fraction
- Cloud water/ice partitioning  
- Basic condensation/evaporation processes
"""

from .shallow_clouds import (
    shallow_cloud_scheme,
    CloudConfig,
    CloudState,
    CloudTendencies,
    calculate_cloud_fraction,
    partition_cloud_phase,
    saturation_specific_humidity
)

__all__ = [
    "shallow_cloud_scheme",
    "CloudConfig",
    "CloudState", 
    "CloudTendencies",
    "calculate_cloud_fraction",
    "partition_cloud_phase",
    "saturation_specific_humidity"
]