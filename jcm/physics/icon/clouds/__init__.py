"""Cloud physics parameterizations for ICON

This module contains cloud microphysics and cloud cover schemes including:
- Shallow cloud scheme with diagnostic cloud fraction
- Cloud water/ice partitioning  
- Basic condensation/evaporation processes
- Comprehensive cloud microphysics with precipitation
"""

from .shallow_clouds import (
    shallow_cloud_scheme,
    CloudParameters,
    CloudState,
    CloudTendencies,
    calculate_cloud_fraction,
    partition_cloud_phase,
    saturation_specific_humidity
)

from .cloud_microphysics import (
    cloud_microphysics,
    MicrophysicsParameters,
    MicrophysicsState,
    MicrophysicsTendencies,
    autoconversion_kk2000,
    accretion_rain_cloud,
    melting_freezing,
    evaporation_sublimation
)

from .cloud_params_2m import ( #TODO move to cloud_microphysics_2m.py
    CloudParams2M
)

from .cloud_microphysics_2m import (
    cloud_microphysics_2m,
    MicrophysicsState_2M,
    MicrophysicsTendencies_2M,
)

__all__ = [
    # Shallow clouds
    "shallow_cloud_scheme",
    "CloudParameters",
    "CloudState", 
    "CloudTendencies",
    "calculate_cloud_fraction",
    "partition_cloud_phase",
    "saturation_specific_humidity",
    # Microphysics
    "cloud_microphysics",
    "MicrophysicsParameters",
    "MicrophysicsState",
    "MicrophysicsTendencies",
    "autoconversion_kk2000",
    "accretion_rain_cloud",
    "melting_freezing",
    "evaporation_sublimation",
    # Cloud parameters for 2m
    "CloudParams2M",
    # Cloud microphysics for 2m
    "cloud_microphysics_2m",
    "MicrophysicsState_2M",
    "MicrophysicsTendencies_2M"
]