"""Two-moment cloud microphysics scheme for ECHAM physics

This module implements a two-moment bulk cloud microphysics scheme, predicting both mass mixing ratios and number 
concentrations of hydrometeor species. 
The scheme represents warm, mixed-phase, and ice-phase cloud processes and their coupling to aerosols.
Based on the mo_cloud_microphysics_2m module from ECHAM6/ICON.

Prognostic hydrometeors:
- Cloud liquid water (mass and number)
- Cloud ice (mass and number)
- Rain (mass and number)
- Snow (mass and number)

Represented processes include:
- Activation of cloud droplets from aerosols (aerosol–cloud coupling) # TODO
- Autoconversion of cloud water to rain
- Accretion of cloud droplets by rain
- Freezing of cloud droplets and rain
- Autoconversion of cloud ice to snow
- Aggregation of ice crystals
- Accretion of cloud ice by snow
- Melting of snow to rain
- Sedimentation of rain and snow
- Evaporation of rain and sublimation of snow
- Bergeron–Findeisen process (vapor deposition growth of ice at the expense of liquid)
- Temperature-dependent partitioning between liquid and ice phases

Planned features:
- Consistent coupling to aerosol microphysics via JAM #TODO

Based on the ECHAM6/ICON microphysics as described in:
- Lohmann et al. (2007): Cloud microphysics and aerosol indirect effects in the global climate model ECHAM5-JAM
- Lohmann & Hoose (2009): Sensitivity studies of different aerosol indirect effects in mixed-phase clouds
- Lohmann & Neubauer (2018): The importance of mixed-phase and ice clouds for climate sensitivity in the global 
  aerosolclimate model ECHAM6-HAM2
- Neubauer et al. (2019): The global aerosol–climate model ECHAM6.3–HAM2.3 – Part 2:  Cloud evaluation, aerosol 
  radiative forcing, and climate sensitivity

Date: 2025-12-15
"""

# The scheme was split from a single ~3700-line module into this package
# (types / sedimentation_melt / deposition_freezing / precip / assembly /
# scheme) on function boundaries only. Everything the old module exported
# is re-exported here so all existing import sites keep working unchanged
# (``from jcm.physics.clouds.lohmann_2m import ...``).

from .types import (
    MicrophysicsTendencies_2M,
    microphysics_dt_constants,
)
from .sedimentation_melt import (
    melting_snow_and_ice,
    sedimentation_ice,
)
from .deposition_freezing import (
    demott2010_inp,
    freezing_below_238K,
    het_mxphase_freezing,
    mixed_phase_deposition_and_corrections,
    WBF_process,
)
from .precip import (
    precip_formation_cold,
    precip_formation_warm,
    sublimation_snow_and_ice_evaporation_rain,
    update_precip_fluxes,
)
from .assembly import (
    diagnostics,
    update_in_cloud_water,
    update_tendencies_and_important_vars,
)
from .scheme import (
    Lohmann2MMicrophysics,
    cloud_microphysics_2m,
)

__all__ = [
    "MicrophysicsTendencies_2M",
    "microphysics_dt_constants",
    "melting_snow_and_ice",
    "sedimentation_ice",
    "demott2010_inp",
    "freezing_below_238K",
    "het_mxphase_freezing",
    "mixed_phase_deposition_and_corrections",
    "WBF_process",
    "precip_formation_cold",
    "precip_formation_warm",
    "sublimation_snow_and_ice_evaporation_rain",
    "update_precip_fluxes",
    "diagnostics",
    "update_in_cloud_water",
    "update_tendencies_and_important_vars",
    "Lohmann2MMicrophysics",
    "cloud_microphysics_2m",
]
