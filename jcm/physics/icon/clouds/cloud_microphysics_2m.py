"""
Two-moment cloud microphysics scheme for ICON physics

This module implements a two-moment bulk cloud microphysics scheme, predicting both mass mixing ratios and number concentrations of hydrometeor species. 
The scheme represents warm, mixed-phase, and ice-phase cloud processes and their coupling to aerosols.

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

Key characteristics:
- Two-moment treatment allows explicit simulation of cloud particle size distributions
- Consistent coupling to aerosol microphysics via HAM #TODO
- Designed for global climate simulations (ECHAM6 / ICON heritage)
- Mixed-phase clouds treated using temperature-dependent ice fraction

Based on the ECHAM6/ICON microphysics as described in:
- Lohmann et al. (2007): Cloud microphysics and aerosol indirect effects in the global climate model ECHAM5-HAM
- Lohmann & Hoose (2009): Sensitivity studies of different aerosol indirect effects in mixed-phase clouds
- Lohmann & Neubauer (2018): The importance of mixed-phase and ice clouds for climate sensitivity in the global aerosolclimate model ECHAM6-HAM2
- Neubauer et al. (2019): The global aerosol–climate model ECHAM6.3–HAM2.3 – Part 2:  Cloud evaluation, aerosol radiative forcing, and climate sensitivity

Notes for developers:
- This scheme is diagnostic–prognostic hybrid: cloud fraction is diagnosed, while hydrometeor mass and number are prognosed.
- Numerical stability relies on sub-stepping and limiter functions for conversion rates.
- Thresholds for autoconversion and freezing depend on both hydrometeor number concentration and temperature.

Date: 2025-12-15
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
import tree_math

from ..constants.physical_constants import (
    cpd, grav, rgrav, rd, alv, als, rv, vtmpc1, vtmpc2, rhoh2o, ak, tmelt, p0s1_bg 
)