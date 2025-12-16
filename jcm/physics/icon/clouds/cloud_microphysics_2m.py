"""
Two-moment cloud microphysics scheme for ICON physics

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
- Consistent coupling to aerosol microphysics via HAM #TODO

Based on the ECHAM6/ICON microphysics as described in:
- Lohmann et al. (2007): Cloud microphysics and aerosol indirect effects in the global climate model ECHAM5-HAM
- Lohmann & Hoose (2009): Sensitivity studies of different aerosol indirect effects in mixed-phase clouds
- Lohmann & Neubauer (2018): The importance of mixed-phase and ice clouds for climate sensitivity in the global 
  aerosolclimate model ECHAM6-HAM2
- Neubauer et al. (2019): The global aerosol–climate model ECHAM6.3–HAM2.3 – Part 2:  Cloud evaluation, aerosol 
  radiative forcing, and climate sensitivity

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

from cloud_params import (
    cqtmin, cvtfall, crhosno, cn0s, ccwmin,
    cthomi,  clmax, clmin, jbmin, jbmax, lonacc,
    ccraut, ceffmin, ceffmax, crhoi, ccsaut
)

from cloud_utils import (epsec, xsec, qsec, eps, mi,
                               ri_vol_mean_1, ri_vol_mean_2,
                               alfased_1, alfased_2, alfased_3,
                               betased_1, betased_2, betased_3,
                               icemin,
                               cdi, mw0, mi0, mi0_rcp, ka, kb,
                               alpha, xmw, fall, rhoice, conv_effr2mvr, clc_min, icemax,
                               dw0, exm1_1, exp_1, exm1_2,
                               exp_2, pirho_rcp, cap, cons4, cons5,
                               fact_PK, pow_PK,
                               get_util_var, get_cloud_bounds, fact_coll_eff, fact_tke
)

@tree_math.struct
class MicrophysicsParameters:
    """Configuration parameters for cloud microphysics"""
    
    # Autoconversion parameters
    ccraut: float        # Critical cloud water for autoconversion (kg/kg)
    ccracl: float        # Accretion coefficient (cloud to rain)
    cauloc: float        # Cloud droplet dispersion parameter
    ceffmin: float       # Minimum cloud droplet radius (microns)
    ceffmax: float       # Maximum cloud droplet radius (microns)
    
    # Ice microphysics parameters
    cn0s: float          # Snow particle number density (1/m^3)
    crhosno: float       # Snow density (kg/m^3)
    cvtfall: float       # Terminal velocity factor for ice
    cthomi: float        # Homogeneous ice nucleation temperature (K)
    csecfrl: float       # Critical ice fraction for Bergeron-Findeisen
    
    # Collection efficiencies
    ccollec: float       # Collection efficiency rain/cloud
    ccollei: float       # Collection efficiency snow/ice
    
    # Time scale parameters
    tau_melt: float      # Melting time scale (s)
    tau_freeze: float    # Freezing time scale (s)
    
    # Evaporation/sublimation parameters
    cevaprain: float     # Rain evaporation coefficient
    cevapsnow: float     # Snow sublimation coefficient
    
    # Sedimentation parameters
    vt_ice: float        # Ice crystal fall speed (m/s)
    vt_snow_a: float     # Snow fall speed coefficient a
    vt_snow_b: float     # Snow fall speed exponent b
    vt_rain_a: float     # Rain fall speed coefficient a
    vt_rain_b: float     # Rain fall speed exponent b
    
    # Numerical parameters
    epsilon: float       # Small number for numerical stability
    dt_sedi: float       # Sub-timestep for sedimentation (s)

    @classmethod
    def default(cls, ccraut=5.0e-4, ccracl=6.0, cauloc=1.0, ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
                 crhosno=100.0, cvtfall=3.29, cthomi=233.15, csecfrl=0.1, ccollec=0.7,
                 ccollei=0.3, tau_melt=100.0, tau_freeze=100.0, cevaprain=1.0e-3,
                 cevapsnow=5.0e-4, vt_ice=0.1, vt_snow_a=8.8, vt_snow_b=0.15,
                 vt_rain_a=386.0, vt_rain_b=0.67, epsilon=1.0e-12, dt_sedi=10.0) -> 'MicrophysicsParameters':
        """Return default microphysics parameters"""
        return cls(
            ccraut=jnp.array(ccraut),
            ccracl=jnp.array(ccracl),
            cauloc=jnp.array(cauloc),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            cn0s=jnp.array(cn0s),
            crhosno=jnp.array(crhosno),
            cvtfall=jnp.array(cvtfall),
            cthomi=jnp.array(cthomi),
            csecfrl=jnp.array(csecfrl),
            ccollec=jnp.array(ccollec),
            ccollei=jnp.array(ccollei),
            tau_melt=jnp.array(tau_melt),
            tau_freeze=jnp.array(tau_freeze),
            cevaprain=jnp.array(cevaprain),
            cevapsnow=jnp.array(cevapsnow),
            vt_ice=jnp.array(vt_ice),
            vt_snow_a=jnp.array(vt_snow_a),
            vt_snow_b=jnp.array(vt_snow_b),
            vt_rain_a=jnp.array(vt_rain_a),
            vt_rain_b=jnp.array(vt_rain_b),
            epsilon=jnp.array(epsilon),
            dt_sedi=jnp.array(dt_sedi)
        )
    
