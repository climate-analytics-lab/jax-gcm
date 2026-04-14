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
from jax import jit
from typing import NamedTuple, Tuple, Optional
import tree_math
from math import pi

from ..constants.physical_constants import (
    cpd, grav, rgrav, rd, alv, als, rv, vtmpc1, vtmpc2, rhoh2o, ak, tmelt, p0s1_bg, alhs, alhc, t0 
)

from .cloud_params_2m import (
    CloudParams2M,
    cqtmin, cvtfall, crhosno, cn0s, ccwmin,
    cthomi,  clmax, clmin, jbmin, jbmax, lonacc,
    ccraut, ceffmin, ceffmax, crhoi, ccsaut, epsec, xsec, qsec, eps, mi,
    ri_vol_mean_1, ri_vol_mean_2,
    alfased_1, alfased_2, alfased_3,
    betased_1, betased_2, betased_3,
    icemin,
    cdi, mw0, mi0, mi0_rcp, ka, kb,
    alpha, xmw, fall, rhoice, conv_effr2mvr, clc_min, icemax,
    dw0, exm1_1, exp_1, exm1_2,
    exp_2, pirho_rcp, cap, cons4, cons5,
    fact_PK, pow_PK, cdnc_min_lower, cdnc_min_upper,
    rcd_vol_max, ldyn_cdnc_min, cdnc_min_fixed, nic_cirrus,
    fact_coll_eff, fact_tke
)

from .cloud_utils import (get_util_var, get_cloud_bounds, eff_ice_crystal_radius, minimum_CDNC,
                          consistency_number_to_mass, gridbox_frac_falling_hydrometeor, threshold_vert_vel, 
                          breadth_factor, effective_2_volmean_radius_param_Schuman_2011
)

# @tree_math.struct
# class MicrophysicsParameters_2M:
#     """Configuration parameters for cloud microphysics"""
    
#     # Autoconversion parameters
#     ccraut: float        # Critical cloud water for autoconversion (kg/kg)
#     ccracl: float        # Accretion coefficient (cloud to rain)
#     cauloc: float        # Cloud droplet dispersion parameter
#     ceffmin: float       # Minimum cloud droplet radius (microns)
#     ceffmax: float       # Maximum cloud droplet radius (microns)
    
#     # Ice microphysics parameters
#     cn0s: float          # Snow particle number density (1/m^3)
#     crhosno: float       # Snow density (kg/m^3)
#     cvtfall: float       # Terminal velocity factor for ice
#     cthomi: float        # Homogeneous ice nucleation temperature (K)
#     csecfrl: float       # Critical ice fraction for Bergeron-Findeisen
    
#     # Collection efficiencies
#     ccollec: float       # Collection efficiency rain/cloud
#     ccollei: float       # Collection efficiency snow/ice
    
#     # Time scale parameters
#     tau_melt: float      # Melting time scale (s)
#     tau_freeze: float    # Freezing time scale (s)
    
#     # Evaporation/sublimation parameters
#     cevaprain: float     # Rain evaporation coefficient
#     cevapsnow: float     # Snow sublimation coefficient
    
#     # Sedimentation parameters
#     vt_ice: float        # Ice crystal fall speed (m/s)
#     vt_snow_a: float     # Snow fall speed coefficient a
#     vt_snow_b: float     # Snow fall speed exponent b
#     vt_rain_a: float     # Rain fall speed coefficient a
#     vt_rain_b: float     # Rain fall speed exponent b
    
#     # Numerical parameters
#     epsilon: float       # Small number for numerical stability
#     dt_sedi: float       # Sub-timestep for sedimentation (s)

#     # Exponents for autoconversion
#     exm1_1: float
#     exp_1: float
#     exm1_2: float
#     exp_2: float

#     @classmethod
#     def default(cls, ccraut=5.0e-4, ccracl=6.0, cauloc=1.0, ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
#                  crhosno=100.0, cvtfall=3.29, cthomi=233.15, csecfrl=0.1, ccollec=0.7,
#                  ccollei=0.3, tau_melt=100.0, tau_freeze=100.0, cevaprain=1.0e-3,
#                  cevapsnow=5.0e-4, vt_ice=0.1, vt_snow_a=8.8, vt_snow_b=0.15,
#                  vt_rain_a=386.0, vt_rain_b=0.67, epsilon=1.0e-12, dt_sedi=10.0, exm1_1 = 2.47 - 1.0,
#                  exp_1 = -1.0 / exm1_1, exm1_2 = 4.7 - 1.0, exp_2 = -1.0 / exm1_2) -> 'MicrophysicsParameters_2M':
#         """Return default microphysics parameters for 2-m scheme"""
#         return cls(
#             ccraut=jnp.array(ccraut),
#             ccracl=jnp.array(ccracl),
#             cauloc=jnp.array(cauloc),
#             ceffmin=jnp.array(ceffmin),
#             ceffmax=jnp.array(ceffmax),
#             cn0s=jnp.array(cn0s),
#             crhosno=jnp.array(crhosno),
#             cvtfall=jnp.array(cvtfall),
#             cthomi=jnp.array(cthomi),
#             csecfrl=jnp.array(csecfrl),
#             ccollec=jnp.array(ccollec),
#             ccollei=jnp.array(ccollei),
#             tau_melt=jnp.array(tau_melt),
#             tau_freeze=jnp.array(tau_freeze),
#             cevaprain=jnp.array(cevaprain),
#             cevapsnow=jnp.array(cevapsnow),
#             vt_ice=jnp.array(vt_ice),
#             vt_snow_a=jnp.array(vt_snow_a),
#             vt_snow_b=jnp.array(vt_snow_b),
#             vt_rain_a=jnp.array(vt_rain_a),
#             vt_rain_b=jnp.array(vt_rain_b),
#             epsilon=jnp.array(epsilon),
#             dt_sedi=jnp.array(dt_sedi),
#             exm1_1=jnp.array(exm1_1),
#             exp_1=jnp.array(exp_1),
#             exm1_2=jnp.array(exm1_2),
#             exp_2=jnp.array(exp_2)
#         )

class MicrophysicsState_2M(NamedTuple):
    """Microphysics state variables and diagnostics"""
    
    # Precipitation fluxes (kg/m²/s)
    rain_flux: jnp.ndarray      # Rain flux at each level
    snow_flux: jnp.ndarray      # Snow flux at each level
    
    # In-cloud values
    qc_in_cloud: jnp.ndarray    # In-cloud liquid water (kg/kg)
    qi_in_cloud: jnp.ndarray    # In-cloud ice (kg/kg)
    qnc_in_cloud: jnp.ndarray   # In-cloud liquid droplet number concentration (1/m³)
    qni_in_cloud: jnp.ndarray   # In-cloud ice crystal number concentration (1/m³)
    
    # Process rates (kg/kg/s)
    autoconv_rate: jnp.ndarray  # Autoconversion rate
    accretion_rate: jnp.ndarray # Accretion rate
    melting_rate: jnp.ndarray   # Melting rate
    freezing_rate: jnp.ndarray  # Freezing rate
    
    # Precipitation at surface
    precip_rain: jnp.ndarray    # Surface rain (kg/m²/s)
    precip_snow: jnp.ndarray    # Surface snow (kg/m²/s)

class MicrophysicsTendencies_2M(NamedTuple):
    """Tendencies from microphysics processes"""
    
    dtedt: jnp.ndarray          # Temperature tendency (K/s)
    dqdt: jnp.ndarray           # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray          # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray          # Cloud ice tendency (kg/kg/s)
    dqncdt: jnp.ndarray         # Cloud droplet number tendency (1/m³/s)
    dqnidt: jnp.ndarray         # Cloud ice crystal number tendency (1/m³/s)
    dqrdt: jnp.ndarray          # Rain water tendency (kg/kg/s)
    dqsdt: jnp.ndarray          # Snow tendency (kg/kg/s)

# Constants
# pi = jnp.pi
# rhoh2o = 1.0  # Placeholder for water density, define appropriately
# cdnc_min_lower = 1.0e6
# cdnc_min_upper = 40.0e6
# rcd_vol_max = 19.0e-6
# ldyn_cdnc_min = True  # Set to True for dynamic CDNC, False for static CDNC
# cdnc_min_fixed = 100.0  # Example value in cm^-3

def microphysics_dt_constants(dt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constants that depend on the microphysics timestep. Here for consistency with ECHAM6,
    where they cannot be parameters.
    Constants are defined locally in each subroutine where needed. """
    
    ztmst = dt
    ztmst_rcp = 1.0 / jnp.maximum(ztmst, eps)
    zcons1 = cpd*vtmpc2
    zcons2 = ztmst * rgrav
    zcons3 = 1.0 / ( pi*crhosno*cn0s*cvtfall**(1.0/1.16) )**0.25
    
    return ztmst, ztmst_rcp, zcons1, zcons2, zcons3

def melting_snow_and_ice(
    melt_mask: jnp.ndarray,
    temperature_previous: jnp.ndarray,
    ice_cloud_previous: jnp.ndarray,
    pressure_thickness: jnp.ndarray,
    icncq: jnp.ndarray,
    lsdcp: jnp.ndarray,
    lvdcp: jnp.ndarray,
    icnc: jnp.ndarray,
    qmel: jnp.ndarray,
    cdnc: jnp.ndarray,
    rain_flux: jnp.ndarray,
    snow_flux: jnp.ndarray,
    ice_flux: jnp.ndarray,
    ice_flux_n: jnp.ndarray,
    ice_tendency: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple:
    """
    Simulates the melting of snow and ice in a cloud microphysics model. This function is a JAX implementation
    of the ECHAM6 `melting_snow_and_ice` routine. It calculates the energy-limited melting capacity based on 
    temperature differences, melts snow flux into rain flux, melts ice-crystal flux into rain water, and handles 
    in-cloud ice melting when the temperature exceeds the melting point.

    The function updates various input arrays in-place and returns updated values for cloud microphysics variables.

    Parameters:
    -----------
    melt_mask : jnp.ndarray
        Boolean mask indicating where the temperature exceeds the melting point (T > tmelt).
    temperature_previous : jnp.ndarray
        Temperature at the previous timestep [K].
    ice_cloud_previous : jnp.ndarray
        Cloud ice mixing ratio at the previous timestep [kg/kg].
    pressure_thickness : jnp.ndarray
        Pressure thickness of the atmospheric layer [Pa].
    icncq : jnp.ndarray
        Temporary ice crystal number concentration to be transferred to droplets during melting [1/m^3].
    lsdcp : jnp.ndarray
        Ratio of sublimation heat to specific heat capacity of dry air (Ls/cpd).
    lvdcp : jnp.ndarray
        Ratio of latent heat of vaporization to specific heat capacity of dry air (Lv/cpd).
    icnc : jnp.ndarray
        Ice crystal number concentration [1/m^3] (INOUT).
    qmel : jnp.ndarray
        Droplet source rate from melting ice [1/m^3/s] (INOUT).
    cdnc : jnp.ndarray
        Cloud droplet number concentration [1/m^3] (INOUT).
    rain_flux : jnp.ndarray
        Rain water flux [kg/m^2/s] (INOUT).
    snow_flux : jnp.ndarray
        Snow flux [kg/m^2/s] (INOUT).
    ice_flux : jnp.ndarray
        Ice crystal mass flux from above [kg/m^2/s] (INOUT).
    ice_flux_n : jnp.ndarray
        Ice crystal number flux from above [1/m^2/s] (INOUT).
    ice_tendency : jnp.ndarray
        Tendency of cloud ice [kg/kg/s] (INOUT).
    dt : jnp.ndarray
        Time step [s].

    Returns:
    --------
    tuple:
        - icnc (jnp.ndarray): Updated ice crystal number concentration [1/m^3].
        - qmel (jnp.ndarray): Updated droplet source rate from melting ice [1/m^3/s].
        - cdnc (jnp.ndarray): Updated cloud droplet number concentration [1/m^3].
        - rain_flux (jnp.ndarray): Updated rain water flux [kg/m^2/s].
        - snow_flux (jnp.ndarray): Updated snow flux [kg/m^2/s].
        - ice_flux (jnp.ndarray): Updated ice crystal mass flux from above [kg/m^2/s].
        - ice_flux_n (jnp.ndarray): Updated ice crystal number flux from above [1/m^2/s].
        - ice_tendency (jnp.ndarray): Updated tendency of cloud ice [kg/kg/s].
        - pimlt (jnp.ndarray): Melting of in-cloud ice (diagnostic) [kg/kg].
        - psmlt (jnp.ndarray): Melting of snow flux (diagnostic) [kg/kg].
        - pximlt (jnp.ndarray): Melting of ice flux from above (diagnostic) [kg/kg].

    Routine Description:
    ---------------------
    1. Computes the energy-limited melting capacity based on the temperature difference above the melting point.
    2. Melts a fraction of the snow flux into rain flux, constrained by the available energy.
    3. Melts a fraction of the ice-crystal flux from above, adjusting both the mass and number fluxes.
    4. Handles in-cloud ice melting when the temperature exceeds the melting point, transferring all ice crystals 
       to cloud droplets and updating the droplet source rate.
    5. Ensures consistency between the number flux and the remaining mass flux of ice crystals.
    6. Outputs diagnostic variables for the melting of in-cloud ice, snow flux, and ice flux from above.
    """

    # Microphysics timestep constants
    ztmst, ztmst_rcp, _, zcons2, _ = microphysics_dt_constants(dt)
    
    # ------------------------------------------------------------
    # 1) Energy-limited melt capacity (per layer) from T - tmelt
    # ------------------------------------------------------------
    ztdif = jnp.maximum(0.0, temperature_previous - tmelt)
    melt_capacity = (
        zcons2
        * ztdif
        * pressure_thickness
        / jnp.maximum(lsdcp - lvdcp, eps)
    )

    # ------------------------------------------------------------
    # 2) Melt snow flux -> rain flux
    # ------------------------------------------------------------
    snow_melt_flux = jnp.minimum(xsec * snow_flux, melt_capacity)  # ztmp2
    rain_flux = rain_flux + snow_melt_flux
    snow_flux = snow_flux - snow_melt_flux

    # Diagnostic melting in mmr units (as in Fortran): psmlt = dt*grav*melt_flux / pdp
    psmlt = ztmst * grav * snow_melt_flux / jnp.maximum(pressure_thickness, eps)

    # ------------------------------------------------------------
    # 3) Melt ice-crystal mass flux from above -> (implicitly) rain water
    # ------------------------------------------------------------
    ice_melt_flux = jnp.minimum(xsec * ice_flux, melt_capacity)

    has_ice_flux = ice_flux > epsec
    ice_melt_flux_n = jnp.where(
        has_ice_flux,
        ice_flux_n * ice_melt_flux / jnp.maximum(ice_flux, epsec),
        0.0,
    )

    ice_flux = ice_flux - ice_melt_flux
    ice_flux_n = ice_flux_n - ice_melt_flux_n

    # Keep number flux consistent with remaining mass flux
    # Expect this helper to exist in the module (or be imported).
    ice_flux_n = consistency_number_to_mass(pthreshold=epsec, pmass=ice_flux, pnumber=ice_flux_n)

    pximlt = ztmst * grav * ice_melt_flux / jnp.maximum(pressure_thickness, eps)

    # ------------------------------------------------------------
    # 4) Melt in-cloud ice mass when melt_mask is True
    # ------------------------------------------------------------
    ice_mass_candidate = jnp.maximum(ice_cloud_previous + ztmst * ice_tendency, 0.0)
    pimlt = jnp.where(melt_mask, ice_mass_candidate, 0.0)
    ice_tendency = ice_tendency - ztmst_rcp * pimlt

    # ------------------------------------------------------------
    # 5) If T > tmelt: melt all ice crystals (number) -> cloud droplets
    # ------------------------------------------------------------
    add_to_cdnc = jnp.where(melt_mask, icncq, 0.0)
    icnc = jnp.where(melt_mask, icemin, icnc)
    cdnc = cdnc + add_to_cdnc
    qmel = qmel + ztmst * add_to_cdnc

    return (
        icnc,
        qmel,
        cdnc,
        rain_flux,
        snow_flux,
        ice_flux,
        ice_flux_n,
        ice_tendency,
        pimlt,
        psmlt,
        pximlt,
    )

def sublimation_snow_and_ice_evaporation_rain(
    precip_mask: jnp.ndarray,                 # ld_precip
    falling_ice_mask: jnp.ndarray,            # ld_falling_ice
    specific_humidity_prev: jnp.ndarray,      # pqm1 [kg/kg]
    temperature_prev: jnp.ndarray,            # ptm1 [K]
    precip_fraction: jnp.ndarray,             # pclcpre [0..1]
    pressure_thickness: jnp.ndarray,          # pdp [Pa]
    dp_over_g: jnp.ndarray,                   # pdpg [kg/m^2] (delta p / g)
    subsat_wrt_ice: jnp.ndarray,              # picesub (sub-saturation w.r.t. ice; scheme-specific)
    lsdcp: jnp.ndarray,                       # plsdcp = Ls/cpd
    inv_air_density: jnp.ndarray,             # pqrho [m^3/kg] = 1/rho
    qsat_ice: jnp.ndarray,                    # pqsi [kg/kg]
    inv_air_density_rcp: jnp.ndarray,         # prho_rcp (also 1/rho; retained for exact port)
    snow_flux: jnp.ndarray,                   # psfl [kg/m^2/s]
    air_density: jnp.ndarray,                 # prho [kg/m^3]
    qsat_water_prev: jnp.ndarray,             # pqsw [kg/kg] at (t-1)
    rain_flux: jnp.ndarray,                   # prfl [kg/m^2/s]
    subsat_wrt_water_evap: jnp.ndarray,       # psusatw_evap (sub-saturation w.r.t. water; scheme-specific)
    thermo_term_water: jnp.ndarray,           # pastbstw (thermodynamic factor, >0)
    falling_ice_fraction: jnp.ndarray,        # pclcfi [0..1] fraction covered by falling ice
    ice_flux: jnp.ndarray,                    # pxiflux (INOUT) [kg/m^2/s]
    ice_flux_n: jnp.ndarray,                  # pxifluxn (INOUT) [1/m^2/s]
    dt: jnp.ndarray,                          # ztmst [s]
) -> tuple[
    jnp.ndarray,  # ice_flux (updated) [kg/m^2/s]
    jnp.ndarray,  # ice_flux_n (updated) [1/m^2/s]
    jnp.ndarray,  # ice_sublim (sublimation of falling ice) [kg/kg]
    jnp.ndarray,  # snow_sublim   (sublimation of snow) [kg/kg]
    jnp.ndarray,  # rain_evap   (evaporation of rain) [kg/kg]
]:
    """
    Sublimation of snow and *falling* ice + evaporation of rain (ICON/ECHAM 2-moment scheme).

    JAX port of the ECHAM6 subroutine `sublimation_snow_and_ice_evaporation_rain`.

    Routine overview
    ----------------
    This routine computes three microphysical sink terms in a grid box / column slice:

    1) **Snow sublimation** (`snow_sublim`, kg/kg):
       Removes snow mass (represented as a snow flux `snow_flux`) when the environment is
       subsaturated with respect to ice. The sink is limited by:
         - the available snow flux per precipitating area,
         - the vapor deficit w.r.t. ice: (qsat_ice - specific_humidity_prev),
         - a diffusion/ventilation coefficient dependent on temperature and density.

    2) **Falling-ice sublimation** (`ice_sublim`, kg/kg):
       Similar to snow sublimation, but applied to the *falling ice mass flux from above*
       (`ice_flux`). This routine then updates:
         - `ice_flux` (mass flux) by removing sublimated mass,
         - `ice_flux_n` (number flux) consistently with mass removal,
         - and enforces physical consistency by zeroing `ice_flux_n` when `ice_flux` is tiny.

       Important: this is **falling ice** sublimation only (matches Fortran comment).
       Sublimation of *cloud ice mixing ratio* is handled elsewhere in the scheme.

    3) **Rain evaporation** (`rain_evap`, kg/kg):
       Evaporates rain flux `rain_flux` under subsaturation with respect to liquid water,
       limited by:
         - available rain flux per precipitating area,
         - vapor deficit w.r.t. water: (qsat_water_prev - specific_humidity_prev),
         - an evaporation coefficient depending on density and a thermodynamic term.

    Inputs
    ------
    precip_mask :
        Boolean array, presence of precipitation (`ld_precip`).
    falling_ice_mask :
        Boolean array, presence of falling ice (`ld_falling_ice`).
    specific_humidity_prev :
        `pqm1`, specific humidity at previous step [kg/kg].
    temperature_prev :
        `ptm1`, temperature at previous step [K].
    precip_fraction :
        `pclcpre`, fraction of grid box covered by precip [0..1].
    pressure_thickness :
        `pdp`, layer pressure thickness [Pa].
    dp_over_g :
        `pdpg`, dp/g [kg/m^2].
    subsat_wrt_ice :
        `picesub`, subsaturation w.r.t. ice (scheme-specific diagnostic).
    lsdcp :
        `plsdcp`, latent heat of sublimation divided by cp [K] (ECHAM convention).
    inv_air_density :
        `pqrho`, inverse air density [m^3/kg] (1/rho).
    qsat_ice :
        `pqsi`, saturation specific humidity w.r.t. ice [kg/kg].
    inv_air_density_rcp :
        `prho_rcp`, inverse air density again (kept for exact Fortran mapping).
    snow_flux :
        `psfl`, snow mass flux [kg/m^2/s].
    air_density :
        `prho`, air density [kg/m^3].
    qsat_water_prev :
        `pqsw`, saturation specific humidity w.r.t. water at (t-1) [kg/kg].
    rain_flux :
        `prfl`, rain mass flux [kg/m^2/s].
    subsat_wrt_water_evap :
        `psusatw_evap`, subsaturation term w.r.t. water used by evaporation formula.
    thermo_term_water :
        `pastbstw`, thermodynamic term in evaporation expression (must be > 0).
    falling_ice_fraction :
        `pclcfi`, fraction of grid box covered by falling ice [0..1].
    ice_flux :
        `pxiflux` (INOUT), falling-ice mass flux into grid box from above [kg/m^2/s].
    ice_flux_n :
        `pxifluxn` (INOUT), falling-ice number flux into grid box from above [1/m^2/s].
    dt :
        `ztmst`, timestep [s].

    Returns
    -------
    ice_flux :
        Updated `pxiflux` after sublimation [kg/m^2/s].
    ice_flux_n :
        Updated `pxifluxn` after sublimation and number/mass consistency fix [1/m^2/s].
    ice_sublim :
        Sublimation of falling ice expressed as a mixing-ratio increment over the timestep [kg/kg].
    snow_sublim :
        Sublimation of snow expressed as a mixing-ratio increment over the timestep [kg/kg].
    rain_evap :
        Evaporation of rain expressed as a mixing-ratio increment over the timestep [kg/kg].
    """

     # Microphysics timestep constants
    ztmst, _, _, zcons2, zcons3 = microphysics_dt_constants(dt)

    # ------------------------------------------------------------------
    # Common diffusion/ventilation coefficient for ice-phase sublimation
    # ------------------------------------------------------------------
    denom = (1.0 / (2.43e-2 * rv)) * (lsdcp**2) / jnp.maximum(temperature_prev**2, eps)
    denom = denom + (1.0 / 0.211e-4) * inv_air_density_rcp / jnp.maximum(qsat_ice, eps)
    zcoeff = 3.0e6 * 2.0 * pi * subsat_wrt_ice * inv_air_density_rcp / jnp.maximum(denom, eps)

    # Avoid division by zero for area fractions: MERGE(frac, 1, mask)
    zclcpre = jnp.where(precip_mask, precip_fraction, 1.0)
    zclcfi = jnp.where(falling_ice_mask, falling_ice_fraction, 1.0)

    # ------------------------------------------------------------------
    # Snow sublimation (snow_sublim)
    # ------------------------------------------------------------------
    ll_snow = jnp.logical_and(snow_flux > cqtmin, precip_mask)

    zclambs_s = zcons3 * (snow_flux / jnp.maximum(zclcpre, eps)) ** (0.25 / 1.16)
    zcfac4c_s = 0.78 * zclambs_s**2 + 232.19 * (inv_air_density**0.25) * (zclambs_s**2.625)
    ztmp2_s = zcfac4c_s * zcoeff * dp_over_g

    zzeps_s = jnp.maximum(-xsec * snow_flux / jnp.maximum(zclcpre, eps), ztmp2_s)
    ztmp3_s = -ztmst * zzeps_s / jnp.maximum(dp_over_g, eps) * zclcpre

    ztmp4_s = jnp.maximum(xsec * (qsat_ice - specific_humidity_prev), 0.0)
    ztmp3_s = jnp.clip(ztmp3_s, 0.0, ztmp4_s)
    snow_sublim = jnp.where(ll_snow, ztmp3_s, 0.0)

    # ------------------------------------------------------------------
    # Falling ice sublimation (ice_sublim) and update ice_flux, ice_flux_n
    # ------------------------------------------------------------------
    ll_ice = jnp.logical_and(ice_flux > cqtmin, falling_ice_mask)

    zclambs_i = zcons3 * (ice_flux / jnp.maximum(zclcfi, eps)) ** (0.25 / 1.16)
    zcfac4c_i = 0.78 * zclambs_i**2 + 232.19 * (inv_air_density**0.25) * (zclambs_i**2.625)
    ztmp2_i = zcfac4c_i * zcoeff * dp_over_g

    zzeps_i = jnp.maximum(-xsec * ice_flux / jnp.maximum(zclcfi, eps), ztmp2_i)
    ztmp3_i = -ztmst * zzeps_i / jnp.maximum(dp_over_g, eps) * zclcfi

    ztmp4_i = jnp.maximum(xsec * (qsat_ice - specific_humidity_prev), 0.0)
    ztmp3_i = jnp.clip(ztmp3_i, 0.0, ztmp4_i)
    ice_sublim = jnp.where(ll_ice, ztmp3_i, 0.0)

    # number flux reduction due to sublimated mass
    zsubin = ice_sublim * ice_flux_n / jnp.maximum(ice_flux, cqtmin)
    zsubin = zcons2 * zsubin * pressure_thickness
    zsubin = jnp.where(ll_ice, zsubin, 0.0)

    ice_flux_n = ice_flux_n - zsubin
    ice_flux = ice_flux - zcons2 * ice_sublim * pressure_thickness

    ice_flux_n = consistency_number_to_mass(pthreshold=epsec, pmass=ice_flux, pnumber=ice_flux_n)

    # ------------------------------------------------------------------
    # Rain evaporation (rain_evap)
    # ------------------------------------------------------------------
    ll_rain = jnp.logical_and(rain_flux > cqtmin, precip_mask)

    ztmp2_r = (
        870.0
        * subsat_wrt_water_evap
        * dp_over_g
        * (rain_flux / jnp.maximum(zclcpre, eps)) ** 0.61
        / (jnp.sqrt(jnp.maximum(air_density, eps)) * jnp.maximum(thermo_term_water, eps))
    )

    zzeps_r = jnp.maximum(-xsec * rain_flux / jnp.maximum(zclcpre, eps), ztmp2_r)
    ztmp3_r = -ztmst * zzeps_r * zclcpre / jnp.maximum(dp_over_g, eps)

    ztmp4_r = jnp.maximum(xsec * (qsat_water_prev - specific_humidity_prev), 0.0)
    ztmp3_r = jnp.clip(ztmp3_r, 0.0, ztmp4_r)
    rain_evap = jnp.where(ll_rain, ztmp3_r, 0.0)

    return ice_flux, ice_flux_n, ice_sublim, snow_sublim, rain_evap

def sedimentation_ice(
    cloud_fraction: jnp.ndarray,          # paclc [0..1]
    air_density_correction: jnp.ndarray,  # paaa  (air-density correction for fall speed)
    pressure_thickness: jnp.ndarray,      # pdp [Pa]
    air_density: jnp.ndarray,             # prho [kg/m^3]
    inv_air_density_rcp: jnp.ndarray,     # prho_rcp [m^3/kg] (1/rho) in ICON naming
    ice_mmr_gridmean: jnp.ndarray,        # pxip1 (INOUT) grid-mean ice mass mixing ratio [kg/kg]
    icnc_in_cloud: jnp.ndarray,           # picnc (INOUT) in-cloud ice crystal number conc. [1/m^3]
    ice_flux: jnp.ndarray,                # pxiflux (INOUT) ice-crystal mass flux into layer from above [kg/m^2/s]
    ice_flux_n: jnp.ndarray,              # pxifluxn (INOUT) ice-crystal number flux into layer from above [1/m^2/s]
    falling_ice_fraction: jnp.ndarray,    # pclcfi (INOUT) fraction of grid box covered by sedimenting/falling ice [0..1]
    dt: jnp.ndarray,                      # ztmst [s]
) -> tuple[
    jnp.ndarray,  # ice_mmr_gridmean (updated) [kg/kg]
    jnp.ndarray,  # icnc_in_cloud (updated) [1/m^3]
    jnp.ndarray,  # ice_flux (updated) [kg/m^2/s]
    jnp.ndarray,  # ice_flux_n (updated) [1/m^2/s]
    jnp.ndarray,  # falling_ice_fraction (updated) [0..1]
    jnp.ndarray,  # ice_sedimentation_rate_in_cloud (pmrateps) [kg/kg]
]:
    """
    Sedimentation of cloud ice (mass + number) and update of falling-ice fluxes (Lin et al. (1983)).

    This is a JAX port of the Fortran subroutine `sedimentation_ice` from ICON/ECHAM
    (mo_cloud_microphysics_2m). It performs a single sedimentation step for **cloud ice**
    and updates the **falling ice fluxes** (mass and number) entering/leaving the layer.

    Conventions / important details
    -------------------------------
    - `ice_mmr_gridmean` is treated as a **grid-mean** cloud-ice mass mixing ratio [kg/kg] (Fortran `pxip1`).
    - `icnc_in_cloud` is treated as **in-cloud** ice crystal number concentration [1/m^3] (Fortran `picnc`).
      The routine converts it to **grid-mean** via: `zicnc_gridmean = icnc_in_cloud * cloud_fraction`
      for the sedimentation update, then converts back to in-cloud where `cloud_fraction > clc_min`.
    - `ice_flux` and `ice_flux_n` are **falling** ice fluxes coming from above (Fortran `pxiflux`, `pxifluxn`).
      They are updated by adding the flux contribution from sedimentation out of this level.
    - The fall speed depends on an effective mean mass-per-crystal proxy and is limited to [0.001, 2.0] m/s.
    - `falling_ice_fraction` is updated with `gridbox_frac_falling_hydrometeor(...)`, consistent with other
      precip/falling-hydrometeor routines in this module.
    - Finally, `ice_flux_n` is passed through `consistency_number_to_mass(...)` to enforce that number flux
      cannot remain nonzero when mass flux is essentially zero (ICON/ECHAM consistency safeguard).

    Parameters
    ----------
    cloud_fraction : array
        Cloud cover `paclc` [0..1].
    air_density_correction : array
        Density correction factor `paaa` used in the ice crystal fall velocity (dimensionless).
    pressure_thickness : array
        Layer pressure thickness `pdp` [Pa].
    air_density : array
        Air density `prho` [kg/m^3].
    inv_air_density_rcp : array
        Inverse air density `prho_rcp` [m^3/kg] (ICON naming; effectively 1/rho).
    ice_mmr_gridmean : array
        Grid-mean ice mass mixing ratio `pxip1` [kg/kg] (INOUT).
    icnc_in_cloud : array
        In-cloud ice crystal number concentration `picnc` [1/m^3] (INOUT).
    ice_flux : array
        Falling-ice *mass* flux entering from above `pxiflux` [kg/m^2/s] (INOUT).
    ice_flux_n : array
        Falling-ice *number* flux entering from above `pxifluxn` [1/m^2/s] (INOUT).
    falling_ice_fraction : array
        Gridbox fraction covered by sedimenting ice `pclcfi` [0..1] (INOUT).
    dt : array or scalar
        Microphysics time step `ztmst` [s].

    Returns
    -------
    ice_mmr_gridmean : array
        Updated grid-mean ice mass mixing ratio [kg/kg].
    icnc_in_cloud : array
        Updated in-cloud ice crystal number concentration [1/m^3].
    ice_flux : array
        Updated falling-ice mass flux [kg/m^2/s].
    ice_flux_n : array
        Updated falling-ice number flux [1/m^2/s].
    falling_ice_fraction : array
        Updated falling-ice fractional coverage [0..1].
    ice_sedimentation_rate_in_cloud : array
        Diagnostic in-cloud sedimented ice amount (`pmrateps`) [kg/kg].
        This is `zxi_delta / max(cloud_fraction, clc_min)` where clouds exist, otherwise the grid-mean `zxi_delta`.
        In ICON/ECHAM it is used for in-cloud scavenging diagnostics.

    """

    # Fortran uses ztmst and zcons2 ( = ztmst * rgrav ) from common timestep constants.
    ztmst, _, _, zcons2, _ = microphysics_dt_constants(dt)

    # --- Keep a copy of grid-mean ice before sedimentation
    zxi_bf_sed = ice_mmr_gridmean

    # --- Convert ICNC to grid-mean and enforce minimum
    zicnc_gridmean = icnc_in_cloud * cloud_fraction
    zicnc_gridmean = jnp.maximum(zicnc_gridmean, icemin)
    zicnc_gridmean_bf_sed = zicnc_gridmean

    # --- Mean mass per crystal proxy
    zmmean = air_density * ice_mmr_gridmean / jnp.maximum(zicnc_gridmean, eps)
    zmmean = jnp.maximum(zmmean, mi)

    # --- Regime selection for sedimentation parameters
    ll_small = zmmean < ri_vol_mean_1
    ll_mid = jnp.logical_and(~ll_small, zmmean < ri_vol_mean_2)

    zalfased = jnp.where(ll_small, alfased_1, alfased_2)
    zalfased = jnp.where(ll_mid, alfased_3, zalfased)

    zbetased = jnp.where(ll_small, betased_1, betased_2)
    zbetased = jnp.where(ll_mid, betased_3, zbetased)

    # --- Fall speed (mass and number use same here), limited as in Fortran
    zxifallmc = fall * zalfased * (zmmean ** zbetased) * air_density_correction
    zxifallmc = jnp.clip(zxifallmc, 0.001, 2.0)
    zxifallnc = zxifallmc

    # --- Exponential coefficients
    zal1 = ztmst * grav * zxifallmc * air_density / jnp.maximum(pressure_thickness, eps)
    zal3 = grav * ztmst * zxifallnc * air_density / jnp.maximum(pressure_thickness, eps)

    # --- Incoming-flux "equilibria" (MERGE to 0 if fall speed is too small)
    ll_mass = zxifallmc > eps
    zal2_raw = ice_flux * inv_air_density_rcp / jnp.maximum(zxifallmc, eps)
    zal2 = jnp.where(ll_mass, zal2_raw, 0.0)

    ll_num = zxifallnc > eps
    zal4_raw = ice_flux_n / jnp.maximum(zxifallnc, eps)
    zal4 = jnp.where(ll_num, zal4_raw, 0.0)

    # --- Update grid-mean ice mmr and grid-mean ICNC via relaxation form
    exp1 = jnp.exp(-zal1)
    exp3 = jnp.exp(-zal3)

    ice_mmr_gridmean = ice_mmr_gridmean * exp1 + zal2 * (1.0 - exp1)
    zicnc_gridmean = zicnc_gridmean * exp3 + zal4 * (1.0 - exp3)

    # --- Convert back to in-cloud ICNC where cloud fraction is meaningful
    has_cloud = cloud_fraction > clc_min
    icnc_in_cloud_candidate = zicnc_gridmean / jnp.maximum(cloud_fraction, clc_min)
    icnc_in_cloud = jnp.where(has_cloud, icnc_in_cloud_candidate, zicnc_gridmean)

    # --- Sedimented grid-mean amount
    # zxi_delta can be negative if the incoming flux equilibrium (zal2) exceeds
    # the initial ice content — the layer gains more mass from above than it loses.
    # In that case zxiflx_from_level would be negative (net absorption), which would
    # *reduce* the outgoing flux below the incoming flux. This is physically valid
    # (the layer is a net sink), but the outgoing flux itself cannot go below zero.
    zxi_delta = zxi_bf_sed - ice_mmr_gridmean

    # --- Flux contribution from this level (can be negative = net absorption from above)
    zxiflx_from_level = zcons2 * zxi_delta * pressure_thickness

    # --- In-cloud sedimentation diagnostic (pmrateps in Fortran)
    # Only meaningful as a positive rate; clamp to zero for the absorption case.
    pmrateps_in_cloud = zxi_delta / jnp.maximum(cloud_fraction, clc_min)
    ice_sedimentation_rate_in_cloud = jnp.where(has_cloud, pmrateps_in_cloud, zxi_delta)
    ice_sedimentation_rate_in_cloud = jnp.maximum(ice_sedimentation_rate_in_cloud, 0.0)

    # --- Update fraction covered by falling ice
    # Only update if there is a positive flux contribution from this level.
    falling_ice_fraction = gridbox_frac_falling_hydrometeor(
        precip_flux_from_above=ice_flux,
        precip_frac_from_above=falling_ice_fraction,
        precip_flux_from_level=jnp.maximum(zxiflx_from_level, 0.0),  # only positive contribution
        precip_frac_from_level=cloud_fraction,
    )

    # --- Update mass flux
    # The outgoing flux = incoming + sedimented_out - absorbed_from_above.
    # Cannot go below zero: if zxiflx_from_level < 0 (net absorption), limit removal
    # to what is available in the incoming flux.
    ice_flux = jnp.maximum(ice_flux + zxiflx_from_level, 0.0)

    # --- Update number flux
    # Same logic: delta_n can be negative (layer absorbs crystals from above).
    # Outgoing number flux cannot go below zero.
    delta_n = zcons2 * (zicnc_gridmean_bf_sed - zicnc_gridmean) * pressure_thickness * inv_air_density_rcp
    ice_flux_n = jnp.maximum(ice_flux_n + delta_n, 0.0)

    # --- Enforce mass/number consistency
    ice_flux_n = consistency_number_to_mass(pthreshold=epsec, pmass=ice_flux, pnumber=ice_flux_n)

    return (
        ice_mmr_gridmean,
        icnc_in_cloud,
        ice_flux,
        ice_flux_n,
        falling_ice_fraction,
        ice_sedimentation_rate_in_cloud,
    )
# ...existing code...

def mixed_phase_deposition_and_corrections(
    pressure: jnp.ndarray,               # papp1 [Pa] pressure at full levels (t-1)
    icnc: jnp.ndarray,                   # picnc [1/m^3] ice crystal number concentration
    specific_humidity_prev: jnp.ndarray, # pqm1 [kg/kg] specific humidity (t-1)
    cloud_fraction: jnp.ndarray,         # paclc [0..1] cloud cover
    sat_vap_pres_ice: jnp.ndarray,       # pesi [Pa] saturation vapour pressure w.r.t. ice
    sat_vap_pres_water: jnp.ndarray,     # pesw [Pa] saturation vapour pressure w.r.t. water
    bergeron_variable: jnp.ndarray,      # peta [-] variable for Bergeron-Findeisen process
    tompkins_genti: jnp.ndarray,         # pgenti [kg/kg] Tompkins cloud cover scheme variable
    lsdcp: jnp.ndarray,                  # plsdcp [K] Ls / cpd
    lvdcp: jnp.ndarray,                  # plvdcp [K] Lv / cpd
    specific_humidity: jnp.ndarray,      # pqp1 [kg/kg] specific humidity (t)
    qsat_prev: jnp.ndarray,              # pqsm1 [kg/kg] saturation specific humidity (t-1)
    air_density: jnp.ndarray,            # prho [kg/m^3]
    temperature: jnp.ndarray,            # ptp1 [K] temperature (t)
    ice_evaporation: jnp.ndarray,        # pxievap [kg/kg] evaporation of cloud ice
    ice_mmr_gridmean: jnp.ndarray,       # pxip1 [kg/kg] ice mass mixing ratio (grid-mean, t)
    ice_detrainment_tendency: jnp.ndarray, # pxite [kg/kg/s] cloud ice tendency from detrainment
    updraft_velocity: jnp.ndarray,       # pvervx [cm/s] updraft velocity
    condensation_rate: jnp.ndarray,      # pcnd [kg/kg] (INOUT) condensation rate
    deposition_rate: jnp.ndarray,        # pdep [kg/kg] (INOUT) deposition rate
    dt: jnp.ndarray,                     # ztmst [s]
    ll_het: bool = True,                 # heterogeneous nucleation flag (module-level in Fortran)
) -> tuple[
    jnp.ndarray,  # condensation_rate (updated pcnd) [kg/kg]
    jnp.ndarray,  # deposition_rate (updated pdep) [kg/kg]
    jnp.ndarray,  # temperature_tmp (ptp1tmp) [K]
    jnp.ndarray,  # specific_humidity_tmp (pqp1tmp) [kg/kg]
    jnp.ndarray,  # qsat_tmp (pqsp1tmp) [kg/kg]
]:
    """
    Mixed-phase deposition and condensation corrections for the ICON/ECHAM 2-moment scheme.

    JAX port of Fortran subroutine `mixed_phase_deposition_and_corrections`
    (mo_cloud_microphysics_2m).

    Overview
    --------
    This routine determines whether a grid box is in the ice or liquid phase,
    computes updated saturation specific humidities at the new temperature,
    and applies condensation/deposition increments accounting for:
      - Bergeron-Findeisen process (ice growth at expense of liquid),
      - Homogeneous vs heterogeneous cirrus nucleation (via nic_cirrus / ll_het),
      - Phase-consistent thermodynamic corrections to temperature and humidity.

    It does NOT perform sedimentation or precipitation — those are handled in
    `sedimentation_ice` and `precip_formation_cold/warm`.

    Steps
    -----
    1. Compute first-guess updated temperature (`temperature_tmp`) and specific
       humidity (`specific_humidity_tmp`) from existing condensation/deposition rates.
    2. Update ice mass mixing ratio (`zxip1`) including detrainment, evaporation,
       Tompkins source (`pgenti`), and deposition.
    3. Compute effective ice crystal radius from `zxip1` and `icnc` (via
       `eff_ice_crystal_radius`), then convert to volume-mean radius using the
       Schumann et al. (2011) parameterisation.
    4. Compute Bergeron-Findeisen threshold vertical velocity (`zvervmax`) from
       saturation vapour pressures, ICNC, ice radius, and `peta`.
    5. Determine phase mask `lo2`:
       - True  (ice)    if T < cthomi, OR if T < tmelt AND updraft < threshold
       - False (liquid) otherwise
    6. Look up saturation vapour pressures at the new temperature using the
       ECHAM lookup-table approach (here replaced by analytic Teten's formula
       consistent with the rest of the JAX scheme).
    7. Compute saturation specific humidities and thermodynamic correction factor
       `zqcon = 1 / (1 + Lc * dqs/dT)`.
    8. Apply deposition increment to `deposition_rate` (ice cases) and condensation
       increment to `condensation_rate` (liquid cases), using phase-dependent
       supersaturation thresholds and the `nic_cirrus` / `ll_het` flags.
    9. Apply final corrections: if the updated humidity falls below `zrhtest`
       (a RH-limited threshold based on t-1 humidity), reduce the
       condensation/deposition so as not to over-dry the grid box.
    10. Recompute `temperature_tmp` and `specific_humidity_tmp` from the corrected rates.

    Parameters
    ----------
    pressure : array
        Full-level pressure at (t-1), `papp1` [Pa].
    icnc : array
        In-cloud ice crystal number concentration `picnc` [1/m^3].
    specific_humidity_prev : array
        Specific humidity at (t-1) `pqm1` [kg/kg].
    cloud_fraction : array
        Cloud cover `paclc` [0..1].
    sat_vap_pres_ice : array
        Saturation vapour pressure w.r.t. ice `pesi` [Pa].
    sat_vap_pres_water : array
        Saturation vapour pressure w.r.t. water `pesw` [Pa].
    bergeron_variable : array
        Variable for the Bergeron-Findeisen threshold velocity `peta` [-].
    tompkins_genti : array
        Ice source term from the Tompkins cloud cover scheme `pgenti` [kg/kg].
    lsdcp : array
        Latent heat of sublimation / cpd `plsdcp` [K].
    lvdcp : array
        Latent heat of vaporisation / cpd `plvdcp` [K].
    specific_humidity : array
        Specific humidity at (t) `pqp1` [kg/kg].
    qsat_prev : array
        Saturation specific humidity at (t-1) `pqsm1` [kg/kg].
    air_density : array
        Air density `prho` [kg/m^3].
    temperature : array
        Temperature at (t) `ptp1` [K].
    ice_evaporation : array
        Evaporation of cloud ice `pxievap` [kg/kg].
    ice_mmr_gridmean : array
        Grid-mean cloud ice mass mixing ratio at (t) `pxip1` [kg/kg].
    ice_detrainment_tendency : array
        Cloud ice tendency from convective detrainment `pxite` [kg/kg/s].
    updraft_velocity : array
        Updraft velocity `pvervx` [cm/s].
    condensation_rate : array
        Condensation rate `pcnd` [kg/kg] (INOUT).
    deposition_rate : array
        Deposition rate `pdep` [kg/kg] (INOUT).
    dt : array or scalar
        Microphysics timestep `ztmst` [s].
    ll_het : bool
        Module-level flag for heterogeneous nucleation path (default False).

    Returns
    -------
    condensation_rate : array
        Updated condensation rate `pcnd` [kg/kg].
    deposition_rate : array
        Updated deposition rate `pdep` [kg/kg].
    temperature_tmp : array
        Updated temperature `ptp1tmp` [K].
    specific_humidity_tmp : array
        Updated specific humidity `pqp1tmp` [kg/kg].
    qsat_tmp : array
        Updated saturation specific humidity `pqsp1tmp` [kg/kg].

    Notes
    -----
    The Fortran lookup table calls (`set_lookup_index`, `tlucua`, `tlucuaw`,
    `tlucub`, `sat_spec_hum`) are replaced here by inline Teten's formula
    computations consistent with the rest of the JAX scheme.
    The `effective_2_volmean_radius_param_Schuman_2011` and
    `threshold_vert_vel` helpers must be available in this module or imported.
    """

    ztmst = dt

    # -------------------------------------------------------------------------
    # 1. First-guess updated temperature and specific humidity
    # -------------------------------------------------------------------------
    temperature_tmp = temperature + lvdcp * condensation_rate + lsdcp * deposition_rate
    specific_humidity_tmp = specific_humidity - condensation_rate - deposition_rate

    # -------------------------------------------------------------------------
    # 2. Updated ice mass mixing ratio (grid-mean)
    #    zxip1 = pxip1 + dt*pxite - pxievap + pgenti + pdep
    # -------------------------------------------------------------------------
    zxip1 = ice_mmr_gridmean + ztmst * ice_detrainment_tendency - ice_evaporation + tompkins_genti + deposition_rate
    zxip1 = jnp.maximum(zxip1, 0.0)

    # -------------------------------------------------------------------------
    # 3. Effective ice crystal radius → volume-mean radius (Schumann 2011)
    #    Convert: grid-mean kg/kg → in-cloud g/m^3
    # -------------------------------------------------------------------------
    ice_gm3 = 1000.0 * zxip1 * air_density / jnp.maximum(cloud_fraction, clc_min)
    zrieff = eff_ice_crystal_radius(ice_gm3, icnc)           # [µm]
    zrieff = jnp.clip(zrieff, ceffmin, ceffmax)

    # Schumann et al. (2011) parameterisation: r_vol from r_eff
    # zrih = -2261 + sqrt(5113188 + 2809*zrieff^3); zrice = 1e-6 * zrih^(1/3)
    zrih = -2261.0 + jnp.sqrt(5113188.0 + 2809.0 * zrieff**3)
    zrice = 1.0e-6 * jnp.maximum(zrih, 0.0) ** (1.0 / 3.0)

    # -------------------------------------------------------------------------
    # 4. Bergeron-Findeisen threshold vertical velocity
    # -------------------------------------------------------------------------
    zvervmax = threshold_vert_vel(
        sat_vap_pres_water=sat_vap_pres_water,
        sat_vap_pres_ice=sat_vap_pres_ice,
        icnc=icnc,
        ice_radius=zrice,
        eta=bergeron_variable,
    )

    # -------------------------------------------------------------------------
    # 5. Phase mask lo2:  True = ice cloud,  False = liquid cloud
    #    lo2 = (T_tmp < cthomi) OR (T_tmp < tmelt AND 0.01*pvervx < zvervmax)
    # -------------------------------------------------------------------------
    lo2 = jnp.logical_or(
        temperature_tmp < cthomi,
        jnp.logical_and(
            temperature_tmp < tmelt,
            0.01 * updraft_velocity < zvervmax,
        ),
    )

    # -------------------------------------------------------------------------
    # 6. Saturation vapour pressures and specific humidities at temperature_tmp
    #    using Teten's formula (replaces Fortran lookup tables).
    #
    #    Over ice  (lo2=True):  e_s = e_s_ice(T_tmp)
    #    Over water(lo2=False): e_s = e_s_water(T_tmp)
    #
    #    sat_spec_hum: q_s = eps * e_s / (p - (1-eps)*e_s)
    #                      ≈ e_s / (p/(eps) - e_s)    [standard approximation]
    #    where eps = Rd/Rv, vtmpc1 = Rv/Rd - 1
    # -------------------------------------------------------------------------

    # Re-evaluate at temperature_tmp (this replaces fortran lookup tables)
    ztmp_ice = (alhs/rv)*(1.0/t0 - 1.0/temperature_tmp)
    ztmp_water = (alhc/rv)*(1.0/t0 - 1.0/temperature_tmp)
    zes_ice_new = 611 * jnp.exp(ztmp_ice)
    zes_water_new = 611 * jnp.exp(ztmp_water)

    # Select phase-appropriate saturation vapour pressure
    zes = jnp.where(lo2, zes_ice_new, zes_water_new)
    zesw = zes_water_new

    # Saturation specific humidity (standard formula)
    # q_s = zes / (p - (1 - Rd/Rv)*zes)  — same form as ECHAM sat_spec_hum
    def _qsat(e, p):
        e_clipped = jnp.minimum(e, 0.4 * p)   # safety clip (Fortran: zes < 0.4)
        return e_clipped / (p - (1.0 - 1.0 / (1.0 + vtmpc1)) * e_clipped)

    qsat_tmp = _qsat(zes, pressure)          # pqsp1tmp: phase-appropriate
    qsat_tmp_water = _qsat(zesw, pressure)   # zqsp1tmpw: always over water

    # zcor: correction factor d(q_s)/d(e_s) * p / (p - e_s)^2  (used in zlcdqsdt)
    # In ECHAM: zcor = 1 / (1 - vtmpc1 * q_s)
    zcor = 1.0 / jnp.maximum(1.0 - vtmpc1 * qsat_tmp, eps)
    zcorw = 1.0 / jnp.maximum(1.0 - vtmpc1 * qsat_tmp_water, eps)

    # -------------------------------------------------------------------------
    # 7. Saturation specific humidity at (t+1) for zdqsdt
    #    In Fortran: zqst1 uses tlucuap1 (lookup at it+1), approximated here
    #    by evaluating at (T_tmp + 1 K) and taking finite difference.
    # -------------------------------------------------------------------------
    ztmp_ice_p1 = jnp.minimum(ak * (temperature_tmp + 1.0 - tmelt) / jnp.maximum(temperature_tmp + 1.0 - 7.66, eps), 700.0)
    ztmp_water_p1 = jnp.minimum(ak * (temperature_tmp + 1.0 - tmelt) / jnp.maximum(temperature_tmp + 1.0 - 35.86, eps), 700.0)

    zes_p1 = jnp.where(lo2, p0s1_bg * jnp.exp(ztmp_ice_p1), p0s1_bg * jnp.exp(ztmp_water_p1))
    zqst1 = zes_p1 / pressure
    zqst1 = jnp.minimum(zqst1, 0.5)
    zqst1 = zqst1 / (1.0 - vtmpc1 * zqst1)

    # zdqsdt = 1000*(q_s(T+1) - q_s(T))  [units: per 1000 K — as in Fortran]
    zdqsdt = 1000.0 * (zqst1 - qsat_tmp)

    # -------------------------------------------------------------------------
    # 8. Thermodynamic correction factor zqcon
    #    Fortran: zlcdqsdt = MERGE(lc*zdqsdt, q_s*zcor*zlucub, ll1)
    #    where ll1 = (zes < 0.4) and zlucub ~ d(ln zes)/dT from the table.
    #    In the analytic port: use lc*zdqsdt for both branches (ll1 captures
    #    a numerical regime of the lookup table; for the analytic formula the
    #    two expressions converge).
    # -------------------------------------------------------------------------
    ll1 = zes < 0.4 * pressure   # equivalent to Fortran ll1 (zes < 0.4 in sat_spec_hum units)

    zlc = jnp.where(lo2, lsdcp, lvdcp)

    # zlucub equivalent: (Lc/Rv) / T^2  (Clausius-Clapeyron derivative of ln e_s)
    zlucub = jnp.where(
        lo2,
        als / (rv * jnp.maximum(temperature_tmp**2, eps)),  # ice
        alv / (rv * jnp.maximum(temperature_tmp**2, eps)),  # water
    )

    ztmp1_zlcd = zlc * zdqsdt
    ztmp2_zlcd = qsat_tmp * zcor * zlucub
    zlcdqsdt = jnp.where(ll1, ztmp1_zlcd, ztmp2_zlcd)

    zqcon = 1.0 / (1.0 + zlcdqsdt)

    # -------------------------------------------------------------------------
    # 9. Supersaturation thresholds
    # -------------------------------------------------------------------------
    zoversat = 0.01 * qsat_tmp           # 1% supersaturation over ice/water
    zoversatw = 0.01 * qsat_tmp_water    # 1% supersaturation over water

    # zrhtest: RH-limited threshold humidity for final correction
    zrhtest = jnp.minimum(specific_humidity_prev / jnp.maximum(qsat_prev, eps), 1.0) * qsat_tmp

    # Heterogeneous onset humidity (only relevant in ice phase)
    zqsp1tmphet_candidate = jnp.minimum(qsat_tmp_water + zoversatw, qsat_tmp * 1.3)
    zqsp1tmphet = jnp.where(lo2, zqsp1tmphet_candidate, 0.0)

    # -------------------------------------------------------------------------
    # 10. Supersaturation increments
    # -------------------------------------------------------------------------
    ztmp1 = (specific_humidity_tmp - qsat_tmp - zoversat) * zqcon          # w.r.t. ice/water
    ztmp2 = (specific_humidity_tmp - qsat_tmp_water - zoversatw) * zqcon   # w.r.t. water
    ztmp3 = (specific_humidity_tmp - zqsp1tmphet) * zqcon                  # w.r.t. heterogeneous onset

    # -------------------------------------------------------------------------
    # 11. Supersaturation condition flags
    # -------------------------------------------------------------------------
    ll1_circ = jnp.array(nic_cirrus == 1)  # constant (not per-point)

    ll2 = specific_humidity_tmp > (qsat_tmp + zoversat)
    ll3 = specific_humidity_tmp > (qsat_tmp_water + zoversatw)
    ll4 = specific_humidity_tmp > zqsp1tmphet
    ll5 = temperature_tmp >= cthomi  # True = mixed-phase (not homogeneous)

    # -------------------------------------------------------------------------
    # 12. Deposition increment (ice cloud cases, lo2=True)
    #     Three mutually exclusive branches:
    #       A: nic_cirrus==1 (or nic_cirrus!=1 but T>=cthomi):  use ztmp1 if ll2
    #       B: nic_cirrus!=1, T<cthomi, not heterogeneous:       use ztmp2 if ll3
    #       C: nic_cirrus!=1, T<cthomi, heterogeneous (ll_het):  use ztmp3 if ll4
    # -------------------------------------------------------------------------
    dep_increment = jnp.zeros_like(deposition_rate)

    # Branch A
    ll6_A = jnp.logical_and(
        lo2,
        jnp.logical_or(
            jnp.logical_and(ll1_circ, ll2),
            jnp.logical_and(~ll1_circ, jnp.logical_and(ll2, ll5)),
        ),
    )
    dep_increment = jnp.where(ll6_A, ztmp1, dep_increment)

    # Branch B: nic_cirrus!=1, T<cthomi (!ll5), not ll_het
    ll6_B = jnp.logical_and(
        lo2,
        jnp.logical_and(
            ~ll1_circ,
            jnp.logical_and(ll3, jnp.logical_and(~ll5, jnp.array(not ll_het))),
        ),
    )
    dep_increment = jnp.where(ll6_B, ztmp2, dep_increment)

    # Branch C: nic_cirrus!=1, T<cthomi (!ll5), ll_het
    ll6_C = jnp.logical_and(
        lo2,
        jnp.logical_and(
            ~ll1_circ,
            jnp.logical_and(ll4, jnp.logical_and(~ll5, jnp.array(ll_het))),
        ),
    )
    dep_increment = jnp.where(ll6_C, ztmp3, dep_increment)

    deposition_rate = deposition_rate + dep_increment

    # -------------------------------------------------------------------------
    # 13. Condensation increment (liquid cloud cases, lo2=False)
    # -------------------------------------------------------------------------
    ll6_liq = jnp.logical_and(~lo2, ll2)
    cnd_increment = jnp.where(ll6_liq, ztmp1, 0.0)
    condensation_rate = condensation_rate + cnd_increment

    # -------------------------------------------------------------------------
    # 14. Final corrections
    #     If the updated q < zrhtest AND q_s(new) <= q_s(t-1),
    #     cap deposition/condensation at (pqp1 - zrhtest) to avoid over-drying.
    # -------------------------------------------------------------------------
    ztmp5 = jnp.maximum(specific_humidity - zrhtest, 0.0)

    ll1_dep = deposition_rate > 0.0
    ll2_cnd = condensation_rate > 0.0
    ll3_rh  = specific_humidity_tmp < zrhtest
    ll4_qs  = qsat_tmp <= qsat_prev

    # Correction for deposition (ice phase)
    ll5_dep = jnp.logical_and(lo2, jnp.logical_and(ll1_dep, jnp.logical_and(ll3_rh, ll4_qs)))
    deposition_rate = jnp.where(ll5_dep, ztmp5, deposition_rate)

    # Correction for condensation (liquid phase)
    ll5_cnd = jnp.logical_and(~lo2, jnp.logical_and(ll2_cnd, jnp.logical_and(ll3_rh, ll4_qs)))
    condensation_rate = jnp.where(ll5_cnd, ztmp5, condensation_rate)

    # -------------------------------------------------------------------------
    # 15. Final updated temperature and specific humidity
    # -------------------------------------------------------------------------
    temperature_tmp = temperature + lvdcp * condensation_rate + lsdcp * deposition_rate
    specific_humidity_tmp = specific_humidity - condensation_rate - deposition_rate

    return (
        condensation_rate,
        deposition_rate,
        temperature_tmp,
        specific_humidity_tmp,
        qsat_tmp,
    )

def freezing_below_238K(
    freezing_condition: jnp.ndarray,    # ld_frz_below_238K
    cloud_cover: jnp.ndarray,           # paclc
    min_cdnc: jnp.ndarray,              # pcdnc_min
    ice_crystal_number: jnp.ndarray,    # picnc
    droplet_freezing_rate: jnp.ndarray, # pqfre
    droplet_number: jnp.ndarray,        # pcdnc
    freezing_rate: jnp.ndarray,         # pfrl
    cloud_ice: jnp.ndarray,             # pxib
    cloud_liquid: jnp.ndarray,          # pxlb
    timestep: float,                    # zdt
    min_liquid_threshold: float         # cqtmin
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Freezing process below 238K for cloud microphysics.

    Overview
    --------
    This routine simulates the freezing of cloud droplets into ice crystals
    below 238K. It updates the ice crystal number concentration (ICNC), cloud
    droplet freezing rate, cloud droplet number concentration (CDNC), freezing
    rate, cloud ice, and cloud liquid water mixing ratios.

    The freezing process is triggered by a boolean mask (`freezing_condition`)
    that identifies grid points where freezing occurs. The routine:
      1. Updates the freezing rate by adding contributions from cloud liquid water.
      2. Transfers cloud liquid water to cloud ice where freezing occurs.
      3. Reduces the cloud liquid water mixing ratio to zero in freezing regions.
      4. Updates the cloud droplet freezing rate and ice crystal number concentration
         based on the available cloud droplet number concentration.
      5. Ensures the cloud droplet number concentration does not fall below a
         minimum threshold (`min_cdnc`).

    Parameters
    ----------
    freezing_condition : jnp.ndarray
        Boolean mask indicating where freezing below 238K occurs `ld_frz_below_238K`.
    cloud_cover : jnp.ndarray
        Cloud cover fraction `paclc` [0..1] .
    min_cdnc : jnp.ndarray
        Minimum cloud droplet number concentration from max radius `pcdnc_min` [1/m^3] .
    ice_crystal_number : jnp.ndarray
        Ice crystal number concentration (ICNC) `picnc` [1/m^3] (INOUT).
    droplet_freezing_rate : jnp.ndarray
        Cloud droplet freezing rate `pqfre` [m^-3 s^-1]  (INOUT).
    droplet_number : jnp.ndarray
        Cloud droplet number concentration (CDNC) `pcdnc` [1/m^3] (INOUT).
    freezing_rate : jnp.ndarray
        Freezing rate `pfrl` [kg/kg] (INOUT).
    cloud_ice : jnp.ndarray
        Cloud ice mixing ratio in the cloudy part of the grid box `pxib` [kg/kg] (INOUT).
    cloud_liquid : jnp.ndarray
        Cloud liquid water mixing ratio in the cloudy part of the grid box `pxlb` [kg/kg] (INOUT).
    timestep : float
        Time step `zdt` [s] .
    min_liquid_threshold : float
        Minimum threshold for cloud liquid water `cqtmin` [kg/kg].
    Returns
    -------
    Updated values of ice_crystal_number, droplet_freezing_rate, droplet_number,
    freezing_rate, cloud_ice, and cloud_liquid.
    """

    # -------------------------------------------------------------------------
    # 1. Update freezing rate by adding contributions from cloud liquid water
    # -------------------------------------------------------------------------
    temp_freezing_rate = freezing_rate + cloud_liquid * cloud_cover
    freezing_rate = jnp.where(freezing_condition, temp_freezing_rate, freezing_rate)

    # -------------------------------------------------------------------------
    # 2. Transfer cloud liquid water to cloud ice where freezing occurs
    # -------------------------------------------------------------------------
    temp_cloud_ice = cloud_ice + cloud_liquid
    cloud_ice = jnp.where(freezing_condition, temp_cloud_ice, cloud_ice)

    # -------------------------------------------------------------------------
    # 3. Reduce cloud liquid water to zero in freezing regions
    # -------------------------------------------------------------------------
    cloud_liquid = jnp.where(freezing_condition, 0.0, cloud_liquid)

    # -------------------------------------------------------------------------
    # 4. Update droplet freezing rate and ice crystal number concentration
    # -------------------------------------------------------------------------
    # Excess droplet number above the minimum threshold
    excess_droplets = jnp.maximum(droplet_number - min_cdnc, 0.0)

    # Update droplet freezing rate
    updated_freezing_rate = droplet_freezing_rate - timestep * excess_droplets
    droplet_freezing_rate = jnp.where(freezing_condition, updated_freezing_rate, droplet_freezing_rate)

    # Update ice crystal number concentration
    updated_ice_crystal_number = ice_crystal_number + excess_droplets
    ice_crystal_number = jnp.where(freezing_condition, updated_ice_crystal_number, ice_crystal_number)

    # -------------------------------------------------------------------------
    # 5. Ensure cloud droplet number concentration does not fall below minimum
    # -------------------------------------------------------------------------
    droplet_number = jnp.where(freezing_condition, min_liquid_threshold, droplet_number)

    return ice_crystal_number, droplet_freezing_rate, droplet_number, freezing_rate, cloud_ice, cloud_liquid

def het_mxphase_freezing(
    freezing_condition: jnp.ndarray,  # Original: ld_mxphase_frz
    pressure: jnp.ndarray,            # Original: papp1
    tke: jnp.ndarray,                 # Original: ptkem1
    vertical_velocity: jnp.ndarray,   # Original: pvervel
    cloud_cover: jnp.ndarray,         # Original: paclc
    bc_soluble_fraction: jnp.ndarray, # Original: pfracbcsol
    bc_insoluble_fraction: jnp.ndarray, # Original: pfracbcinsol
    dust_soluble_fraction: jnp.ndarray, # Original: pfracdusol
    dust_accumulation_fraction: jnp.ndarray, # Original: pfracduai
    dust_coarse_fraction: jnp.ndarray, # Original: pfracduci
    air_density: jnp.ndarray,         # Original: prho
    inv_air_density: jnp.ndarray,     # Original: prho_rcp
    wet_radius_aitken: jnp.ndarray,   # Original: prwetki
    wet_radius_accumulation: jnp.ndarray, # Original: prwetai
    wet_radius_coarse: jnp.ndarray,   # Original: prwetci
    temperature: jnp.ndarray,         # Original: ptp1tmp
    min_cdnc: jnp.ndarray,            # Original: pcdnc_min
    ice_crystal_number: jnp.ndarray,  # Original: picnc (INOUT)
    droplet_number: jnp.ndarray,      # Original: pcdnc (INOUT)
    freezing_rate: jnp.ndarray,       # Original: pfrl (INOUT)
    cloud_ice: jnp.ndarray,           # Original: pxib (INOUT)
    cloud_liquid: jnp.ndarray,        # Original: pxlb (INOUT)
    timestep: float,                  # Original: ztmst
    min_liquid_threshold: float       # Original: cqtmin
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Heterogeneous mixed-phase freezing for cloud microphysics.

    Overview
    --------
    This routine simulates heterogeneous freezing in mixed-phase clouds, including
    contact and immersion freezing by dust and soot aerosols. It updates the ice
    crystal number concentration (ICNC), cloud droplet number concentration (CDNC),
    freezing rate, cloud ice, and cloud liquid water mixing ratios.

    Parameters
    ----------
    freezing_condition : jnp.ndarray
        Boolean mask indicating where heterogeneous freezing occurs (original: ld_mxphase_frz).
    pressure : jnp.ndarray
        Pressure at full levels (t-1) [Pa] (original: papp1).
    tke : jnp.ndarray
        Turbulent kinetic energy (t-1) [m^2/s^2] (original: ptkem1).
    vertical_velocity : jnp.ndarray
        Large-scale vertical velocity [m/s] (original: pvervel).
    cloud_cover : jnp.ndarray
        Cloud cover fraction [0..1] (original: paclc).
    bc_soluble_fraction : jnp.ndarray
        Fraction of BC in all soluble mixed modes (original: pfracbcsol).
    bc_insoluble_fraction : jnp.ndarray
        Fraction of BC in all insoluble modes (original: pfracbcinsol).
    dust_soluble_fraction : jnp.ndarray
        Fraction of dust aerosols in all soluble mixed modes (original: pfracdusol).
    dust_accumulation_fraction : jnp.ndarray
        Fraction of dust in the insoluble accumulation mode (original: pfracduai).
    dust_coarse_fraction : jnp.ndarray
        Fraction of dust in the insoluble coarse mode (original: pfracduci).
    air_density : jnp.ndarray
        Air density [kg/m^3] (original: prho).
    inv_air_density : jnp.ndarray
        Inverse air density [m^3/kg] (original: prho_rcp).
    wet_radius_aitken : jnp.ndarray
        Wet radius of Aitken insoluble mode [m] (original: prwetki).
    wet_radius_accumulation : jnp.ndarray
        Wet radius of accumulation insoluble mode [m] (original: prwetai).
    wet_radius_coarse : jnp.ndarray
        Wet radius of coarse insoluble mode [m] (original: prwetci).
    temperature : jnp.ndarray
        Temperature at (t) [K] (original: ptp1tmp).
    min_cdnc : jnp.ndarray
        Minimum CDNC concentration computed from maximum radius [1/m^3] (original: pcdnc_min).
    ice_crystal_number : jnp.ndarray
        Ice crystal number concentration (ICNC) [1/m^3] (INOUT) (original: picnc).
    droplet_number : jnp.ndarray
        Cloud droplet number concentration (CDNC) [1/m^3] (INOUT) (original: pcdnc).
    freezing_rate : jnp.ndarray
        Freezing rate [kg/kg] (INOUT) (original: pfrl).
    cloud_ice : jnp.ndarray
        Cloud ice mixing ratio in the cloudy part of the grid box [kg/kg] (INOUT) (original: pxib).
    cloud_liquid : jnp.ndarray
        Cloud liquid water mixing ratio in the cloudy part of the grid box [kg/kg] (INOUT) (original: pxlb).
    timestep : float
        Time step [s] (original: ztmst).
    min_liquid_threshold : float
        Minimum threshold for cloud liquid water [kg/kg] (original: cqtmin).

    Returns
    -------
    Updated values of ice_crystal_number, droplet_number, freezing_rate,
    cloud_ice, cloud_liquid, and freezing_rate_number.
    """

    # -------------------------------------------------------------------------
    # 1. Aerosol diffusivity due to Brownian motion
    # -------------------------------------------------------------------------
    # Compute aerosol diffusivity for different modes
    ztmp1 = 1.0 + 1.26 * 6.6e-8 / (wet_radius_aitken + 1e-12) * (p0s1_bg / pressure) * (temperature / tmelt)
    ztmp2 = 1.0 + 1.26 * 6.6e-8 / (wet_radius_accumulation + 1e-12) * (p0s1_bg / pressure) * (temperature / tmelt)
    ztmp3 = 1.0 + 1.26 * 6.6e-8 / (wet_radius_coarse + 1e-12) * (p0s1_bg / pressure) * (temperature / tmelt)

    zeta_air = 1e-5 * (1.718 + 0.0049 * (temperature - tmelt) - 1.2e-5 * (temperature - tmelt) ** 2)

    aerosol_diffusivity_bc = ak * temperature * ztmp1 / (6.0 * pi * zeta_air * (wet_radius_aitken + 1e-12))
    aerosol_diffusivity_bc = jnp.where(wet_radius_aitken < 1e-12, 0.0, aerosol_diffusivity_bc)

    aerosol_diffusivity_dust_accum = ak * temperature * ztmp2 / (6.0 * pi * zeta_air * (wet_radius_accumulation + 1e-12))
    aerosol_diffusivity_dust_accum = jnp.where(wet_radius_accumulation < 1e-12, 0.0, aerosol_diffusivity_dust_accum)

    aerosol_diffusivity_dust_coarse = ak * temperature * ztmp3 / (6.0 * pi * zeta_air * (wet_radius_coarse + 1e-12))
    aerosol_diffusivity_dust_coarse = jnp.where(wet_radius_coarse < 1e-12, 0.0, aerosol_diffusivity_dust_coarse)

    # -------------------------------------------------------------------------
    # 2. Freezing rates (contact and immersion freezing)
    # -------------------------------------------------------------------------
    # Compute mean volume radius of cloud droplets
    droplet_radius = (0.75 * cloud_liquid * air_density / (pi * rhoh2o * droplet_number)) ** (1.0 / 3.0)

    # Contact freezing by dust and soot
    contact_freezing_dust = jnp.minimum(1.0, jnp.maximum(0.0, -(0.1014 * (temperature - tmelt) + 0.3277)))
    contact_freezing_bc = 0.0  # BC contact freezing disabled

    # Immersion freezing by dust and soot
    immersion_freezing_dust = 32.3 * dust_soluble_fraction
    immersion_freezing_bc = 2.91e-3 * bc_soluble_fraction

    # Compute freezing rates
    freezing_rate_contact = (
        cloud_liquid / droplet_number * air_density * 4.0 * pi * droplet_radius * droplet_number * inv_air_density
        * (contact_freezing_dust * (aerosol_diffusivity_dust_accum * dust_accumulation_fraction
                                    + aerosol_diffusivity_dust_coarse * dust_coarse_fraction)
           + contact_freezing_bc * aerosol_diffusivity_bc * bc_insoluble_fraction)
        * (droplet_number + ice_crystal_number)
    )

    freezing_rate_immersion = -(
        (immersion_freezing_dust + immersion_freezing_bc) * air_density / rhoh2o
        * jnp.exp(tmelt - temperature) * jnp.minimum(vertical_velocity - fact_tke * jnp.sqrt(tke) * air_density * grav, 0.0)
    )

    freezing_rate_contact = cloud_liquid * (1.0 - jnp.exp(-freezing_rate_contact / jnp.maximum(cloud_liquid, min_liquid_threshold) * timestep))
    freezing_rate_immersion = cloud_liquid * (1.0 - jnp.exp(-freezing_rate_immersion * cloud_liquid / droplet_number * timestep))

    # Total freezing rate
    total_freezing_rate = freezing_rate_contact + freezing_rate_immersion
    total_freezing_rate = jnp.clip(total_freezing_rate, 0.0, cloud_liquid)

    # Freezing rate for number concentration
    freezing_rate_number = droplet_number * total_freezing_rate / (cloud_liquid + 1e-12)
    freezing_rate_number = jnp.maximum(freezing_rate_number, 0.0)

    # -------------------------------------------------------------------------
    # 3. Update cloud properties
    # -------------------------------------------------------------------------
    freezing_rate = jnp.where(freezing_condition, total_freezing_rate, freezing_rate)
    freezing_rate_number = jnp.where(freezing_condition, freezing_rate_number, 0.0)

    droplet_number = jnp.where(
        freezing_condition,
        jnp.maximum(droplet_number - freezing_rate_number, min_cdnc),
        droplet_number
    )

    ice_crystal_number = jnp.where(
        freezing_condition,
        jnp.maximum(ice_crystal_number + freezing_rate_number, min_liquid_threshold),
        ice_crystal_number
    )

    cloud_liquid = jnp.where(
        freezing_condition,
        cloud_liquid - freezing_rate,
        cloud_liquid
    )

    cloud_ice = jnp.where(
        freezing_condition,
        cloud_ice + freezing_rate,
        cloud_ice
    )

    return ice_crystal_number, droplet_number, freezing_rate, cloud_ice, cloud_liquid, freezing_rate_number

def WBF_process(
    wbf_mask: jnp.ndarray,                 # Original: ld_WBF
    cloud_fraction: jnp.ndarray,           # Original: paclc
    lsdcp: jnp.ndarray,                    # Original: plsdcp  (Ls/cpd)
    lvdcp: jnp.ndarray,                    # Original: plvdcp  (Lv/cpd)
    cdnc: jnp.ndarray,                     # Original: pcdnc   (INOUT) CDNC [1/m^3]
    cloud_liquid_in_cloud: jnp.ndarray,    # Original: pxlb    (INOUT) in-cloud liquid [kg/kg]
    cloud_ice_in_cloud: jnp.ndarray,       # Original: pxib    (INOUT) in-cloud ice [kg/kg]
    cloud_liquid_tendency: jnp.ndarray,    # Original: pxlte   (INOUT) liquid tendency [kg/kg/s]
    cloud_ice_tendency: jnp.ndarray,       # Original: pxite   (INOUT) ice tendency [kg/kg/s]
    temp_tendency: jnp.ndarray,            # Original: ptte    (INOUT) temperature tendency [K/s]
    dt: jnp.ndarray                        # Microphysics timestep (used to form ztmst_rcp)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Warm-bridge/freeze (WBF) process: transfer of in-cloud liquid to ice under WBF conditions.

    JAX port of Fortran subroutine `WBF_process` (mo_cloud_microphysics_2m).

    Overview
    --------
    Implements the WBF phase-transfer step used in the ICON/ECHAM 2‑moment microphysics.
    Where the WBF condition holds (wbf_mask), in-cloud liquid is transferred to in-cloud ice,
    tendencies for liquid/ice and temperature are adjusted to reflect the transfer and latent-heat
    effects, and cloud droplet number concentration (CDNC) is reset to a minimum value.

    Steps
    -----
    1. Compute transfer proxy ztmp1 = ztmst_rcp * pxlb * paclc (Fortran: ztmst_rcp*pxlb*paclc).
    2. Subtract ztmp1 from the cloud-liquid tendency:
         pxlte <- pxlte - ztmp1 (applied where wbf_mask True).
    3. Add ztmp1 to the cloud-ice tendency:
         pxite <- pxite + ztmp1 (applied where wbf_mask True).
    4. Apply thermodynamic correction to temperature tendency:
         ptte <- ptte + (plsdcp - plvdcp) * ztmp1 (applied where wbf_mask True).
    5. Enforce minimum CDNC where WBF applies:
         pcdnc <- cqtmin (Fortran MERGE(cqtmin, pcdnc, ld_WBF)).
    6. Transfer remaining in-cloud liquid to in-cloud ice and zero liquid:
         pxib <- pxib + pxlb ; pxlb <- 0  (applied where wbf_mask True).

    Parameters
    ----------
    wbf_mask : jnp.ndarray
        Logical mask where the WBF process is active. (Fortran: ld_WBF)
    cloud_fraction : jnp.ndarray
        Cloud cover fraction in the layer. (Fortran: paclc)
    lsdcp : jnp.ndarray
        Latent heat of sublimation divided by cpd (Ls/cpd). (Fortran: plsdcp)
    lvdcp : jnp.ndarray
        Latent heat of vaporization divided by cpd (Lv/cpd). (Fortran: plvdcp)
    cdnc : jnp.ndarray
        Cloud droplet number concentration (pcdnc) [1/m^3] (INOUT).
    cloud_liquid_in_cloud : jnp.ndarray
        In-cloud cloud liquid mixing ratio (pxlb) [kg/kg] (INOUT).
    cloud_ice_in_cloud : jnp.ndarray
        In-cloud cloud ice mixing ratio (pxib) [kg/kg] (INOUT).
    cloud_liquid_tendency : jnp.ndarray
        Tendency of in-cloud liquid (pxlte) [kg/kg/s] (INOUT).
    cloud_ice_tendency : jnp.ndarray
        Tendency of in-cloud ice (pxite) [kg/kg/s] (INOUT).
    temp_tendency : jnp.ndarray
        Temperature tendency (ptte) [K/s] (INOUT).
    dt : jnp.ndarray or float
        Microphysics timestep ztmst [s] used to form ztmst_rcp = 1/ztmst.

    Returns
    -------
    cdnc :
        Updated cloud droplet number concentration (pcdnc) [1/m^3].
    cloud_liquid_in_cloud :
        Updated in-cloud liquid mixing ratio (pxlb) [kg/kg].
    cloud_ice_in_cloud :
        Updated in-cloud ice mixing ratio (pxib) [kg/kg].
    cloud_liquid_tendency :
        Updated liquid tendency (pxlte) [kg/kg/s].
    cloud_ice_tendency :
        Updated ice tendency (pxite) [kg/kg/s].
    temp_tendency :
        Updated temperature tendency (ptte) [K/s].

    Notes
    -----
    - ztmst_rcp (Fortran ztmst_rcp) is obtained from microphysics_dt_constants(dt).
    - All operations are vectorised and preserve input shapes; values are only changed
      where wbf_mask is True.
    - cqtmin is used as the minimum CDNC (Fortran constant cqtmin).
    """
    
    # get reciprocal timestep constant (ztmst_rcp = 1 / ztmst)
    _, ztmst_rcp, *_ = microphysics_dt_constants(dt)

    # ztmp1 = ztmst_rcp * pxlb * paclc  (evap / WBF proxy)
    ztmp1 = ztmst_rcp * cloud_liquid_in_cloud * cloud_fraction

    # cloud liquid tendency: pxlte <- MERGE(pxlte - ztmp1, pxlte, ld_WBF)
    cloud_liquid_tendency = jnp.where(wbf_mask, cloud_liquid_tendency - ztmp1, cloud_liquid_tendency)

    # cloud ice tendency: pxite <- MERGE(pxite + ztmp1, pxite, ld_WBF)
    cloud_ice_tendency = jnp.where(wbf_mask, cloud_ice_tendency + ztmp1, cloud_ice_tendency)

    # temperature tendency: ptte <- MERGE(ptte + (plsdcp - plvdcp)*ztmp1, ptte, ld_WBF)
    temp_tendency = jnp.where(wbf_mask, temp_tendency + (lsdcp - lvdcp) * ztmp1, temp_tendency)

    # cdnc <- MERGE(cqtmin, pcdnc, ld_WBF)  (set to minimum where WBF occurs)
    cdnc = jnp.where(wbf_mask, cqtmin, cdnc)

    # pxib <- MERGE(pxib + pxlb, pxib, ld_WBF)  (transfer liquid mass to ice)
    cloud_ice_in_cloud = jnp.where(wbf_mask, cloud_ice_in_cloud + cloud_liquid_in_cloud, cloud_ice_in_cloud)

    # pxlb <- MERGE(0.0, pxlb, ld_WBF)  (zero liquid where WBF occurs)
    cloud_liquid_in_cloud = jnp.where(wbf_mask, 0.0, cloud_liquid_in_cloud)

    return (
        cdnc,
        cloud_liquid_in_cloud,
        cloud_ice_in_cloud,
        cloud_liquid_tendency,
        cloud_ice_tendency,
        temp_tendency,
    )

def precip_formation_warm(
    warm_precip_mask: jnp.ndarray,
    autoconversion_factor: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    minimum_cloud_precip_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    rain_water: jnp.ndarray,
    minimum_droplet_number: jnp.ndarray,
    droplet_number: jnp.ndarray,
    cloud_water: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Warm-rain precipitation formation for the 2-moment microphysics scheme.

    This is a JAX port of the ECHAM6/ICON Fortran routine `precip_formation_warm`
    for the "nauto == 2" branch: Khairoutdinov & Kogan (2000).

    The routine computes, in a grid-box (or column slice) of independent points:
      1) Autoconversion of cloud liquid water to rain (mass transfer),
      2) Accretion of cloud water by rain falling from above (zrac1),
      3) Accretion of cloud water by newly formed rain in the same grid-box (zrac2),
      4) Associated rain production rates (mass and number),
      5) Updated cloud droplet number concentration (droplet_number),
      6) Updated in-cloud cloud water (cloud_water).

    Parameters
    ----------
    warm_precip_mask : bool array
        Equivalent to ECHAM `ld_prcp_warm`. True where warm precipitation
        formation is physically allowed (e.g., T > 0C and clouds present).
    autoconversion_factor : array
        Equivalent to ECHAM `pauloc`. Fraction of gridbox participating in
        accretion with newly formed condensate (unitless).
    cloud_fraction : array
        Equivalent to ECHAM `paclc`. Cloud cover fraction (0..1).
    minimum_cloud_precip_fraction : array
        Equivalent to ECHAM `pclcstar`. min(cloud cover, precipitation cover).
        Used to weight accretion with rain from above.
    air_density : array
        Equivalent to ECHAM `prho` [kg/m^3].
    rain_water : array
        Equivalent to ECHAM `pxrp1`. Rain mixing ratio [kg/kg] (at time t),
        representing "rain from above" for accretion term zrac1.
    minimum_droplet_number : array
        Equivalent to ECHAM `pcdnc_min`. Minimum allowed droplet number
        computed from max droplet radius [1/m^3].
    droplet_number : array
        Equivalent to ECHAM `pcdnc`. Cloud droplet number concentration [1/m^3].
        (INOUT) Updated to reflect loss due to autoconversion/accretion.
    cloud_water : array
        Equivalent to ECHAM `pxlb`. In-cloud liquid water mixing ratio [kg/kg].
        (INOUT) Decreased by autoconversion + accretion.
    dt : array or scalar
        Equivalent to ECHAM `ztmst`. Microphysics timestep [s].

    Returns
    -------
    droplet_number : array
        Updated cloud droplet number concentration [1/m^3].
    cloud_water : array
        Updated in-cloud cloud water [kg/kg].
    autoconversion_rate_in_cloud : array
        Rain formation rate in cloudy part (for scavenging) [kg/kg].
    autoconversion_rate : array
        Gridbox-mean rain formation rate [kg/kg].
    droplet_number_removal_rate : array
        Rain formation rate for number concentration [1/m^3].

    Notes mapping to ECHAM names (nauto==2 branch)
    -----------------------------------------------
    zraut    : autoconversion mass removed from cloud water [kg/kg]
    zrac1    : accretion with rain from above [kg/kg]
    zrac2    : accretion with newly formed rain in gridbox [kg/kg]
    autoconversion_rate     : paclc*(zraut+zrac2) + pclcstar*zrac1
    autoconversion_rate_in_cloud : zraut+zrac1+zrac2 (only where ld_prcp_warm)
    droplet_number_removal_rate    : (zraut+zrac1+zrac2)/(old_cloud_water+eps) (only where ld_prcp_warm)
    """
    # -------------------------------------------------------------------------
    # Allocate outputs (same shapes as cloud_water)
    # -------------------------------------------------------------------------
    autoconversion_rate_in_cloud = jnp.zeros_like(cloud_water)  # For in-cloud scavenging
    autoconversion_rate = jnp.zeros_like(cloud_water)      # Rain formation rate [kg/kg]
    droplet_number_removal_rate = jnp.zeros_like(cloud_water)     # Number formation rate proxy [1/m^3]
    accretion_rate_from_above = jnp.zeros_like(cloud_water)      # zrac1: accretion with rain from above [kg/kg]
    accretion_rate_in_cloud = jnp.zeros_like(cloud_water)        # zrac2: accretion with newly formed rain [kg/kg]

    # Local process rates (mass increments) [kg/kg]
    zraut = jnp.zeros_like(cloud_water)     # autoconversion
    # zrautself not used yet (present in Fortran); kept for completeness
    # zrautself = jnp.zeros_like(cloud_water)

    # -------------------------------------------------------------------------
    # 1) Autoconversion (Khairoutdinov & Kogan 2000)
    # -------------------------------------------------------------------------

    # Here, `droplet_number` is pcdnc and `cloud_water` is pxlb.
    ztmp1 = ccraut * 1350.0 * (1e-6 * droplet_number) ** (-1.79)

    # The expression below is a time-integrated sink form used in the Fortran.
    # It is constructed so that zraut is bounded by cloud_water (after MIN).
    ztmp1 = cloud_water * (
        1.0
        - (
            1.0
            + dt * exm1_1 * ztmp1 * cloud_water ** exm1_1
        ) ** exp_1
    )

    # Ensure autoconversion cannot remove more liquid than exists.
    ztmp1 = jnp.minimum(cloud_water, ztmp1)

    # Apply physical mask: only do it where warm precip can form.
    zraut = jnp.where(warm_precip_mask, ztmp1, 0.0)

    # Update in-cloud liquid water after autoconversion.
    # Keep original pre-autoconversion cloud water for later use in droplet_number_removal_rate.
    cloud_water_before = cloud_water
    cloud_water = jnp.where(warm_precip_mask, cloud_water - zraut, cloud_water)

    # -------------------------------------------------------------------------
    # 2) Accretion with rain from above (zrac1)
    # -------------------------------------------------------------------------
    #   A fraction of cloud_water is collected by existing rain (rain_water).
    #   The term exp(-3.7*dt*rain_water) acts like a survival fraction.
    ztmp1 = jnp.exp(-3.7 * dt * rain_water)
    ztmp1 = cloud_water * (1.0 - ztmp1)
    zrac1 = jnp.where(warm_precip_mask, ztmp1, 0.0)
    accretion_rate_from_above = zrac1  # output: zrac1 [kg/kg]

    # Remove accreted cloud water
    cloud_water = cloud_water - zrac1

    # -------------------------------------------------------------------------
    # 3) Accretion with newly formed rain inside the grid box (zrac2)
    # -------------------------------------------------------------------------
    # The exponent uses: autoconversion_factor (pauloc), air_density (prho),
    # and the newly formed rain amount (zraut).
    ztmp1 = -3.7 * dt * autoconversion_factor * air_density * zraut
    ztmp1 = jnp.where(warm_precip_mask, ztmp1, 0.0)  # MERGE
    ztmp1 = cloud_water * (1.0 - jnp.exp(ztmp1))
    zrac2 = jnp.where(warm_precip_mask, ztmp1, 0.0)
    accretion_rate_in_cloud = zrac2  # output: zrac2 [kg/kg]

    # Remove further accreted cloud water
    cloud_water = cloud_water - zrac2

    # -------------------------------------------------------------------------
    # 4) Gridbox-mean rain production rate (mass): autoconversion_rate
    # -------------------------------------------------------------------------
    #   - zraut and zrac2 are weighted by cloud fraction (in-cloud processes).
    #   - zrac1 uses minimum_cloud_precip_fraction (precip cover coupling).
    autoconversion_rate = cloud_fraction * (zraut + zrac2) + minimum_cloud_precip_fraction * zrac1

    # -------------------------------------------------------------------------
    # 5) In-cloud scavenging rate output: autoconversion_rate_in_cloud
    # -------------------------------------------------------------------------
    ztmp1 = zraut + zrac1 + zrac2
    autoconversion_rate_in_cloud = jnp.where(warm_precip_mask, ztmp1, 0.0)

    # -------------------------------------------------------------------------
    # 6) Droplet-number impact of autoconversion/accretion: droplet_number_removal_rate and updated pcdnc
    # -------------------------------------------------------------------------
    droplet_number_removal_rate = jnp.where(
        warm_precip_mask,
        (zraut + zrac1 + zrac2) / (cloud_water_before + eps),
        0.0,
    )

    # Only limit droplet number when cloud water is still meaningful (> cqtmin).
    ll1 = jnp.logical_and(warm_precip_mask, cloud_water > cqtmin)

    # Enforce a minimum allowed droplet number (pcdnc_min) only when ll1 is true.
    min_allowed = jnp.where(ll1, minimum_droplet_number, 0.0)

    # Available droplet number above the minimum
    available = droplet_number - min_allowed

    # "Requested" droplet reduction based on droplet_number_removal_rate proxy:
    requested = droplet_number * droplet_number_removal_rate

    # Actual reduction is limited by what is available above minimum
    droplet_number_removal_rate = jnp.where(warm_precip_mask, jnp.minimum(available, requested), 0.0)

    # Update droplet number concentration, keep >= cqtmin
    droplet_number_new = jnp.maximum(droplet_number - droplet_number_removal_rate, cqtmin)
    droplet_number = jnp.where(warm_precip_mask, droplet_number_new, droplet_number)

    return droplet_number, cloud_water, autoconversion_rate_in_cloud, autoconversion_rate, droplet_number_removal_rate, accretion_rate_from_above, accretion_rate_in_cloud

def precip_formation_cold(
    cloud_mask: jnp.ndarray,                      # ld_cc
    autoconversion_factor: jnp.ndarray,            # pauloc
    cloud_fraction: jnp.ndarray,                   # paclc
    minimum_cloud_precip_fraction: jnp.ndarray,    # pclcstar
    inverse_air_density: jnp.ndarray,              # pqrho  (m^3/kg)  NOTE: in ICON this is 1/prho
    inverse_air_density_rcp: jnp.ndarray,          # prho_rcp (should be 1/prho too; keep both for exact port)
    temperature: jnp.ndarray,                      # ptp1tmp [K]
    dynamic_viscosity: jnp.ndarray,                # pviscos
    snow_mass_mmr_from_above: jnp.ndarray,         # pxsp1  (snow mass mixing ratio from above) [kg/kg] (name inferred)
    air_density: jnp.ndarray,                      # prho [kg/m^3]
    minimum_droplet_number: jnp.ndarray,           # pcdnc_min [1/m^3]
    ice_number: jnp.ndarray,                       # picnc [1/m^3] (INOUT)
    droplet_number: jnp.ndarray,                   # pcdnc [1/m^3] (INOUT)
    snow_rate_in_cloud: jnp.ndarray,               # pmrateps [kg/kg] (INOUT)  (in-cloud snow formation used for scavenging)
    in_cloud_ice: jnp.ndarray,                     # pxib [kg/kg] (INOUT)
    in_cloud_liquid: jnp.ndarray,                  # pxlb [kg/kg] (INOUT)
    dt: jnp.ndarray,                               # ztmst [s]
) -> tuple[
    jnp.ndarray,  # ice_number
    jnp.ndarray,  # droplet_number
    jnp.ndarray,  # snow_rate_in_cloud (pmrateps)
    jnp.ndarray,  # in_cloud_ice
    jnp.ndarray,  # in_cloud_liquid
    jnp.ndarray,  # psprn  snow number formation [1/m^3]
    jnp.ndarray,  # psacl  snow-droplet accretion mass [kg/kg]
    jnp.ndarray,  # psacln snow-droplet accretion number [1/m^3]
    jnp.ndarray,  # pmsnowacl in-cloud accretion mass for scavenging (?) [kg/kg]
    jnp.ndarray,  # pspr  grid-mean snow formation mass [kg/kg]
]:
    """
    Cold-phase precipitation formation for the ICON/ECHAM 2-moment scheme.

    JAX port of Fortran `precip_formation_cold` (mo_cloud_microphysics_2m).

    Processes represented (subset as in the Fortran):
      1) Aggregation of ice crystals to snow (zsaut): ice mass -> snow mass.
      2) Riming: accretion of cloud droplets by snow (zsaclin -> psacl, psacln), with
         collision efficiency based on Stokes/Reynolds numbers.
      3) Accretion of cloud ice by snow (zsaci).
      4) Diagnostics of snow formation rates (pspr, pmrateps) and ice-number loss by
         "break-up" / self-collection style terms (psprn).

    Notes / important caveats
    -------------------------
    - This is a direct translation of a complex Fortran block with many "MERGE"/mask
      operations and temporary variables. It should be validated against the Fortran.
    - The Fortran contains optional secondary ice production (lsecprod). That block is
      not included here (set zsecprod=0), matching the common default (off).
    - `pxsp1` meaning: treated here as snow mass mixing ratio entering from above.
    - This routine expects in-cloud condensates (pxib/pxlb) as in ICON/ECHAM conventions.
    """

    # Allocate outputs
    pspr = jnp.zeros_like(in_cloud_ice)     # snow formation (grid-mean) [kg/kg]
    psprn = jnp.zeros_like(in_cloud_ice)    # snow formation for number conc [1/m^3]
    pmsnowacl = jnp.zeros_like(in_cloud_ice)  # in-cloud snow-droplet accretion mass [kg/kg]
    psacl = jnp.zeros_like(in_cloud_ice)    # snow-droplet accretion mass (grid-mean) [kg/kg]
    psacln = jnp.zeros_like(in_cloud_ice)   # snow-droplet accretion number [1/m^3]

    # Local variables
    zxibold = jnp.maximum(in_cloud_ice, eps)  # store pxib with security for later use
    zsaut = jnp.zeros_like(in_cloud_ice)      # aggregation mass [kg/kg]
    zxsp2 = jnp.zeros_like(in_cloud_ice)      # snow formed inside box (mass conc proxy) [??]
    zsaclin = jnp.zeros_like(in_cloud_ice)    # in-cloud droplet mass accreted by snow [kg/kg]
    zsaci = jnp.zeros_like(in_cloud_ice)      # ice accreted by snow [kg/kg]
    zsecprod = jnp.zeros_like(in_cloud_ice)   # secondary ice production mass [kg/kg] (not implemented here)

    # ---------------------------------------------------------------------
    # 0) Early mask: only proceed where there is cloud and enough ice
    # ---------------------------------------------------------------------
    ll1 = jnp.logical_and(cloud_mask, in_cloud_ice > cqtmin)

    # If ll1 is false everywhere, Fortran returns early. In JAX we just mask.
    # (no-op if all masked)
    # ---------------------------------------------------------------------
    # 1) Compute effective ice-crystal "size" zris based on effective radius
    # ---------------------------------------------------------------------
    # Convert in-cloud ice from kg/kg to in-cloud g/m^3: 1000*pxib*prho
    ice_gm3 = 1000.0 * in_cloud_ice * air_density

    # eff_ice_crystal_radius expects (ice_gm3, icnc). If you already have such a helper,
    # call it; otherwise this will need to be implemented.
    zrieff = eff_ice_crystal_radius(ice_gm3, ice_number)  # [micron] typically (scheme-dependent)

    # Clip effective radius bounds
    zrieff = jnp.minimum(jnp.maximum(zrieff, ceffmin), ceffmax)

    # Compute zrih then zris = 1e-6 * zrih**(1/3)
    zrih = -2261.0 + jnp.sqrt(5113188.0 + 2809.0 * zrieff**3)
    zris = 1.0e-6 * (zrih ** (1.0 / 3.0))

    # Fortran MERGE(..., 1., ll1): just ensure non-zero where masked off
    zris = jnp.where(ll1, zris, 1.0)

    # ---------------------------------------------------------------------
    # 2) Temperature-dependent collision efficiency for aggregation
    # ---------------------------------------------------------------------
    zcolleffi = jnp.exp(fact_coll_eff * (temperature - tmelt))
    zcolleffi = jnp.where(ll1, zcolleffi, 0.0)

    # ---------------------------------------------------------------------
    # 3) Aggregation of ice crystals to snow (zsaut)
    # ---------------------------------------------------------------------
    zc1 = 17.5 / crhoi * air_density * (inverse_air_density ** 0.33)

    # zdt2 = -6/zc1 * log10(1e4*zris); then ztmp1 = ccsaut / zdt2
    zdt2 = (-6.0 / jnp.maximum(zc1, eps)) * jnp.log10(1.0e4 * jnp.maximum(zris, eps))
    ztmp1 = ccsaut / jnp.maximum(zdt2, eps)
    ztmp1 = jnp.where(ll1, ztmp1, 0.0)

    # zsaut = pxib*(1 - 1/(1+ ztmp1*dt*pxib))
    zsaut = in_cloud_ice * (1.0 - 1.0 / (1.0 + ztmp1 * dt * in_cloud_ice))

    # update in_cloud_ice = pxib - zsaut (only where ll1)
    zxibold2 = in_cloud_ice  # store pxib pre-update for later
    in_cloud_ice = jnp.where(ll1, in_cloud_ice - zsaut, in_cloud_ice)

    # snow formed inside the grid box (mass concentration proxy)
    zxsp2 = autoconversion_factor * air_density * zsaut
    zxsp2 = jnp.where(ll1, zxsp2, 0.0)

    # total snow mass mixing ratio available (from above + newly formed)
    zxsp = snow_mass_mmr_from_above + zxsp2

    # ---------------------------------------------------------------------
    # 4) Riming: accretion of snow with cloud droplets (zsaclin, psacl, psacln)
    # ---------------------------------------------------------------------
    ll2 = jnp.logical_and(
        ll1,
        jnp.logical_and(
            zxsp > cqtmin,
            jnp.logical_and(in_cloud_liquid > cqtmin, droplet_number >= minimum_droplet_number),
        ),
    )

    # droplet mean radius proxy (zdw)
    zdw = (6.0 * pirho_rcp * air_density * in_cloud_liquid / jnp.maximum(droplet_number, eps)) ** (1.0 / 3.0)
    zdw = jnp.maximum(zdw, 1.0e-6)

    zudrop = 1.19e4 * 2500.0 * zdw**2 * (1.3 * inverse_air_density_rcp) ** 0.35

    # planar snowflake max dimension (constant)
    zdplanar = 447.0e-6

    zusnow = 2.34 * (100.0 * zdplanar) ** 0.3 * (1.3 * inverse_air_density_rcp) ** 0.35

    zstokes = 2.0 * rgrav * (zusnow - zudrop) * zudrop / zdplanar
    zstokes = jnp.maximum(zstokes, cqtmin)

    zrey = air_density * zdplanar * zusnow / jnp.maximum(dynamic_viscosity, eps)
    zrey = jnp.maximum(zrey, cqtmin)

    ll3 = zrey <= 5.0
    ll4 = jnp.logical_and(zrey > 5.0, zrey < 40.0)
    ll5 = zrey >= 40.0

    zstcrit = jnp.ones_like(zrey)
    zstcrit = jnp.where(ll3, 5.52 * zrey ** (-1.12), zstcrit)
    zstcrit = jnp.where(ll4, 1.53 * zrey ** (-0.325), zstcrit)

    zcsacl = 0.2 * (jnp.log10(zstokes) - jnp.log10(zstcrit) - 2.236) ** 2
    zcsacl = jnp.minimum(zcsacl, 1.0 - cqtmin)
    zcsacl = jnp.maximum(zcsacl, 0.0)
    zcsacl = jnp.sqrt(1.0 - zcsacl)

    ll6 = jnp.logical_and(ll5, zstokes <= 0.06)
    ll7 = jnp.logical_and(ll5, jnp.logical_and(zstokes > 0.06, zstokes <= 0.25))
    ll8 = jnp.logical_and(ll5, jnp.logical_and(zstokes > 0.25, zstokes <= 1.00))

    zcsacl = jnp.where(ll5, (zstokes + 1.1) ** 2 / (zstokes + 1.6) ** 2, zcsacl)
    zcsacl = jnp.where(ll6, 1.034 * zstokes ** 1.085, zcsacl)
    zcsacl = jnp.where(ll7, 0.787 * zstokes ** 0.988, zcsacl)
    zcsacl = jnp.where(ll8, 0.7475 * jnp.log10(zstokes) + 0.65, zcsacl)

    zcsacl = jnp.clip(zcsacl, 0.01, 1.0)
    zcsacl = jnp.where(ll2, zcsacl, 0.0)

    # lambda_snow proxy and collection coefficient
    zlamsm = cons4 * zxsp ** 0.8125
    ztmp2 = pi * cn0s * 3.078 * zlamsm * (inverse_air_density ** 0.5)
    ztmp2 = jnp.where(ll2, ztmp2, 0.0)

    # integrated riming sink on liquid water
    survival = jnp.exp(-dt * ztmp2 * zcsacl)
    zsaclin = in_cloud_liquid * (1.0 - survival)
    zsaclin = jnp.where(ll2, zsaclin, 0.0)

    # update in_cloud_liquid (remove accreted)
    pxlb_before = in_cloud_liquid
    in_cloud_liquid = jnp.where(ll2, in_cloud_liquid - zsaclin, in_cloud_liquid)

    # grid-mean accretion mass
    psacl = jnp.where(ll2, cloud_fraction * zsaclin, 0.0)

    # number accretion (droplet number loss), only if liquid remains meaningful
    ll2b = in_cloud_liquid > cqtmin
    psacln_raw = droplet_number * zsaclin / (pxlb_before + eps)
    psacln_raw = jnp.minimum(psacln_raw, droplet_number - minimum_droplet_number)
    psacln_raw = jnp.maximum(psacln_raw, 0.0)
    psacln = jnp.where(ll2b, psacln_raw, 0.0)

    # apply number loss to droplets where ll1 (as in Fortran MERGE)
    droplet_number = jnp.where(ll1, droplet_number - psacln, droplet_number)
    pmsnowacl = jnp.where(ll1, zsaclin, 0.0)

    # ---------------------------------------------------------------------
    # 5) Accretion of snow with ice crystals (zsaci)
    # ---------------------------------------------------------------------
    ll1b = jnp.logical_and(cloud_mask, in_cloud_ice > cqtmin)
    ll2 = jnp.logical_and(ll1b, zxsp > cqtmin)

    zlamsm = cons4 * zxsp ** 0.8125
    ztmp1 = pi * cn0s * 3.078 * zlamsm * (inverse_air_density ** 0.5)
    survival = jnp.exp(-dt * ztmp1 * zcolleffi)
    zsaci = in_cloud_ice * (1.0 - survival)
    zsaci = jnp.where(ll2, zsaci, 0.0)

    in_cloud_ice = in_cloud_ice - zsaci

    # ---------------------------------------------------------------------
    # 6) Snow formation mass outputs (grid-mean + in-cloud scavenging)
    # ---------------------------------------------------------------------
    pspr = jnp.where(ll1b, cloud_fraction * (zsaut + zsaci), 0.0)

    snow_rate_in_cloud = jnp.where(ll1b, (zsaut + zsaci), snow_rate_in_cloud)

    # ---------------------------------------------------------------------
    # 7) Ice-number change due to (aggregation + accretion + self-collection - secprod)
    # ---------------------------------------------------------------------
    ll_ice_num = jnp.logical_and(
        cloud_mask,
        jnp.logical_and(in_cloud_ice > epsec, ice_number >= icemin),
    )

    zxibold_sec = jnp.maximum(zxibold2, 0.0)  # Fortran zxibold used here
    zsprn1 = ice_number * (zsaci + zsaut) / (zxibold_sec + eps)
    zself = 0.5 * dt * zc1 * ice_number * in_cloud_ice
    zsecprodn = mi0_rcp * air_density * zsecprod

    psprn_val = zsprn1 + zself - zsecprodn
    psprn_val = jnp.minimum(psprn_val, ice_number)
    psprn = jnp.where(ll_ice_num, psprn_val, 0.0)

    ice_number_new = jnp.maximum(ice_number - psprn, cqtmin)
    ice_number = jnp.where(ll_ice_num, ice_number_new, ice_number)

    return (
        ice_number,
        droplet_number,
        snow_rate_in_cloud,
        in_cloud_ice,
        in_cloud_liquid,
        psprn,
        psacl,
        psacln,
        pmsnowacl,
        pspr,
    )

def update_precip_fluxes(
    cloud_fraction: jnp.ndarray,            # Original: paclc
    pressure_thickness: jnp.ndarray,        # Original: pdp
    rain_evap_mmr: jnp.ndarray,             # Original: pevp (evaporation of rain, kg/kg)
    lsdcp: jnp.ndarray,                     # Original: plsdcp
    lvdcp: jnp.ndarray,                     # Original: plvdcp
    rain_formation: jnp.ndarray,            # Original: prpr
    snow_accretion: jnp.ndarray,            # Original: psacl
    snow_formation: jnp.ndarray,            # Original: pspr
    snow_sublimation_mmr: jnp.ndarray,      # Original: psub (kg/kg)
    temp_tmp: jnp.ndarray,                  # Original: ptp1tmp (K)
    ice_flux_from_above: jnp.ndarray,       # Original: pxiflux
    precip_cover: jnp.ndarray,              # Original: pclcpre (INOUT)
    rain_flux: jnp.ndarray,                 # Original: prfl (INOUT) [kg/m2/s]
    snow_flux: jnp.ndarray,                 # Original: psfl (INOUT) [kg/m2/s]
    snow_melt: jnp.ndarray,                 # Original: psmlt (INOUT) [kg/kg]
    dt: jnp.ndarray,                        # microphysics timestep used to form zcons2
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,  # updated inout: precip_cover, rain_flux, snow_flux, snow_melt
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray   # out: pfevapr, pfrain, pfsnow, pfsubls
]:
    """
    Update precipitation fluxes entering/leaving a layer.

    The routine computes, in a grid-box (or column slice) of independent points:
      1) Rain and snow mass produced in this level (autoconversion/accretion/aggregation).
      2) Top-level melting of incoming ice into rain where temperature permits.
      3) Update of the precip-covered fraction due to falling hydrometeors.
      4) In-cloud rain/snow fluxes (pfrain, pfsnow) and area-integrated evaporation/sublimation.
      5) Update of column rain_flux / snow_flux by adding produced flux and removing evaporated/sublimated mass.
      6) Diagnostic outputs of surface-area-integrated evaporation/sublimation and in-cloud fluxes.

    Parameters
    ----------
    cloud_fraction : jnp.ndarray
        paclc, cloud cover fraction (0..1).
    pressure_thickness : jnp.ndarray
        pdp, layer pressure thickness [Pa].
    rain_evap_mmr : jnp.ndarray
        pevp, rain evaporation expressed as mixing-ratio [kg/kg].
    lsdcp : jnp.ndarray
        plsdcp, latent heat of sublimation / cpd [K].
    lvdcp : jnp.ndarray
        plvdcp, latent heat of vaporisation / cpd [K].
    rain_formation : jnp.ndarray
        prpr, rain production rate (grid-mean) [kg/kg].
    snow_accretion : jnp.ndarray
        psacl, snow accretion mass (grid-mean) [kg/kg].
    snow_formation : jnp.ndarray
        pspr, snow formation mass (grid-mean) [kg/kg].
    snow_sublimation_mmr : jnp.ndarray
        psub, snow sublimation expressed as mixing-ratio [kg/kg].
    temp_tmp : jnp.ndarray
        ptp1tmp, layer temperature used for melting decisions [K].
    ice_flux_from_above : jnp.ndarray
        pxiflux, falling-ice mass flux entering from above [kg/m^2/s].
    precip_cover : jnp.ndarray
        pclcpre (INOUT), precip-covered fraction (0..1).
    rain_flux : jnp.ndarray
        prfl (INOUT), column rain mass flux [kg/m^2/s].
    snow_flux : jnp.ndarray
        psfl (INOUT), column snow mass flux [kg/m^2/s].
    snow_melt : jnp.ndarray
        psmlt (INOUT), accumulated melting diagnostic [kg/kg].
    dt : jnp.ndarray
        Microphysics timestep (used to form zcons2 = dt * rgrav) [s].

    Returns
    -------
    precip_cover : jnp.ndarray
        Updated precip-covered fraction (pclcpre) [0..1].
    rain_flux : jnp.ndarray
        Updated column rain flux (prfl) [kg/m^2/s].
    snow_flux : jnp.ndarray
        Updated column snow flux (psfl) [kg/m^2/s].
    snow_melt : jnp.ndarray
        Updated accumulated snow melt diagnostic (psmlt) [kg/kg].
    pfevapr : jnp.ndarray
        Area-integrated rain evaporation [kg/m^2/s].
    pfrain : jnp.ndarray
        In-cloud rain flux (area-averaged) [kg/m^2/s].
    pfsnow : jnp.ndarray
        In-cloud snow flux (area-averaged) [kg/m^2/s].
    pfsubls : jnp.ndarray
        Area-integrated snow sublimation [kg/m^2/s].
    """
    # 1) Rain & Snow Production (autoconversion / accretion / aggregation)
    # timestep-dependent constant (zcons2 = dt * rgrav) and small guards
    _, _, _, zcons2, _ = microphysics_dt_constants(dt)

    # Precipitation produced in this level (mass flux units [kg/m2/s])
    zzdrr = zcons2 * pressure_thickness * rain_formation
    zzdrs = zcons2 * pressure_thickness * (snow_formation + snow_accretion)

    # If ice_flux_from_above is non-zero it must be included (caller should pass pxiflux at top level)
    # Top-level melting: convert part of snow -> rain if T > tmelt (uses plsdcp/plvdcp)
    # Note: Fortran gated with (kk .EQ. klev); here caller should incorporate ice_flux_from_above only when appropriate.
    # We perform the melting step unconditionally where ice_flux_from_above>0 and temp_tmp>tmelt to preserve behaviour.
    has_incoming_ice = ice_flux_from_above > 0.0
    zzdrs = zzdrs + jnp.where(has_incoming_ice, ice_flux_from_above, 0.0)

    # 2) Top-level Melting of Incoming Ice into Rain
    # melting capacity (per area) limited by available energy
    melt_capacity = zcons2 * pressure_thickness / jnp.maximum(lsdcp - lvdcp, eps) * jnp.maximum(0.0, (temp_tmp - tmelt))
    # limit melting to a fraction xsec*zzdrs (same heuristic as Fortran)
    ztmp2 = jnp.minimum(xsec * zzdrs, melt_capacity)
    # apply melting where incoming ice exists and melting capacity>0
    melt_applied = jnp.where(has_incoming_ice, ztmp2, 0.0)
    zzdrr = zzdrr + melt_applied
    zzdrs = zzdrs - melt_applied
    # psmlt accumulates melting mass in kg/kg units (Fortran: psmlt += ztmp2/(zcons2*pdp))
    snow_melt = snow_melt + melt_applied / jnp.maximum(zcons2 * pressure_thickness, eps)

    # 3) Update Precip-covered Fraction due to Falling Hydrometeors
    # Total precip from above (existing fluxes) and produced here (zpredel)
    zpretot = rain_flux + snow_flux
    zpredel = zzdrr + zzdrs

    # Update precip-covered fraction using helper
    # gridbox_frac_falling_hydrometeor signature:
    #   (precip_flux_from_above, precip_frac_from_above, precip_flux_from_level, precip_frac_from_level)
    precip_cover = gridbox_frac_falling_hydrometeor(
        precip_flux_from_above=zpretot,
        precip_frac_from_above=precip_cover,
        precip_flux_from_level=zpredel,
        precip_frac_from_level=cloud_fraction,
    )

    # 4) In-cloud Rain/Snow Fluxes and Area-integrated Evaporation/Sublimation
    # in-cloud (area-averaged) rain/snow fluxes before evaporation/sublimation
    ll1 = precip_cover > epsec

    ztmp1 = (rain_flux + zzdrr) / jnp.maximum(precip_cover, epsec)
    ztmp2 = (snow_flux + zzdrs) / jnp.maximum(precip_cover, epsec)

    pfrain = jnp.where(ll1, ztmp1, 0.0)
    pfsnow = jnp.where(ll1, ztmp2, 0.0)

    # evaporation / sublimation area-integrated (kg/m2/s)
    ztmp3 = (zcons2 * pressure_thickness * rain_evap_mmr) / jnp.maximum(precip_cover, epsec)
    ztmp4 = (zcons2 * pressure_thickness * snow_sublimation_mmr) / jnp.maximum(precip_cover, epsec)

    pfevapr = jnp.where(ll1, ztmp3, 0.0)
    pfsubls = jnp.where(ll1, ztmp4, 0.0)

    # 5) Update Column Rain / Snow Fluxes (add produced, remove evaporated/sublimated)
    # update column fluxes: add produced mass, remove evaporated/sublimated mass
    rain_flux = rain_flux + zzdrr - zcons2 * pressure_thickness * rain_evap_mmr
    snow_flux = snow_flux + zzdrs - zcons2 * pressure_thickness * snow_sublimation_mmr

    # 6) Diagnostics: return updated cover/fluxes and area-integrated/in-cloud diagnostics
    return (
        precip_cover,
        rain_flux,
        snow_flux,
        snow_melt,
        pfevapr,
        pfrain,
        pfsnow,
        pfsubls,
    )

def update_tendencies_and_important_vars(
    icnc: jnp.ndarray,                       # picnc
    cdnc: jnp.ndarray,                       # pcdnc
    ice_mmr_prev: jnp.ndarray,               # pxim1
    liq_mmr_prev: jnp.ndarray,               # pxlm1
    tracer_tm1_cdnc: jnp.ndarray,            # pxtm1_cdnc
    tracer_tm1_icnc: jnp.ndarray,            # pxtm1_icnc
    condensation_rate: jnp.ndarray,          # pcnd
    deposition_rate: jnp.ndarray,            # pdep
    rain_evap_mmr: jnp.ndarray,              # pevp
    freezing_rate: jnp.ndarray,              # pfrl
    tompkins_ice: jnp.ndarray,               # pgenti
    tompkins_liq: jnp.ndarray,               # pgentl
    incloud_ice_melt: jnp.ndarray,           # pimlt
    lsdcp: jnp.ndarray,                      # plsdcp
    lvdcp: jnp.ndarray,                      # plvdcp
    air_density: jnp.ndarray,                # prho
    inv_air_density: jnp.ndarray,            # prho_rcp
    rain_formation: jnp.ndarray,             # prpr
    snow_accretion: jnp.ndarray,             # psacl
    snow_formation: jnp.ndarray,             # pspr
    cloud_ice_evap: jnp.ndarray,             # pxievap
    ice_flux_melt: jnp.ndarray,              # pximlt
    pxitec: jnp.ndarray,                     # pxitec
    pxlevap: jnp.ndarray,                    # pxlevap
    pxltec: jnp.ndarray,                     # pxltec
    pxisub: jnp.ndarray,                     # pxisub
    snow_sublimation_mmr: jnp.ndarray,       # psub
    snow_melt: jnp.ndarray,                  # psmlt
    cloud_ice_in_cloud: jnp.ndarray,         # pxib
    cloud_liquid_in_cloud: jnp.ndarray,      # pxlb
    temp_tmp: jnp.ndarray,                   # ptp1tmp
    liquid_cloud_flag: jnp.ndarray,          # ld_liqcl (logical)
    ice_cloud_flag: jnp.ndarray,             # ld_icecl (logical)
    # INOUTs
    cloud_fraction: jnp.ndarray,             # paclc (INOUT)
    specific_humidity_tendency: jnp.ndarray, # pqte (INOUT)
    temp_tendency: jnp.ndarray,              # ptte (INOUT)
    ice_tendency: jnp.ndarray,               # pxite (INOUT)
    liq_tendency: jnp.ndarray,               # pxlte (INOUT)
    tracer_tendency_cdnc: jnp.ndarray,       # pxtte_cdnc (INOUT)
    tracer_tendency_icnc: jnp.ndarray,       # pxtte_icnc (INOUT)
    incloud_liq_before_rain: jnp.ndarray,    # pmlwc (INOUT)
    incloud_ice_before_snow: jnp.ndarray,    # pmiwc (INOUT)
    # time constant
    dt: jnp.ndarray,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Update tendencies and compute in-cloud effective radii.

    Overview
    --------
    - Accumulates temperature and humidity tendencies from microphysical sources.
    - Advances prognostic in-cloud liquid/ice mixing ratios and updates their tendencies.
    - Computes tracer tendencies for prognostic CDNC/ICNC and applies corrections to
      prevent negative in-cloud mass.
    - Computes effective in-cloud liquid and ice radii (µm).

    Steps
    -----
    1. Form timestep constants (ztmst, ztmst_rcp).
    2. Accumulate specific-humidity and temperature tendencies from condensation,
       deposition, evaporation/sublimation, melting, freezing and Tompkins sources.
    3. Advance in-cloud liquid and ice prognostics and update pxlte/pxite tendencies.
    4. Compute tracer tendencies for CDNC/ICNC from current-incloud values and
       previous tracer fields.
    5. Apply corrections when prognostic in-cloud mass falls below thresholds:
       - remove negative bias via zdxlcor/zdxicor,
       - adjust tracer tendencies accordingly,
       - possibly set cloud fraction to zero or clamp to clc_min.
    6. Compute effective liquid droplet radius (preffl) using breadth_factor and
       in-cloud liquid; compute ice effective radius (preffi) via eff_ice_crystal_radius
       with cirrus correction when nic_cirrus==1 and cold.
    7. Return updated INOUTs and effective radii.

    Parameters
    ----------
    icnc, cdnc :
        ICNC and CDNC (picnc, pcdnc).
    ice_mmr_prev, liq_mmr_prev :
        Previous in-cloud ice/liquid mmr (pxim1, pxlm1).
    tracer_tm1_cdnc, tracer_tm1_icnc :
        Tracer fields at t-1 for CDNC/ICNC (pxtm1_cdnc, pxtm1_icnc).
    condensation_rate, deposition_rate, rain_evap_mmr, freezing_rate :
        Process rates (pcnd, pdep, pevp, pfrl).
    tompkins_ice, tompkins_liq :
        Tompkins source terms (pgenti, pgentl).
    incloud_ice_melt, ice_flux_melt, snow_melt :
        Melting diagnostics (pimlt, pximlt, psmlt).
    lsdcp, lvdcp :
        Latent-heat constants (Ls/cpd, Lv/cpd).
    air_density, inv_air_density :
        prho, prho_rcp.
    rain_formation, snow_accretion, snow_formation :
        Rain/snow production (prpr, psacl, pspr).
    cloud_ice_evap, pxlevap, pxitec, pxltec, pxisub, snow_sublimation_mmr :
        Additional process terms used in tendencies.
    cloud_ice_in_cloud, cloud_liquid_in_cloud :
        In-cloud mixing ratios (pxib, pxlb).
    temp_tmp :
        Temporary layer temperature (ptp1tmp).
    liquid_cloud_flag, ice_cloud_flag :
        Logical masks for liquid/ice cloud presence.
    cloud_fraction, specific_humidity_tendency, temp_tendency, ice_tendency,
    liq_tendency, tracer_tendency_cdnc, tracer_tendency_icnc,
    incloud_liq_before_rain, incloud_ice_before_snow :
        INOUT arrays updated in-place.
    dt :
        Microphysics timestep ztmst [s].

    Returns
    -------
    Tuple (updated INOUTs + effective radii):
    - cloud_fraction
    - specific_humidity_tendency
    - temp_tendency
    - ice_tendency
    - liq_tendency
    - tracer_tendency_cdnc
    - tracer_tendency_icnc
    - incloud_liq_before_rain
    - incloud_ice_before_snow
    - out_liq_eff_radius_um (preffl) : liquid effective radius [µm]
    - out_ice_eff_radius_um (preffi) : ice effective radius [µm]

    Notes
    -----
    - Timestep constants are obtained via microphysics_dt_constants(dt).
    - Correction thresholds use module constants (ccwmin, clc_min, eps, etc.).
    - Breadth and ice-radius helpers (breadth_factor, eff_ice_crystal_radius)
      are used to compute effective radii. Cirrus branch applied when nic_cirrus==1.
    """
    # timestep constants
    ztmst, ztmst_rcp, _, _, _ = microphysics_dt_constants(dt)

    # --- 1) temperature & humidity tendencies accumulated from microphysical sources
    specific_humidity_tendency = specific_humidity_tendency + ztmst_rcp * (
        -condensation_rate - tompkins_liq + rain_evap_mmr + pxlevap
        - deposition_rate - tompkins_ice + snow_sublimation_mmr + cloud_ice_evap
        + pxisub
    )

    temp_tendency = temp_tendency + ztmst_rcp * (
        lvdcp * (condensation_rate + tompkins_liq - rain_evap_mmr - pxlevap)
        + lsdcp * (deposition_rate + tompkins_ice - snow_sublimation_mmr - cloud_ice_evap - pxisub)
        + (lsdcp - lvdcp) * (-snow_melt - incloud_ice_melt - ice_flux_melt + freezing_rate + snow_accretion)
    )

    # --- 2) liquid prognostic advance and tendencies
    ztmp1 = pxltec + liq_tendency
    ztmp2 = incloud_ice_melt + ice_flux_melt - freezing_rate - rain_formation - snow_accretion + condensation_rate + tompkins_liq - pxlevap
    liq_mmr_next = liq_mmr_prev + ztmst * ztmp1 + ztmp2
    liq_tendency = ztmp1 + ztmst_rcp * ztmp2

    # --- 3) ice prognostic advance and tendencies
    ztmp1 = pxitec + ice_tendency
    ztmp2 = freezing_rate - snow_formation + deposition_rate + tompkins_ice - cloud_ice_evap
    ice_mmr_next = ice_mmr_prev + ztmst * ztmp1 + ztmp2
    ice_tendency = ztmp1 + ztmst_rcp * ztmp2

    # --- 4) tracer tendencies for prognostic CDNC/ICNC (mapped exactly)
    tracer_tendency_cdnc = ztmst_rcp * (cdnc * inv_air_density - tracer_tm1_cdnc)
    tracer_tendency_icnc = ztmst_rcp * (icnc * inv_air_density - tracer_tm1_icnc)

    # --- 5) Corrections to avoid negative in-cloud mass (merge logic)
    # liquid
    ll_liq_neg = liq_mmr_next < ccwmin
    zdxlcor = jnp.where(ll_liq_neg, -ztmst_rcp * liq_mmr_next, 0.0)
    liq_tendency = liq_tendency + zdxlcor

    # adjust tracer tendency for cdnc where negative-correction applied
    tracer_tendency_cdnc = jnp.where(
        ll_liq_neg,
        tracer_tendency_cdnc - ztmst_rcp * cdnc * inv_air_density,
        tracer_tendency_cdnc,
    )

    # ice
    ll_ice_neg = ice_mmr_next < ccwmin
    zdxicor = jnp.where(ll_ice_neg, -ztmst_rcp * ice_mmr_next, 0.0)
    ice_tendency = ice_tendency + zdxicor

    tracer_tendency_icnc = jnp.where(
        ll_ice_neg,
        tracer_tendency_icnc - ztmst_rcp * icnc * inv_air_density,
        tracer_tendency_icnc,
    )

    # where both liquid and ice are tiny, set cloud_fraction to 0
    cloud_fraction = jnp.where(jnp.logical_and(ll_liq_neg, ll_ice_neg), 0.0, cloud_fraction)

    # clamp small cloud fraction values to zero (Fortran MERGE with clc_min)
    ll_small_clc = cloud_fraction < clc_min
    cloud_fraction = jnp.where(ll_small_clc, 0.0, cloud_fraction)

    # zero tiny in-cloud accumulators (Fortran used 1e-20 checks)
    pmlwc_flag = jnp.logical_or(ll_small_clc, incloud_liq_before_rain < 1e-20)
    incloud_liq_before_rain = jnp.where(pmlwc_flag, 0.0, incloud_liq_before_rain)

    pmiwc_flag = jnp.logical_or(ll_small_clc, incloud_ice_before_snow < 1e-20)
    incloud_ice_before_snow = jnp.where(pmiwc_flag, 0.0, incloud_ice_before_snow)

    # adjust tendencies by removing the correction contributions
    specific_humidity_tendency = specific_humidity_tendency - zdxlcor - zdxicor
    temp_tendency = temp_tendency + lvdcp * zdxlcor + lsdcp * zdxicor

    # --- 6) effective liquid droplet radius [um] (preffl)
    # breadth_factor returns dimensionless breadth parameter (Fortran breadth_factor)
    breadth = breadth_factor(cdnc)
    # convert to effective radius (um): 1e6 * breadth * ((3/(4*pi*rhoh2o)) * pxlb * prho / pcdnc)^(1/3)
    liq_eff_radius = 1.0e6 * breadth * ((3.0 / (4.0 * pi * rhoh2o)) * cloud_liquid_in_cloud * air_density / jnp.maximum(cdnc, eps)) ** (1.0 / 3.0)
    liq_eff_radius = jnp.where(liquid_cloud_flag, liq_eff_radius, 0.0)

    # --- 7) ice crystal effective radius [um] (preffi)
    # convert in-cloud ice kg/kg -> g/m^3: 1000 * pxib * prho
    ice_gm3 = 1000.0 * cloud_ice_in_cloud * air_density
    ice_eff_rad = eff_ice_crystal_radius(ice_gm3, icnc)  # returns microns (as in module helpers)

    # cirrus correction branch as in Fortran when nic_cirrus==1 and cold
    if int(nic_cirrus) == 1:
        is_cold = temp_tmp < cthomi
        ztmp2 = 83.8 * (1e3 * jnp.maximum(cloud_ice_in_cloud, eps) * air_density) ** 0.216
        ice_eff_rad = jnp.where(is_cold, ztmp2, ice_eff_rad)

    # clip bounds
    ice_eff_rad = jnp.maximum(ice_eff_rad, ceffmin)
    ice_eff_rad = jnp.minimum(ice_eff_rad, ceffmax)
    ice_eff_rad = jnp.where(ice_cloud_flag, ice_eff_rad, 0.0)

    # --- finalize returns (match Fortran order)
    out_preffl = liq_eff_radius
    out_preffi = ice_eff_rad

    return (
        cloud_fraction,
        specific_humidity_tendency,
        temp_tendency,
        ice_tendency,
        liq_tendency,
        tracer_tendency_cdnc,
        tracer_tendency_icnc,
        incloud_liq_before_rain,
        incloud_ice_before_snow,
        out_preffl,
        out_preffi,
    )

def lookup_1d_interp(
    table: jnp.ndarray,
    pt: jnp.ndarray,
    scale: float,
    i_min: int,
    i_max: int,
    overflow_penalty_scale: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable 1D lookup via linear interpolation.
    Returns (value, overflow_penalty).
    - table indexed with integer indices [i_min..i_max]
    - continuous index = pt * scale

    Notes:
    Intended as differentiable approximation to replace set_lookup_index_1d from ECHAM.
    Compute a continuous index, clip it to table bounds (no hard error), and return a linearly-interpolated 
    value (or soft-weighted sum) that keeps gradients w.r.t. pt.
    """
    idx_raw = pt * scale
    # detect overflow BEFORE clipping (differentiable boolean -> float penalty)
    overflow_low = jnp.maximum(0.0, (i_min - idx_raw))
    overflow_high = jnp.maximum(0.0, (idx_raw - i_max))
    overflow_penalty = overflow_penalty_scale * (overflow_low**2 + overflow_high**2)

    # clip into safe continuous range that allows interpolation between i and i+1
    idx = jnp.clip(idx_raw, i_min, jnp.maximum(i_min, i_max - 1e-6))

    i0 = jnp.floor(idx).astype(int)
    w = idx - jnp.floor(idx)
    v0 = jnp.take(table, i0, axis=0)
    v1 = jnp.take(table, jnp.clip(i0 + 1, 0, table.shape[0] - 1), axis=0)
    value = (1.0 - w) * v0 + w * v1

    return value, overflow_penalty


def lookup_2d_interp(
    table: jnp.ndarray,
    pt1: jnp.ndarray,
    pt2: jnp.ndarray,
    scale1: float,
    scale2: float,
    i1_min: int,
    i1_max: int,
    i2_min: int,
    i2_max: int,
    overflow_penalty_scale: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable 2D lookup via bilinear interpolation.
    table shape: [N1, N2]; pt1, pt2 shapes broadcastable to each other.
    Returns (value, overflow_penalty).

    Notes:
    Intended as differentiable approximation to replace set_lookup_index_2d from ECHAM.
    Compute a continuous index, clip it to table bounds (no hard error), and return a linearly-interpolated 
    value (or soft-weighted sum) that keeps gradients w.r.t. pt. 
    """
    idx1_raw = pt1 * scale1
    idx2_raw = pt2 * scale2

    # overflow penalties
    ol1 = jnp.maximum(0.0, (i1_min - idx1_raw))
    oh1 = jnp.maximum(0.0, (idx1_raw - i1_max))
    ol2 = jnp.maximum(0.0, (i2_min - idx2_raw))
    oh2 = jnp.maximum(0.0, (idx2_raw - i2_max))
    overflow_penalty = overflow_penalty_scale * (ol1**2 + oh1**2 + ol2**2 + oh2**2)

    # clip to safe continuous ranges
    idx1 = jnp.clip(idx1_raw, i1_min, jnp.maximum(i1_min, i1_max - 1e-6))
    idx2 = jnp.clip(idx2_raw, i2_min, jnp.maximum(i2_min, i2_max - 1e-6))

    i1 = jnp.floor(idx1).astype(int)
    i2 = jnp.floor(idx2).astype(int)
    w1 = idx1 - jnp.floor(idx1)
    w2 = idx2 - jnp.floor(idx2)

    i1p = jnp.clip(i1 + 1, 0, table.shape[0] - 1)
    i2p = jnp.clip(i2 + 1, 0, table.shape[1] - 1)

    # gather four corners
    v00 = table[i1, i2]
    v10 = table[i1p, i2]
    v01 = table[i1, i2p]
    v11 = table[i1p, i2p]

    # bilinear interpolation
    value = (1 - w1) * (1 - w2) * v00 + w1 * (1 - w2) * v10 + (1 - w1) * w2 * v01 + w1 * w2 * v11

    return value, overflow_penalty

def sat_spec_hum(
    pressure: jnp.ndarray,         # Original: pap
    es_rd_over_rv: jnp.ndarray,    # Original: ptlucu  (lookup value e_s * Rd/Rv)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Saturation vapour pressure / correction factor / saturation specific humidity.

    Overview
    --------
    Compute:
      - pes : non-dimensional ratio (e_s * Rd/Rv) / p  (Fortran `pes`),
              clipped to 0.5 to avoid pathological values from lookup tables;
      - pcor: thermodynamic correction factor (Fortran `pcor`) = 1 / (1 - vtmpc1 * pes);
      - saturation_specific_humidity  : saturation specific humidity (Fortran `pq`) = pes * pcor.

    Steps
    -----
    1. pes = es_rd_over_rv / pressure
    2. Clip pes to a maximum of 0.5 (Fortran: MIN(pes, 0.5_dp))
    3. pcor = 1.0 / (1.0 - vtmpc1 * pes) with a small safety floor on the denominator
    4. pq = pes * pcor

    Parameters
    ----------
    pressure : jnp.ndarray
        Full-level pressure (Fortran: pap) [Pa]. Can be 1D or 2D (or higher) as long as
        shapes match es_rd_over_rv.
    es_rd_over_rv : jnp.ndarray
        Lookup-table value (Fortran: ptlucu) equal to e_s * Rd/Rv at the lookup temperature.
        Same shape as pressure.

    Returns
    -------
    pes : jnp.ndarray
        Scaled saturation vapour pressure (ECHAM: pes) (dimensionless, clipped <= 0.5).
    pcor : jnp.ndarray
        Correction factor (ECHAM: pcor) = 1 / (1 - vtmpc1 * pes).
    saturation_specific_humidity : jnp.ndarray
        Saturation specific humidity (ECHAM: pq) [kg/kg].

    Notes
    -----
    - This single function replaces the 1D/2D ECHAM variants by operating elementwise
      on arrays of arbitrary compatible shape.
    - A safety floor (eps) is used when forming the denominator to avoid division-by-zero.
    - ECHAM names: pap -> pressure, ptlucu -> es_rd_over_rv, pes/pcor/pq preserved.
    """
    # pes = ptlucu / pap
    pes = es_rd_over_rv / pressure
    pes = jnp.minimum(pes, 0.5)

    # pcor = 1 / (1 - vtmpc1 * pes)  (protect denominator)
    from .cloud_params_2m import eps as _eps  # local small number from params
    denom = jnp.maximum(1.0 - vtmpc1 * pes, _eps)
    pcor = 1.0 / denom

    # pq = pes * pcor
    saturation_specific_humidity = pes * pcor

    return pes, pcor, saturation_specific_humidity

def update_in_cloud_water(
    pressure: jnp.ndarray,               # Original: pap
    activated_cdnc: jnp.ndarray,         # Original: pcdncact
    condensation_rate: jnp.ndarray,      # Original: pcnd
    deposition_rate: jnp.ndarray,        # Original: pdep
    tompkins_genti: jnp.ndarray,         # Original: pgenti
    tompkins_gentl: jnp.ndarray,         # Original: pgentl
    newly_formed_ice: jnp.ndarray,       # Original: pnicex
    specific_humidity_tmp: jnp.ndarray,  # Original: pqp1tmp
    sat_spec_humidity_tmp: jnp.ndarray,  # Original: pqsp1tmp
    air_density: jnp.ndarray,            # Original: prho
    ice_radius_mean: jnp.ndarray,        # Original: prid
    temp_prev: jnp.ndarray,              # Original: ptm1
    cloud_flag: jnp.ndarray,             # Original: ld_cc (INOUT)
    ice_crystal_number: jnp.ndarray,     # Original: picnc (INOUT)
    nucleation_rate: jnp.ndarray,        # Original: pqnuc (INOUT)
    droplet_number: jnp.ndarray,         # Original: pcdnc (INOUT)
    cloud_fraction: jnp.ndarray,         # Original: paclc (INOUT)
    cloud_ice_in_cloud: jnp.ndarray,     # Original: pxib (INOUT)
    cloud_liquid_in_cloud: jnp.ndarray,  # Original: pxlb (INOUT)
    dt: jnp.ndarray                       # Microphysics timestep (used for pqnuc accumulation)
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """
    Update in-cloud water/ice, CDNC/ICNC activation/nucleation and cloud cover.

    Overview
    --------
    Updates in-cloud mixing ratios (liquid/ice), cloud fraction, CDNC/ICNC and
    accumulates nucleation diagnostics following ICON/ECHAM logic.

    Steps
    -----
    1. Compute relative humidity and positive condensation/deposition sources.
    2. Update in-cloud pxib/pxlb using deposition/condensation scaled by cloud fraction
       (lower-limited by clc_min) where cloud already exists.
    3. If no cloud but there are positive condensation/deposition sources, set cloud
       fraction from relative humidity (clipped to [0.01,1.0]) and set in-cloud values
       from source-per-cloud-area.
    4. Compute minimum CDNC from in-cloud liquid mass density via minimum_CDNC().
    5. Update cloud flag (ld_cc) from cloud_fraction>0.
    6. Where cloud formed and liquid present, allow activation: increase CDNC up to
       activated_cdnc and accumulate nucleation rate (pqnuc += dt * delta_cdnc).
    7. Enforce CDNC >= computed minimum (pcdnc_min) or = cqtmin where no cloud present.
    8. Update ICNC where cloud ice present and below icemin:
       - nic_cirrus==1: prognostic conversion from ice mass -> number using rhoice and prid
       - nic_cirrus==2: use pnicex (capped by pressure*1e6)
       Then enforce picnc >= icemin (or cqtmin where no cloud).

    Parameters
    ----------
    (see argument list above; original Fortran names in parentheses)

    Returns
    -------
    Updated (in same-ish order):
      - cloud_flag (ld_cc)
      - ice_crystal_number (picnc)
      - nucleation_rate (pqnuc)
      - droplet_number (pcdnc)
      - cloud_fraction (paclc)
      - cloud_ice_in_cloud (pxib)
      - cloud_liquid_in_cloud (pxlb)
      - pcdnc_min : minimum CDNC computed from max radius [1/m^3]

    Notes
    -----
    - Uses jnp.where (Fortran MERGE) to preserve values where masks are False.
    - Uses helper minimum_CDNC(...) to compute pcdnc_min from in-cloud liquid mass density.
    - The logic mirrors the Fortran ordering and masks; numerical safeguards (clipping,
      max denominators) follow the Fortran intent.
    """
    # safety eps already imported as eps; other constants available (clc_min, cqtmin, icemin, nic_cirrus, rhoice)
    # 1) relative humidity
    relhum = specific_humidity_tmp / jnp.maximum(sat_spec_humidity_tmp, eps)

    # positive deposition / condensation sources (limit negative contributions)
    src_dep = jnp.maximum(deposition_rate + tompkins_genti, 0.0)
    src_cnd = jnp.maximum(condensation_rate + tompkins_gentl, 0.0)

    # 2) update in-cloud ice/liquid where cloud already exists:
    # pxib_new = pxib + pdep / max(paclc, clc_min)
    pxib_candidate = cloud_ice_in_cloud + src_dep / jnp.maximum(cloud_fraction, clc_min)
    pxib_candidate = jnp.maximum(pxib_candidate, 0.0)
    cloud_ice_in_cloud = jnp.where(cloud_flag, pxib_candidate, cloud_ice_in_cloud)

    pxlb_candidate = cloud_liquid_in_cloud + src_cnd / jnp.maximum(cloud_fraction, clc_min)
    pxlb_candidate = jnp.maximum(pxlb_candidate, 0.0)
    cloud_liquid_in_cloud = jnp.where(cloud_flag, pxlb_candidate, cloud_liquid_in_cloud)

    # 3) if no cloud but there are positive sources, set cloud fraction from relhum and
    #    set in-cloud values to source-per-cloud-area
    make_cloud_mask = jnp.logical_and(~cloud_flag, jnp.logical_or(src_dep > 0.0, src_cnd > 0.0))

    paclc_from_rh = jnp.clip(relhum, 0.01, 1.0)
    cloud_fraction = jnp.where(make_cloud_mask, paclc_from_rh, cloud_fraction)

    pxib_from_src = src_dep / jnp.maximum(cloud_fraction, clc_min)
    pxib_from_src = jnp.maximum(pxib_from_src, 0.0)
    cloud_ice_in_cloud = jnp.where(make_cloud_mask, pxib_from_src, cloud_ice_in_cloud)

    pxlb_from_src = src_cnd / jnp.maximum(cloud_fraction, clc_min)
    pxlb_from_src = jnp.maximum(pxlb_from_src, 0.0)
    cloud_liquid_in_cloud = jnp.where(make_cloud_mask, pxlb_from_src, cloud_liquid_in_cloud)

    # 4) compute minimum CDNC from in-cloud liquid mass density (kg/kg * rho -> kg/m^3)
    liquid_mass_density = cloud_liquid_in_cloud * air_density  # [kg/m^3]
    pcdnc_min = minimum_CDNC(liquid_mass_density)

    # 5) redefine cloud flag
    cloud_flag = cloud_fraction > 0.0

    # 6) activation / nucleation: only where cloud exists and liquid > cqtmin
    ll1 = jnp.logical_and(cloud_flag, cloud_liquid_in_cloud > cqtmin)
    ll2 = jnp.logical_and(ll1, jnp.logical_and(droplet_number <= pcdnc_min, temp_prev > cthomi))

    # desired additional droplets
    delta_cdnc = jnp.maximum(activated_cdnc - droplet_number, 0.0)

    # only count activation where ll2
    delta_cdnc_applied = jnp.where(ll2, delta_cdnc, 0.0)

    # update droplet number and nucleation-rate diagnostic (pqnuc += dt * delta)
    droplet_number = droplet_number + delta_cdnc_applied
    nucleation_rate = nucleation_rate + dt * delta_cdnc_applied

    # 7) enforce minimum CDNC or set to cqtmin where no meaningful cloud (Fortran MERGE semantics)
    # ztmp1 = max(pcdnc, pcdnc_min)
    tmp_cdnc_max = jnp.maximum(droplet_number, pcdnc_min)
    # Fortran: pcdnc = MERGE( ztmp1, cqtmin, ll1 ) -> if ll1 True -> tmp_cdnc_max else -> cqtmin
    droplet_number = jnp.where(ll1, tmp_cdnc_max, cqtmin)

    # 8) update ICNC similarly
    ll1_ic = jnp.logical_and(cloud_flag, cloud_ice_in_cloud > cqtmin)
    ll2_ic = jnp.logical_and(ll1_ic, ice_crystal_number <= icemin)

    # compute candidate ICNC depending on nic_cirrus
    if int(nic_cirrus) == 1:
        # 0.75 / (pi * rhoice) * prho * pxib / prid^3  (note units)
        icnc_candidate = 0.75 / (pi * rhoice) * air_density * cloud_ice_in_cloud / jnp.maximum(ice_radius_mean**3, eps)
    elif int(nic_cirrus) == 2:
        # min(pnicex, pap*1e6)
        icnc_candidate = jnp.minimum(newly_formed_ice, pressure * 1.0e6)
    else:
        # default: leave unchanged candidate (set to existing to be MERGE-safe)
        icnc_candidate = ice_crystal_number

    ice_crystal_number = jnp.where(ll2_ic, icnc_candidate, ice_crystal_number)

    # enforce minimum icnc or set to cqtmin where no cloud-ice
    tmp_icnc_max = jnp.maximum(ice_crystal_number, icemin)
    ice_crystal_number = jnp.where(ll1_ic, tmp_icnc_max, cqtmin)

    return (
        cloud_flag,
        ice_crystal_number,
        nucleation_rate,
        droplet_number,
        cloud_fraction,
        cloud_ice_in_cloud,
        cloud_liquid_in_cloud,
        pcdnc_min,
    )

def diagnostics(
    cdnc: jnp.ndarray,                    # pcdnc
    icnc: jnp.ndarray,                    # picnc
    cloud_fraction: jnp.ndarray,          # paclc
    dp_over_g: jnp.ndarray,               # pdpg
    layer_thickness: jnp.ndarray,         # pdz
    freezing_number_rate: jnp.ndarray,    # pfrln
    air_density: jnp.ndarray,             # prho
    rain_number_formation: jnp.ndarray,   # prprn
    snow_number_accretion: jnp.ndarray,   # psacln
    incloud_ice: jnp.ndarray,             # pxib
    incloud_liquid: jnp.ndarray,          # pxlb
    temp_tmp: jnp.ndarray,                # ptp1tmp
    eff_radius_liq: jnp.ndarray,          # preffl (µm)
    eff_radius_ice: jnp.ndarray,          # preffi (µm)
    liquid_cloud_flag: jnp.ndarray,       # ld_liqcl (logical)
    ice_cloud_flag: jnp.ndarray,          # ld_icecl (logical)
    # INOUT accumulators (order preserved from Fortran)
    cdnc_ave: jnp.ndarray,                # pcdnc_ave
    cdnc_ave_acc: jnp.ndarray,            # pcdnc_ave_acc
    cdnc_ave_burd: jnp.ndarray,           # pcdnc_ave_burd
    cdnc_ct: jnp.ndarray,                 # pcdnc_ct
    cld_ice_time: jnp.ndarray,            # pcliwc_time
    cld_liq_time: jnp.ndarray,            # pcloud_time
    icnc_ave: jnp.ndarray,                # picnc_ave
    icnc_ave_acc: jnp.ndarray,            # picnc_ave_acc
    icnc_ave_burd: jnp.ndarray,           # picnc_ave_burd
    ice_water_content_acc: jnp.ndarray,   # piwc_acc
    iwp_tovs: jnp.ndarray,                # piwp_tovs
    liq_water_content_acc: jnp.ndarray,   # plwc_acc
    cdnc_accretion: jnp.ndarray,          # pqacc
    cdnc_autoconv: jnp.ndarray,           # pqaut
    cdnc_freezing: jnp.ndarray,           # pqfre
    eff_radius_ice_acc: jnp.ndarray,      # preffi_acc
    eff_radius_ice_time: jnp.ndarray,     # preffi_time
    eff_radius_ice_tovs: jnp.ndarray,     # preffi_tovs
    eff_radius_liq_acc: jnp.ndarray,      # preffl_acc
    eff_radius_liq_ct: jnp.ndarray,       # preffl_ct
    eff_radius_liq_time: jnp.ndarray,     # preffl_time
    cdnc_burden: jnp.ndarray,             # pcdnc_burden
    icnc_burden: jnp.ndarray,             # picnc_burden
    tau1i: jnp.ndarray,                   # ptau1i
    eff_radius_ct_m: jnp.ndarray,         # preffct (m)
    cloud_fraction_acc: jnp.ndarray,      # paclcac
    ktop: jnp.ndarray,                    # ktop (integer flags per column top)
    level_index: int,                     # kk (current level index)
    dt: jnp.ndarray,                      # microphysics timestep (s) -> used as zdt / zdtime
) -> tuple:
    """
    Diagnostics accumulator updates.

    Overview
    --------
    - Update time-accumulated diagnostics and burdens for liquid/ice clouds,
      CDNC/ICNC, effective radii, TOVS-style IWP diagnostics and related accumulators.

    Steps
    -----
    1. Subtract instantaneous number-process contributions (autoconversion, freezing, accretion).
    2. Update liquid-cloud accumulators (CDNC averages, liquid water content, times, burdens).
    3. Update cloud-top liquid diagnostics where applicable.
    4. Update ice-cloud accumulators (ICNC averages, ice water content, times, burdens).
    5. Compute TOVS-style cirrus diagnostics and select sampling candidates.
    6. Accumulate icnc/liquid burdens and total cloud-fraction accumulation.
    7. Return updated INOUT accumulators in the original ECHAM order.

    Parameters
    ----------
    cdnc, icnc : jnp.ndarray
        Cloud droplet and ice-crystal number concentrations (pcdnc, picnc).
    cloud_fraction : jnp.ndarray
        Cloud cover fraction (paclc).
    dp_over_g : jnp.ndarray
        dp/g (pdpg).
    layer_thickness : jnp.ndarray
        Layer thickness (pdz).
    freezing_number_rate : jnp.ndarray
        Number of freezing events per timestep (pfrln).
    air_density : jnp.ndarray
        Air density (prho).
    rain_number_formation : jnp.ndarray
        Rain number formation rate (prprn).
    snow_number_accretion : jnp.ndarray
        Snow number accretion (psacln).
    incloud_ice, incloud_liquid : jnp.ndarray
        In-cloud ice/liquid mixing ratios (pxib, pxlb).
    temp_tmp : jnp.ndarray
        Layer temperature used in diagnostics (ptp1tmp).
    eff_radius_liq, eff_radius_ice : jnp.ndarray
        Effective radii (preffl, preffi) in µm.
    liquid_cloud_flag, ice_cloud_flag : jnp.ndarray
        Logical masks for liquid/ice cloud presence (ld_liqcl, ld_icecl).
    INOUT accumulators : jnp.ndarray
        Various accumulator arrays (order preserved from Fortran):
        cdnc_ave, cdnc_ave_acc, cdnc_ave_burd, cdnc_ct, cld_ice_time, cld_liq_time,
        icnc_ave, icnc_ave_acc, icnc_ave_burd, ice_water_content_acc, iwp_tovs,
        liq_water_content_acc, cdnc_accretion, cdnc_autoconv, cdnc_freezing,
        eff_radius_ice_acc, eff_radius_ice_time, eff_radius_ice_tovs,
        eff_radius_liq_acc, eff_radius_liq_ct, eff_radius_liq_time,
        cdnc_burden, icnc_burden, tau1i, eff_radius_ct_m, cloud_fraction_acc.
    ktop : jnp.ndarray
        Column-top level flags.
    level_index : int
        Current level index (kk).
    dt : jnp.ndarray
        Microphysics timestep (zdt / zdtime).

    Returns
    -------
    Tuple of updated INOUT accumulators in the same order as provided:
    (cdnc_ave, cdnc_ave_acc, cdnc_ave_burd, cdnc_ct, cld_ice_time, cld_liq_time,
     icnc_ave, icnc_ave_acc, icnc_ave_burd, ice_water_content_acc, iwp_tovs,
     liq_water_content_acc, cdnc_accretion, cdnc_autoconv, cdnc_freezing,
     eff_radius_ice_acc, eff_radius_ice_time, eff_radius_ice_tovs,
     eff_radius_liq_acc, eff_radius_liq_ct, eff_radius_liq_time,
     cdnc_burden, icnc_burden, tau1i, eff_radius_ct_m, cloud_fraction_acc)

    Notes
    -----
    - Time-step scalars zdt and zdtime are taken equal to dt.
    """

    # time-step scalars used in Fortran as zdt / zdtime
    zdt = dt
    zdtime = dt

    # 1) subtract instantaneous number-process contributions over the timestep
    cdnc_autoconv = cdnc_autoconv - zdt * rain_number_formation
    cdnc_freezing = cdnc_freezing - zdt * freezing_number_rate
    cdnc_accretion = cdnc_accretion - zdt * snow_number_accretion

    # 2) liquid-cloud diagnostics (update only where liquid cloud flag True)
    tmp = cdnc_ave_acc + zdtime * cdnc
    cdnc_ave_acc = jnp.where(liquid_cloud_flag, tmp, cdnc_ave_acc)

    tmp = liq_water_content_acc + zdtime * incloud_liquid * air_density
    liq_water_content_acc = jnp.where(liquid_cloud_flag, tmp, liq_water_content_acc)

    tmp = cld_liq_time + zdtime
    cld_liq_time = jnp.where(liquid_cloud_flag, tmp, cld_liq_time)

    tmp = cdnc_burden + cdnc * layer_thickness
    cdnc_burden = jnp.where(liquid_cloud_flag, tmp, cdnc_burden)

    tmp = cdnc_ave + zdtime * cdnc * cloud_fraction
    cdnc_ave = jnp.where(liquid_cloud_flag, tmp, cdnc_ave)

    tmp = cdnc_ave_burd + zdtime * cdnc * layer_thickness * cloud_fraction
    cdnc_ave_burd = jnp.where(liquid_cloud_flag, tmp, cdnc_ave_burd)

    # accumulated in-cloud liquid effective radius (unconditional add)
    eff_radius_liq_acc = eff_radius_liq_acc + zdtime * eff_radius_liq

    # 3) cloud-top liquid diagnostics (complex mask ll1)
    # ll1 = (
    #     jnp.logical_and.reduce(
    #         (
    #             liquid_cloud_flag,
    #             ktop == level_index,
    #             temp_tmp > tmelt,
    #             eff_radius_ct_m < 4.0,
    #             eff_radius_liq >= 4.0,
    #         )
    #     )
    # )
    ll1 = jnp.logical_and.reduce(
    jnp.stack(
        (
            liquid_cloud_flag,
            (ktop == level_index),
            (temp_tmp > tmelt),
            (eff_radius_ct_m < 4.0),
            (eff_radius_liq >= 4.0),
        ),
        axis=0,
    ),
    axis=0,
)

    tmp = eff_radius_liq_ct + zdtime * eff_radius_liq
    eff_radius_liq_ct = jnp.where(ll1, tmp, eff_radius_liq_ct)

    tmp = cdnc_ct + zdtime * cdnc * cloud_fraction
    cdnc_ct = jnp.where(ll1, tmp, cdnc_ct)

    tmp = eff_radius_liq_time + zdtime
    eff_radius_liq_time = jnp.where(ll1, tmp, eff_radius_liq_time)

    eff_radius_ct_m = jnp.where(ll1, eff_radius_liq, eff_radius_ct_m)

    # 4) ice-cloud diagnostics (update only where ice cloud flag True)
    tmp = icnc_ave_acc + zdtime * icnc
    icnc_ave_acc = jnp.where(ice_cloud_flag, tmp, icnc_ave_acc)

    tmp = ice_water_content_acc + zdtime * incloud_ice * air_density
    ice_water_content_acc = jnp.where(ice_cloud_flag, tmp, ice_water_content_acc)

    eff_radius_ice_acc = eff_radius_ice_acc + zdtime * eff_radius_ice

    tmp = cld_ice_time + zdtime
    cld_ice_time = jnp.where(ice_cloud_flag, tmp, cld_ice_time)

    # 5) TOVS-style semi-transparent cirrus diagnostics
    ll2 = jnp.logical_and(ice_cloud_flag, jnp.logical_not(ll1))

    ztmp3 = 1000.0 * incloud_ice * cloud_fraction * dp_over_g  # IWP [g/m2]
    ztmp4 = tau1i + 1.9787 * ztmp3 * jnp.maximum(eff_radius_ice, ceffmin) ** (-1.0365)
    tau1i = jnp.where(ll2, ztmp4, tau1i)

    # 6) selection for TOVS sampling
    ll3 = jnp.logical_and(ll2, jnp.logical_and(tau1i > 0.7, tau1i < 3.8))

    tmp = eff_radius_ice_tovs + zdtime * eff_radius_ice
    eff_radius_ice_tovs = jnp.where(ll3, tmp, eff_radius_ice_tovs)

    tmp = eff_radius_ice_time + zdtime
    eff_radius_ice_time = jnp.where(ll3, tmp, eff_radius_ice_time)

    tmp = iwp_tovs + zdtime * ztmp3
    iwp_tovs = jnp.where(ll3, tmp, iwp_tovs)

    # 7) icnc burden / averages (ice)
    tmp = icnc_burden + icnc * layer_thickness
    icnc_burden = jnp.where(ice_cloud_flag, tmp, icnc_burden)

    tmp = icnc_ave + zdtime * icnc * cloud_fraction
    icnc_ave = jnp.where(ice_cloud_flag, tmp, icnc_ave)

    tmp = icnc_ave_burd + zdtime * icnc * layer_thickness * cloud_fraction
    icnc_ave_burd = jnp.where(ice_cloud_flag, tmp, icnc_ave_burd)

    # 8) accumulate cloud fraction
    cloud_fraction_acc = cloud_fraction_acc + zdtime * cloud_fraction

    # return updated INOUTs in the same order as arguments were provided
    return (
        cdnc_ave,
        cdnc_ave_acc,
        cdnc_ave_burd,
        cdnc_ct,
        cld_ice_time,
        cld_liq_time,
        icnc_ave,
        icnc_ave_acc,
        icnc_ave_burd,
        ice_water_content_acc,
        iwp_tovs,
        liq_water_content_acc,
        cdnc_accretion,
        cdnc_autoconv,
        cdnc_freezing,
        eff_radius_ice_acc,
        eff_radius_ice_time,
        eff_radius_ice_tovs,
        eff_radius_liq_acc,
        eff_radius_liq_ct,
        eff_radius_liq_time,
        cdnc_burden,
        icnc_burden,
        tau1i,
        eff_radius_ct_m,
        cloud_fraction_acc,
    )

def cloud_microphysics_2m(
    temperature: jnp.ndarray,           # Temperature [K] (nlev, ncols)
    specific_humidity: jnp.ndarray,     # Specific humidity [kg/kg] (nlev, ncols)
    pressure: jnp.ndarray,              # Pressure at full levels [Pa] (nlev, ncols)
    cloud_water: jnp.ndarray,           # Cloud water mixing ratio [kg/kg] (nlev, ncols)
    cloud_ice: jnp.ndarray,             # Cloud ice mixing ratio [kg/kg] (nlev, ncols)
    cloud_fraction: jnp.ndarray,        # Cloud fraction [1] (nlev, ncols)
    air_density: jnp.ndarray,           # Air density [kg/m³] (nlev, ncols)
    layer_thickness: jnp.ndarray,       # Layer thickness [m] (nlev, ncols)
    droplet_number: jnp.ndarray,        # Droplet number concentration [1/m³] (nlev, ncols)
    tke: jnp.ndarray,                   # Turbulent kinetic energy [m²/s²] (nlev, ncols)
    vertical_velocity: jnp.ndarray,     # Vertical velocity [m/s] (nlev, ncols)
    params: CloudParams2M,              # Cloud parameters structure
    dt: jnp.ndarray,                    # Microphysics timestep [s]
    ktdia: int = 1,                     # Top diagnostic level
) -> tuple[MicrophysicsTendencies_2M, MicrophysicsState_2M]:
    """
    Main entry point for the two-moment cloud microphysics scheme.

    Based on ECHAM6's cloud_micro_interface subroutine. Orchestrates the sequence of
    microphysical processes and returns tendencies and state diagnostics.

    Args:
        temperature: Temperature at full levels [K]
        specific_humidity: Specific humidity at full levels [kg/kg]
        pressure: Pressure at full levels [Pa]
        cloud_water: Cloud water mixing ratio [kg/kg]
        cloud_ice: Cloud ice mixing ratio [kg/kg]
        cloud_fraction: Cloud fraction [1]
        air_density: Air density [kg/m³] (nlev, ncols)
        layer_thickness: Layer thickness [m] (nlev, ncols)
        droplet_number: Droplet number concentration [1/m³] (nlev, ncols)
        tke: Turbulent kinetic energy [m²/s²] (nlev, ncols). Used to compute
            updraft velocity for deposition and WBF parameterizations.
        vertical_velocity: Vertical velocity [m/s] (nlev, ncols). Combined with
            TKE to compute representative updraft velocity for microphysics.
        params: CloudParams2M structure with tunable parameters
        dt: Microphysics timestep [s]
        ktdia: Top level for diagnostics (default 1)

    Returns:
        tuple of (MicrophysicsTendencies_2M, MicrophysicsState_2M)

    Notes:
        - Skips ARG activation (uses cdnc_factor from aerosol scheme instead)
        - Handles ice crystal number via nic_cirrus parameter:
          * nic_cirrus=1: Constant ice crystal number
          * nic_cirrus=2: Prognostic ice crystal number (xfrzmstr)
        - Processes are applied in sequence following ECHAM6 order
    """
    nlev, ncols = temperature.shape

    # Get timestep constants
    ztmst, ztmst_rcp, zcons1, zcons2, zcons3 = microphysics_dt_constants(dt)

    # Initialize working arrays
    zeps = params.eps
    tmelt = params.tmelt
    cthomi = params.cthomi
    cdnc = droplet_number

    # Latent heat / cpd ratios (from ECHAM6/ICON physical constants)
    # als = latent heat of sublimation, alv = latent heat of vaporization, cpd = 1004.64 J/kg/K
    # ECHAM uses q-correction: lsdcp = als / (cpd + zcons1*pqm1) where zcons1 = rd/cp
    # This accounts for the temperature dependence of the saturation specific humidity.
    # Compute reference qsat_water (for latent heat ratio correction) at a representative T.
    # Use T = tmelt as reference (the latent heat correction is fairly insensitive to T).
    esat_water_ref = es0 * jnp.exp(c1 * (tmelt - t0_sat) / (tmelt - t2))
    qsat_water_ref = eps * esat_water_ref / jnp.maximum(pressure[0] - (1.0 - eps) * esat_water_ref, zeps)
    lsdcp = als / (cpd + rd * qsat_water_ref)  # [K] (lsdcp used for ice-related terms)
    lvdcp = alv / (cpd + rd * qsat_water_ref)  # [K] (lvdcp used for liquid-related terms)

    # Teten's formula saturation vapor pressure helpers (Magnus formula)
    # esat_water(T)  = p0s1_bg * exp(c1*(T-t0)/(T-t2))   for T >= t0
    # esat_water(T)  = p0s1_bg * exp(c3*(T-t0)/(T-t4))   for T <  t0
    # esat_ice(T)    = p0s1_bg * exp(c3*(T-t0)/(T-t4))   (pure ice, T < t0)
    # Constants (ICON/ECHAM6 values):
    #   c1=17.502, t2=32.19  (water, T >= 273.16 K)
    #   c3=22.587, t4=-0.7   (water T < 273.16 K, and pure ice)
    t0_sat = t0                           # 273.15 K
    es0 = p0s1_bg                         # 611.21 Pa
    c1, t2 = 17.502, 32.19               # water above freezing
    c3, t4 = 22.587, -0.7                # water below freezing / pure ice

    def sat_vap_pressure_water(T: jnp.ndarray) -> jnp.ndarray:
        """Saturation vapor pressure over liquid water [Pa] using Teten's formula."""
        above = T >= t0_sat
        es_above = es0 * jnp.exp(c1 * (T - t0_sat) / (T - t2))
        es_below = es0 * jnp.exp(c3 * (T - t0_sat) / (T - t4))
        return jnp.where(above, es_above, es_below)

    def sat_vap_pressure_ice(T: jnp.ndarray) -> jnp.ndarray:
        """Saturation vapor pressure over ice [Pa] using Teten's formula."""
        return es0 * jnp.exp(c3 * (T - t0_sat) / (T - t4))

    def qsat_from_esat(esat: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """Specific humidity at saturation [kg/kg] from esat [Pa] and pressure [Pa]."""
        return eps * esat / jnp.maximum(p - (1.0 - eps) * esat, zeps)

    # Initialize prognostic ice crystal number (xfrzmstr) if nic_cirrus=2
    # For nic_cirrus=1, use prescribed constant value
    # Use lax.cond for JAX compatibility with static parameter
    def prognostic_icnc(_):
        return jnp.where(
            cloud_ice > params.epsec,
            cloud_ice * params.mi0_rcp,
            0.0
        )

    def prescribed_icnc(_):
        return jnp.where(
            temperature < cthomi,
            params.cn0s * 1e-6,  # Convert from 1/m³ to 1/kg equivalent
            0.0
        )

    icnc = lax.cond(
        int(params.nic_cirrus) == 2,
        prognostic_icnc,
        prescribed_icnc,
        None
    )

    # Initialize flux arrays (column-integrated, surface boundary conditions)
    rain_flux = jnp.zeros((nlev, ncols))
    snow_flux = jnp.zeros((nlev, ncols))

    # Initialize tendency accumulators
    specific_humidity_tendency = jnp.zeros((nlev, ncols))
    temp_tendency = jnp.zeros((nlev, ncols))
    ice_tendency = jnp.zeros((nlev, ncols))
    liq_tendency = jnp.zeros((nlev, ncols))
    tracer_tendency_cdnc = jnp.zeros((nlev, ncols))
    tracer_tendency_icnc = jnp.zeros((nlev, ncols))

    # Initialize diagnostic accumulators
    incloud_liq_before_rain = jnp.zeros((nlev, ncols))
    incloud_ice_before_snow = jnp.zeros((nlev, ncols))

    # Initialize process rate diagnostics
    autoconv_rate = jnp.zeros((nlev, ncols))
    accretion_rate = jnp.zeros((nlev, ncols))
    melting_rate = jnp.zeros((nlev, ncols))
    freezing_rate = jnp.zeros((nlev, ncols))

    # Temperature-dependent dynamic viscosity of air [kg/m/s]
    # From ECHAM6: zkair = 4.1867e-3 * (5.69 + 0.017*(T - tmelt))
    # Note: factor of 4.1867e-3 converts cal/(cm·s·K) to J/(m·s·K)
    dynamic_viscosity = 4.1867e-3 * (5.69 + 0.017 * (temperature - tmelt))

    # Updraft velocity for mixed-phase deposition and WBF [cm/s]
    # ECHAM6: zverv = fact_tke * sqrt(max(2*TKE, 0)) * 100 [cm/s]
    #               + abs(vertical_velocity) * 100 [cm/s]
    # where fact_tke = 0.7. The sqrt(2*TKE) represents the turbulent
    # contribution; the second term is the resolved vertical velocity.
    updraft_velocity = (
        params.fact_tke * jnp.sqrt(jnp.maximum(2.0 * tke, 0.0)) * 100.0
        + jnp.abs(vertical_velocity) * 100.0
    )

    # Initialize in-cloud values
    cloud_liquid_in_cloud = jnp.where(
        cloud_fraction > zeps,
        cloud_water / jnp.maximum(cloud_fraction, zeps),
        0.0
    )
    cloud_ice_in_cloud = jnp.where(
        cloud_fraction > zeps,
        cloud_ice / jnp.maximum(cloud_fraction, zeps),
        0.0
    )

    # Initialize number concentrations in cloud
    droplet_number = jnp.where(
        cloud_fraction > zeps,
        cdnc,
        0.0
    )
    ice_number = icnc.copy()

    # Initialize precip_cover (precipitation-covered fraction)
    precip_cover = jnp.zeros((nlev, ncols))

    # Initialize snow melt diagnostic
    snow_melt = jnp.zeros((nlev, ncols))

    # Pressure thickness for flux calculations
    pressure_thickness = layer_thickness * air_density * params.grav  # Approximate

    # --- Process sequence following ECHAM6 cloud_micro_interface ---

    # Step 1: Warm rain processes (autoconversion and accretion)
    # Only active where temperature > 273.15 K
    warm_mask = temperature > tmelt

    # Compute warm rain processes - use correct argument names matching function signature
    warm_results = precip_formation_warm(
        warm_precip_mask=warm_mask,
        autoconversion_factor=jnp.ones_like(temperature),  # pauloc - full participation
        cloud_fraction=cloud_fraction,
        minimum_cloud_precip_fraction=jnp.zeros_like(temperature),  # pclcstar - no rain from above initially
        air_density=air_density,
        rain_water=jnp.zeros_like(temperature),  # pxrp1 - no rain initially
        minimum_droplet_number=jnp.zeros_like(temperature),  # pcdnc_min - will be computed if needed
        droplet_number=droplet_number,
        cloud_water=cloud_liquid_in_cloud,
        dt=dt,
    )

    (
        droplet_number_updated,
        cloud_water_updated,
        autoconv_rate_in_cloud,
        autoconv_rate_gridmean,
        droplet_number_removal_rate,
        zrac1,
        zrac2,
    ) = warm_results

    rain_formation_warm = autoconv_rate_gridmean
    # Accretion: Beheng (1994) accretion rates from precip_formation_warm outputs.
    # zrac1 = accretion with rain from above (weighted by pclcstar = min(cloud_frac, precip_cover))
    # zrac2 = accretion with newly formed rain in grid-box (weighted by cloud_fraction)
    # Gridbox-mean accretion: cloud_fraction * zrac2 + pclcstar * zrac1
    pclcstar = jnp.minimum(cloud_fraction, precip_cover)
    accretion_warm = cloud_fraction * zrac2 + pclcstar * zrac1

    # Update accumulators and in-cloud values from warm processes
    autoconv_rate = autoconv_rate + rain_formation_warm
    accretion_rate = accretion_rate + accretion_warm
    cloud_liquid_in_cloud = cloud_water_updated
    droplet_number = droplet_number_updated

    # Step 2: Cold rain/ice processes (deposition, Bergeron-Findeisen, freezing)

    # Mixed-phase deposition and corrections
    # Use actual function signature from line 1257
    # Compute saturation vapor pressures (Teten's formula) and saturation q
    esat_water = sat_vap_pressure_water(temperature)
    esat_ice = sat_vap_pressure_ice(temperature)
    qsat_ice = qsat_from_esat(esat_ice, pressure)
    qsat_water = qsat_from_esat(esat_water, pressure)
    # qsat_prev: use the saturation q at current state as the "previous" estimate
    qsat_prev = jnp.where(temperature < tmelt, qsat_ice, qsat_water)

    # Bergeron variable: ratio of supersaturation over ice vs water
    # peta = (q - qsat_ice) / (qsat_water - qsat_ice) when mixed-phase
    # In pure liquid: peta = 0; in pure ice: peta = 1
    bergeron_variable = jnp.clip(
        (specific_humidity - qsat_ice) / jnp.maximum(qsat_water - qsat_ice, zeps),
        0.0, 1.0
    )

    mixed_results = mixed_phase_deposition_and_corrections(
        pressure=pressure,
        icnc=ice_number,
        specific_humidity_prev=specific_humidity,
        cloud_fraction=cloud_fraction,
        sat_vap_pres_ice=esat_ice,
        sat_vap_pres_water=esat_water,
        bergeron_variable=bergeron_variable,
        tompkins_genti=jnp.zeros_like(temperature),  # pgenti (not used in base scheme)
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        specific_humidity=specific_humidity,
        qsat_prev=qsat_prev,
        air_density=air_density,
        temperature=temperature,
        ice_evaporation=jnp.zeros_like(temperature),  # pxievap (updated by sublimation)
        ice_mmr_gridmean=cloud_ice_in_cloud,
        ice_detrainment_tendency=jnp.zeros_like(temperature),  # pxite (no detrainment here)
        updraft_velocity=updraft_velocity,  # pvervx [cm/s] (computed from TKE + vertical velocity)
        condensation_rate=jnp.zeros_like(temperature),  # pcnd (INOUT)
        deposition_rate=jnp.zeros_like(temperature),  # pdep (INOUT)
        dt=dt,
    )

    (
        condensation_rate,
        deposition_rate,
        temperature_tmp,
        specific_humidity_tmp,
        qsat_tmp,
    ) = mixed_results

    # Diagnostic correction terms for Tompkins cloud scheme (not used in base scheme = 0)
    tompkins_ice = jnp.zeros((nlev, ncols))
    tompkins_liq = jnp.zeros((nlev, ncols))
    # Ice sublimation / evaporation (from sublimation_snow_and_ice step)
    cloud_ice_evap = snow_sub  # [kg/kg] grid-mean ice evaporation/sublimation rate
    # Tendency corrections: in the full ECHAM scheme these come from subcolumns
    # In the base single-column scheme they are zero
    pxitec = jnp.zeros((nlev, ncols))  # ice tendency correction
    pxltec = jnp.zeros((nlev, ncols))  # liquid tendency correction
    pxisub = jnp.zeros((nlev, ncols))  # ice sublimation correction
    pxlevap = rain_evap  # Rain evaporation rate used for cloud cover correction

    # Freezing processes (below 238K)
    freezing_results = freezing_below_238K(
        temperature=temperature,
        cloud_liquid=cloud_liquid_in_cloud,
        cloud_ice=cloud_ice_in_cloud,
        droplet_number=droplet_number,
        params=params,
        dt=dt,
    )

    (
        freezing_rate_col,
        cloud_ice_after_freezing,
        cloud_liquid_after_freezing,
    ) = freezing_results

    freezing_rate = freezing_rate + freezing_rate_col

    # Heterogeneous mixed-phase freezing
    het_freezing_results = het_mxphase_freezing(
        temperature=temperature,
        cloud_liquid=cloud_liquid_after_freezing,
        cloud_ice=cloud_ice_after_freezing,
        droplet_number=droplet_number,
        ice_number=ice_number,
        params=params,
    )

    (
        ice_number_het,
        cloud_ice_het,
        cloud_liquid_het,
    ) = het_freezing_results

    ice_number = ice_number_het
    cloud_ice_in_cloud = cloud_ice_het
    cloud_liquid_in_cloud = cloud_liquid_het

    # WBF (Bergeron-Findeisen) process
    # WBF activates in mixed-phase conditions: T < tmelt, both liquid and ice present
    wbf_mask = (temperature < tmelt) & (cloud_liquid_in_cloud > zeps) & (cloud_ice_in_cloud > zeps)

    wbf_results = WBF_process(
        wbf_mask=wbf_mask,
        cloud_fraction=cloud_fraction,
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        cdnc=droplet_number,
        cloud_liquid_in_cloud=cloud_liquid_in_cloud,
        cloud_ice_in_cloud=cloud_ice_in_cloud,
        cloud_liquid_tendency=liq_tendency,
        cloud_ice_tendency=ice_tendency,
        temp_tendency=temp_tendency,
        dt=dt,
    )

    (
        wbf_rate,
        cloud_ice_wbf,
        cloud_liquid_wbf,
        droplet_number_wbf,
        liq_tendency_wbf,
        ice_tendency_wbf,
    ) = wbf_results

    cloud_ice_in_cloud = cloud_ice_wbf
    cloud_liquid_in_cloud = cloud_liquid_wbf
    droplet_number = droplet_number_wbf
    liq_tendency = liq_tendency_wbf
    ice_tendency = ice_tendency_wbf

    # Cold precipitation formation (snow autoconversion, aggregation)
    # Only active below freezing where ice cloud is present
    cold_mask = (temperature <= tmelt) & (cloud_ice_in_cloud > zeps)

    cold_precip_results = precip_formation_cold(
        cloud_mask=cold_mask,
        autoconversion_factor=jnp.ones_like(temperature),  # pauloc - full participation
        cloud_fraction=cloud_fraction,
        minimum_cloud_precip_fraction=jnp.zeros_like(temperature),  # pclcstar
        inverse_air_density=1.0 / jnp.maximum(air_density, zeps),
        inverse_air_density_rcp=jnp.maximum(air_density, zeps),  # already 1/rho, but keep for API
        temperature=temperature,
        dynamic_viscosity=dynamic_viscosity,  # pvko [kg/m/s], temperature-dependent from ECHAM
        snow_mass_mmr_from_above=jnp.zeros((nlev, ncols)),  # pxsp1 - no snow from above
        air_density=air_density,
        minimum_droplet_number=jnp.zeros_like(temperature),  # pcdnc_min
        ice_number=ice_number,
        droplet_number=droplet_number,
        snow_rate_in_cloud=jnp.zeros((nlev, ncols)),  # pmrateps (INOUT)
        in_cloud_ice=cloud_ice_in_cloud,
        in_cloud_liquid=cloud_liquid_in_cloud,
        dt=dt,
    )

    (
        ice_number_cold,
        droplet_number_cold,
        snow_rate_in_cloud,
        cloud_ice_cold,
        cloud_liquid_cold,
        ice_number_formation,
        snow_accretion_mass,
        snow_accretion_number,
        snow_accretion_mass_scav,
        snow_formation_gridmean,
    ) = cold_precip_results

    snow_formation = snow_formation_gridmean
    snow_accretion = snow_accretion_mass
    aggregation_rate = ice_number_formation  # Ice number loss by aggregation -> snow number
    snow_sublimation = jnp.zeros((nlev, ncols))  # Sublimation handled in step 5

    ice_number = ice_number_cold
    cloud_ice_in_cloud = cloud_ice_cold
    cloud_liquid_in_cloud = cloud_liquid_cold
    droplet_number = droplet_number_cold

    # Update ice tendency from cold processes
    ice_tendency = ice_tendency - snow_formation + aggregation_rate

    # Step 3: Sedimentation of ice crystals
    sedi_results = sedimentation_ice(
        cloud_ice=cloud_ice_in_cloud,
        ice_number=ice_number,
        pressure=pressure,
        temperature=temperature,
        cloud_fraction=cloud_fraction,
        params=params,
    )

    (
        ice_flux,
        ice_sedi_source,
    ) = sedi_results

    # Step 4: Melting of snow and ice (through melting level)
    melt_mask = temperature > tmelt

    melt_results = melting_snow_and_ice(
        melt_mask=melt_mask,
        temperature_previous=temperature,
        ice_cloud_previous=cloud_ice_in_cloud,
        pressure_thickness=pressure_thickness,
        icncq=ice_number,
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        icnc=ice_number,
        qmel=jnp.zeros((nlev, ncols)),  # Initialize
        cdnc=cdnc,
        rain_flux=rain_flux,
        snow_flux=snow_flux,
        ice_flux=ice_flux,
        ice_flux_n=jnp.zeros((nlev, ncols)),  # Initialize
        ice_tendency=ice_tendency,
        dt=dt,
    )

    (
        ice_number_melt,
        cdnc_melt,
        rain_flux_melt,
        snow_flux_melt,
        ice_flux_melt,
        melting_diag,
        ice_tendency_melt,
    ) = melt_results

    rain_flux = rain_flux_melt
    snow_flux = snow_flux_melt
    melting_rate = melting_rate + melting_diag
    ice_tendency = ice_tendency + ice_tendency_melt

    # Step 5: Sublimation of snow and evaporation of rain
    sub_evap_results = sublimation_snow_and_ice_evaporation_rain(
        rain_flux=rain_flux,
        snow_flux=snow_flux,
        temperature=temperature,
        pressure=pressure,
        specific_humidity=specific_humidity,
        cloud_fraction=cloud_fraction,
        params=params,
        dt=dt,
    )

    (
        rain_evap,
        snow_sub,
        rain_flux_sub,
        snow_flux_sub,
    ) = sub_evap_results

    rain_flux = rain_flux_sub
    snow_flux = snow_flux_sub

    # Step 6: Update precipitation fluxes through column
    precip_update_results = update_precip_fluxes(
        cloud_fraction=cloud_fraction,
        pressure_thickness=pressure_thickness,
        rain_evap_mmr=rain_evap,
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        rain_formation=rain_formation_warm,
        snow_accretion=snow_accretion,
        snow_formation=snow_formation,
        snow_sublimation_mmr=snow_sub,
        temp_tmp=temperature,
        ice_flux_from_above=ice_flux,
        precip_cover=precip_cover,
        rain_flux=rain_flux,
        snow_flux=snow_flux,
        snow_melt=snow_melt,
        dt=dt,
    )

    (
        precip_cover_upd,
        rain_flux_final,
        snow_flux_final,
        snow_melt_upd,
        pfevapr,
        pfrain,
        pfsnow,
        pfsubls,
    ) = precip_update_results

    precip_cover = precip_cover_upd
    rain_flux = rain_flux_final
    snow_flux = snow_flux_final
    snow_melt = snow_melt_upd

    # Step 7: Update tendencies and compute important diagnostics
    update_results = update_tendencies_and_important_vars(
        icnc=ice_number,
        cdnc=cdnc,
        ice_mmr_prev=cloud_ice_in_cloud,
        liq_mmr_prev=cloud_liquid_in_cloud,
        tracer_tm1_cdnc=cdnc,  # Previous timestep CDNC
        tracer_tm1_icnc=ice_number,  # Previous timestep ICNC
        condensation_rate=condensation_rate,
        deposition_rate=deposition_rate,
        rain_evap_mmr=rain_evap,
        freezing_rate=freezing_rate,
        tompkins_ice=tompkins_ice,
        tompkins_liq=tompkins_liq,
        incloud_ice_melt=jnp.zeros((nlev, ncols)),  # Diagnostic
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        air_density=air_density,
        inv_air_density=1.0 / jnp.maximum(air_density, zeps),
        rain_formation=rain_formation_warm,
        snow_accretion=snow_accretion,
        snow_formation=snow_formation,
        cloud_ice_evap=cloud_ice_evap,
        ice_flux_melt=jnp.zeros((nlev, ncols)),  # Diagnostic
        pxitec=pxitec,
        pxlevap=pxlevap,
        pxltec=pxltec,
        pxisub=pxisub,
        snow_sublimation_mmr=snow_sub,
        snow_melt=snow_melt,
        cloud_ice_in_cloud=cloud_ice_in_cloud,
        cloud_liquid_in_cloud=cloud_liquid_in_cloud,
        temp_tmp=temperature,
        liquid_cloud_flag=temperature > tmelt,
        ice_cloud_flag=temperature <= tmelt,
        cloud_fraction=cloud_fraction,
        specific_humidity_tendency=specific_humidity_tendency,
        temp_tendency=temp_tendency,
        ice_tendency=ice_tendency,
        liq_tendency=liq_tendency,
        tracer_tendency_cdnc=tracer_tendency_cdnc,
        tracer_tendency_icnc=tracer_tendency_icnc,
        incloud_liq_before_rain=incloud_liq_before_rain,
        incloud_ice_before_snow=incloud_ice_before_snow,
        dt=dt,
    )

    (
        cloud_fraction_final,
        specific_humidity_tendency_final,
        temp_tendency_final,
        ice_tendency_final,
        liq_tendency_final,
        tracer_tendency_cdnc_final,
        tracer_tendency_icnc_final,
        incloud_liq_final,
        incloud_ice_final,
        liq_eff_radius,
        ice_eff_radius,
    ) = update_results

    # Surface precipitation (last level)
    precip_rain = rain_flux_final[-1, :] if nlev > 1 else rain_flux_final[0, :]
    precip_snow = snow_flux_final[-1, :] if nlev > 1 else snow_flux_final[0, :]

    # Build tendency result
    tendencies = MicrophysicsTendencies_2M(
        dtedt=temp_tendency_final,
        dqdt=specific_humidity_tendency_final,
        dqcdt=liq_tendency_final,
        dqidt=ice_tendency_final,
        dqncdt=tracer_tendency_cdnc_final,
        dqnidt=tracer_tendency_icnc_final,
        dqrdt=jnp.zeros((nlev, ncols)),  # Rain tendency (diagnostic)
        dqsdt=jnp.zeros((nlev, ncols)),  # Snow tendency (diagnostic)
    )

    # Build state result
    state = MicrophysicsState_2M(
        rain_flux=rain_flux_final,
        snow_flux=snow_flux_final,
        qc_in_cloud=cloud_liquid_in_cloud,
        qi_in_cloud=cloud_ice_in_cloud,
        qnc_in_cloud=droplet_number,
        qni_in_cloud=ice_number,
        autoconv_rate=autoconv_rate,
        accretion_rate=accretion_rate,
        melting_rate=melting_rate,
        freezing_rate=freezing_rate,
        precip_rain=precip_rain,
        precip_snow=precip_snow,
    )

    return tendencies, state