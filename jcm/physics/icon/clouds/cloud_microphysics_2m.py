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
                          consistency_number_to_mass, gridbox_frac_falling_hydrometeor, threshold_vert_vel
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

def cloud_micro_interface():
    """Placeholder for cloud microphysics interface function. 
    Link between microphysics and aerosol scheme via activation TODO"""
    pass

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

import jax.numpy as jnp

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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    # Local process rates (mass increments) [kg/kg]
    zrac1 = jnp.zeros_like(cloud_water)     # accretion with rain from above
    zrac2 = jnp.zeros_like(cloud_water)     # accretion with newly formed rain
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

    return droplet_number, cloud_water, autoconversion_rate_in_cloud, autoconversion_rate, droplet_number_removal_rate

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
    kk: int,
    klev: int,
    cloud_fraction: jnp.ndarray,                  # paclc
    pressure_thickness: jnp.ndarray,              # pdp [Pa]
    evaporation_rain_mmr: jnp.ndarray,            # rain_evap [kg/kg]
    lsdcp: jnp.ndarray,                           # plsdcp = Ls/cpd
    lvdcp: jnp.ndarray,                           # plvdcp = Lv/cpd
    rain_formation_rate: jnp.ndarray,             # prpr [kg/kg]
    snow_droplet_accretion_mmr: jnp.ndarray,      # psacl [kg/kg]
    snow_formation_mmr: jnp.ndarray,              # pspr [kg/kg]
    sublimation_snow_mmr: jnp.ndarray,            # snow_sublim [kg/kg]
    temperature: jnp.ndarray,                     # ptp1tmp [K]
    ice_flux_from_above: jnp.ndarray,             # pxiflux [kg/m2/s]
    precip_fraction: jnp.ndarray,                 # pclcpre (INOUT) [0..1]
    rain_flux: jnp.ndarray,                       # prfl (INOUT) [kg/m2/s]
    snow_flux: jnp.ndarray,                       # psfl (INOUT) [kg/m2/s]
    snow_melt_mmr: jnp.ndarray,                   # psmlt (INOUT) [kg/kg]
    dt: jnp.ndarray,                              # ztmst [s]
) -> tuple[
    jnp.ndarray,  # precip_fraction (updated)
    jnp.ndarray,  # rain_flux (updated)
    jnp.ndarray,  # snow_flux (updated)
    jnp.ndarray,  # snow_melt_mmr (updated)
    jnp.ndarray,  # pfevapr  evaporation of rain [kg/m2/s]
    jnp.ndarray,  # pfrain   rain flux before evaporation [kg/m2/s]
    jnp.ndarray,  # pfsnow   snow flux before sublimation [kg/m2/s]
    jnp.ndarray,  # pfsubls  sublimation of snow [kg/m2/s]
]:
    """
    JAX port of Fortran subroutine `update_precip_fluxes`.

    This updates rain/snow precipitation fluxes with local source terms (rain/snow formation),
    applies evaporation/sublimation sinks, conditionally melts snow at the lowest level, and
    updates precipitating area fraction `precip_fraction`.

    Notes
    -----
    - Fortran indexing uses (kbdim,kproma) and level counters (kk,klev). Here we assume
      arrays already correspond to the horizontal slice (kproma), and kk/klev are Python ints.
    - Uses helper `gridbox_frac_falling_hydrometeor(...)` (must exist in this module or be imported).
    """
    # Microphysics timestep constants
    _, _, _, zcons2, _ = microphysics_dt_constants(dt)

    # local formed precip flux increments in this layer
    zzdrr = zcons2 * pressure_thickness * rain_formation_rate
    zzdrs = zcons2 * pressure_thickness * (snow_formation_mmr + snow_droplet_accretion_mmr)

    # lowest level special treatment (kk == klev): add ice flux from above and melt snow if T > Tmelt
    is_lowest = (kk == klev)

    if is_lowest:
        zzdrs = zzdrs + ice_flux_from_above

        # zcons = zcons2*pdp/(plsdcp-plvdcp) * max(0, T-Tmelt)
        zcons = (
            zcons2
            * pressure_thickness
            / jnp.maximum(lsdcp - lvdcp, eps)
            * jnp.maximum(0.0, temperature - tmelt)
        )

        # zsnmlt = min(xsec*zzdrs, zcons)
        zsnmlt = jnp.minimum(xsec * zzdrs, zcons)

        # transfer melted snow to rain
        zzdrr = zzdrr + zsnmlt
        zzdrs = zzdrs - zsnmlt

        # update diagnostic snow melt mixing ratio tendency (kg/kg)
        snow_melt_mmr = snow_melt_mmr + zsnmlt / jnp.maximum(zcons2 * pressure_thickness, eps)

    # total precip from above and formed here
    zpretot = rain_flux + snow_flux
    zpredel = zzdrr + zzdrs

    # update precipitating fraction (area of falling hydrometeors)
    precip_fraction = gridbox_frac_falling_hydrometeor(
        zpretot=zpretot,
        precip_fraction=precip_fraction,
        zpredel=zpredel,
        cloud_fraction=cloud_fraction,
    )

    # in-cloud scavenging diagnostics, only where precip_fraction > epsec
    ll1 = precip_fraction > epsec

    denom = jnp.maximum(precip_fraction, epsec)

    # rain/snow flux before evaporation/sublimation (per precipitating area)
    pfrain_raw = (rain_flux + zzdrr) / denom
    pfsnow_raw = (snow_flux + zzdrs) / denom
    pfrain = jnp.where(ll1, pfrain_raw, 0.0)
    pfsnow = jnp.where(ll1, pfsnow_raw, 0.0)

    # evaporation/sublimation flux rates per precipitating area
    pfevapr_raw = (zcons2 * pressure_thickness * evaporation_rain_mmr) / denom
    pfsubls_raw = (zcons2 * pressure_thickness * sublimation_snow_mmr) / denom
    pfevapr = jnp.where(ll1, pfevapr_raw, 0.0)
    pfsubls = jnp.where(ll1, pfsubls_raw, 0.0)

    # update fluxes with sources and sinks (grid-mean)
    rain_flux = rain_flux + zzdrr - zcons2 * pressure_thickness * evaporation_rain_mmr
    snow_flux = snow_flux + zzdrs - zcons2 * pressure_thickness * sublimation_snow_mmr

    return (
        precip_fraction,
        rain_flux,
        snow_flux,
        snow_melt_mmr,
        pfevapr,
        pfrain,
        pfsnow,
        pfsubls,
    )

def update_in_cloud_water(
    aerosol_total: jnp.ndarray, 
    activated_cdnc: jnp.ndarray, 
    condensation_increment: jnp.ndarray, 
    deposition_increment: jnp.ndarray, 
    cloud_cover_vari_i: jnp.ndarray, 
    cloud_cover_vari_l: jnp.ndarray, 
    activated_icnc: jnp.ndarray, 
    specific_humidity: jnp.ndarray, 
    saturation_specific_humidity: jnp.ndarray, 
    air_density: jnp.ndarray, 
    ice_mean_volume_radius: jnp.ndarray, 
    temperature_previous: jnp.ndarray,
    cloud_flag: jnp.ndarray, 
    icnc: jnp.ndarray, 
    droplet_nucleation_accumulated: jnp.ndarray, 
    cdnc: jnp.ndarray, 
    cloud_fraction: jnp.ndarray, 
    in_cloud_ice_mixing_ratio: jnp.ndarray,
    in_cloud_water_mixing_ratio: jnp.ndarray, 
    dt: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update in-cloud condensate (liquid/ice), droplet/ice number concentrations, and cloud cover.
    TODO : debug to account for the dependency on time step length

    This function is a close port of the ECHAM6/ICON Fortran subroutine
    `update_in_cloud_water` (see mo_cloud_microphysics_2m). It performs a
    bookkeeping/consistency update after condensation/deposition tendencies have
    been computed, and after droplet activation / ice nucleation sources are known.

    Conceptually, this routine:
      (A) Updates in-cloud liquid water (in_cloud_water_mixing_ratio) and in-cloud ice (in_cloud_ice_mixing_ratio) by adding
          condensation/deposition increments, accounting for cloud fraction.
      (B) Creates cloud cover (cloud_fraction) in newly cloudy grid boxes based on relative
          humidity if condensation/deposition occurs when no cloud existed.
      (C) Computes a minimum CDNC (cdnc_min) implied by a maximum droplet size
          constraint (via `minimum_CDNC`).
      (D) Updates CDNC (cdnc) using newly activated droplet number (activated_cdnc),
          but only if CDNC is below the minimum and the temperature is warm enough.
      (E) Updates ICNC (icnc) if there is cloud ice and existing ICNC is too small,
          using a cirrus nucleation parametrization selector (nic_cirrus).

    Notes on ICON/ECHAM conventions
    -------------------------------
    - Many quantities are "in-cloud" (i.e., defined only in the cloudy fraction).
      Specifically:
        * in_cloud_water_mixing_ratio: in-cloud cloud liquid water mixing ratio [kg/kg]
        * in_cloud_ice_mixing_ratio: in-cloud cloud ice mixing ratio [kg/kg]
      When adding condensation/deposition increments (condensation_increment/deposition_increment), the increment is
      divided by max(cloud_fraction, clc_min) to convert from grid-mean increment to the
      in-cloud increment (Fortran uses MAX(cloud_fraction, clc_min) to avoid division by 0).

    - `cloud_flag` is a boolean flag indicating whether the layer is considered cloudy.
      In the Fortran code it is redefined midroutine from cloud_fraction (see below), so
      its meaning changes slightly.

    - JAX port: Fortran MERGE(a,b,mask) is implemented as jnp.where(mask, a, b).

    Parameters (names match Fortran)
    --------------------------------
    aerosol_total : jnp.ndarray
        Total number of aerosols available (used in one cirrus option).
    activated_cdnc : jnp.ndarray
        Number concentration of newly activated cloud droplets [1/m^3].
    condensation_increment : jnp.ndarray
        Condensation increment / rate proxy for this step [kg/kg].
        (In Fortran: "condensation rate", but treated as an increment here.)
    deposition_increment : jnp.ndarray
        Deposition increment / rate proxy for this step [kg/kg].
    cloud_cover_vari_i, cloud_cover_vari_l : jnp.ndarray
        Additional sources used in Tompkins cloud cover scheme bookkeeping.
        In Fortran they are added to deposition_increment/condensation_increment and then clipped to >= 0.
    activated_icnc : jnp.ndarray
        Number concentration of newly formed ice crystals (used in cirrus option 2).
    specific_humidity : jnp.ndarray
        Updated specific humidity at time t [kg/kg].
    saturation_specific_humidity : jnp.ndarray
        Updated saturation specific humidity [kg/kg].
    air_density : jnp.ndarray
        Air density [kg/m^3].
    ice_mean_volume_radius : jnp.ndarray
        Mean volume radius of ice crystals [m] (used in cirrus option 1).
    temperature_previous : jnp.ndarray
        Temperature at previous step (t-1) [K].

    cloud_flag : jnp.ndarray (bool)
        Cloud-present flag (in/out).
    icnc : jnp.ndarray
        Ice crystal number concentration (ICNC) [1/m^3] (in/out).
    droplet_nucleation_accumulated : jnp.ndarray
        Accumulated droplet nucleation number (in/out).
        NOTE: In the original Fortran droplet_nucleation_accumulated is [1/m^3/s] and is incremented by zdt*ΔN.
        In this Python port you currently add ΔN directly (see comment below).
    cdnc : jnp.ndarray
        Cloud droplet number concentration (CDNC) [1/m^3] (in/out).
    cloud_fraction : jnp.ndarray
        Cloud fraction / cloud cover [0..1] (in/out).
    in_cloud_ice_mixing_ratio : jnp.ndarray
        In-cloud ice mixing ratio [kg/kg] (in/out).
    in_cloud_water_mixing_ratio : jnp.ndarray
        In-cloud liquid mixing ratio [kg/kg] (in/out).

    Returns
    -------
    cloud_flag : bool array
        Updated cloud-present flag (rederived from cloud_fraction).
    icnc : array
        Updated ICNC [1/m^3].
    droplet_nucleation_accumulated : array
        Updated nucleation accumulator.
    cdnc : array
        Updated CDNC [1/m^3].
    cloud_fraction : array
        Updated cloud cover [0..1].
    in_cloud_ice_mixing_ratio : array
        Updated in-cloud ice mixing ratio [kg/kg].
    in_cloud_water_mixing_ratio : array
        Updated in-cloud liquid mixing ratio [kg/kg].
    cdnc_min : array
        Minimum CDNC [1/m^3] computed from maximum droplet-size constraint.
    """

    # ---------------------------------------------------------------------
    # 0) Relative humidity and "effective" condensation/deposition increments
    # ---------------------------------------------------------------------
    # Here we guard saturation_specific_humidity to avoid inf/nan when saturation is tiny.
    relative_humidity = specific_humidity / jnp.maximum(saturation_specific_humidity, 1e-12)

    # These "ztmp1/ztmp2" represent non-negative sources that can establish or
    # enhance cloud condensate.
    ztmp1 = jnp.maximum(deposition_increment + cloud_cover_vari_i, 0.0)   # (ice-side) deposition-like source
    ztmp2 = jnp.maximum(condensation_increment + cloud_cover_vari_l, 0.0)   # (liquid-side) condensation-like source

    # ---------------------------------------------------------------------
    # 1) If a grid box is already cloudy (cloud_flag=True), update in-cloud condensate
    # ---------------------------------------------------------------------
    # Convert grid-mean condensation/deposition increments to in-cloud increments
    # by dividing by max(cloud_fraction, clc_min). This avoids huge increments when cloud_fraction ~ 0.
   
    ztmp3 = in_cloud_ice_mixing_ratio + deposition_increment / jnp.maximum(cloud_fraction, clc_min) # in-cloud ice growth increment
    ztmp3 = jnp.maximum(ztmp3, 0.0) # 

    ztmp4 = in_cloud_water_mixing_ratio + condensation_increment / jnp.maximum(cloud_fraction, clc_min) # in-cloud liquid growth increment
    ztmp4 = jnp.maximum(ztmp4, 0.0)

    # Apply only where cloud_flag indicates "cloud exists".
    # (MERGE(ztmp3, in_cloud_ice_mixing_ratio, cloud_flag) => where(cloud_flag, ztmp3, in_cloud_ice_mixing_ratio))
    in_cloud_ice_mixing_ratio = jnp.where(cloud_flag, ztmp3, in_cloud_ice_mixing_ratio)
    in_cloud_water_mixing_ratio = jnp.where(cloud_flag, ztmp4, in_cloud_water_mixing_ratio)

    # ---------------------------------------------------------------------
    # 2) If there is no cloud yet, but condensation/deposition wants to occur,
    #    create cloud cover and initialize in-cloud condensate.
    # ---------------------------------------------------------------------
    # i.e. "new cloud formation condition".
    ll1 = jnp.logical_and(~cloud_flag, jnp.logical_or(ztmp1 > 0.0, ztmp2 > 0.0))

    # if new cloud is created, set cloud cover to a RH-based value.
    ztmp3 = jnp.clip(relative_humidity, 0.01, 1.0)
    cloud_fraction = jnp.where(ll1, ztmp3, cloud_fraction)

    # Initialize in-cloud ice/liquid in those newly cloudy points using the
    # same "grid-mean to in-cloud" conversion.
    ztmp3 = ztmp1 / jnp.maximum(cloud_fraction, clc_min)  # deposition source -> in-cloud ice
    ztmp4 = ztmp2 / jnp.maximum(cloud_fraction, clc_min)  # condensation source -> in-cloud liquid

    in_cloud_ice_mixing_ratio = jnp.where(ll1, ztmp3, in_cloud_ice_mixing_ratio)
    in_cloud_water_mixing_ratio = jnp.where(ll1, ztmp4, in_cloud_water_mixing_ratio)

    # ---------------------------------------------------------------------
    # 3) Compute minimum CDNC implied by maximum droplet radius constraint
    # ---------------------------------------------------------------------
    # Here ztmp1 is cloud liquid water mass concentration [kg/m^3]
    # (because in_cloud_water_mixing_ratio is [kg/kg] and air_density is [kg/m^3]).
    ztmp1 = in_cloud_water_mixing_ratio * air_density
    cdnc_min = minimum_CDNC(ztmp1)

    # ---------------------------------------------------------------------
    # 4) Re-derive cloud_flag from cloud fraction
    # ---------------------------------------------------------------------
    cloud_flag = cloud_fraction > 0.0

    # ---------------------------------------------------------------------
    # 5) Update CDNC (cdnc) + accumulate nucleation proxy (droplet_nucleation_accumulated)
    # ---------------------------------------------------------------------
    # ll1: only if cloudy and enough liquid exists.
    ll1 = jnp.logical_and(cloud_flag, in_cloud_water_mixing_ratio > cqtmin)

    # ll2: only if CDNC is <= minimum and temperature is above homogeneous
    # nucleation threshold (temperature_previous > cthomi).
    # Interpretation: If CDNC is too low compared with what maximum droplet size
    # would allow, then allow additional activated droplets (activated_cdnc) to increase
    # CDNC, but not in very cold conditions.
    ll2 = jnp.logical_and(ll1, jnp.logical_and(cdnc <= cdnc_min, temperature_previous > cthomi))

    # Candidate increase in CDNC from activation: ΔN = max(activated_cdnc - cdnc, 0)
    ztmp1 = jnp.maximum(0.0, activated_cdnc - cdnc)

    # nucleation_rate_cdnc holds actual nucleation increment (for diagnostics in Fortran).
    nucleation_rate_cdnc = jnp.where(ll2, ztmp1, 0.0)

    # Apply the increase to CDNC and the accumulator.
    cdnc = cdnc + nucleation_rate_cdnc

    # The Fortran code accumulates droplet nucleation as droplet_nucleation_accumulated = droplet_nucleation_accumulated + zdt * ΔN.
    # In this Python implementation, the `dt` parameter is explicitly passed to the function.
    # The accumulation is performed as droplet_nucleation_accumulated += dt * nucleation_rate_cdnc,
    # ensuring that droplet_nucleation_accumulated is updated correctly as [1/m^3/s].
    droplet_nucleation_accumulated += dt * nucleation_rate_cdnc

    # Enforce at least minimum CDNC if cloudy & has liquid, otherwise set to cqtmin.
    ztmp1 = jnp.maximum(cdnc, cdnc_min)
    cdnc = jnp.where(ll1, ztmp1, cqtmin)

    # ---------------------------------------------------------------------
    # 6) Update ICNC (icnc) based on chosen cirrus nucleation scheme
    # ---------------------------------------------------------------------
    # ll1: only if cloudy and enough ice exists.
    ll1 = jnp.logical_and(cloud_flag, in_cloud_ice_mixing_ratio > cqtmin)

    # ll2: only if ICNC is too small (<= icemin) in icy cloud.
    ll2 = jnp.logical_and(ll1, icnc <= icemin)

    # This produces a diagnostic ICNC consistent with available ice mass and
    # a prescribed crystal size (option 1), or from an explicit nucleation number
    # limited by aerosol availability (option 2).
    if nic_cirrus == 1:
        ztmp1 = 0.75 / (pi * rhoice) * air_density * in_cloud_ice_mixing_ratio / jnp.maximum(ice_mean_volume_radius**3, 1e-12)
    elif nic_cirrus == 2:
        ztmp1 = jnp.minimum(activated_icnc, aerosol_total * 1e6)

    # Update icnc only where ll2 requests a reset/increase.
    icnc = jnp.where(ll2, ztmp1, icnc)

    # Enforce a minimum ICNC in icy clouds; otherwise set to cqtmin (as in Fortran).
    ztmp1 = jnp.maximum(icnc, icemin)
    icnc = jnp.where(ll1, ztmp1, cqtmin)
    
    #return cloud_flag, icnc, droplet_nucleation_accumulated, cdnc, cloud_fraction, in_cloud_ice_mixing_ratio, in_cloud_water_mixing_ratio, cdnc_min
    pass

