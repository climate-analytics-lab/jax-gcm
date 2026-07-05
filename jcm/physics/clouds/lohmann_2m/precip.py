"""Precipitation formation and fluxes for the Lohmann 2M scheme.

Warm-rain (autoconversion/accretion) and cold (aggregation/accretion)
precipitation formation, precipitation flux updates, and snow/ice
sublimation plus rain evaporation. Split out of the monolithic
``lohmann_2m.py`` module (pure move, no numerical change).
"""

import jax.numpy as jnp
from math import pi

import jcm.constants as c

from ..lohmann_2m_params import CloudParams2M
from ..cloud_utils import (
    consistency_number_to_mass,
    eff_ice_crystal_radius,
    gridbox_frac_falling_hydrometeor,
)
from .types import microphysics_dt_constants


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
    params: CloudParams2M,
) -> tuple[
    jnp.ndarray,  # ice_flux (updated) [kg/m^2/s]
    jnp.ndarray,  # ice_flux_n (updated) [1/m^2/s]
    jnp.ndarray,  # ice_sublim (sublimation of falling ice) [kg/kg]
    jnp.ndarray,  # snow_sublim   (sublimation of snow) [kg/kg]
    jnp.ndarray,  # rain_evap   (evaporation of rain) [kg/kg]
]:
    """Sublimation of snow and *falling* ice + evaporation of rain (ICON/ECHAM 2-moment scheme).

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
    ztmst, _, _, zcons2, zcons3 = microphysics_dt_constants(dt, params)

    # ------------------------------------------------------------------
    # Common diffusion/ventilation coefficient for ice-phase sublimation
    # ------------------------------------------------------------------
    denom = (1.0 / (2.43e-2 * c.rv)) * (lsdcp**2) / jnp.maximum(temperature_prev**2, params.eps)
    denom = denom + (1.0 / 0.211e-4) * inv_air_density_rcp / jnp.maximum(qsat_ice, params.eps)
    zcoeff = 3.0e6 * 2.0 * pi * subsat_wrt_ice * inv_air_density_rcp / jnp.maximum(denom, params.eps)

    # Avoid division by zero for area fractions: MERGE(frac, 1, mask)
    zclcpre = jnp.where(precip_mask, precip_fraction, 1.0)
    zclcfi = jnp.where(falling_ice_mask, falling_ice_fraction, 1.0)

    # ------------------------------------------------------------------
    # Snow sublimation (snow_sublim)
    # ------------------------------------------------------------------
    ll_snow = jnp.logical_and(snow_flux > params.cqtmin, precip_mask)

    # Double-where guard: the fractional power has an infinite derivative at
    # a zero base, and ``snow_sublim`` is where-masked below — masking the
    # output alone still yields NaN gradients (0 cotangent × ∞ derivative).
    # A safe base of 1.0 where ll_snow is False keeps forward values
    # unchanged while keeping the backward pass finite.
    zclambs_s_base = jnp.where(ll_snow, snow_flux / jnp.maximum(zclcpre, params.eps), 1.0)
    zclambs_s = zcons3 * zclambs_s_base ** (0.25 / 1.16)
    zcfac4c_s = 0.78 * zclambs_s**2 + 232.19 * (inv_air_density**0.25) * (zclambs_s**2.625)
    ztmp2_s = zcfac4c_s * zcoeff * dp_over_g

    zzeps_s = jnp.maximum(-params.xsec * snow_flux / jnp.maximum(zclcpre, params.eps), ztmp2_s)
    ztmp3_s = -ztmst * zzeps_s / jnp.maximum(dp_over_g, params.eps) * zclcpre

    ztmp4_s = jnp.maximum(params.xsec * (qsat_ice - specific_humidity_prev), 0.0)
    ztmp3_s = jnp.clip(ztmp3_s, 0.0, ztmp4_s)
    snow_sublim = jnp.where(ll_snow, ztmp3_s, 0.0)

    # ------------------------------------------------------------------
    # Falling ice sublimation (ice_sublim) and update ice_flux, ice_flux_n
    # ------------------------------------------------------------------
    ll_ice = jnp.logical_and(ice_flux > params.cqtmin, falling_ice_mask)

    # Same double-where guard as snow sublimation above (ice_sublim and
    # zsubin are where-masked on ll_ice).
    zclambs_i_base = jnp.where(ll_ice, ice_flux / jnp.maximum(zclcfi, params.eps), 1.0)
    zclambs_i = zcons3 * zclambs_i_base ** (0.25 / 1.16)
    zcfac4c_i = 0.78 * zclambs_i**2 + 232.19 * (inv_air_density**0.25) * (zclambs_i**2.625)
    ztmp2_i = zcfac4c_i * zcoeff * dp_over_g

    zzeps_i = jnp.maximum(-params.xsec * ice_flux / jnp.maximum(zclcfi, params.eps), ztmp2_i)
    ztmp3_i = -ztmst * zzeps_i / jnp.maximum(dp_over_g, params.eps) * zclcfi

    ztmp4_i = jnp.maximum(params.xsec * (qsat_ice - specific_humidity_prev), 0.0)
    ztmp3_i = jnp.clip(ztmp3_i, 0.0, ztmp4_i)
    ice_sublim = jnp.where(ll_ice, ztmp3_i, 0.0)

    # number flux reduction due to sublimated mass
    zsubin = ice_sublim * ice_flux_n / jnp.maximum(ice_flux, params.cqtmin)
    zsubin = zcons2 * zsubin * pressure_thickness
    zsubin = jnp.where(ll_ice, zsubin, 0.0)

    ice_flux_n = ice_flux_n - zsubin
    ice_flux = ice_flux - zcons2 * ice_sublim * pressure_thickness

    ice_flux_n = consistency_number_to_mass(pthreshold=params.epsec, pmass=ice_flux, pnumber=ice_flux_n)

    # ------------------------------------------------------------------
    # Rain evaporation (rain_evap)
    # ------------------------------------------------------------------
    ll_rain = jnp.logical_and(rain_flux > params.cqtmin, precip_mask)

    # Same double-where guard as snow sublimation above (rain_evap is
    # where-masked on ll_rain).
    zrain_pow_base = jnp.where(ll_rain, rain_flux / jnp.maximum(zclcpre, params.eps), 1.0)
    ztmp2_r = (
        870.0
        * subsat_wrt_water_evap
        * dp_over_g
        * zrain_pow_base ** 0.61
        / (jnp.sqrt(jnp.maximum(air_density, params.eps)) * jnp.maximum(thermo_term_water, params.eps))
    )

    zzeps_r = jnp.maximum(-params.xsec * rain_flux / jnp.maximum(zclcpre, params.eps), ztmp2_r)
    ztmp3_r = -ztmst * zzeps_r * zclcpre / jnp.maximum(dp_over_g, params.eps)

    ztmp4_r = jnp.maximum(params.xsec * (qsat_water_prev - specific_humidity_prev), 0.0)
    ztmp3_r = jnp.clip(ztmp3_r, 0.0, ztmp4_r)
    rain_evap = jnp.where(ll_rain, ztmp3_r, 0.0)

    return ice_flux, ice_flux_n, ice_sublim, snow_sublim, rain_evap


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
    params: CloudParams2M,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Warm-rain precipitation formation for the 2-moment microphysics scheme.

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
    ztmp1 = params.ccraut * 1350.0 * (1e-6 * droplet_number) ** (-1.79)

    # The expression below is a time-integrated sink form used in the Fortran.
    # It is constructed so that zraut is bounded by cloud_water (after MIN).
    # Double-where the power base: in partially-cloudy columns the
    # in-cloud cloud_water can be exactly 0 (or transiently negative
    # from upstream arithmetic), and x**1.47 / the outer **(-0.68)
    # then NaN the parameter cotangents even though the warm mask
    # discards the value (0*NaN in reverse mode).
    has_lw = cloud_water > 0.0
    lw_safe = jnp.where(has_lw, cloud_water, 1.0)
    ztmp1 = jnp.where(
        has_lw,
        lw_safe * (
            1.0
            - (
                1.0
                + dt * params.exm1_1 * ztmp1 * lw_safe ** params.exm1_1
            ) ** params.exp_1
        ),
        0.0,
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
        (zraut + zrac1 + zrac2) / (cloud_water_before + params.eps),
        0.0,
    )

    # Only limit droplet number when cloud water is still meaningful (> cqtmin).
    ll1 = jnp.logical_and(warm_precip_mask, cloud_water > params.cqtmin)

    # Enforce a minimum allowed droplet number (pcdnc_min) only when ll1 is true.
    min_allowed = jnp.where(ll1, minimum_droplet_number, 0.0)

    # Available droplet number above the minimum
    available = droplet_number - min_allowed

    # "Requested" droplet reduction based on droplet_number_removal_rate proxy:
    requested = droplet_number * droplet_number_removal_rate

    # Actual reduction is limited by what is available above minimum
    droplet_number_removal_rate = jnp.where(warm_precip_mask, jnp.minimum(available, requested), 0.0)

    # Update droplet number concentration, keep >= cqtmin
    droplet_number_new = jnp.maximum(droplet_number - droplet_number_removal_rate, params.cqtmin)
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
    params: CloudParams2M,                         # threaded scheme parameters
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
    """Cold-phase precipitation formation for the ICON/ECHAM 2-moment scheme.

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
    zxibold = jnp.maximum(in_cloud_ice, params.eps)  # noqa: F841 — store pxib with security for later use (Phase 5b)
    zsaut = jnp.zeros_like(in_cloud_ice)      # aggregation mass [kg/kg]
    zxsp2 = jnp.zeros_like(in_cloud_ice)      # snow formed inside box (mass conc proxy) [??]
    zsaclin = jnp.zeros_like(in_cloud_ice)    # in-cloud droplet mass accreted by snow [kg/kg]
    zsaci = jnp.zeros_like(in_cloud_ice)      # ice accreted by snow [kg/kg]
    zsecprod = jnp.zeros_like(in_cloud_ice)   # secondary ice production mass [kg/kg] (not implemented here)

    # ---------------------------------------------------------------------
    # 0) Early mask: only proceed where there is cloud and enough ice
    # ---------------------------------------------------------------------
    ll1 = jnp.logical_and(cloud_mask, in_cloud_ice > params.cqtmin)

    # If ll1 is false everywhere, Fortran returns early. In JAX we just mask.
    # (no-op if all masked)
    # ---------------------------------------------------------------------
    # 1) Compute effective ice-crystal "size" zris based on effective radius
    # ---------------------------------------------------------------------
    # Convert in-cloud ice from kg/kg to in-cloud g/m^3: 1000*pxib*prho
    ice_gm3 = 1000.0 * in_cloud_ice * air_density

    # eff_ice_crystal_radius expects (ice_gm3, icnc). If you already have such a helper,
    # call it; otherwise this will need to be implemented.
    zrieff = eff_ice_crystal_radius(ice_gm3, ice_number, params)  # [micron] typically (scheme-dependent)

    # Clip effective radius bounds
    zrieff = jnp.minimum(jnp.maximum(zrieff, params.ceffmin), params.ceffmax)

    # Compute zrih then zris = 1e-6 * zrih**(1/3)
    zrih = -2261.0 + jnp.sqrt(5113188.0 + 2809.0 * zrieff**3)
    zris = 1.0e-6 * (zrih ** (1.0 / 3.0))

    # Fortran MERGE(..., 1., ll1): just ensure non-zero where masked off
    zris = jnp.where(ll1, zris, 1.0)

    # ---------------------------------------------------------------------
    # 2) Temperature-dependent collision efficiency for aggregation
    # ---------------------------------------------------------------------
    zcolleffi = jnp.exp(params.fact_coll_eff * (temperature - c.tmelt))
    zcolleffi = jnp.where(ll1, zcolleffi, 0.0)

    # ---------------------------------------------------------------------
    # 3) Aggregation of ice crystals to snow (zsaut)
    # ---------------------------------------------------------------------
    zc1 = 17.5 / params.crhoi * air_density * (inverse_air_density ** 0.33)

    # zdt2 = -6/zc1 * log10(1e4*zris); then ztmp1 = ccsaut / zdt2
    zdt2 = (-6.0 / jnp.maximum(zc1, params.eps)) * jnp.log10(1.0e4 * jnp.maximum(zris, params.eps))
    ztmp1 = params.ccsaut / jnp.maximum(zdt2, params.eps)
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
            zxsp > params.cqtmin,
            jnp.logical_and(in_cloud_liquid > params.cqtmin, droplet_number >= minimum_droplet_number),
        ),
    )

    # droplet mean radius proxy (zdw). Double-where guard: the cube root has
    # an infinite derivative when in_cloud_liquid == 0, and everything
    # downstream of zdw (zcsacl) is where-masked on ll2 — safe base 1.0 in
    # the masked region keeps forward values unchanged and gradients finite.
    zdw_base = jnp.where(
        ll2,
        6.0 * params.pirho_rcp * air_density * in_cloud_liquid / jnp.maximum(droplet_number, params.eps),
        1.0,
    )
    zdw = zdw_base ** (1.0 / 3.0)
    zdw = jnp.maximum(zdw, 1.0e-6)

    zudrop = 1.19e4 * 2500.0 * zdw**2 * (1.3 * inverse_air_density_rcp) ** 0.35

    # planar snowflake max dimension (constant)
    zdplanar = 447.0e-6

    zusnow = 2.34 * jnp.maximum(100.0 * zdplanar, 1.0e-30) ** 0.3 * (1.3 * inverse_air_density_rcp) ** 0.35

    zstokes = 2.0 * c.rgrav * (zusnow - zudrop) * zudrop / zdplanar
    zstokes = jnp.maximum(zstokes, params.cqtmin)

    zrey = air_density * zdplanar * zusnow / jnp.maximum(dynamic_viscosity, params.eps)
    zrey = jnp.maximum(zrey, params.cqtmin)

    ll3 = zrey <= 5.0
    ll4 = jnp.logical_and(zrey > 5.0, zrey < 40.0)
    ll5 = zrey >= 40.0

    zstcrit = jnp.ones_like(zrey)
    zstcrit = jnp.where(ll3, 5.52 * zrey ** (-1.12), zstcrit)
    zstcrit = jnp.where(ll4, 1.53 * zrey ** (-0.325), zstcrit)

    zcsacl = 0.2 * (jnp.log10(zstokes) - jnp.log10(zstcrit) - 2.236) ** 2
    zcsacl = jnp.minimum(zcsacl, 1.0 - params.cqtmin)
    zcsacl = jnp.maximum(zcsacl, 0.0)
    zcsacl = jnp.sqrt(jnp.maximum(1.0 - zcsacl, 1.0e-30))

    ll6 = jnp.logical_and(ll5, zstokes <= 0.06)
    ll7 = jnp.logical_and(ll5, jnp.logical_and(zstokes > 0.06, zstokes <= 0.25))
    ll8 = jnp.logical_and(ll5, jnp.logical_and(zstokes > 0.25, zstokes <= 1.00))

    zcsacl = jnp.where(ll5, (zstokes + 1.1) ** 2 / (zstokes + 1.6) ** 2, zcsacl)
    zcsacl = jnp.where(ll6, 1.034 * zstokes ** 1.085, zcsacl)
    zcsacl = jnp.where(ll7, 0.787 * zstokes ** 0.988, zcsacl)
    zcsacl = jnp.where(ll8, 0.7475 * jnp.log10(zstokes) + 0.65, zcsacl)

    zcsacl = jnp.clip(zcsacl, 0.01, 1.0)
    zcsacl = jnp.where(ll2, zcsacl, 0.0)

    # lambda_snow proxy and collection coefficient. Double-where guard on
    # the fractional power (zxsp can be 0 where ll2 is False; ztmp2 is
    # where-masked, so the safe base only changes the masked region).
    zlamsm = params.cons4 * jnp.where(ll2, zxsp, 1.0) ** 0.8125
    ztmp2 = pi * params.cn0s * 3.078 * zlamsm * (inverse_air_density ** 0.5)
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
    ll2b = in_cloud_liquid > params.cqtmin
    psacln_raw = droplet_number * zsaclin / (pxlb_before + params.eps)
    psacln_raw = jnp.minimum(psacln_raw, droplet_number - minimum_droplet_number)
    psacln_raw = jnp.maximum(psacln_raw, 0.0)
    psacln = jnp.where(ll2b, psacln_raw, 0.0)

    # apply number loss to droplets where ll1 (as in Fortran MERGE)
    droplet_number = jnp.where(ll1, droplet_number - psacln, droplet_number)
    pmsnowacl = jnp.where(ll1, zsaclin, 0.0)

    # ---------------------------------------------------------------------
    # 5) Accretion of snow with ice crystals (zsaci)
    # ---------------------------------------------------------------------
    ll1b = jnp.logical_and(cloud_mask, in_cloud_ice > params.cqtmin)
    ll2 = jnp.logical_and(ll1b, zxsp > params.cqtmin)

    # Double-where guard on the fractional power (zsaci is where-masked on
    # ll2, so the safe base only changes the masked region).
    zlamsm = params.cons4 * jnp.where(ll2, zxsp, 1.0) ** 0.8125
    ztmp1 = pi * params.cn0s * 3.078 * zlamsm * (inverse_air_density ** 0.5)
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
        jnp.logical_and(in_cloud_ice > params.epsec, ice_number >= params.icemin),
    )

    zxibold_sec = jnp.maximum(zxibold2, 0.0)  # Fortran zxibold used here
    zsprn1 = ice_number * (zsaci + zsaut) / (zxibold_sec + params.eps)
    zself = 0.5 * dt * zc1 * ice_number * in_cloud_ice
    zsecprodn = params.mi0_rcp * air_density * zsecprod

    psprn_val = zsprn1 + zself - zsecprodn
    psprn_val = jnp.minimum(psprn_val, ice_number)
    psprn = jnp.where(ll_ice_num, psprn_val, 0.0)

    ice_number_new = jnp.maximum(ice_number - psprn, params.cqtmin)
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
    params: CloudParams2M,                  # threaded scheme parameters
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,  # updated inout: precip_cover, rain_flux, snow_flux, snow_melt
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray   # out: pfevapr, pfrain, pfsnow, pfsubls
]:
    """Update precipitation fluxes entering/leaving a layer.

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
    _, _, _, zcons2, _ = microphysics_dt_constants(dt, params)

    # Precipitation produced in this level (mass flux units [kg/m2/s])
    zzdrr = zcons2 * pressure_thickness * rain_formation
    zzdrs = zcons2 * pressure_thickness * (snow_formation + snow_accretion)

    # Fold the sedimenting ice flux into the snow flux. The caller gates
    # this with ECHAM's ``kk == klev`` condition (bottom level only) by
    # passing zero elsewhere — see the column sweep.
    zzdrs = zzdrs + jnp.maximum(ice_flux_from_above, 0.0)

    # 2) Top-level Melting of Incoming Ice into Rain
    # melting capacity (per area) limited by available energy
    melt_capacity = zcons2 * pressure_thickness / jnp.maximum(lsdcp - lvdcp, params.eps) * jnp.maximum(0.0, (temp_tmp - c.tmelt))
    # limit melting to a fraction xsec*zzdrs (same heuristic as Fortran)
    ztmp2 = jnp.minimum(params.xsec * zzdrs, melt_capacity)
    # apply melting where incoming ice exists and melting capacity>0
    melt_applied = jnp.where(ice_flux_from_above > 0.0, ztmp2, 0.0)
    zzdrr = zzdrr + melt_applied
    zzdrs = zzdrs - melt_applied
    # psmlt accumulates melting mass in kg/kg units (Fortran: psmlt += ztmp2/(zcons2*pdp))
    snow_melt = snow_melt + melt_applied / jnp.maximum(zcons2 * pressure_thickness, params.eps)

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
        params=params,
    )

    # 4) In-cloud Rain/Snow Fluxes and Area-integrated Evaporation/Sublimation
    # in-cloud (area-averaged) rain/snow fluxes before evaporation/sublimation
    ll1 = precip_cover > params.epsec

    ztmp1 = (rain_flux + zzdrr) / jnp.maximum(precip_cover, params.epsec)
    ztmp2 = (snow_flux + zzdrs) / jnp.maximum(precip_cover, params.epsec)

    pfrain = jnp.where(ll1, ztmp1, 0.0)
    pfsnow = jnp.where(ll1, ztmp2, 0.0)

    # evaporation / sublimation area-integrated (kg/m2/s)
    ztmp3 = (zcons2 * pressure_thickness * rain_evap_mmr) / jnp.maximum(precip_cover, params.epsec)
    ztmp4 = (zcons2 * pressure_thickness * snow_sublimation_mmr) / jnp.maximum(precip_cover, params.epsec)

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

