"""Ice sedimentation and snow/ice melting for the Lohmann 2M scheme.

Split out of the monolithic ``lohmann_2m.py`` module (pure move, no
numerical change).
"""

import jax.numpy as jnp

import jcm.constants as c

from ..lohmann_2m_params import CloudParams2M
from ..cloud_utils import (
    consistency_number_to_mass,
    gridbox_frac_falling_hydrometeor,
)
from .types import microphysics_dt_constants


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
    params: CloudParams2M,
) -> tuple:
    """Simulate the melting of snow and ice in a cloud microphysics model. This function is a JAX implementation
    of the ECHAM6 `melting_snow_and_ice` routine. It calculates the energy-limited melting capacity based on 
    temperature differences, melts snow flux into rain flux, melts ice-crystal flux into rain water, and handles 
    in-cloud ice melting when the temperature exceeds the melting point.

    The function updates various input arrays in-place and returns updated values for cloud microphysics variables.

    Parameters
    ----------
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

    Returns
    -------
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
    ztmst, ztmst_rcp, _, zcons2, _ = microphysics_dt_constants(dt, params)
    
    # ------------------------------------------------------------
    # 1) Energy-limited melt capacity (per layer) from T - tmelt
    # ------------------------------------------------------------
    ztdif = jnp.maximum(0.0, temperature_previous - c.tmelt)
    melt_capacity = (
        zcons2
        * ztdif
        * pressure_thickness
        / jnp.maximum(lsdcp - lvdcp, params.eps)
    )

    # ------------------------------------------------------------
    # 2) Melt snow flux -> rain flux
    # ------------------------------------------------------------
    snow_melt_flux = jnp.minimum(params.xsec * snow_flux, melt_capacity)  # ztmp2
    rain_flux = rain_flux + snow_melt_flux
    snow_flux = snow_flux - snow_melt_flux

    # Diagnostic melting in mmr units (as in Fortran): psmlt = dt*grav*melt_flux / pdp
    psmlt = ztmst * c.grav * snow_melt_flux / jnp.maximum(pressure_thickness, params.eps)

    # ------------------------------------------------------------
    # 3) Melt ice-crystal mass flux from above -> (implicitly) rain water
    # ------------------------------------------------------------
    ice_melt_flux = jnp.minimum(params.xsec * ice_flux, melt_capacity)

    has_ice_flux = ice_flux > params.epsec
    ice_melt_flux_n = jnp.where(
        has_ice_flux,
        ice_flux_n * ice_melt_flux / jnp.maximum(ice_flux, params.epsec),
        0.0,
    )

    ice_flux = ice_flux - ice_melt_flux
    ice_flux_n = ice_flux_n - ice_melt_flux_n

    # Keep number flux consistent with remaining mass flux
    # Expect this helper to exist in the module (or be imported).
    ice_flux_n = consistency_number_to_mass(pthreshold=params.epsec, pmass=ice_flux, pnumber=ice_flux_n)

    pximlt = ztmst * c.grav * ice_melt_flux / jnp.maximum(pressure_thickness, params.eps)

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
    icnc = jnp.where(melt_mask, params.icemin, icnc)
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
    params: CloudParams2M,
) -> tuple[
    jnp.ndarray,  # ice_mmr_gridmean (updated) [kg/kg]
    jnp.ndarray,  # icnc_in_cloud (updated) [1/m^3]
    jnp.ndarray,  # ice_flux (updated) [kg/m^2/s]
    jnp.ndarray,  # ice_flux_n (updated) [1/m^2/s]
    jnp.ndarray,  # falling_ice_fraction (updated) [0..1]
    jnp.ndarray,  # ice_sedimentation_rate_in_cloud (pmrateps) [kg/kg]
]:
    """Sedimentation of cloud ice (mass + number) and update of falling-ice fluxes (Lin et al. (1983)).

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
    ztmst, _, _, zcons2, _ = microphysics_dt_constants(dt, params)

    # --- Keep a copy of grid-mean ice before sedimentation
    zxi_bf_sed = ice_mmr_gridmean

    # --- Convert ICNC to grid-mean and enforce minimum
    zicnc_gridmean = icnc_in_cloud * cloud_fraction
    zicnc_gridmean = jnp.maximum(zicnc_gridmean, params.icemin)
    zicnc_gridmean_bf_sed = zicnc_gridmean

    # --- Mean mass per crystal proxy
    zmmean = air_density * ice_mmr_gridmean / jnp.maximum(zicnc_gridmean, params.eps)
    zmmean = jnp.maximum(zmmean, params.mi)

    # --- Regime selection for sedimentation parameters
    ll_small = zmmean < params.ri_vol_mean_1
    ll_mid = jnp.logical_and(~ll_small, zmmean < params.ri_vol_mean_2)

    zalfased = jnp.where(ll_small, params.alfased_1, params.alfased_2)
    zalfased = jnp.where(ll_mid, params.alfased_3, zalfased)

    zbetased = jnp.where(ll_small, params.betased_1, params.betased_2)
    zbetased = jnp.where(ll_mid, params.betased_3, zbetased)

    # --- Fall speed (mass and number use same here), limited as in Fortran
    zxifallmc = params.fall * zalfased * (zmmean ** zbetased) * air_density_correction
    zxifallmc = jnp.clip(zxifallmc, 0.001, 2.0)
    zxifallnc = zxifallmc

    # --- Exponential coefficients
    zal1 = ztmst * c.grav * zxifallmc * air_density / jnp.maximum(pressure_thickness, params.eps)
    zal3 = c.grav * ztmst * zxifallnc * air_density / jnp.maximum(pressure_thickness, params.eps)

    # --- Incoming-flux "equilibria" (MERGE to 0 if fall speed is too small)
    ll_mass = zxifallmc > params.eps
    zal2_raw = ice_flux * inv_air_density_rcp / jnp.maximum(zxifallmc, params.eps)
    zal2 = jnp.where(ll_mass, zal2_raw, 0.0)

    ll_num = zxifallnc > params.eps
    zal4_raw = ice_flux_n / jnp.maximum(zxifallnc, params.eps)
    zal4 = jnp.where(ll_num, zal4_raw, 0.0)

    # --- Update grid-mean ice mmr and grid-mean ICNC via relaxation form
    exp1 = jnp.exp(-zal1)
    exp3 = jnp.exp(-zal3)

    ice_mmr_gridmean = ice_mmr_gridmean * exp1 + zal2 * (1.0 - exp1)
    zicnc_gridmean = zicnc_gridmean * exp3 + zal4 * (1.0 - exp3)

    # --- Convert back to in-cloud ICNC where cloud fraction is meaningful
    has_cloud = cloud_fraction > params.clc_min
    icnc_in_cloud_candidate = zicnc_gridmean / jnp.maximum(cloud_fraction, params.clc_min)
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
    pmrateps_in_cloud = zxi_delta / jnp.maximum(cloud_fraction, params.clc_min)
    ice_sedimentation_rate_in_cloud = jnp.where(has_cloud, pmrateps_in_cloud, zxi_delta)
    ice_sedimentation_rate_in_cloud = jnp.maximum(ice_sedimentation_rate_in_cloud, 0.0)

    # --- Update fraction covered by falling ice
    # Only update if there is a positive flux contribution from this level.
    falling_ice_fraction = gridbox_frac_falling_hydrometeor(
        precip_flux_from_above=ice_flux,
        precip_frac_from_above=falling_ice_fraction,
        precip_flux_from_level=jnp.maximum(zxiflx_from_level, 0.0),  # only positive contribution
        precip_frac_from_level=cloud_fraction,
        params=params,
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
    ice_flux_n = consistency_number_to_mass(pthreshold=params.epsec, pmass=ice_flux, pnumber=ice_flux_n)

    return (
        ice_mmr_gridmean,
        icnc_in_cloud,
        ice_flux,
        ice_flux_n,
        falling_ice_fraction,
        ice_sedimentation_rate_in_cloud,
    )

