"""Tendency assembly and diagnostics for the Lohmann 2M scheme.

Final tendency/state assembly (``update_tendencies_and_important_vars``),
in-cloud water bookkeeping, and diagnostic outputs. Split out of the
monolithic ``lohmann_2m.py`` module (pure move, no numerical change).
"""

import jax.numpy as jnp
from math import pi

import jcm.constants as c

from ..lohmann_2m_params import CloudParams2M
from ..cloud_utils import (
    eff_ice_crystal_radius,
    eff_liquid_droplet_radius,
    minimum_CDNC,
)
from .types import microphysics_dt_constants


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
    params: CloudParams2M,                   # threaded scheme parameters
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Update tendencies and compute in-cloud effective radii.

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
    - Timestep constants are obtained via microphysics_dt_constants(dt, params).
    - Correction thresholds use module constants (ccwmin, clc_min, eps, etc.).
    - Breadth and ice-radius helpers (breadth_factor, eff_ice_crystal_radius)
      are used to compute effective radii. Cirrus branch applied when nic_cirrus==1.

    """
    # timestep constants
    ztmst, ztmst_rcp, _, _, _ = microphysics_dt_constants(dt, params)

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
    ll_liq_neg = liq_mmr_next < params.ccwmin
    zdxlcor = jnp.where(ll_liq_neg, -ztmst_rcp * liq_mmr_next, 0.0)
    liq_tendency = liq_tendency + zdxlcor

    # adjust tracer tendency for cdnc where negative-correction applied
    tracer_tendency_cdnc = jnp.where(
        ll_liq_neg,
        tracer_tendency_cdnc - ztmst_rcp * cdnc * inv_air_density,
        tracer_tendency_cdnc,
    )

    # ice
    ll_ice_neg = ice_mmr_next < params.ccwmin
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
    ll_small_clc = cloud_fraction < params.clc_min
    cloud_fraction = jnp.where(ll_small_clc, 0.0, cloud_fraction)

    # zero tiny in-cloud accumulators (Fortran used 1e-20 checks)
    pmlwc_flag = jnp.logical_or(ll_small_clc, incloud_liq_before_rain < 1e-20)
    incloud_liq_before_rain = jnp.where(pmlwc_flag, 0.0, incloud_liq_before_rain)

    pmiwc_flag = jnp.logical_or(ll_small_clc, incloud_ice_before_snow < 1e-20)
    incloud_ice_before_snow = jnp.where(pmiwc_flag, 0.0, incloud_ice_before_snow)

    # adjust tendencies by removing the correction contributions
    specific_humidity_tendency = specific_humidity_tendency - zdxlcor - zdxicor
    temp_tendency = temp_tendency + lvdcp * zdxlcor + lsdcp * zdxicor

    # --- 6) effective liquid droplet radius [um] (preffl); the ECHAM law and its
    # cube-root gradient guard live in the shared helper the 1M scheme also uses.
    liq_eff_radius = eff_liquid_droplet_radius(
        cloud_liquid_in_cloud, air_density, cdnc, params.eps,
        liquid_cloud_flag=liquid_cloud_flag,
    )

    # --- 7) ice crystal effective radius [um] (preffi)
    # convert in-cloud ice kg/kg -> g/m^3: 1000 * pxib * prho
    ice_gm3 = 1000.0 * cloud_ice_in_cloud * air_density
    ice_eff_rad = eff_ice_crystal_radius(ice_gm3, icnc, params)  # returns microns (as in module helpers)

    # cirrus correction branch as in Fortran when nic_cirrus==1 and cold
    if params.nic_cirrus == 1:
        is_cold = temp_tmp < params.cthomi
        ztmp2 = 83.8 * (1e3 * jnp.maximum(cloud_ice_in_cloud, params.eps) * air_density) ** 0.216
        ice_eff_rad = jnp.where(is_cold, ztmp2, ice_eff_rad)

    # clip bounds
    ice_eff_rad = jnp.maximum(ice_eff_rad, params.ceffmin)
    ice_eff_rad = jnp.minimum(ice_eff_rad, params.ceffmax)
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
    ice_radius_mean: jnp.ndarray,        # Original: prid — volume-mean, METRES
    temp_prev: jnp.ndarray,              # Original: ptm1
    cloud_flag: jnp.ndarray,             # Original: ld_cc (INOUT)
    ice_crystal_number: jnp.ndarray,     # Original: picnc (INOUT)
    nucleation_rate: jnp.ndarray,        # Original: pqnuc (INOUT)
    droplet_number: jnp.ndarray,         # Original: pcdnc (INOUT)
    cloud_fraction: jnp.ndarray,         # Original: paclc (INOUT)
    cloud_ice_in_cloud: jnp.ndarray,     # Original: pxib (INOUT)
    cloud_liquid_in_cloud: jnp.ndarray,  # Original: pxlb (INOUT)
    dt: jnp.ndarray,                      # Microphysics timestep (used for pqnuc accumulation)
    params: CloudParams2M,                # threaded scheme parameters
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Update in-cloud water/ice, CDNC/ICNC activation/nucleation and cloud cover.

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
    # numeric guards and constants (eps, clc_min, cqtmin, icemin, rhoice) come from the threaded params struct
    # 1) relative humidity
    relhum = specific_humidity_tmp / jnp.maximum(sat_spec_humidity_tmp, params.eps)

    # positive deposition / condensation sources (limit negative contributions)
    src_dep = jnp.maximum(deposition_rate + tompkins_genti, 0.0)
    src_cnd = jnp.maximum(condensation_rate + tompkins_gentl, 0.0)

    # 2) update in-cloud ice/liquid where cloud already exists:
    # pxib_new = pxib + pdep / max(paclc, clc_min)
    pxib_candidate = cloud_ice_in_cloud + src_dep / jnp.maximum(cloud_fraction, params.clc_min)
    pxib_candidate = jnp.maximum(pxib_candidate, 0.0)
    cloud_ice_in_cloud = jnp.where(cloud_flag, pxib_candidate, cloud_ice_in_cloud)

    pxlb_candidate = cloud_liquid_in_cloud + src_cnd / jnp.maximum(cloud_fraction, params.clc_min)
    pxlb_candidate = jnp.maximum(pxlb_candidate, 0.0)
    cloud_liquid_in_cloud = jnp.where(cloud_flag, pxlb_candidate, cloud_liquid_in_cloud)

    # 3) if no cloud but there are positive sources, set cloud fraction from relhum and
    #    set in-cloud values to source-per-cloud-area
    make_cloud_mask = jnp.logical_and(~cloud_flag, jnp.logical_or(src_dep > 0.0, src_cnd > 0.0))

    paclc_from_rh = jnp.clip(relhum, 0.01, 1.0)
    cloud_fraction = jnp.where(make_cloud_mask, paclc_from_rh, cloud_fraction)

    pxib_from_src = src_dep / jnp.maximum(cloud_fraction, params.clc_min)
    pxib_from_src = jnp.maximum(pxib_from_src, 0.0)
    cloud_ice_in_cloud = jnp.where(make_cloud_mask, pxib_from_src, cloud_ice_in_cloud)

    pxlb_from_src = src_cnd / jnp.maximum(cloud_fraction, params.clc_min)
    pxlb_from_src = jnp.maximum(pxlb_from_src, 0.0)
    cloud_liquid_in_cloud = jnp.where(make_cloud_mask, pxlb_from_src, cloud_liquid_in_cloud)

    # 4) compute minimum CDNC from in-cloud liquid mass density (kg/kg * rho -> kg/m^3)
    liquid_mass_density = cloud_liquid_in_cloud * air_density  # [kg/m^3]
    pcdnc_min = minimum_CDNC(liquid_mass_density, params)

    # 5) redefine cloud flag
    cloud_flag = cloud_fraction > 0.0

    # 6) activation / nucleation: only where cloud exists and liquid > cqtmin
    ll1 = jnp.logical_and(cloud_flag, cloud_liquid_in_cloud > params.cqtmin)
    ll2 = jnp.logical_and(ll1, jnp.logical_and(droplet_number <= pcdnc_min, temp_prev > params.cthomi))

    # desired additional droplets. A smooth (hyperbolic) maximum when
    # ``params.activation_smoothing > 0``: ``0.5*(d + sqrt(d^2 + w^2))``
    # overshoots ``max(d, 0)`` by at most ``w/2`` at the corner and is exact
    # away from it. The hard max makes the loss piecewise in the aerosol
    # activation parameters (the SPA prefactor/exponent enter only through
    # ``activated_cdnc``): a cell whose carried CDNC crosses the floor flips
    # branches and the exact local gradient stops tracking the large-scale
    # Twomey response. Double-where on the width so ``0`` recovers the hard
    # max exactly without the unselected branch differentiating ``sqrt`` at
    # zero (its derivative there is infinite).
    act_gap = activated_cdnc - droplet_number
    act_smoothing_on = params.activation_smoothing > 0.0
    act_w_safe = jnp.where(act_smoothing_on, params.activation_smoothing, 1.0)
    delta_cdnc_smooth = 0.5 * (
        act_gap + jnp.sqrt(act_gap * act_gap + act_w_safe * act_w_safe)
    )
    delta_cdnc = jnp.where(
        act_smoothing_on, delta_cdnc_smooth, jnp.maximum(act_gap, 0.0)
    )

    # only count activation where ll2
    delta_cdnc_applied = jnp.where(ll2, delta_cdnc, 0.0)

    # update droplet number and nucleation-rate diagnostic (pqnuc += dt * delta)
    droplet_number = droplet_number + delta_cdnc_applied
    nucleation_rate = nucleation_rate + dt * delta_cdnc_applied

    # 7) enforce minimum CDNC or set to cqtmin where no meaningful cloud (Fortran MERGE semantics)
    # ztmp1 = max(pcdnc, pcdnc_min)
    tmp_cdnc_max = jnp.maximum(droplet_number, pcdnc_min)
    # Fortran: pcdnc = MERGE( ztmp1, cqtmin, ll1 ) -> if ll1 True -> tmp_cdnc_max else -> cqtmin
    droplet_number = jnp.where(ll1, tmp_cdnc_max, params.cqtmin)

    # 8) update ICNC similarly
    ll1_ic = jnp.logical_and(cloud_flag, cloud_ice_in_cloud > params.cqtmin)
    ll2_ic = jnp.logical_and(ll1_ic, ice_crystal_number <= params.icemin)

    # compute candidate ICNC depending on nic_cirrus
    if params.nic_cirrus == 1:
        # N = rho*q_i / ((4/3)*pi*prid^3*rho_ice): crystal number from ice mass
        # and the volume-mean radius ``prid``, which is in METRES.
        # The floor must be a pure divide-by-zero guard: a realistic prid^3 is
        # ~1e-13 m^3, so ``eps`` (~1e-7) would clamp every cell and force the
        # candidate to zero. ``d_epsilon`` (1e-30) sits below any physical value.
        icnc_candidate = 0.75 / (pi * params.rhoice) * air_density * cloud_ice_in_cloud / jnp.maximum(ice_radius_mean**3, params.d_epsilon)
    elif params.nic_cirrus == 2:
        # min(pnicex, pap*1e6)
        icnc_candidate = jnp.minimum(newly_formed_ice, pressure * 1.0e6)
    else:
        # default: leave unchanged candidate (set to existing to be MERGE-safe)
        icnc_candidate = ice_crystal_number

    ice_crystal_number = jnp.where(ll2_ic, icnc_candidate, ice_crystal_number)

    # enforce minimum icnc or set to cqtmin where no cloud-ice
    tmp_icnc_max = jnp.maximum(ice_crystal_number, params.icemin)
    ice_crystal_number = jnp.where(ll1_ic, tmp_icnc_max, params.cqtmin)

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
    params: CloudParams2M,                # threaded scheme parameters
) -> tuple:
    """Diagnostics accumulator updates.

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
    ll1 = jnp.logical_and.reduce(
    jnp.stack(
        (
            liquid_cloud_flag,
            (ktop == level_index),
            (temp_tmp > c.tmelt),
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
    ztmp4 = tau1i + 1.9787 * ztmp3 * jnp.maximum(eff_radius_ice, params.ceffmin) ** (-1.0365)
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

# ---------------------------------------------------------------------------
# DeMott (2010) INP parameterization
# ---------------------------------------------------------------------------


