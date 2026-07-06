"""Mixed-phase deposition, WBF, and freezing for the Lohmann 2M scheme.

Mixed-phase vapor deposition and corrections, homogeneous freezing below
238 K, heterogeneous mixed-phase freezing, the Wegener-Bergeron-Findeisen
process, and the DeMott (2010) ice-nucleating-particle diagnostic. Split
out of the monolithic ``lohmann_2m.py`` module (pure move, no numerical
change).
"""

import jax.numpy as jnp
from math import pi

import jcm.constants as c

from ..lohmann_2m_params import CloudParams2M
from ..cloud_utils import (
    eff_ice_crystal_radius,
    threshold_vert_vel,
)
from .types import microphysics_dt_constants


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
    params: CloudParams2M,               # threaded scheme parameters
    ll_het: bool = True,                 # heterogeneous nucleation flag (module-level in Fortran)
) -> tuple[
    jnp.ndarray,  # condensation_rate (updated pcnd) [kg/kg]
    jnp.ndarray,  # deposition_rate (updated pdep) [kg/kg]
    jnp.ndarray,  # temperature_tmp (ptp1tmp) [K]
    jnp.ndarray,  # specific_humidity_tmp (pqp1tmp) [kg/kg]
    jnp.ndarray,  # qsat_tmp (pqsp1tmp) [kg/kg]
]:
    """Mixed-phase deposition and condensation corrections for the ICON/ECHAM 2-moment scheme.

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
    ice_gm3 = 1000.0 * zxip1 * air_density / jnp.maximum(cloud_fraction, params.clc_min)
    zrieff = eff_ice_crystal_radius(ice_gm3, icnc, params)   # [µm]
    zrieff = jnp.clip(zrieff, params.ceffmin, params.ceffmax)

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
        params=params,
    )

    # -------------------------------------------------------------------------
    # 5. Phase mask lo2:  True = ice cloud,  False = liquid cloud
    #    lo2 = (T_tmp < cthomi) OR (T_tmp < tmelt AND 0.01*pvervx < zvervmax)
    # -------------------------------------------------------------------------
    lo2 = jnp.logical_or(
        temperature_tmp < params.cthomi,
        jnp.logical_and(
            temperature_tmp < c.tmelt,
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
    ztmp_ice = (c.alhs/c.rv)*(1.0/c.tmelt - 1.0/temperature_tmp)
    ztmp_water = (c.alhc/c.rv)*(1.0/c.tmelt - 1.0/temperature_tmp)
    zes_ice_new = 611 * jnp.exp(ztmp_ice)
    zes_water_new = 611 * jnp.exp(ztmp_water)

    # Select phase-appropriate saturation vapour pressure
    zes = jnp.where(lo2, zes_ice_new, zes_water_new)
    zesw = zes_water_new

    # Saturation specific humidity (standard formula)
    # q_s = zes / (p - (1 - Rd/Rv)*zes)  — same form as ECHAM sat_spec_hum
    def _qsat(e, p):
        # q_s = eps·e_s / (p − (1−eps)·e_s). The leading eps was missing
        # (qsat 1.61× high — review finding 2.20); note 1/(1+vtmpc1) ≡ eps.
        e_clipped = jnp.minimum(e, 0.4 * p)   # safety clip (Fortran: zes < 0.4)
        return c.eps * e_clipped / (p - (1.0 - c.eps) * e_clipped)

    qsat_tmp = _qsat(zes, pressure)          # pqsp1tmp: phase-appropriate
    qsat_tmp_water = _qsat(zesw, pressure)   # zqsp1tmpw: always over water

    # zcor: correction factor d(q_s)/d(e_s) * p / (p - e_s)^2  (used in zlcdqsdt)
    # In ECHAM: zcor = 1 / (1 - vtmpc1 * q_s)
    zcor = 1.0 / jnp.maximum(1.0 - c.vtmpc1 * qsat_tmp, params.eps)
    zcorw = 1.0 / jnp.maximum(1.0 - c.vtmpc1 * qsat_tmp_water, params.eps)  # noqa: F841 — used in Phase 5b

    # -------------------------------------------------------------------------
    # 7. Saturation specific humidity at (t+1) for zdqsdt
    #    In Fortran: zqst1 uses tlucuap1 (lookup at it+1), approximated here
    #    by evaluating at (T_tmp + 1 K) and taking finite difference.
    # -------------------------------------------------------------------------
    # ECHAM mo_convect_tables Tetens pairs: ice c3ies=21.875/c4ies=7.66,
    # water c3les=17.269/c4les=35.86, prefactor c2es = c1es·(Rd/Rv) =
    # 610.78·eps — the lookup table tlucua stores eps·e_s, which is why the
    # /pressure form below carries no explicit eps. The previous code used
    # c.ak (the BOLTZMANN constant, 1.38e-23) as the exponent coefficient
    # and c.p0s1_bg (101325 Pa!) as the prefactor: exp(~0)·101325/p pinned
    # zqst1 at its 0.5 cap, made zdqsdt ~ +490 and the zqcon
    # thermodynamic factor ~1e-6 — suppressing internal condensation/
    # deposition by six orders of magnitude (review finding 1.2).
    ztmp_ice_p1 = jnp.minimum(21.875 * (temperature_tmp + 1.0 - c.tmelt) / jnp.maximum(temperature_tmp + 1.0 - 7.66, params.eps), 700.0)
    ztmp_water_p1 = jnp.minimum(17.269 * (temperature_tmp + 1.0 - c.tmelt) / jnp.maximum(temperature_tmp + 1.0 - 35.86, params.eps), 700.0)

    c2es = 610.78 * c.eps
    zes_p1 = jnp.where(lo2, c2es * jnp.exp(ztmp_ice_p1), c2es * jnp.exp(ztmp_water_p1))
    zqst1 = zes_p1 / pressure
    zqst1 = jnp.minimum(zqst1, 0.5)
    zqst1 = zqst1 / (1.0 - c.vtmpc1 * zqst1)

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
        c.alhs / (c.rv * jnp.maximum(temperature_tmp**2, params.eps)),  # ice
        c.alhc / (c.rv * jnp.maximum(temperature_tmp**2, params.eps)),  # water
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
    zrhtest = jnp.minimum(specific_humidity_prev / jnp.maximum(qsat_prev, params.eps), 1.0) * qsat_tmp

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
    ll1_circ = jnp.array(params.nic_cirrus == 1)  # constant (not per-point)

    ll2 = specific_humidity_tmp > (qsat_tmp + zoversat)
    ll3 = specific_humidity_tmp > (qsat_tmp_water + zoversatw)
    ll4 = specific_humidity_tmp > zqsp1tmphet
    ll5 = temperature_tmp >= params.cthomi  # True = mixed-phase (not homogeneous)

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
    # 12b. Koop homogeneous-freezing floor (interim toward #552).
    # Below cthomi, vapor above the Koop et al. (2000) homogeneous
    # nucleation threshold S_crit(T) = 2.349 − T/259 CANNOT persist —
    # solution droplets freeze explosively on a timescale of seconds.
    # The full Kaercher-Lohmann scheme (#552) resolves the competition
    # for that vapor; until it lands, the excess above S_crit deposits
    # within the step. Without this floor, cells in the (~20 K too cold)
    # winter stratosphere accumulate S_ice well beyond 2 faster than
    # ICNC-limited depositional growth can consume it, and the latent-
    # heat spike when the state finally collapses NaN'd the coupled
    # T63L47 runs three times (days 30/90/110). Rides the deposition
    # ledger, so water/enthalpy bookkeeping is exact by construction.
    scrit_koop = 2.349 - temperature_tmp / 259.0
    koop_excess = jnp.where(
        jnp.logical_and(lo2, temperature_tmp < params.cthomi),
        jnp.maximum(specific_humidity_tmp - scrit_koop * qsat_tmp, 0.0),
        0.0,
    )
    deposition_rate = deposition_rate + koop_excess

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
        zvervmax,
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
    """Freezing process below 238K for cloud microphysics.

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
    min_liquid_threshold: float,      # Original: cqtmin
    params: CloudParams2M,            # threaded scheme parameters
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Heterogeneous mixed-phase freezing for cloud microphysics.

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
    ztmp1 = 1.0 + 1.26 * 6.6e-8 / (wet_radius_aitken + 1e-12) * (c.p0s1_bg / pressure) * (temperature / c.tmelt)
    ztmp2 = 1.0 + 1.26 * 6.6e-8 / (wet_radius_accumulation + 1e-12) * (c.p0s1_bg / pressure) * (temperature / c.tmelt)
    ztmp3 = 1.0 + 1.26 * 6.6e-8 / (wet_radius_coarse + 1e-12) * (c.p0s1_bg / pressure) * (temperature / c.tmelt)

    zeta_air = 1e-5 * (1.718 + 0.0049 * (temperature - c.tmelt) - 1.2e-5 * (temperature - c.tmelt) ** 2)

    aerosol_diffusivity_bc = c.ak * temperature * ztmp1 / (6.0 * pi * zeta_air * (wet_radius_aitken + 1e-12))
    aerosol_diffusivity_bc = jnp.where(wet_radius_aitken < 1e-12, 0.0, aerosol_diffusivity_bc)

    aerosol_diffusivity_dust_accum = c.ak * temperature * ztmp2 / (6.0 * pi * zeta_air * (wet_radius_accumulation + 1e-12))
    aerosol_diffusivity_dust_accum = jnp.where(wet_radius_accumulation < 1e-12, 0.0, aerosol_diffusivity_dust_accum)

    aerosol_diffusivity_dust_coarse = c.ak * temperature * ztmp3 / (6.0 * pi * zeta_air * (wet_radius_coarse + 1e-12))
    aerosol_diffusivity_dust_coarse = jnp.where(wet_radius_coarse < 1e-12, 0.0, aerosol_diffusivity_dust_coarse)

    # -------------------------------------------------------------------------
    # 2. Freezing rates (contact and immersion freezing)
    # -------------------------------------------------------------------------
    # Compute mean volume radius of cloud droplets. Double-where guard on
    # the cube root: it has an infinite derivative at cloud_liquid == 0 and
    # the freezing rates below vanish there (and are additionally
    # where-masked on freezing_condition), so without a safe base the
    # backward pass multiplies 0 × ∞ = NaN at liquid-free points. Forward
    # values are unchanged (radius was 0 there, and every consumer scales
    # by cloud_liquid).
    has_liquid = cloud_liquid > 0.0
    droplet_radius_base = jnp.where(
        has_liquid,
        0.75 * cloud_liquid * air_density / (pi * c.rhow * droplet_number),
        1.0,
    )
    droplet_radius = jnp.where(has_liquid, droplet_radius_base ** (1.0 / 3.0), 0.0)

    # Contact freezing by dust and soot
    contact_freezing_dust = jnp.minimum(1.0, jnp.maximum(0.0, -(0.1014 * (temperature - c.tmelt) + 0.3277)))
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
        (immersion_freezing_dust + immersion_freezing_bc) * air_density / c.rhow
        * jnp.exp(c.tmelt - temperature) * jnp.minimum(vertical_velocity - params.fact_tke * jnp.sqrt(tke) * air_density * c.grav, 0.0)
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
    dt: jnp.ndarray,                       # Microphysics timestep (used to form ztmst_rcp)
    params: CloudParams2M,                 # threaded scheme parameters
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Warm-bridge/freeze (WBF) process: transfer of in-cloud liquid to ice under WBF conditions.

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
    - ztmst_rcp (Fortran ztmst_rcp) is obtained from microphysics_dt_constants(dt, params).
    - All operations are vectorised and preserve input shapes; values are only changed
      where wbf_mask is True.
    - cqtmin is used as the minimum CDNC (Fortran constant cqtmin).

    """
    # get reciprocal timestep constant (ztmst_rcp = 1 / ztmst)
    _, ztmst_rcp, *_ = microphysics_dt_constants(dt, params)

    # ztmp1 = ztmst_rcp * pxlb * paclc  (evap / WBF proxy)
    ztmp1 = ztmst_rcp * cloud_liquid_in_cloud * cloud_fraction

    # cloud liquid tendency: pxlte <- MERGE(pxlte - ztmp1, pxlte, ld_WBF)
    cloud_liquid_tendency = jnp.where(wbf_mask, cloud_liquid_tendency - ztmp1, cloud_liquid_tendency)

    # cloud ice tendency: pxite <- MERGE(pxite + ztmp1, pxite, ld_WBF)
    cloud_ice_tendency = jnp.where(wbf_mask, cloud_ice_tendency + ztmp1, cloud_ice_tendency)

    # temperature tendency: ptte <- MERGE(ptte + (plsdcp - plvdcp)*ztmp1, ptte, ld_WBF)
    temp_tendency = jnp.where(wbf_mask, temp_tendency + (lsdcp - lvdcp) * ztmp1, temp_tendency)

    # cdnc <- MERGE(cqtmin, pcdnc, ld_WBF)  (set to minimum where WBF occurs)
    cdnc = jnp.where(wbf_mask, params.cqtmin, cdnc)

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


def demott2010_inp(
    temperature: jnp.ndarray,
    n_aer_coarse_cm3: float,
) -> jnp.ndarray:
    """Ice nucleating particle concentration via DeMott et al. (2010).

    Returns INP concentration in 1/m³ for the mixed-phase temperature range
    (−9 °C to −35 °C, i.e. 264 K to 238 K). Outside this range, returns 0.

    Args:
        temperature: Temperature [K].
        n_aer_coarse_cm3: Total aerosol number > 0.5 μm diameter [cm⁻³ STP].

    Reference:
        DeMott et al. (2010), PNAS, doi:10.1073/pnas.0910818107

    """
    a, b, c, d = 5.94e-5, 3.33, 0.0264, 0.0033
    delta_T = 273.16 - temperature
    delta_T_clipped = jnp.clip(delta_T, 0.0, 35.0)
    n_aer_safe = jnp.maximum(n_aer_coarse_cm3, 0.01)

    # n_INP in std L⁻¹ → convert to m⁻³ (* 1e3)
    n_inp_per_litre = a * delta_T_clipped ** b * n_aer_safe ** (c * delta_T_clipped + d)
    n_inp_per_m3 = n_inp_per_litre * 1e3

    # Only active in the valid range (238 K to 264 K)
    active = (temperature <= 264.0) & (temperature >= 238.0)
    return jnp.where(active, n_inp_per_m3, 0.0)

