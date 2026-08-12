"""Lohmann 2M column-sweep orchestrator and composable physics term.

``cloud_microphysics_2m`` runs the full two-moment process chain over a
column (lax.scan sweep), and ``Lohmann2MMicrophysics`` wraps it as a
composable ``PhysicsTerm``. Split out of the monolithic ``lohmann_2m.py``
module (pure move, no numerical change).
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from flax import nnx

import jcm.constants as c
from jcm.forcing import ForcingData
from jcm.physics import thermodynamics
from jcm.physics.aerosol.spa import spa_activated_cdnc
from jcm.physics.diagnostics.moist_air_state import advance_thermo_run
from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

from ..lohmann_2m_params import CloudParams2M
from ..cloud_utils import (
    eff_ice_crystal_radius,
    minimum_CDNC,
)
from .types import MicrophysicsTendencies_2M
from .sedimentation_melt import melting_snow_and_ice, sedimentation_ice
from .deposition_freezing import (
    demott2010_inp,
    freezing_below_238K,
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
    update_in_cloud_water,
    update_tendencies_and_important_vars,
)


# ---------------------------------------------------------------------------
# Column-sweep orchestrator
# ---------------------------------------------------------------------------


def cloud_microphysics_2m(
    temperature: jnp.ndarray,       # (nlev,)  K
    specific_humidity: jnp.ndarray, # (nlev,)  kg/kg
    pressure: jnp.ndarray,          # (nlev,)  Pa
    qc: jnp.ndarray,                # (nlev,)  kg/kg cloud liquid mass mixing ratio
    qi: jnp.ndarray,                # (nlev,)  kg/kg cloud ice mass mixing ratio
    qnc: jnp.ndarray,               # (nlev,)  kg^-1 cloud droplet number per kg of air
    qni: jnp.ndarray,               # (nlev,)  kg^-1 ice crystal number per kg of air
    qr: jnp.ndarray,                # (nlev,)  kg/kg rain mixing ratio (from prev step)
    qs: jnp.ndarray,                # (nlev,)  kg/kg snow mixing ratio (from prev step)
    cloud_fraction: jnp.ndarray,    # (nlev,)  [0,1]
    air_density: jnp.ndarray,       # (nlev,)  kg/m^3
    layer_thickness: jnp.ndarray,   # (nlev,)  m   (dz, full-level layer depths)
    tke: jnp.ndarray,               # (nlev,)  m²/s²  turbulent kinetic energy
    activated_cdnc: jnp.ndarray,    # (nlev,)  1/m³   aerosol-activated CDNC (from MACv2-SP)
    ice_nuclei: jnp.ndarray,        # (nlev,)  1/m³   immersion het INP (JAM #494); 0 → DeMott floor
    ice_nuclei_deposition: jnp.ndarray,  # (nlev,) 1/m³  deposition INP → cirrus nucleation
    dt: jnp.ndarray,                # scalar   seconds
    params: CloudParams2M,          # tunable parameters
) -> tuple[
    MicrophysicsTendencies_2M,      # per-level tendencies
    jnp.ndarray, jnp.ndarray,       # surface rain / snow flux [kg/m^2/s]
    jnp.ndarray, jnp.ndarray,       # liq / ice effective radius [um] (nlev,)
    jnp.ndarray, jnp.ndarray,       # rain / snow(+ice) flux leaving each layer [kg/m^2/s] (nlev,)
]:
    """Column-sweep orchestrator for the two-moment microphysics scheme.

    Processes (in ECHAM6 order):

      1. **Warm precipitation** (level-independent): qc → qr via KK2000
         autoconversion + accretion (:func:`precip_formation_warm`).
      2. **Mixed-phase deposition** (level-independent): vapor ↔ ice/liquid
         deposition/condensation (:func:`mixed_phase_deposition_and_corrections`).
      3. **Homogeneous freezing** (level-independent): all liquid → ice
         where T < 238 K (:func:`freezing_below_238K`).
      4. **Heterogeneous mixed-phase freezing** (level-independent):
         DeMott (2010) INP parameterization (:func:`demott2010_inp`).
         Uses prescribed coarse-mode aerosol + temperature.
      5. **WBF** (level-independent): remaining liquid → ice in
         mixed-phase clouds (:func:`WBF_process`).
      6. **Cold precipitation** (level-independent): qi → qs aggregation +
         qc → qs riming (:func:`precip_formation_cold`).
      7. **Flux-coupled column sweep** (top-down ``lax.scan``):
         - Ice sedimentation (:func:`sedimentation_ice`)
         - Melting of snow / ice (:func:`melting_snow_and_ice`)
         Precipitation fluxes (rain, snow, ice mass/number) propagate
         downward through the scan carry.

    qnc / qni are stored per kg of air; the scheme interior uses per-m^3,
    so we convert at the boundary.
    """
    eps_dt = jnp.finfo(qc.dtype).eps

    # ECHAM's per-level loop clamps icnc to ``[icemin, icemax]`` and
    # forces cdnc to ``[cqtmin, cdnc_min_upper]``-or-above (lines 1252-3
    # of mo_cloud_micro_2m.f90 and the activation block in
    # update_in_cloud_water). Mirror that on the orchestrator's INPUT so
    # the dynamical-core's spectral round-trip ringing — which can leave
    # small negative artefacts that ``update_in_cloud_water`` amplifies
    # via the ``delta_cdnc = activated_cdnc - droplet_number`` step —
    # cannot drive a multi-day runaway. Upper bound chosen as
    # ``cdnc_max_phys`` (1e11 / m^3, well above any realistic activation
    # output) and ``icemax`` (1e7 / m^3) so realistic clouds are
    # unaffected.
    _cdnc_max_phys_per_m3 = 1.0e11
    inv_rho_safe = 1.0 / jnp.maximum(air_density, eps_dt)
    qnc = jnp.clip(qnc, 0.0, _cdnc_max_phys_per_m3 * inv_rho_safe)
    qni = jnp.clip(qni, 0.0, params.icemax * inv_rho_safe)

    # Number-per-kg-of-air → per-m^3 at the scheme's API boundary.
    cdnc = qnc * air_density
    icnc = qni * air_density

    # Minimum cloud-droplet number — the SAME ECHAM ``minimum_CDNC`` the warm
    # microphysics uses below (the dynamic max-radius floor or the fixed
    # ``cdnc_min_fixed``, selected by ``ldyn_cdnc_min``; calibratable via the
    # ``cdnc_min_*`` parameters). The KK2000 autoconversion rate scales as
    # ``Nc^-1.79``, so without a floor a clean column (Nc -> 0; e.g. when the
    # MACv2-SP aerosol AOD is ~0) autoconverts essentially all cloud water to
    # rain instantly, leaving ~no cloud (LWP ~0.2 g/m2 vs the 1M scheme's ~20).
    # Flooring the droplet number by that same minimum keeps a realistic
    # cloud-water reservoir.
    # minimum_CDNC expects the in-cloud water content in kg/m³ (only
    # consumed when ldyn_cdnc_min=True); passing grid-mean kg/kg fed the
    # dynamic branch values ~ρ·cf too small (review finding 2.24).
    inv_cf_min = 1.0 / jnp.maximum(cloud_fraction, params.epsec)
    qc_in_cloud_kgm3 = jnp.where(
        cloud_fraction > params.epsec, qc * inv_cf_min * air_density, 0.0,
    )
    cdnc_min = minimum_CDNC(qc_in_cloud_kgm3, params)
    cdnc = jnp.maximum(cdnc, cdnc_min)

    # pauloc==1 and pclcstar==cloud_fraction are conservative first-pass
    # approximations — refine in later 5b steps.
    autoconv_factor = jnp.ones_like(qc)
    min_cloud_precip_fraction = cloud_fraction

    # ------------------------------------------------------------------
    # Warm precipitation formation (KK2000 autoconversion + accretion)
    # ------------------------------------------------------------------
    # ECHAM ll_prcp_warm (mo_cloud_micro_2m.f90:1662-1664): cloud cell
    # with liquid present and CDNC at/above the activation floor —
    # NO temperature condition. Warm-rain coalescence operates on
    # supercooled liquid too; a (T > tmelt) gate left every supercooled
    # stratus deck (polar boundary layers, storm tracks, 238-273 K)
    # without its only liquid sink once the corrected mixed-phase
    # partitioning (#554, finding 2.27) started producing supercooled
    # liquid there — cloud water built up ~50x over a month of coupled
    # T63L47 integration and NaN'd the run radiatively. The
    # cdnc >= cdnc_min leg holds by construction after the entry floor.
    warm_precip_mask = (cloud_fraction > params.epsec) & (qc > params.ccwmin)

    # ECHAM runs the KK2000 warm-rain chain on the IN-CLOUD liquid (zxlb)
    # and area-weights the products; feeding grid-mean qc underestimated
    # the qc^2.47 autoconversion by ~cf^1.47 (review finding 2.22). Convert
    # at the boundary: in-cloud in, grid-mean bookkeeping out.
    qc_ic_warm = jnp.where(
        cloud_fraction > params.epsec,
        qc / jnp.maximum(cloud_fraction, params.epsec),
        0.0,
    )
    (cdnc_warm, qc_ic_after_warm, _autoconv_in_cloud, _autoconv_rate,
     _dcdnc_removal, autoconv_only, accretion_only) = (
        precip_formation_warm(
            warm_precip_mask,
            autoconv_factor,
            cloud_fraction,
            min_cloud_precip_fraction,
            air_density,
            qr,
            cdnc_min,
            cdnc,
            qc_ic_warm,
            dt,
            params,
        )
    )
    qc_after_warm = jnp.where(
        cloud_fraction > params.epsec,
        qc_ic_after_warm * cloud_fraction,
        qc,
    )
    qr_gain_warm = qc - qc_after_warm  # grid-mean mass moved qc → rain (kg/kg)

    # ------------------------------------------------------------------
    # Derived quantities used across multiple process steps
    # ------------------------------------------------------------------
    inv_cf = jnp.where(
        cloud_fraction > params.epsec,
        1.0 / jnp.maximum(cloud_fraction, params.epsec),
        0.0,
    )
    in_cloud_liquid = qc_after_warm * inv_cf
    in_cloud_ice = qi * inv_cf
    inv_rho = 1.0 / jnp.maximum(air_density, eps_dt)
    lsdcp = c.alhs / c.cpd
    lvdcp = c.alhc / c.cpd
    zero = jnp.zeros_like(qc)

    # ------------------------------------------------------------------
    # Mixed-phase deposition and corrections
    # ------------------------------------------------------------------
    # Saturation vapour pressures from the shared ECHAM coefficients
    # (jcm.physics.thermodynamics). ``es_water`` uses the LIQUID-WATER
    # coefficients at ALL temperatures — the Bergeron-Findeisen variable
    # below and the threshold vertical velocity depend on the water/ice
    # saturation *difference* below freezing, which degenerates to zero
    # (killing the Bergeron process) if es_water switches to the ice
    # coefficients below 0 °C.
    es_water = thermodynamics.saturation_vapor_pressure(temperature, phase="water")
    es_ice = thermodynamics.saturation_vapor_pressure(temperature, phase="ice")

    qsat_water = c.eps * es_water / jnp.maximum(pressure - (1.0 - c.eps) * es_water, params.epsec)
    qsat_ice = c.eps * es_ice / jnp.maximum(pressure - (1.0 - c.eps) * es_ice, params.epsec)
    qsat_prev = jnp.where(temperature < params.tmelt, qsat_ice, qsat_water)

    bergeron_variable = jnp.clip(
        (specific_humidity - qsat_ice) / jnp.maximum(qsat_water - qsat_ice, params.epsec),
        0.0, 1.0,
    )

    # Updraft velocity [cm/s] from TKE (vertical velocity not plumbed yet).
    updraft_velocity = params.fact_tke * jnp.sqrt(jnp.maximum(2.0 * tke, 0.0)) * 100.0

    (
        condensation_rate, deposition_rate,
        temp_tmp, q_tmp, qsat_tmp,
        zvervmax_wbf,
    ) = mixed_phase_deposition_and_corrections(
        pressure,
        icnc,
        specific_humidity,
        cloud_fraction,
        es_ice, es_water,
        bergeron_variable,
        zero,               # tompkins_genti
        lsdcp, lvdcp,
        specific_humidity,
        qsat_prev,
        air_density,
        temperature,
        zero,               # ice_evaporation
        qi,
        zero,               # ice_detrainment_tendency
        updraft_velocity,
        zero,               # condensation_rate (INOUT, start at 0)
        zero,               # deposition_rate (INOUT, start at 0)
        dt,
        params,
    )

    # ------------------------------------------------------------------
    # Update in-cloud water/ice from deposition/condensation + activation
    # ------------------------------------------------------------------
    cloud_flag = cloud_fraction > 0.0

    # Mean ice crystal radius for ICNC nucleation path.
    ice_radius = eff_ice_crystal_radius(qi * air_density, icnc, params)

    (
        cloud_flag, icnc_uicw, _nucleation_rate, cdnc_uicw,
        cloud_fraction_uicw, in_cloud_ice_uicw, in_cloud_liquid_uicw,
        cdnc_min_uicw,
    ) = update_in_cloud_water(
        pressure,
        activated_cdnc,       # aerosol-activated CDNC (from MACv2-SP)
        condensation_rate,
        deposition_rate,
        zero,                 # tompkins_genti
        zero,                 # tompkins_gentl
        ice_nuclei_deposition,  # newly_formed_ice: cirrus deposition nucleation (#494)
        q_tmp,                # specific_humidity_tmp
        qsat_tmp,             # sat_spec_humidity_tmp
        air_density,
        ice_radius,
        temperature,          # temp_prev
        cloud_flag,
        icnc,
        zero,                 # nucleation_rate accumulator
        cdnc_warm,
        cloud_fraction,
        in_cloud_ice,
        in_cloud_liquid,
        dt,
        params,
    )

    # ------------------------------------------------------------------
    # Freezing below 238 K (homogeneous freezing, level-independent)
    # ------------------------------------------------------------------
    freezing_condition = temperature < params.cthomi
    (
        icnc_frz, _droplet_freezing_rate, cdnc_frz,
        _freezing_rate, in_cloud_ice_frz, in_cloud_liquid_frz,
    ) = freezing_below_238K(
        freezing_condition,
        cloud_fraction_uicw,
        cdnc_min_uicw,
        icnc_uicw,
        zero,             # droplet_freezing_rate accumulator
        cdnc_uicw,
        zero,             # freezing_rate accumulator
        in_cloud_ice_uicw,
        in_cloud_liquid_uicw,
        dt,
        params.cqtmin,
    )

    # ------------------------------------------------------------------
    # Heterogeneous mixed-phase freezing INP [1/m³]
    #
    # Prefer the online JAM heterogeneous ice nucleation (immersion+deposition
    # on prognostic dust/BC, #494) where it is active; fall back to the DeMott
    # (2010) parameterization on a prescribed coarse-aerosol number wherever the
    # online source is empty (≈0) — e.g. clean air or before the JAM tracers
    # spin up. Mirrors the ``activated_cdnc``/SPA-floor pattern for droplets.
    # ------------------------------------------------------------------
    het_condition = (temperature < params.tmelt) & (temperature >= params.cthomi)

    demott_floor = demott2010_inp(temperature, params.n_aer_coarse)
    n_inp = jnp.where(ice_nuclei > 0.0, ice_nuclei, demott_floor)

    # Where het freezing is active and INP > current ICNC, set ICNC to
    # INP and freeze a corresponding amount of liquid → ice.
    icnc_het = jnp.where(het_condition & (n_inp > icnc_frz), n_inp, icnc_frz)

    # Freeze liquid proportional to the new ice crystals formed, assuming
    # each INP freezes one droplet with mass = mean droplet mass.
    new_crystals = jnp.maximum(icnc_het - icnc_frz, 0.0)
    mean_droplet_mass = jnp.where(
        cdnc_frz > params.epsec,
        in_cloud_liquid_frz * air_density / jnp.maximum(cdnc_frz, params.epsec),
        0.0,
    )
    frozen_mass = new_crystals * mean_droplet_mass * inv_rho  # kg/kg
    frozen_mass = jnp.minimum(frozen_mass, in_cloud_liquid_frz)

    in_cloud_ice_het = in_cloud_ice_frz + frozen_mass
    in_cloud_liquid_het = in_cloud_liquid_frz - frozen_mass
    cdnc_het = jnp.where(
        het_condition, jnp.maximum(cdnc_frz - new_crystals, params.cqtmin), cdnc_frz,
    )

    # ------------------------------------------------------------------
    # WBF (Wegener-Bergeron-Findeisen): liquid → ice in mixed-phase
    # ------------------------------------------------------------------
    # ECHAM ll_WBF (mo_cloud_micro_2m.f90:1590-1594): the Bergeron
    # conversion fires only in the mixed-phase window (cthomi < T < tmelt;
    # below cthomi freezing_below_238K already emptied the liquid), with
    # active deposition (zdep > 0), enough droplets (cdnc ≥ floor), AND —
    # the criterion the port dropped — a weak updraft:
    # 0.01·zvervx < zvervmax, the Korolev/Mazin threshold velocity below
    # which ice grows at the liquid's expense. Without it every
    # mixed-phase cloud glaciated in one step and no supercooled liquid
    # survived (review finding 2.19). zvervmax comes from the deposition
    # block (ECHAM recomputes it from post-freezing zxib — a second-order
    # refinement over reusing the deposition-stage value).
    wbf_mask = (
        (temperature < params.tmelt)
        & (temperature > params.cthomi)
        & (in_cloud_liquid_het > params.epsec)
        & (in_cloud_ice_het > params.epsec)
        & (deposition_rate > 0.0)
        & (0.01 * updraft_velocity < zvervmax_wbf)
    )
    (
        cdnc_wbf, in_cloud_liquid_wbf, in_cloud_ice_wbf,
        _liq_tend_wbf, _ice_tend_wbf, dtedt_wbf,
    ) = WBF_process(
        wbf_mask,
        cloud_fraction,
        lsdcp, lvdcp,
        cdnc_het,
        in_cloud_liquid_het,
        in_cloud_ice_het,
        zero,             # cloud_liquid_tendency accumulator
        zero,             # cloud_ice_tendency accumulator
        zero,             # temp_tendency accumulator
        dt,
        params,
    )

    # ------------------------------------------------------------------
    # Cold precipitation formation (ice aggregation → snow + riming)
    # Uses post-freezing/WBF in-cloud values.
    # ------------------------------------------------------------------
    dynamic_viscosity = 4.1867e-3 * (5.69 + 0.017 * (temperature - params.tmelt))
    cold_mask = (temperature <= params.tmelt) & (qi > params.ccwmin)

    (
        icnc_cold,
        cdnc_cold,
        _snow_rate_in_cloud,
        in_cloud_ice_cold,
        in_cloud_liquid_cold,
        _psprn,
        psacl,
        _psacln,
        _pmsnowacl,
        snow_formation_gridmean,
    ) = precip_formation_cold(
        cold_mask,
        autoconv_factor,
        cloud_fraction,
        min_cloud_precip_fraction,
        inv_rho,
        inv_rho,
        temperature,
        dynamic_viscosity,
        qs,
        air_density,
        cdnc_min,
        icnc_het,         # WBF doesn't modify icnc; chain from het step
        cdnc_wbf,
        jnp.zeros_like(qc),
        in_cloud_ice_wbf,
        in_cloud_liquid_wbf,
        dt,
        params,
    )

    # Convert in-cloud → grid-mean for tendency computation.
    qi_after_cold = in_cloud_ice_cold * cloud_fraction
    # (qc_to_snow / qi_to_snow state differences removed with the qr/qs
    # ledger — see the dqrdt/dqsdt note below.)

    # ------------------------------------------------------------------
    # Flux-coupled column sweep (top-down lax.scan)
    #
    # Sedimentation and melting couple across levels via precipitation
    # fluxes: the flux leaving level k enters level k+1. We use
    # jax.lax.scan from top of atmosphere to surface to propagate
    # rain_flux, snow_flux, ice_flux, ice_flux_n, and
    # falling_ice_fraction correctly.
    # ------------------------------------------------------------------
    # Precompute per-level inputs for the scan.
    pressure_thickness = air_density * params.grav * layer_thickness
    air_density_correction = (1.3 * inv_rho) ** 0.4
    melt_mask = temperature > params.tmelt

    # Pre-compute sublimation/evaporation quantities for the scan.
    # ECHAM conventions (mo_cloud_micro_2m): the subsaturations are the
    # NEGATIVE relative deficits ``min(q/qs − 1, 0)`` — the sublimation/
    # evaporation chain needs the sign to produce a sink (zzeps =
    # max(−…, coeff·subsat) with a final clip at 0). The previous
    # positive-definite ``max(qs − q, 0)`` inverted the chain so the clip
    # floored rain evaporation and snow sublimation at exactly zero —
    # both processes were dead code (survey finding; §2.12-2M analog).
    dp_over_g = pressure_thickness * c.rgrav
    subsat_wrt_ice = jnp.minimum(
        specific_humidity / jnp.maximum(qsat_ice, params.epsec) - 1.0, 0.0,
    )
    subsat_wrt_water = jnp.minimum(
        specific_humidity / jnp.maximum(qsat_water, params.epsec) - 1.0, 0.0,
    )
    # Rotstayn thermodynamic + vapour-diffusion factor (ECHAM zastbstw =
    # zast + zbst), the same chain the 1M rain evaporation uses:
    #   zast = Lv·(Lv/(Rv·T) − 1)/(T·0.024),  zbst = Rv·T/(Dv·esw),
    # with Dv = 2.21/p. The previous value was the psychrometric factor
    # 1 + L²qs/(Rv·cp·T²) (~2-5) — ~6 orders of magnitude smaller than
    # zast+zbst, which would have made the (revived) evaporation ~1e6×
    # too strong.
    esw_orch = qsat_water * pressure / jnp.maximum(
        c.eps + (1.0 - c.eps) * qsat_water, params.epsec,
    )
    zdv_orch = 2.21 / jnp.maximum(pressure, params.epsec)
    zast_orch = (
        c.alhc * (c.alhc / (c.rv * jnp.maximum(temperature, 1.0)) - 1.0)
        / jnp.maximum(temperature, 1.0) / 0.024
    )
    zbst_orch = c.rv * temperature / jnp.maximum(zdv_orch * esw_orch, params.epsec)
    thermo_term_water = zast_orch + zbst_orch

    def _flux_coupled_step(carry, level_in):
        """Process one level: sedi → melt → sublim/evap → update_precip."""
        (rain_flux, snow_flux, ice_flux, ice_flux_n,
         falling_ice_frac, precip_cover) = carry
        (cf_k, adc_k, dp_k, rho_k, inv_rho_k, qi_k, icnc_k, cdnc_k,
         t_k, melt_k,
         q_k, dpg_k, subice_k, subwat_k, qsi_k, qsw_k, thermo_k,
         rain_form_k, snow_accr_k, snow_form_k,
         is_bottom_k,
         ) = level_in

        # --- Sedimentation ---
        (
            qi_post_sedi, icnc_post_sedi,
            ice_flux, ice_flux_n, falling_ice_frac,
            _sedi_rate,
        ) = sedimentation_ice(
            cf_k, adc_k, dp_k, rho_k, inv_rho_k,
            qi_k, icnc_k,
            ice_flux, ice_flux_n, falling_ice_frac,
            dt,
            params,
        )

        # --- Melting --- (per-level pimlt/psmlt/pximlt are KEPT: the
        # in-cloud melt mass goes to cloud liquid, the falling-ice melt to
        # cloud liquid, the snow melt to rain — all with per-level fusion
        # cooling in update_tendencies. The previous scan discarded all
        # three (melted in-cloud ice VANISHED from the mass ledger, only
        # the ICNC→CDNC number transfer survived) and broadcast a
        # column-accumulated snow_melt to every level (finding 2.23).
        (
            icnc_post_melt, _qmel, cdnc_post_melt,
            rain_flux, snow_flux, ice_flux, ice_flux_n,
            _ice_tend, pimlt_k, psmlt_k, pximlt_k,
        ) = melting_snow_and_ice(
            melt_k, t_k, qi_k, dp_k,
            icnc_post_sedi, lsdcp, lvdcp,
            icnc_post_sedi,
            jnp.array(0.0),  # qmel accumulator
            cdnc_k,
            rain_flux, snow_flux, ice_flux, ice_flux_n,
            jnp.array(0.0),  # ice_tendency
            dt,
            params,
        )

        # --- Sublimation / evaporation ---
        precip_mask = (rain_flux > params.cqtmin) | (snow_flux > params.cqtmin)
        falling_ice_mask_k = ice_flux > params.cqtmin

        (
            ice_flux, ice_flux_n,
            ice_sublim_k, snow_sublim_k, rain_evap_k,
        ) = sublimation_snow_and_ice_evaporation_rain(
            precip_mask, falling_ice_mask_k,
            q_k, t_k,
            precip_cover, dp_k, dpg_k,
            subice_k, lsdcp, inv_rho_k,
            qsi_k, inv_rho_k,
            snow_flux, rho_k,
            qsw_k, rain_flux,
            subwat_k, thermo_k,
            falling_ice_frac,
            ice_flux, ice_flux_n,
            dt,
            params,
        )

        # --- Update precipitation fluxes ---
        (
            precip_cover, rain_flux, snow_flux, snow_melt,
            _pfevapr, _pfrain, _pfsnow, _pfsubls,
        ) = update_precip_fluxes(
            cf_k, dp_k,
            rain_evap_k, lsdcp, lvdcp,
            rain_form_k, snow_accr_k, snow_form_k,
            snow_sublim_k, t_k,
            # ECHAM folds the sedimenting ice flux into the snow flux ONLY
            # at the bottom level (Fortran ``kk == klev`` gate). Passing
            # the undepleted carry at every level counted a constant
            # cirrus flux into snow once per level below it — ~nlev× snow
            # (review finding 2.17).
            jnp.where(is_bottom_k, ice_flux, 0.0),
            # Per-level melt bookkeeping (ECHAM zeroes zsmlt each jk):
            # feed 0 in and take the routine's output as this level's
            # increment, added to the melting-subroutine psmlt below.
            precip_cover, rain_flux, snow_flux, jnp.array(0.0),
            dt,
            params,
        )
        psmlt_level = psmlt_k + snow_melt

        carry_out = (rain_flux, snow_flux, ice_flux, ice_flux_n,
                     falling_ice_frac, precip_cover)
        # Per-level flux profiles for downstream (COSP/CloudSat)
        # diagnostics: the grid-mean rain / frozen fluxes LEAVING this
        # layer (the carry values after update_precip_fluxes). The frozen
        # profile adds the sedimenting cloud-ice flux at interior levels
        # so it is the total falling frozen water; at the bottom level
        # ``update_precip_fluxes`` has already folded ``ice_flux`` into
        # ``snow_flux`` (ECHAM ``kk == klev`` gate), so adding it again
        # there would double-count — hence the ``is_bottom_k`` guard,
        # which also makes the bottom row equal ``surface_snow_flux``
        # exactly.
        frozen_flux_k = snow_flux + jnp.where(is_bottom_k, 0.0, ice_flux)
        level_out = (qi_post_sedi, icnc_post_melt, cdnc_post_melt,
                     ice_sublim_k, snow_sublim_k, rain_evap_k,
                     psmlt_level, pimlt_k, pximlt_k,
                     rain_flux, frozen_flux_k)
        return carry_out, level_out

    # Stack per-level inputs: shape (nlev,) each → scanned along axis 0.
    nlev_scan = temperature.shape[0]
    is_bottom_level = jnp.arange(nlev_scan) == (nlev_scan - 1)
    scan_inputs = (
        cloud_fraction, air_density_correction, pressure_thickness,
        air_density, inv_rho, qi_after_cold, icnc_cold, cdnc_cold,
        temperature, melt_mask,
        specific_humidity, dp_over_g, subsat_wrt_ice, subsat_wrt_water,
        qsat_ice, qsat_water, thermo_term_water,
        qr_gain_warm, psacl, snow_formation_gridmean,
        is_bottom_level,
    )

    zero_scalar = jnp.array(0.0, dtype=qc.dtype)
    init_carry = (zero_scalar, zero_scalar, zero_scalar,
                  zero_scalar, zero_scalar, zero_scalar)

    _final_carry, scan_outs = jax.lax.scan(
        _flux_coupled_step, init_carry, scan_inputs,
    )
    (qi_after_scan, icnc_after_scan, cdnc_after_scan,
     ice_sublim, snow_sublim, rain_evap,
     psmlt_per_level, pimlt_per_level, pximlt_per_level,
     rain_flux_profile, snow_flux_profile) = scan_outs

    # Extract carry state at the bottom of the column. The first two
    # elements are the surface rain and snow flux (kg/m^2/s) — these are
    # the large-scale precipitation diagnostics that callers need.
    (
        surface_rain_flux, surface_snow_flux,
        _, _, _, _,
    ) = _final_carry

    # ------------------------------------------------------------------
    # update_tendencies_and_important_vars: full ECHAM6 accounting step
    # ------------------------------------------------------------------
    liquid_cloud_flag = temperature > params.tmelt
    ice_cloud_flag = temperature <= params.tmelt

    (
        cloud_fraction_final,
        dqdt, dtedt, dqidt, dqcdt,
        dqncdt_m3, dqnidt_m3,
        _incloud_liq, _incloud_ice,
        liq_eff_radius, ice_eff_radius,
    ) = update_tendencies_and_important_vars(
        icnc=icnc_after_scan,
        cdnc=cdnc_after_scan,
        ice_mmr_prev=in_cloud_ice_cold,
        liq_mmr_prev=in_cloud_liquid_cold,
        # ECHAM convention: pxtm1_cdnc / pxtm1_icnc are the previous-step
        # tracer values in per-kg-of-air. ``cdnc`` and ``icnc`` here are
        # the working per-m^3 values (qnc * rho, qni * rho), so we pass
        # the per-kg ``qnc``/``qni`` instead. With the original (per-m^3)
        # values the formula mixes per-kg with per-m^3 in the same
        # subtraction and the resulting per-step amplification (~1/rho^2
        # at upper levels) compounds qnc/qni 10+ orders of magnitude
        # over a few days, producing the day-6 NaN.
        tracer_tm1_cdnc=qnc,
        tracer_tm1_icnc=qni,
        condensation_rate=condensation_rate,
        deposition_rate=deposition_rate,
        rain_evap_mmr=rain_evap,
        freezing_rate=_freezing_rate,
        tompkins_ice=zero,
        tompkins_liq=zero,
        incloud_ice_melt=pimlt_per_level,
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        air_density=air_density,
        inv_air_density=inv_rho,
        rain_formation=qr_gain_warm,
        snow_accretion=psacl,
        snow_formation=snow_formation_gridmean,
        cloud_ice_evap=zero,         # not extracted from scan
        ice_flux_melt=pximlt_per_level,
        pxitec=zero,
        # pxlevap is the clear-cell CLOUD-liquid evaporation (ECHAM
        # zxlevap), not rain evaporation — passing rain_evap here as well
        # as via pevp double-counted its moistening/cooling and removed
        # the water from both qc and the rain flux (review finding 2.21).
        pxlevap=zero,
        pxltec=zero,
        # Falling-ice sublimation (ECHAM zxisub): the sublimation routine
        # deducts this mass from the falling ice flux, so it must re-enter
        # the column as vapor here — feeding zero destroyed water at the
        # sublimation rate (only visible in cold columns where sedimenting
        # ice crosses subsaturated layers).
        pxisub=ice_sublim,
        snow_sublimation_mmr=snow_sublim,
        snow_melt=psmlt_per_level,
        cloud_ice_in_cloud=in_cloud_ice_cold,
        cloud_liquid_in_cloud=in_cloud_liquid_cold,
        temp_tmp=temperature,
        liquid_cloud_flag=liquid_cloud_flag,
        ice_cloud_flag=ice_cloud_flag,
        cloud_fraction=cloud_fraction_uicw,
        specific_humidity_tendency=zero,
        temp_tendency=dtedt_wbf,     # seed with WBF contribution
        # ECHAM folds ice sedimentation into pxite BEFORE this ledger
        # (mo_cloud_micro_2m.f90 section 4: pxite = ztmst_rcp·(zxip1 −
        # pxim1) after sedimentation_ice) — seeding zero here emitted the
        # bottom-reaching ice flux as surface snow with no qi debit,
        # opening the flux-form budget by exactly that flux (Codex review
        # on #554).
        ice_tendency=(qi_after_scan - qi_after_cold) / dt,
        liq_tendency=zero,
        tracer_tendency_cdnc=zero,
        tracer_tendency_icnc=zero,
        incloud_liq_before_rain=in_cloud_liquid,  # before warm step
        incloud_ice_before_snow=in_cloud_ice_uicw, # before cold step
        dt=dt,
        params=params,
    )

    # Mass tendencies: warm qc→qr, cold qi→qs + qc→qs, sedi+melt loss
    # ECHAM carries NO rain/snow mixing-ratio tracers: precipitation
    # leaves each level exclusively through the prfl/psfl fluxes assembled
    # in update_precip_fluxes, and the surface fluxes are the outputs. The
    # previous state-difference dqrdt/dqsdt double-booked the same mass
    # (once in prognostic qr/qs, once in the scan fluxes), went negative
    # whenever qi_after_cold gained from WBF/het/deposition, and was then
    # silently clipped by the non-negative-tracer guard — hidden mass
    # destruction (review finding 2.18).
    dqrdt = jnp.zeros_like(qc)
    dqsdt = jnp.zeros_like(qc)

    # update_tendencies' tracer_tendency_{cdnc,icnc} is already in per-kg-
    # of-air per second once we pass qnc/qni (per-kg) as the tm1 tracers
    # — see the fix above. The legacy ``* inv_rho`` here was a second
    # units error compounded with the per-kg-vs-per-m^3 swap above.
    dqncdt = dqncdt_m3
    dqnidt = dqnidt_m3

    tendencies = MicrophysicsTendencies_2M(
        dtedt=dtedt,
        dqdt=dqdt,
        dqcdt=dqcdt,
        dqidt=dqidt,
        dqncdt=dqncdt,
        dqnidt=dqnidt,
        dqrdt=dqrdt,
        dqsdt=dqsdt,
    )
    # Column-integrated rain sources [kg/m^2/s], split by pathway: the
    # warm chain (KK2000 autoconversion + accretion, ``qr_gain_warm``) and
    # snow melt (``psmlt_per_level``). Their ratio is the model's
    # warm-rain fraction, the CloudSat-style observable that constrains
    # the warm-rain parameters (ccraut, and the SPA activation fit through
    # CDNC). Both are per-step grid-mean mixing-ratio increments, so the
    # column flux is sum(dq * rho * dz) / dt.
    air_mass = air_density * layer_thickness  # [kg/m^2] per level
    rain_formation_warm = jnp.sum(qr_gain_warm * air_mass) / dt
    rain_from_melt = jnp.sum(psmlt_per_level * air_mass) / dt

    # AeroCom process rates [kg/m^2/s], same column-integral convention.
    # autoconv and accretn split the warm chain above into its two
    # pathways (their sum is qr_gain_warm up to the droplet-number
    # limiter); wbf is the liquid mass converted to ice by the
    # Wegener-Bergeron-Findeisen process. All are grid-mean, so the
    # in-cloud WBF increment is weighted by cloud fraction.
    autoconv_rate_col = jnp.sum(autoconv_only * air_mass) / dt
    accretion_rate_col = jnp.sum(accretion_only * air_mass) / dt
    wbf_rate_col = jnp.sum(
        (in_cloud_liquid_het - in_cloud_liquid_wbf) * cloud_fraction
        * air_mass) / dt

    # Microphysical effective radii (ECHAM preffl/preffi, um) — consumed
    # by the radiation term via the clouds carry (finding 2.36: the
    # radiation-side fabricated r_eff(T)*clip(IWC) saturated at the LUT
    # edge for thin cirrus, mis-forcing the TTL).
    # The (nlev,) rain / frozen flux profiles (flux leaving each layer,
    # stacked from the scan ys) go last so existing ``tend, rain, snow,
    # *_`` call sites keep working; their bottom row equals the surface
    # fluxes by construction (same carry values).
    # Per-level precip process rates for JAM wet scavenging (#499), grid
    # mean [kg/kg/s]. Formation is the full condensate→precip ledger the
    # flux update integrates: the warm chain (KK2000 autoconversion +
    # accretion, ``qr_gain_warm``), riming (``psacl``) and the cold snow
    # formation. Evaporation is rain evap + snow sublimation; the falling
    # cloud-ice sublimation (``ice_sublim``) is deliberately excluded — the
    # sedimenting ice flux is not a scavenging carrier.
    precip_formation_rate = (
        qr_gain_warm + psacl + snow_formation_gridmean
    ) / dt
    precip_evaporation_rate = (rain_evap + snow_sublim) / dt

    # NOTE the (nlev,) rain / frozen flux profiles stay LAST: call sites and
    # tests unpack them positionally from the end (``*_, rain_b, snow_b``),
    # so new outputs are inserted before them, not appended.
    return tendencies, surface_rain_flux, surface_snow_flux, \
        liq_eff_radius, ice_eff_radius, rain_formation_warm, rain_from_melt, \
        autoconv_rate_col, accretion_rate_col, wbf_rate_col, \
        precip_formation_rate, precip_evaporation_rate, \
        rain_flux_profile, snow_flux_profile


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------



class Lohmann2MMicrophysics(PhysicsTerm):
    """ECHAM 2-moment cloud microphysics (Lohmann/Seifert-Beheng-style) term.

    Drop-in 2M alternative to :class:`Echam1MMicrophysics`. Declares the
    full prognostic-tracer set (``qc``, ``qi``, ``qnc``, ``qni``, ``qr``,
    ``qs``) — the ``qnc`` / ``qni`` number concentrations are stored per
    kg of air with ``nondimensionalize=False`` so the modal/nodal
    converters don't apply the gram/kg scaling that mass mixing ratios
    get.

    Reads the post-condensation ``cloud_fraction`` / ``qc`` / ``qi`` from
    the public ``"clouds"`` key (set by :class:`SundqvistCloudFraction`
    upstream), TKE from ``"vertical_diffusion"``, and the SPA-style
    activated CDNC floor from the public ``"aerosol"`` Nccn. Writes the
    surface rain / snow precip flux into ``"clouds"`` along with the
    qnc / qni state-carry needed for the next step's update.

    Must be composed downstream of ``SundqvistCloudFraction`` and
    (because it reads TKE) downstream of ``TteTkeVerticalDiffusion``.
    """

    name: ClassVar[str] = "lohmann_2m_microphysics"
    category: ClassVar[str] = "clouds"
    # ``vertical_diffusion`` is intentionally not in ``requires``: in the
    # default ECHAM physc ordering (vdiff → convection → microphysics) the
    # vdiff term runs upstream, so the TKE read here is same-step — but the
    # scheme must also compose in vdiff-free stacks (unit tests, minimal
    # RCE), where the soft read falls back to the carried/zero value.
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "air_density", "layer_thickness",
        "clouds", "aerosol",
    )
    provides: ClassVar[tuple[str, ...]] = (
        "autoconv", "accretn", "wbf", "clouds",
    )

    def __init__(self, params: 'CloudParams2M | None' = None):
        """Hold the scheme-native :class:`CloudParams2M`."""
        if params is None:
            params = CloudParams2M.default()
        self.params = nnx.Param(params)
        # SPA-activation knobs currently live on ``AerosolParameters``;
        # cache them here so the term doesn't have to read them through
        # the aerosol typed sub-struct (where they may not be present in
        # custom compositions).
        self._spa_prefactor = nnx.Param(jnp.array(1.0))
        self._spa_exponent = nnx.Param(jnp.array(0.5))
        self._spa_cap_smoothing = nnx.Param(jnp.array(0.0))

    def configure_spa(self, prefactor: float, exponent: float,
                      cap_smoothing: float = 0.0) -> None:
        """Set the SPA prefactor / exponent / cap-smoothing (factory hook)."""
        self._spa_prefactor = nnx.Param(jnp.asarray(prefactor))
        self._spa_exponent = nnx.Param(jnp.asarray(exponent))
        self._spa_cap_smoothing = nnx.Param(jnp.asarray(cap_smoothing))

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """Declare the full 2M prognostic tracer set."""
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
            TracerSpec("qnc", units="kg^-1", nondimensionalize=False),
            TracerSpec("qni", units="kg^-1", nondimensionalize=False),
            # NOTE: rain/snow are NOT prognostic tracers in ECHAM's 2M —
            # precipitation lives entirely in the within-step prfl/psfl
            # fluxes (finding 2.18). The former qr/qs tracers double-booked
            # that mass.
        )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute 2M microphysics tendencies and update ``"clouds"``."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params_2m = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        air_density = diagnostics["air_density"]
        layer_thickness = diagnostics["layer_thickness"]

        # Post-(vdiff+convection) thermodynamic state (sequential
        # vdiff->convection->cloud coupling, ECHAM physc order): the upstream
        # vdiff and convection terms have already advanced ``thermo_run`` with
        # their tendencies and convection forwarded its detrained condensate
        # into ``clouds.qc/qi``. Doing the saturation balance + clear-sky
        # evaporation on this post-upstream (T, q) — instead of the step-start state — is
        # what makes the in-step moist-energy balance consistent and the
        # clear-sky evaporation stable (see
        # ``.claude/coupled_cloud_operator_design.md``). Tendencies are returned
        # relative to this state; the host's additive sum with convection's
        # tendency telescopes back to the correct final state. Falls back to the
        # step-start state if no upstream term seeded ``thermo_run``.
        thermo_run = diagnostics.get("thermo_run")
        if thermo_run is None:
            temperature_in = state.temperature
            specific_humidity_in = state.specific_humidity
        else:
            temperature_in = thermo_run["temperature"]
            specific_humidity_in = thermo_run["specific_humidity"]

        clouds = diagnostics["clouds"]
        qc_interim = clouds.qc
        qi_interim = clouds.qi
        cloud_fraction = clouds.cloud_fraction

        zeros = jnp.zeros_like(state.temperature)
        qnc = state.tracers.get("qnc", zeros)
        qni = state.tracers.get("qni", zeros)
        qr = state.tracers.get("qr", zeros)
        qs = state.tracers.get("qs", zeros)

        if "vertical_diffusion" in diagnostics:
            tke = diagnostics["vertical_diffusion"].tke
        else:
            tke = jnp.zeros_like(state.temperature)

        # Activated CDNC source. The inline SPA floor (from the MACv2-SP Nccn)
        # is always computed as a baseline. An upstream activation term (e.g.
        # JAM's ARG, #461) may write an explicit ``activated_cdnc``; when it
        # does we use it, but fall back to the SPA floor wherever that online
        # source is empty (≈0) — e.g. before the prognostic JAM aerosol tracers
        # spin up — so the default JAM+2M run still activates droplets. Both
        # paths are differentiable and produce the same (nlev, ncols) field.
        Nccn = diagnostics["aerosol"].Nccn
        spa_floor = spa_activated_cdnc(
            Nccn=Nccn[jnp.newaxis, :],
            cloud_fraction=cloud_fraction,
            prefactor=self._spa_prefactor.get_value(),
            exponent=self._spa_exponent.get_value(),
            cap_smoothing=self._spa_cap_smoothing.get_value(),
        )
        arg_cdnc = diagnostics.get("activated_cdnc")
        if arg_cdnc is None:
            activated_cdnc = spa_floor
        else:
            activated_cdnc = jnp.where(arg_cdnc > 1.0, arg_cdnc, spa_floor)

        # Online heterogeneous ice nuclei (JAM #494); 0 where absent so the
        # core falls back to its DeMott floor. Immersion drives the mixed-phase
        # het freezing; deposition drives the cirrus nucleation hook.
        zeros_2d = jnp.zeros_like(state.temperature)
        ice_nuclei = diagnostics.get("ice_nuclei", zeros_2d)
        ice_nuclei_deposition = diagnostics.get(
            "ice_nuclei_deposition", zeros_2d
        )

        (tend_all, surface_rain_flux, surface_snow_flux,
         r_eff_liq_all, r_eff_ice_all, rain_formation_warm, rain_from_melt,
         autoconv_all, accretion_all, wbf_all,
         precip_form_all, precip_evap_all,
         rain_flux_all, snow_flux_all) = jax.vmap(
            cloud_microphysics_2m,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, None),
            out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )(
            temperature_in, specific_humidity_in, pressure_full,
            qc_interim, qi_interim, qnc, qni, qr, qs,
            cloud_fraction, air_density, layer_thickness, tke,
            activated_cdnc, ice_nuclei, ice_nuclei_deposition, dt, params_2m,
        )

        # Grid-mean Sundqvist condensation / evaporation — the implicit
        # saturation adjustment the 1M column-sweep performs but the 2M
        # microphysics lacked. Without it the 2M scheme forms almost no
        # stratiform cloud in saturated cells (an A/B spin-up gave a
        # radiatively-active LWP ~50x smaller than 1M) and the condensate it
        # does carry — convective detrainment advected into sub-saturated
        # cells — accumulates unbounded. This step condenses vapour -> qc/qi
        # where the post-convection grid box is supersaturated (forming
        # radiatively-active cloud) and evaporates qc/qi where it is
        # sub-saturated, capped at the available cloud water, via the
        # warming-feedback-damped Newton step (so it is stable, unlike the
        # clear-cell-evaporation bolt-on). ``condensation_evaporation`` is
        # broadcasting-native, so it runs directly on the (nlev, ncols) state.
        from ..sundqvist import (
            condensation_evaporation as _sundqvist_cond_evap,
            CloudParameters as _SundqvistCloudParams,
        )
        dtedt_strat, dqdt_strat, dqcdt_strat, dqidt_strat = _sundqvist_cond_evap(
            temperature_in, specific_humidity_in, qc_interim, qi_interim,
            cloud_fraction, pressure_full, dt, _SundqvistCloudParams.default(),
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=tend_all.dtedt.T + dtedt_strat,
            specific_humidity=tend_all.dqdt.T + dqdt_strat,
            tracers={
                "qc": tend_all.dqcdt.T + dqcdt_strat,
                "qi": tend_all.dqidt.T + dqidt_strat,
                "qnc": tend_all.dqncdt.T,
                "qni": tend_all.dqnidt.T,
            },
        )

        # Stash current-step qnc/qni as tm1 for the next step's
        # update_tendencies_and_important_vars; expose surface precip
        # diagnostics from the lax.scan.
        clouds_next = clouds.copy(
            qnc_prev=qnc, qni_prev=qni,
            precip_rain=surface_rain_flux,
            precip_snow=surface_snow_flux,
            # Per-level precipitation flux profiles for satellite-simulator
            # diagnostics (COSP/CloudSat). The vmap over columns puts the
            # column axis first — transpose back to the (nlev, ncols)
            # CloudData layout, same as the effective radii below.
            rain_flux=rain_flux_all.T,
            snow_flux=snow_flux_all.T,
            # Per-level precip formation / evaporation rates [kg/kg/s] for
            # JAM wet scavenging (#499); see cloud_microphysics_2m.
            precip_formation_rate=precip_form_all.T,
            precip_evaporation_rate=precip_evap_all.T,
            # Microphysical effective radii (um) for the radiation term
            # (ECHAM preffl/preffi; consumed next step via the carry —
            # same lag as every cross-term diagnostic).
            r_eff_liq=r_eff_liq_all.T,
            r_eff_ice=r_eff_ice_all.T,
            # Rain-source split [kg/m^2/s]: warm-chain formation vs snow
            # melt. Their ratio is the warm-rain fraction, the CloudSat-
            # style observable for the warm-rain calibration.
            rain_formation_warm=rain_formation_warm,
            rain_from_melt=rain_from_melt,
        )
        # Advance the running condensate view so terms downstream (the
        # satellite simulators and the AeroCom diagnostics) describe the
        # POST-microphysics atmosphere, matching the tracers saved at the
        # same timestamp. ``thermo_run`` is a parallel diagnostic view,
        # never the prognostic state, so this cannot alter the trajectory
        # (see ``advance_thermo_run``).
        # AeroCom microphysical process rates [kg/m^2/s], column-integrated
        # (jax-gcm#585). Published unconditionally so the diagnostics key set
        # stays static across steps — the dict is part of the scan carry.
        diagnostics = {**diagnostics,
                       "autoconv": autoconv_all,
                       "accretn": accretion_all,
                       "wbf": wbf_all}
        diagnostics = advance_thermo_run(
            diagnostics, dt,
            d_temperature=tendency.temperature,
            d_specific_humidity=tendency.specific_humidity,
            d_qc=tendency.tracers["qc"], d_qi=tendency.tracers["qi"])

        return tendency, {**diagnostics, "clouds": clouds_next}
