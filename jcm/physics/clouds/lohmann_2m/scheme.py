"""Lohmann 2M column-sweep orchestrator and composable physics term.

``cloud_microphysics_2m`` runs the full two-moment process chain over a
column as one flux-coupled top-down ``lax.scan`` — a faithful
transcription of ECHAM's ``column_processes`` loop — and
``Lohmann2MMicrophysics`` wraps it as a composable ``PhysicsTerm``.
Design rationale and the state-splitting convention:
``docs/source/design/lohmann_2m_column_processes.md``.
"""

from typing import ClassVar
from math import pi

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
    ice_volume_mean_radius,
    minimum_CDNC,
    threshold_vert_vel,
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
    temperature: jnp.ndarray,       # (nlev,)  K      post-upstream provisional T
    specific_humidity: jnp.ndarray, # (nlev,)  kg/kg  post-upstream provisional q
    pressure: jnp.ndarray,          # (nlev,)  Pa
    qc: jnp.ndarray,                # (nlev,)  kg/kg post-upstream cloud liquid
    qi: jnp.ndarray,                # (nlev,)  kg/kg post-upstream cloud ice
    qnc: jnp.ndarray,               # (nlev,)  kg^-1 cloud droplet number per kg of air
    qni: jnp.ndarray,               # (nlev,)  kg^-1 ice crystal number per kg of air
    cloud_fraction: jnp.ndarray,    # (nlev,)  [0,1]
    air_density: jnp.ndarray,       # (nlev,)  kg/m^3
    layer_thickness: jnp.ndarray,   # (nlev,)  m   (dz, full-level layer depths)
    tke: jnp.ndarray,               # (nlev,)  m²/s²  turbulent kinetic energy
    activated_cdnc: jnp.ndarray,    # (nlev,)  1/m³   aerosol-activated CDNC (from MACv2-SP)
    ice_nuclei: jnp.ndarray,        # (nlev,)  1/m³   immersion het INP (JAM #494); 0 → DeMott floor
    ice_nuclei_deposition: jnp.ndarray,  # (nlev,) 1/m³  deposition INP → cirrus nucleation
    dt: jnp.ndarray,                # scalar   seconds
    params: CloudParams2M,          # tunable parameters
    temperature_m1: jnp.ndarray | None = None,        # (nlev,) K   step-start T (ECHAM ptm1)
    specific_humidity_m1: jnp.ndarray | None = None,  # (nlev,)     step-start q (ECHAM pqm1)
    qc_m1: jnp.ndarray | None = None,                 # (nlev,)     step-start qc (ECHAM pxlm1)
    qi_m1: jnp.ndarray | None = None,                 # (nlev,)     step-start qi (ECHAM pxim1)
) -> tuple[
    MicrophysicsTendencies_2M,      # per-level tendencies
    jnp.ndarray, jnp.ndarray,       # surface rain / snow flux [kg/m^2/s]
    jnp.ndarray, jnp.ndarray,       # liq / ice effective radius [um] (nlev,)
    jnp.ndarray, jnp.ndarray,       # rain / snow(+ice) flux leaving each layer [kg/m^2/s] (nlev,)
]:
    """Column orchestrator for the two-moment microphysics scheme.

    A faithful transcription of the ECHAM6-HAM ``mo_cloud_micro_2m.f90``
    ``column_processes`` loop: the WHOLE process chain runs inside one
    top-down ``lax.scan``, because in the reference every process at level
    ``jk`` sees the precipitation state (``prfl``/``pssfl``/``zclcpre``/
    ``zxiflux``) that the levels above produced *this step*. Splitting the
    "level-independent" processes out of the sweep — the previous layout —
    silently severed exactly those couplings: rain/snow-from-above
    accretion ran on tracers that no longer existed (#662 finding 5), the
    precipitation-cover geometry ``zclcstar`` was unavailable (#685), and
    ice created mid-step never met its aggregation sink (#686).

    Per level, in ECHAM section order (numbers = Fortran comments):

      4.    Ice sedimentation (:func:`sedimentation_ice`), then
      3.1   melting of snow / falling ice / in-cloud ice
            (:func:`melting_snow_and_ice`). NOTE this sediment→melt order
            is deliberately MG/PUMAS's (micro_pumas_v1: sediment 3093 →
            melt 3293), not ECHAM's melt→sediment; the melt acts on the
            post-sedimentation ice via the threaded tendency so the two
            sinks cannot claim the same mass (#662 finding 2).
      3.2/3 Snow/ice sublimation + rain evaporation on the incoming
            fluxes (:func:`sublimation_snow_and_ice_evaporation_rain`).
      (4b)  In-cloud condensate prep with clear-sky evaporation
            ``zxlevap``/``zxievap`` (ECHAM 1310-1385): condensate in
            cells with no cloud, and the clear-sky share of positive
            upstream increments, evaporates back to vapour (#667).
      5.    Grid-scale condensation source ``zqcdif`` → ``zcnd``/``zdep``
            (the Sundqvist moisture-convergence closure, ECHAM 1389-1470)
            followed by the supersaturation corrections
            (:func:`mixed_phase_deposition_and_corrections`). The scheme
            OWNS saturation adjustment — there is no external
            condensation bolt-on (#667).
      5.5   In-cloud water update + droplet activation / ICNC nucleation
            (:func:`update_in_cloud_water`).
      6.1   Homogeneous freezing below ``cthomi``
            (:func:`freezing_below_238K`).
      6.2   Heterogeneous mixed-phase freezing (JAM immersion INP with
            the DeMott (2010) fallback) and the WBF process with the
            Korolev/Mazin threshold updraft recomputed from the
            post-freezing ice (:func:`WBF_process`).
      7.    Precipitation geometry: ``zclcstar = min(paclc, zclcpre)``,
            the layer-depth ``zauloc`` ramp, and the Marshall-Palmer
            inversion of the carry fluxes into ``zxrp1``/``zxsp1`` (rain/
            snow water content seen by accretion; ECHAM 1614-1655 /
            Roeckner et al. 2003 eqs. 10.70, 10.74).
      7.1   Warm-rain formation (:func:`precip_formation_warm`) — AFTER
            condensation and activation, as in both references.
      7.2   Cold precipitation formation (:func:`precip_formation_cold`).
      7.3   Precipitation-flux update (:func:`update_precip_fluxes`).

    Section 8 (:func:`update_tendencies_and_important_vars`) is per-level
    algebra with no cross-level coupling, so it runs vectorized after the
    sweep on the stacked per-level outputs.

    State-splitting convention (operator-split host vs ECHAM leapfrog):
    the primary ``temperature``/``specific_humidity``/``qc``/``qi`` are
    the POST-UPSTREAM provisional state (ECHAM ``ptm1 + ztmst·ptte``
    etc.), which is what the returned tendencies are relative to. The
    optional ``*_m1`` arguments are the step-start state (ECHAM ``ptm1``/
    ``pqm1``/``pxlm1``/``pxim1``): saturation anchors evaluate there, and
    the differences ``(x - x_m1)`` play the role of ECHAM's accumulated
    tendencies ``ztmst·pqte``/``ztmst·pxlte`` in the condensation closure
    and the clear-sky-evaporation split. When omitted they default to the
    provisional state (zero upstream increments), which reduces section 5
    to a pure saturation adjustment.

    The large-scale vertical velocity is not plumbed to this scheme yet:
    ECHAM's ``zvervx`` (updraft for the WBF gate) uses only the TKE term
    here, and the ``knvb``/``lonacc`` inversion-level exception on
    ``zauloc`` is omitted (it needs ``pvervel``) — tracked in #705.

    qnc / qni are stored per kg of air; the scheme interior uses per-m^3,
    so we convert at the boundary.
    """
    if temperature_m1 is None:
        temperature_m1 = temperature
    if specific_humidity_m1 is None:
        specific_humidity_m1 = specific_humidity
    if qc_m1 is None:
        qc_m1 = qc
    if qi_m1 is None:
        qi_m1 = qi

    eps_dt = jnp.finfo(qc.dtype).eps
    zero = jnp.zeros_like(qc)
    lsdcp = c.alhs / c.cpd
    lvdcp = c.alhc / c.cpd

    # ------------------------------------------------------------------
    # Upstream increments (ECHAM's accumulated tendencies × ztmst)
    # ------------------------------------------------------------------
    dT_up = temperature - temperature_m1          # ztmst·ptte
    dq_up = specific_humidity - specific_humidity_m1  # ztmst·pqte
    dqc_up = qc - qc_m1                           # ztmst·(pxlte + detrainment)
    dqi_up = qi - qi_m1                           # ztmst·(pxite + detrainment)

    # ------------------------------------------------------------------
    # Entry clamps on the number tracers (jcm addition, see below)
    # ------------------------------------------------------------------
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
    #
    # The icnc LOWER bound is deliberately 0, not ECHAM's ``icemin``:
    # arrivals at or below ``icemin`` are re-diagnosed from ice mass in
    # ``update_in_cloud_water`` (the ``<=`` test fires either way), so the
    # floor would only inject a spurious icemin-per-step tracer source into
    # ice-free cells. Note this floor is NOT why mixed-phase r_eff_ice
    # saturates at ``ceffmax`` — those cells arrive with ICNC well above
    # icemin but INP-limited (~1e3 /m^3) — see #728.
    _cdnc_max_phys_per_m3 = 1.0e11
    inv_rho = 1.0 / jnp.maximum(air_density, eps_dt)
    qnc = jnp.clip(qnc, 0.0, _cdnc_max_phys_per_m3 * inv_rho)
    qni = jnp.clip(qni, 0.0, params.icemax * inv_rho)

    # Number-per-kg-of-air → per-m^3 at the scheme's API boundary.
    cdnc0 = qnc * air_density
    icnc0 = qni * air_density

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
    cdnc0 = jnp.maximum(cdnc0, minimum_CDNC(qc_in_cloud_kgm3, params))

    # ------------------------------------------------------------------
    # Step-start (t-1) thermodynamic fields — ECHAM section 1
    # ------------------------------------------------------------------
    # Saturation anchors are evaluated at the STEP-START state, exactly as
    # ECHAM evaluates zqsi/zqsw/zeta/the subsaturations at (ptm1, pqm1);
    # the provisional state enters only through the increments above.
    # ``es_water`` uses the LIQUID-WATER coefficients at ALL temperatures —
    # the Bergeron/WBF machinery depends on the water/ice saturation
    # *difference* below freezing, which degenerates to zero if es_water
    # switches to the ice coefficients below 0 °C.
    es_water = thermodynamics.saturation_vapor_pressure(
        temperature_m1, phase="water")
    es_ice = thermodynamics.saturation_vapor_pressure(
        temperature_m1, phase="ice")
    qsat_water, dqsw_dt = (
        thermodynamics.saturation_specific_humidity_and_derivative(
            temperature_m1, pressure, phase="water"))
    qsat_ice, dqsi_dt = (
        thermodynamics.saturation_specific_humidity_and_derivative(
            temperature_m1, pressure, phase="ice"))

    # Subsaturations for rain evaporation / snow sublimation: the NEGATIVE
    # relative deficits ``min(q/qs − 1, 0)`` (ECHAM zsusatw_evap/zicesub) —
    # the sublimation/evaporation chain needs the sign to produce a sink.
    subsat_wrt_ice = jnp.minimum(
        specific_humidity_m1 / jnp.maximum(qsat_ice, params.epsec) - 1.0, 0.0,
    )
    subsat_wrt_water = jnp.minimum(
        specific_humidity_m1 / jnp.maximum(qsat_water, params.epsec) - 1.0, 0.0,
    )

    # Rotstayn thermodynamic + vapour-diffusion factor (ECHAM zastbstw =
    # zast + zbst), the same chain the 1M rain evaporation uses:
    #   zast = Lv·(Lv/(Rv·T) − 1)/(T·ka),  zbst = Rv·T/(Dv·esw),
    # with Dv = 2.21/p and ka = 0.024 W/m/K.
    t_safe = jnp.maximum(temperature_m1, 1.0)
    zdv = 2.21 / jnp.maximum(pressure, params.epsec)
    zast = c.alhc * (c.alhc / (c.rv * t_safe) - 1.0) / (t_safe * 0.024)
    zbst = c.rv * temperature_m1 / jnp.maximum(zdv * es_water, params.epsec)
    thermo_term_water = zast + zbst

    # Bergeron/WBF diffusional-growth factor (ECHAM zeta, line 856 of
    # mo_cloud_micro_2m.f90). This is the ``peta`` that
    # ``threshold_vert_vel`` multiplies by (esw−esi)/esi·ICNC·r to get the
    # Korolev/Mazin threshold updraft [m/s]. The previous port fed a
    # dimensionless 0..1 saturation-ratio clip here, which is not the
    # reference quantity at all — the WBF gate and the lo2 phase decision
    # were miscalibrated by orders of magnitude (#667).
    zkair = 4.1867e-3 * (5.69 + 0.017 * (temperature_m1 - c.tmelt))
    zeta_a = (1.0 / jnp.maximum(specific_humidity_m1, params.eps)
              + lsdcp * c.alhc / (c.rv * t_safe ** 2))
    zeta_b = c.grav * (lvdcp * c.rd / c.rv / t_safe - 1.0) / (c.rd * t_safe)
    zeta_c = 1.0 / jnp.maximum(
        params.crhoi * c.alhs ** 2 / (jnp.maximum(zkair, params.epsec)
                                      * c.rv * t_safe ** 2)
        + params.crhoi * c.rv * t_safe
        / jnp.maximum(es_ice * zdv, params.epsec),
        params.epsec,
    )
    bergeron_eta = (zeta_a / zeta_b * zeta_c
                    * 4.0 * pi * params.crhoi * params.cap * inv_rho)

    # Updraft velocity [cm/s] from TKE (ECHAM zvervx; the large-scale
    # vertical-velocity contribution is not plumbed yet).
    updraft_velocity = params.fact_tke * jnp.sqrt(
        jnp.maximum(2.0 * tke, 0.0)) * 100.0

    # Dynamic viscosity of air (ECHAM zviscos, Pruppacher & Klett 13-18a).
    dynamic_viscosity = 4.1867e-3 * (
        5.69 + 0.017 * (temperature_m1 - c.tmelt))

    # Geometry / density helpers.
    pressure_thickness = air_density * params.grav * layer_thickness
    dp_over_g = pressure_thickness * c.rgrav
    zqrho = 1.3 * inv_rho                      # ECHAM zqrho = 1.3/ρ
    air_density_correction = zqrho ** 0.4      # ECHAM zaaa
    melt_mask = temperature_m1 > params.tmelt  # ECHAM ll_mlt (ptm1)

    # Heterogeneous mixed-phase INP [1/m³]: prefer the online JAM source
    # (immersion on prognostic dust/BC, #494); fall back to the DeMott
    # (2010) diagnostic on prescribed coarse aerosol where it is empty.
    demott_floor = demott2010_inp(temperature_m1, params.n_aer_coarse)
    n_inp = jnp.where(ice_nuclei > 0.0, ice_nuclei, demott_floor)

    # ------------------------------------------------------------------
    # The flux-coupled column sweep: ECHAM's column_processes loop
    # ------------------------------------------------------------------
    nlev_scan = temperature.shape[0]
    is_bottom_level = jnp.arange(nlev_scan) == (nlev_scan - 1)

    def _column_level_step(carry, level_in):
        """One level of the ECHAM column_processes loop (sections 3-8)."""
        (rain_flux, snow_flux, ice_flux, ice_flux_n,
         falling_ice_frac, precip_cover) = carry
        (cf_k, t_m1_k, q_m1_k, dT_up_k, dq_up_k, dqc_up_k, dqi_up_k,
         qc_m1_k, qi_m1_k, qc_run_k, qi_run_k,
         p_k, rho_k, inv_rho_k, dp_k, dpg_k, dz_k, adc_k, zqrho_k,
         cdnc0_k, icnc0_k,
         esw_k, esi_k, qsw_k, qsi_k, dqsw_k, dqsi_k,
         subice_k, subwat_k, thermo_k, eta_k, verv_k, visc_k, melt_k,
         act_cdnc_k, n_inp_k, inp_dep_k, is_bottom_k) = level_in

        zero_s = jnp.zeros_like(cf_k)

        # --- 4. Sedimentation of cloud ice (grid-mean) -----------------
        # Acts on the provisional grid-mean ice (ECHAM zxip1 = pxim1 +
        # ztmst·pxite, with the upstream increments folded into qi here).
        (zxip1, icnc_sedi, ice_flux, ice_flux_n, falling_ice_frac,
         _sedi_rate) = sedimentation_ice(
            cf_k, adc_k, dp_k, rho_k, inv_rho_k,
            jnp.maximum(qi_run_k, 0.0), icnc0_k,
            ice_flux, ice_flux_n, falling_ice_frac,
            dt, params,
        )
        sedi_tend = (zxip1 - qi_run_k) / dt

        # --- 3.1 Melting (fluxes + in-cloud ice) -----------------------
        # Runs after sedimentation (MG/PUMAS order, see docstring); the
        # running ice tendency is threaded THROUGH the routine so
        # ``pimlt = max(qi + ztmst·pxite, 0)`` reconstructs the ice left
        # AFTER sedimentation and the two sinks cannot claim the same
        # mass (#662 finding 2).
        (icnc_melt, _qmel, cdnc_melt,
         rain_flux, snow_flux, ice_flux, ice_flux_n,
         ice_tend_k, pimlt_k, psmlt_a, pximlt_k) = melting_snow_and_ice(
            melt_k, t_m1_k, qi_run_k, dp_k,
            icnc_sedi, lsdcp, lvdcp,
            icnc_sedi,
            jnp.array(0.0),  # qmel accumulator
            cdnc0_k,
            rain_flux, snow_flux, ice_flux, ice_flux_n,
            sedi_tend,
            dt,
            params,
        )

        # --- 3.2/3.3 Sublimation of snow/falling ice + rain evap -------
        precip_mask = precip_cover > 0.0        # ECHAM ll_precip
        falling_ice_mask_k = falling_ice_frac > 0.0  # ECHAM ll_falling_ice
        (ice_flux, ice_flux_n,
         xisub_k, sub_k, evp_k) = sublimation_snow_and_ice_evaporation_rain(
            precip_mask, falling_ice_mask_k,
            q_m1_k, t_m1_k,
            precip_cover, dp_k, dpg_k,
            subice_k, lsdcp,
            zqrho_k,          # ECHAM pqrho = zqrho = 1.3/ρ (was 1/ρ)
            qsi_k, inv_rho_k,
            snow_flux, rho_k,
            qsw_k, rain_flux,
            subwat_k, thermo_k,
            falling_ice_frac,
            ice_flux, ice_flux_n,
            dt,
            params,
        )

        # --- In-cloud condensate prep + clear-sky evaporation ----------
        # ECHAM 1310-1385. The step's total non-microphysical increments
        # (upstream + sedimentation + melting) are split ECHAM-style: in
        # cloudy cells positive increments enter the in-cloud state at
        # their grid-mean magnitude while their clear-sky share
        # ``(1−paclc)·max(增, 0)`` evaporates (the two add back to the full
        # grid-mean increment); negative increments deplete in-cloud
        # values clamped at zero; in cloud-FREE cells the entire
        # condensate — carried plus incremented — evaporates. This is the
        # clear-sky condensate sink the scheme previously lacked (#667):
        # a cf=0 cell holding qc/qi now returns it to vapour with the
        # matching latent cooling instead of carrying it untouchable.
        ll_cc = cf_k > params.clc_min
        cf_safe = jnp.maximum(cf_k, params.clc_min)

        zxidt = dqi_up_k + dt * ice_tend_k
        zxldt = dqc_up_k + pximlt_k + pimlt_k
        ll_ipos = zxidt > 0.0
        ll_lpos = zxldt > 0.0
        zxidtstar = jnp.maximum(zxidt, 0.0)
        zxldtstar = jnp.maximum(zxldt, 0.0)

        zxib = jnp.where(ll_cc, qi_m1_k / cf_safe, 0.0)
        incr_i = jnp.where(ll_ipos, zxidt,
                           jnp.maximum(zxidt / cf_safe, -zxib))
        zxib = zxib + jnp.where(ll_cc, incr_i, 0.0)
        zxim1evp = (jnp.where(ll_cc, 0.0, qi_m1_k)
                    + jnp.where(jnp.logical_and(~ll_cc, ~ll_ipos),
                                zxidt, 0.0))

        zxlb = jnp.where(ll_cc, qc_m1_k / cf_safe, 0.0)
        incr_l = jnp.where(ll_lpos, zxldt,
                           jnp.maximum(zxldt / cf_safe, -zxlb))
        zxlb = zxlb + jnp.where(ll_cc, incr_l, 0.0)
        zxlm1evp = (jnp.where(ll_cc, 0.0, qc_m1_k)
                    + jnp.where(jnp.logical_and(~ll_cc, ~ll_lpos),
                                zxldt, 0.0))

        zxievap = (1.0 - cf_k) * zxidtstar + zxim1evp
        zxlevap = (1.0 - cf_k) * zxldtstar + zxlm1evp

        zxib = jnp.maximum(zxib, 0.0)
        zxlb = jnp.maximum(zxlb, 0.0)
        zxilb = zxib + zxlb

        # --- Phase decision lo2 (ECHAM section 4 end) ------------------
        # Ice-vs-liquid regime from the Korolev/Mazin threshold updraft,
        # computed on the post-sedimentation ice. ``zrice`` is the
        # volume-mean radius in METRES (ECHAM prid); the shared helper
        # carries the clip + Schumann conversion so this copy cannot
        # drift from precip.py / deposition_freezing.py (#725).
        ice_gm3 = 1000.0 * zxip1 * rho_k / cf_safe
        zrice = ice_volume_mean_radius(ice_gm3, icnc_melt, params)
        zvervmax = threshold_vert_vel(
            sat_vap_pres_water=esw_k, sat_vap_pres_ice=esi_k,
            icnc=icnc_melt, ice_radius=zrice, eta=eta_k, params=params)
        lo2 = jnp.logical_or(
            t_m1_k < params.cthomi,
            jnp.logical_and(t_m1_k < params.tmelt,
                            0.01 * verv_k < zvervmax),
        )

        # --- 5. Condensation source zqcdif → zcnd / zdep ---------------
        # The Sundqvist moisture-convergence closure (ECHAM 1389-1470):
        # the humidity increment this step, minus the saturation-humidity
        # change implied by the temperature increment (damped by the
        # warming feedback), condenses into the cloudy fraction.
        zlc = jnp.where(lo2, lsdcp, lvdcp)
        zqsm1 = jnp.where(lo2, qsi_k, qsw_k)
        zdqsdt = jnp.where(lo2, dqsi_k, dqsw_k)

        zdtdt = (dT_up_k
                 - lvdcp * (evp_k + zxlevap)
                 - (lsdcp - lvdcp) * (psmlt_a + pximlt_k + pimlt_k)
                 - lsdcp * (sub_k + zxievap + xisub_k))
        zqp1 = jnp.maximum(q_m1_k + dq_up_k, 0.0)
        ztp1 = t_m1_k + zdtdt

        zdqsat = (zdtdt
                  + cf_k * (zlc * dq_up_k
                            + lvdcp * (evp_k + zxlevap)
                            + lsdcp * (sub_k + zxievap + xisub_k)))
        zdqsat = (zdqsat * zdqsdt
                  / (1.0 + cf_k * zlc * zdqsdt))
        zqcdif = (dq_up_k - zdqsat) * cf_k
        # Bounds: dissipation limited to the available condensate,
        # condensation to (almost) the available vapour (ECHAM qsec·zqp1,
        # qsec = 1 − cqtmin ≈ xsec).
        zqcdif = jnp.clip(zqcdif, -zxilb * cf_k, params.xsec * zqp1)

        ll_dissip = zqcdif < 0.0
        zifrac = jnp.clip(zxib / jnp.maximum(zxilb, params.epsec), 0.0, 1.0)
        frac = jnp.where(ll_dissip, zifrac, 1.0)
        zcnd0 = jnp.where(ll_dissip, zqcdif * (1.0 - zifrac), 0.0)
        if params.nic_cirrus == 2:
            # ECHAM: zdep = zqinucl·zifrac — the Kärcher-Lohmann
            # nucleated vapour, which jcm does not compute (#552).
            zdep0 = zero_s
        else:
            zdep0 = zqcdif * frac
        ll_growth_liq = jnp.logical_and(~ll_dissip, ~lo2)
        zdep0 = jnp.where(ll_growth_liq, 0.0, zdep0)
        # Saturation adjustment for water condensation (ECHAM #485): in
        # the liquid-growth regime the full zqcdif condenses.
        zcnd0 = jnp.where(ll_growth_liq, zqcdif, zcnd0)

        # --- 5.4 Supersaturation corrections ---------------------------
        (zcnd, zdep, ztp1tmp, zqp1tmp, zqsp1tmp,
         _zvervmax_dep) = mixed_phase_deposition_and_corrections(
            p_k, icnc_melt, q_m1_k, cf_k,
            esi_k, esw_k,
            eta_k,
            zero_s,             # tompkins_genti
            lsdcp, lvdcp,
            zqp1, zqsm1,
            rho_k, ztp1,
            zxievap,
            zxip1,
            zero_s,             # detrainment tendency (folded into qi)
            verv_k,
            zcnd0, zdep0,       # INOUT, seeded from section 5
            dt,
            params,
        )

        # --- 5.5 In-cloud water update + activation / nucleation -------
        (cloud_flag, icnc_u, _nucl, cdnc_u, paclc, zxib, zxlb,
         cdnc_min_k) = update_in_cloud_water(
            p_k,
            act_cdnc_k,
            zcnd, zdep,
            zero_s, zero_s,     # Tompkins sources
            inp_dep_k,          # newly_formed_ice: cirrus dep-INP (#494)
            zqp1tmp, zqsp1tmp,
            rho_k,
            zrice,              # prid: volume-mean ice radius [m]
            t_m1_k,             # ptm1 (activation gates on step-start T)
            ll_cc,
            icnc_melt,
            zero_s,             # nucleation_rate accumulator
            cdnc_melt,
            cf_k,
            zxib, zxlb,
            dt,
            params,
        )

        # --- 6.1 Homogeneous freezing below cthomi ---------------------
        frz_below = ztp1tmp <= params.cthomi
        (icnc_f, _qfre, cdnc_f, zfrl, zxib, zxlb) = freezing_below_238K(
            frz_below, paclc, cdnc_min_k,
            icnc_u,
            zero_s,             # droplet_freezing_rate accumulator
            cdnc_u,
            zero_s,             # freezing_rate accumulator (pfrl)
            zxib, zxlb,
            dt,
            params.cqtmin,
        )

        # --- 6.2 Heterogeneous mixed-phase freezing + WBF --------------
        # ECHAM ll_mxphase_frz: liquid present, mixed-phase window on the
        # corrected temperature, droplets at/above the floor, cloud
        # present. The jcm INP substitution (JAM immersion / DeMott
        # fallback) freezes droplets up to the INP number, moving number,
        # mass AND fusion heat together (#662 finding 3).
        ll_mxfrz = (
            (zxlb > params.cqtmin)
            & (ztp1tmp < params.tmelt)
            & (ztp1tmp > params.cthomi)
            & (cdnc_f >= cdnc_min_k)
            & cloud_flag
        )
        icnc_het = jnp.where(
            jnp.logical_and(ll_mxfrz, n_inp_k > icnc_f), n_inp_k, icnc_f)
        new_crystals = jnp.maximum(icnc_het - icnc_f, 0.0)
        mean_droplet_mass = jnp.where(
            cdnc_f > params.epsec,
            zxlb * rho_k / jnp.maximum(cdnc_f, params.epsec),
            0.0,
        )
        frozen_mass = jnp.minimum(
            new_crystals * mean_droplet_mass * inv_rho_k, zxlb)
        zxib = zxib + frozen_mass
        zxlb = zxlb - frozen_mass
        cdnc_h = jnp.where(
            ll_mxfrz,
            jnp.maximum(cdnc_f - new_crystals, params.cqtmin),
            cdnc_f,
        )
        # Grid-mean freezing ledger (ECHAM pfrl): only the het leg is
        # converted here — freezing_below_238K already area-weights
        # internally (pfrl += pxlb·paclc).
        zfrl = zfrl + frozen_mass * paclc

        # WBF with the threshold updraft recomputed from the
        # post-freezing in-cloud ice (ECHAM 1580-1594).
        # ``zxib`` is already in-cloud, so no cloud-fraction division here.
        ice_gm3_wbf = 1000.0 * zxib * rho_k
        zrice_wbf = ice_volume_mean_radius(ice_gm3_wbf, icnc_het, params)
        zvervmax_wbf = threshold_vert_vel(
            sat_vap_pres_water=esw_k, sat_vap_pres_ice=esi_k,
            icnc=icnc_het, ice_radius=zrice_wbf, eta=eta_k, params=params)
        ll_wbf = (
            ll_mxfrz
            & cloud_flag
            & (zdep > 0.0)
            & (zxlb > 0.0)
            & (0.01 * verv_k < zvervmax_wbf)
        )
        # ``WBF_process`` computes the grid-mean transfer once
        # (pxlb·paclc/dt) and reports it three ways — liquid debit, ice
        # credit, fusion warming. All three seed the ledger together:
        # they are one transfer, and the enthalpy budget only closes if
        # the mass and the heat travel with it (#662 finding 1).
        (cdnc_w, zxlb, zxib,
         wbf_liq_tend, wbf_ice_tend, wbf_dtedt) = WBF_process(
            ll_wbf, paclc, lsdcp, lvdcp,
            cdnc_h, zxlb, zxib,
            zero_s, zero_s, zero_s,
            dt,
            params,
        )
        wbf_transfer_k = -dt * wbf_liq_tend   # grid-mean kg/kg moved liq→ice

        # --- 7. Precipitation geometry + Marshall-Palmer inversion -----
        # zclcstar: the cloud ∩ precipitation overlap that weights
        # accretion by rain/snow from above (#685 — was paclc, i.e. the
        # assumption that precip always covers at least the cloud).
        zclcstar = jnp.minimum(paclc, precip_cover)
        # zauloc: layer-depth-dependent fraction of the box in which
        # newly formed rain participates in accretion (#685 — was 1).
        zauloc = jnp.clip(params.cauloc / 5000.0 * dz_k,
                          params.clmin, params.clmax)

        zxlb = jnp.maximum(zxlb, 1.0e-20)
        zxib = jnp.maximum(zxib, 1.0e-20)
        zmlwc_k = zxlb          # in-cloud liquid before rain formation
        zmiwc_k = zxib          # in-cloud ice before snow formation

        # Rain/snow water content diagnosed from the carry fluxes by the
        # Marshall-Palmer inversions (Roeckner et al. 2003 eqs. 10.70 /
        # 10.74; ECHAM 1638-1654). This replaces the dead ``qr``/``qs``
        # tracer reads (#662 finding 5): ECHAM carries no rain/snow
        # tracers — the accretion "rain from above" is the flux the
        # levels above just produced, inverted to a mixing ratio.
        # Double-where guards on the fractional powers (infinite
        # derivative at 0 base under the masked branch).
        ll_pre = precip_cover > params.epsec
        rain_present = jnp.logical_and(ll_pre, rain_flux > params.cqtmin)
        snow_present = jnp.logical_and(ll_pre, snow_flux > params.cqtmin)
        zclcpre_safe = jnp.maximum(precip_cover, params.epsec)
        zqrho_sqrt = jnp.sqrt(zqrho_k)
        zxrp1_base = jnp.where(
            rain_present,
            jnp.maximum(rain_flux, params.cqtmin)
            / (12.45 * zclcpre_safe * zqrho_sqrt),
            1.0,
        )
        zxrp1 = jnp.where(rain_present, zxrp1_base ** (8.0 / 9.0), 0.0)
        zxsp1_base = jnp.where(
            snow_present,
            jnp.maximum(snow_flux, params.cqtmin)
            / (params.cvtfall * zclcpre_safe),
            1.0,
        )
        zxsp1 = jnp.where(snow_present, zxsp1_base ** (1.0 / 1.16), 0.0)

        # --- 7.1 Warm-rain formation (KK2000) --------------------------
        # ECHAM ll_prcp_warm: cloud present, liquid present, droplets
        # at/above the activation floor — NO temperature condition
        # (coalescence operates on supercooled liquid too).
        ll_warm = (
            cloud_flag
            & (zxlb > params.cqtmin)
            & (cdnc_w >= cdnc_min_k)
        )
        (cdnc_p, zxlb, _mratepr, zrpr, _rprn,
         auto_only_k, accr_only_k) = precip_formation_warm(
            ll_warm,
            zauloc,
            paclc,
            zclcstar,
            rho_k,
            zxrp1,
            cdnc_min_k,
            cdnc_w,
            zxlb,
            dt,
            params,
        )

        # --- 7.2 Cold precipitation formation --------------------------
        # Gate is ECHAM's: the cloud flag only — the internal
        # ``zxib > cqtmin`` check runs on the CURRENT in-cloud ice, so
        # ice deposited/frozen/WBF-transferred THIS step meets its
        # aggregation and riming sinks in the same step (#686).
        (icnc_c, cdnc_c, _mrateps, zxib, zxlb,
         _sprn, zsacl, _sacln, _msnowacl, zspr) = precip_formation_cold(
            cloud_flag,
            zauloc,
            paclc,
            zclcstar,
            zqrho_k,            # ECHAM pqrho = 1.3/ρ (was 1/ρ)
            inv_rho_k,
            ztp1tmp,
            visc_k,
            zxsp1,
            rho_k,
            cdnc_min_k,
            icnc_het,
            cdnc_p,
            zero_s,
            zxib, zxlb,
            dt,
            params,
        )

        # --- 7.3 Update precipitation fluxes ---------------------------
        (precip_cover, rain_flux, snow_flux, snow_melt_b,
         _pfevapr, _pfrain, _pfsnow, _pfsubls) = update_precip_fluxes(
            paclc, dp_k,
            evp_k, lsdcp, lvdcp,
            zrpr, zsacl, zspr,
            sub_k, ztp1tmp,
            # ECHAM folds the sedimenting ice flux into the snow flux
            # ONLY at the bottom level (Fortran ``kk == klev`` gate).
            jnp.where(is_bottom_k, ice_flux, 0.0),
            precip_cover, rain_flux, snow_flux, jnp.array(0.0),
            dt,
            params,
        )
        psmlt_k = psmlt_a + snow_melt_b

        # --- 8-prep: phase-presence flags for the effective radii ------
        # ECHAM ll_liqcl/ll_icecl (1755-1760): actual condensate + number
        # above its floor — NOT a temperature split.
        ll_liqcl_k = jnp.logical_and(zxlb > params.epsec,
                                     cdnc_c >= cdnc_min_k)
        ll_icecl_k = jnp.logical_and(zxib > params.epsec,
                                     icnc_c >= params.icemin)

        # Per-level flux profiles for downstream (COSP/CloudSat)
        # diagnostics: the grid-mean rain / frozen fluxes LEAVING this
        # layer. The frozen profile adds the sedimenting cloud-ice flux
        # at interior levels; at the bottom ``update_precip_fluxes`` has
        # already folded it into snow (adding again would double-count).
        frozen_flux_k = snow_flux + jnp.where(is_bottom_k, 0.0, ice_flux)

        carry_out = (rain_flux, snow_flux, ice_flux, ice_flux_n,
                     falling_ice_frac, precip_cover)
        level_out = (
            zcnd, zdep, zfrl, zrpr, zsacl, zspr,
            pimlt_k, pximlt_k, psmlt_k,
            xisub_k, sub_k, evp_k,
            zxlevap, zxievap,
            ice_tend_k, wbf_liq_tend, wbf_ice_tend, wbf_dtedt,
            zxib, zxlb, paclc, icnc_c, cdnc_c, cdnc_min_k, ztp1tmp,
            zmlwc_k, zmiwc_k,
            auto_only_k, accr_only_k,
            rain_flux, frozen_flux_k,
            wbf_transfer_k, ll_liqcl_k, ll_icecl_k,
        )
        return carry_out, level_out

    scan_inputs = (
        cloud_fraction, temperature_m1, specific_humidity_m1,
        dT_up, dq_up, dqc_up, dqi_up,
        qc_m1, qi_m1, qc, qi,
        pressure, air_density, inv_rho, pressure_thickness, dp_over_g,
        layer_thickness, air_density_correction, zqrho,
        cdnc0, icnc0,
        es_water, es_ice, qsat_water, qsat_ice, dqsw_dt, dqsi_dt,
        subsat_wrt_ice, subsat_wrt_water, thermo_term_water,
        bergeron_eta, updraft_velocity, dynamic_viscosity, melt_mask,
        activated_cdnc, n_inp, ice_nuclei_deposition, is_bottom_level,
    )

    zero_scalar = jnp.array(0.0, dtype=qc.dtype)
    init_carry = (zero_scalar, zero_scalar, zero_scalar,
                  zero_scalar, zero_scalar, zero_scalar)

    _final_carry, scan_outs = jax.lax.scan(
        _column_level_step, init_carry, scan_inputs,
    )
    (condensation_rate, deposition_rate, freezing_rate,
     rain_formation, snow_accretion, snow_formation,
     pimlt_per_level, pximlt_per_level, psmlt_per_level,
     ice_sublim, snow_sublim, rain_evap,
     xlevap, xievap,
     ice_tendency_scan, liq_tend_wbf, ice_tend_wbf, dtedt_wbf,
     in_cloud_ice_final, in_cloud_liquid_final, paclc_final,
     icnc_final, cdnc_final, cdnc_min_final, ztp1tmp_all,
     zmlwc, zmiwc,
     autoconv_only, accretion_only,
     rain_flux_profile, snow_flux_profile,
     wbf_transfer, ll_liqcl, ll_icecl) = scan_outs

    # Surface precipitation fluxes: the carry at the bottom of the column.
    (surface_rain_flux, surface_snow_flux, _, _, _, _) = _final_carry

    # ------------------------------------------------------------------
    # 8. update_tendencies_and_important_vars: the ECHAM6 accounting step
    # ------------------------------------------------------------------
    cloud_fraction_in = cloud_fraction

    (
        cloud_fraction_final,
        dqdt, dtedt, dqidt, dqcdt,
        dqncdt_perkg, dqnidt_perkg,
        _incloud_liq, _incloud_ice,
        liq_eff_radius, ice_eff_radius,
        zdxlcor, zdxicor,
    ) = update_tendencies_and_important_vars(
        icnc=icnc_final,
        cdnc=cdnc_final,
        # ECHAM's pxim1/pxlm1 are the grid-mean condensate the increments
        # accumulate on. Here the upstream increments are already inside
        # ``qi``/``qc`` and the seeds below carry only the scheme's own
        # tendencies, so the reconstruction pxim1 + ztmst·(upstream+own)
        # + increments equals ``qi + ztmst·own + increments`` — the same
        # number, with the negative-mass guard testing the actual
        # end-of-step grid-mean state against ``ccwmin`` (#662 finding 6).
        ice_mmr_prev=qi,
        liq_mmr_prev=qc,
        # ECHAM convention: pxtm1_cdnc / pxtm1_icnc are the previous-step
        # tracer values in per-kg-of-air (the working cdnc/icnc are
        # per-m³, so the tendency subtracts per-kg from per-m³·1/ρ).
        tracer_tm1_cdnc=qnc,
        tracer_tm1_icnc=qni,
        condensation_rate=condensation_rate,
        deposition_rate=deposition_rate,
        rain_evap_mmr=rain_evap,
        freezing_rate=freezing_rate,
        tompkins_ice=zero,
        tompkins_liq=zero,
        incloud_ice_melt=pimlt_per_level,
        lsdcp=lsdcp,
        lvdcp=lvdcp,
        air_density=air_density,
        inv_air_density=inv_rho,
        rain_formation=rain_formation,
        snow_accretion=snow_accretion,
        snow_formation=snow_formation,
        # Clear-sky evaporation of cloud ice / liquid (ECHAM zxievap /
        # zxlevap): the in-scheme sink for condensate in cloud-free cells
        # and for the clear-sky share of upstream increments (#667).
        cloud_ice_evap=xievap,
        ice_flux_melt=pximlt_per_level,
        pxitec=zero,
        pxlevap=xlevap,
        pxltec=zero,
        # Falling-ice sublimation (ECHAM zxisub): deducted from the
        # falling ice flux, so it must re-enter the column as vapour.
        pxisub=ice_sublim,
        snow_sublimation_mmr=snow_sublim,
        snow_melt=psmlt_per_level,
        cloud_ice_in_cloud=in_cloud_ice_final,
        cloud_liquid_in_cloud=in_cloud_liquid_final,
        temp_tmp=ztp1tmp_all,
        liquid_cloud_flag=ll_liqcl,
        ice_cloud_flag=ll_icecl,
        cloud_fraction=paclc_final,
        specific_humidity_tendency=zero,
        # WBF reports its transfer as a tendency, so its three legs seed
        # the three INOUT accumulators (heat, ice credit, liquid debit).
        temp_tendency=dtedt_wbf,
        # ECHAM folds ice sedimentation AND melting into pxite before
        # this ledger (section 4), so the sweep's combined per-level
        # tendency arrives as a seed; WBF's ice credit joins it.
        ice_tendency=ice_tendency_scan + ice_tend_wbf,
        liq_tendency=liq_tend_wbf,
        tracer_tendency_cdnc=zero,
        tracer_tendency_icnc=zero,
        incloud_liq_before_rain=zmlwc,
        incloud_ice_before_snow=zmiwc,
        dt=dt,
        params=params,
    )

    # ECHAM carries NO rain/snow mixing-ratio tracers: precipitation
    # leaves each level exclusively through the prfl/psfl fluxes and the
    # surface fluxes are the outputs (review finding 2.18).
    dqrdt = jnp.zeros_like(qc)
    dqsdt = jnp.zeros_like(qc)

    # The microphysics may REMOVE cloud, never create it. ECHAM's write-back
    # to ``paclc`` exists to clear cells the scheme has just emptied of both
    # condensates; the upward branch of ``update_in_cloud_water`` — which
    # sets cf = clip(RH, 0.01, 1) wherever a clear cell has any condensation
    # or deposition — is a second, RH-based cloud-cover closure, and
    # ``SundqvistCloudFraction`` is the one this stack uses. Publishing the
    # raw value substitutes it: an ice-supersaturated stratospheric column
    # above ``cloud_top_pressure_pa``, which Sundqvist deliberately reports
    # as cloud-free, comes back overcast (cf = 1) on ~1e-6 kg/kg of ice, and
    # COSP, AeroCom and the JAM cloud-borne / aqueous / wetdep terms all read
    # it. Clipping to the incoming cover keeps the emptying behaviour and
    # drops the closure substitution.
    cloud_fraction_final = jnp.minimum(cloud_fraction_final, cloud_fraction_in)

    # update_tendencies' tracer_tendency_{cdnc,icnc} is already in per-kg-
    # of-air per second once qnc/qni (per-kg) are passed as the tm1
    # tracers.
    dqncdt = dqncdt_perkg
    dqnidt = dqnidt_perkg

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
    # warm chain (KK2000 autoconversion + accretion, ``rain_formation``)
    # and snow melt. Their ratio is the model's warm-rain fraction, the
    # CloudSat-style observable that constrains the warm-rain parameters
    # (ccraut, and the SPA activation fit through CDNC). Both are
    # per-step grid-mean mixing-ratio increments, so the column flux is
    # sum(dq * rho * dz) / dt.
    air_mass = air_density * layer_thickness  # [kg/m^2] per level
    rain_formation_warm = jnp.sum(rain_formation * air_mass) / dt
    rain_from_melt = jnp.sum(psmlt_per_level * air_mass) / dt

    # AeroCom process rates [kg/m^2/s], same column-integral convention.
    # autoconv and accretn split the warm chain into its two pathways;
    # wbf is the grid-mean liquid mass converted to ice by the
    # Wegener-Bergeron-Findeisen process.
    autoconv_rate_col = jnp.sum(autoconv_only * air_mass) / dt
    accretion_rate_col = jnp.sum(accretion_only * air_mass) / dt
    wbf_rate_col = jnp.sum(wbf_transfer * air_mass) / dt

    # Per-level precip process rates for JAM wet scavenging (#499), grid
    # mean [kg/kg/s]. Formation is the full condensate→precip ledger the
    # flux update integrates: the warm chain, riming (``zsacl``) and the
    # cold snow formation. Evaporation is rain evap + snow sublimation;
    # the falling cloud-ice sublimation is deliberately excluded — the
    # sedimenting ice flux is not a scavenging carrier.
    precip_formation_rate = (
        rain_formation + snow_accretion + snow_formation
    ) / dt
    precip_evaporation_rate = (rain_evap + snow_sublim) / dt

    # Negative-mass-repair diagnostic (#689): the column-integrated
    # latent heating of the zdxlcor/zdxicor guard [W/m²]. The repair is
    # ECHAM-faithful and thermodynamically consistent, but sign-definite
    # — every condensate undershoot the dycore leaves becomes warming +
    # drying, never the reverse — so its magnitude and geographic pattern
    # must be observable in a run rather than silently folded into
    # dtedt/dqdt. Positive values = spurious heating from repairing
    # negative/sub-ccwmin condensate.
    negative_mass_repair = jnp.sum(
        (c.alhc * zdxlcor + c.alhs * zdxicor) * air_mass)

    # NOTE the (nlev,) rain / frozen flux profiles stay LAST: call sites and
    # tests unpack them positionally from the end (``*_, rain_b, snow_b``),
    # so new outputs are inserted before them, not appended.
    return tendencies, surface_rain_flux, surface_snow_flux, \
        liq_eff_radius, ice_eff_radius, rain_formation_warm, rain_from_melt, \
        autoconv_rate_col, accretion_rate_col, wbf_rate_col, \
        precip_formation_rate, precip_evaporation_rate, cloud_fraction_final, \
        negative_mass_repair, \
        rain_flux_profile, snow_flux_profile


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------



class Lohmann2MMicrophysics(PhysicsTerm):
    """ECHAM 2-moment cloud microphysics (Lohmann/Seifert-Beheng-style) term.

    Drop-in 2M alternative to :class:`Echam1MMicrophysics`. Declares the
    prognostic-tracer set (``qc``, ``qi``, ``qnc``, ``qni``) — the
    ``qnc`` / ``qni`` number concentrations are stored per kg of air with
    ``nondimensionalize=False`` so the modal/nodal converters don't apply
    the gram/kg scaling that mass mixing ratios get.

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
        # into ``clouds.qc/qi``. The provisional state is what the returned
        # tendencies are relative to (the host's additive sum with the
        # upstream tendencies telescopes back to the correct final state),
        # while the STEP-START state supplies ECHAM's (ptm1, pqm1, pxlm1,
        # pxim1) anchors: saturation evaluates there, and the differences
        # play the role of the accumulated tendencies in the condensation
        # closure and the clear-sky-evaporation split (see
        # ``cloud_microphysics_2m``). Falls back to the step-start state if
        # no upstream term seeded ``thermo_run``.
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
        # Step-start tracers — the baseline the upstream increments in
        # ``clouds.qc``/``clouds.qi`` accumulated on (ECHAM pxlm1/pxim1).
        qc_m1 = state.tracers.get("qc", zeros)
        qi_m1 = state.tracers.get("qi", zeros)

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

        # The core owns grid-scale condensation now (the ECHAM section-5
        # zqcdif closure + supersaturation corrections run inside the
        # sweep), so there is NO external Sundqvist condensation bolt-on
        # any more. The bolt-on dated from when the internal adjustment
        # was suppressed ~1e6x by the c.ak/zdqsdt transcription bugs
        # (#667): with those fixed, summing both would remove
        # supersaturation twice per step with double the latent heating.
        (tend_all, surface_rain_flux, surface_snow_flux,
         r_eff_liq_all, r_eff_ice_all, rain_formation_warm, rain_from_melt,
         autoconv_all, accretion_all, wbf_all,
         precip_form_all, precip_evap_all, cloud_fraction_all,
         negative_mass_repair_all,
         rain_flux_all, snow_flux_all) = jax.vmap(
            cloud_microphysics_2m,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     None, None, 1, 1, 1, 1),
            out_axes=(0,) * 16,
        )(
            temperature_in, specific_humidity_in, pressure_full,
            qc_interim, qi_interim, qnc, qni,
            cloud_fraction, air_density, layer_thickness, tke,
            activated_cdnc, ice_nuclei, ice_nuclei_deposition, dt, params_2m,
            state.temperature, state.specific_humidity, qc_m1, qi_m1,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=tend_all.dtedt.T,
            specific_humidity=tend_all.dqdt.T,
            tracers={
                "qc": tend_all.dqcdt.T,
                "qi": tend_all.dqidt.T,
                "qnc": tend_all.dqncdt.T,
                "qni": tend_all.dqnidt.T,
            },
        )

        # Stash current-step qnc/qni as tm1 for the next step's
        # update_tendencies_and_important_vars; expose surface precip
        # diagnostics from the lax.scan.
        clouds_next = clouds.copy(
            # ECHAM writes the post-microphysics cloud fraction back to
            # ``paclc``: cells the scheme has just emptied of both condensates,
            # or driven below ``clc_min``, are no longer cloudy. Radiation and
            # the aerosol cloud-borne partition read this, and must see the
            # cloud the step actually leaves behind.
            cloud_fraction=cloud_fraction_all.T,
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
            # Column-integrated latent heating of the negative-mass
            # repair [W/m²] (#689): sign-definite spurious warming from
            # returning sub-ccwmin/negative condensate to vapour.
            negative_mass_repair=negative_mass_repair_all,
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
