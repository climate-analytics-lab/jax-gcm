"""Lohmann 2M column-sweep orchestrator and composable physics term.

``cloud_microphysics_2m`` runs the full two-moment process chain over a
column as one flux-coupled top-down ``lax.scan`` — the flux-coupling
structure of ECHAM's ``column_processes`` loop, but with a process order
that matches neither ECHAM nor CAM exactly (see the term docstring) — and
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
from jcm.physics.clouds.cloud_data import CLOUD_OUTPUT_ATTRS
from jcm.physics.diagnostics.moist_air_state import advance_thermo_run
from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

from ..lohmann_2m_params import CloudParams2M
from ..cloud_utils import (
    air_dynamic_viscosity,
    detrained_ice_crystal_number,
    ice_fall_speed_air_density_factor,
    ice_volume_mean_radius_from_temperature,
    ice_volume_mean_radius_schumann,
    latent_heat_over_cp,
    minimum_CDNC,
    threshold_vert_vel,
    turbulent_updraft_velocity,
)
from .types import (
    HeterogeneousFreezingAerosol,
    MicrophysicsTendencies_2M,
    ScavengingLedger,
)
from .sedimentation_melt import melting_snow_and_ice, sedimentation_ice
from .deposition_freezing import (
    demott2010_inp,
    freezing_below_238K,
    het_mxphase_freezing,
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
    ice_nuclei: jnp.ndarray,        # (nlev,)  1/m³   external immersion INP for the DeMott closure (none in-tree)
    ice_nuclei_deposition: jnp.ndarray,  # (nlev,) 1/m³  deposition INP (pnicex; read only at nic_cirrus=2)
    dt: jnp.ndarray,                # scalar   seconds
    params: CloudParams2M,          # tunable parameters
    temperature_m1: jnp.ndarray | None = None,        # (nlev,) K   step-start T (ECHAM ptm1)
    specific_humidity_m1: jnp.ndarray | None = None,  # (nlev,)     step-start q (ECHAM pqm1)
    qc_m1: jnp.ndarray | None = None,                 # (nlev,)     step-start qc (ECHAM pxlm1)
    qi_m1: jnp.ndarray | None = None,                 # (nlev,)     step-start qi (ECHAM pxim1)
    detrained_qc: jnp.ndarray | None = None,  # (nlev,) kg/kg per step: liquid detrainment (ECHAM ztmst·pxtecl)
    detrained_qi: jnp.ndarray | None = None,  # (nlev,) kg/kg per step: ice detrainment (ECHAM ztmst·pxteci)
    freezing_aerosol: HeterogeneousFreezingAerosol | None = None,  # (nlev,) leaves: HAM freezing inputs
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
    "level-independent" processes out of the sweep would sever exactly those
    couplings: rain/snow-from-above accretion would run on tracers that do
    not yet exist (#662 finding 5), the precipitation-cover geometry
    ``zclcstar`` would be unavailable (#685), and ice created mid-step would
    never meet its aggregation sink (#686).

    Per level, ordered by ECHAM's section numbering (numbers = Fortran
    comments) but matching neither reference exactly — the sediment→melt
    sweep is MG/PUMAS's (3.1) and warm precipitation runs after condensation
    and activation (7.1):

      4.    Ice sedimentation (:func:`sedimentation_ice`) of the ice
            present BEFORE this step's convective detrainment (ECHAM
            ``zxip1 = pxim1 + ztmst·pxite``, 1227-1248), after which the
            crystal number of the detrained ice ``znidetr`` joins the ICNC
            (1251-1252), then
      3.1   melting of snow / falling ice / in-cloud ice
            (:func:`melting_snow_and_ice`). NOTE this sediment→melt order
            is deliberately MG/PUMAS's (micro_pumas_v1: sediment 3093 →
            melt 3293), not ECHAM's melt→sediment; the melt acts on the
            post-sedimentation ice via the threaded tendency so the two
            sinks cannot claim the same mass (#662 finding 2).
      3.2/3 Snow/ice sublimation + rain evaporation on the incoming
            fluxes (:func:`sublimation_snow_and_ice_evaporation_rain`).
      (4b)  Phase decision ``lo2`` (ECHAM 1276-1298), which also
            re-splits the detrained condensate into ice and liquid with
            the matching latent-heat correction (1301-1317); then the
            in-cloud condensate prep with clear-sky evaporation
            ``zxlevap``/``zxievap`` (1319-1385): condensate in cells with
            no cloud, and the clear-sky share of positive increments,
            evaporates back to vapour.
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
      6.2   Heterogeneous mixed-phase freezing and the WBF process with
            the Korolev/Mazin threshold updraft recomputed from the
            post-freezing ice (:func:`WBF_process`). With
            ``freezing_aerosol`` (the HAM freezing inputs of a prognostic
            aerosol, JAM) the freezing is ECHAM-HAM's contact + immersion
            rates (:func:`het_mxphase_freezing`, F 2675-2840); without it,
            jcm's aerosol-free closure that freezes droplets up to the
            DeMott (2010) INP number.
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
    etc.), which is what the returned tendencies are relative to; ``qc``
    and ``qi`` include this step's convective detrainment. The optional
    ``*_m1`` arguments are the step-start state (ECHAM ``ptm1``/``pqm1``/
    ``pxlm1``/``pxim1``): saturation anchors evaluate there, and the
    differences ``(x - x_m1)`` — less the detrained condensate for ``qc``
    and ``qi`` — play the role of ECHAM's accumulated tendencies
    ``ztmst·ptte``/``ztmst·pqte``/``ztmst·pxlte``/``ztmst·pxite`` in the
    condensation closure, the clear-sky-evaporation split and (for ice)
    the sedimentation input.

    What those increments hold in the composed ECHAM stack
    (``echam_physics``) differs by variable. ``temperature`` and
    ``specific_humidity`` come from the running thermodynamic view
    ``thermo_run``, so ``dT``/``dq`` carry this step's vertical-diffusion
    and convection increments. ``qc``/``qi`` come from ``clouds.qc/qi``,
    which ``SundqvistCloudFraction`` snapshots from ``thermo_run`` AHEAD of
    vertical diffusion (the factory composes the cover term before
    ``TteTkeVerticalDiffusion``); vertical diffusion advances only
    ``thermo_run``, never ``clouds``, and after it only the convection term
    adds to ``clouds.qc/qi`` — its detrainment. The pure condensate
    increments are therefore zero in that stack (up to the convection
    term's clip of ``clouds.qc/qi`` at zero, which leaves them positive
    where the step-start tracer is negative): vertical diffusion's
    condensate increment never reaches the scheme, and the sedimentation
    input is the step-start ice ``qi_m1``. The dynamics and radiation
    increments ECHAM also accumulates in its tendencies reach none of the
    four. #940 rewires the stack to supply the condensate, dynamics and
    radiation increments. When omitted, the ``*_m1`` arguments default to
    the provisional state less the detrained condensate (zero upstream
    increments), which reduces section 5 to a pure saturation adjustment.

    Convective detrainment arrives separately as ``detrained_qc`` /
    ``detrained_qi`` [kg/kg per step], the liquid and ice parts of the
    condensate the convection scheme added to ``qc``/``qi`` this step
    (ECHAM ``ztmst·pxtecl``/``ztmst·pxteci``; default zero). ECHAM's 2M
    uses only their sum ``zxtec`` (the boundary condition 'Detrained
    condensate', 555-570): it is not sedimented this step, it brings its
    own crystal number ``znidetr`` where ECHAM's ``ll_cv`` holds, and the
    section-4 ``lo2`` criterion re-splits it into ice and liquid.

    The large-scale vertical velocity is not plumbed to this scheme yet:
    ECHAM's ``zvervx`` (updraft for the WBF gate) uses only the TKE term
    here, and the ``knvb``/``lonacc`` inversion-level exception on
    ``zauloc`` is omitted (it needs ``pvervel``) — listed in #941.

    qnc / qni are stored per kg of air; the scheme interior uses per-m^3,
    so we convert at the boundary.
    """
    if temperature_m1 is None:
        temperature_m1 = temperature
    if specific_humidity_m1 is None:
        specific_humidity_m1 = specific_humidity
    if detrained_qc is None:
        detrained_qc = jnp.zeros_like(qc)
    if detrained_qi is None:
        detrained_qi = jnp.zeros_like(qi)
    if qc_m1 is None:
        qc_m1 = qc - detrained_qc
    if qi_m1 is None:
        qi_m1 = qi - detrained_qi

    eps_dt = jnp.finfo(qc.dtype).eps
    zero = jnp.zeros_like(qc)
    # Latent-heat-to-heat-capacity ratios built from the MOIST heat capacity
    # ``cpd·(1 + vtmpc2·q)`` evaluated at the step-start humidity (ECHAM
    # zlvdcp = alv/pcair, zlsdcp = als/pcair; mo_cloud_micro_2m.f90:844-848,
    # pcair from pqm1). Per-level (nlev,) arrays — every ledger term below and
    # the ``update_tendencies`` accounting divide by this same per-level cp,
    # so the column enthalpy identity closes against ``cp·dT`` (the enthalpy
    # gate uses the same moist cp). #706.
    lvdcp, lsdcp = latent_heat_over_cp(specific_humidity_m1)

    # ------------------------------------------------------------------
    # Upstream increments (ECHAM's accumulated tendencies × ztmst)
    # ------------------------------------------------------------------
    dT_up = temperature - temperature_m1          # ztmst·ptte
    dq_up = specific_humidity - specific_humidity_m1  # ztmst·pqte
    # The condensate increments EXCLUDE this step's convective detrainment:
    # ECHAM keeps it out of pxlte/pxite and hands it to the 2M separately as
    # zxtec (555-570), and the sweep treats it by its own rules (not
    # sedimented this step, own crystal number, re-split by lo2).
    dqc_up = (qc - qc_m1) - detrained_qc          # ztmst·pxlte
    dqi_up = (qi - qi_m1) - detrained_qi          # ztmst·pxite
    zxtec = detrained_qc + detrained_qi           # ztmst·zxtec, both phases

    # ------------------------------------------------------------------
    # Entry floor on the number tracers
    # ------------------------------------------------------------------
    # ECHAM's section-1 numbers are ρ·(pxtm1 + ztmst·pxtte) floored at
    # cqtmin = 1e-12 /m³ (600-605), with no upper bound: ICNC is capped at
    # icemax only once the detrained crystal number has joined it (1252),
    # and CDNC is not capped at all. The floor keeps the dynamical core's
    # spectral ringing — small negative tracer values — out of
    # ``update_in_cloud_water``, whose ``delta_cdnc = activated_cdnc -
    # droplet_number`` step would amplify it.
    #
    # The floor is cqtmin, not 0, because the two are different states to
    # the Korolev/Mazin threshold updraft, which is proportional to ICNC:
    # at ICNC = 0 it is exactly 0, and the strict section-1 criterion
    # ``lo2_2d = 0.01·zvervx < zvervmax`` (885) is then false at zero
    # updraft (the lowest level, 815, or TKE = 0), where at cqtmin it is
    # true. In an ice-free mixed-phase cell — the normal state after the
    # negative-mass repair has removed the number with the ice — a 0 floor
    # would fail ``ll_cv`` there and drop the crystal number of the
    # detrained ice ``znidetr``, leaving that ice number-less.
    #
    # For ICNC the floor is deliberately below ECHAM's ``icemin``, whose
    # floors (1127-1131, 1253) this scheme does not apply: arrivals at or
    # below ``icemin`` are re-diagnosed from ice mass in
    # ``update_in_cloud_water`` (the ``<=`` test fires either way), so an
    # icemin floor would only inject a spurious icemin-per-step tracer
    # source into ice-free cells.
    #
    # The floor shapes only the WORKING numbers. The number tendencies are
    # taken against the RAW step-start tracers, as ECHAM passes
    # pxtm1(:,jk,idt_cdnc/idt_icnc) unfloored to
    # update_tendencies_and_important_vars (1781) and forms
    # pxtte = (n/ρ − pxtm1)/ztmst (3625-3628): the end-of-step tracer is
    # then exactly the scheme's number, so an out-of-range raw value is
    # removed within the step instead of being carried and re-floored.
    qnc_raw = qnc
    qni_raw = qni
    inv_rho = 1.0 / jnp.maximum(air_density, eps_dt)

    # Number-per-kg-of-air → per-m^3 at the scheme's API boundary, where
    # ECHAM applies the floor (per m³, 600-605).
    cdnc0 = jnp.maximum(qnc * air_density, params.cqtmin)
    icnc0 = jnp.maximum(qni * air_density, params.cqtmin)

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
    # consumed when ldyn_cdnc_min=True), of the liquid present BEFORE this
    # step's detrainment: ECHAM evaluates zcdnc_min on pxlm1 + ztmst·pxlte
    # (609-610) and floors CDNC with it at 1124-1125, there only in cells
    # with paclc >= epsec and ptm1 > cthomi; this scheme floors the working
    # CDNC with it at entry in every cell.
    inv_cf_min = 1.0 / jnp.maximum(cloud_fraction, params.epsec)
    qc_in_cloud_kgm3 = jnp.where(
        cloud_fraction > params.epsec,
        (qc_m1 + dqc_up) * inv_cf_min * air_density, 0.0,
    )
    cdnc0 = jnp.maximum(cdnc0, minimum_CDNC(qc_in_cloud_kgm3, params))

    # ------------------------------------------------------------------
    # Step-start (t-1) thermodynamic fields — ECHAM section 1
    # ------------------------------------------------------------------
    # Saturation anchors are evaluated at the STEP-START state, exactly as
    # ECHAM evaluates zqsi/zqsw/zeta/the subsaturations at (ptm1, pqm1);
    # the provisional state enters only through the increments above.
    # ECHAM reads them from the 0.001 K tables (mo_cloud_micro_2m.f90
    # l.648-702): the "water" set from ``tlucuaw`` — Sonntag over liquid
    # water at ALL temperatures, which the Bergeron/WBF machinery needs,
    # since it depends on the water/ice saturation *difference* below
    # freezing — and the "ice" set from ``tlucua``, the ``ua`` table: Sonntag
    # over ice at and below tmelt and over water above (``phase="auto"``).
    # The vapour pressures are ``sat_spec_hum``'s capped ``zes`` times
    # ``p·rv/rd`` (``zesw_2d``, ``zesi``, l.668 and 690), so they are held at
    # ``0.5·p·rv/rd`` where that cap binds (the top few levels).
    # The slopes are section 5's ``zdqsdt = 1000·(qs(it+1) − qs(it))`` of
    # those capped ``qs`` (l.1393-1397): the fit's analytic slope, and zero
    # where the cap binds, as the difference of two capped knots is.
    es_cap = 0.5 * pressure * (c.rv / c.rd)
    es_water_fit = thermodynamics.saturation_vapor_pressure(
        temperature_m1, phase="water")
    es_ice_fit = thermodynamics.saturation_vapor_pressure(
        temperature_m1, phase="auto")
    es_water = jnp.minimum(es_water_fit, es_cap)
    es_ice = jnp.minimum(es_ice_fit, es_cap)
    qsat_water, dqsw_dt = (
        thermodynamics.saturation_specific_humidity_and_derivative(
            temperature_m1, pressure, phase="water"))
    qsat_ice, dqsi_dt = (
        thermodynamics.saturation_specific_humidity_and_derivative(
            temperature_m1, pressure, phase="auto"))
    dqsw_dt = jnp.where(es_water_fit < es_cap, dqsw_dt, 0.0)
    dqsi_dt = jnp.where(es_ice_fit < es_cap, dqsi_dt, 0.0)

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

    # Updraft velocity [cm/s] of the Wegener-Bergeron-Findeisen and phase
    # (lo2) criteria, ECHAM zvervx (mo_cloud_micro_2m.f90:814-816):
    #   zvervx = −100·ω/(g·ρ) + 100·fact_tke·sqrt(TKE),
    # with the turbulent term zeroed at the lowest level (line 815). The
    # large-scale term −100·ω/(g·ρ) needs the pressure velocity, which is
    # not plumbed to this scheme (#941); it is the term to add here.
    updraft_velocity = turbulent_updraft_velocity(tke, params)

    # ------------------------------------------------------------------
    # Section-1 crystal number of convective detrainment (ECHAM 858-982)
    # ------------------------------------------------------------------
    # Temperature-parameterised volume-mean crystal radius zrid [m] at the
    # step-start temperature (945-956): the size detrained crystals are
    # created at (znidetr below) and the radius the section-5.5 ICNC
    # diagnosis inverts (prid, 1511).
    zrid = ice_volume_mean_radius_from_temperature(temperature_m1, params)

    # Section-1 Wegener-Bergeron-Findeisen criterion lo2_2d (858-885): the
    # Korolev/Mazin threshold updraft of the ice present BEFORE this step's
    # detrainment (pxim1 + ztmst·pxite, 860-861) at the step-start crystal
    # number, against the updraft. It has no temperature terms; the
    # temperature gates are ll_cv's (958-963) and the section-4 lo2's.
    # ``icnc0`` is the number floored at cqtmin on entry, as ECHAM's is
    # (604-605); that floor is what lets an ice-free cell pass this strict
    # test at zero updraft (see the entry floor). ECHAM's CDNC↔ICNC phase
    # consistency step (607-628) has no counterpart in this scheme.
    zxip1_sec1 = jnp.maximum(qi_m1 + dqi_up, 0.0)
    ice_gm3_sec1 = (1000.0 * zxip1_sec1 * air_density
                    / jnp.maximum(cloud_fraction, params.clc_min))
    zrice_sec1 = ice_volume_mean_radius_schumann(ice_gm3_sec1, icnc0, params)
    zvervmax_sec1 = threshold_vert_vel(
        sat_vap_pres_water=es_water, sat_vap_pres_ice=es_ice,
        icnc=icnc0, ice_radius=zrice_sec1, eta=bergeron_eta, params=params)
    lo2_2d = 0.01 * updraft_velocity < zvervmax_sec1

    # In-cloud crystal number of the detrained ice znidetr [1/m³]
    # (958-982), added to the post-sedimentation ICNC in the sweep. ECHAM's
    # cirrus nucleation zninucl (986-999) and the zicncq floors (1119-1131)
    # are not part of this scheme.
    znidetr = detrained_ice_crystal_number(
        zxtec, temperature_m1, lo2_2d, cloud_fraction, air_density, zrid,
        params)

    # Dynamic viscosity of air for the snow Reynolds number in riming,
    # ECHAM pviscos (mo_cloud_utils.f90::get_util_var, line 132).
    dynamic_viscosity = air_dynamic_viscosity(temperature_m1)

    # Geometry / density helpers.
    pressure_thickness = air_density * params.grav * layer_thickness
    dp_over_g = pressure_thickness * c.rgrav
    zqrho = 1.3 * inv_rho                      # ECHAM zqrho = 1.3/ρ
    # Ice fall-speed air-density factor, ECHAM paaa
    # (mo_cloud_utils.f90::get_util_var, line 129).
    air_density_correction = ice_fall_speed_air_density_factor(
        pressure, temperature_m1)
    melt_mask = temperature_m1 > params.tmelt  # ECHAM ll_mlt (ptm1)

    # Heterogeneous mixed-phase INP [1/m³] of the aerosol-free freezing
    # closure (section 6.2): the DeMott (2010) diagnostic on prescribed
    # coarse aerosol at ambient density, or a larger external INP a caller
    # supplies as ``ice_nuclei`` (no in-tree composition does). Unused when
    # ``freezing_aerosol`` selects ECHAM-HAM's rates.
    demott_floor = demott2010_inp(
        temperature_m1, params.n_aer_coarse, air_density)
    n_inp = jnp.maximum(ice_nuclei, demott_floor)
    # The HAM freezing inputs ride the level scan with the TKE the immersion
    # cooling rate needs (F 2800). ECHAM reads the previous step's TKE
    # (ptkem1); this is the TKE the scheme receives, the one its updraft
    # zvervx also uses. A static Python switch: the aerosol-free composition
    # traces exactly the closure it had before.
    if freezing_aerosol is None:
        freezing_levels = ()
    else:
        freezing_levels = (
            tke, freezing_aerosol.dust_soluble,
            freezing_aerosol.dust_insoluble_accumulation,
            freezing_aerosol.dust_insoluble_coarse,
            freezing_aerosol.bc_soluble, freezing_aerosol.bc_insoluble,
            freezing_aerosol.wet_radius_insoluble_aitken,
            freezing_aerosol.wet_radius_insoluble_accumulation,
            freezing_aerosol.wet_radius_insoluble_coarse,
        )

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
         qc_m1_k, qi_m1_k, det_qc_k, zxtec_k, zrid_k, znidetr_k,
         p_k, rho_k, inv_rho_k, dp_k, dpg_k, dz_k, adc_k, zqrho_k,
         cdnc0_k, icnc0_k,
         esw_k, esi_k, qsw_k, qsi_k, dqsw_k, dqsi_k,
         subice_k, subwat_k, thermo_k, eta_k, verv_k, visc_k, melt_k,
         act_cdnc_k, n_inp_k, inp_dep_k, is_bottom_k,
         lvdcp_k, lsdcp_k, freezing_k) = level_in

        zero_s = jnp.zeros_like(cf_k)

        # --- 4. Sedimentation of cloud ice (grid-mean) -----------------
        # Acts on the ice present BEFORE this step's convective
        # detrainment, ECHAM zxip1 = pxim1 + ztmst·pxite (1227-1228):
        # pxite excludes the detrained condensate, which ECHAM carries
        # separately as zxtec. The detrained ice therefore does not fall
        # this step; it joins the in-cloud state after sedimentation
        # through zxidt (1316), with its own crystal number (below).
        # ECHAM floors the input at EPSILON(1._dp) = 2.2e-16; this scheme
        # floors at 0, because the float32 epsilon (1.2e-7 kg/kg) is a
        # sizeable amount of ice.
        zxip1_pre = qi_m1_k + dqi_up_k
        (zxip1, icnc_sedi, ice_flux, ice_flux_n, falling_ice_frac,
         mrateps_sedi_k) = sedimentation_ice(
            cf_k, adc_k, dp_k, rho_k, inv_rho_k,
            jnp.maximum(zxip1_pre, 0.0), icnc0_k,
            ice_flux, ice_flux_n, falling_ice_frac,
            dt, params,
        )
        # ECHAM pxite = (zxip1 − pxim1)/ztmst (1248): the post-sedimentation
        # ice relative to the UNFLOORED pre-sedimentation ice, so that
        # pxim1 + ztmst·pxite reconstructs zxip1 exactly in the ledger.
        sedi_tend = (zxip1 - zxip1_pre) / dt

        # The crystal number of the detrained ice joins the post-
        # sedimentation ICNC, capped at icemax (1251-1252). ECHAM's floor at
        # icemin (1253) is not applied: the ICNC lower bound stays cqtmin
        # (znidetr's own floor, 978, as at entry) and number-less ice is
        # re-diagnosed in update_in_cloud_water (see the entry floor).
        icnc_sedi = jnp.minimum(icnc_sedi + znidetr_k, params.icemax)

        # --- 3.1 Melting (fluxes + in-cloud ice) -----------------------
        # Runs after sedimentation (MG/PUMAS order, see docstring); the
        # running ice tendency is threaded THROUGH the routine so
        # ``pimlt = max(zxip1_pre + ztmst·pxite, 0)`` reconstructs the ice
        # left AFTER sedimentation and the two sinks cannot claim the same
        # mass (#662 finding 2). The number melted (``zicncq``) and the
        # number reset (``picnc``) are both the post-znidetr ICNC: ECHAM's
        # zicncq includes znidetr (982), and znidetr is only its cqtmin
        # floor where melting acts (ll_cv needs T_m1 < tmelt, melting
        # T_m1 > tmelt), so ECHAM's melt-then-add order and this
        # add-then-melt order give the same number.
        (icnc_melt, _qmel, cdnc_melt,
         rain_flux, snow_flux, ice_flux, ice_flux_n,
         ice_tend_k, pimlt_k, psmlt_a, pximlt_k) = melting_snow_and_ice(
            melt_k, t_m1_k, zxip1_pre, dp_k,
            icnc_sedi, lsdcp_k, lvdcp_k,
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
            subice_k, lsdcp_k,
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

        # --- Phase decision lo2 (ECHAM section 4 end, 1276-1298) -------
        # Ice-vs-liquid regime from the Korolev/Mazin threshold updraft,
        # computed on the post-sedimentation, pre-detrainment ice and on
        # the ICNC that already carries znidetr (1251 precedes 1281).
        # ``zrice`` is ECHAM's volume-mean radius for this threshold
        # (0.9·r_eff, line 1288), in METRES; the shared helper is also
        # what the section-1 criterion, deposition_freezing.py and the WBF
        # gate below use, so the four decisions cannot drift.
        ll_cc = cf_k > params.clc_min
        cf_safe = jnp.maximum(cf_k, params.clc_min)
        ice_gm3 = 1000.0 * zxip1 * rho_k / cf_safe
        zrice = ice_volume_mean_radius_schumann(ice_gm3, icnc_melt, params)
        zvervmax = threshold_vert_vel(
            sat_vap_pres_water=esw_k, sat_vap_pres_ice=esi_k,
            icnc=icnc_melt, ice_radius=zrice, eta=eta_k, params=params)
        lo2 = jnp.logical_or(
            t_m1_k < params.cthomi,
            jnp.logical_and(t_m1_k < params.tmelt,
                            0.01 * verv_k < zvervmax),
        )

        # --- Re-split of the detrained condensate (1301-1317) ----------
        # ECHAM's 2M ignores the convection scheme's phase split of the
        # detrained condensate and assigns the whole of zxtec by lo2: ice
        # where lo2 holds, liquid elsewhere (zxite2/zxlte2, 1310-1314).
        # ``move`` is the net mass this reclassifies from the convection's
        # ice to liquid (> 0) or from its liquid to ice (< 0). Convection
        # heated the detrained condensate with the latent heat of its own
        # phase, so the reclassification carries the fusion-heat
        # difference, added to the temperature increment before section 5
        # exactly where ECHAM modifies ptte (1301-1307; zdtdt =
        # ztmst·ptte − …, 1406). ECHAM's gate
        # ``ztconv <= tmelt .AND. .NOT. lo2`` (1302-1303) reads the
        # temperature the convection scheme split the condensate on
        # (ztconv is cudtdq's pten, mo_cumastr.f90:248), so it is exactly
        # "convection counted it as ice, lo2 makes it liquid", which here is
        # ``detrained_qi`` with lo2 false. Two deliberate deviations keep
        # the scheme's column enthalpy identity exact: (1) the correction
        # divides by the per-level MOIST cp (lsdcp − lvdcp, #706) where
        # ECHAM divides by cpd (1305); (2) it also acts in the other
        # direction, which ECHAM leaves uncorrected: condensate convection
        # counted as liquid (its temperature above tmelt while T_m1 is
        # below it) that lo2 makes ice.
        ice_part = jnp.where(lo2, zxtec_k, 0.0)      # ztmst·zxite2
        liq_part = jnp.where(lo2, 0.0, zxtec_k)      # ztmst·zxlte2
        move = liq_part - det_qc_k
        split_dT = -(lsdcp_k - lvdcp_k) * move

        # --- In-cloud condensate prep + clear-sky evaporation ----------
        # ECHAM 1319-1385. The step's total non-microphysical increments
        # (upstream + re-split detrainment + sedimentation + melting) are
        # split ECHAM-style: in cloudy cells positive increments enter the
        # in-cloud state at their grid-mean magnitude while their
        # clear-sky share ``(1−paclc)·max(increment, 0)`` evaporates (the
        # two add back to the full grid-mean increment); negative
        # increments deplete in-cloud values clamped at zero; in
        # cloud-FREE cells the entire condensate — carried plus
        # incremented — evaporates, returning it to vapour with the
        # matching latent cooling.
        zxidt = dqi_up_k + ice_part + dt * ice_tend_k     # 1316
        zxldt = dqc_up_k + liq_part + pximlt_k + pimlt_k  # 1317
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

        # --- 5. Condensation source zqcdif → zcnd / zdep ---------------
        # The Sundqvist moisture-convergence closure (ECHAM 1389-1470):
        # the humidity increment this step, minus the saturation-humidity
        # change implied by the temperature increment (damped by the
        # warming feedback), condenses into the cloudy fraction.
        zlc = jnp.where(lo2, lsdcp_k, lvdcp_k)
        zqsm1 = jnp.where(lo2, qsi_k, qsw_k)
        zdqsdt = jnp.where(lo2, dqsi_k, dqsw_k)

        zdtdt = (dT_up_k + split_dT
                 - lvdcp_k * (evp_k + zxlevap)
                 - (lsdcp_k - lvdcp_k) * (psmlt_a + pximlt_k + pimlt_k)
                 - lsdcp_k * (sub_k + zxievap + xisub_k))
        zqp1 = jnp.maximum(q_m1_k + dq_up_k, 0.0)
        ztp1 = t_m1_k + zdtdt

        zdqsat = (zdtdt
                  + cf_k * (zlc * dq_up_k
                            + lvdcp_k * (evp_k + zxlevap)
                            + lsdcp_k * (sub_k + zxievap + xisub_k)))
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
            lsdcp_k, lvdcp_k,
            zqp1, zqsm1,
            rho_k, ztp1,
            zxievap,
            zxip1,
            # pxite: the ice part of the re-split detrainment [kg/kg/s],
            # ECHAM zxite (1490), which the routine adds to the
            # post-sedimentation zxip1 for its own lo2 test (2361-2362).
            ice_part / dt,
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
            inp_dep_k,          # pnicex: read only by the nic_cirrus=2 branch
            zqp1tmp, zqsp1tmp,
            rho_k,
            zrid_k,             # prid: ECHAM zrid [m] (1511)
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
        # ECHAM ll_mxphase_frz (1541-1545): liquid present, mixed-phase
        # window on the corrected temperature, droplets at/above the floor,
        # cloud present.
        ll_mxfrz = (
            (zxlb > params.cqtmin)
            & (ztp1tmp < params.tmelt)
            & (ztp1tmp > params.cthomi)
            & (cdnc_f >= cdnc_min_k)
            & cloud_flag
        )
        if freezing_k:
            # ECHAM-HAM's het_mxphase_freezing (1552-1567, 2675-2840):
            # Brownian contact freezing on insoluble dust and immersion
            # freezing of dust/BC-bearing droplets, as rates over the step,
            # on the HAM freezing inputs of the composition's aerosol. The
            # immersion rate needs the cooling of the vertical motion,
            # ECHAM's ztte = (omega - fact_tke*sqrt(TKE)*rho*g)/(cpd*rho)
            # (2800-2802). The turbulent part is here; the large-scale
            # omega is not plumbed to this scheme (#705), so it enters as 0
            # and ascent-driven immersion freezing beyond the TKE updraft is
            # absent, as is the suppression in large-scale subsidence.
            (tke_k, fdusol_k, fduai_k, fduci_k, fbcsol_k, fbcinsol_k,
             rwetki_k, rwetai_k, rwetci_k) = freezing_k
            (icnc_het, cdnc_h, zfrl, zxib, zxlb,
             new_crystals) = het_mxphase_freezing(
                ll_mxfrz, p_k, tke_k,
                zero_s,             # pvervel: large-scale omega (#705)
                paclc, fbcsol_k, fbcinsol_k, fdusol_k, fduai_k, fduci_k,
                rho_k, inv_rho_k, rwetki_k, rwetai_k, rwetci_k,
                ztp1tmp, cdnc_min_k,
                icnc_f, cdnc_f, zfrl, zxib, zxlb,
                dt, params.cqtmin, params,
            )
        else:
            # The aerosol-free closure: freeze one mean-mass droplet per new
            # crystal up to the INP number, moving number, mass AND fusion
            # heat together (#662 finding 3). The new crystals are capped by
            # the droplets available, as the frozen mass is capped by the
            # liquid, so an INP number above CDNC cannot create crystals
            # from nothing. ECHAM's aerosol-free freezing is its lccnclim
            # mode, which needs the large-scale omega (#705); until then the
            # INP number is DeMott (2010). ECHAM's het_mxphase_freezing caps
            # the frozen number at the droplets above cdnc_min (2818-2820);
            # this closure leaves at least cqtmin.
            new_crystals = jnp.where(
                ll_mxfrz,
                jnp.minimum(jnp.maximum(n_inp_k - icnc_f, 0.0), cdnc_f),
                0.0,
            )
            icnc_het = icnc_f + new_crystals
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
        zrice_wbf = ice_volume_mean_radius_schumann(
            ice_gm3_wbf, icnc_het, params)
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
            ll_wbf, paclc, lsdcp_k, lvdcp_k,
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
        (cdnc_p, zxlb, mratepr_k, zrpr, _rprn,
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
        (icnc_c, cdnc_c, mrateps_k, zxib, zxlb,
         _sprn, zsacl, _sacln, msnowacl_k, zspr) = precip_formation_cold(
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
            # zmrateps INOUT seed: the in-cloud sedimented-ice amount from
            # section 4 (ECHAM 1243 → 1703) — ECHAM-HAM counts sedimenting
            # ice as a scavenging carrier via this ledger entry. Where the
            # cold chain runs it OVERWRITES the seed (Fortran MERGE at
            # 3310); elsewhere the seed survives to ``cloud_subm_2``.
            mrateps_sedi_k,
            zxib, zxlb,
            dt,
            params,
        )

        # --- 7.3 Update precipitation fluxes ---------------------------
        (precip_cover, rain_flux, snow_flux, snow_melt_b,
         _pfevapr, _pfrain, _pfsnow, _pfsubls) = update_precip_fluxes(
            paclc, dp_k,
            evp_k, lsdcp_k, lvdcp_k,
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
            mratepr_k, mrateps_k, msnowacl_k,
            auto_only_k, accr_only_k,
            rain_flux, frozen_flux_k,
            wbf_transfer_k, ll_liqcl_k, ll_icecl_k,
            move, split_dT,
        )
        return carry_out, level_out

    scan_inputs = (
        cloud_fraction, temperature_m1, specific_humidity_m1,
        dT_up, dq_up, dqc_up, dqi_up,
        qc_m1, qi_m1, detrained_qc, zxtec, zrid, znidetr,
        pressure, air_density, inv_rho, pressure_thickness, dp_over_g,
        layer_thickness, air_density_correction, zqrho,
        cdnc0, icnc0,
        es_water, es_ice, qsat_water, qsat_ice, dqsw_dt, dqsi_dt,
        subsat_wrt_ice, subsat_wrt_water, thermo_term_water,
        bergeron_eta, updraft_velocity, dynamic_viscosity, melt_mask,
        activated_cdnc, n_inp, ice_nuclei_deposition, is_bottom_level,
        lvdcp, lsdcp, freezing_levels,
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
     zmratepr, zmrateps, zmsnowacl,
     autoconv_only, accretion_only,
     rain_flux_profile, snow_flux_profile,
     wbf_transfer, ll_liqcl, ll_icecl,
     detrainment_move, detrainment_split_dT) = scan_outs

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
        incloud_liq_scav, incloud_ice_scav,
        liq_eff_radius, ice_eff_radius,
        zdxlcor, zdxicor,
    ) = update_tendencies_and_important_vars(
        icnc=icnc_final,
        cdnc=cdnc_final,
        # ECHAM's pxim1/pxlm1 are the grid-mean condensate the increments
        # accumulate on, and pxitec/pxltec (zxite/zxlte) the re-split
        # detrainment. Here the upstream increments and the detrainment
        # are already inside ``qi``/``qc`` as convection split it, so the
        # re-split provisional state ``qi − move`` / ``qc + move`` stands
        # for pxim1 + ztmst·(pxite + pxitec) / pxlm1 + ztmst·(pxlte +
        # pxltec), and the seeds below carry only the scheme's own
        # tendencies: the reconstruction is the same number, with the
        # negative-mass guard testing the actual end-of-step grid-mean
        # state against ``ccwmin`` (#662 finding 6).
        ice_mmr_prev=qi - detrainment_move,
        liq_mmr_prev=qc + detrainment_move,
        # ECHAM convention: pxtm1_cdnc / pxtm1_icnc are the step-start
        # tracer values in per-kg-of-air, RAW (1781; see the entry floor).
        # The working cdnc/icnc are per-m³, so the tendency subtracts
        # per-kg from per-m³·1/ρ, and the negative-mass repair removes the
        # whole number where it zeroes the condensate (3632-3652).
        tracer_tm1_cdnc=qnc_raw,
        tracer_tm1_icnc=qni_raw,
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

    # The ledger ran on the re-split provisional state; the host adds the
    # returned tendencies to ITS provisional (qc, qi, temperature), which
    # still holds convection's split, so the reclassification and its
    # fusion-heat correction are returned as tendencies too.
    tendencies = MicrophysicsTendencies_2M(
        dtedt=dtedt + detrainment_split_dT / dt,
        dqdt=dqdt,
        dqcdt=dqcdt + detrainment_move / dt,
        dqidt=dqidt - detrainment_move / dt,
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

    # The ECHAM-HAM wet-scavenging interface (#708): the process-time
    # ledger cloud_subm_2 receives, published for the JAM wetdep and
    # cloud-borne terms (see ScavengingLedger). The pools are the
    # POST-assembly zmlwc/zmiwc (the Fortran passes them INOUT through
    # update_tendencies_2, which zeroes them in emptied cells — a zeroed
    # pool with a positive formation rate is the fraction=1 marker). The
    # process cover is clipped to the incoming Sundqvist cover for the
    # same reason the published cloud_fraction is: the RH-raising branch
    # of update_in_cloud_water is a closure this stack does not use.
    scav_ledger = ScavengingLedger(
        incloud_liquid=incloud_liq_scav,
        incloud_ice=incloud_ice_scav,
        rain_formation=zmratepr / dt,
        snow_formation=zmrateps / dt,
        liquid_riming=zmsnowacl / dt,
        process_cloud_fraction=jnp.minimum(paclc_final, cloud_fraction_in),
        condensate_evaporation=(xlevap + xievap) / dt,
    )

    # NOTE the (nlev,) rain / frozen flux profiles stay LAST: call sites and
    # tests unpack them positionally from the end (``*_, rain_b, snow_b``),
    # so new outputs are inserted before them, not appended.
    return tendencies, surface_rain_flux, surface_snow_flux, \
        liq_eff_radius, ice_eff_radius, rain_formation_warm, rain_from_melt, \
        autoconv_rate_col, accretion_rate_col, wbf_rate_col, \
        precip_formation_rate, precip_evaporation_rate, cloud_fraction_final, \
        negative_mass_repair, scav_ledger, \
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
    upstream), together with this step's convective detrainment
    ``clouds.conv_detrainment_qc/qi`` (written by ``TiedtkeConvection``,
    zero without a convection term), TKE from ``"vertical_diffusion"``, the
    activated CDNC (JAM's ``activated_cdnc``, or the SPA-style floor from the
    public ``"aerosol"`` Nccn), and, where a prognostic aerosol publishes
    them, ECHAM-HAM's heterogeneous-freezing inputs ``"freezing_aerosol"``
    (a :class:`HeterogeneousFreezingAerosol`), which switch section 6.2 from
    the DeMott closure to ECHAM's contact + immersion rates. Writes the
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
    # CF/units metadata for the ``clouds.*`` output fields (#740). Shared with
    # the cover term; this term fills the precip / 2M / process-rate fields.
    output_attrs: ClassVar[dict[str, dict[str, str]]] = CLOUD_OUTPUT_ATTRS

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

    def adopt_runtime_configuration(self, previous) -> None:
        """Inherit the SPA activation tuning from a displaced 2M term.

        Set by ``echam_physics`` after composition, from the aerosol module's
        parameters, so a term swapped in afterwards would otherwise silently
        fall back to the (1.0, 0.5, 0.0) constructor defaults and change the
        droplet number the whole cloud scheme keys off.
        """
        for name in ("_spa_prefactor", "_spa_exponent", "_spa_cap_smoothing"):
            param = getattr(previous, name, None)
            if param is not None:
                setattr(self, name, nnx.Param(jnp.asarray(param.get_value())))

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

        # Provisional state (sequential vdiff->convection->cloud coupling,
        # ECHAM physc order). T and q are the running view ``thermo_run``,
        # which the upstream vdiff and convection terms have advanced with
        # their tendencies. qc and qi are ``clouds.qc/qi``: the cover term
        # snapshots them from ``thermo_run`` before vertical diffusion runs
        # (echam_physics composes SundqvistCloudFraction ahead of
        # TteTkeVerticalDiffusion, and vdiff advances only ``thermo_run``),
        # and the convection term then adds its detrainment. So the T and q
        # increments carry vdiff + convection, while the condensate
        # increments less the detrainment are zero up to the convection
        # term's clip at zero: vdiff's condensate increment does not reach
        # the scheme, and the ice it sediments is the step-start ``qi_m1``.
        # #940 rewires this to supply the condensate (and the dynamics and
        # radiation) increments; the scheme already treats whatever arrives
        # as ECHAM's accumulated tendencies.
        #
        # The provisional state is what the returned tendencies are relative
        # to (the host's additive sum with the upstream tendencies telescopes
        # back to the correct final state), while the STEP-START state
        # supplies ECHAM's (ptm1, pqm1, pxlm1, pxim1) anchors: saturation
        # evaluates there, and the differences, less the detrained
        # condensate, play the role of the accumulated tendencies (see
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
        # This step's convective detrainment, kg/kg per step: the liquid and
        # ice parts the convection term added to ``clouds.qc/qi`` (ECHAM
        # ztmst·pxtecl / ztmst·pxteci; zero without a convection term, as
        # the cover term resets the fields every step). The scheme needs it
        # apart from the other increments: it is not sedimented this step,
        # carries its own crystal number and is re-split by lo2.
        detrained_qc = dt * clouds.conv_detrainment_qc
        detrained_qi = dt * clouds.conv_detrainment_qi

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

        # Activated CDNC source. Which baseline fills the cells an online
        # activation model leaves empty depends on the aerosol scheme, and the
        # presence of the ``activated_cdnc`` diagnostic tells the two apart:
        #
        #  * MACv2-SP path (no online activation term, ``activated_cdnc``
        #    absent): the SPA floor derived from the prescribed-plume
        #    ``aerosol.Nccn`` IS the activation source — the scheme's only
        #    Twomey link (Lin et al. 2025, ``jcm.physics.aerosol.spa``).
        #
        #  * JAM path (``ArgActivation`` writes ``activated_cdnc``, #461): JAM
        #    carries no prescribed-plume ``Nccn`` (it was removed from the JAM
        #    composition in #640), so the SPA floor would be identically zero
        #    and leave ARG-empty cloudy cells with no droplets. Fall back
        #    instead to the scheme's OWN minimum-CDNC — the ECHAM-HAM fixed
        #    ``cdnc_min_fixed`` (40 cm⁻³) or the dynamic max-radius floor,
        #    selected by ``ldyn_cdnc_min`` (#674). This is the same
        #    ``minimum_CDNC`` the per-column warm microphysics enforces on the
        #    droplet number (see ``minimum_CDNC`` below): feeding it as the
        #    activation *target* here only sets the nucleation source in
        #    ARG-empty cells; the per-column floor still owns the actual number
        #    bound, so the droplet number is not doubly floored.
        arg_cdnc = diagnostics.get("activated_cdnc")
        if arg_cdnc is None:
            Nccn = diagnostics["aerosol"].Nccn
            activated_cdnc = spa_activated_cdnc(
                Nccn=Nccn[jnp.newaxis, :],
                cloud_fraction=cloud_fraction,
                prefactor=self._spa_prefactor.get_value(),
                exponent=self._spa_exponent.get_value(),
                cap_smoothing=self._spa_cap_smoothing.get_value(),
            )
        else:
            # In-cloud liquid mass density (kg/m³) for the dynamic branch of
            # ``minimum_CDNC``; the fixed branch ignores it. Masked to cloudy
            # cells so the fallback (like the SPA floor it replaces) is zero in
            # clear air — the activation gate is cloud-only anyway.
            inv_cf_min = 1.0 / jnp.maximum(cloud_fraction, params_2m.epsec)
            qc_in_cloud_kgm3 = jnp.where(
                cloud_fraction > params_2m.epsec,
                qc_interim * inv_cf_min * air_density, 0.0,
            )
            cdnc_min_floor = jnp.where(
                cloud_fraction > params_2m.epsec,
                minimum_CDNC(qc_in_cloud_kgm3, params_2m), 0.0,
            )
            activated_cdnc = jnp.where(arg_cdnc > 1.0, arg_cdnc, cdnc_min_floor)

        # Aerosol inputs of the heterogeneous mixed-phase freezing. A
        # prognostic aerosol (JAM) publishes ECHAM-HAM's freezing inputs as
        # ``freezing_aerosol`` (the ham_IN_setup fields), which selects
        # ECHAM's contact + immersion rates in section 6.2; without it the
        # scheme uses its aerosol-free DeMott (2010) closure. ``ice_nuclei``
        # is an optional external INP number for that closure (no in-tree
        # term publishes it). ``ice_nuclei_deposition`` reaches
        # ``update_in_cloud_water`` as ECHAM's ``pnicex``, read only by the
        # nic_cirrus=2 branch (#552).
        zeros_2d = jnp.zeros_like(state.temperature)
        ice_nuclei = diagnostics.get("ice_nuclei", zeros_2d)
        ice_nuclei_deposition = diagnostics.get(
            "ice_nuclei_deposition", zeros_2d
        )
        freezing_aerosol = diagnostics.get("freezing_aerosol")

        # The core owns grid-scale condensation now (the ECHAM section-5
        # zqcdif closure + supersaturation corrections run inside the
        # sweep), so there is NO external Sundqvist condensation bolt-on
        # any more. The bolt-on dated from when the internal adjustment
        # was suppressed ~1e6x by the c.ak/zdqsdt transcription bugs
        # (#667): with those fixed, summing both would remove
        # supersaturation twice per step with double the latent heating.
        # The scheme's own preffl/preffi (ECHAM cloud_micro_2m outputs) are
        # not published: the radii radiation uses, and the ``clouds.r_eff_*``
        # diagnostic, are formed by the radiation term from the step's state,
        # as in ECHAM's cloud_optics.
        (tend_all, surface_rain_flux, surface_snow_flux,
         _preffl, _preffi, rain_formation_warm, rain_from_melt,
         autoconv_all, accretion_all, wbf_all,
         precip_form_all, precip_evap_all, cloud_fraction_all,
         negative_mass_repair_all, scav_ledger_all,
         rain_flux_all, snow_flux_all) = jax.vmap(
            cloud_microphysics_2m,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     None, None, 1, 1, 1, 1, 1, 1,
                     None if freezing_aerosol is None else 1),
            out_axes=(0,) * 17,
        )(
            temperature_in, specific_humidity_in, pressure_full,
            qc_interim, qi_interim, qnc, qni,
            cloud_fraction, air_density, layer_thickness, tke,
            activated_cdnc, ice_nuclei, ice_nuclei_deposition, dt, params_2m,
            state.temperature, state.specific_humidity, qc_m1, qi_m1,
            detrained_qc, detrained_qi, freezing_aerosol,
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

        # ``qnc_prev``/``qni_prev`` keep this step's RAW step-start number
        # tracers: the baseline the number tendencies are taken against
        # (ECHAM pxtm1, mo_cloud_micro_2m.f90:1781), not the scheme's
        # entry-floored working numbers. No term reads them; they record the
        # state the scheme started from. The surface precipitation comes
        # from the lax.scan carry.
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
            # Rain-source split [kg/m^2/s]: warm-chain formation vs snow
            # melt. Their ratio is the warm-rain fraction, the CloudSat-
            # style observable for the warm-rain calibration.
            rain_formation_warm=rain_formation_warm,
            rain_from_melt=rain_from_melt,
            # Column-integrated latent heating of the negative-mass
            # repair [W/m²] (#689): sign-definite spurious warming from
            # returning sub-ccwmin/negative condensate to vapour.
            negative_mass_repair=negative_mass_repair_all,
            # The ECHAM-HAM wet-scavenging ledger (#708): the process-time
            # quantities cloud_subm_2 receives, for the JAM wetdep and
            # cloud-borne terms (see ScavengingLedger in types.py). The
            # vmap puts the column axis first — transpose like the rest.
            incloud_liquid=scav_ledger_all.incloud_liquid.T,
            incloud_ice=scav_ledger_all.incloud_ice.T,
            incloud_rain_formation=scav_ledger_all.rain_formation.T,
            incloud_snow_formation=scav_ledger_all.snow_formation.T,
            incloud_riming=scav_ledger_all.liquid_riming.T,
            process_cloud_fraction=scav_ledger_all.process_cloud_fraction.T,
            condensate_evaporation_rate=(
                scav_ledger_all.condensate_evaporation.T),
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
