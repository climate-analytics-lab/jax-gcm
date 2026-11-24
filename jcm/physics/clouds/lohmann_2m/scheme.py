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
from jcm.physics.aerosol.spa import spa_activated_cdnc, spa_activation_potential
from jcm.physics.clouds.cloud_data import CLOUD_OUTPUT_ATTRS
from jcm.physics.clouds.cloud_inputs import cloud_scheme_inputs
from jcm.physics.clouds.echam_1m import cached_inversion_range, resolve_lonacc_mask
from jcm.physics.diagnostics.moist_air_state import advance_thermo_run
from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

from ..lohmann_2m_params import AerosolFreeFreezing, CloudParams2M
from ..cloud_utils import (
    air_dynamic_viscosity,
    convective_detrainment_droplet_number,
    detrained_ice_crystal_number,
    ice_fall_speed_air_density_factor,
    ice_volume_mean_radius_from_temperature,
    ice_volume_mean_radius_schumann,
    karcher_lohmann_deposition_rate,
    latent_heat_over_cp,
    minimum_CDNC,
    rain_mixing_ratio_from_flux,
    snow_mixing_ratio_from_flux,
    stratiform_activation_at_cloud_base,
    sundqvist_condensation,
    threshold_vert_vel,
    turbulent_updraft_velocity,
)
from .types import (
    HeterogeneousFreezingAerosol,
    MicrophysicsTendencies_2M,
    ScavengingLedger,
)
from .ccnclim import ccnclim_freezing_aerosol
from .sedimentation_melt import (
    melt_falling_flux,
    melt_in_cloud_ice,
    sedimentation_ice,
)
from .cirrus import xfrzmstr
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
    temperature_m1: jnp.ndarray,       # (nlev,)  K      anchor T (ECHAM ptm1)
    specific_humidity_m1: jnp.ndarray, # (nlev,)  kg/kg  anchor q (ECHAM pqm1)
    pressure: jnp.ndarray,             # (nlev,)  Pa
    qc_m1: jnp.ndarray,                # (nlev,)  kg/kg  anchor cloud liquid (ECHAM pxlm1)
    qi_m1: jnp.ndarray,                # (nlev,)  kg/kg  anchor cloud ice (ECHAM pxim1)
    qnc_m1: jnp.ndarray,               # (nlev,)  kg^-1  anchor droplet number per kg of air
    qni_m1: jnp.ndarray,               # (nlev,)  kg^-1  anchor crystal number per kg of air
    cloud_fraction: jnp.ndarray,    # (nlev,)  [0,1]
    air_density: jnp.ndarray,       # (nlev,)  kg/m^3
    layer_thickness: jnp.ndarray,   # (nlev,)  m   (dz, full-level layer depths)
    tke: jnp.ndarray,               # (nlev,)  m²/s²  turbulent kinetic energy
    activated_cdnc: jnp.ndarray,    # (nlev,)  1/m³   aerosol-activated CDNC (from MACv2-SP); the fallback for ``activation_potential`` below when a caller does not pass that separately (every in-tree caller does, so in practice this value itself no longer reaches the per-level scan -- see ``activation_potential``'s own comment)
    ice_nuclei: jnp.ndarray,        # (nlev,)  1/m³   external immersion INP for the AerosolFreeFreezing.DEMOTT closure (not the default since #1149; none in-tree)
    ice_nuclei_deposition: jnp.ndarray,  # (nlev,) 1/m³  deposition INP (no in-tree producer since #953 removed it; tracked under #1017); NOT read when nic_cirrus=2, see cirrus_aerosol_number
    dt: jnp.ndarray,                # scalar   seconds
    params: CloudParams2M,          # tunable parameters
    temperature_increment: jnp.ndarray | None = None,  # (nlev,) K      ztmst·ptte
    humidity_increment: jnp.ndarray | None = None,     # (nlev,) kg/kg  ztmst·pqte
    qc_increment: jnp.ndarray | None = None,           # (nlev,) kg/kg  ztmst·pxlte
    qi_increment: jnp.ndarray | None = None,           # (nlev,) kg/kg  ztmst·pxite
    qnc_increment: jnp.ndarray | None = None,          # (nlev,) kg^-1  ztmst·pxtte(cdnc)
    qni_increment: jnp.ndarray | None = None,          # (nlev,) kg^-1  ztmst·pxtte(icnc)
    detrained_qc: jnp.ndarray | None = None,           # (nlev,) kg/kg  ztmst·pxtecl
    detrained_qi: jnp.ndarray | None = None,           # (nlev,) kg/kg  ztmst·pxteci
    freezing_aerosol: HeterogeneousFreezingAerosol | None = None,  # (nlev,) leaves: HAM freezing inputs
    cirrus_aerosol_number: jnp.ndarray | None = None,  # (nlev,) 1/m³  papnx (HAM soluble aerosol number available for cirrus freezing; ham_cirrus_aerosol); read only at nic_cirrus=2
    vertical_velocity: jnp.ndarray | None = None,  # (nlev,) Pa/s  large-scale pressure velocity (ECHAM pvervel, positive descending, #705); None = no large-scale contribution (zvervx is TKE-only, pvervel enters het_mxphase_freezing as 0)
    lonacc_mask: jnp.ndarray | None = None,  # (nlev,) bool  ECHAM's lonacc exception on zauloc (knvb/subsidence, #705); None = no exception. UNLIKE the 1M scheme, this is NOT inert at the 2M default: ECHAM6.3-HAM2.3's 2M cauloc = 3.0 (mo_cloud_micro_2m.f90), not 0
    cloud_fraction_prev: jnp.ndarray | None = None,  # (nlev,) [0,1]  previous-step cloud cover (ECHAM zaclc_tm1, #955); None = this step's cloud_fraction (the "cover grew" trigger below never fires)
    convective_base_activated_cdnc: jnp.ndarray | None = None,  # scalar  1/m³  the ACTIVATION POTENTIAL (never cloud-fraction-weighted, #1160) gathered at the convective cloud-base level (ECHAM cdncact_cv at iclbas, #955); None = 0 (no convective droplet source)
    has_convective_cloud_base: jnp.ndarray | None = None,  # scalar  bool  this column has an active convective updraft this step (ECHAM iclbas>0, #955); None = False
    activation_potential: jnp.ndarray | None = None,  # (nlev,) 1/m³  the UNMASKED activation potential (ECHAM zcdncact/cdncact_cv) -- feeds all THREE Fortran sites that read it unmasked (742-782, 889-941, and update_in_cloud_water's own 2594-2608, jax-gcm#1160/#1161): the two section-1 droplet-number sources below AND the per-level scan's section-5.5 re-activation; None = activated_cdnc (JAM's is already unmasked, so this is a no-op there; for MACv2-SP this reproduces the pre-#1160/#1161 behaviour, i.e. a caller that has not been updated to pass the real potential)
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

    Section 1 also runs two droplet-number sources, vectorized over the
    whole column BEFORE the scan (ECHAM runs them before its own level
    loop too — ``mo_cloud_micro_2m.f90``'s ``DO 831`` starts well after
    both): :func:`~jcm.physics.clouds.cloud_utils.
    stratiform_activation_at_cloud_base` (742-782) replenishes CDNC at
    cloud base on four parcel-theory triggers and copies the result up
    through the cloud, and :func:`~jcm.physics.clouds.cloud_utils.
    convective_detrainment_droplet_number` (889-941) gives the convective
    detrained liquid a droplet number and blends it into CDNC by in-cloud
    liquid water content (jax-gcm#955). Both act on the SAME pre-scan
    ``cdnc0`` that used to go straight from the entry floor to the
    ``minimum_CDNC`` floor with nothing in between — a column that never
    satisfies either source's gates reaches the scan with exactly the
    value it always did, since the ``minimum_CDNC`` floor is unchanged,
    just applied after these two instead of before them. Both read
    ``activation_potential``, not ``activated_cdnc`` (jax-gcm#1160):
    ECHAM's own ``zcdncact``/``cdncact_cv`` is never cloud-fraction-
    weighted at any of the three Fortran sites that read it (742-782,
    889-941, and ``update_in_cloud_water``'s own 2594-2608), so a
    cover-masked stand-in reads a spurious zero wherever the cell it is
    read at (a convective base, say) is not the cell its masking was
    computed for. The third site runs inside the per-level scan below,
    which is fed the resolved ``activation_potential`` (not the raw
    ``activated_cdnc`` parameter) for exactly the same reason
    (jax-gcm#1161).

    Per level, ordered by ECHAM's section numbering (numbers = Fortran
    comments) — warm precipitation runs after condensation and activation
    (7.1), the one deliberate deviation left; everything through section 4
    now matches ECHAM's own order exactly (#972/#1110/#1116). ECHAM's
    single ``melting_snow_and_ice`` routine (3.1) is two calls here, back
    to back, both before sublimation and sedimentation — see
    :func:`melt_falling_flux` and :func:`melt_in_cloud_ice`'s own
    docstrings for why it is split at all (two numerically independent
    pools) and the #1116 history of where the in-cloud piece sat before
    this PR:

      3.1   Melt of the falling snow/ice-crystal FLUX entering the level
            (:func:`melt_falling_flux`) and of the in-cloud ice ALREADY
            IN this level (:func:`melt_in_cloud_ice`) where ``T_m1 >
            tmelt`` — matching ECHAM's own order for both pieces: they
            run before both sublimation and this level's own
            sedimentation (F 1170 < 1200 < 1221), so a warm level's own
            ice is melted before any of it could sediment out as a
            still-frozen flux (ECHAM's `picnc`/`pxite` are melt's INOUT
            arguments at F 1176-1182 and sedimentation's own F 1227-1232
            input reads them post-melt).
      3.2/3 Snow/ice sublimation + rain evaporation
            (:func:`sublimation_snow_and_ice_evaporation_rain`) on the
            post-melt flux ENTERING the level — matching ECHAM's own
            order for this piece too (F 1200 < 1221). #972/#1110: an
            earlier version of this sweep ran sublimation AFTER this
            level's own sedimentation (sediment → melt → sublimate), so
            a level with no incoming ice (the ordinary case for the
            first level that sediments any) sublimated ice it had
            itself just produced, before that ice ever reached the
            level below — a ~1.8% low mass/number flux into a purely-
            receiving level, compounding across stacked
            detraining+sedimenting levels into the "unexplained" ICNC
            residual #972 first measured.
      4.    Ice sedimentation (:func:`sedimentation_ice`) of the
            melt-adjusted ice present before this step's convective
            detrainment (ECHAM ``zxip1 = pxim1 + ztmst·pxite``,
            1227-1248, with the melt-updated ``pxite``), after which the
            crystal number of the detrained ice ``znidetr`` joins the ICNC
            (1251-1252).
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
            post-freezing ice (:func:`WBF_process`). The freezing is
            ECHAM-HAM's contact + immersion rates
            (:func:`het_mxphase_freezing`, F 2675-2840) fed either a
            prognostic aerosol's HAM inputs (``freezing_aerosol``, JAM) or,
            without one, ECHAM's own ``lccnclim`` climatology
            (:func:`~jcm.physics.clouds.lohmann_2m.ccnclim.
            ccnclim_freezing_aerosol`, #1149, the ``aerosol_free_freezing``
            default); ``AerosolFreeFreezing.DEMOTT`` keeps jcm's older,
            non-ECHAM closure (freeze droplets up to the DeMott (2010) INP
            number) as an explicit opt-in.
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

    Inputs, in ECHAM's terms (``ztmst = dt``; ``cloud_micro_interface``
    receives the same set, ``physc.f90:1073-1081``):

    - the anchor state ``temperature_m1``, ``specific_humidity_m1``,
      ``qc_m1``, ``qi_m1``, ``qnc_m1``, ``qni_m1`` — ECHAM's ``ptm1``,
      ``pqm1``, ``pxlm1``, ``pxim1`` and ``pxtm1`` of the number tracers.
      Saturation, the step-start subsaturations, the moist heat capacity and
      every other quantity ECHAM evaluates at the previous time level are
      evaluated here;
    - the increments ``*_increment`` — ECHAM's ``ztmst·ptte``,
      ``ztmst·pqte``, ``ztmst·pxlte``, ``ztmst·pxite``, ``ztmst·pxtte``:
      everything that changed the state since the anchor (the dynamics and
      every physics process upstream of the cloud scheme), EXCLUDING
      convective detrainment. Omitted increments are zero;
    - ``detrained_qc``, ``detrained_qi`` — ECHAM's ``ztmst·pxtecl``,
      ``ztmst·pxteci``: the liquid and ice parts of this step's convective
      detrainment, as mass per step, kept apart from the increments.
      Omitted is zero. ECHAM's 2M uses only their sum ``zxtec`` (the
      boundary condition 'Detrained condensate', 555-570): it is not
      sedimented this step, it brings its own crystal number ``znidetr``
      where ECHAM's ``ll_cv`` holds, and the section-4 ``lo2`` criterion
      re-splits it into ice and liquid.

    The provisional state the returned tendencies are relative to is formed
    here as ``anchor + increment`` (``anchor + (increment + detrained)`` for
    the condensate, as convection split it), the same expression
    :func:`jcm.physics.clouds.cloud_inputs.cloud_scheme_inputs` uses, so the
    term's and the column's provisional state agree to the bit. With every
    increment omitted the provisional state is the anchor, and section 5
    reduces to a pure saturation adjustment.

    The large-scale vertical velocity (#705) enters ECHAM's ``zvervx``
    updraft (the WBF/lo2 gates, Section 1), the heterogeneous mixed-phase
    freezing's immersion cooling rate (Section 6.2, ``het_mxphase_freezing``)
    and the ``knvb``/``lonacc`` inversion-level exception on ``zauloc``
    (Section 7) via ``vertical_velocity``/``lonacc_mask``. Both fall back to
    "no large-scale contribution" (zeros/``None``) when the host has no
    pressure-velocity provider — see
    :class:`~jcm.physics.clouds.lohmann_2m.Lohmann2MMicrophysics`'s docstring
    for which configurations that is.

    qnc / qni are stored per kg of air; the scheme interior uses per-m^3,
    so we convert at the boundary.
    """
    def _or_zeros(value):
        return jnp.zeros_like(qc_m1) if value is None else value

    temperature_increment = _or_zeros(temperature_increment)
    humidity_increment = _or_zeros(humidity_increment)
    qc_increment = _or_zeros(qc_increment)
    qi_increment = _or_zeros(qi_increment)
    qnc_increment = _or_zeros(qnc_increment)
    qni_increment = _or_zeros(qni_increment)
    detrained_qc = _or_zeros(detrained_qc)
    detrained_qi = _or_zeros(detrained_qi)
    cirrus_aerosol_number = _or_zeros(cirrus_aerosol_number)
    vertical_velocity = _or_zeros(vertical_velocity)
    lonacc_mask = (jnp.zeros_like(qc_m1, dtype=bool) if lonacc_mask is None
                  else lonacc_mask)
    cloud_fraction_prev = (cloud_fraction if cloud_fraction_prev is None
                          else cloud_fraction_prev)
    convective_base_activated_cdnc = (
        jnp.array(0.0, dtype=qc_m1.dtype)
        if convective_base_activated_cdnc is None
        else convective_base_activated_cdnc)
    has_convective_cloud_base = (
        jnp.array(False) if has_convective_cloud_base is None
        else has_convective_cloud_base)
    # Falls back to ``activated_cdnc`` (see the parameter comment above):
    # for JAM the two values already coincide, so this is a true no-op;
    # the fallback exists only so a caller that passes neither keeps
    # running rather than raising.
    activation_potential = (activated_cdnc if activation_potential is None
                            else activation_potential)

    # ------------------------------------------------------------------
    # Upstream increments (ECHAM's accumulated tendencies × ztmst)
    # ------------------------------------------------------------------
    dT_up = temperature_increment                 # ztmst·ptte
    dq_up = humidity_increment                    # ztmst·pqte
    # The condensate increments EXCLUDE this step's convective detrainment:
    # ECHAM keeps it out of pxlte/pxite and hands it to the 2M separately as
    # zxtec (555-570), and the sweep treats it by its own rules (not
    # sedimented this step, own crystal number, re-split by lo2).
    dqc_up = qc_increment                         # ztmst·pxlte
    dqi_up = qi_increment                         # ztmst·pxite
    zxtec = detrained_qc + detrained_qi           # ztmst·zxtec, both phases

    # Provisional state (ECHAM ptm1 + ztmst·ptte, pxlm1 + ztmst·(pxlte +
    # pxtecl), ...): what the returned tendencies are relative to. The
    # condensate carries the detrainment as convection split it.
    temperature = temperature_m1 + dT_up
    qc = qc_m1 + (qc_increment + detrained_qc)
    qi = qi_m1 + (qi_increment + detrained_qi)
    qnc = qnc_m1 + qnc_increment
    qni = qni_m1 + qni_increment

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
    # taken against the RAW provisional tracers (anchor + increment,
    # unfloored), the state the host adds them to. ECHAM passes the unfloored
    # pxtm1(:,jk,idt_cdnc/idt_icnc) to update_tendencies_and_important_vars
    # (1781) and replaces its tendency with pxtte = (n/ρ − pxtm1)/ztmst
    # (3625-3628); for jcm's additive host the same end state is reached from
    # the provisional tracer. The end-of-step tracer is then exactly the
    # scheme's number, so an out-of-range raw value is removed within the
    # step instead of being carried and re-floored.
    qnc_raw = qnc
    qni_raw = qni
    inv_rho = 1.0 / jnp.maximum(air_density, eps_dt)

    # Number-per-kg-of-air → per-m^3 at the scheme's API boundary, where
    # ECHAM applies the floor (per m³, 600-605). ``cdnc0`` here is ECHAM's
    # own entry value (cqtmin-floored ONLY, lines 600-601) — the two
    # activation sources below (#955) need exactly this, unfloored by
    # ``minimum_CDNC``, to tell whether a cell has already been reactivated
    # this step; the ``minimum_CDNC`` floor itself is applied once, at the
    # end of both, in its original place in ECHAM's own ordering (floored
    # only much later, in ``update_in_cloud_water``'s section 5.5).
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
    # CDNC with it at entry in every cell (deferred below, past the two
    # activation sources).
    inv_cf_min = 1.0 / jnp.maximum(cloud_fraction, params.epsec)
    qc_in_cloud_kgm3 = jnp.where(
        cloud_fraction > params.epsec,
        (qc_m1 + dqc_up) * inv_cf_min * air_density, 0.0,
    )
    cdnc_min = minimum_CDNC(qc_in_cloud_kgm3, params)

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
    # SIGNED water supersaturation (ECHAM zsusatw, the full ``q/qsw - 1``,
    # not the negative-only form above): one of the four cloud-base
    # activation triggers (#955, mo_cloud_micro_2m.f90:747).
    water_supersaturation = (
        specific_humidity_m1 / jnp.maximum(qsat_water, params.epsec) - 1.0
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
    # with the turbulent term zeroed at the lowest level (line 815) and the
    # large-scale term from the host's pressure velocity (#705;
    # ``vertical_velocity`` is ECHAM's ``pvervel``, Pa/s, positive
    # descending -- hence the sign flip to a positive cm/s updraft on
    # ascent, matching the turbulent term's sign).
    updraft_velocity = (turbulent_updraft_velocity(tke, params)
                        - 100.0 * vertical_velocity / (c.grav * air_density))

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

    # ------------------------------------------------------------------
    # Section-1 droplet-number sources (ECHAM 742-782, 889-941; #955)
    # ------------------------------------------------------------------
    # Stratiform activation at cloud base, copied to the cloud above it
    # (742-782): a SEPARATE, earlier replenishment path from the
    # ``cdnc <= cdnc_min`` re-activation ``update_in_cloud_water`` (the
    # scan's section 5.5) already applies — see
    # :func:`~jcm.physics.clouds.cloud_utils.stratiform_activation_at_cloud_base`
    # for the four triggers and the cloud-base search. Acts on ``cdnc0``
    # before the ``minimum_CDNC`` floor (deferred to the end of this
    # block, as ECHAM applies it only much later). Reads
    # ``activation_potential``, NOT ``activated_cdnc``: F749's
    # ``ztmp1_2d = zcdncact(...) - zcdnc(...)`` reads ECHAM's own
    # ``zcdncact`` directly, with no ``paclc``/cloud-fraction factor
    # anywhere in the Fortran (jax-gcm#1160) — the base being "in cloud"
    # comes entirely from ``is_base`` (this function's own cloud-base
    # search), not from scaling the activation value by cover.
    cdnc0 = stratiform_activation_at_cloud_base(
        cdnc0, cdnc_min, activation_potential, temperature_m1,
        dqc_up, cloud_fraction, cloud_fraction_prev,
        water_supersaturation, params,
    )

    # Droplet number of detrained convective liquid ``zqlnuccv`` (889-941):
    # the lesser of the activated number at this level and at the
    # convective cloud base, LWC-weighted into CDNC, never decreasing —
    # see :func:`~jcm.physics.clouds.cloud_utils.
    # convective_detrainment_droplet_number` for why ``activation_potential``
    # stands in for ECHAM's separate ``cdncact_cv`` diagnostic (the
    # deliberate simplification jax-gcm#955 calls "build it once"). BOTH
    # operands of its ``min()`` (this level, and the convective base) read
    # the UNMASKED potential (jax-gcm#1160): F905-917's ``cdncact_cv`` is
    # read unmasked at both the current level and (gathered through
    # ``iclbas``) the convective base, so using the cover-weighted
    # ``activated_cdnc`` at either operand would read a spurious zero in a
    # convective column sitting over clear stratiform sky, the exact gap
    # the Codex review on PR #1160 found at the convective-base gather.
    # Acts on the OUTPUT of the stratiform activation above, exactly as
    # ECHAM's ``zcdnc`` carries both updates in sequence (section 1, lines
    # 782 then 933-939, both before the main level loop).
    cdnc0 = convective_detrainment_droplet_number(
        cdnc0, activation_potential, convective_base_activated_cdnc,
        has_convective_cloud_base, temperature_m1, lo2_2d, zxtec,
        cloud_fraction, qc_m1 + dqc_up, params,
    )

    # The ``minimum_CDNC`` floor, deferred from the entry-floor block
    # above: ECHAM floors CDNC to ``cdnc_min`` only in
    # ``update_in_cloud_water`` (section 5.5), well after both sources
    # above; applying it here (once, after both) rather than at entry
    # reproduces that ordering without re-running the floor twice.
    cdnc0 = jnp.maximum(cdnc0, cdnc_min)

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

    # Heterogeneous mixed-phase INP [1/m³] of the AerosolFreeFreezing.DEMOTT
    # fallback (section 6.2, not the default since #1149): the DeMott (2010)
    # diagnostic on prescribed coarse aerosol at ambient density, or a larger
    # external INP a caller supplies as ``ice_nuclei`` (no in-tree
    # composition does). Unused under LCCNCLIM or when ``freezing_aerosol``
    # selects a prognostic aerosol's rates.
    demott_floor = demott2010_inp(
        temperature_m1, params.n_aer_coarse, air_density)
    n_inp = jnp.maximum(ice_nuclei, demott_floor)
    # The HAM freezing inputs ride the level scan with the TKE the immersion
    # cooling rate needs (F 2800). ECHAM reads the previous step's TKE
    # (ptkem1); this is the TKE the scheme receives, the one its updraft
    # zvervx also uses.
    #
    # Three-way static Python switch (#1149): a prognostic aerosol's own HAM
    # inputs always win when published; otherwise ``aerosol_free_freezing``
    # picks between ECHAM's ``lccnclim`` climatology (the default) and the
    # DeMott fallback. ``freezing_levels`` is non-empty in BOTH the JAM and
    # the ``lccnclim`` case -- ``het_mxphase_freezing`` runs identically on
    # either's fractions, so the per-level scan below (``freezing_k``) does
    # not need to know which one it got; only the DeMott fallback leaves it
    # empty, keeping its own closure (the ``else`` branch inside the scan).
    if freezing_aerosol is not None:
        ham_inputs = freezing_aerosol
    elif params.aerosol_free_freezing == AerosolFreeFreezing.LCCNCLIM:
        ham_inputs = ccnclim_freezing_aerosol(air_density, params)
    else:
        ham_inputs = None

    if ham_inputs is None:
        freezing_levels = ()
    else:
        freezing_levels = (
            tke, ham_inputs.dust_soluble,
            ham_inputs.dust_insoluble_accumulation,
            ham_inputs.dust_insoluble_coarse,
            ham_inputs.bc_soluble, ham_inputs.bc_insoluble,
            ham_inputs.wet_radius_insoluble_aitken,
            ham_inputs.wet_radius_insoluble_accumulation,
            ham_inputs.wet_radius_insoluble_coarse,
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
         activation_potential_k, n_inp_k, inp_dep_k, papnx_k, is_bottom_k,
         lvdcp_k, lsdcp_k, freezing_k,
         vervel_k, zauloc_off_k) = level_in

        zero_s = jnp.zeros_like(cf_k)

        # --- Kaercher-Lohmann cirrus homogeneous nucleation (#552, #1017
        # task 2 part 2b) ------------------------------------------------
        # ECHAM computes ZSUSATIX and calls XFRZMSTR once per level inside
        # SECTION 1's OWN loop (mo_cloud_micro_2m.f90:1009-1038) -- BEFORE
        # sedimentation, melting or section 5 -- on the step-start humidity
        # sice = pqm1/zqsi - 1 (:688-689), never section 5's adjusted
        # zqp1tmp/zqsp1tmp (part 2b's claim (b)). The result (``znicex``)
        # is used TWICE in the real Fortran: added directly into zicncq
        # here (:1050-1058), and reused, UNCHANGED, as ``pnicex`` at
        # update_in_cloud_water's own call far later (:1507) -- not a
        # second xfrzmstr call. Computed once here and threaded through
        # for both, matching that structure (an earlier version of this
        # port only wired the second use, which this task's own end-to-end
        # harness verification caught: section 1's direct addition is
        # where nucleation actually reaches ICNC on every column except
        # the rare one where pre-existing ice already exceeds icemin, so
        # skipping it left non-degenerate cirrus columns pinned at the
        # floor regardless of the aerosol number).
        #
        # The depletion reference at THIS point is ECHAM's zicncq right
        # before its nic_cirrus block (:982) -- picnc's entry value plus
        # this step's detrained-ice number, BEFORE sedimentation or
        # melting have touched it (both run later, in section 4) -- not
        # ``icnc_sedi`` (the post-sedimentation, post-join quantity that
        # does not exist yet at this point in the sweep; part 2b's claim
        # (a), also caught by the harness verification).
        #
        # ``nic_cirrus`` is a static (pytree_node=False) config field, so
        # the plain Python ``if`` does not need to trace both branches --
        # identical to how ``assembly.py``'s own
        # ``nic_cirrus==1``/``==2`` dispatch works.
        if params.nic_cirrus == 2:
            sice_k = jnp.maximum(
                q_m1_k / jnp.maximum(qsi_k, params.eps) - 1.0, 0.0)
            zicncq_early_k = icnc0_k + znidetr_k
            # zapnx = MAX(1e-6*(papnx - zicncq), 1e-6) [1/cm3]
            # (mo_cloud_micro_2m.f90:1015,1018): the available aerosol
            # number depleted by the ICNC already present.
            apn_cm3_k = jnp.maximum(
                1.0e-6 * (papnx_k - zicncq_early_k), 1.0e-6)
            # verv_k is ECHAM's zvervx, already [cm/s] (turbulent_updraft_
            # velocity's own docstring); xfrzmstr's own contract takes m/s
            # (matching its updraft argument's name), so convert back.
            cirrus_ri_raw_k, cirrus_pnicex_k = xfrzmstr(
                sice_k, verv_k / 100.0, apn_cm3_k, t_m1_k, p_k, dt, params)
            # The SAME depleted number, converted back to 1/m3, is HAM's
            # ``pap`` -- update_in_cloud_water's own cap on the candidate
            # (#552; see that function's nic_cirrus==2 branch).
            cirrus_aerosol_number_available_k = apn_cm3_k * 1.0e6
            # zninucl = MERGE(MIN(zap*1e6, znicex), 0, ll_ice) (:1050-1058):
            # section 1's OWN cap+gate on znicex for the early join below --
            # DIFFERENT from the raw znicex ``cirrus_pnicex_k`` itself
            # passed to update_in_cloud_water, which applies its own,
            # separate MIN(pnicex, pap*1e6) cap at a different point
            # (:2618). ``ll_ice = (zsusatix>0)&(ptm1<zthomi)`` is already a
            # SUBSET of xfrzmstr's own internal gate, so ``cirrus_pnicex_k``
            # is already exactly 0 wherever ll_ice is false -- only the cap
            # at ``cirrus_aerosol_number_available_k`` remains to apply.
            zninucl_k = jnp.minimum(
                cirrus_pnicex_k, cirrus_aerosol_number_available_k)
            # The Kaercher-Lohmann deposition rate ``zqinucl``
            # (mo_cloud_micro_2m.f90:1046-1102): the vapour
            # deposited onto the crystals ``xfrzmstr`` just nucleated, with
            # ventilation, computed here (step-start quantities only) and
            # carried to section 5 below, matching ECHAM's own section-1
            # placement -- entirely before sedimentation/melting/section 4.
            # The ICNC ``icnc_before_floor`` is ECHAM's ``zicncq`` right
            # after ``+= zninucl`` (:1058), the SAME pre-sedimentation sum
            # ``icnc_sedi`` below adds (:982,1050-1058); this call floors
            # it independently (it is read, not mutated, before that add).
            _icncq_floored_k, zqinucl_k = karcher_lohmann_deposition_rate(
                cirrus_ri_raw_k, zrid_k, zicncq_early_k + zninucl_k,
                qi_m1_k, cf_k, rho_k, adc_k, visc_k, sice_k, t_m1_k, p_k,
                esi_k, q_m1_k, qsi_k, dt, params)
        else:
            cirrus_pnicex_k = inp_dep_k
            cirrus_aerosol_number_available_k = zero_s
            # NOT inp_dep_k: zninucl's own nic_cirrus=1 formula (zascs-based,
            # #955) is unimplemented and distinct from ice_nuclei_deposition
            # regardless -- this early join must stay exactly 0 whenever
            # nic_cirrus != 2, never leak a future #679 producer into it.
            zninucl_k = zero_s

        # --- 3.1a Melt the falling snow/ice flux ENTERING the level ----
        # ECHAM's melt (F 1170-1190) runs first, on the flux the level
        # above produced, before this level's own sublimation (1200) or
        # sedimentation (1221) touch it. Split out of the single
        # ``melting_snow_and_ice`` call (:func:`melt_falling_flux`) so it
        # can sit here, where ECHAM has it, next to :func:`melt_in_cloud_ice`
        # below — see both functions' docstrings for why the one Fortran
        # routine is now two calls, and #1116 (closed by this reorder) for
        # why they are called TOGETHER, here, rather than split across
        # sedimentation.
        (rain_flux, snow_flux, ice_flux, ice_flux_n, psmlt_a,
         pximlt_k) = melt_falling_flux(
            t_m1_k, dp_k, lsdcp_k, lvdcp_k,
            rain_flux, snow_flux, ice_flux, ice_flux_n,
            dt, params,
        )

        # --- 3.1b Melt the in-cloud ice ALREADY PRESENT this level ------
        # ECHAM's melt (F 1170-1190) is ONE call touching both the falling
        # flux (above) and the in-cloud ice (here) before sedimentation
        # (F 1221) ever runs: `picnc`/`pxite` are its INOUT arguments
        # (F 1176-1182), and F 1227's ``zxip1 = pxim1 + ztmst·pxite`` reads
        # the melt-adjusted ``pxite`` sedimentation then acts on — so a
        # warm level (``T_m1 > tmelt``) melts its own in-cloud ice BEFORE
        # any of it can sediment out as still-frozen flux; it is a
        # physical impossibility in ECHAM for a warm level to emit a
        # falling-ice flux it created from its own melted ice.
        #
        # An earlier version of this fix (#972/#1110) kept this call AFTER
        # sedimentation, carrying forward the position jcm had used since
        # #662 (micro_pumas_v1.F90's sediment-before-melt, cited there as
        # "fine" because the FINDING #662 fixed was the discarded melt
        # tendency, not the order — #662 never claimed ECHAM's own order
        # was wrong or unneeded). That position was safe only incidentally,
        # because of the self-sublimation/double-spend bugs this PR fixes
        # above: once sublimation and flux-melt correctly see the flux
        # ENTERING the level, a warm level with any in-cloud ice would
        # sediment it as a real outgoing flux before this call ever got a
        # chance to melt it (a local Codex review caught this on the fixed
        # code — #1116, closed here by retiring the deviation rather than
        # leaving it dormant). ``ice_cloud_previous=qi_m1_k`` and
        # ``ice_tendency=dqi_up_k/dt`` are ECHAM's own ``pxim1``/``pxite``
        # at F 1176 (the upstream tendency only — sedimentation has not
        # run yet), exactly matching ECHAM's own pre-sedimentation melt
        # input instead of jcm's old post-sedimentation one.
        pre_sed_ice_tend_k = dqi_up_k / dt
        (icnc_premelt_k, _qmel, cdnc_melt, ice_tend_pre_sed_k,
         pimlt_k) = melt_in_cloud_ice(
            melt_k, qi_m1_k,
            icnc0_k, icnc0_k,
            jnp.array(0.0),  # qmel accumulator
            cdnc0_k,
            pre_sed_ice_tend_k,
            dt,
            params,
        )
        # The ice actually entering sedimentation (F 1227-1228): ECHAM's
        # own post-melt ``pxim1 + ztmst·pxite``, MAX'd at EPSILON(1d0) —
        # this scheme's 0 floor, as already documented at the
        # ``sedimentation_ice`` call below.
        zxip1_pre_for_sed = jnp.maximum(
            qi_m1_k + dt * ice_tend_pre_sed_k, 0.0)

        # --- 3.2/3.3 Sublimation of snow/falling ice + rain evap -------
        # ECHAM calls this (F 1200) on the flux ENTERING the level, after
        # its own melt has already acted on it (above) and strictly
        # before this level's own sedimentation (F 1232) adds to it. Run
        # here, before :func:`sedimentation_ice` below, using the carry
        # exactly as this level received it (post-melt): ``falling_ice_
        # frac``, ``ice_flux`` and ``ice_flux_n`` are still the level-
        # above's OUTGOING values (sedimentation has not yet touched them
        # this level), matching ECHAM's ``zclcfi``/``zxiflux``/
        # ``zxifluxn`` at the point its own section 3.2/3.3 reads them.
        #
        # #972/#1110: before this reorder, the call sat AFTER
        # ``sedimentation_ice`` (sediment → melt → sublimate), so
        # ``falling_ice_frac``/``ice_flux``/``ice_flux_n`` were already
        # THIS level's own just-settled output. A level with no incoming
        # ice (the ordinary case for the first level that sediments any)
        # then spuriously sublimated part of the ice it had itself only
        # just created, before that ice ever reached the level below —
        # measured as a ~1.8% low mass/number flux into a purely
        # receiving level on the `sediment_then_detrain` fixture column
        # (#1110), and, compounded across the three stacked
        # detraining+sedimenting levels of `detrainment_mixed_lo2_true`
        # (35→36→37), the "unexplained" ICNC residual #972 first measured
        # (growing 4.6e-4 → 1.75e-3 → 3.6e-3 there). Moving sublimation
        # alone (keeping the flux-melt above where it always was, right
        # after sedimentation) would have left the snow pool double-spent:
        # melt and sublimation would both have read the SAME pre-melt
        # snow_flux, so the snow this level sublimated was never actually
        # subtracted from what melt also claimed (a Codex review finding
        # on the first version of this fix, reproduced with a negative
        # outgoing snow flux). Moving the flux-melt piece up here with it
        # restores ECHAM's own melt-then-sublimate sequencing for both
        # pools; the in-cloud melt piece above is in the same position
        # for the same reason (#1116).
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

        # --- 4. Sedimentation of cloud ice (grid-mean) -----------------
        # Acts on the ice present BEFORE this step's convective
        # detrainment and AFTER this level's own in-cloud melt (above),
        # ECHAM zxip1 = pxim1 + ztmst·pxite (1227-1228) with the melt-
        # adjusted ``pxite`` — ``zxip1_pre_for_sed``, computed above.
        # pxite excludes the detrained condensate, which ECHAM carries
        # separately as zxtec. The detrained ice therefore does not fall
        # this step; it joins the in-cloud state after sedimentation
        # through zxidt (1316), with its own crystal number (below).
        # ECHAM floors the input at EPSILON(1._dp) = 2.2e-16; this scheme
        # floors at 0, because the float32 epsilon (1.2e-7 kg/kg) is a
        # sizeable amount of ice. ``ice_flux``/``ice_flux_n``/
        # ``falling_ice_frac`` entering here are the just-melted-and-
        # sublimated incoming values (above), exactly as ECHAM's section 4
        # reads ``zxiflux``/``zxifluxn``/``zclcfi`` after its own 3.1-3.3.
        # ``icnc_premelt_k`` is ECHAM's own melt-adjusted ``picnc`` INOUT
        # (F 1176-1182, icemin where ``melt_k`` holds) — sedimentation
        # reads it at F 1232, after melt has already run.
        (zxip1, icnc_sedi, ice_flux, ice_flux_n, falling_ice_frac,
         mrateps_sedi_k) = sedimentation_ice(
            cf_k, adc_k, dp_k, rho_k, inv_rho_k,
            zxip1_pre_for_sed, icnc_premelt_k,
            ice_flux, ice_flux_n, falling_ice_frac,
            dt, params,
        )
        # ECHAM pxite = (zxip1 − pxim1)/ztmst (1248): the FULL tendency
        # (upstream increment + melt + sedimentation) relative to the
        # UNFLOORED, pre-melt, pre-sedimentation ice ``zxip1_pre`` —
        # ``zxip1`` already reflects the melt-adjusted entry above, so
        # this one subtraction folds both sinks in without claiming
        # either twice (#662 finding 2's double-claim was the discarded
        # tendency across routines, not this algebra).
        zxip1_pre = qi_m1_k + dqi_up_k
        ice_tend_k = (zxip1 - zxip1_pre) / dt

        # The crystal number of the detrained ice -- and, at nic_cirrus=2,
        # the Kaercher-Lohmann cirrus nucleation computed above
        # (mo_cloud_micro_2m.f90's own ``zicncq += znidetr + zninucl``,
        # :982,1050-1058, both folded into one addition here since jcm
        # computes them at the same point) -- joins the post-sedimentation
        # ICNC, capped at icemax (1251-1252). ECHAM's floor at icemin
        # (1253) is not applied: the ICNC lower bound stays cqtmin
        # (znidetr's own floor, 978, as at entry) and number-less ice is
        # re-diagnosed in update_in_cloud_water (see the entry floor).
        icnc_sedi = jnp.minimum(
            icnc_sedi + znidetr_k + zninucl_k, params.icemax)

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
        zrice = ice_volume_mean_radius_schumann(ice_gm3, icnc_sedi, params)
        zvervmax = threshold_vert_vel(
            sat_vap_pres_water=esw_k, sat_vap_pres_ice=esi_k,
            icnc=icnc_sedi, ice_radius=zrice, eta=eta_k, params=params)
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
        # The bounds and the phase split are the 1M scheme's too (shared
        # helper): dissipation limited to the available condensate,
        # condensation to (almost) the available vapour (ECHAM qsec·zqp1,
        # qsec = 1 − cqtmin ≈ xsec); in the liquid-growth regime the full
        # zqcdif condenses (the saturation adjustment of ECHAM #485).
        _zqcdif, zcnd0, zdep0 = sundqvist_condensation(
            dq_up_k, zdqsat, cf_k, zxib, zxlb, zqp1,
            lo2.astype(zqp1.dtype), params.xsec, params.epsec)
        if params.nic_cirrus == 2:
            # ECHAM 1449-1458: at nic_cirrus=2
            # the deposition leg is ``zqinucl`` (computed in section 1
            # above) standing in for ``zqcdif`` -- zcnd is UNAFFECTED
            # (always built from zqcdif, 1437-1441). The Fortran writes
            # ``zdep = zqinucl*zifrac`` unconditionally and only zeroes it
            # where ``ll2 = (NOT dissipation) & (NOT lo2)`` (1455-1458);
            # since its own ``zifrac`` MERGEs to 1.0 outside dissipation
            # (1443), recomputing the dissipation-branch fraction here and
            # dispatching it exactly as ``sundqvist_condensation`` dispatches
            # zqcdif's own zdep reproduces that MERGE: dissipation uses the
            # clipped in-cloud ice fraction, growth uses the ice/liquid
            # ``lo2`` weight, condensation (ll2) is zero either way.
            dissipation = _zqcdif < 0.0
            zifrac = jnp.clip(
                zxib / jnp.maximum(zxib + zxlb, params.epsec), 0.0, 1.0)
            zdep0 = jnp.where(
                dissipation, zqinucl_k * zifrac,
                lo2.astype(zqp1.dtype) * zqinucl_k)

        # --- 5.4 Supersaturation corrections ---------------------------
        (zcnd, zdep, ztp1tmp, zqp1tmp, zqsp1tmp,
         _zvervmax_dep) = mixed_phase_deposition_and_corrections(
            p_k, icnc_sedi, q_m1_k, cf_k,
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
            cirrus_aerosol_number_available_k,  # pap: NOT pressure (#552)
            # The resolved ``activation_potential`` (jax-gcm#1161), not the
            # raw ``activated_cdnc`` parameter -- see that parameter's own
            # comment and ``update_in_cloud_water``'s ``activated_cdnc``
            # Parameters entry for why this is the unmasked potential.
            activation_potential_k,
            zcnd, zdep,
            zero_s, zero_s,     # Tompkins sources
            cirrus_pnicex_k,    # pnicex: read only by the nic_cirrus=2 branch
            zqp1tmp, zqsp1tmp,
            rho_k,
            zrid_k,             # prid: ECHAM zrid [m] (1511)
            t_m1_k,             # ptm1 (activation gates on step-start T)
            ll_cc,
            icnc_sedi,
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
            # on the HAM freezing inputs of either a prognostic aerosol
            # (JAM's ``freezing_aerosol``) or, without one, ECHAM's own
            # ``lccnclim`` climatology (#1149, the ``aerosol_free_freezing``
            # default) -- the same call either way, since both suppliers
            # fill the identical ``HeterogeneousFreezingAerosol`` slots (see
            # ``freezing_levels``'s construction above the scan). The
            # immersion rate needs the cooling of the vertical motion,
            # ECHAM's ztte = (omega - fact_tke*sqrt(TKE)*rho*g)/(cpd*rho)
            # (2800-2802); ``vervel_k`` is that raw omega (#705) -- the
            # turbulent part is formed inside ``het_mxphase_freezing`` itself
            # from ``tke_k``.
            (tke_k, fdusol_k, fduai_k, fduci_k, fbcsol_k, fbcinsol_k,
             rwetki_k, rwetai_k, rwetci_k) = freezing_k
            (icnc_het, cdnc_h, zfrl, zxib, zxlb,
             new_crystals) = het_mxphase_freezing(
                ll_mxfrz, p_k, tke_k,
                vervel_k,           # pvervel: large-scale omega (#705)
                paclc, fbcsol_k, fbcinsol_k, fdusol_k, fduai_k, fduci_k,
                rho_k, inv_rho_k, rwetki_k, rwetai_k, rwetci_k,
                ztp1tmp, cdnc_min_k,
                icnc_f, cdnc_f, zfrl, zxib, zxlb,
                dt, params.cqtmin, params,
            )
        else:
            # The DeMott (2010) INP-number closure (``AerosolFreeFreezing.
            # DEMOTT``, an explicit non-default opt-in since #1149 --
            # ``lccnclim`` above is ECHAM's own aerosol-free mode and this
            # parameterisation is in neither the ECHAM nor the CAM source
            # tree): freeze one mean-mass droplet per new crystal up to the
            # INP number, moving number, mass AND fusion heat together
            # (#662 finding 3). The new crystals are capped by the droplets
            # available, as the frozen mass is capped by the liquid, so an
            # INP number above CDNC cannot create crystals from nothing.
            # ECHAM's het_mxphase_freezing caps the frozen number at the
            # droplets above cdnc_min (2818-2820); this closure leaves at
            # least cqtmin.
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
        # ``lonacc`` exception (#705, mo_cloud.f90 917-928): zeroed at the
        # cover's diagnosed low-level inversion and the level below it when
        # the air subsides there (``zauloc_off_k``, from
        # :func:`jcm.physics.clouds.echam_1m.lonacc_levels`).
        zauloc = jnp.where(zauloc_off_k, 0.0, zauloc)

        zxlb = jnp.maximum(zxlb, 1.0e-20)
        zxib = jnp.maximum(zxib, 1.0e-20)
        zmlwc_k = zxlb          # in-cloud liquid before rain formation
        zmiwc_k = zxib          # in-cloud ice before snow formation

        # Rain/snow water content diagnosed from the carry fluxes by the
        # Marshall-Palmer inversions (Roeckner et al. 2003 eqs. 10.70 /
        # 10.74; ECHAM 1638-1654). This replaces the dead ``qr``/``qs``
        # tracer reads (#662 finding 5): ECHAM carries no rain/snow
        # tracers — the accretion "rain from above" is the flux the
        # levels above just produced, inverted to a mixing ratio. Shared
        # with the direct Fortran-reference test (jax-gcm#1087) so both
        # feed ``precip_formation_warm``/``_cold`` the identical
        # inversion jcm actually runs.
        zxrp1 = rain_mixing_ratio_from_flux(rain_flux, precip_cover, zqrho_k, params)
        zxsp1 = snow_mixing_ratio_from_flux(snow_flux, precip_cover, params)

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
         _pfevapr, pfrain_k, pfsnow_k, _pfsubls) = update_precip_fluxes(
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
            # The POST-update precipitating-area fraction (the same value
            # that becomes this level's share of ``carry_out``, i.e. AFTER
            # ``update_precip_fluxes`` above) -- ECHAM's own per-level
            # ``zclcpre_2d(:,jk)`` (mo_cloud_micro_2m.f90:1742, written right
            # after its own ``update_precip_fluxes`` call at 1719-1733, and
            # consumed as ``pclc`` by ``cloud_subm_2``/``ham_wetdep`` for
            # HAM's below-cloud scavenging). ``pfrain_k``/``pfsnow_k`` are
            # that SAME call's own ``pfrain``/``pfsnow`` outputs -- the
            # in-cloud, cover-normalised, pre-evaporation rain/snow flux
            # ECHAM threads to ``cloud_subm_2`` as ``zfrain``/``zfsnow``
            # (mo_cloud_micro_2m.f90:1813, the ``CALL cloud_subm_2(...,
            # zfrain(:,:), zfsnow(:,:), ...)`` right after the per-level
            # sweep; ``cloud_subm_2``'s own dummy-argument comment at
            # mo_submodel_interface.f90:1676-1677 reads "rain/snow flux
            # before evaporation [kg/m2/s]", matching exactly). Threaded
            # out here, not added to the carry tuple above (that one is
            # unchanged) or to ``CloudData`` -- see
            # ``configure_wetdep_hydro_diagnostics``.
            precip_cover, pfrain_k, pfsnow_k,
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
        # The RESOLVED ``activation_potential`` (jax-gcm#1161), not the raw
        # ``activated_cdnc`` parameter: section 5.5's re-activation, inside
        # the scan below, is the third Fortran site (2594-2608) that reads
        # ECHAM's ``pcdncact`` unmasked, exactly like the two section-1
        # sources above it in this function -- see this function's own
        # docstring and ``activation_potential``'s parameter comment.
        activation_potential, n_inp, ice_nuclei_deposition, cirrus_aerosol_number,
        is_bottom_level, lvdcp, lsdcp, freezing_levels,
        vertical_velocity, lonacc_mask,
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
     detrainment_move, detrainment_split_dT,
     precip_cover_profile, pfrain_profile, pfsnow_profile) = scan_outs

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
        # The baseline of the number tendencies: the RAW provisional tracers
        # per kg of air (anchor + increment; ECHAM's pxtm1 for its
        # replacing tendency, 1781; see the entry floor). The working
        # cdnc/icnc are per-m³, so the tendency subtracts per-kg from
        # per-m³·1/ρ, and the negative-mass repair removes the whole number
        # where it zeroes the condensate (3632-3652).
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
    # raw value substitutes it: a clear cell (cover exactly 0, below the
    # critical humidity) with a trace of ice deposition comes back with
    # cf = clip(RH, 0.01, 1), and COSP, AeroCom and the JAM cloud-borne /
    # aqueous / wetdep terms all read it. Clipping to the incoming cover
    # keeps the emptying behaviour and drops the closure substitution.
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
        precip_cover_profile, pfrain_profile, pfsnow_profile, \
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

    Its thermodynamic and condensate inputs are ECHAM's: the anchor state,
    the increments since it and the convective detrainment, formed by
    :func:`jcm.physics.clouds.cloud_inputs.cloud_scheme_inputs` from the
    carried post-physics state, the upstream running tendency and the
    convection scheme's detrainment (``clouds.conv_detrainment_qc/qi``,
    written by ``TiedtkeConvection``, zero without a convection term). Reads
    ``cloud_fraction`` from the public ``"clouds"`` key (set by :class:`SundqvistCloudFraction`
    upstream), TKE from ``"vertical_diffusion"``, the activated CDNC (JAM's
    ``activated_cdnc``, or the SPA-style floor from the public ``"aerosol"``
    Nccn), and, where a prognostic aerosol publishes them, ECHAM-HAM's
    heterogeneous-freezing inputs ``"freezing_aerosol"`` (a
    :class:`HeterogeneousFreezingAerosol`), which section 6.2 uses in place
    of its own ``lccnclim`` climatology (#1149, the default without a
    prognostic aerosol) to feed ECHAM's contact + immersion rates. Writes the
    surface rain / snow precip flux into ``"clouds"`` along with the
    number tracers the step started from (``qnc_prev``/``qni_prev``).

    Two droplet-number sources (#955) read two more soft, composition-
    optional diagnostics: ``"cloud_fraction_prev"`` (also published by
    ``SundqvistCloudFraction``, the previous step's cover, for the
    cloud-base activation's "cover grew" trigger) and ``"convection"``
    (published by ``TiedtkeConvection``, the convective cloud-base level
    and whether an updraft is active, for the detrained-liquid droplet
    number). Both default to a no-op without their producer. Both sources
    feed on ``"activation_potential"`` (#1160), not ``"activated_cdnc"``:
    JAM's ``ArgActivation``/``HamActivation`` publish it as a plain alias
    of ``activated_cdnc`` (their own output is never cloud-fraction-
    weighted), and this term derives it for MACv2-SP from
    :func:`jcm.physics.aerosol.spa.spa_activation_potential` — the
    unmasked counterpart of the ``spa_activated_cdnc`` floor, matching
    ECHAM's own ``zcdncact``/``cdncact_cv`` (never cover-weighted; see
    that module's docstring). The scan's own section-5.5 re-activation
    (``update_in_cloud_water``) reads the SAME unmasked
    ``activation_potential`` (jax-gcm#1161) for the identical reason as
    the two sources above it — ECHAM's own F2594-2608 never cloud-
    fraction-weights ``pcdncact`` either; ``"activated_cdnc"`` now serves
    only as the fallback value for ``activation_potential`` when a
    caller does not publish the latter.

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
    # ``reffl``/``reffi`` are the scheme's own in-cloud droplet and crystal
    # effective radii, ECHAM's ``mo_activ`` ``REFFL``/``REFFI`` streams, which
    # HAM's in-cloud impaction scavenging reads (``ic_scav_imp``).
    provides: ClassVar[tuple[str, ...]] = (
        "autoconv", "accretn", "wbf", "clouds", "reffl", "reffi",
    )
    # ECHAM's anchor (ptm1, pqm1, pxlm1, pxim1, pxtm1 of cdnc/icnc) is the
    # previous step's post-physics state; the model carries it for this term
    # (``jcm.physics.clouds.cloud_inputs``).
    requires_post_physics_fields: ClassVar[tuple[str, ...]] = (
        "temperature", "specific_humidity", "qc", "qi", "qnc", "qni",
    )
    # Composition-optional soft reads, each with a documented fallback at
    # its ``__call__`` site (see the ``vertical_diffusion`` comment above,
    # and the aerosol-inputs comment below for four more): ``activated_cdnc``
    # / ``freezing_aerosol`` are JAM-only (upstream ``ArgActivation`` /
    # ``IceNucleation``); ``cirrus_aerosol_number`` is published by
    # ``IceNucleation`` only for a population with ``cirrus_aerosol_modes``
    # (M7); ``ice_nuclei`` / ``ice_nuclei_deposition`` are currently unfed by
    # any in-tree term at all, which the ``__call__`` comment explains
    # (jax-gcm#673). ``inversion_level``: published by
    # ``SundqvistCloudFraction`` (#705); read via ``.get()`` so this term
    # stays composable without it (falls back to "no lonacc exception", see
    # ``cache_coords``/``__call__``). ``cloud_fraction_prev``: also
    # published by ``SundqvistCloudFraction`` (#955, the previous step's
    # cover); falls back to this step's own cover (the "cover grew"
    # activation trigger becomes a no-op). ``convection``: published by
    # ``TiedtkeConvection`` (#955, the convective cloud-base bookkeeping
    # the detrained-liquid droplet source needs); falls back to "no
    # convective base" (that source is correctly a no-op without it).
    # ``activation_potential``: published by ``ArgActivation``/
    # ``HamActivation`` as a plain alias of ``activated_cdnc`` (#1160,
    # their own output is never cloud-fraction-weighted); falls back to
    # the locally-computed ``spa_activation_potential`` under MACv2-SP,
    # or to ``activated_cdnc`` itself if even that is absent (a direct-
    # construction unit test with no aerosol diagnostics at all).
    optional_requires: ClassVar[tuple[str, ...]] = (
        "vertical_diffusion", "activated_cdnc", "activation_potential",
        "freezing_aerosol", "cirrus_aerosol_number", "ice_nuclei",
        "ice_nuclei_deposition", "inversion_level", "cloud_fraction_prev",
        "convection",
    )
    # CF/units metadata for the ``clouds.*`` output fields (#740). Shared with
    # the cover term; this term fills the precip / 2M / process-rate fields.
    output_attrs: ClassVar[dict[str, dict[str, str]]] = {
        **CLOUD_OUTPUT_ATTRS,
        "reffl": {"units": "um",
                  "long_name": "in-cloud droplet effective radius after the "
                               "two-moment microphysics"},
        "reffi": {"units": "um",
                  "long_name": "in-cloud ice crystal effective radius after "
                               "the two-moment microphysics"},
    }

    def __init__(self, params: 'CloudParams2M | None' = None, *,
                 params_are_defaults: bool = False):
        """Hold the scheme-native :class:`CloudParams2M`.

        ``params_are_defaults`` marks parameters that are the resolution
        defaults of a grid rather than the caller's own choice (the factory
        and the Hydra runner set it), so ``cache_coords`` can check them
        against the grid; parameters left ``None`` are defaults too.
        """
        self.params_are_defaults = params is None or params_are_defaults
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
        # Off by default: the ``"precip_cover"``/``"pfrain"``/``"pfsnow"``
        # diagnostics keys this scheme can publish (see
        # ``configure_wetdep_hydro_diagnostics``) would otherwise change the
        # diagnostics dict's key set on every existing composition, a
        # default-path change. Deliberately NOT fields on
        # ``CloudData``/``"clouds"``: that struct rides the physics carry, so
        # new fields there would change the checkpoint pytree of every 2M
        # preset and break restart from already-stamped checkpoints. A plain
        # Python bool, not an ``nnx.Param``: it selects which diagnostics
        # keys this term writes, not a differentiable quantity.
        self._publish_wetdep_hydro = False
        # Set by ``cache_coords``; ``None`` disables the ``lonacc`` exception
        # (#705) rather than raising, so a caller that never cached coords
        # (e.g. an existing direct-construction unit test) keeps working.
        self._inversion_range: tuple[int, int] | None = None

    def cache_coords(self, coords) -> None:
        """Warn if default parameters were built for another grid.

        The parameters were fixed at construction and are not re-resolved;
        parameters the caller supplied are not checked.

        Also caches ``jbmin``/``jbmax`` (#705) via
        :func:`~jcm.physics.clouds.echam_1m.cached_inversion_range` -- see
        that function's docstring for why this is independent of
        :class:`~jcm.physics.clouds.sundqvist.SundqvistCloudFraction`'s own
        copy, and tolerant of a coords object with no level boundaries.
        """
        if self.params_are_defaults:
            from jcm.physics.resolution_defaults import check_defaults_grid
            check_defaults_grid(type(self).__name__,
                                self.params.get_value().defaults_truncation, coords)
        self._inversion_range = cached_inversion_range(coords)

    def configure_spa(self, prefactor: float, exponent: float,
                      cap_smoothing: float = 0.0) -> None:
        """Set the SPA prefactor / exponent / cap-smoothing (factory hook)."""
        self._spa_prefactor = nnx.Param(jnp.asarray(prefactor))
        self._spa_exponent = nnx.Param(jnp.asarray(exponent))
        self._spa_cap_smoothing = nnx.Param(jnp.asarray(cap_smoothing))

    def configure_wetdep_hydro_diagnostics(self, enabled: bool) -> None:
        """Publish the per-level precipitation inputs of HAM's below-cloud scavenging.

        ``"precip_cover"``, ``"pfrain"`` and ``"pfsnow"``. ``echam_physics``
        turns this on exactly when ``jam_wetdep_scheme`` selects a HAM
        below-cloud pathway (jax-gcm#1017). The scheme computes all three in
        its per-level sweep and discards them unless this is enabled:

        - ``"precip_cover"`` is ECHAM's ``pclc``/``zclcpre``: the precipitating
          area fraction after this level's ``update_precip_fluxes``, the value
          that becomes this level's share of the carry.
        - ``"pfrain"``/``"pfsnow"`` are that same call's ``pfrain``/``pfsnow``
          outputs, the in-cloud, cover-normalised rain/snow flux before
          evaporation that ECHAM passes to ``cloud_subm_2`` as
          ``zfrain``/``zfsnow`` (``mo_cloud_micro_2m.f90:1813``). They are not
          the scan's carry ``rain_flux``/``snow_flux`` (ECHAM's column
          ``prfl``/``psfl``), which are post-evaporation and not
          cover-normalised.

        The in-cloud effective radii ``"reffl"``/``"reffi"`` are not part of
        this switch: every 2M run publishes them (the class-level
        ``provides``), for in-cloud impaction scavenging.

        Enabling also adds the three keys to ``self.provides``: ``WetScavenging``
        lists them in its ``requires``, and ``ComposablePhysics._validate_ordering``
        checks that against the upstream ``provides`` at compose time.
        """
        self._publish_wetdep_hydro = bool(enabled)
        keys = ("precip_cover", "pfrain", "pfsnow")
        base = tuple(t for t in type(self).provides if t not in keys)
        self.provides = (*base, *keys) if enabled else base

    def adopt_runtime_configuration(self, previous) -> None:
        """Inherit the SPA tuning and the wetdep-hydro toggle from a displaced 2M term.

        Set by ``echam_physics`` after composition, from the aerosol module's
        parameters, so a term swapped in afterwards would otherwise silently
        fall back to the (1.0, 0.5, 0.0) constructor defaults and change the
        droplet number the whole cloud scheme keys off -- and, for the
        wetdep-hydro flag, would silently stop publishing keys a
        downstream HAM wet-deposition term requires.
        """
        for name in ("_spa_prefactor", "_spa_exponent", "_spa_cap_smoothing"):
            param = getattr(previous, name, None)
            if param is not None:
                setattr(self, name, nnx.Param(jnp.asarray(param.get_value())))
        publish = getattr(previous, "_publish_wetdep_hydro", None)
        if publish is not None:
            self.configure_wetdep_hydro_diagnostics(bool(publish))

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

        # ECHAM's cloud_micro_interface inputs (physc.f90:1073-1081): the
        # anchor state at the previous time level, the increments since then
        # (the dynamics of the last step and every physics term upstream of
        # this one, radiation and vertical diffusion's condensate included),
        # and the convective detrainment by itself. The anchor is the
        # previous step's post-physics state carried by the model where it is
        # valid, the received state otherwise (first step, single column);
        # see ``jcm.physics.clouds.cloud_inputs``.
        inputs = cloud_scheme_inputs(
            state, diagnostics, tracers=("qc", "qi", "qnc", "qni"))
        anchor, increment = inputs.anchor, inputs.increment
        provisional = inputs.provisional

        # Air density: ECHAM's zrho = papm1/(rd·ptvm1)
        # (mo_cloud_micro_2m.f90:578), the anchor's virtual density, with
        # ptvm1 = ptm1·(1 + vtmpc1·pqm1 − (pxlm1 + pxim1)) (physc.f90:267-268),
        # as the 1M forms it, at the received state's pressure (the carry has
        # no pressures; see the 1M's note). The layer depth goes with it,
        # dz = Δp/(ρ·g) (the
        # virtual-temperature depth, as ECHAM's zdz from the geopotential), so
        # the layer mass the column forms as ρ·g·dz stays the moist-air
        # diagnostics' Δp.
        virtual_temperature = anchor.temperature * (
            1.0 + c.vtmpc1 * anchor.specific_humidity
            - (anchor.tracers["qc"] + anchor.tracers["qi"]))
        air_density = pressure_full / (c.rd * virtual_temperature)
        layer_mass = diagnostics["air_density"] * diagnostics["layer_thickness"]
        layer_thickness = layer_mass / air_density

        clouds = diagnostics["clouds"]
        cloud_fraction = clouds.cloud_fraction

        if "vertical_diffusion" in diagnostics:
            tke = diagnostics["vertical_diffusion"].tke
        else:
            tke = jnp.zeros_like(state.temperature)

        # #705: the large-scale pressure velocity, ECHAM's ``pvervel``, feeds
        # the ``zvervx`` updraft (Section 1), the immersion-freezing cooling
        # rate (Section 6.2) and the ``lonacc`` exception on ``zauloc``
        # (Section 7). Read via ``.get`` on ``_dycore_fields`` (the same
        # soft pattern :class:`~jcm.physics.diagnostics.omega.OmegaDiagnostic`
        # uses) rather than declaring ``requires_dycore_fields`` here: that
        # would force every composition of this term to require omega, and
        # the only existing 2M-on-pySES configurations
        # (``ma-ne30-l47``/``ma-ne30-l95``) deliberately run with
        # ``physics.cu_lmfmid: false`` (TiedtkeConvection's own omega
        # requirement) pending that campaign's own stability validation --
        # not for lack of an omega provider, which pySES now has
        # (jax-gcm#698) -- so this term must stay composable without one
        # too. In every composition that DOES require omega for another
        # reason (chiefly ``cu_lmfmid=True``, the ECHAM default), it is
        # simply there to read.
        vertical_velocity = diagnostics.get(
            "_dycore_fields", {}).get("omega", jnp.zeros_like(state.temperature))
        lonacc_mask = resolve_lonacc_mask(
            diagnostics, state, self._inversion_range, vertical_velocity)

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
            # The UNMASKED activation potential (#1160): the SAME Lin
            # (2025) fit, on the full-column ``Nccn``, with no
            # ``cloud_fraction`` factor — matching ECHAM's own
            # ``zcdncact``/``cdncact_cv``, which the Fortran never
            # cloud-fraction-weights (see ``spa.py``'s module docstring).
            # ``activated_cdnc`` above keeps its existing, cover-masked
            # value -- it is passed to ``cloud_microphysics_2m`` purely as
            # the fallback default for ``activation_potential`` (unused
            # here since the real potential is always computed and passed
            # below). Every consumer that reads activation, including the
            # scan's own section-5.5 re-activation (jax-gcm#1161), reads
            # this unmasked potential, not the masked floor.
            #
            # Broadcast to the full ``(nlev, ncols)`` grid explicitly
            # (Codex P1 round 2 on #1160): unlike ``spa_activated_cdnc``
            # above, this function has no ``cloud_fraction`` factor to
            # broadcast the column-mean ``Nccn`` against, so its raw
            # output stays the per-column-scalar shape ``(1, ncols)``.
            # Every consumer documents and indexes this as a per-level
            # ``(nlev, ...)`` field: the convective-base gather below
            # (``jnp.take_along_axis(..., axis=0)``) needs a real
            # ``nlev``-sized level axis to index into, and the per-level
            # scan feeding section 5.5 (below, and jax-gcm#1161) requires
            # every one of its ``scan_inputs`` to share the SAME leading
            # axis size -- ``jax.lax.scan`` raises outright on a mismatch,
            # it does not silently broadcast. Left at size 1, a base level
            # above the surface reads out of bounds (silent NaN, via
            # ``jnp.take_along_axis``'s default fill mode) and the scan
            # raises a leading-axis-size ``ValueError``. The SPA potential
            # is physically uniform over the column anyway (no per-level
            # input), so broadcasting changes no value, only the shape.
            activation_potential = jnp.broadcast_to(
                spa_activation_potential(
                    Nccn=Nccn[jnp.newaxis, :],
                    prefactor=self._spa_prefactor.get_value(),
                    exponent=self._spa_exponent.get_value(),
                    cap_smoothing=self._spa_cap_smoothing.get_value(),
                ),
                activated_cdnc.shape,
            )
        else:
            # In-cloud liquid mass density (kg/m³) for the dynamic branch of
            # ``minimum_CDNC``; the fixed branch ignores it. Masked to cloudy
            # cells so the fallback (like the SPA floor it replaces) is zero in
            # clear air — the activation gate is cloud-only anyway.
            inv_cf_min = 1.0 / jnp.maximum(cloud_fraction, params_2m.epsec)
            qc_in_cloud_kgm3 = jnp.where(
                cloud_fraction > params_2m.epsec,
                provisional.tracers["qc"] * inv_cf_min * air_density, 0.0,
            )
            cdnc_min_floor = jnp.where(
                cloud_fraction > params_2m.epsec,
                minimum_CDNC(qc_in_cloud_kgm3, params_2m), 0.0,
            )
            activated_cdnc = jnp.where(arg_cdnc > 1.0, arg_cdnc, cdnc_min_floor)
            # The JAM activation term (``ArgActivation``/``HamActivation``)
            # already publishes the unmasked potential under this key, as
            # a plain alias of ``activated_cdnc`` (#1160: their own output
            # is never cloud-fraction-weighted, so the alias is exact).
            # The ``.get`` fallback only matters for a direct-construction
            # unit test that hand-builds a diagnostics dict with
            # ``activated_cdnc`` but not ``activation_potential``.
            activation_potential = diagnostics.get(
                "activation_potential", arg_cdnc)

        # Aerosol inputs of the heterogeneous mixed-phase freezing. A
        # prognostic aerosol (JAM) publishes ECHAM-HAM's freezing inputs as
        # ``freezing_aerosol`` (the ham_IN_setup fields), which selects
        # ECHAM's contact + immersion rates in section 6.2; without one the
        # scheme defaults to ECHAM's own aerosol-free mode, ``lccnclim``
        # (#1149), with ``AerosolFreeFreezing.DEMOTT`` as a non-default
        # fallback to jcm's older DeMott (2010) closure. ``ice_nuclei`` is an
        # optional external INP number for that DEMOTT closure only (no
        # in-tree term publishes it).
        #
        # ``ice_nuclei_deposition`` is HAM's deposition-INP slot: its one
        # in-tree producer (Niemand 2012, ``jam/ice_nucleation/niemand.py``)
        # was removed by #953 in favour of the ``freezing_aerosol`` contact
        # + immersion path above, so no in-tree term publishes it and it
        # stays zero. It is only READ by ``cloud_microphysics_2m`` when
        # ``nic_cirrus != 2``, where it is unused anyway (today's default
        # everywhere), so this is a dead read, not a live one. A real
        # HAM-faithful producer is tracked under #1017, not here.
        #
        # ``cirrus_aerosol_number`` (HAM's ``papnx``, from
        # ``ham_cirrus_aerosol`` via :class:`IceNucleation`, published only
        # for a population with ``cirrus_aerosol_modes`` set -- M7) is what
        # the ``nic_cirrus=2`` branch actually reads for ECHAM's ``pnicex``
        # (#552, jax-gcm#1017 task 2 part 2b): :func:`cloud_microphysics_2m`
        # feeds it through :func:`jcm.physics.clouds.lohmann_2m.cirrus.
        # xfrzmstr` itself, not a diagnostics passthrough like the two
        # above.
        zeros_2d = jnp.zeros_like(state.temperature)
        ice_nuclei = diagnostics.get("ice_nuclei", zeros_2d)
        ice_nuclei_deposition = diagnostics.get(
            "ice_nuclei_deposition", zeros_2d
        )
        cirrus_aerosol_number = diagnostics.get(
            "cirrus_aerosol_number", zeros_2d
        )
        freezing_aerosol = diagnostics.get("freezing_aerosol")

        # Section-1 droplet-number sources (#955): the previous step's
        # cloud cover (ECHAM zaclc_tm1, the "cover grew" activation
        # trigger), and the convective-base bookkeeping the detrained-
        # liquid droplet number needs. ``cloud_fraction_prev`` is
        # published by ``SundqvistCloudFraction`` from the carry it
        # received THIS step, before overwriting it with the RH-diagnosed
        # cover this term reads above; its absence (a stack with no cover
        # term, e.g. a direct-construction unit test) falls back to this
        # step's own cover, which makes the "cover grew" trigger an
        # always-false no-op rather than raising.
        cloud_fraction_prev = diagnostics.get(
            "cloud_fraction_prev", cloud_fraction)

        # ``convection.cloud_base`` / ``.ktype`` (#955): the convective
        # cloud-base level index and whether an updraft is active this
        # column, published by ``TiedtkeConvection``. Absent in a stack
        # without a convection term (RCE/single-column setups), where the
        # detrained-liquid droplet source is correctly a no-op (there is
        # no detrainment to give a number to either). jcm has no second,
        # convective-updraft activation call the way ECHAM-HAM does
        # (``cdncact_cv``, filled by the convection scheme's own ARG
        # call); the SAME ``activation_potential`` computed above stands
        # in, gathered at the convective base instead of the stratiform
        # one -- see ``convective_detrainment_droplet_number``'s docstring
        # for why, and the design doc for the design decision.
        #
        # Gathers from ``activation_potential``, NOT ``activated_cdnc``
        # (#1160): a convective updraft's base can sit in a grid cell with
        # little or no STRATIFORM cloud cover, where the MACv2-SP
        # ``activated_cdnc`` (masked by THAT cell's stratiform
        # ``cloud_fraction``) reads zero or a tiny fraction of the real
        # aerosol-activation potential -- exactly the gap a Codex review
        # on PR #1160 found at this gather. ``activation_potential`` has
        # no such masking (JAM: unmasked by construction; MACv2-SP: the
        # unmasked ``spa_activation_potential`` computed above), so the
        # gather now reads the same potential regardless of the
        # convective base cell's own stratiform cover.
        convection = diagnostics.get("convection")
        if convection is None:
            has_convective_base = jnp.zeros(
                cloud_fraction.shape[1:], dtype=bool)
            convective_base_activated_cdnc = jnp.zeros_like(cloud_fraction[0])
        else:
            has_convective_base = convection.ktype > 0
            safe_base_level = jnp.clip(convection.cloud_base, 0, nlev - 1)
            convective_base_activated_cdnc = jnp.take_along_axis(
                activation_potential, safe_base_level[jnp.newaxis, ...],
                axis=0,
            )[0]

        # The core owns grid-scale condensation now (the ECHAM section-5
        # zqcdif closure + supersaturation corrections run inside the
        # sweep), so there is NO external Sundqvist condensation bolt-on
        # any more. The bolt-on dated from when the internal adjustment
        # was suppressed ~1e6x by the c.ak/zdqsdt transcription bugs
        # (#667): with those fixed, summing both would remove
        # supersaturation twice per step with double the latent heating.
        # The scheme's own preffl/preffi (ECHAM cloud_micro_2m outputs, the
        # ``mo_activ`` REFFL/REFFI streams) are published as ``reffl``/``reffi``
        # for HAM's in-cloud impaction scavenging, which reads exactly these.
        # Radiation does not use them: the radii it radiates with, and the
        # ``clouds.r_eff_*`` diagnostic, are formed by the radiation term from
        # the step's state, as in ECHAM's cloud_optics.
        (tend_all, surface_rain_flux, surface_snow_flux,
         preffl_all, preffi_all, rain_formation_warm, rain_from_melt,
         autoconv_all, accretion_all, wbf_all,
         precip_form_all, precip_evap_all, cloud_fraction_all,
         negative_mass_repair_all, scav_ledger_all,
         precip_cover_all, pfrain_all, pfsnow_all,
         rain_flux_all, snow_flux_all) = jax.vmap(
            cloud_microphysics_2m,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     None, None, 1, 1, 1, 1, 1, 1, 1, 1,
                     None if freezing_aerosol is None else 1, 1,
                     1,  # vertical_velocity: always a real array here (soft-read default is zeros, never None)
                     None if lonacc_mask is None else 1,
                     1,  # cloud_fraction_prev: always a real array (soft-read default is cloud_fraction, never None)
                     0, 0,  # convective_base_activated_cdnc / has_convective_base: per-column scalars (ncols,)
                     1,  # activation_potential: always a real array (resolved above, never None)
                     ),
            out_axes=(0,) * 20,
        )(
            anchor.temperature, anchor.specific_humidity, pressure_full,
            anchor.tracers["qc"], anchor.tracers["qi"],
            anchor.tracers["qnc"], anchor.tracers["qni"],
            cloud_fraction, air_density, layer_thickness, tke,
            activated_cdnc, ice_nuclei, ice_nuclei_deposition, dt, params_2m,
            increment.temperature, increment.specific_humidity,
            increment.tracers["qc"], increment.tracers["qi"],
            increment.tracers["qnc"], increment.tracers["qni"],
            inputs.detrained_qc, inputs.detrained_qi, freezing_aerosol,
            cirrus_aerosol_number, vertical_velocity, lonacc_mask,
            cloud_fraction_prev,
            convective_base_activated_cdnc, has_convective_base,
            activation_potential,
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

        # Publish the resolved CDNC into ``clouds.droplet_number`` (#1053).
        # ``CloudData`` is created with this field at zero and only the 1M
        # path (``echam_1m.py``) used to fill it; the 2M carries its number
        # as the prognostic tracer ``qnc`` [kg^-1] instead, so this path left
        # the CF-advertised CDNC at its zero seed forever.
        #
        # GRID-MEAN, not in-cloud: ECHAM-HAM hands its radiation the grid-mean
        # number unmasked by cover, ``pacdnc(:,jk) = zcdnc(:,jk)``
        # (``mo_cloud_micro_2m.f90:1710``) with ``zcdnc`` the grid-mean
        # ``ρ·(pxtm1 + ztmst·pxtte)`` (line 600) — there is no ``/paclc``
        # anywhere on that path. The same convention is already load-bearing
        # elsewhere in jcm: ``cloud_optics.radiation_effective_radii``'s
        # prognostic branch forms ``max(qnc, 0)·air_density`` and
        # ``AerocomDiagnostics._number_concentrations`` derives its grid-mean
        # ``cdnc_gm`` the identical way. This differs from the 1M path's
        # CONVENTION for the same field (a characteristic in-cloud value,
        # nonzero in clear sky — see that field's entry in
        # ``CLOUD_OUTPUT_ATTRS`` and ``cloud_utils.prescribed_droplet_number``):
        # the 1M's ``acdnc`` is a profile ECHAM prescribes from pressure alone
        # and hands unmasked to both the in-cloud microphysics formulas and
        # the radiation, so it was never grid-mean to begin with; the 2M's
        # number is a genuine prognostic grid-mean tracer, so masking it by
        # cover here would invent an in-cloud value ECHAM-HAM's own CDNC
        # output does not have.
        #
        # Timing: formed from THIS STEP's saved ``qnc``, not the step-start
        # tracer the term receives. Operator splitting means the host sums
        # EVERY term's tendency (this one plus any upstream term that also
        # touches ``qnc``) onto ``state.tracers["qnc"]`` only after the whole
        # step's physics has run, so reading the step-start tracer plus only
        # this term's own tendency would both date the published CDNC a step
        # and drop any upstream term's contribution — the lag
        # ``aerocom._post_physics_tracer`` / ``cosp_cloudsat
        # ._post_micro_condensate`` exist to avoid, and exactly the gap
        # Codex review flagged on #1053 (PR #1085). ``inputs.provisional``
        # is already ``state.tracers["qnc"] + dt`` times the running upstream
        # tendency (``cloud_scheme_inputs``' ``anchor + increment``, with no
        # detrainment term for ``qnc``) — the same baseline the scheme's own
        # ``tracer_tm1_cdnc=qnc_raw`` ledger above uses, so adding this
        # term's tendency to it reproduces exactly what the dycore will save,
        # whether or not a future upstream term ever writes a ``qnc``
        # tendency (today, none does).
        qnc_post_physics = jnp.maximum(
            inputs.provisional.tracers["qnc"] + dt * tendency.tracers["qnc"],
            0.0)
        droplet_number = qnc_post_physics * diagnostics["air_density"]

        # ``qnc_prev``/``qni_prev`` record the raw number tracers of the
        # state this term receives, not the scheme's entry-floored working
        # numbers. No term reads them. The surface precipitation comes from
        # the lax.scan carry.
        clouds_next = clouds.copy(
            droplet_number=droplet_number,
            # ECHAM writes the post-microphysics cloud fraction back to
            # ``paclc``: cells the scheme has just emptied of both condensates,
            # or driven below ``clc_min``, are no longer cloudy. Radiation and
            # the aerosol cloud-borne partition read this, and must see the
            # cloud the step actually leaves behind.
            cloud_fraction=cloud_fraction_all.T,
            qnc_prev=state.tracers.get("qnc", jnp.zeros_like(state.temperature)),
            qni_prev=state.tracers.get("qni", jnp.zeros_like(state.temperature)),
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
                       "wbf": wbf_all,
                       # (nlev, ncols) [um], 0 where the level holds no
                       # liquid / ice cloud; transposed like the profiles
                       # above.
                       "reffl": preffl_all.T,
                       "reffi": preffi_all.T}
        if self._publish_wetdep_hydro:
            # A static flag, so the diagnostics key set is fixed for a given
            # composition, as the scan carry requires.
            diagnostics = {**diagnostics,
                           "precip_cover": precip_cover_all.T,
                           "pfrain": pfrain_all.T,
                           "pfsnow": pfsnow_all.T}
        diagnostics = advance_thermo_run(
            diagnostics, dt,
            d_temperature=tendency.temperature,
            d_specific_humidity=tendency.specific_humidity,
            d_qc=tendency.tracers["qc"], d_qi=tendency.tracers["qi"])

        return tendency, {**diagnostics, "clouds": clouds_next}
