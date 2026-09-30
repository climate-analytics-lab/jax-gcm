# Clouds and microphysics

**What we do.** A diagnostic cloud fraction plus a choice of **single-moment** or
**two-moment** microphysics:

- **Sundqvist diagnostic cloud fraction**
  (``jcm/physics/clouds/sundqvist.py::SundqvistCloudFraction``) — RH-based cloud
  fraction with a stratocumulus inversion enhancement (``mo_cover.f90``). The
  enhancement boosts the apparent RH at a **single** boundary-layer-top level
  over ice-free ocean with no active convection, chosen as ECHAM's ``zknvb``
  scan does — the most inversion-like BL level, resolving to the *lowest*
  (nearest-surface) one on a tie; the differentiable softmax surrogate keeps
  that single-level behaviour rather than smearing the boost across the tied
  levels. It is
  a **pure diagnostic** — the term emits zero T/q/qc/qi tendencies; the
  saturation adjustment lives downstream in each microphysics scheme (the 2M
  path's ``mixed_phase_deposition_and_corrections``, and ``echam_1m.py``'s own
  port of the same linearised-Newton step). The humidity the cover closure sees,
  ``q/q_s`` with ``q_s`` over ice where the cell holds cloud ice below
  ``t_ice`` (``mo_cover.f90``'s ``lo2`` switch), is published as
  ``cover_relative_humidity``. It is a scheme-internal closure variable — it
  jumps by up to ~25 % across the cloud-ice threshold between adjacent cold
  cells — so it is kept apart from the model's one public ``relative_humidity``,
  which ``MoistAirColumnState`` computes with respect to liquid water at every
  temperature (the WMO definition, written as CMIP ``hur``) and which no
  composition changes the meaning of.
- **ECHAM 1-moment microphysics**
  (``jcm/physics/clouds/echam_1m.py::Echam1MMicrophysics``) — a flux-coupled
  top-down column sweep: autoconversion (Beheng 1994 default or KK2000),
  accretion, ice→snow aggregation (Levkov 1992), riming, snow/ice melt, ice
  sedimentation, Rotstayn (1997) rain evaporation. Ports the ECHAM6/ICON
  ``mo_cloud.f90`` single-moment branch. The ice/snow fall-speed factor
  ``cvtfall = 2.5`` is ECHAM's value for jcm's default T63 grid
  (``mo_echam_cloud_params.f90``, ``nn == 63``), the same the 2M scheme uses.
- **Lohmann 2-moment microphysics**
  (``jcm/physics/clouds/lohmann_2m/scheme.py``: ``cloud_microphysics_2m`` and its
  ``Lohmann2MMicrophysics`` term) carries droplet and ice-crystal number as well
  as mass. The process chain runs as one flux-coupled top-down ``lax.scan`` in
  the section order of ECHAM's ``column_processes`` loop: ice sedimentation →
  melt → sublimation/rain-evaporation → clear-sky evaporation → grid-scale
  condensation and supersaturation corrections → in-cloud water update +
  droplet activation / ICNC diagnosis → homogeneous freezing → mixed-phase
  freezing + WBF → precipitation geometry → warm-rain (KK2000) and cold
  precipitation formation. Before the sweep the scheme evaluates the ECHAM
  section-1 fields the sweep needs: the crystal radius ``zrid``, the
  detrainment phase criterion ``lo2_2d`` and the detrained crystal number
  ``znidetr``. The process formulas and constants are ECHAM-HAM's
  (``mo_cloud_utils.f90`` / ``mo_echam_cloud_params.f90`` /
  ``mo_cloud_micro_2m.f90``, collected in
  ``jcm/physics/clouds/lohmann_2m_params.py::CloudParams2M``); no CAM
  ``micro_mg`` / PUMAS ``qsmall`` / ``mincld`` / ``dcs`` constants enter. The
  scheme is not a complete port of ECHAM-HAM. Several section-1 number
  sources are absent, the mixed-phase heterogeneous freezing is a jcm
  closure, and a small set of jcm-only bounds remains (for example the
  droplet-number entry cap and the Koop homogeneous-freezing floor). The
  absent processes and the deliberate deviations are listed below. See
  {doc}`../design/lohmann_2m_column_processes`.

Both schemes convert a condensed/evaporated/frozen mixing-ratio increment to a
temperature increment with ``L / cp`` where ``cp`` is the **moist** isobaric
heat capacity ``cpd·(1 + vtmpc2·q)`` evaluated per-level at the step-start
humidity — ECHAM's ``zlvdcp = alv/pcair`` / ``zlsdcp = als/pcair``
(``mo_cloud.f90``, ``mo_cloud_micro_2m.f90``), shared through
``cloud_utils.latent_heat_over_cp``. Dry ``cpd`` would over-heat every
condensation event by ``vtmpc2·q`` (~1.5 % in the moist tropics); the column
enthalpy budget closes against this same moist ``cp``.

The 2M scheme's utility fields are ECHAM's, shared through ``cloud_utils``:

- **Viscosity of air** in the snow Reynolds number of riming,
  ``pviscos = (1.512 + 0.0052·(T − 233.15))·10⁻⁵`` kg m⁻¹ s⁻¹ at the step-start
  temperature (``mo_cloud_utils.f90::get_util_var``, line 132;
  ``air_dynamic_viscosity``). It puts the Reynolds number of the 447 µm planar
  flake at about 15–30 through the troposphere, and the collection efficiency
  of 10–20 µm droplets at about 0.8 (``precip.riming_collection_efficiency``,
  ``mo_cloud_micro_2m.f90::precip_formation_cold``, lines 3198–3250; Lohmann
  2004). The thermal conductivity of air ``zkair`` (line 715) is a different
  quantity; it enters only the diffusional-growth factors.
- **Ice fall-speed air-density factor**
  ``paaa = (p/30000)^−0.178·(T/233)^−0.394`` (``get_util_var``, line 129;
  Heymsfield & Iaquinta 2000; ``ice_fall_speed_air_density_factor``), 1 at
  300 hPa and 233 K, scaling the sedimentation speed of ice mass and number
  alike (``sedimentation_ice``, ``mo_cloud_micro_2m.f90`` line 2224).
- **Turbulent updraft** of the phase choice ``lo2`` and the
  Wegener–Bergeron–Findeisen gate, ``100·fact_tke·√TKE`` cm s⁻¹ with
  ``fact_tke = 0.7``, zero at the lowest level (``mo_cloud_micro_2m.f90``
  lines 814–815; ``turbulent_updraft_velocity``). ECHAM's ``zvervx`` adds the
  large-scale ``−100·ω/(g·ρ)`` (line 816), which is not plumbed to the scheme
  (#705).
- **Volume-mean ice radius of the WBF threshold** ``0.9·r_eff``
  (``conv_effr2mvr``; ``effective_2_volmean_radius_param_Schuman_2011``, lines
  4059–4085; ``ice_volume_mean_radius_schumann``), with ``r_eff`` the
  Lohmann (2008) effective radius clipped to 10–150 µm. ECHAM uses it at every
  threshold-velocity decision, and jcm ports all four: the phase of convective
  detrainment before the level loop (``lo2_2d``, lines 872–885), section 4
  (line 1288), the section-5 supersaturation correction (line 2374) and the
  WBF gate (line 1582). Aggregation uses the plate radius
  ``zrih = −2261 + √(5113188 + 2809·r_eff³)`` µm³ (``ice_volume_mean_radius``,
  lines 3160–3166), as ECHAM does. The ICNC diagnosis uses the
  temperature-parameterised radius ``zrid`` described next.

**Ice mass and number from convective detrainment.** Detrainment supplies most
of the 2M scheme's cloud-ice mass. ECHAM gives that ice a crystal number and a
phase by rules that need no aerosol, in section 1 before the level loop and in
section 4 inside it (``mo_cloud_micro_2m.f90``). jcm applies them:

- **Crystal radius** ``zrid`` (lines 945–956):
  ``0.9·max(23.2·exp(0.015·min(T − tmelt, 0)), 1)`` µm at the step-start
  temperature, with the 1 µm floor of the Schumann conversion (line 4083;
  ``cloud_utils.effective_2_volmean_radius_param_Schuman_2011``). It falls
  from about 21 µm at the melting point to about 9 µm at 220 K. It is the
  radius of the crystals detrainment adds and of the ICNC diagnosis in
  ``update_in_cloud_water`` (passed as ``prid``, line 1511; used at line
  2616).
- **Phase criterion of the detrained condensate** ``lo2_2d`` (lines 859–885):
  the WBF test ``0.01·zvervx < zvervmax`` on the ice present before this
  step's detrainment and on the incoming crystal number. It carries no
  temperature test of its own; those enter through ``ll_cv`` below.
- **Detrained crystal number** ``znidetr`` (lines 958–983). It is nonzero
  where condensate detrains, the cover exceeds ``clc_min``, and the
  step-start temperature is below ``cthomi``, or below ``tmelt`` with
  ``lo2_2d`` true (``ll_cv``). There it is
  ``conv_effr2mvr·(0.5·10⁻²)^pow_PK·1000/fact_PK · ρ·Δq_det / (max(cf, clc_min)·zrid^pow_PK)``,
  floored at ``cqtmin``, with ``Δq_det`` the whole condensate detrained this
  step in both phases (lines 970–972). The prefactor is ECHAM's as written.
  It is not the exact inverse of the plate mass–radius law, and jcm keeps it.
- **Sedimentation acts on the ice present before detrainment** (lines
  1227–1248): ``pxim1 + ztmst·pxite``, the step-start ice plus the upstream
  increments without the detrained part. Detrained ice joins the cell after
  sedimentation (``zxidt``, lines 1316–1317), so it does not fall at the old
  crystal number in the step it arrives.
- **Crystal number after sedimentation** (lines 1251–1252):
  ``ICNC = min(ICNC_sedimented + znidetr, icemax)``.
- **Phase of the detrained condensate in section 4** (lines 1276–1316).
  ``lo2``, evaluated on the post-sedimentation ice and the crystal number that
  includes ``znidetr``, assigns the whole detrained condensate to ice where it
  holds and to liquid elsewhere. The convection scheme has already heated the
  column for its own split: Tiedtke counts detrained condensate as ice where
  its environment temperature is at or below ``tmelt``. Where the 2M scheme
  reassigns convective ice to liquid it removes the fusion heat, ``(Ls − Lv)/cp``
  per unit mass (lines 1300–1308). Where it reassigns convective liquid to ice
  it adds that heat. The reassigned condensate then enters the in-cloud state,
  the clear-sky evaporation and the section-5 closure together with the other
  increments (``zxidt``/``zxldt``, lines 1316–1317).
- **Number tendencies are taken against the raw tracer** (the call at lines
  1780–1781; lines 3625–3628). For its own arithmetic the scheme clips the
  incoming number tracers to ``[0, icemax/ρ]`` for ice and
  ``[0, 10¹¹ m⁻³/ρ]`` for droplets. The returned tendency is
  ``(N_end/ρ − q_N)/Δt`` against the unclipped tracer ``q_N``. The tracer
  therefore ends the step at the scheme's number, or at zero where the
  negative-mass repair removed the condensate (lines 3641–3652), and an
  out-of-range value does not persist from step to step.

The Tiedtke term records the condensate it detrains each step as
``clouds.conv_detrainment_qc`` and ``clouds.conv_detrainment_qi``
[kg kg⁻¹ s⁻¹], its own split at ``tmelt``, alongside the ``clouds.qc`` /
``clouds.qi`` it advances. Both fields are written to the output. The 2M
scheme receives them separately from the other increments and re-splits
their sum as described above.

Cloud parameters are ``flax.struct.dataclass`` leaves (differentiable), threaded
through the scheme via ``nnx.Param``; only genuine code-path switches
(``nic_cirrus``, ``ldyn_cdnc_min``) are static aux.

**What ECHAM/CAM does.** ECHAM6-HAM2.3 uses the **Sundqvist (1989)** statistical
cloud cover (Lohmann & Roeckner 1996) in ``mo_cover.f90``, with **Lohmann et al.
two-moment microphysics** (``mo_cloud_micro_2m.f90``; Lohmann et al. 2007, Lohmann
& Hoose 2009; Roeckner et al. 2003 for the Marshall-Palmer precipitation
inversion). CAM6 uses **MG2/PUMAS** two-moment microphysics (Gettelman & Morrison
2015; ``micro_mg`` / ``micro_pumas_v1``). DeMott et al. (2010), *PNAS*
doi:10.1073/pnas.0910818107 is the ice-nucleating-particle count. Without
prognostic HAM aerosol, ECHAM's two-moment scheme takes its aerosol inputs
from the ``lccnclim`` mode: ``ccnclim_IN_setup``
(``mo_ccnclim.f90:524-583``) feeds the aerosol freezing routine
``het_mxphase_freezing`` constant dust and black-carbon fractions (lines
69–73) and fixed insoluble-mode radii (lines 74–77), and sets the cirrus
soluble-aerosol number from a CCN climatology with a floor of 10⁷ kg⁻¹.

**Why we differ.**
- `science` (deliberate, documented): the 2M column sweep uses **MG/PUMAS
  sediment→melt ordering** (ice sedimentation before melt), *not* ECHAM's
  melt→sediment. The melt acts on the post-sedimentation ice through the threaded
  tendency so the two sinks cannot claim the same mass. This is the one intentional
  cross-reference ordering deviation in the sweep; the process docstring in
  ``scheme.py`` names it explicitly.
- `science` (deliberate): the fusion-heat correction of the detrainment
  re-split uses jcm's per-level moist ``cp`` and runs in both directions.
  ECHAM divides by dry ``cpd`` and corrects only convective ice turned liquid
  (``ztconv <= tmelt .AND. .NOT. lo2``, lines 1300–1308), so convective liquid
  turned ice gains no fusion heat there. jcm's form keeps the column
  moist-enthalpy identity exact, which the conservation tests pin.
- `science` (deliberate): ECHAM floors the crystal number at ``icemin`` in
  cold cloudy cells before the loop (lines 1127–1131) and after the section-4
  additions (line 1253). jcm applies neither floor. A cell at or below
  ``icemin`` is re-diagnosed from its ice mass in ``update_in_cloud_water``,
  and the floor would inject ``icemin`` crystals per step into cells with no
  ice.
- `science` (deliberate): the ICNC diagnosis is capped at ``icemax``.
  ECHAM's diagnosis (lines 2610–2624) is uncapped. jcm applies the bound
  ECHAM puts on the section-1 additions (line 1252).
- `differentiability`: activation, freezing and precipitation gates are
  formulated to keep cloud parameters differentiable; there is no import-time
  default parameter instance (that would sever gradients / overrides).

**Status & known limitations (stated openly).**
- **Mixed-phase heterogeneous freezing is a jcm closure.** In cells with
  liquid between ``cthomi`` and ``tmelt`` (ECHAM's ``ll_mxfrz``), droplets of
  mean mass freeze until the crystal number reaches the INP number ``n_inp``.
  The droplets available cap the number frozen, the liquid available caps the
  mass, and number, mass and fusion heat move together. The droplet number is
  floored at ``cqtmin``, where ECHAM's ``het_mxphase_freezing`` stops at
  ``cdnc_min`` (line 2819).
  - Without JAM, ``n_inp`` is the **DeMott et al. (2010)** parameterisation
    (``jcm/physics/clouds/lohmann_2m/deposition_freezing.py::demott2010_inp``)
    on a prescribed number of particles larger than 0.5 µm, ``n_aer_coarse``
    (0.5 cm⁻³ at standard conditions, spatially constant). DeMott et al.
    report INP per standard litre, at 273.15 K and about 1013 hPa, so the
    function converts to ambient air by ``ρ/ρ_STP``. It applies only within
    the fitted range, 238–264 K. The parameterisation appears in neither the
    ECHAM nor the CAM source tree.
  - Because detrained ice carries crystal number, heterogeneous freezing is
    a second-order crystal source. In T63 January test runs it supplied
    about 200 crystals m⁻² s⁻¹ against about 3×10⁵ from detrainment. The
    choice of INP closure therefore moves a small term, and DeMott stays the
    closure until ECHAM's ``lccnclim`` mode can replace it.
  - With JAM, ``n_inp = max(ice_nuclei, DeMott)``, where ``ice_nuclei`` is
    JAM's immersion INP on prognostic dust and black carbon. This is a
    stopgap. JAM's immersion INP is positive in nearly every mixed-phase cloud
    cell but about four orders of magnitude below the DeMott value, for a
    reason outside the cloud scheme (#953). Taking the larger keeps
    heterogeneous freezing active in JAM. JAM sets ``n_inp`` only where it
    exceeds DeMott, which at present is rare, so aerosol–ice coupling through
    this path is weak.
  - JAM's deposition INP (``ice_nuclei_deposition``) is read only by the
    ``nic_cirrus = 2`` branch of ``update_in_cloud_water``, so it is inert at
    the default ``nic_cirrus = 1`` (#679, #552).
- **Absent by decision.** These ECHAM-HAM processes are not in the 2M scheme:
  - the droplet number of detrained liquid, ``zqlnuccv`` (lines 889–941), and
    stratiform activation at cloud base copied to the levels above (lines
    742–782). Both are droplet sources outside the ice treatment, and
    ``zqlnuccv`` needs the activated number at convective cloud base
    (``cdncact_cv``), which the Tiedtke scheme does not provide. Detrained
    liquid therefore joins the existing droplet population, and droplets are
    activated only in ``update_in_cloud_water``, where the droplet number has
    fallen to ``cdnc_min`` (lines 2594–2608). Tracked in #941.
  - cirrus nucleation ``zninucl`` at ``nic_cirrus = 1`` (lines 986–999). Its
    cap is the soluble-aerosol number ``zascs``, which the scheme does not
    receive, and the ice budget stands without it.
  - Kärcher–Lohmann cirrus, ``nic_cirrus = 2`` (lines 1001–1117; #552).
  - aerosol-driven mixed-phase freezing, ``het_mxphase_freezing`` (lines
    2675–2840). jcm's transliteration
    (``deposition_freezing.py::het_mxphase_freezing``) is exported but
    unused. It carries two transcription errors: its immersion rate
    multiplies by the pressure velocity ``ω − fact_tke·√TKE·ρ·g`` itself,
    where ECHAM multiplies by the cooling rate ``ztte = ω/(cpd·ρ)`` built from
    it (line 2802), and its returned freezing rate lacks ECHAM's cover
    weighting ``·paclc`` (line 2837).
  - ECHAM's aerosol-free freezing mode, ``lccnclim`` (see *What ECHAM/CAM
    does*). It is the faithful replacement for the DeMott closure. It waits
    for the large-scale vertical velocity, which the immersion rate needs
    (#705), and for the two transcription errors above to be fixed.
  - the large-scale term ``−100·ω/(g·ρ)`` of the updraft ``zvervx`` (line
    816; #705).
- **Known biases of the 2M ice.** Three biases remain, measured in 10-day T63
  January runs from a spun-up state. No parameter has been tuned to them. A
  retune follows together with the convection retune (#682).
  - Glaciation is too warm. The supercooled share of the condensate mass
    falls to one half near −7 °C. CALIOP places the crossing near −20 °C, but
    CALIOP's quantity is a cloud-top phase frequency, not a mass fraction, and
    that figure comes from a secondary summary (Hu et al. 2010; Tan et al.
    2016).
  - Cold ice cloud holds too many crystals. About a fifth to two fifths of
    ice-containing cells exceed 10⁶ crystals m⁻³, above the 90th percentile
    of the in-situ cirrus climatology of Krämer et al. (2020, *ACP* 20,
    12569). The cause is the small radius ``zrid`` gives at cirrus
    temperatures, which makes ``znidetr`` large.
  - Liquid water path lies below the observed range (50–84 g m⁻² over the
    oceans; Lohmann et al. 2007, *ACP* 7, 3425, Table 2).
- **Cirrus ICNC diagnosis (default ``nic_cirrus = 1``).** Where a cloudy cell
  holds ice at or below ``icemin`` crystals, ``update_in_cloud_water``
  diagnoses the number from the ice mass at the radius ``zrid``,
  ``N = 0.75·ρ·q_i / (π·ρ_ice·zrid³)`` with ``q_i`` in-cloud
  (``jcm/physics/clouds/lohmann_2m/assembly.py::update_in_cloud_water``;
  ECHAM lines 2610–2624). The number is capped at ``icemax`` (10⁷ m⁻³; see
  *Why we differ*). The radius floor ``cirrus_min_ice_radius`` (10⁻⁷ m)
  remains a parameter but never binds, because ``zrid`` is at least 1 µm. The
  inversion is non-dimensionalised by a static 1 µm scale, so no gradient
  path, through the state or a parameter, divides by a tiny cube
  (`differentiability`).
- Both the 1M and 2M paths publish an LWC-dependent radiative liquid radius
  from the shared ECHAM Martin/Bower law (``eff_liquid_droplet_radius``);
  radiation reads it from the carried ``clouds`` state one step lagged, because
  the ECHAM term order runs radiation before microphysics. The constant
  ``effective_radius_liquid`` fallback therefore survives only where that carry
  is still zero, resolved **cell by cell** — the cold-start first step, and
  thereafter any cloudy cell that was clear the previous step (a level newly
  turning cloudy falls back even mid-rollout in an otherwise-cloudy column) —
  not the steady state the 1M ``physics=echam`` path used to run on. The
  radiative **ice** radius is the 2M's published end-of-step ``r_eff_ice``:
  the plate-crystal law of ``eff_ice_crystal_radius`` on the end-of-step ice
  mass and crystal number at and above ``cthomi``, and ECHAM's
  ``83.8·IWC^0.216`` below it, clipped to 10–150 µm (ECHAM ``preffi``, lines
  3680–3698). Its mixed-phase value follows the crystal number the scheme
  carries, which comes mainly from ``znidetr`` and the ``zrid`` diagnosis.
- Clear-sky evaporation of decorrelated condensate (the radiation-side contract in
  ``mcica.in_cloud_path``) is owned by the 2M scheme's clear-sky evaporation step.

**Code pointers.**
- ``jcm/physics/clouds/sundqvist.py`` — ``SundqvistCloudFraction``,
  ``calculate_cloud_fraction``, ``condensation_evaporation``.
- ``jcm/physics/clouds/echam_1m.py`` — ``Echam1MMicrophysics``.
- ``jcm/physics/clouds/lohmann_2m/`` — ``scheme.py`` (``cloud_microphysics_2m``,
  ``Lohmann2MMicrophysics``, process-order docstring, the section-1 fields),
  ``deposition_freezing.py``
  (``demott2010_inp`` used; ``het_mxphase_freezing`` defined/exported/unused),
  ``sedimentation_melt.py``, ``precip.py``, ``assembly.py``,
  ``jcm/physics/clouds/lohmann_2m/types.py``;
  ``jcm/physics/clouds/lohmann_2m_params.py`` (``CloudParams2M``);
  ``jcm/physics/clouds/cloud_utils.py``
  (``effective_2_volmean_radius_param_Schuman_2011``);
  ``jcm/physics/clouds/cloud_data.py`` (``CloudData``, including
  ``conv_detrainment_qc`` / ``conv_detrainment_qi``).

**Validation evidence.** ``jcm/physics/clouds/sundqvist_test.py`` and
``sundqvist_smooth_gradients_test.py``, ``echam_1m_test.py``,
``lohmann_2m_test.py``, ``cloud_utils_test.py``, ``cloud_data_test.py``. Design
reference: {doc}`../design/lohmann_2m_column_processes`.
