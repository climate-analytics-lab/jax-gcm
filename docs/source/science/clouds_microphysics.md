# Clouds and microphysics

**What we do.** A diagnostic cloud fraction plus a choice of **single-moment** or
**two-moment** microphysics:

- **Cloud cover** (``jcm/physics/clouds/sundqvist.py::SundqvistCloudFraction``)
  — ECHAM6.3's ``mo_cover.f90::cover``, the Sundqvist (1989) / Lohmann and
  Roeckner (1996) relative-humidity closure, used by the 1M, the 2M and the
  JAM configurations alike. The formulation is ECHAM's at every level (the T63
  values of ``crt``, ``crs``, ``nex``, ``csatsc`` and ``cinv`` are jcm's
  calibrated set, below):
  ``q_s`` in ECHAM's form over ice or water by ECHAM's ``lo2`` switch; the
  critical relative humidity ``crt + (crs − crt)·exp(1 − (p_s/p)^nex)``;
  over ice-free ocean without convection (previous step's ``ktype``, as in
  ECHAM), ECHAM's inversion search from the lowest level up to its level
  ``jbmin``, which takes the level with the largest ``min(0, dT/dz)`` (the
  lowest one on a tie), applies no enhancement if that level lies below
  ``jbmax``, and otherwise divides ``q_s`` by
  ``zsat = min(1, csatsc + max(0, −dT/dz·cp/g))`` there and ``nadd`` levels
  below; ``b0 = (q/(q_s·zsat) − rhc)/(1 − rhc)`` clipped to ``[0, 1]`` and
  ``cover = 1 − sqrt(1 − b0)``. The cover is therefore exactly 0 in every cell
  at or below the critical humidity and exactly 1 at saturation, which the
  microphysics' clear-cell rules read. There is no stratospheric cutoff:
  ECHAM computes every level (``ktdia = 1``). ``jbmin``/``jbmax`` follow
  ECHAM's ``sucloud`` rule (the first levels from the top below 2000 m and
  500 m, heights ``(p_s − p)/(1.25 g)`` at a 101320 Pa surface) on the model's
  own levels, 40/45 at L47 and 88/93 at L95, and ``dT/dz`` uses the model
  geopotential, as ECHAM's uses ``pgeo``. Every column of the ECHAM Fortran
  reference (``jcm/data/test/echam_cloud_reference/``) is reproduced at the
  reference tolerance, at T31, T63, T127 and T255, with ECHAM's own constants
  for each. It is a **pure diagnostic**
  — the term emits zero T/q/qc/qi tendencies; condensation lives in each
  microphysics scheme, as in ECHAM's ``cloud``. The humidity the closure sees,
  ``q/q_s`` with ``q_s`` over ice where ``lo2`` selects it, is published as
  ``cover_relative_humidity``. It is a scheme-internal closure variable — it
  jumps by up to ~25 % across the cloud-ice threshold between adjacent cold
  cells — so it is kept apart from the model's one public ``relative_humidity``,
  which ``MoistAirColumnState`` computes with respect to liquid water at every
  temperature (the WMO definition, written as CMIP ``hur``) and which no
  composition changes the meaning of.
- **ECHAM 1-moment microphysics**
  (``jcm/physics/clouds/echam_1m.py::Echam1MMicrophysics``) — a transcription
  of ECHAM6.3's ``mo_cloud.f90::cloud`` (Lohmann & Roeckner 1996; Roeckner et
  al. 2003, §10): one top-down column sweep that runs ECHAM's sections in
  ECHAM's order at every level, with the falling rain and snow, the
  precipitating fraction and the sedimenting ice carried between levels within
  the step. Melting of the incoming snow and of cloud ice above ``tmelt``,
  sublimation of the incoming snow (Lin et al. 1983) and evaporation of the
  incoming rain (Rotstayn 1997), all at the anchor state (below); ice
  sedimentation; the ``lo2`` phase switch (ice below ``cthomi``, or below
  ``tmelt`` where the cloud ice exceeds ``csecfrl``), which selects the latent
  heat and ice or water saturation; the return of all condensate of a clear
  cell to vapour; **tendency-driven condensation** in the cloudy part,
  ``zqcdif = (Δq − Δq_sat)·paclc`` with the cloudy part saturated at the
  anchor, followed by the whole-box 1 % supersaturation check and the
  treatment of a clear cell that gained condensate as cloudy for the step;
  homogeneous freezing below ``cthomi``, Bigg and contact freezing between
  ``cthomi`` and ``tmelt``; Beheng (1994) autoconversion and accretion by rain;
  Levkov et al. (1992) aggregation, accretion of ice and riming by snow; the
  precipitating-fraction update with its reset to the local cover; and the
  return of condensate below ``ccwmin`` to vapour with the cover write-back.
  Every output, and every intermediate the Fortran harness exposes (under the
  Sonntag variant at T63), is compared with the Fortran routine, whose code is
  unmodified and whose saturation tables the harness evaluates analytically
  (they agree to 3e-11), on 42 designed and sampled columns
  (``echam_fortran_reference_test.py``). The anchor and the increments ``Δq``,
  ``ΔT``, ``Δq_c``, ``Δq_i`` are the 2M's (``cloud_scheme_inputs``, below): the
  previous step's post-physics state, and the dynamics since then plus ``dt``
  times the running tendency of every physics term composed before the 1M term
  (radiation, vertical diffusion with its condensate, the surface,
  convection), with the convective detrainment passed separately as ECHAM's
  ``pxtecl``/``pxteci``.
  The saturation vapour pressure and its slope are ECHAM's Sonntag (1990) fit
  of ``thermodynamics``, read through ``echam_saturation`` as the cover reads it. ``cvtfall``,
  ``csecfrl`` and ``clwprat`` are ordinary tunable parameters whose defaults follow ECHAM's per-truncation
  values (T63: 2.5, 5e-6, 4.0). The **droplet number** is ECHAM's prescribed
  ``acdnc`` (``physc.f90`` §3.12; ICON-A ``mo_echam_phy_diag.f90::droplet_number``):
  80 cm⁻³ over sea and 180 cm⁻³ over land that is not glacier from the surface
  to 800 hPa, ``20 + (zn2 − 20)·exp(1 − min(8, 80000/p)²)`` cm⁻³ above,
  continuous at 800 hPa and 20 cm⁻³ in the upper troposphere. ECHAM passes that
  one field to ``cloud`` as ``pacdnc``, where it enters the Beheng
  autoconversion and the Bigg and contact freezing. The radiation's droplet
  number (``cloud_utils.prescribed_droplet_number``, published as
  ``clouds.droplet_number``) is that profile times the MACv2-SP Twomey factor
  ``cdnc_factor``. `science` (deliberate, documented) — the same factor scales
  the autoconversion's droplet number (``autoconversion_twomey``, on by
  default), so the aerosol-cloud interaction acts on precipitation formation
  and not only on the radiation. MPI-ESM1.2 scales the radiation's droplet
  number only (Mauritsen et al. 2019, *JAMES*, doi:10.1029/2018MS001400,
  §2.2); the departure is the maintainer's choice, recorded in #932. The
  freezing of section 6.2 reads ECHAM's unscaled profile. The jcm option
  ``autoconversion_scheme="kk2000"`` (Khairoutdinov & Kogan 2000) is off by
  default.
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
  sources are absent, the mixed-phase heterogeneous freezing is ECHAM-HAM's
  only where a prognostic aerosol supplies its inputs (JAM) and a jcm closure
  otherwise, and a small set of jcm-only bounds remains (the Koop
  homogeneous-freezing floor, the ``icemax`` cap on the ICNC diagnosis and
  the falling-ice cover threshold). The
  absent processes and the deliberate deviations are listed below. See
  {doc}`../design/lohmann_2m_column_processes`.

  **What drives its condensation.** The 2M is tendency-driven, as ECHAM's
  ``cloud_micro_interface``: section 5 condenses ``zqcdif = (ztmst·pqte −
  zdqsat)·paclc``, the humidity increment the saturation humidity does not
  absorb, into the cloudy part of the cell, which is taken to be saturated at
  the anchor. The term takes ECHAM's inputs from
  ``jcm/physics/clouds/cloud_inputs.py::cloud_scheme_inputs``: the anchor
  (``ptm1``, ``pqm1``, ``pxlm1``, ``pxim1`` and the number tracers) is the
  previous step's post-physics state, carried by the model; the increments
  are the dynamics of the last step plus every term upstream of the scheme —
  radiation, vertical diffusion with its condensate change, the surface and
  convection; and the convective detrainment arrives by itself as
  ``detrained_qc``/``detrained_qi`` (``pxtecl``/``pxteci``), which the scheme
  treats by ECHAM's rules for detrained condensate (below). So large-scale ascent and cloud-top radiative
  cooling condense in a partly cloudy cell in the step they happen, without
  waiting for the whole cell to saturate. Every quantity ECHAM evaluates at
  the previous time level (the saturation humidities, the subsaturations for
  evaporation and sublimation, the moist heat capacity) is evaluated at this
  anchor. On a first step, after a restart from a checkpoint without the
  carried state and in the single-column and RCE hosts, the anchor is the
  step-start state and the dynamics increment is zero. See
  {doc}`../design/operator_split_physics`.

Both schemes convert a condensed/evaporated/frozen mixing-ratio increment to a
temperature increment with ``L / cp`` where ``cp`` is the **moist** isobaric
heat capacity ``cpd·(1 + vtmpc2·q)`` evaluated per-level at the step-start
humidity — ECHAM's ``zlvdcp = alv/pcair`` / ``zlsdcp = als/pcair``
(``mo_cloud.f90``, ``mo_cloud_micro_2m.f90``), shared through
``cloud_utils.latent_heat_over_cp``. Dry ``cpd`` would over-heat every
condensation event by ``vtmpc2·q`` (~1.5 % in the moist tropics); the column
enthalpy budget closes against this same moist ``cp``.

Saturation is Sonntag (1990), the formula ECHAM's lookup tables hold, from
``jcm/physics/thermodynamics.py`` (see {doc}`constants`); the cover and the 1M
read it through ``echam_saturation``. The cover's ``q_s`` and the 1M's
condensation take the ice fit where ``lo2`` holds and the water fit elsewhere
(``mo_cover.f90`` l.215-224; ``mo_cloud.f90`` l.697-699, and
``lookup_ua_eor_uaw_spline`` for the whole-box check, l.763-765). The 1M's
snow sublimation reads ECHAM's ``ua`` table (ice at and below ``tmelt``, water
above; l.451) and its rain evaporation the ``uaw`` table (water at every
temperature; l.520-522), both at the step-start temperature (l.392-394). The
2M scheme reads ECHAM's ``uaw`` for its water saturations and the ``ua``
table for its ice saturations, and picks between them with its own ``lo2`` in
the condensation (``mo_cloud_micro_2m.f90``).

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
  step in both phases (lines 970–972). The prefactor is ECHAM's as written:
  ``conv_effr2mvr`` times the inverse of the plate mass–radius law at the
  plate dimension ``2·zrid``, since ``(0.5·10⁻²/zrid)^pow_PK`` is that
  dimension in centimetres to the power ``−pow_PK``.
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
  1780–1781; lines 3625–3628). For its own arithmetic the scheme floors the
  incoming number tracers at ``cqtmin`` (lines 600–605) and applies no upper
  bound at entry: ICNC is capped at ``icemax`` only after ``znidetr`` joins
  it (line 1252), and CDNC is not capped. The returned tendency is
  ``(N_end/ρ − q_N)/Δt`` against the unclipped provisional tracer ``q_N``
  (anchor plus increment), which the host adds it to. The tracer
  therefore ends the step at the scheme's number, or at zero where the
  negative-mass repair removed the condensate (lines 3641–3652), and an
  out-of-range value does not persist from step to step.

The Tiedtke term records the condensate it detrains each step as
``clouds.conv_detrainment_qc`` and ``clouds.conv_detrainment_qi``
[kg kg⁻¹ s⁻¹], its own split at ``tmelt``, alongside the ``clouds.qc`` /
``clouds.qi`` it advances; the cover term resets them to zero every step.
Both fields are written to the output. Both cloud schemes receive them
through ``cloud_scheme_inputs``, separately from the other increments: the
1M uses Tiedtke's split as ECHAM's ``cloud`` does, and the 2M re-splits their
sum as described above.

Cloud parameters are ``flax.struct.dataclass`` leaves (differentiable), threaded
through the scheme via ``nnx.Param``; only genuine code-path switches
(``nic_cirrus``, ``ldyn_cdnc_min``) are static aux.

**What ECHAM/CAM does.** ECHAM6-HAM2.3 uses the **Sundqvist (1989)** statistical
cloud cover (Lohmann & Roeckner 1996) in ``mo_cover.f90``, with **Lohmann et al.
two-moment microphysics** (``mo_cloud_micro_2m.f90``; Lohmann et al. 2007, Lohmann
& Hoose 2009; Roeckner et al. 2003 for the Marshall-Palmer precipitation
inversion). CAM6 uses **MG2/PUMAS** two-moment microphysics (Gettelman & Morrison
2015; ``micro_mg`` / ``micro_pumas_v1``). DeMott et al. (2010), *PNAS*
doi:10.1073/pnas.0910818107 is the ice-nucleating-particle count. With
prognostic HAM aerosol, ``cloud_subm_1`` calls ``mo_ham_freezing.f90::ham_IN_setup``,
which reduces the M7 modes to the dust and black-carbon fractions and insoluble
radii ``het_mxphase_freezing`` reads (mo_cloud_micro_2m.f90 lines 726–734,
1552–1567). Without it, ECHAM's two-moment scheme takes its aerosol inputs
from the ``lccnclim`` mode: ``ccnclim_IN_setup``
(``mo_ccnclim.f90:524-583``) feeds the aerosol freezing routine
``het_mxphase_freezing`` constant dust and black-carbon fractions (lines
69–73) and fixed insoluble-mode radii (lines 74–77), and sets the cirrus
soluble-aerosol number from a CCN climatology with a floor of 10⁷ kg⁻¹.

**Why we differ.**
- Cover, `differentiability` — the cover's value is ECHAM's, but where its
  derivative is useless the derivative is that of a named smooth surrogate
  ({doc}`../design/surrogate_gradients`). The clip of ``b0`` has plateaux
  with zero slope and the square root an unbounded slope at saturation; the
  derivative is that of ``1 − sqrt(1 − b0_s)`` with the softplus clip
  ``b0_s = w·softplus(b0/w) − w·softplus((b0 − 1)/w)``, width
  ``smooth_b0 = 0.02``, whose slope is bounded by ``1/(2 sqrt(w ln 2))`` ≈ 4.2
  and which lies within ``sqrt(w ln 2)`` ≈ 0.12 of the cover (at ``b0 = 1``).
  The inversion search's stability test ``best > −cinv·g/cp`` is a threshold;
  its derivative is that of the sigmoid ``σ((best + cinv·g/cp)/w)``, width
  ``smooth_inv_thr = 2e-4`` K/m, which gives ``cinv`` and the chosen level's
  lapse rate a gradient near the threshold. Which level the search chooses is
  piecewise constant in the temperature and keeps its reference derivative,
  zero: a smooth selection over levels would differ from the value by the
  whole enhancement wherever two levels compete. The ``lo2`` phase switch
  keeps its reference derivative (that of the branch in use). Widths of zero
  select the reference derivatives.
- Cover, saturation vapour pressure — ECHAM's: the Sonntag (1990) fit that
  ECHAM's lookup tables hold (``mo_echam_convect_tables.f90``), evaluated
  analytically rather than through the 0.025 K spline (they agree to 1e-10),
  from ``jcm/physics/thermodynamics.py`` through
  ``jcm/physics/clouds/echam_saturation.py``. With it the cover
  reproduces every column of ECHAM's own reference.
- Radiation's cover, `science` — ECHAM's radiation uses the cover only where
  the step-start grid-mean condensate it radiates is positive
  (``mo_radiation.f90`` l.428-434, ``xq = MAX(xlm1, 0)``,
  ``MERGE(cld_frc, 0, xq_liq > 0 .OR. xq_ice > 0)``), and hands the same
  masked cover to COSP. jcm's radiation does the same in
  ``cloud_data.radiation_cloud_fields``, which every radiation scheme reads.
  jcm's COSP term applies the same mask to a different state: the
  post-microphysics cover with the post-physics condensate and temperature it
  simulates, where ECHAM's COSP reads the radiation's masked step-start
  fields (``mo_psrad_interface.f90`` l.414; ``physc.f90`` l.1348-1356). A cell the
  mask clears has no condensate and so no optical depth; under maximum-random
  overlap it separates the cloud banks above and below it, as in ECHAM's
  sampler. The mask keeps its reference derivative (`differentiability`): the
  cleared cell contributes nothing to any flux through its own optics, and
  the one discrete effect, the bank separation, is already piecewise constant
  in McICA's sampling. The price is that in a cell with cover but no
  condensate the radiation's derivative with respect to that cell's own
  condensate is exactly zero, although a trace of condensate would switch its
  optics on at the full cover; a surrogate on the mask cannot change that,
  because the in-cloud path's clear-cell guard and McICA's sampled sub-column
  masks select on the masked cover (#973). The AeroCom diagnostics read the post-microphysics
  cover, which is ECHAM's written-back ``aclc`` (zero where both condensates
  are below ``ccwmin``), and are not masked again.
- Cover and radiation condensate, time level — the cover reads the state the
  physics receives, which contains the step's dynamics, and so does the
  radiation's condensate (``cloud_data.radiation_cloud_fields``); ECHAM's
  cover reads the ``t − Δt`` fields (``physc.f90`` l.543-548) and its
  radiation the ``xlm1``/``xim1`` of that time level (l.566-573), one
  dynamics step earlier. The cloud schemes map ``xlm1`` to their anchor. The carry holds that
  state's temperature, humidity and condensate (the cloud schemes' anchor)
  but not its pressures or geopotential; the cover reads the received state
  by the maintainer's decision. The cloud schemes' ``papm1`` (their air
  density and saturation) is likewise the received state's pressure, one
  dynamics step after the anchor: the difference is the surface-pressure
  change of one step, about 3e-4 of the pressure for a cyclone deepening
  1 hPa per hour in a 900 s step.
- Resolution-dependent defaults, `science` — ECHAM sets ``crs``, ``crt``,
  ``nex``, ``nadd``, ``csatsc``, ``cinv``, ``cvtfall``, ``csecfrl`` and
  ``clwprat`` per truncation (``mo_echam_cloud_params.f90::sucloud``; the 2M
  reads the same ``cvtfall``, ``mo_cloud_micro_2m.f90`` l.97, 536) and
  defines them for T31, T63, T127 and T255 only; it has no T106
  configuration. jcm builds these defaults for the run's truncation at physics
  construction (``echam_physics(coords=...)``; the Hydra runner passes the
  grid): the table's values at its four truncations (ECHAM's, with jcm's
  calibrated T63 cover values below), linear interpolation in the truncation
  number between them for the real-valued ones, and the nearer truncation's
  value for the integers ``nex`` and ``nadd``. T106 therefore gets
  ``crs = 0.9632``, ``crt = 0.7267``, ``csatsc = 0.7814``, ``cinv = 0.2379``,
  ``cvtfall = 2.836``, ``csecfrl = 8.36e-6``, ECHAM's ``nex = 2`` and
  T63's (= T127's) ``nadd``, ``clwprat``. The interpolated values are jcm's
  choice, made so that the resolution trend the table encodes is continued
  rather than switched, and they are **untuned**. Outside T31–T255, and on a
  grid with no spectral truncation, the nearest (or T63) values are used with
  a warning. An explicit parameter object or a field override always wins
  (``jcm/physics/resolution_defaults.py``).
- Cover parameters at T63, `science` — the T63 values of ``crt``, ``crs``,
  ``nex``, ``csatsc`` and ``cinv`` are jcm's, not ECHAM's (0.75, 0.975, 2, 0.7,
  0.25): **0.679016061, 0.9, 1.84856084, 0.948216414 and 0.213005383**. The
  cloud fraction is what the 2M host's convection and microphysics levers
  cannot move without adding liquid water, and these five parameters set it.
  The values are the interior arm of a 25-arm Gaussian-process search over
  exactly these parameters on the 2M host (30-day January and July windows, a
  loss over cloud cover, SW and LW cloud radiative effect, liquid water path
  and precipitation against the jcm-monitor climatologies), confirmed in a
  365-day T63 L47 year. That year reads a radiation cover of 0.65 (observed
  0.63; ECHAM's values give 0.58), a net TOA flux of +5.6 W/m² (CERES +1.0;
  +9.8), a reflected SW of 96.9 W/m² (99.0; 90.8) and a LW CRE of 28.2 (27.9;
  26.1), with a SW CRE 2.6 W/m² too strong and 49.6 g/m² of liquid water path
  (ESA-CCI 36.4; 42.1); precipitation, the clear-sky OLR and the tropical
  precipitation extremes are those of ECHAM's values. About half of the added
  cloud lies in the lowest five model levels (the lowest level's mean cover is
  0.17, ECHAM's values give 0.10), because ``crs`` 0.9 is below the humidity of
  a moist surface layer; the sweep's loss does not weigh the vertical placement
  of cloud. The search's own optimum
  lay on the edges of its box (``nex`` 3.96 of an upper bound 4, ``cinv`` 0.5
  of 0.5, ``crs`` 0.9 of a lower bound 0.9) and is not the default. The adopted
  ``crs`` is itself on that lower bound, below every ECHAM value (0.95 at T31 to
  0.994 at T127), and ``csatsc`` and ``cinv`` are weakly constrained by the
  search (#1014). ``nex`` is an INTEGER in ECHAM, but the closure needs none: the
  profile ``rhc`` has a base ``p_s/p >= 1`` and so is continuous and
  differentiable in a real exponent, equal to ``crs`` at the surface and
  tending to ``crt`` aloft. The 1M, the 2M and the JAM-2M hosts read these
  values through the same cover; they were calibrated on the 2M host, and the
  1M host, T106 and the cubed sphere have no calibration of their own (#1014).
  The 1M host's 365-day year with the set reads a cover of 0.48 offline and 0.60
  radiation (0.42 and 0.52 with ECHAM's values), a SW CRE of -45.8 W/m² (-35.5;
  observed -45.7), a reflected SW of 94.7 W/m² (84.2; 99.0), an annual TOA flux
  of -2.7 W/m² (+5.8; CERES +1.0) and a liquid water path of 82 g/m² (61; ESA-CCI
  36); the LW CRE (16.9 against 27.9 observed) and the thin ice are
  microphysics, not cover.
  The JAM-2M host's droplet number comes from the interactive aerosol, so the
  cover and the aerosol scales are one calibration there: with the aerosol
  emission scales fitted on ECHAM's cover parameters (dust 0.379095663, sea salt
  2) the extra cloud adds wet removal (a sea-salt burden of 9.2 mg/m², a total
  AOD of 0.052) and the loss on the Stage-2 windows (461.1) is worse than that of
  the same host on ECHAM's cover (377.6). The shipped aerosol scales (dust 0.344,
  sea salt 4) are fitted with this cover set in place (Stage 2c of the design
  page): the JAM-2M year passes the cover gate (0.52), reads a reflected SW of
  99.3 W/m² and a LW CRE of 27.4 (observed 99.0, 27.9), a net TOA flux of
  +2.7 W/m², a SW CRE 3.1 W/m² too strong, 57 g/m² of liquid water path and a
  total AOD of 0.074 (observed 0.145), and its loss (325.8) is below the
  aerosol-only configuration's (377.6).
  ECHAM's own row stays in ``ECHAM_CLOUD_DEFAULTS``, and the Fortran comparison
  runs on the constants its reference data recorded. Evidence and the year
  tables: {doc}`../design/jam_aerosol_retune` ("Stage 2b: cloud fraction").
- ``csecfrl`` and ``cthomi``, two copies — ECHAM has one ``csecfrl``
  (``mo_echam_cloud_params.f90`` l.76, set per truncation) and one
  ``cthomi`` (l.54), which its cover and its ``cloud`` both read. jcm holds
  each twice: ``CloudParameters.csecfrl``/``t_ice`` for the cover and
  ``MicrophysicsParameters.csecfrl``/``cthomi`` for the 1M (the 2M carries
  its own ``cthomi``). The defaults agree. An override or a calibration of
  one copy leaves the other scheme's value unchanged, and ``echam_physics()``
  warns when the copies it builds differ.
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
- `differentiability` — the 1M scheme keeps ECHAM's values at every switch
  and power law and gives six of them a surrogate derivative
  ({doc}`../design/surrogate_gradients`): the ``lo2`` phase switch (logistic in
  temperature, width ``phase_switch_width`` = 1 K, and in the cloud ice
  relative to ``csecfrl``, width ``phase_switch_ice_width`` = 0.1 of
  ``csecfrl``), the melt of all cloud ice above ``tmelt`` and the freezing of
  all cloud water at or below ``cthomi`` (logistic in temperature, the same
  1 K), the ice fall speed ``cvtfall·(ρ·q_i)^0.16`` (below
  ``ice_fall_speed_gradient_cutoff`` = 1e-7 kg/m³ the derivative is that of a
  parabola through the origin that matches the power law's value and slope at
  the cutoff, and below zero, for negative provisional ice, that of the
  parabola's tangent at the origin; thin cirrus holds 1e-6 to 1e-4 kg/m³, so
  the cutoff lies below real cloud, and the largest slope is 1.4e6 for every
  ice content), and the mean droplet radius of
  contact freezing (the same parabola in the in-cloud liquid below
  ``contact_freezing_liquid_cutoff`` = 1e-10 kg/kg, about a 0.08 µm droplet).
  The sixth is the KK2000 option's threshold, a hard gate with a logistic
  surrogate of width ``smooth_ccraut``. All widths are static fields; zero selects the
  reference derivative. Switches whose value jumps but which keep their
  reference derivative: the clear-cell criterion ``paclc > 0`` (its jump has
  no smooth continuation without a model of partial-cell evaporation, which
  ECHAM 6.3 comments out), the precipitating-fraction reset, the ``ccwmin``
  correction and cover write-back (jumps of at most ``ccwmin``), and the
  low-pressure ``ub`` branch of the supersaturation check.
- `differentiability`: ``het_mxphase_freezing`` evaluates each singular
  operation (the cube root of the droplet volume, ``√TKE``, the ``1/r`` of the
  Brownian diffusivities, ``exp(tmelt − T)``) on a safe argument where ECHAM
  discards the result, so its values are ECHAM's and its derivatives stay
  finite at zero aerosol, TKE, radius or liquid and outside the gate. ECHAM's
  caps on the frozen mass and number are kinks, kept as written.

**Status & known limitations (stated openly).**
- **Mixed-phase heterogeneous freezing** acts in cells with liquid between
  ``cthomi`` and ``tmelt`` (ECHAM's gate ``ll_mxphase_frz``, lines 1541–1545).
  Which formulation runs is set by the composition.
  - **With JAM: ECHAM-HAM's rates.** ``het_mxphase_freezing``
    (``jcm/physics/clouds/lohmann_2m/deposition_freezing.py``; lines
    2675–2840; Lohmann & Diehl 2006, *J. Atmos. Sci.* 63, 968) freezes cloud
    water over the step by Brownian contact freezing on insoluble dust and by
    immersion freezing of droplets that hold dust or black carbon,
    ``(32.3·f_du + 2.91·10⁻³·f_bc)·exp(tmelt − T)·(−min(ztte, 0))·V_drop``
    per second, with the droplet volume ``V_drop = ρ·q_l/(ρ_w·N_l)``
    (montmorillonite; the coefficients are the differentiable
    ``CloudParams2M.immersion_coefficient_dust/_bc``). The frozen number is
    capped at the droplets above ``cdnc_min``; mass, number and fusion heat
    move together and the returned freezing is cover-weighted (line 2837).
    The dust and BC fractions and the insoluble-mode radii are ECHAM-HAM's
    ``ham_IN_setup`` fields, which JAM computes from its population (see
    {doc}`aerosol`, *Aerosol inputs to ice formation*) and publishes as
    ``freezing_aerosol``. With MAM4 there is no insoluble dust, and ECHAM
    disables black-carbon contact freezing (line 2784), so contact freezing
    is zero and the JAM mixed-phase freezing is immersion freezing.
  - The immersion rate acts only while the air cools. ECHAM's cooling rate
    ``ztte`` is the adiabatic cooling of the vertical motion,
    ``(ω − fact_tke·√TKE·ρ·g)/(cpd·ρ)`` (lines 2800–2802), not the model's
    temperature tendency. The large-scale ``ω`` is not plumbed to the scheme
    (#705) and enters as 0: in 10-day T63 JAM runs the TKE updraft cools every
    gate cell, and adding the dycore's ``ω`` changes the immersion rate
    (weighted by ``exp(tmelt − T)`` and the droplet volume) by 0.5 % and turns
    it off in no cell.
  - **Without JAM: an aerosol-free closure.** Droplets of mean mass freeze
    until the crystal number reaches the INP number ``n_inp``, the **DeMott
    et al. (2010)** parameterisation
    (``deposition_freezing.py::demott2010_inp``) on a prescribed number of
    particles larger than 0.5 µm, ``n_aer_coarse`` (0.5 cm⁻³ at standard
    conditions, spatially constant). DeMott et al. report INP per standard
    litre, at 273.15 K and about 1013 hPa, so the function converts to
    ambient air by ``ρ/ρ_STP``. It applies only within the fitted range,
    238–264 K. The droplets available cap the number frozen, the liquid
    available caps the mass, and number, mass and fusion heat move together.
    The droplet number is floored at ``cqtmin``; ECHAM's
    ``het_mxphase_freezing`` instead caps the number frozen at the droplets
    above ``cdnc_min`` (lines 2818–2820). The parameterisation appears in
    neither the ECHAM nor the CAM source tree; ECHAM's own aerosol-free mode
    ``lccnclim`` (see *What ECHAM/CAM does*) feeds ``het_mxphase_freezing``
    constant fractions and would replace it once ``ω`` is plumbed (#705).
  - Heterogeneous freezing is a second-order ice source in both. Detrained ice
    carries its own crystal number, about 10⁵ crystals m⁻² s⁻¹ in T63 January
    runs, against 10²–10³ from heterogeneous freezing, and the freezing itself
    moves 0.01–0.04 g m⁻² d⁻¹ of a mixed-phase ice budget of about 500. Its
    indirect effect is larger under JAM: where ECHAM's immersion rate
    completes, at the cold end of the mixed phase, every frozen droplet
    becomes a crystal, and those crystals seed the WBF transfer and
    deposition. In 10-day T63 JAM runs with ``ccsaut = 900``, going from the
    closure to ECHAM-HAM's rates raised the WBF transfer from 6.3 to 6.9 and the deposition from 252
    to 271 g m⁻² d⁻¹, lowered the liquid water path from 45.6 to 43.1 g m⁻²
    and the supercooled fraction at 238–243 K from 0.52 to 0.45, and changed
    the ice water path by +0.2 g m⁻² (run-to-run noise: 0.75 g m⁻² in the liquid
    path).
  - ``ice_nuclei`` remains an optional input of the closure (an external INP
    number, the larger of it and DeMott is used); no in-tree term publishes
    it. ``ice_nuclei_deposition`` reaches ``update_in_cloud_water`` as
    ECHAM's ``pnicex``, read only by the ``nic_cirrus = 2`` branch (#552).
- **Absent by decision.** These ECHAM-HAM processes are not in the 2M scheme:
  - the droplet number of detrained liquid, ``zqlnuccv`` (lines 889–941), and
    stratiform activation at cloud base copied to the levels above (lines
    742–782). Both are droplet sources outside the ice treatment, and
    ``zqlnuccv`` needs the activated number at convective cloud base
    (``cdncact_cv``), which the Tiedtke scheme does not provide. Detrained
    liquid therefore joins the existing droplet population, and droplets are
    activated only in ``update_in_cloud_water``, where the droplet number has
    fallen to ``cdnc_min`` (lines 2594–2608). Tracked in #955.
  - cirrus nucleation ``zninucl`` at ``nic_cirrus = 1`` (lines 986–999). Its
    cap is the soluble-aerosol number ``zascs``, which the scheme does not
    receive, and the ice budget stands without it (#955).
  - Kärcher–Lohmann cirrus, ``nic_cirrus = 2`` (lines 1001–1117; #552).
  - ECHAM's aerosol-free freezing mode, ``lccnclim`` (see *What ECHAM/CAM
    does*). It is the faithful replacement for the DeMott closure and waits
    for the large-scale vertical velocity (#705).
  - the large-scale ``ω`` in the updraft ``zvervx`` (``−100·ω/(g·ρ)``, line
    816) and in the immersion cooling rate (line 2800; #705).
- **Known biases of the 2M ice.** Four biases remain, measured in 10-day T63
  January runs from a spun-up state. No parameter has been tuned to them: the
  retune swept the 2M levers (``ccraut``, ``ccsaut`` and the three Tiedtke
  levers) and kept the defaults, because its best arm sat on the lower edge of
  four of its five levers and bought SW cloud effect with liquid water path
  (see {doc}`../design/jam_aerosol_retune`).
  - Glaciation is too warm. The supercooled share of the condensate mass
    falls to one half near −7 °C. CALIOP places the crossing near −20 °C, but
    CALIOP's quantity is a cloud-top phase frequency, not a mass fraction, and
    that figure comes from a secondary summary (Hu et al. 2010; Tan et al.
    2016).
  - Cold ice cloud holds too many crystals. In the daily-mean output, two
    fifths of the mixed-phase and three fifths of the cold ice-containing
    cells exceed 10⁶ crystals m⁻³, above the 90th percentile
    of the in-situ cirrus climatology of Krämer et al. (2020, *ACP* 20,
    12569). The cause is the small radius ``zrid`` gives at cirrus
    temperatures, which makes ``znidetr`` large.
  - Liquid water path lies below the observed range (50–84 g m⁻² over the
    oceans; Lohmann et al. 2007, *ACP* 7, 3425, Table 2).
  - Under JAM the mixed-phase condensate stays more liquid than in the non-JAM
    member. Both aggregate ice with ECHAM's generic ``ccsaut = 95``
    (``mo_echam_cloud_params.f90``, line 59). ECHAM-HAM retunes it to 900 in
    ``mo_activ.f90`` at T63 L47 with prognostic CDNC, AR&G activation and
    ``cdnc_min_fixed = 40`` (lines 392–408). That retune belongs with HAM's own
    aerosol and its insoluble dust mode, which jcm's MAM4-based JAM does not
    reproduce, so JAM keeps the generic value. ``ccsaut`` for JAM was one of the
    retune's eight Stage-1 levers (0.5-2 times 95) and keeps 95: its
    standardised effect on the loss was +0.05, against +3.21 for the dust scale.
    In 10-day T63 January runs from a JAM state 10
    days past a cold start, 95 gives an ice water path of 22.3 g m⁻² against
    7.0 with 900. The supercooled mass fraction at 253–258 K is 0.48 against
    0.83. The non-JAM member had 27.8 g m⁻² and 0.28, measured before the
    Sonntag saturation. The aerosol–ice coupling does not explain the gap:
    replacing JAM's INP by DeMott's in the closure changed the ice water path
    by 0.05 g m⁻², and ECHAM-HAM's rates change it by +0.2 g m⁻² (above).
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
- Neither microphysics scheme publishes the radiative effective radii: as in
  ECHAM, the radiation forms them inside its own call from the step's
  condensate and droplet/crystal number (``mo_cloud_optics.f90::cloud_optics``;
  see {doc}`radiation`), from the laws in ``cloud_utils`` that the 2M scheme
  also evaluates for its own ``preffl``/``preffi``. The 1M radiation and the
  1M autoconversion read the prescribed droplet number times the Twomey
  factor, and the 1M's Bigg and contact freezing the unscaled ``acdnc``
  (above). The
  radiative **ice** radius on the 2M path follows the crystal number the
  scheme carries, which comes mainly from ``znidetr`` and the ``zrid``
  diagnosis of the ice-number sources above.
- Clear-sky evaporation of decorrelated condensate (the radiation-side contract in
  ``mcica.in_cloud_path``) is owned by the 2M scheme's clear-sky evaporation step.
- **1M: gravity-wave and orographic drag heating** reach the cloud scheme one
  step late: those terms run after it (≤ 0.03 K/day in the troposphere).
- **1M: the ``lonacc`` zeroing of the local-rain factor** at the cover's
  inversion level is implemented in the column function but not supplied by
  the term, which needs the cover's inversion level (``knvb``) and the
  large-scale vertical velocity; #705 tracks that plumbing, for the 2M's
  matching ``zauloc`` gate too. It is inert at ECHAM6.3, which fixes
  ``cauloc = 0`` as a parameter (``mo_echam_cloud_params.f90`` l.71), as are
  the local-rain accretion and the in-layer snow, which the Fortran
  comparison therefore does not exercise (unit tests pin their formulas).

**Code pointers.**
- ``jcm/physics/clouds/sundqvist.py`` — ``SundqvistCloudFraction``,
  ``calculate_cloud_fraction``.
- ``jcm/physics/clouds/echam_saturation.py`` — ``es_water``, ``es_ice``,
  ``lo2_ice_phase``, ``qsat_from_es``: the Sonntag functions of
  thermodynamics.py under the default formula.
- ``jcm/physics/clouds/cloud_data.py`` — ``radiation_cloud_fields``,
  ``condensate_masked_cover``.
- ``jcm/physics/clouds/echam_cloud_defaults.py`` — ``echam_cloud_defaults``,
  ``ECHAM_CLOUD_DEFAULTS``, ``JCM_CALIBRATED_COVER_T63``,
  ``JCM_CLOUD_DEFAULTS``, ``inversion_levels``.
- ``jcm/physics/resolution_defaults.py`` — ``resolution_defaults``.
- ``jcm/physics/clouds/echam_1m.py`` — ``Echam1MMicrophysics``,
  ``cloud_microphysics_column_sweep``.
- ``jcm/physics/clouds/lohmann_2m/`` — ``scheme.py`` (``cloud_microphysics_2m``,
  ``Lohmann2MMicrophysics``, process-order docstring, the section-1 fields),
  ``deposition_freezing.py``
  (``het_mxphase_freezing``, ECHAM-HAM's rates under JAM; ``demott2010_inp``,
  the aerosol-free closure's INP),
  ``sedimentation_melt.py``, ``precip.py``, ``assembly.py``,
  ``jcm/physics/clouds/lohmann_2m/types.py`` (``HeterogeneousFreezingAerosol``,
  the HAM freezing inputs);
  ``jcm/physics/clouds/lohmann_2m_params.py`` (``CloudParams2M``);
  ``jcm/physics/clouds/cloud_utils.py``
  (``effective_2_volmean_radius_param_Schuman_2011``);
  ``jcm/physics/clouds/cloud_data.py`` (``CloudData``, including
  ``conv_detrainment_qc`` / ``conv_detrainment_qi``).

**Validation evidence.** ``jcm/physics/clouds/echam_fortran_reference_test.py``
(the cover and 1M against the ECHAM6.3 Fortran, column by column, with
``jcm/data/test/echam_cloud_reference/``),
``sundqvist_test.py`` (designed points, the Fortran columns at ECHAM's four
truncations, the surrogates), ``cloud_data_test.py`` (radiation's condensate
mask and its effect on McICA overlap), ``echam_cloud_defaults_test.py``,
``echam_saturation_test.py``, ``echam_1m_test.py``,
``lohmann_2m_test.py``, ``cloud_utils_test.py``.
The ECHAM-HAM comparisons: ``lohmann_2m_fortran_reference_test.py`` (the ice
sources against the compiled ECHAM6.3-HAM2.3 ``cloud_micro_interface``, building
block by building block and end to end: the crystal radius, both ``lo2``
criteria, the sedimentation of the pre-detrainment ice, the ICNC diagnosis, the
detrained crystal number and the number tendencies) and
``lohmann_2m_freezing_reference_test.py`` (heterogeneous
mixed-phase freezing against the compiled ``cloud_micro_interface`` with the HAM
freezing inputs set: ``het_mxphase_freezing`` agrees to round-off on every
freezing column; the large-scale-``ω`` column is a strict xfail, #705);
``lohmann_2m_freezing_test.py`` (gradients, and the aerosol-free path pinned to
its output before the ECHAM-HAM rates). Design
reference: {doc}`../design/lohmann_2m_column_processes`.
