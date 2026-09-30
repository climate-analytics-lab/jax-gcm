# Clouds and microphysics

**What we do.** A diagnostic cloud fraction plus a choice of **single-moment** or
**two-moment** microphysics:

- **Cloud cover** (``jcm/physics/clouds/sundqvist.py::SundqvistCloudFraction``)
  — ECHAM6.3's ``mo_cover.f90::cover``, the Sundqvist (1989) / Lohmann and
  Roeckner (1996) relative-humidity closure, used by the 1M, the 2M and the
  JAM configurations alike. The value is ECHAM's at every level:
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
  reference tolerance, at T31, T63, T127 and T255. It is a **pure diagnostic**
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
  incoming rain (Rotstayn 1997), all at the anchor (step-start) state; ice
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
  Every output and every intermediate the Fortran harness exposes is compared
  with the unmodified Fortran routine on 42 designed and sampled columns
  (``echam_fortran_reference_test.py``). The increments ``Δq``, ``ΔT``,
  ``Δq_c``, ``Δq_i`` are ``dt`` times the running tendency of every physics
  term composed before the 1M term (radiation, vertical diffusion with its
  condensate, the surface, convection) on the step-start state as anchor, with
  the convective detrainment passed separately as ECHAM's ``pxtecl``/``pxteci``.
  The saturation vapour pressure and its slope are ECHAM's Sonntag (1990) fit,
  from ``echam_saturation``, the module the cover reads. ``cvtfall``,
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
  (``jcm/physics/clouds/lohmann_2m/scheme.py`` — ``cloud_microphysics_2m`` and its
  ``Lohmann2MMicrophysics`` term) — the full two-moment process chain (droplet and
  ICNC number) run as one flux-coupled top-down ``lax.scan``, a faithful
  transcription of ECHAM's column loop: ice sedimentation → melt →
  sublimation/rain-evaporation → clear-sky evaporation → grid-scale condensation
  and supersaturation corrections → in-cloud water update + droplet activation /
  ICNC nucleation → homogeneous freezing → mixed-phase freezing + WBF →
  precipitation geometry → warm-rain (KK2000) and cold precipitation formation.
  This tree is **essentially pure ECHAM-HAM**: every constant traces to
  ``mo_cloud_utils.f90`` / ``mo_echam_cloud_params.f90`` / ``mo_cloud_micro_2m.f90``
  (``jcm/physics/clouds/lohmann_2m_params.py::CloudParams2M``), with no CAM
  ``micro_mg`` / PUMAS ``qsmall`` / ``mincld`` / ``dcs`` constants. See
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
  ``detrained_qc``/``detrained_qi`` (``pxtecl``/``pxteci``), used where ECHAM
  uses ``pxlte + pxtecl``. So large-scale ascent and cloud-top radiative
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

Cloud parameters are ``flax.struct.dataclass`` leaves (differentiable), threaded
through the scheme via ``nnx.Param``; only genuine code-path switches
(``nic_cirrus``, ``ldyn_cdnc_min``) are static aux.

**What ECHAM/CAM does.** ECHAM6-HAM2.3 uses the **Sundqvist (1989)** statistical
cloud cover (Lohmann & Roeckner 1996) in ``mo_cover.f90``, with **Lohmann et al.
two-moment microphysics** (``mo_cloud_micro_2m.f90``; Lohmann et al. 2007, Lohmann
& Hoose 2009; Roeckner et al. 2003 for the Marshall-Palmer precipitation
inversion). CAM6 uses **MG2/PUMAS** two-moment microphysics (Gettelman & Morrison
2015; ``micro_mg`` / ``micro_pumas_v1``). DeMott et al. (2010), *PNAS*
doi:10.1073/pnas.0910818107 is the ice-nucleating-particle count.

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
  from ``jcm/physics/clouds/echam_saturation.py``. With it the cover
  reproduces every column of ECHAM's own reference.
- Radiation's cover, `science` — ECHAM's radiation uses the cover only where
  the step-start grid-mean condensate it radiates is positive
  (``mo_radiation.f90`` l.428-434, ``xq = MAX(xlm1, 0)``,
  ``MERGE(cld_frc, 0, xq_liq > 0 .OR. xq_ice > 0)``), and hands the same
  masked cover to COSP. jcm does the same in
  ``cloud_data.radiation_cloud_fields``, which every radiation scheme reads,
  and in the COSP term (masked by the condensate COSP is given). A cell the
  mask clears has no condensate and so no optical depth; under maximum-random
  overlap it separates the cloud banks above and below it, as in ECHAM's
  sampler. The mask keeps its reference derivative (`differentiability`): the
  cleared cell contributes nothing to any flux through its own optics, and
  the one discrete effect, the bank separation, is already piecewise constant
  in McICA's sampling. The AeroCom diagnostics read the post-microphysics
  cover, which is ECHAM's written-back ``aclc`` (zero where both condensates
  are below ``ccwmin``), and are not masked again.
- Cover, time level — the cover reads the state the physics receives, which
  contains the step's dynamics; ECHAM's reads the ``t − Δt`` fields
  (``physc.f90`` l.543-548), one dynamics step earlier. Reading ECHAM's state
  would need the previous step's post-physics state in the carry.
- Resolution-dependent defaults, `science` — ECHAM sets ``crs``, ``crt``,
  ``nex``, ``nadd``, ``csatsc``, ``cinv``, ``cvtfall``, ``csecfrl`` and
  ``clwprat`` per truncation (``mo_echam_cloud_params.f90::sucloud``) and
  defines them for T31, T63, T127 and T255 only; it has no T106
  configuration. jcm builds these defaults for the run's truncation at physics
  construction (``echam_physics(coords=...)``; the Hydra runner passes the
  grid): ECHAM's values at its four truncations, linear interpolation in the
  truncation number between them for the real-valued ones, and the nearer
  truncation's value for the integers ``nex`` and ``nadd``. T106 therefore
  gets ``crs = 0.9878``, ``cvtfall = 2.836``, ``csecfrl = 8.36e-6`` and T63's
  (= T127's) other values. The interpolated values are jcm's choice, made so
  that the resolution trend ECHAM encodes is continued rather than switched,
  and they are **untuned**. Outside T31–T255, and on a grid with no spectral
  truncation, the nearest (or T63) values are used with a warning. An explicit
  parameter object or a field override always wins
  (``jcm/physics/resolution_defaults.py``).
- `science` (deliberate, documented) — the 2M column sweep uses **MG/PUMAS
  sediment→melt ordering** (ice sedimentation before melt), *not* ECHAM's
  melt→sediment. The melt acts on the post-sedimentation ice through the threaded
  tendency so the two sinks cannot claim the same mass. This is the one intentional
  cross-reference ordering deviation in an otherwise pure-ECHAM tree; the process
  docstring in ``scheme.py`` names it explicitly.
- `differentiability` — activation, freezing and precipitation gates are
  formulated to keep cloud parameters differentiable; there is no import-time
  default parameter instance (that would sever gradients / overrides).
- `differentiability` — the 1M scheme keeps ECHAM's values at every switch
  and power law and gives four of them a surrogate derivative
  ({doc}`../design/surrogate_gradients`): the ``lo2`` phase switch (logistic in
  temperature, width ``phase_switch_width`` = 1 K, and in the cloud ice
  relative to ``csecfrl``, width ``phase_switch_ice_width`` = 0.1 of
  ``csecfrl``), the melt of all cloud ice above ``tmelt`` and the freezing of
  all cloud water at or below ``cthomi`` (logistic in temperature, the same
  1 K), the ice fall speed ``cvtfall·(ρ·q_i)^0.16`` (below
  ``ice_fall_speed_gradient_cutoff`` = 1e-7 kg/m³ the derivative is that of a
  parabola through the origin that matches the power law's value and slope at
  the cutoff; thin cirrus holds 1e-6 to 1e-4 kg/m³, so the cutoff lies below
  real cloud, and the largest slope is 1.4e6), and the mean droplet radius of
  contact freezing (the same parabola in the in-cloud liquid below
  ``contact_freezing_liquid_cutoff`` = 1e-10 kg/kg, about a 0.08 µm droplet).
  The KK2000 option's threshold is a hard gate with a logistic surrogate of
  width ``smooth_ccraut``. All widths are static fields; zero selects the
  reference derivative. Switches whose value jumps but which keep their
  reference derivative: the clear-cell criterion ``paclc > 0`` (its jump has
  no smooth continuation without a model of partial-cell evaporation, which
  ECHAM 6.3 comments out), the precipitating-fraction reset, the ``ccwmin``
  correction and cover write-back (jumps of at most ``ccwmin``), and the
  low-pressure ``ub`` branch of the supersaturation check.

**Status & known limitations (stated openly).**
- **Ice treatment is much unresolved** and depends on choices that exist in no
  single reference. The live heterogeneous ice-nucleating-particle path is a
  prognostic JAM ``ice_nuclei`` field where an online dust/BC source exists —
  ``n_inp = where(ice_nuclei > 0, ice_nuclei, demott_floor)`` — so JAM+2M
  configurations run genuine aerosol–ice-cloud coupling and the **DeMott et al.
  (2010)** parameterisation
  (``jcm/physics/clouds/lohmann_2m/deposition_freezing.py::demott2010_inp``,
  called from ``jcm/physics/clouds/lohmann_2m/scheme.py``) is the floor in cells
  with no online source. That floor appears in neither the ECHAM nor the CAM
  source tree. A
  faithful ECHAM-style ``het_mxphase_freezing`` transliteration is defined and
  exported alongside it but is currently **unused**.
- **Cirrus ICNC diagnosis (default ``nic_cirrus=1``) is bounded by a maximum
  crystal number, not a minimum size.** In cold, cloudy cells that arrive at or
  below ``icemin`` the ice-crystal number is diagnosed by inverting the
  ice-mass / volume-mean-radius relation,
  ``N = rho q_i / ((4/3) pi r^3 rho_ice)``
  (``jcm/physics/clouds/lohmann_2m/assembly.py::update_in_cloud_water``). The
  diagnosed number is **capped at ``icemax``** (1e7 m⁻³). This follows ECHAM-HAM,
  whose own ``nic_cirrus==1`` branch caps the nucleated number by the available
  soluble-aerosol count, ``MIN(candidate, zascs)``
  (``mo_cloud_micro_2m.f90``, the Lohmann (2002) ICNC scheme) — a bound on
  crystal *number*. jcm does not plumb the aerosol number into this scheme, so
  ``icemax`` — the same maximum-plausible ICNC the scheme already clamps the ice
  tracer to (``scheme.py``) — stands in as the ceiling. A fixed minimum crystal
  *radius* was rejected as the physical bound: at ~100 nm the mass inversion
  still yields ~1e11–1e12 m⁻³ (far above realistic cirrus, ~1e3–1e6 m⁻³, and
  above ``icemax``). The 100 nm value survives only as ``cirrus_min_ice_radius``,
  the coarse-mode-aerosol lower bound (a differentiable parameter leaf) that
  floors the radius; the inversion itself is non-dimensionalised by a *static*
  1 µm scale so no gradient path — through the state **or** the parameter —
  divides by a tiny cube (`differentiability`; the bare ``C/r³`` form's
  ``1/r⁶`` gradient overflowed float32 below r ≈ 3e-7 m, and using the
  parameter itself as the scale put the same overflow on its own gradient).
- Neither microphysics scheme publishes the radiative effective radii: as in
  ECHAM, the radiation forms them inside its own call from the step's
  condensate and droplet/crystal number (``mo_cloud_optics.f90::cloud_optics``;
  see {doc}`radiation`), from the laws in ``cloud_utils`` that the 2M scheme
  also evaluates for its own ``preffl``/``preffi``. The 1M radiation and
  microphysics see the same prescribed droplet number (above). The
  radiative **ice** radius on the 2M path is limited by the crystal number:
  mixed-phase ICNC is INP-limited (~1e3 m⁻³), which puts most warm-branch
  crystals at the top of the size range the radiation's tables cover (#728).
- Clear-sky evaporation of decorrelated condensate (the radiation-side contract in
  ``mcica.in_cloud_path``) is owned by the 2M scheme's clear-sky evaporation step.
- **1M: the dynamics is not in the increments.** ECHAM's increments at
  ``cloud`` contain the dynamics of the step (advection and the adiabatic
  term); jcm's contain the upstream physics only, because the dynamics is
  already in the step-start state. In a partly cloudy box, large-scale ascent
  therefore forms no condensate through ``zqcdif`` until the whole box exceeds
  saturation and the 1 % supersaturation check condenses the excess. The missed
  in-cloud forcing is about +6 g/kg/day in extratropical ascent (T63 samples).
  Supplying it needs the previous step's post-physics state as the anchor.
- **1M: gravity-wave and orographic drag heating** reach the cloud scheme one
  step late: those terms run after it (≤ 0.03 K/day in the troposphere).
- **1M: the ``lonacc`` zeroing of the local-rain factor** at the cover's
  inversion level is implemented in the column function but not supplied by
  the term (it needs the cover's inversion level); it is inert at ECHAM6.3's
  ``cauloc = 0``, as are the local-rain accretion and the in-layer snow, which
  the Fortran comparison therefore does not exercise (unit tests pin their
  formulas).

**Code pointers.**
- ``jcm/physics/clouds/sundqvist.py`` — ``SundqvistCloudFraction``,
  ``calculate_cloud_fraction``.
- ``jcm/physics/clouds/echam_saturation.py`` — ``es_water``, ``es_ice``,
  ``lo2_ice_phase``, ``qsat_from_es``.
- ``jcm/physics/clouds/cloud_data.py`` — ``radiation_cloud_fields``,
  ``condensate_masked_cover``.
- ``jcm/physics/clouds/echam_cloud_defaults.py`` — ``echam_cloud_defaults``,
  ``inversion_levels``.
- ``jcm/physics/resolution_defaults.py`` — ``resolution_defaults``.
- ``jcm/physics/clouds/echam_1m.py`` — ``Echam1MMicrophysics``,
  ``cloud_microphysics_column_sweep``.
- ``jcm/physics/clouds/lohmann_2m/`` — ``scheme.py`` (``cloud_microphysics_2m``,
  ``Lohmann2MMicrophysics``, process-order docstring), ``deposition_freezing.py``
  (``demott2010_inp`` used; ``het_mxphase_freezing`` defined/exported/unused),
  ``sedimentation_melt.py``, ``precip.py``, ``assembly.py``,
  ``jcm/physics/clouds/lohmann_2m/types.py``;
  ``jcm/physics/clouds/lohmann_2m_params.py`` (``CloudParams2M``).

**Validation evidence.** ``jcm/physics/clouds/echam_fortran_reference_test.py``
(the cover and 1M against the ECHAM6.3 Fortran, column by column, with
``jcm/data/test/echam_cloud_reference/``),
``sundqvist_test.py`` (designed points, the Fortran columns at ECHAM's four
truncations, the surrogates), ``cloud_data_test.py`` (radiation's condensate
mask and its effect on McICA overlap), ``echam_cloud_defaults_test.py``,
``echam_saturation_test.py``, ``echam_1m_test.py``,
``lohmann_2m_test.py``, ``cloud_utils_test.py``, ``cloud_data_test.py``. Design
reference: {doc}`../design/lohmann_2m_column_processes`.
