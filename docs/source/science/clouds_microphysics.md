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
  port of the same linearised-Newton step).
- **ECHAM 1-moment microphysics**
  (``jcm/physics/clouds/echam_1m.py::Echam1MMicrophysics``) — a flux-coupled
  top-down column sweep: autoconversion (Beheng 1994 default or KK2000),
  accretion, ice→snow aggregation (Levkov 1992), riming, snow/ice melt, ice
  sedimentation, Rotstayn (1997) rain evaporation. Ports the ECHAM6/ICON
  ``mo_cloud.f90`` single-moment branch. The ice/snow fall-speed factor
  ``cvtfall = 2.5`` is ECHAM's value for jcm's default T63 grid
  (``mo_echam_cloud_params.f90``, ``nn == 63``), the same the 2M scheme uses.
  Its cloud-ice fall speed retains ECHAM's forward density power
  ``v = cvtfall (max(rho q_i, d_epsilon))^0.16`` and original zero-ice outer
  gate exactly. During automatic differentiation, the default
  ``ice_fall_speed_derivative_cutoff=1e-10`` kg m-3 substitutes the bounded
  local slope of a C1 continuation below the cutoff; at and above the cutoff
  it uses the true ECHAM derivative. This is a surrogate derivative: forward
  AD below the cutoff intentionally differs from finite differences of the
  unchanged primal. Pass ``ice_fall_speed_derivative_cutoff=0`` for the
  original JAX derivative. The static default is a numerical threshold, not a
  physically validated cloud-ice boundary.
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
- `science` (deliberate, documented) — the 2M column sweep uses **MG/PUMAS
  sediment→melt ordering** (ice sedimentation before melt), *not* ECHAM's
  melt→sediment. The melt acts on the post-sedimentation ice through the threaded
  tendency so the two sinks cannot claim the same mass. This is the one intentional
  cross-reference ordering deviation in an otherwise pure-ECHAM tree; the process
  docstring in ``scheme.py`` names it explicitly.
- `differentiability` — activation, freezing and precipitation gates are
  formulated to keep cloud parameters differentiable; there is no import-time
  default parameter instance (that would sever gradients / overrides).

**Status & known limitations (stated openly).**
- The default 1M low-ice fall-speed surrogate derivative is a
  differentiability-driven numerical treatment related to the
  gradient-regularisation work in #843. It bounds the formal ``q_i -> 0+``
  local slope without deleting ice or changing any forward result or the
  resolved-ice derivative. A one-date, zero-radiation calibration diagnostic
  found finite three- and eight-step gradients and three successful updates,
  but this is not full-model reverse-mode qualification, physical validation
  of the threshold, or evidence that every coupled multi-step gradient is
  useful.
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
- Both the 1M and 2M paths publish an LWC-dependent radiative liquid radius
  from the shared ECHAM Martin/Bower law (``eff_liquid_droplet_radius``);
  radiation reads it from the carried ``clouds`` state one step lagged, because
  the ECHAM term order runs radiation before microphysics. The constant
  ``effective_radius_liquid`` fallback therefore survives only where that carry
  is still zero, resolved **cell by cell** — the cold-start first step, and
  thereafter any cloudy cell that was clear the previous step (a level newly
  turning cloudy falls back even mid-rollout in an otherwise-cloudy column) —
  not the steady state the 1M ``physics=echam`` path used to run on. The
  radiative **ice** radius remains limited: mixed-phase ICNC is
  INP-limited (~1e3 m⁻³), pinning most warm-branch ``r_eff_ice`` at the 150 µm
  clip (#728).
- Clear-sky evaporation of decorrelated condensate (the radiation-side contract in
  ``mcica.in_cloud_path``) is owned by the 2M scheme's clear-sky evaporation step.

**Code pointers.**
- ``jcm/physics/clouds/sundqvist.py`` — ``SundqvistCloudFraction``,
  ``calculate_cloud_fraction``, ``condensation_evaporation``.
- ``jcm/physics/clouds/echam_1m.py`` — ``Echam1MMicrophysics``.
- ``jcm/physics/clouds/lohmann_2m/`` — ``scheme.py`` (``cloud_microphysics_2m``,
  ``Lohmann2MMicrophysics``, process-order docstring), ``deposition_freezing.py``
  (``demott2010_inp`` used; ``het_mxphase_freezing`` defined/exported/unused),
  ``sedimentation_melt.py``, ``precip.py``, ``assembly.py``,
  ``jcm/physics/clouds/lohmann_2m/types.py``;
  ``jcm/physics/clouds/lohmann_2m_params.py`` (``CloudParams2M``).

**Validation evidence.** ``jcm/physics/clouds/sundqvist_test.py`` and
``sundqvist_smooth_gradients_test.py``, ``echam_1m_test.py``,
``lohmann_2m_test.py``, ``cloud_utils_test.py``, ``cloud_data_test.py``. Design
reference: {doc}`../design/lohmann_2m_column_processes`.
