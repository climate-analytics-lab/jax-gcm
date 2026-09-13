# Clouds and microphysics

**What we do.** A diagnostic cloud fraction plus a choice of **single-moment** or
**two-moment** microphysics:

- **Sundqvist diagnostic cloud fraction**
  (``jcm/physics/clouds/sundqvist.py::SundqvistCloudFraction``) — RH-based cloud
  fraction with a stratocumulus inversion enhancement (``mo_cover.f90``). It is
  a **pure diagnostic** — the term emits zero T/q/qc/qi tendencies; the
  saturation adjustment lives downstream in each microphysics scheme (the 2M
  path's ``mixed_phase_deposition_and_corrections``, and ``echam_1m.py``'s own
  port of the same linearised-Newton step).
- **ECHAM 1-moment microphysics**
  (``jcm/physics/clouds/echam_1m.py::Echam1MMicrophysics``) — a flux-coupled
  top-down column sweep: autoconversion (Beheng 1994 default or KK2000),
  accretion, ice→snow aggregation (Levkov 1992), riming, snow/ice melt, ice
  sedimentation, Rotstayn (1997) rain evaporation. Ports the ECHAM6/ICON
  ``mo_cloud.f90`` single-moment branch.
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
- The 1M ``physics=echam`` path has no LWC dependence in its radiative liquid
  radius (#717) — live on the release-validated ``t63-echam-1m`` /
  ``t106-echam-1m`` configurations; 2M paths use microphysical radii.
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
