# Gravity-wave drag and upper-boundary dissipation

**What we do.** Four separate composable terms cover the gravity-wave spectrum,
plus two upper-boundary dissipation terms:

- **Hines non-orographic** (``jcm/physics/gravity_waves/hines/hines.py::HinesGwd``):
  a Doppler-spread spectral scheme launching an 8-azimuth wave spectrum from a
  fixed launch level, sweeping cutoff vertical wavenumbers upward, with momentum
  deposition, heating and diffusion.
- **Lott-Miller SSO** (``jcm/physics/gravity_waves/sso/lott_miller.py::LottMillerSso``):
  sub-grid orographic drag with blocked-flow form drag below the blocking level
  and a saturated gravity-wave stress profile above, from seven sub-grid
  orography descriptors (derived on the fly from Baines-Palmer statistics when no
  preprocessed DEM is available).
- **Frontal spectral GWD** (``jcm/physics/gravity_waves/spectral/term.py::FrontalGravityWaveDrag``):
  a faithful JAX port of CAM's spectral non-orographic scheme with a
  frontogenesis-triggered source. See {doc}`../design/frontal_gravity_wave_drag`.
- **Simple GWD fallback** (``jcm/physics/gravity_waves/simple/simple_gwd.py::SimpleGwd``):
  a single-wave orographic-source breaking scheme.

Upper-boundary dissipation is two terms: the ECHAM-style **upper sponge**
(``jcm/physics/dissipation/upper_sponge.py::UpperSponge``) — Rayleigh drag on
(u, v) and relaxation of T toward its zonal mean over the top N levels — and
``jcm/physics/dissipation/upper_temperature_relaxation.py::UpperTemperatureRelaxation``,
a Newtonian relaxation of the top-level *temperatures* toward a reference profile
(e.g. USSA-1976) purpose-built for finite mesospheric lids, with an *optional*
Rayleigh-friction wind branch following CAM's ``rayleigh_friction.F90`` tanh
profile (Euler-backward, dissipated KE returned as heat).

**What ECHAM/CAM does.** Hines (1997, *JGR*, Doppler-spread parameterization) is
ECHAM's ``gwspectrum`` (``mo_gwspectrum.f90``). Lott & Miller (1997, *QJRMS*)
sub-grid orographic drag is ECHAM's ``ssodrag`` (``mo_ssodrag.f90``). The frontal
scheme is CAM's non-orographic spectral GWD with the Charron & Manzini (2002)
frontogenesis source (ESCOMP/CAM ``cam_cesm2_2_rel``: ``gw_common.F90`` +
``gw_front.F90`` + ``gw_drag.F90``). ECHAM's upper sponge is ``uspnge``
(``mo_upper_sponge.f90``); CAM's momentum lid damping is ``rayleigh_friction.F90``.

**Why we differ.**
- `science` — Hines omits the four optional ECHAM branches
  (latitude/precipitation-modulated launch variance, ``lfront``, ``icutoff``
  damping — all off in ECHAM-A defaults) and hard-codes orographic coupling
  ``sigsqmcw = 0`` (orographic waves handled by Lott-Miller). Lott-Miller omits
  the mountain-lift branch (``gklift = 0`` in production ECHAM). The frontal port
  omits ``gw_diffusion.F90`` (matching CAM's ``lapply_vdiff=.false.``), the
  ridge/IGW/top-taper options and history plumbing (full list in
  {doc}`../design/frontal_gravity_wave_drag`).
- `differentiability` — a heating-bounded stability limiter (``limit_tendency_sum``,
  default on) caps ``Σ_l |gwut_l|`` rather than only the net, because ECHAM/CAM
  never operate this scheme with a lid layer as thin as ECHAM L47 (``ρ→0`` drives
  ~123 K/day heating there); every masked division/sqrt keeps its safe operand
  inside ``jnp.where`` for finite reverse-mode gradients (#558).
- `compute` — the upper sponge damps the full (u, v) field rather than only
  ECHAM's m≠0 spectral modes; ``enspodi`` defaults to 2.0 (softening downward)
  rather than ECHAM's uniform 1.0.

**Status & known limitations.** ``echam_physics(gw_scheme=...)`` selects
``"hines"`` (default), ``"frontal"``, ``"both"`` (Hines broad-spectrum
background + frontal storm-track deposition — frontal-only under-drags the
subtropical jet, and some double-counting near strong fronts is accepted;
retune ``taubgnd`` and the Hines source strength jointly if it shows) or
``"none"``. The frontal term is inert without a dycore-supplied
``"frontogenesis"`` diagnostic; a lat-lon provider exists for the dinosaur
backend (``DinosaurDycore(compute_frontogenesis=True)``), while the pySES/pg2
unstructured-grid provider is a follow-up (#568/#564).
``UpperTemperatureRelaxation`` is primarily a temperature relaxation; the CAM
``rayleigh_friction.F90`` tanh form appears only in its optional wind branch.

**Code pointers.**
- ``jcm/physics/gravity_waves/hines/hines.py`` — ``HinesGwd``, ``hines_gwd``.
- ``jcm/physics/gravity_waves/sso/lott_miller.py`` — ``LottMillerSso``,
  ``sso_drag``.
- ``jcm/physics/gravity_waves/spectral/`` — ``term.py`` (``FrontalGravityWaveDrag``),
  ``solver.py`` (← ``gw_common.F90``), ``frontal.py`` (← ``gw_front.F90``),
  ``frontogenesis.py``.
- ``jcm/physics/gravity_waves/simple/simple_gwd.py`` — ``SimpleGwd``.
- ``jcm/physics/dissipation/upper_sponge.py`` (``UpperSponge``),
  ``upper_temperature_relaxation.py`` (``UpperTemperatureRelaxation``).

**Validation evidence.** ``jcm/physics/gravity_waves/hines/hines_test.py``;
``sso/lott_miller_test.py``, ``lott_miller_host_test.py``;
``spectral/solver_test.py`` (NumPy float64 reference), ``frontal_test.py``,
``frontogenesis_test.py``, ``term_test.py``; ``simple/simple_gwd_test.py``;
``dissipation/upper_temperature_relaxation_test.py``. Design reference:
{doc}`../design/frontal_gravity_wave_drag`.
