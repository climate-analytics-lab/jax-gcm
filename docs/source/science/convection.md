# Convection

**What we do.** Three interchangeable convection schemes:

- **Tiedtke-Nordeng mass-flux**
  (``jcm/physics/convection/tiedtke_nordeng/tiedtke_nordeng.py::TiedtkeConvection``)
  — the ECHAM/ICON scheme: deep, shallow and mid-level convection, convective
  momentum transport, and downdrafts (``updraft.py``, ``downdraft.py``,
  ``flux_tendencies.py``). Organized (Nordeng) entrainment and detrainment are
  the **metre-based fractional rates** of ``mo_cuascent.f90`` — organized
  entrainment carries the ``zbuoyz·0.5/(1+∫buoyancy) + zdrodz`` density-lapse
  term and organized detrainment is the ``tan``-profile in height that scales
  as 1/(cloud depth) — each clamped to ECHAM's hard cap ``centrmax = 3.0e-4
  m⁻¹``, and per-layer detrained mass is capped at 0.75 of the plume
  (``cu_asc`` line 500) so the detrained-condensate ledger can never exceed the
  plume mass. Momentum transport is the ``cududv`` deviation-flux divergence
  with SEPARATE updraft and downdraft fluxes, each carrying its own prognostic
  plume wind (mass-weighted entrainment of the environment), the upstream
  ``jk−1`` environment offset, the sub-cloud pressure-ratio taper, and the
  explicit surface-layer closure. The ``cudtdq`` ledger keys its
  condensate-flux latent heat to the phase (``alhs`` below the melting point,
  ``alhc`` above) and writes a tendency to the surface layer (ECHAM's
  ``jk == klev`` branch). Moisture convergence *classifies* deep vs shallow
  (ECHAM's ``mo_cumastr.f90`` test), while the cloud-base *closure* routes
  independently of that type: any active surface plume takes the
  moisture-budget flux ``E/(q_u−q_e)`` where it is valid (ECHAM's ``zlo1``
  test — near-saturated cloud bases and negligible evaporation fail it) and
  ECHAM's constant fallback ``zmfub = 0.01 kg m⁻² s⁻¹``
  (``mo_cumastr.f90:567``) otherwise, so surface evaporation, not CAPE, sets
  the steady-state flux of a healthy plume; the deep amplitude is then set by
  the Nordeng ``zmfub1 = zcape·zmfub/(zheat·tau)`` rescale, which is where the
  CAPE-consumption timescale ``tau`` lives. Mid-level (``cubasmc``) plumes take
  neither: their base flux is the resolved ascent that triggered them. The saturation
  adjustment ``cuadjtq`` is a faithful linearised-Newton port of ``mo_cuadjust.f90``
  (``adjustment.py``) with the three ``kcall`` modes. The precipitation budget
  (rain/snow partition, snow melt, sub-cloud Kessler evaporation, proportional
  depletion) transcribes ECHAM ``cuflx`` (``flux_tendencies.py``,
  ``mo_cufluxdts.f90``). The fractional precipitation cover the sub-cloud
  evaporation acts over follows ECHAM's submodel dependence
  (``mo_cufluxdts.f90:414-420``): plain ECHAM uses the constant
  ``zcucov = 0.05``, but with the JAM aerosol chain composed
  (``aerosol_module='jam'``, jcm's analogue of ECHAM's ``lham``) it uses the
  **updraft area** ``pmfu/(zwu·ρ_u)`` — ``ρ_u = p/(R_d·T_u)`` the updraft
  density, ``zwu = 2`` m/s the assumed in-cloud updraft speed — evaluated
  over the plume profile and ECHAM's sub-cloud taper. This is the identical
  footprint the JAM convective wet deposition uses for below-cloud washout
  (``updraft_area_cover`` is shared between the two), so the rain the
  aerosol scavenging sees and the rain the evaporation depletes agree by
  construction.
- **SPEEDY convection** (``jcm/physics/convection/speedy_convection.py::diagnose_convection``)
  — SPEEDY's simplified Tiedtke (1993) mass-flux scheme with a
  conditional-instability trigger on saturation moist static energy.
- **Betts-Miller**
  (``jcm/physics/convection/betts_miller/betts_miller_terms.py::BettsMillerConvection``;
  core ``betts_miller.py``) — a faithful port of Isca's ``betts_miller.f90``
  (Frierson 2007 Simplified Betts-Miller), relaxing T and q toward a moist-adiabatic
  reference at RH ``rhbm`` over ``tau_bm``, with ``do_shallower`` / ``do_changeqref``
  siblings. Written broadcasting-native (vertical on axis 0).

**Tiedtke's** activation is a **smooth sigmoid trigger** on CAPE rather than a hard
``cape > threshold`` branch, so tau / entrainment / threshold parameters carry
nonzero gradients near the trigger. Saturation thermodynamics are shared
(``jcm/physics/convection/saturation.py``, Tetens).

**What ECHAM/CAM does.** ECHAM6-HAM2.3 uses the **Tiedtke (1989) bulk mass-flux
scheme with Nordeng (1994) CAPE closure** (``mo_cumastr.f90`` master driver,
``mo_cuasc.f90`` / ``mo_cuascn.f90`` updraft ascent, ``mo_cudlfs`` / ``mo_cuddraf``
downdrafts, ``mo_cuadjust.f90`` saturation and per-level adjustment,
``mo_cufluxdts.f90`` fluxes). References: Tiedtke, M. (1989), *Mon. Wea. Rev.* 117,
1779-1800; Nordeng, T.E. (1994), ECMWF Tech. Memo. 206. Betts-Miller's reference
is Betts & Miller (1986) as simplified by Frierson, D.M.W. (2007), *J. Atmos. Sci.*
64, 1959-1976 (Isca ``betts_miller.f90``).

**Why we differ.**
- `differentiability` — the hard ``ldcum`` activation and the deep/shallow
  split are replaced by smooth sigmoid weights, so those convective parameters
  are differentiable. **Mid-level selection stays discrete** (a boolean
  ``use_midlev`` that forces full trigger weight, as ECHAM's ``cubasmc``
  conditions ARE the activation), so gradients do not flow across mid-level
  onset.
- `science` / `compute` (stopgap) — the **per-level moist-adjustment limits in
  ``mo_cuadjust.f90`` are not yet ported**. The cloud-base mass-flux **CFL cap**
  ``zmfmax = layer_mass/dt`` bounds the column-integrated flux but not per-level
  latent-heat spikes inside the updraft loop. Until the per-level limits land, an
  explicitly-labelled stopgap caps the convective T-tendency at 5 K/hr
  (``_DTDT_MAX``) and rescales the **whole** ledger homogeneously — T, q,
  qc/qi, precipitation, the mass fluxes with the tracer transport they drive,
  **and the momentum tendencies** ``dudt``/``dvdt``, which share the same mass
  flux — preserving column conservation by linearity, as ECHAM's ``zmfub1``
  amplitude scaling does. This cap is the documented cause of a cap-pinned
  single-layer heating artifact in pathological columns.

**Status & known limitations.**
- The 5 K/hr tendency cap is a **safety net, not physics**; it fires only where
  the parcel-vs-environment balance has gone pathological (healthy tropical deep
  convection is ~1 K/hr). It remains until the ``mo_cuadjust`` per-level limits are
  ported.
- Cloud-base closure falls back to ECHAM's constant ``zmfub = 0.01`` first
  guess when the moisture-budget denominator collapses under a near-saturated
  cloud base or spectral supersaturation ringing; deep columns then take the
  Nordeng CAPE rescale, so ``tau`` still sets their amplitude.
- SPEEDY and Betts-Miller are idealized alternatives; Betts-Miller is
  specific-humidity-formulated (Isca's mixing-ratio form differs at second order).

**Code pointers.**
- ``jcm/physics/convection/tiedtke_nordeng/`` — ``tiedtke_nordeng.py``
  (``TiedtkeConvection``, the CFL cap, the ``moisture_valid`` closure gate
  [ECHAM zlo1], ``_DTDT_MAX``, the unported-mo_cuadjust note),
  ``adjustment.py`` (``cuadjtq``), ``flux_tendencies.py``
  (``convective_precip_fluxes`` [ECHAM cuflx], ``mass_flux_closure_blend``),
  ``updraft.py``, ``downdraft.py``.
- ``jcm/physics/convection/speedy_convection.py`` — ``diagnose_convection``.
- ``jcm/physics/convection/betts_miller/`` — ``betts_miller.py``,
  ``betts_miller_terms.py`` (``BettsMillerConvection``).
- Convective *tracer* transport and in-plume scavenging live with the aerosol
  chain — see {doc}`aerosol`.

**Validation evidence.** ``jcm/physics/convection/tiedtke_nordeng/`` test suite
(``tiedtke_nordeng_test.py``, ``adjustment_test.py``, ``updraft_test.py``,
``downdraft_test.py``, ``deep_shallow_test.py``, ``midlevel_trigger_test.py``,
``rce_integration_test.py``, ``convection_units_test.py``,
``smooth_gradients_test.py``, ``cloud_depth_test.py``);
``betts_miller/betts_miller_test.py``; ``speedy_convection_test.py``.

## Heat capacity of the Tiedtke plume and ledger

**What we do.** Every static-energy and latent-heat conversion the Tiedtke
port shares with ECHAM ``cumastr`` uses the **moist** isobaric heat capacity
``cp = cpd·(1 + vtmpc2·max(q, 0))``
(``jcm/physics/thermodynamics.py::moist_isobaric_heat_capacity``), evaluated
per level from the **step-start** humidity: the cloud-base and mid-level parcel
lifts (``find_cloud_base``, ``find_midlevel_cloud_base``, the ``calculate_updraft``
seed), the updraft and downdraft dry-static-energy mixing
(``calculate_updraft``, ``downdraft_step``), the Nordeng ``zheat`` lapse term,
and the ``cudtdq`` ledger — the DSE deviation fluxes ``cp·(T_plume − T)·M`` and
the conversion of the whole heat ledger to a temperature tendency. The column
enthalpy the ledger deposits is therefore ``Σ cp·dT·Δp/g``. Three sites keep
dry ``cpd`` because the reference does: the ``cuadjtq`` Newton step and the
wet-bulb adjustment (``adjustment.py``, ``saturation.py``), and the ``cuflx``
melting constant, which applies its own ``(1 + vtmpc2·q)`` factor with the
provisional humidity. jcm's own trigger diagnostic ``calculate_cape_cin`` has no
ECHAM counterpart and uses the textbook dry-``cpd`` parcel.

**What ECHAM does.** ``mo_cumastr.f90:229`` builds ``zcpq = cpd·(1 +
vtmpc2·MAX(pqm1, 0))`` from the step-start humidity ``pqm1`` (not the
provisional ``zqp1`` the plume sees) and passes it as ``pcpen``; ``cuini``
averages it to half levels (``pcpcu``). ``cubase``/``cuasc``/``cubasmc``/
``cuddraf`` carry plume heat as ``pcpcu·T + pgeoh``; ``cuflx`` subtracts the
environment's ``pcpcu·ptenh + pgeoh`` (``mo_cufluxdts.f90:198-204``);
``cudtdq`` divides by ``pcpen`` (``zrcpm``, ``mo_cufluxdts.f90:648``); ``zheat``
uses ``zcpcui = 1/zcpcu`` (``mo_cumastr.f90:598/849``). ``cuadjtq`` reads
``L/cp`` from tables built with ``alv/cpd`` (``mo_echam_convect_tables.f90:214``).

**Why we differ.** We do not: the port matches the reference's choice at every
site. Because ``cp`` is the **environment's** at each level, a parcel lifted
through an environment that dries with height gains ``≈ T·vtmpc2·Δq_env``
relative to a dry-``cpd`` lift (the heat content is carried with the lower
level's larger ``cp`` and divided by the upper level's smaller one). That is
the reference's thermodynamics and ECHAM's convective parameters were tuned
with it, so it is kept rather than replaced by the parcel's own heat capacity.

**Status & known limitations.** ECHAM's ``pcpcu`` lives on half levels; jcm's
convection is full-level throughout (#530), so the full-level ``cp`` serves
both.

## Cloud-base trigger and the sub-cloud layer

**What we do.** Tiedtke's cloud base is ECHAM's ``cubase`` ``klab`` walk
(``jcm/physics/convection/tiedtke_nordeng/tiedtke_nordeng.py::find_cloud_base``):
a parcel starts at the lowest level with the environment's temperature and
humidity and is lifted upward conserving dry static energy. At each level a dry
buoyancy test ``zbuo = Tv_u - Tv_e + zlift`` decides whether the walk continues;
the first level at which the parcel condenses is the LCL, where a second test —
the same buoyancy with condensate loading — decides whether a cloud base exists.
``zlift`` is the sub-grid thermal excess of the warmest boundary-layer plumes,
``min(clip(thvsig·cbfac, cminbuoy, cmaxbuoy), 1)`` K, taken from vdiff's
prognostic θ_v variance where one is available and from
``ConvectionParameters.cu_thvsig`` otherwise. A column whose parcel loses its
buoyancy before reaching its own LCL gets **no surface-based convection at all**,
whatever its CAPE: the walk stops, and CAPE never enters. The second,
independent entry is the mid-level ``cubasmc`` trigger
(``find_midlevel_cloud_base``), which starts a plume with no surface connection
where the environment is nearly saturated and resolved ascent is lifting it.

**What ECHAM does.** ``mo_cuinitialize.f90::cubase`` (the ``klab`` walk, ``zlift``
from ``pthvsig``) and ``mo_cuascent.f90::cubasmc`` (the mid-level trigger).
ECHAM evaluates the walk on half levels whose environment temperature is the
**dry-static-energy upper envelope** of the two adjacent full levels
(``mo_cuinitialize.f90::cuini``: ``ptenh = (MAX(s(jk-1), s(jk)) - geoh)/cpm``,
then monotonized upward), which flattens any dry-neutral or dry-unstable layer
before the parcel is compared against it.

**Why we differ.**
- `compute` — the walk runs on **full levels**, since jcm's convection path is
  full-level throughout (the scheme-wide staggering approximation, #530). For a
  stably stratified profile the DSE-envelope half level carries the full level
  above it, so the dry test is equivalent up to one level index; the
  condensation test is evaluated half a layer higher than the reference's.

**Status & known limitations.**
- The trigger is **strict by construction, and this is the reference's
  behaviour, not an approximation of it**: because ``zlift`` is capped at 1 K, a
  sounding whose sub-cloud layer is stably stratified loses more parcel
  buoyancy per level than the excess can cover and never reaches its LCL. The
  walk's moist heat capacity (see below) credits the parcel
  ``≈ T·vtmpc2·Δq_env`` per level where the environment dries with height, so
  a moist 6.5 K/km surface layer can still reach its LCL; a genuinely stable
  (e.g. 4 K/km or inverted) one cannot. Convection in such
  a column is the job of ``cubasmc``, which needs resolved ascent and a nearly
  saturated environment.
- It follows that **any prescribed or idealised column handed to Tiedtke assumes
  a well-mixed (dry-adiabatic) sub-cloud layer**, the state a prognostic run's
  vertical diffusion maintains and real tropical soundings have. The
  release-validation single-column state
  (``jcm/rce.py::rce_initial_state``, ``jcm/rce.py::jam_scavenging_column``)
  supplies one explicitly via ``mixed_layer_top_m``, since a prescribed column
  re-imposed every step never lets vdiff build it. Failure of this assumption is
  silent: the plume simply never exists, while turbulent mixing keeps the
  profiles plausible. See {doc}`../design/convective_trigger_soundings`.
- jcm fixes **one cloud base per column per step**; ECHAM can re-seed above a
  ``cubase`` plume that dies partway up (#700).
