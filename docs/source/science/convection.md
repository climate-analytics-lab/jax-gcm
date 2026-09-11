# Convection

**What we do.** Three interchangeable convection schemes:

- **Tiedtke-Nordeng mass-flux**
  (``jcm/physics/convection/tiedtke_nordeng/tiedtke_nordeng.py::TiedtkeConvection``)
  — the ECHAM/ICON scheme: deep, shallow and mid-level convection, convective
  momentum transport, and downdrafts (``updraft.py``, ``downdraft.py``,
  ``flux_tendencies.py``). Moisture convergence *classifies* deep vs shallow
  (ECHAM's ``mo_cumastr.f90`` test), while the cloud-base *closure* routes
  independently of that type: any active surface plume takes the
  moisture-budget flux ``E/(q_u−q_e)`` where it is valid (ECHAM's ``zlo1``
  test — near-saturated cloud bases and negligible evaporation fail it) and
  the bounded CAPE flux otherwise, so surface evaporation, not CAPE, sets the
  steady-state flux of a healthy plume. Mid-level (``cubasmc``) plumes take
  neither: their base flux is the resolved ascent that triggered them. The saturation
  adjustment ``cuadjtq`` is a faithful linearised-Newton port of ``mo_cuadjust.f90``
  (``adjustment.py``) with the three ``kcall`` modes. The precipitation budget
  (rain/snow partition, snow melt, sub-cloud Kessler evaporation, proportional
  depletion) transcribes ECHAM ``cuflx`` (``flux_tendencies.py``,
  ``mo_cufluxdts.f90``).
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
  (``_DTDT_MAX``) and rescales the thermodynamic ledger homogeneously — T, q,
  qc/qi, precipitation, and the mass fluxes with the tracer transport they
  drive — preserving column conservation by linearity, as ECHAM's ``zmfub1``
  amplitude scaling does. The **momentum tendencies are the exception**:
  ``dudt``/``dvdt`` are returned unscaled, so a capped plume's momentum
  transport keeps full amplitude (tracked with the other ledger gaps in #676).
  This cap is the documented cause of a cap-pinned single-layer heating
  artifact in pathological columns.

**Status & known limitations.**
- The 5 K/hr tendency cap is a **safety net, not physics**; it fires only where
  the parcel-vs-environment balance has gone pathological (healthy tropical deep
  convection is ~1 K/hr). It remains until the ``mo_cuadjust`` per-level limits are
  ported.
- Cloud-base closure falls back to the bounded CAPE flux (rather than ECHAM's tiny
  flux) when the moisture-budget denominator collapses under a near-saturated cloud
  base or spectral supersaturation ringing.
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

### Cloud-base trigger and the sub-cloud layer

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
  sounding whose lapse rate runs to the surface loses more parcel buoyancy per
  level than the excess can cover and never reaches its LCL. Convection in such
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
