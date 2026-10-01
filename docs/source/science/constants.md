# Physical constants

**What we do.** ``jcm/constants.py`` is the single source of truth for the
general physical constants shared across all physics packages.
``PhysicalConstants`` is a ``NamedTuple`` in which every *independent* quantity is
a field exactly once under one canonical name (``cpd``, ``rd``… no aliases), and
every algebraically derived quantity is a ``@property`` that recomputes on access
(``rd = akap·cpd``, ``cvd = cpd − rd``, ``rgrav = 1/grav``, ``alhf = alhs − alhc``,
the ``vtmpc*`` moisture coefficients, molar masses), so they can never drift out
of sync with the bases even after an override. The module owns one live singleton;
``set_constants(...)`` rebinds it (a whole instance, or base-field kwargs), and a
module-level ``__getattr__`` forwards bare-name access (``jcm.constants.grav``) to
the singleton so attribute-access consumers honour overrides. The dynamical core
reads the live singleton at construction.

**What ECHAM/CAM does.** ECHAM6 keeps physical constants as module-level
``PARAMETER``s in ``mo_physical_constants.f90`` (compile-time fixed; derived
constants such as ``alf = als − alv`` computed once at init). CAM uses
``physconst`` / ``shr_const_mod``.

**Why we differ.**
- `differentiability` / design — constants live in one mutable process-global
  singleton read by attribute access, rather than a set of frozen compile-time
  ``PARAMETER``s. This lets ``set_constants(...)`` retune a base value (a
  different planet, a sensitivity study, gradient-based calibration) before model
  construction, with the derived-quantity properties recomputing consistently,
  for the dynamics and for every consumer that follows the contract's
  ``c.<name>`` attribute access. The documented trap: ``from jcm.constants
  import grav`` binds the value at import time and will *not* track overrides —
  new code must use attribute access.

**Status & known limitations.** Only *base* fields may be overridden by keyword;
passing a derived quantity to ``set_constants`` raises. ``alhf`` is derived (not
an independent base) so the fusion enthalpy always equals ``alhs − alhc``.
Every consumer in the package now reads constants **when the value is used**
— inside the function, at construction, or at trace time — so an override
reaches all of them: the dinosaur dycore wrapper takes the live singleton at
construction, and the JAM activation / sedimentation / dry-deposition /
aqueous-chemistry chain, the 2M heterogeneous freezing, the TTE-TKE closure, the emissions
preparation step and the WMO-tropopause diagnostic all read theirs per call.

The contract is about *timing*, not merely about the import form, and the
guard in ``jcm/constants_test.py`` enforces it that way: it parses every
non-test module and rejects three distinct captures, each of which silently
froze Earth values after an override.

1. ``from jcm.constants import grav`` — binds the float at import.
2. ``from jcm.constants import physical_constants`` — binds the singleton
   *object*, equally stale because ``set_constants`` rebinds the module global
   rather than mutating it (``PhysicalConstants`` is a ``NamedTuple``).
3. Evaluating ``c.<name>`` at import time *despite* using the approved module
   alias — a derived module constant (``_MW_AIR = c.m_air * 1000.0``), a
   default argument (``def f(..., gravity=c.grav)``, evaluated once when the
   ``def`` executes), or a class-body attribute. This is the easiest form to
   miss: the TTE-TKE closure read ``c.cpd`` live and took gravity from a
   frozen default two lines earlier, in the same expression.

Only ``PhysicalConstants`` itself may be imported by name — it is a type and
binds no value; the dycore uses it as an annotation.

The guard is structural, so it has one blind spot worth naming: a module-level
*call* that reads constants inside itself (``_TABLE = _build_table()``) is not
detected. The package's one such value, ``dycore.PHYSICS_SPECS``, is built from
``PhysicalConstants.default()`` rather than the live singleton and is
referenced only by tests, so it is a fixed default by construction rather than
a stale override — the dycore's own specs are built at construction from the
live values.

Two boundaries remain, both by design rather than oversight. ``set_constants``
must be called **before** the model is built: a constant read inside a jitted
term is baked in when that term is traced, so an override afterwards does not
propagate until recompilation. And constants internal to ``mam4-jax`` belong to
that package — JAM's calls into it use its values, which ``set_constants`` does
not reach.

**Code pointers.**
- ``jcm/constants.py`` — ``PhysicalConstants``, the ``physical_constants``
  singleton, ``set_constants``, the module-level ``__getattr__``, and the derived
  ``@property`` definitions.
- Consumed at dycore construction:
  ``jcm/dycore/dinosaur/dycore.py`` (``physics_specs_from_constants``).

**Validation evidence.** The override / derived-quantity behaviour is exercised
through dycore construction (``jcm/dycore/dinosaur/dycore_test.py``) and the
SPEEDY-specific ``jcm/physics/speedy/physical_constants_test.py``.
``jcm/constants_test.py`` adds the structural import guard plus per-module
behavioural checks: with gravity overridden, the tropopause geopotential
height, the ARG maximum supersaturation, the Stokes settling velocity, the
quasi-laminar deposition resistance and the 2M immersion freezing (through its
cooling rate ``fact_tke·√TKE·g/cpd``) all move, and each restores the original
constants afterwards.

## Saturation vapour pressure of the ECHAM physics

**What we do.** Every ECHAM scheme in jcm takes its saturation vapour
pressure, saturation specific humidity and their temperature derivatives from
one module, ``jcm/physics/thermodynamics.py``, which evaluates the five-term
fit of Sonntag (1990, *Z. Meteorol.* 70, 340-344),
``ln es = a1/T + a2 + a3·0.01·T + a4·1e-5·T² + a5·ln T``, with separate
coefficients over liquid water and over ice. The phase is chosen per scheme as
ECHAM chooses it:

- ``es_ua`` — ECHAM's ``ua`` table: ice at and below the melting point, water
  above. Tiedtke-Nordeng convection throughout (``cuini``, ``cubase``,
  ``cuasc``, ``cudlfs``/``cuddraf`` and the ``cuadjtq`` adjustment, whose
  latent heat switches at the same point), the TTE-TKE vertical diffusion,
  the ocean, land and sea-ice surface saturation, and every "ice" saturation
  of the 2M scheme.
- ``es_water`` — ECHAM's ``uaw`` table: water at all temperatures. The 1M
  rain evaporation and the 2M scheme's water saturation (rain evaporation,
  Bergeron-Findeisen, the water branch of its condensation).
- ``es_ice`` or ``es_water`` per cell by the ``lo2`` switch — the cloud
  cover's saturation (ice where ``T < cthomi`` or where ``T < tmelt`` with
  cloud ice above ``csecfrl``) and the 2M condensation (ice where
  ``T < cthomi`` or where ``T < tmelt`` and the updraft is below the
  Korolev-Mazin threshold).

The saturation specific humidity is ECHAM's ``x = MIN(es·rd/rv/p, 0.5)``,
``qs = x/(1 − vtmpc1·x)``, and its slope the one ECHAM's Newton adjustments
use, ``dqs/dT = zcor²·d(es·rd/rv)/dT / p`` (``qsat_from_es``,
``dqsat_dT_from_es``).

**What ECHAM/CAM does.** ECHAM6.3-HAM2.3 tabulates the same Sonntag fit in
``mo_echam_convect_tables.f90::init_convect_tables`` — the ``ua`` table with
the ice fit where ``T ≤ tmelt``, the ``uaw`` table with the water fit — and
reads it through cubic Hermite splines on 0.025 K knots
(``lookup_ua_spline``, ``lookup_uaw_spline``, ``lookup_ua_eor_uaw_spline``
with the ``lo2`` phase test in ``prepare_ua_index_spline``); the 2M scheme
reads the 0.001 K tables ``tlucua``/``tlucuaw`` at the nearest knot
(``mo_cloud_micro_2m.f90``). The Tetens constants ``c1es``, ``c3les``,
``c4les``, ``c3ies``, ``c4ies`` in ``mo_physical_constants.f90`` serve only
the 2 m dew-point inversion of the land tile's post-processing
(``mo_surface_land.f90::postproc_land``). CAM uses Goff-Gratch (1946)
through ``wv_sat_methods``.

**Why we differ.**
- `compute` / `differentiability` — jcm evaluates the fit instead of
  interpolating a table: no table memory or gather, and a derivative that is
  the fit's own. Against ECHAM's compiled tables the fit differs by the
  splines' interpolation error, at most 4e-12 in ``es`` and 1.8e-9 in
  ``d ln es/dT`` between 150 and 330 K
  (``jcm/data/test/echam_saturation_tables/``), so the two are the same
  formulation.
- `differentiability` — the ``ua`` table switches phase at 273.15 K while the
  two fits meet at the triple point, 273.16 K, so ``es`` steps by −9.7e-5 of
  its value (0.059 Pa, what a 1.3 mK warming changes) and ``d ln es/dT`` by
  +13 % across the switch. Automatic differentiation returns each side's
  analytic slope, which is the slope ECHAM's derivative tables hold; the step
  is too small to be worth a surrogate gradient, so there is none.

**Status & known limitations.** jcm's constants are its own unified set, not
ECHAM's (``rv = 461.0`` against ECHAM's 461.51, ``rd = akap·cpd``), so
``rd/rv`` is 0.62265 where ECHAM's is 0.62196. The saturation formula is
unaffected; the ``qs`` built from it follows the constants. The 1M scheme's
saturation adjustment blends the two fits linearly between 238.15 K and
``tmelt`` instead of switching with ``lo2`` (#940). The idealised schemes
keep their own references: Betts-Miller and JAM's MAM4 humidity and ice
nucleation use the Tetens form of ``jcm/physics/convection/saturation.py``,
JAM's placeholder microphysics a WMO Magnus fit, SPEEDY its own
``speedy_humidity.get_qsat``, the RCE testbed's fixed-RH closure its own
Tetens blend, and the public ``relative_humidity`` diagnostic Bolton (1980)
over water. JAM's ARG activation uses the same Magnus fit where ECHAM-HAM's
activation reads the 2M scheme's Sonntag ``zesw_2d``, a deviation tracked in
#932. The ECHAM surface's 2 m humidity diagnostic
(``surface/echam/turbulent_fluxes.py::compute_surface_humidity``), which no
tendency reads, keeps its Clausius-Clapeyron form.

**Code pointers.**
- ``jcm/physics/thermodynamics.py`` — ``es_water``, ``es_ice``, ``es_ua``,
  ``ua_ice_phase``, the ``dlnes_dT_*`` slopes, ``qsat_from_es``,
  ``dqsat_dT_from_es`` and the phase-selected
  ``saturation_specific_humidity(_and_derivative)``.
- ``jcm/physics/convection/tiedtke_nordeng/cuadjtq.py`` —
  ``saturation_mixing_ratio``, ``cuadjtq``, ``cuadjtq_newton``,
  ``cuadjtq_newton_evap`` (the convection's saturation and its adjustment).
- ``jcm/physics/clouds/sundqvist.py::_qs_cover`` — the cover's ``lo2``
  saturation.

**Validation evidence.** ``jcm/physics/thermodynamics_test.py`` compares the
functions with ECHAM's own compiled tables in float64 (rtol 1e-11 in ``es``,
4e-9 in the slope) and float32 (2e-5, 2e-6), pins the phase rule, the jump at
the melting point, ECHAM's ``qs`` form and its slope.
``jcm/physics/convection/tiedtke_nordeng/cuadjtq_test.py`` compares the
convection's saturation adjustment with ECHAM's compiled ``cuadjtq`` in all
three ``kcall`` modes, float64 and float32
(``jcm/data/test/echam_cuadjtq_reference/``).
