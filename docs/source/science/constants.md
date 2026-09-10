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
Overrides do **not** yet reach everything: seven modules — the JAM
activation / sedimentation / dry-deposition / dust / ice-nucleation chain, the
dinosaur dycore wrapper and the WMO-tropopause diagnostic — bind values with
``from jcm.constants import <name>`` at import time and would silently keep
Earth values after an override (#772). Consumers following the contract
(``import jcm.constants as c``, read ``c.<name>`` when traced or at
construction) are unaffected — so until #772 lands, a ``set_constants`` run is
consistent everywhere except a composition using those seven.

**Code pointers.**
- ``jcm/constants.py`` — ``PhysicalConstants``, the ``physical_constants``
  singleton, ``set_constants``, the module-level ``__getattr__``, and the derived
  ``@property`` definitions.
- Consumed at dycore construction:
  ``jcm/dycore/dinosaur/dycore.py`` (``physics_specs_from_constants``).

**Validation evidence.** The override / derived-quantity behaviour is exercised
through dycore construction (``jcm/dycore/dinosaur/dycore_test.py``) and the
SPEEDY-specific ``jcm/physics/speedy/physical_constants_test.py``.
