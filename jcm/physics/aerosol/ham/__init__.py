"""HAM aerosol harness.

A HAMMOZ-style, microphysics-agnostic process harness (emissions, dry/wet
deposition, sedimentation, ARG activation) wrapping an interchangeable modal
microphysics core. See issue #461 and ``.claude/aerosol_harness_plan.md``.

The ``ham_aerosol_physics`` factory and the per-process terms are added in
later phases; Phase 0 exposes the population contract and the placeholder
core.
"""

from jcm.physics.aerosol.ham.ham_state import HamAerosolState
from jcm.physics.aerosol.ham.microphysics import (
    MAM4_JAX_COMMIT,
    MAM4_SPEC,
    ModalMicrophysicsTerm,
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.ham.population import (
    AerosolMode,
    AerosolSpecies,
    ModalAerosolSpec,
)
from jcm.physics.aerosol.ham.tracer_layout import (
    mass_name,
    number_name,
    tracer_specs,
)

__all__ = [
    "AerosolMode",
    "AerosolSpecies",
    "ModalAerosolSpec",
    "HamAerosolState",
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "MAM4_SPEC",
    "MAM4_JAX_COMMIT",
    "mass_name",
    "number_name",
    "tracer_specs",
]
