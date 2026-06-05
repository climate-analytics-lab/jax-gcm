"""JAM aerosol harness.

A HAMMOZ-style, microphysics-agnostic process harness (emissions, dry/wet
deposition, sedimentation, ARG activation) wrapping an interchangeable modal
microphysics core. See issue #461 and ``.claude/aerosol_harness_plan.md``.

The ``jam_aerosol_physics`` factory and the per-process terms are added in
later phases; Phase 0 exposes the population contract and the placeholder
core.
"""

from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.jam_terms import jam_aerosol_physics
from jcm.physics.aerosol.jam.microphysics import (
    MAM4_SPEC,
    ModalMicrophysicsTerm,
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.population import (
    AerosolMode,
    AerosolSpecies,
    ModalAerosolSpec,
)
from jcm.physics.aerosol.jam.tracer_layout import (
    mass_name,
    number_name,
    tracer_specs,
)

__all__ = [
    "AerosolMode",
    "AerosolSpecies",
    "ModalAerosolSpec",
    "JamAerosolState",
    "jam_aerosol_physics",
    "ModalMicrophysicsTerm",
    "PlaceholderMicrophysics",
    "MAM4_SPEC",
    "mass_name",
    "number_name",
    "tracer_specs",
]
