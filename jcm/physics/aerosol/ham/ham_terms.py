"""``ham_aerosol_physics()`` factory — the ordered HAM harness term list.

Returns the HAMMOZ-style process chain (emissions → microphysics core →
activation → sedimentation → dry deposition → wet deposition) as a list of
``PhysicsTerm``s, ready to splice into ``echam_physics``. The microphysics
core is the swap point: pass ``"placeholder"`` (default κ-Köhler equilibrium)
or any ``ModalMicrophysicsTerm`` instance (e.g. a future MAM4-JAX wrapper,
#490). Every harness term is handed the core's population so they all agree
on mode/species layout.
"""

from __future__ import annotations

from jcm.physics.aerosol.ham.activation.arg_term import (
    ArgActivation,
    ArgParameters,
)
from jcm.physics.aerosol.ham.drydep.drydep_term import (
    DryDepParameters,
    HamDryDeposition,
)
from jcm.physics.aerosol.ham.emissions.emissions_term import (
    EmissionParameters,
    HamEmissions,
)
from jcm.physics.aerosol.ham.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.ham.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.ham.sedimentation.sedi_term import (
    HamSedimentation,
    SedParameters,
)
from jcm.physics.aerosol.ham.wetdep.wetdep_term import (
    HamWetDeposition,
    WetDepParameters,
)
from jcm.physics.physics_term import PhysicsTerm

_MICROPHYSICS = {
    "placeholder": PlaceholderMicrophysics,
}


def _resolve_microphysics(
    microphysics: ModalMicrophysicsTerm | str,
) -> ModalMicrophysicsTerm:
    if isinstance(microphysics, ModalMicrophysicsTerm):
        return microphysics
    try:
        return _MICROPHYSICS[microphysics]()
    except KeyError:
        raise ValueError(
            f"Unknown ham microphysics {microphysics!r}. "
            f"Choose one of {sorted(_MICROPHYSICS)} or pass a "
            "ModalMicrophysicsTerm instance."
        ) from None


def ham_aerosol_physics(
    *,
    microphysics: ModalMicrophysicsTerm | str = "placeholder",
    arg_variant: str = "arg2000",
    emissions: EmissionParameters | None = None,
    activation: ArgParameters | None = None,
    sedimentation: SedParameters | None = None,
    drydep: DryDepParameters | None = None,
    wetdep: WetDepParameters | None = None,
) -> list[PhysicsTerm]:
    """Build the ordered HAM harness term list.

    Args:
        microphysics: the swappable core — ``"placeholder"`` or a
            ``ModalMicrophysicsTerm`` instance.
        arg_variant: ``"arg2000"`` (default) or ``"ghosh2025"`` activation.
        emissions/activation/sedimentation/drydep/wetdep: optional per-process
            ``Parameters`` overrides (each ``None`` resolves to its default).

    Returns:
        ``[HamEmissions, <core>, ArgActivation, HamSedimentation,
        HamDryDeposition, HamWetDeposition]``.

    """
    core = _resolve_microphysics(microphysics)
    spec = core.spec
    return [
        HamEmissions(params=emissions, spec=spec),
        core,
        ArgActivation(params=activation, spec=spec, variant=arg_variant),
        HamSedimentation(params=sedimentation, spec=spec),
        HamDryDeposition(params=drydep, spec=spec),
        HamWetDeposition(params=wetdep, spec=spec),
    ]
