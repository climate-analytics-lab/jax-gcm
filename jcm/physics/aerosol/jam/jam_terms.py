"""``jam_aerosol_physics()`` factory — the ordered JAM harness term list.

Returns the HAMMOZ-style process chain (emissions → microphysics core →
activation → sedimentation → dry deposition → wet deposition) as a list of
``PhysicsTerm``s, ready to splice into ``echam_physics``. The microphysics
core is the swap point: pass ``"placeholder"`` (default κ-Köhler equilibrium)
or any ``ModalMicrophysicsTerm`` instance (e.g. a future MAM4-JAX wrapper,
#490). Every harness term is handed the core's population so they all agree
on mode/species layout.
"""

from __future__ import annotations

from jcm.physics.aerosol.jam.activation.arg_term import (
    ArgActivation,
    ArgParameters,
)
from jcm.physics.aerosol.jam.drydep.drydep_term import (
    DryDepParameters,
    SlinnDryDeposition,
)
from jcm.physics.aerosol.jam.emissions.emissions_term import (
    EmissionParameters,
    JamEmissions,
)
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.sedimentation.sedi_term import (
    StokesSedimentation,
    SedParameters,
)
from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
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
            f"Unknown aer microphysics {microphysics!r}. "
            f"Choose one of {sorted(_MICROPHYSICS)} or pass a "
            "ModalMicrophysicsTerm instance."
        ) from None


def jam_aerosol_physics(
    *,
    microphysics: ModalMicrophysicsTerm | str = "placeholder",
    arg_variant: str = "arg2000",
    emissions: EmissionParameters | None = None,
    activation: ArgParameters | None = None,
    sedimentation: SedParameters | None = None,
    drydep: DryDepParameters | None = None,
    wetdep: WetDepParameters | None = None,
) -> list[PhysicsTerm]:
    """Build the ordered JAM harness term list.

    Args:
        microphysics: the swappable core — ``"placeholder"`` or a
            ``ModalMicrophysicsTerm`` instance.
        arg_variant: ``"arg2000"`` (default) or ``"ghosh2025"`` activation.
        emissions/activation/sedimentation/drydep/wetdep: optional per-process
            ``Parameters`` overrides (each ``None`` resolves to its default).

    Returns:
        ``[JamEmissions, <core>, ArgActivation, StokesSedimentation,
        SlinnDryDeposition, WetScavenging]``.

    """
    core = _resolve_microphysics(microphysics)
    spec = core.spec
    return [
        JamEmissions(params=emissions, spec=spec),
        core,
        ArgActivation(params=activation, spec=spec, variant=arg_variant),
        StokesSedimentation(params=sedimentation, spec=spec),
        SlinnDryDeposition(params=drydep, spec=spec),
        WetScavenging(params=wetdep, spec=spec),
    ]
