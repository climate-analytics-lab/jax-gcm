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
from jcm.physics.aerosol.jam.chemistry.aqueous import (
    AqueousSulfur,
    AqueousSulfurParameters,
)
from jcm.physics.aerosol.jam.chemistry.oxidants import (
    OxidantParameters,
    PrescribedOxidants,
)
from jcm.physics.aerosol.jam.chemistry.sulfur_gas import (
    SulfurGasChemistry,
    SulfurGasParameters,
)
from jcm.physics.aerosol.jam.drydep.drydep_term import (
    DryDepParameters,
    SlinnDryDeposition,
)
from jcm.physics.aerosol.jam.emissions.dms import DmsEmissions, DmsParameters
from jcm.physics.aerosol.jam.emissions.dust import DustEmissions, DustParameters
from jcm.physics.aerosol.jam.emissions.seasalt import (
    SeaSaltEmissions,
    SeaSaltParameters,
)
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.optics.optics_term import JamOpticsTerm
from jcm.physics.aerosol.jam.sedimentation.sedi_term import (
    StokesSedimentation,
    SedParameters,
)
from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
)
from jcm.physics.physics_term import PhysicsTerm

def _load_mam4_jax() -> type[ModalMicrophysicsTerm]:
    """Import the MAM4-JAX core lazily (optional GPL-3.0 dependency)."""
    from jcm.physics.aerosol.jam.microphysics.mam4_jax import (
        Mam4JaxMicrophysics,
    )

    return Mam4JaxMicrophysics


# Core resolvers. ``placeholder`` is built-in; ``mam4_jax`` is loaded lazily so
# the optional GPL-3.0 ``mam4-jax`` dependency is only imported when selected.
_MICROPHYSICS = {
    "placeholder": lambda: PlaceholderMicrophysics(),
    "mam4_jax": lambda: _load_mam4_jax()(),
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
    optics: bool = True,
    seasalt: SeaSaltParameters | None = None,
    dms: DmsParameters | None = None,
    dust: DustParameters | None = None,
    oxidants: OxidantParameters | None = None,
    sulfur_gas: SulfurGasParameters | None = None,
    aqueous: AqueousSulfurParameters | None = None,
    aqueous_scheme: str = "full",
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
        seasalt/dms/dust: optional ``Parameters`` overrides for the natural
            emission schemes (Gong sea salt, Nightingale DMS, Tegen dust).
        oxidants/sulfur_gas/aqueous: optional ``Parameters`` for the
            prescribed-oxidant + gas-phase + aqueous sulfur chemistry (#496).
        aqueous_scheme: ``"full"`` (default, HAM ``ham_wet_chemistry`` port) or
            ``"simple"`` (H2O2-limited stoichiometric oxidation).
        activation/sedimentation/drydep/wetdep: optional per-process
            ``Parameters`` overrides (each ``None`` resolves to its default).

    Returns:
        The ordered term list: natural emissions, prescribed oxidants and
        gas-phase sulfur chemistry, the microphysics core (optionally followed
        by online optics), activation, sedimentation, dry deposition, in-cloud
        aqueous sulfur chemistry, and wet deposition.

    """
    core = _resolve_microphysics(microphysics)
    spec = core.spec
    pre_core = [
        SeaSaltEmissions(params=seasalt, spec=spec),
        DmsEmissions(params=dms, spec=spec),
        DustEmissions(params=dust, spec=spec),
        # Sulfur chemistry: oxidants → gas-phase DMS/SO2 oxidation, producing
        # the H2SO4/SOAG gas the core condenses + nucleates this same step.
        PrescribedOxidants(params=oxidants),
        SulfurGasChemistry(params=sulfur_gas),
    ]
    # Online aerosol direct radiative effect (#495): placed right after the core
    # (needs ``_jam_state``); overwrites the MACv2-SP ``aerosol`` optics.
    optics_terms = [JamOpticsTerm(spec=spec)] if optics else []
    post_core = [
        ArgActivation(params=activation, spec=spec, variant=arg_variant),
        StokesSedimentation(params=sedimentation, spec=spec),
        SlinnDryDeposition(params=drydep, spec=spec),
        # In-cloud aqueous SO2 oxidation → cloud-borne sulfate; runs in the
        # post-cloud block (needs current clouds), just before wet scavenging.
        AqueousSulfur(params=aqueous, spec=spec, scheme=aqueous_scheme),
        WetScavenging(params=wetdep, spec=spec),
    ]
    terms = [*pre_core, core, *optics_terms, *post_core]
    return terms
