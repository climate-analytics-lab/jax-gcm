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
        activation/sedimentation/drydep/wetdep: optional per-process
            ``Parameters`` overrides (each ``None`` resolves to its default).

    Returns:
        ``[SeaSaltEmissions, DmsEmissions, DustEmissions, <core>,
        ArgActivation, StokesSedimentation, SlinnDryDeposition,
        WetScavenging]``.

    """
    core = _resolve_microphysics(microphysics)
    spec = core.spec
    terms = [
        SeaSaltEmissions(params=seasalt, spec=spec),
        DmsEmissions(params=dms, spec=spec),
        DustEmissions(params=dust, spec=spec),
        core,
        ArgActivation(params=activation, spec=spec, variant=arg_variant),
        StokesSedimentation(params=sedimentation, spec=spec),
        SlinnDryDeposition(params=drydep, spec=spec),
        WetScavenging(params=wetdep, spec=spec),
    ]
    if optics:
        # Online aerosol direct radiative effect (#495): overwrites the
        # MACv2-SP optics in the ``aerosol`` diagnostic. Placed after the core
        # (needs ``_jam_state``); reads the MACv2-SP ``aerosol`` struct that
        # ``echam_physics`` provides upstream.
        terms.insert(4, JamOpticsTerm(spec=spec))
    return terms
