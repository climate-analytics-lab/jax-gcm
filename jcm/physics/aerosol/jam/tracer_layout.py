"""Tracer-name conventions for the JAM aerosol harness.

Aerosol mass and number live as ordinary entries in ``state.tracers`` so the
dynamical core transports them and the existing diagnostics infrastructure
works unchanged. The ``<class>`` token below is a per-class short name — a
mode short for a modal scheme, a bin label for a sectional one — so the same
conventions serve any population family. The *cloud-borne* mirror keys are
emitted for cores that prognose an interstitial + cloud-borne population
(e.g. MAM4); a core without that distinction simply declares only the
interstitial keys.

Flat key conventions (the ``state.tracers`` API is flat-keyed):

    mass, interstitial   ``m_<species>_<class>``    [kg/kg]
    mass, cloud-borne    ``mc_<species>_<class>``   [kg/kg]
    number, interstitial ``n_<class>``              [kg^-1]
    number, cloud-borne  ``nc_<class>``             [kg^-1]

Number tracers use ``nondimensionalize=False`` (they are #/kg, not a
mixing ratio), matching the 2M scheme's ``qnc``/``qni`` convention.
"""

from __future__ import annotations

from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.physics_term import TracerSpec


def gas_name(species: str) -> str:
    """Tracer key for a gas-phase precursor mixing ratio (e.g. ``g_so2``)."""
    return f"g_{species}"


def gas_tracer_specs(species: tuple[str, ...]) -> tuple[TracerSpec, ...]:
    """TracerSpecs for the given gas-phase precursor tokens.

    Gas precursors are ordinary mass mixing ratios [kg/kg] (transported and
    nondimensionalised like aerosol mass), one per token.
    """
    return tuple(TracerSpec(gas_name(s), units="kg/kg") for s in species)


def mass_name(species: str, mode_short: str, *, cloud_borne: bool = False) -> str:
    """Tracer key for a (species, mode) mass mixing ratio."""
    prefix = "mc" if cloud_borne else "m"
    return f"{prefix}_{species}_{mode_short}"


def number_name(mode_short: str, *, cloud_borne: bool = False) -> str:
    """Tracer key for a mode number mixing ratio."""
    prefix = "nc" if cloud_borne else "n"
    return f"{prefix}_{mode_short}"


def tracer_specs(spec: ModalAerosolSpec) -> tuple[TracerSpec, ...]:
    """All ``TracerSpec``s for a population (interstitial + cloud-borne).

    Returns one mass spec per (mode, species) and one number spec per mode,
    each doubled for the cloud-borne mirror when the population prognoses
    one (``spec.cloud_borne``); an implicit-in-droplet population declares
    only the interstitial keys.
    """
    phases = (False, True) if spec.cloud_borne else (False,)
    out: list[TracerSpec] = []
    for mode in spec.modes:
        for cb in phases:
            out.append(
                TracerSpec(
                    number_name(mode.short, cloud_borne=cb),
                    units="kg^-1",
                    nondimensionalize=False,
                )
            )
            for sp in mode.species:
                out.append(
                    TracerSpec(
                        mass_name(sp, mode.short, cloud_borne=cb),
                        units="kg/kg",
                    )
                )
    return tuple(out)
