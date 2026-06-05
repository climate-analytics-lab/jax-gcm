"""Tracer-name conventions for the JAM aerosol harness.

Aerosol mass and number live as ordinary entries in ``state.tracers`` so the
dynamical core transports them and the existing diagnostics infrastructure
works unchanged. MAM4 carries both an *interstitial* population and a
*cloud-borne* mirror, so every quantity has an interstitial and a
cloud-borne key.

Flat key conventions (the ``state.tracers`` API is flat-keyed):

    mass, interstitial   ``m_<species>_<mode_short>``    [kg/kg]
    mass, cloud-borne    ``mc_<species>_<mode_short>``   [kg/kg]
    number, interstitial ``n_<mode_short>``              [kg^-1]
    number, cloud-borne  ``nc_<mode_short>``             [kg^-1]

Number tracers use ``nondimensionalize=False`` (they are #/kg, not a
mixing ratio), matching the 2M scheme's ``qnc``/``qni`` convention.
"""

from __future__ import annotations

from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.physics_term import TracerSpec


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
    each doubled for the cloud-borne mirror.
    """
    out: list[TracerSpec] = []
    for mode in spec.modes:
        for cb in (False, True):
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


def mass_names_for_mode(
    spec: ModalAerosolSpec, mode_short: str, *, cloud_borne: bool = False
) -> tuple[str, ...]:
    """Mass tracer keys for one mode, in the population's species order."""
    mode = spec.mode(mode_short)
    return tuple(
        mass_name(sp, mode.short, cloud_borne=cloud_borne) for sp in mode.species
    )
