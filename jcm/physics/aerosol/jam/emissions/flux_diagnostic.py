"""Shared emission-flux diagnostic for the JAM emission terms.

AeroCom (MMPPE) asks for per-species emission fluxes ``emi_<species>``
[kg m-2 s-1]. Every emission term already computes them — as the tracer
tendencies it returns — so rather than re-deriving each term's flux
formula, this helper column-integrates the term's own mass tendencies.
That way the diagnostic cannot drift from the mass the term actually adds,
whatever scheme it uses internally.

Fluxes ACCUMULATE across terms: several terms emit the same species (SO2
from anthropogenic sources and from DMS oxidation, sea salt from the
interactive scheme and possibly from prescribed files), and the AeroCom
request is the total. Each term adds its own contribution.

The emitted key set is fixed (``_EMITTED_SPECIES``) rather than derived
from whichever tracers a term happens to carry: the diagnostics dict is
part of the scan carry, so a key set that varies between the initial carry
probe and the real step changes the pytree and the scan rejects it.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

# Species published as emission fluxes. Fixed so the carry pytree is stable;
# a species no term emits simply stays zero.
EMITTED_SPECIES: tuple[str, ...] = (
    "so2", "so4", "bc", "oc", "poa", "soa", "ss", "du", "moa", "dms",
)


def _species_of(tracer_name: str) -> str | None:
    """Species a tracer name belongs to, or None if it is not aerosol mass.

    Mass tracers are ``m_<species>_<mode>`` (interstitial) and
    ``mc_<species>_<mode>`` (cloud-borne); gases are ``g_<species>``.
    Number tracers (``n_<mode>``) carry no mass and are skipped.
    """
    for prefix in ("mc_", "m_"):
        if tracer_name.startswith(prefix):
            rest = tracer_name[len(prefix):]
            species = rest.rsplit("_", 1)[0]
            return species if species in EMITTED_SPECIES else None
    if tracer_name.startswith("g_"):
        species = tracer_name[2:]
        return species if species in EMITTED_SPECIES else None
    return None


def accumulate_emission_fluxes(
    diagnostics: dict,
    tracer_tendencies: dict,
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
) -> dict:
    """Add this term's per-species emission fluxes to ``diagnostics``.

    ``tracer_tendencies`` are the term's own mixing-ratio tendencies
    [kg/kg/s], ``(nlev, *horiz)``; integrating them over the layer mass
    ``rho * dz`` gives the mass added per unit area per second, which is
    the emission flux the protocol asks for. (``rho * dz`` equals ``dp/g``
    hydrostatically; it is used here because every emission term already
    reads both fields, so the diagnostic adds no new requirement.) Terms
    that inject above the surface — elevated industrial, biomass burning —
    are handled without special cases, because the integral spans the
    whole column.

    Returns a new diagnostics dict; the caller must use the result.
    """
    dm = air_density * layer_thickness
    horiz = dm.shape[1:]
    dtype = dm.dtype

    totals: dict[str, jnp.ndarray] = {}
    for name, tend in tracer_tendencies.items():
        species = _species_of(name)
        if species is None or jnp.ndim(tend) < 1:
            continue
        contrib = jnp.sum(tend * dm, axis=0)
        totals[species] = totals.get(species, 0.0) + contrib

    out = dict(diagnostics)
    for species in EMITTED_SPECIES:
        key = f"emi_{species}"
        prev = out.get(key)
        if prev is None:
            prev = jnp.zeros(horiz, dtype=dtype)
        out[key] = prev + totals.get(species, jnp.zeros(horiz, dtype=dtype))
    return out


def emission_flux_keys() -> tuple[str, ...]:
    """Return the diagnostics keys :func:`accumulate_emission_fluxes` publishes."""
    return tuple(f"emi_{s}" for s in EMITTED_SPECIES)


class ResetEmissionFluxes(PhysicsTerm):
    """Zero the ``emi_*`` accumulators at the start of each physics step.

    ``accumulate_emission_fluxes`` adds each term's contribution to what is
    already in the diagnostics dict, because several terms emit the same
    species. That dict is threaded back in from the PREVIOUS step as
    ``prev_physics_data``, so without this the accumulation runs across
    timesteps as well as across terms and ``emi_*`` grows without bound with
    run length — snapshots inflate and any time-average is meaningless.

    Placed at the head of the emissions chain by ``jam_aerosol_physics`` so
    the ordering is structural rather than a convention each term has to
    honour. Writing zeros costs nothing and keeps the key set static.
    """

    name: ClassVar[str] = "reset_emission_fluxes"
    # Same category as the emitters it precedes: it is part of the
    # emissions block, not a separate stage.
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = emission_flux_keys()

    def __call__(self, state, diagnostics, forcing, terrain):
        zero = jnp.zeros(state.temperature.shape[1:],
                         dtype=state.temperature.dtype)
        return (PhysicsTendency.zeros(state.temperature.shape),
                {**diagnostics, **{k: zero for k in emission_flux_keys()}})


