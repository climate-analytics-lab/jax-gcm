"""Conservation ledger for the modal core (AER-09).

Every microphysical process moves material between reservoirs; none
creates or destroys an element. This module names the reservoir totals
("budgets") a population carries, says which of them each process must
leave untouched, and audits a process against that contract. The driver
can run the audit per process and publish the relative changes, and the
tests assert them — a budget failure then bisects to one process rather
than to the whole step.

Budgets, all per kg of air, summed over the interstitial and (when the
population carries it) cloud-borne phases:

- ``sulfur``: moles of S — sulfate in every mode at the aerosol molar mass
  the processes book it with, plus DMS, SO₂ and H₂SO₄ vapour. Conserved by
  every process (the chemistry that converts between the gases lives
  outside the core).
- ``organic``: kg of carbonaceous material — POA, SOA and marine organics
  in every mode plus the SOAG vapour, which the exchange step books to SOA
  mass one-to-one. Conserved by every process.
- one budget per inert species (``bc``, ``du``, ``ss``, ...): its total
  mass. Conserved by every process.
- ``number``: total particle number. Conserved by exchange, renaming and
  ageing; coagulation only lowers it and new-particle formation only
  raises it, so those two carry a sign rather than an equality.

Not conserved, and deliberately not budgets: the per-mode split of
anything (that is what the processes change), and water, which the core
does not prognose.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

#: Species counted as sulfur [gas name → S atoms per molecule].
_SULFUR_GASES = {"dms": 1, "so2": 1, "h2so4": 1}
_SULFUR_AEROSOL = ("so4",)
#: Species counted as organic material, and the vapour that becomes SOA.
_ORGANIC_AEROSOL = ("poa", "soa", "moa")
_ORGANIC_GASES = ("soag",)

#: What each process must conserve exactly (``None`` means every budget),
#: and the sign of the one it may change.
CONTRACT: dict[str, dict[str, str]] = {
    "gas_aerosol_exchange": {"number": "equal"},
    "rename": {"number": "equal"},
    "nucleation": {"number": "non-decreasing"},
    "coagulation": {"number": "non-increasing"},
    "carbonaceous_ageing": {"number": "equal"},
}


def _mode_species_total(population, spec, species):
    """Mass of one species over every mode and both phases."""
    total = None
    for mode in spec.modes:
        if species not in mode.species:
            continue
        for cloud_borne in (False, True):
            key = mass_name(species, mode.short, cloud_borne=cloud_borne)
            store = population.cloud_borne if cloud_borne else population.masses
            if key in store:
                total = store[key] if total is None else total + store[key]
    return total


def _gas(population, name):
    return population.gases.get(gas_name(name))


def budgets(population, spec) -> dict[str, jnp.ndarray]:
    """Return every budget of ``population`` as ``{name: (nlev, ncols)}``."""
    out: dict[str, jnp.ndarray] = {}
    zeros = jnp.zeros_like(next(iter(population.numbers.values())))

    sulfur = zeros
    for species in _SULFUR_AEROSOL:
        total = _mode_species_total(population, spec, species)
        if total is not None:
            sulfur = sulfur + total / spec.species_props(species).molar_mass
    for gas, atoms in _SULFUR_GASES.items():
        value = _gas(population, gas)
        if value is not None:
            sulfur = sulfur + atoms * value / GAS_SPECIES[gas].molar_mass
    out["sulfur"] = sulfur

    organic = zeros
    for species in _ORGANIC_AEROSOL:
        total = _mode_species_total(population, spec, species)
        if total is not None:
            organic = organic + total
    for gas in _ORGANIC_GASES:
        value = _gas(population, gas)
        if value is not None:
            organic = organic + value
    out["organic"] = organic

    for species in sorted({s for mode in spec.modes for s in mode.species}):
        if species in _SULFUR_AEROSOL or species in _ORGANIC_AEROSOL:
            continue
        total = _mode_species_total(population, spec, species)
        if total is not None:
            out[species] = total

    number = zeros
    for mode in spec.modes:
        number = number + population.numbers[number_name(mode.short)]
        key = number_name(mode.short, cloud_borne=True)
        if key in population.cloud_borne:
            number = number + population.cloud_borne[key]
    out["number"] = number
    return out


def relative_change(before: dict, after: dict) -> dict[str, jnp.ndarray]:
    """``(after − before) / max(|before|, tiny)`` per budget.

    The scale is the budget's own size, so an empty reservoir that stays
    empty reads as zero change rather than as 0/0.
    """
    out = {}
    for name, old in before.items():
        new = after[name]
        scale = jnp.maximum(jnp.abs(old), jnp.finfo(old.dtype).tiny)
        out[name] = (new - old) / scale
    return out


def violations(process: str, change: dict[str, jnp.ndarray], *,
               rtol: float) -> dict[str, float]:
    """Budgets whose change breaks ``process``'s contract, with the worst value.

    Concrete numbers (host side), for a test or a health gate: a budget
    with an ``equal`` contract fails above ``rtol`` in magnitude; a signed
    contract fails on the wrong sign beyond ``rtol``.
    """
    contract = CONTRACT[process]
    bad = {}
    for name, value in change.items():
        rule = contract.get(name, "equal")
        worst = float(jnp.max(jnp.abs(value)))
        if rule == "equal" and worst > rtol:
            bad[name] = worst
        elif rule == "non-decreasing" and float(jnp.min(value)) < -rtol:
            bad[name] = float(jnp.min(value))
        elif rule == "non-increasing" and float(jnp.max(value)) > rtol:
            bad[name] = float(jnp.max(value))
    return bad
