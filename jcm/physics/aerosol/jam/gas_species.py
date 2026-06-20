"""Gas-phase aerosol-precursor species for the JAM sulfur cycle (#496).

Molar masses [kg/mol] for the prognostic gas tracers the gas-phase chemistry
carries: DMS, SO₂, sulfuric-acid vapour, and a single lumped SOA gas. Values
match the MAM4-JAX gas table (``mam4_jax.data.MW_GAS`` / ``ADV_MASS``) so the
adapter's hand-off is unit-consistent.

Only :data:`MAM4_GAS` (``h2so4``, ``soag``) is passed into the MAM4-JAX core —
its condensation/nucleation consume those two. ``dms``/``so2`` are jcm-side
precursors (their oxidation chemistry is jcm's; MAM4 ignores SO₂/DMS).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GasSpecies:
    """A prognostic gas-phase precursor (pure data, compose-time only)."""

    name: str
    molar_mass: float   # kg/mol
    long_name: str


# token -> GasSpecies. Molar masses are the standard compound values, matching
# MAM4-JAX (H₂SO₄ 98.0784, SOA-gas 150.0, SO₂ 64.0648, DMS 62.1324 g/mol).
GAS_SPECIES: dict[str, GasSpecies] = {
    "dms": GasSpecies("dms", molar_mass=0.0621324,
                      long_name="dimethyl-sulfide"),
    "so2": GasSpecies("so2", molar_mass=0.0640648,
                      long_name="sulfur-dioxide"),
    "h2so4": GasSpecies("h2so4", molar_mass=0.0980784,
                        long_name="sulfuric-acid-vapour"),
    # SOAG = the lumped semi-volatile organic *gas* that condenses to form
    # secondary organic aerosol (SOA); it is the gas-phase SOA precursor, hence
    # not itself an aerosol despite the MAM4 "SOAG" name.
    "soag": GasSpecies("soag", molar_mass=0.150,
                       long_name="secondary-organic-aerosol precursor gas"),
}

#: All prognostic gas tracers carried by the sulfur cycle, in chain order.
SULFUR_GASES: tuple[str, ...] = ("dms", "so2", "h2so4", "soag")

#: The subset fed into the MAM4-JAX core's ``q`` gas slots (amicphys consumes
#: these); the others stay jcm-side.
MAM4_GAS: tuple[str, ...] = ("h2so4", "soag")
