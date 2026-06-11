"""Super-sector emission characteristics for anthropogenic CEDS emissions (#498).

HAMMOZ varies emissions by **injection type** and **source size**, not economic
activity per se. We collapse CEDS's ~8 activity sectors into **3 characteristic
super-sectors**, each with a default injection height/thickness and a size→mode
placement. The HAMMOZ values here are *defaults*; the load-bearing uncertainties
(injection heights, the primary-SO₄ fraction) are exposed as differentiable
``EmissionParameters`` on the term.

Species handling (HAMMOZ ``mo_ham_m7_emissions`` / ``mo_hammoz_emissions``):

* **SO₂** — a small fraction (default 2.5 %, ``zfacso2 = 0.975``) of the sulfur
  is emitted as primary particulate **SO₄** (split ~50/50 Aitken/accum per
  ``cmr_sk``/``cmr_sa``); the remainder enters the **``g_so2``** gas tracer and
  is oxidised by the gas-phase sulfur chemistry (#496).
* **BC / OC** — primary carbonaceous mass into the MAM4 **primary_carbon** mode
  (the Aitken mode carries neither BC nor POA); OC→POA uses **OM:OC = 1.4**.
"""

from __future__ import annotations

from dataclasses import dataclass

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.species import SPECIES

#: Super-sectors in a fixed order — the ``EmissionParameters`` arrays index by
#: this, and the forcing fields are keyed ``emis_<sector>_<species>``.
SUPER_SECTORS: tuple[str, ...] = (
    "surface_combustion",   # CEDS TRA, RCO, AGR, WST, SLV — surface
    "elevated_industrial",  # CEDS ENE, IND — ~50 m
    "shipping",             # CEDS SHP — marine surface
)

#: Aerosol-relevant CEDS species carried here (gas precursors NH3/NOx/CO are
#: out of scope until nitrate / fuller chemistry lands).
EMITTED_SPECIES: tuple[str, ...] = ("so2", "bc", "oc")


@dataclass(frozen=True)
class SectorDefaults:
    """Default smooth-injection geometry for a super-sector [m]."""

    injection_height: float     # Gaussian centre height
    injection_thickness: float  # Gaussian width


SECTOR_DEFAULTS: dict[str, SectorDefaults] = {
    "surface_combustion": SectorDefaults(injection_height=0.0,
                                         injection_thickness=30.0),
    "elevated_industrial": SectorDefaults(injection_height=50.0,
                                          injection_thickness=30.0),
    "shipping": SectorDefaults(injection_height=0.0,
                               injection_thickness=30.0),
}

# --- HAMMOZ species-handling constants (differentiable defaults on the term) --
SO4_PRIMARY_FRACTION = 0.025     # fraction of SO2 sulfur → primary SO4 (zfacso2)
SO4_AITKEN_FRACTION = 0.5        # primary-SO4 split into Aitken vs accum
OM_OC_RATIO = 1.4                # OM:OC mass ratio for OC → POA

#: SO₂ mass → SO₄ mass factor (one S atom each).
SO2_TO_SO4_MASS = SPECIES["so4"].molar_mass / GAS_SPECIES["so2"].molar_mass

#: MAM4 mode shorts that receive the primary emissions.
SO4_MODES: tuple[str, str] = ("ait", "acc")   # primary sulfate
CARBON_MODE = "pcm"                            # primary_carbon (BC + POA)
