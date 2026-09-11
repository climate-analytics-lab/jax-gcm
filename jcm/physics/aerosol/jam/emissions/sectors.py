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
#: this, and the forcing fields are keyed ``emis_<sector>_<species>``. The first
#: three are CEDS anthropogenic activity super-sectors; ``biomass_burning`` is
#: open (GFED/van Marle) burning, distinguished only by its deeper FIRE injection
#: profile (HAMMOZ ``EM_FIRE``). All four go through the same speciation +
#: smooth-injection machinery and are independently gated by which
#: ``emis_<sector>_<species>`` forcing channels are supplied (absent ⇒ inert), so
#: an anthropogenic-only run simply omits the biomass channels.
SUPER_SECTORS: tuple[str, ...] = (
    "surface_combustion",   # CEDS TRA, RCO, AGR, WST, SLV — surface
    "elevated_industrial",  # CEDS ENE, IND — ~50 m
    "shipping",             # CEDS SHP — marine surface
    "biomass_burning",      # open burning (GFED) — deep FIRE injection profile
)

#: Aerosol-relevant CEDS species carried here (gas precursors NH3/NOx/CO are
#: out of scope until nitrate / fuller chemistry lands).
EMITTED_SPECIES: tuple[str, ...] = ("so2", "bc", "oc")


@dataclass(frozen=True)
class SectorDefaults:
    """Default injection geometry [m] and emitted sizes [m] for a super-sector.

    The emitted sizes are the volume-mean diameters CESM's emission-file
    generator uses to turn a mass flux into a number flux (the per-file
    ``mapping_equation`` attribute of the CMIP7 ``num_*`` products, e.g.
    ``num_bc_a4 = 1.0*BC ... 0.134e-6 1700``): accumulation sulfate 0.134 µm
    for surface/agricultural sources and 0.261 µm for energy/industrial and
    shipping, Aitken sulfate 0.0504 µm, primary carbon 0.134 µm.
    """

    injection_height: float     # Gaussian centre height
    injection_thickness: float  # Gaussian width
    so4_accum_diameter: float = 0.134e-6
    so4_aitken_diameter: float = 0.0504e-6
    carbon_diameter: float = 0.134e-6


SECTOR_DEFAULTS: dict[str, SectorDefaults] = {
    "surface_combustion": SectorDefaults(injection_height=0.0,
                                         injection_thickness=30.0),
    "elevated_industrial": SectorDefaults(injection_height=50.0,
                                          injection_thickness=30.0,
                                          so4_accum_diameter=0.261e-6),
    "shipping": SectorDefaults(injection_height=0.0,
                               injection_thickness=30.0,
                               so4_accum_diameter=0.261e-6),
    # Open biomass burning (HAMMOZ ``EM_FIRE``): smoke is lofted through a deep
    # layer rather than emitted at the surface. Defaults centre the smooth
    # Gaussian ~1 km up with a ~1.5 km width — a clearly elevated, deep profile
    # vs the near-surface sectors. Fire injection height is a large, poorly
    # constrained uncertainty (most fires inject within the boundary layer;
    # pyroconvection lofts the tail far higher), so these are deliberately just
    # defaults — ``injection_height``/``injection_thickness`` are differentiable
    # and meant to be calibrated.
    "biomass_burning": SectorDefaults(injection_height=1000.0,
                                      injection_thickness=1500.0),
}

# --- HAMMOZ species-handling constants (differentiable defaults on the term) --
SO4_PRIMARY_FRACTION = 0.025     # fraction of SO2 sulfur → primary SO4 (zfacso2)
OM_OC_RATIO = 1.4                # OM:OC mass ratio for OC → POA

#: SO₂ mass → SO₄ mass factor (one S atom each).
SO2_TO_SO4_MASS = SPECIES["so4"].molar_mass / GAS_SPECIES["so2"].molar_mass

# Which population classes receive primary SO4 / BC / POA — and in what
# proportion — is **not** decided here: it is the population's policy, owned by
# the aerosol spec's ``primary_emission`` table and queried via
# ``spec.primary_split(species)``. This keeps the term representation-agnostic
# (a sectional population supplies its own bin split) and centralises the
# assumption with the population, HAMMOZ-style. For MAM4 the split is 50/50
# Aitken/accumulation sulfate and primary-carbon-mode BC/POA (see
# ``mam4_data._MAM4_PRIMARY_EMISSION``).
#
# Note on emitted size: MAM4 carries a *single* primary-carbon mode, and CESM
# emits both anthropogenic and open-burning carbon at the same 0.134 µm
# volume-mean diameter, so biomass and anthropogenic carbon differ only in
# their injection profile. jcm's "surface_combustion" super-sector aggregates
# CEDS activities that CESM emits at different accumulation-sulfate sizes
# (0.134 µm agricultural, 0.261 µm solvents/waste); the aggregate takes the
# 0.134 µm value. CESM sends transport/residential sulfate to the Aitken mode
# and agricultural/energy/shipping sulfate to accumulation, where the
# population here splits every sector 50/50 (HAMMOZ ``cmr_sk``/``cmr_sa``).
