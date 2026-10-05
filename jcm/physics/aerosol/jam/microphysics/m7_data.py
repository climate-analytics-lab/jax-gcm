"""M7 (ECHAM-HAM) aerosol population, as a ``ModalAerosolSpec`` (jax-gcm#1017).

Builds the M7 population *shape* the ``echam-ham-m7`` preset runs on, the
way ``mam4_data.py`` builds MAM4's: a second population on the SAME JAM
harness (#461), never touching the global ``SPECIES`` table or
``MAM4_SPEC``. The real M7 microphysics core is a later task (the lead's
``microphysics/m7_jax.py``); ``M7_SPEC`` is exercised today by
``PlaceholderMicrophysics`` (registered as the ``"m7_placeholder"`` core in
``jam_terms.py``).

Provenance
----------
Every number below is transcribed from ECHAM6.3-HAM2.3 r7492
(``/data/dwatsonparris/echam6.3.0-ham2.3-moz1.0.r7492/src``, MPI-M/HAMMOZ
licence — ported as NUMBERS here, never as committed Fortran text):

* **Modes** (σ_g, size bounds, soluble/sediments/activation flags,
  ``csr_conv``): ``mo_ham_m7ctl.f90`` — the ``sizeclass(1..7)`` block
  (lines 331-396, classnames/shortnames/``lsoluble``/``lsed``/
  ``lactivation``), ``sigma_fine``/``sigma_coarse`` (``mo_ham.f90:320-321``)
  assembled into the per-mode ``sigma`` array (``mo_ham_m7ctl.f90:171-172``),
  the dry-radius mode boundaries ``crdiv`` (``mo_ham_m7ctl.f90:164``) and
  ``csr_conv`` (``mo_ham_m7ctl.f90:518``).
* **Species** (molar mass, density, κ, electrolyte flag ``nion``):
  ``mo_ham_species.f90`` — SO4 (294-311), BC (343-355), the ``HAM_M7``
  branch of OC (362-375), sea salt (382-398), dust (404-416) and aerosol
  water (422-435); SO4's molar mass itself is ``mo_ham.f90:311``
  (``mw_so4``). Defaults for an unset ``kappa``/``nion`` are 0
  (``mo_species.f90:204``).
* **Wet-removal tables** (``csr_strat_wat/mix/ice``, ``cbcr``, ``cbcs``,
  ``caccso4``): ``mo_ham_m7ctl.f90:213`` (``caccso4``) and ``515-526``
  (the rest) — carried here for the ``nwetdep`` variants a later task wires
  up; the M7 preset does not read them yet.

Mode order and short tokens follow HAM's own two-letter class names in
lower case (``ns ks as cs ki ai ci``: nucleation/Aitken/accumulation/coarse
soluble, then Aitken/accumulation/coarse insoluble — ``inucs=1`` ..
``icoai=7``, ``mo_ham_m7ctl.f90:150-151``), matching the design note
``docs/source/design/ham_m7_configuration.md``.
"""

from __future__ import annotations

import math

from jcm.physics.aerosol.jam.emissions.dust import m7_dust_emission_policy
from jcm.physics.aerosol.jam.ice_nucleation.ham_freezing import HamFreezingClasses
from jcm.physics.aerosol.jam.population import (
    AerosolMode,
    AerosolSpecies,
    ModalAerosolSpec,
)

# --- Species (mo_ham_species.f90; SO4 molar mass from mo_ham.f90:311) ------
M7_SPECIES: tuple[AerosolSpecies, ...] = (
    AerosolSpecies("so4", molar_mass=0.0960631, density=1841.0,
                   hygroscopicity=0.60, long_name="sulfate"),
    AerosolSpecies("bc", molar_mass=0.01201, density=2000.0,
                   hygroscopicity=0.0, long_name="black-carbon"),
    AerosolSpecies("oc", molar_mass=0.180, density=2000.0,
                   hygroscopicity=0.06, long_name="organic-carbon"),
    AerosolSpecies("ss", molar_mass=0.058443, density=2165.0,
                   hygroscopicity=1.0, long_name="sea-salt"),
    AerosolSpecies("du", molar_mass=0.250, density=2650.0,
                   hygroscopicity=0.0, long_name="dust"),
    # Aerosol water: the condensate, not a dry solute (κ=0). Not carried by
    # any mode below (the core diagnoses it, like MAM4's own "h2o" entry;
    # see population.py's AerosolMode.species docstring).
    AerosolSpecies("h2o", molar_mass=0.018, density=1000.0,
                   hygroscopicity=0.0, long_name="aerosol-water"),
)

#: token -> molar mass [kg/mol], for callers that want a quick lookup
#: without constructing the tuple above into a dict themselves.
M7_SPECIES_BY_NAME: dict[str, AerosolSpecies] = {s.name: s for s in M7_SPECIES}

#: Electrolyte flag ``nion`` per species (``mo_ham_species.f90``; default 0,
#: ``mo_species.f90:204``). Loaded by a later HAM-activation task (Köhler
#: A/B terms distinguish electrolyte species); carried on the population here
#: so that table has one source.
#: so4 (mo_ham_species.f90:307), ss (:393) are electrolytes; bc/oc/du take
#: the un-set default of 0 (mo_species.f90:204).
M7_NION: dict[str, int] = {"so4": 2, "bc": 0, "oc": 0, "ss": 2, "du": 0}

# --- Modes (mo_ham_m7ctl.f90) -----------------------------------------------
#
# Dry-radius mode boundaries ``crdiv`` = 0.0005, 0.005, 0.05, 0.5 µm
# (mo_ham_m7ctl.f90:164) as DIAMETERS: 1 nm, 10 nm, 100 nm, 1 µm. M7 has no
# upper bound on its coarse modes; 10 µm is the edge of HAM's sea-salt/dust
# size integrations (the design note), not an M7 mode-geometry constant.
_D_NM1, _D_NM10, _D_NM100, _D_UM1, _D_UM10 = (
    1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5)


def _geometric_midpoint(lo: float, hi: float) -> float:
    """``dgnum``: the geometric midpoint of a mode's diameter bounds."""
    return math.sqrt(lo * hi)


M7_MODES: tuple[AerosolMode, ...] = (
    AerosolMode(
        name="nucleation_soluble", short="ns", geom_std_dev=1.59,
        dgnum=_geometric_midpoint(_D_NM1, _D_NM10),
        dgnum_lo=_D_NM1, dgnum_hi=_D_NM10,
        species=("so4",), soluble=True, can_activate=False, sediments=False,
        csr_conv=0.20,
    ),
    AerosolMode(
        name="aitken_soluble", short="ks", geom_std_dev=1.59,
        dgnum=_geometric_midpoint(_D_NM10, _D_NM100),
        dgnum_lo=_D_NM10, dgnum_hi=_D_NM100,
        species=("so4", "bc", "oc"),
        soluble=True, can_activate=True, sediments=False,
        csr_conv=0.60,
    ),
    AerosolMode(
        name="accumulation_soluble", short="as", geom_std_dev=1.59,
        dgnum=_geometric_midpoint(_D_NM100, _D_UM1),
        dgnum_lo=_D_NM100, dgnum_hi=_D_UM1,
        species=("so4", "bc", "oc", "ss", "du"),
        soluble=True, can_activate=True, sediments=True,
        csr_conv=0.99,
    ),
    AerosolMode(
        name="coarse_soluble", short="cs", geom_std_dev=2.0,
        dgnum=_geometric_midpoint(_D_UM1, _D_UM10),
        dgnum_lo=_D_UM1, dgnum_hi=_D_UM10,
        species=("so4", "bc", "oc", "ss", "du"),
        soluble=True, can_activate=True, sediments=True,
        csr_conv=0.99,
    ),
    AerosolMode(
        name="aitken_insoluble", short="ki", geom_std_dev=1.59,
        dgnum=_geometric_midpoint(_D_NM10, _D_NM100),
        dgnum_lo=_D_NM10, dgnum_hi=_D_NM100,
        species=("bc", "oc"),
        soluble=False, can_activate=False, sediments=False,
        csr_conv=0.20,
    ),
    AerosolMode(
        name="accumulation_insoluble", short="ai", geom_std_dev=1.59,
        dgnum=_geometric_midpoint(_D_NM100, _D_UM1),
        dgnum_lo=_D_NM100, dgnum_hi=_D_UM1,
        species=("du",),
        soluble=False, can_activate=False, sediments=True,
        csr_conv=0.40,
    ),
    AerosolMode(
        name="coarse_insoluble", short="ci", geom_std_dev=2.0,
        dgnum=_geometric_midpoint(_D_UM1, _D_UM10),
        dgnum_lo=_D_UM1, dgnum_hi=_D_UM10,
        species=("du",),
        soluble=False, can_activate=False, sediments=True,
        csr_conv=0.40,
    ),
)

# --- Per-mode HAM tables not yet consumed by the M7 preset ------------------
#
# Stratiform in-cloud scavenging fractions and the below-cloud mean-mass
# scavenging coefficients (mo_ham_m7ctl.f90:515-526), and the H2SO4
# accommodation coefficient (mo_ham_m7ctl.f90:213) — keyed by mode SHORT, in
# the order ns ks as cs ki ai ci (inucs..icoai, mo_ham_m7ctl.f90:150-151).
_SHORTS = tuple(m.short for m in M7_MODES)


def _table(values: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(_SHORTS, values))


#: In-cloud scavenging fraction, stratiform warm cloud (mo_ham_m7ctl.f90:515).
M7_CSR_STRAT_WAT = _table((0.10, 0.25, 0.85, 0.99, 0.20, 0.40, 0.40))
#: ... mixed-phase cloud (mo_ham_m7ctl.f90:516).
M7_CSR_STRAT_MIX = _table((0.10, 0.40, 0.75, 0.75, 0.10, 0.40, 0.40))
#: ... ice cloud (mo_ham_m7ctl.f90:517).
M7_CSR_STRAT_ICE = _table((0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10))
#: Below-cloud mean-mass scavenging coefficient normalised by rain rate
#: [kg m⁻²] (mo_ham_m7ctl.f90:521-522).
M7_CBCR = _table((5.0e-4, 1.0e-4, 1.0e-3, 1.0e-1, 1.0e-4, 1.0e-3, 1.0e-1))
#: ... by snow rate (mo_ham_m7ctl.f90:525-526).
M7_CBCS = _table((5.0e-3,) * 7)
#: H2SO4 accommodation coefficient, reduced for insoluble modes
#: (mo_ham_m7ctl.f90:213).
M7_CACCSO4 = _table((1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3))

# --- Primary-emission policy (the population's own split table) ------------
#
# so4 -> Aitken/accumulation soluble, 50/50 (HAM's fossil-fuel default split,
# cmr_sk/cmr_sa, mo_ham_m7_emissions.f90:165-168); bc/oc (fresh, hydrophobic)
# -> Aitken insoluble, 100% (mo_ham_m7_emissions.f90, the KI pfactor terms);
# du's accumulation/coarse-insoluble SPLIT here mirrors MAM4's 10/90
# acc/cor convention (see mam4_data._MAM4_PRIMARY_EMISSION) for the sectored
# emission terms that read it — M7's OWN dust source (BGC/Tegen) does not use
# this table at all; it goes through ``M7_SPEC.dust_emission`` instead (see
# below), which is the real HAM AI/CI policy.
_M7_PRIMARY_EMISSION = {
    "so4": (("ks", 0.5), ("as", 0.5)),
    "bc": (("ki", 1.0),),
    "oc": (("ki", 1.0),),
    "du": (("ai", 0.1), ("ci", 0.9)),
}

#: Which classes play HAM's roles in the aerosol inputs to mixed-phase
#: freezing (``ice_nucleation/ham_freezing.py``): soluble accumulation/coarse
#: for immersion, insoluble Aitken/accumulation/coarse for contact — this IS
#: HAM's own class set, unlike MAM4's approximating mapping.
M7_FREEZING_ROLES = HamFreezingClasses(
    soluble=("as", "cs"), insoluble_aitken="ki",
    insoluble_accumulation="ai", insoluble_coarse="ci",
)

#: The M7 population: seven log-normal modes, no explicit cloud-borne phase
#: (HAM scavenges interstitial aerosol by its activated fraction; see
#: ``ModalAerosolSpec.cloud_borne``'s docstring).
M7_SPEC = ModalAerosolSpec(
    modes=M7_MODES,
    species=M7_SPECIES,
    family="modal",
    cloud_borne=False,
    primary_emission=_M7_PRIMARY_EMISSION,
    accumulation_mode="as",
    freezing_roles=M7_FREEZING_ROLES,
    # HAM sums the pH-setting sulfate over, and splits the produced sulfate
    # over, exactly the soluble accumulation/coarse classes
    # (mo_ham_chemistry.f90 ``ham_wet_chemistry``; chemistry/aqueous.py).
    aqueous_sulfate_modes=("as", "cs"),
    dust_emission=m7_dust_emission_policy(
        M7_SPECIES_BY_NAME["du"].density),
)
