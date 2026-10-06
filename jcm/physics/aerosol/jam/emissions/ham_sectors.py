"""HAM's per-sector-class primary emission targets for a population (jax-gcm#1017).

Port of ``mo_ham_m7_emissions.f90``: the number-median radii and mass-to-
number factors ``ham_m7_init_emissions`` sets up (lines 90-262) and the
per-sector-class ``pfactor`` assignment ``ham_m7_emissions`` makes for BC,
OC and primary SO4 (lines 564-646).

HAM branches by Fortran sector index, which collapses onto exactly three
size/mode "classes":

* ``"fossil"`` — the Fortran's default/else branch: fossil-fuel-like
  surface and elevated sources (AEROCOM/IPCC ``idsec_fossil``, ``idsec_ind``,
  ``idsec_tra``, ``idsec_wst``, and by omission ``idsec_agr``/``idsec_slv``).
* ``"energy_ships"`` — ``idsec_ene`` (energy-sector smoke stacks) and
  ``idsec_ships`` (international shipping): SO4 only (lines 628-636), which
  HAM special-cases into accumulation+coarse instead of Aitken+accumulation;
  BC/OC have no ENE/SHIPS-specific branch, so they take the SAME targets as
  ``"fossil"``.
* ``"biomass_like"`` — ``idsec_fire``/``idsec_ffire``/``idsec_gfire``/
  ``idsec_awb``/``idsec_dom``/``idsec_biofuel``: BC/OC take HAM's biomass
  number-median radius ``cmr_bb`` (lines 569-574, 586-591); **SO4 does
  NOT** — ``zm2n_s4ks_bb``/``zm2n_s4as_bb`` (lines 253-254) are textually
  IDENTICAL to ``zm2n_s4ks_ff``/``zm2n_s4as_ff`` (lines 250-251): both use
  ``cmr_sk``/``cmr_sa``, not a distinct biomass radius. This looks like an
  oversight in the reference Fortran (every OTHER species does get a
  biomass-specific ``cmr``), but it is what the compiled code does, so
  ``"biomass_like"``'s SO4 targets are a faithful, bit-for-bit copy of
  ``"fossil"``'s.

A :class:`SectorTarget` is ``(species, mode, mass_fraction, cmr_m)``:
``mass_fraction`` of the species' ALREADY-SCALED emitted mass (BC: the raw
BC flux; OC: the raw OC flux, the OM:OC scaling applied once by the caller,
not baked in here, matching HAM's own ``pfactor(idx_mocki) = zom2oc·(...)``
structure; SO4: the primary-SO4 mass the caller already derived from the
SO2 flux — HAM's ``zso2tso4·(1−zfacso2)``, see ``sectors.py``), and
``cmr_m`` the number-median radius [m] HAM assumes for that target.

HAM's mass->number conversion (``zm2n = 3/(4·pi·rho·(cmr·cmr2ram(mode))³)``,
``cmr2ram(mode) = exp(1.5·ln(sigma_mode)²)``, ``mo_ham_m7ctl.f90:427``) is
reproduced EXACTLY by jcm's existing ``distributors.particle_mean_mass``
monodisperse-diameter path, given the volume-mean diameter
``D = 2·cmr·cmr2ram(sigma_mode)`` (:func:`cmr_to_emission_diameter`) — see
``ham_sectors_test.py`` for the algebraic check against a direct
transcription of the Fortran formula.
"""

from __future__ import annotations

import dataclasses
import math

#: HAM's assumed number-median radii [m] (``mo_ham_m7_emissions.f90:150-175``).
CMR_FF = 0.03e-6    # fossil fuel (BC/OC insoluble Aitken; SO4 Aitken half)
CMR_BB = 0.075e-6   # biomass burning (BC/OC)
CMR_SK = 0.03e-6    # primary SO4 -> Aitken soluble
CMR_SA = 0.075e-6   # primary SO4 -> accumulation soluble
CMR_SC = 0.75e-6    # primary SO4 -> coarse soluble (energy/ships)

#: HAM's biomass-burning water-soluble-OC fraction, ``zbb_wsoc_perc``
#: (``mo_ham_m7_emissions.f90:90``): the soluble (Aitken-soluble) share of
#: biomass OC; ``1 - this`` is the insoluble (Aitken-insoluble) share.
BB_WSOC_FRACTION = 0.65

#: Three HAM emission classes every jcm super-sector resolves to.
HAM_SECTOR_CLASSES: tuple[str, ...] = ("fossil", "energy_ships", "biomass_like")


def _cmr2ram(sigma_g: float) -> float:
    """HAM's count-median-radius -> radius-of-average-mass factor.

    ``mo_ham_m7ctl.f90:427``: ``cmr2ram(mode) = exp(1.5·ln(sigma_mode)²)``.
    """
    return math.exp(1.5 * math.log(sigma_g) ** 2)


def cmr_to_emission_diameter(cmr_m: float, sigma_g: float) -> float:
    """Volume-mean diameter [m] reproducing HAM's ``zm2n`` for ``(cmr, sigma_g)``.

    HAM's ``zm2n = 3/(4·pi·rho·(cmr·cmr2ram(sigma_g))³)`` is the reciprocal
    of a MONODISPERSE particle mass at radius ``r_ram = cmr·cmr2ram(sigma_g)``
    (``m_p = rho·(4/3)·pi·r_ram³``), i.e. exactly
    ``distributors.particle_mean_mass(mode, rho, emission_diameter=D)`` with
    ``D = 2·r_ram`` — see ``ham_sectors_test.py`` for the algebraic check.
    """
    return 2.0 * cmr_m * _cmr2ram(sigma_g)


@dataclasses.dataclass(frozen=True)
class SectorTarget:
    """One ``(species, mode, mass fraction, number-median radius)`` target."""

    species: str
    mode: str
    mass_fraction: float
    cmr_m: float


@dataclasses.dataclass(frozen=True)
class HamSectorPolicy:
    """Per-HAM-class primary emission targets plus the OC->OM factor.

    ``targets[ham_class][species]`` is a tuple of :class:`SectorTarget`
    whose ``mass_fraction``s are HAM's own split for that
    ``(ham_class, species)`` pair. ``om_oc`` is the organic-carbon->organic-
    matter mass multiplier (``zom2oc``, ``mo_ham_m7_emissions.f90:93``),
    applied once by the caller to the raw OC flux before the per-target
    mass fractions (mirroring the Fortran's own ``pfactor(idx_mocki) =
    zom2oc·(...)`` structure, where ``zom2oc`` multiplies every OC pfactor).
    """

    targets: dict[str, dict[str, tuple[SectorTarget, ...]]]
    om_oc: float


def m7_sector_policy(om_oc: float) -> HamSectorPolicy:
    """Build the M7 :class:`HamSectorPolicy` (``om_oc`` from ``sectors.OM_OC_RATIO``).

    Mode names are M7's own short tokens (``ks ai cs`` etc. as
    ``microphysics/m7_data.py`` defines them): ``ki`` = insoluble Aitken,
    ``ks``/``as``/``cs`` = soluble Aitken/accumulation/coarse.
    """
    fossil = {
        "bc": (SectorTarget("bc", "ki", 1.0, CMR_FF),),
        "oc": (SectorTarget("oc", "ki", 1.0, CMR_FF),),
        "so4": (
            SectorTarget("so4", "ks", 0.5, CMR_SK),
            SectorTarget("so4", "as", 0.5, CMR_SA),
        ),
    }
    energy_ships = {
        # No ENE/SHIPS-specific BC/OC branch in the Fortran -> same as fossil
        # (mo_ham_m7_emissions.f90:564-606 only branches on the biomass-like
        # sector list for BC/OC).
        "bc": fossil["bc"],
        "oc": fossil["oc"],
        # lines 628-636: ms4ks zeroed, ms4as left at its fossil default
        # (never reassigned here, so it keeps cmr_sa), ms4cs newly set to
        # the other half at cmr_sc.
        "so4": (
            SectorTarget("so4", "as", 0.5, CMR_SA),
            SectorTarget("so4", "cs", 0.5, CMR_SC),
        ),
    }
    biomass_like = {
        "bc": (SectorTarget("bc", "ki", 1.0, CMR_BB),),
        "oc": (
            SectorTarget("oc", "ki", 1.0 - BB_WSOC_FRACTION, CMR_BB),
            SectorTarget("oc", "ks", BB_WSOC_FRACTION, CMR_BB),
        ),
        # lines 639-642: only the NUMBER factors are reassigned here, and
        # (per the module docstring) they are textually identical to the
        # fossil ones -- so the mass/cmr targets are fossil's, unchanged.
        "so4": fossil["so4"],
    }
    return HamSectorPolicy(
        targets={
            "fossil": fossil,
            "energy_ships": energy_ships,
            "biomass_like": biomass_like,
        },
        om_oc=om_oc,
    )


#: jcm super-sector -> (its own/"main" HAM class, optional
#: ``(subset_channel_name, subset HAM class)``). The main channel's flux
#: INCLUDES the subset's (it is not additional mass): the subset's own part
#: is ``emis_<subset_channel_name>_<species>`` when supplied, and the
#: remainder (main minus subset) takes the main class's targets.
SECTOR_ROUTING: dict[str, tuple[str, tuple[str, str] | None]] = {
    "surface_combustion": ("fossil", ("residential", "biomass_like")),
    "elevated_industrial": ("fossil", ("energy", "energy_ships")),
    "shipping": ("energy_ships", None),
    "biomass_burning": ("biomass_like", None),
}
