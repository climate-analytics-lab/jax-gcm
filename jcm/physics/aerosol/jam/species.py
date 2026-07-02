"""Canonical aerosol species table.

Density and hygroscopicity (κ) are the load-bearing properties for the
κ-Köhler placeholder and ARG activation; molar masses are informational
until the real microphysics core (gas–aerosol exchange) needs them.

Provenance: density and κ are transcribed from the E3SM/MAM4 source
``rad_constituents.F90`` (MODAL_AERO_4MODE_MOM branch,
``specdens_amode`` / ``spechygro``), cross-checked against the MAM4-JAX
port (``mam4_jax/data.py`` @ commit ccd872b). They are physical constants,
not code. Molar masses are standard textbook values for the representative
compound of each type and are approximate.
"""

from __future__ import annotations

from jcm.physics.aerosol.jam.population import AerosolSpecies

# token -> AerosolSpecies. The short token is the canonical key a
# microphysics-core adapter maps onto its own species slots (e.g. MAM4's
# SPECNAME_AMODE); ``long_name`` is just a readable label.
SPECIES: dict[str, AerosolSpecies] = {
    "so4": AerosolSpecies("so4", molar_mass=0.115, density=1770.0,
                          hygroscopicity=0.507, long_name="sulfate"),
    "nh4": AerosolSpecies("nh4", molar_mass=0.018, density=1770.0,
                          hygroscopicity=0.507, long_name="ammonium"),
    "no3": AerosolSpecies("no3", molar_mass=0.062, density=1770.0,
                          hygroscopicity=0.507, long_name="nitrate"),
    "poa": AerosolSpecies("poa", molar_mass=0.012, density=1000.0,
                          hygroscopicity=0.010, long_name="particulate-organic"),
    "soa": AerosolSpecies("soa", molar_mass=0.012, density=1000.0,
                          hygroscopicity=0.140, long_name="secondary-organic"),
    "bc": AerosolSpecies("bc", molar_mass=0.012, density=1700.0,
                         hygroscopicity=1.0e-10, long_name="black-carbon"),
    "ss": AerosolSpecies("ss", molar_mass=0.058, density=1900.0,
                         hygroscopicity=1.160, long_name="sea-salt"),
    "du": AerosolSpecies("du", molar_mass=0.135, density=2600.0,
                         hygroscopicity=0.068, long_name="dust"),
    "moa": AerosolSpecies("moa", molar_mass=0.250, density=1601.0,
                          hygroscopicity=0.100, long_name="marine-organic"),
    # Aerosol water — κ=0 (it is the condensate, not a dry solute).
    "h2o": AerosolSpecies("h2o", molar_mass=0.018, density=1000.0,
                          hygroscopicity=0.0, long_name="aerosol-water"),
}
