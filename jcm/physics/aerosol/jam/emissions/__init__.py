"""Aerosol emissions for the JAM harness."""

from jcm.physics.aerosol.jam.emissions.distributors import (
    distribute_surface_flux,
    particle_mean_mass,
)
from jcm.physics.aerosol.jam.emissions.emissions_term import (
    EmissionParameters,
    JamEmissions,
)

__all__ = [
    "JamEmissions",
    "EmissionParameters",
    "distribute_surface_flux",
    "particle_mean_mass",
]
