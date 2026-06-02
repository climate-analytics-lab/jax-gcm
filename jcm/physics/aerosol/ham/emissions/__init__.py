"""Aerosol emissions for the HAM harness."""

from jcm.physics.aerosol.ham.emissions.distributors import (
    distribute_surface_flux,
    particle_mean_mass,
)
from jcm.physics.aerosol.ham.emissions.emissions_term import (
    EmissionParameters,
    HamEmissions,
)

__all__ = [
    "HamEmissions",
    "EmissionParameters",
    "distribute_surface_flux",
    "particle_mean_mass",
]
