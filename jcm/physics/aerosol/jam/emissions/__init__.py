"""Aerosol emissions for the JAM harness.

Each natural source is a faithful port of its HAMMOZ scheme, as its own
``PhysicsTerm`` with a calibratable ``Parameters`` object:

- :class:`SeaSaltEmissions` — Gong (2003) wind-driven sea salt.
- :class:`DmsEmissions` — Nightingale (2000) oceanic DMS → sulfate.
- :class:`DustEmissions` — Tegen et al. (2002) wind-erosion flux.

``distribute_surface_flux`` maps ``(species, mode, flux)`` triples to
lowest-layer modal tracer tendencies (mass + implied number) and is reused by
the DMS/dust terms (and the future prescribed-emissions path, #498).
"""

from jcm.physics.aerosol.jam.emissions.distributors import (
    distribute_surface_flux,
    particle_mean_mass,
)
from jcm.physics.aerosol.jam.emissions.dms import DmsEmissions, DmsParameters
from jcm.physics.aerosol.jam.emissions.dust import DustEmissions, DustParameters
from jcm.physics.aerosol.jam.emissions.seasalt import (
    SeaSaltEmissions,
    SeaSaltParameters,
)

__all__ = [
    "SeaSaltEmissions",
    "SeaSaltParameters",
    "DmsEmissions",
    "DmsParameters",
    "DustEmissions",
    "DustParameters",
    "distribute_surface_flux",
    "particle_mean_mass",
]
