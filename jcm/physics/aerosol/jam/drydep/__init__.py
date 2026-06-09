"""Aerosol dry deposition for the JAM harness."""

from jcm.physics.aerosol.jam.drydep.drydep_term import (
    DryDepParameters,
    SlinnDryDeposition,
)
from jcm.physics.aerosol.jam.drydep.resistances import (
    aerodynamic_resistance,
    deposition_velocity,
    quasi_laminar_resistance,
)

__all__ = [
    "SlinnDryDeposition",
    "DryDepParameters",
    "deposition_velocity",
    "aerodynamic_resistance",
    "quasi_laminar_resistance",
]
