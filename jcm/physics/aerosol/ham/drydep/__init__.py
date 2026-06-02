"""Aerosol dry deposition for the HAM harness."""

from jcm.physics.aerosol.ham.drydep.drydep_term import (
    DryDepParameters,
    HamDryDeposition,
)
from jcm.physics.aerosol.ham.drydep.resistances import (
    aerodynamic_resistance,
    deposition_velocity,
    quasi_laminar_resistance,
)

__all__ = [
    "HamDryDeposition",
    "DryDepParameters",
    "deposition_velocity",
    "aerodynamic_resistance",
    "quasi_laminar_resistance",
]
