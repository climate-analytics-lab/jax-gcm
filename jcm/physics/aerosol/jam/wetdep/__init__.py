"""Aerosol wet deposition for the JAM harness."""

from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
    reinjection_budget,
)

__all__ = [
    "WetScavenging",
    "WetDepParameters",
    "reinjection_budget",
    "below_cloud_rate",
]
