"""Aerosol wet deposition for the JAM harness."""

from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
    in_cloud_rate,
    reinjection_budget,
)

__all__ = [
    "WetScavenging",
    "WetDepParameters",
    "reinjection_budget",
    "in_cloud_rate",
    "below_cloud_rate",
]
