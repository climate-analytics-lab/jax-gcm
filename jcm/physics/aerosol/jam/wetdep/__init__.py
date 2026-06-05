"""Aerosol wet deposition for the JAM harness."""

from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
    in_cloud_rate,
    precip_formation_rate,
)

__all__ = [
    "WetScavenging",
    "WetDepParameters",
    "precip_formation_rate",
    "in_cloud_rate",
    "below_cloud_rate",
]
