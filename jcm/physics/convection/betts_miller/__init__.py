"""Betts-Miller convective adjustment (Isca / Frierson 2007 family).

A flexible, level-agnostic moist convective-adjustment scheme that relaxes
temperature and humidity toward a moist-adiabatic reference profile with a
target relative humidity. Ported from Isca's ``betts_miller.f90``.

Public API:
    * :class:`~jcm.physics.convection.betts_miller.params.BettsMillerParameters`
    * :class:`~jcm.physics.convection.betts_miller.betts_miller_terms.BettsMillerConvection`
"""

from jcm.physics.convection.betts_miller.params import (
    BettsMillerParameters,
    ShallowScheme,
)
from jcm.physics.convection.betts_miller.betts_miller_terms import (
    BettsMillerConvection,
)

__all__ = [
    "BettsMillerParameters",
    "ShallowScheme",
    "BettsMillerConvection",
]
