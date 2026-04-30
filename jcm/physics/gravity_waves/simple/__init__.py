"""Simple single-amplitude gravity-wave drag scheme.

This is the original placeholder GWD that used to live under ``hines/`` —
*not* the actual Hines (1997) parameterisation, just a cheap monochromatic
drag with a Richardson-number breaking criterion plus an optional
mountain-wave source. Kept as a coexisting cheap option for aquaplanet
tests where running the full Hines spectrum or Lott-Miller SSO is
overkill. The proper schemes live in
``jcm/physics/gravity_waves/{hines,sso}``.
"""

from .simple_gwd import (
    SimpleGwdParameters,
    SimpleGwdState,
    SimpleGwdTendencies,
    brunt_vaisala_frequency,
    orographic_source,
    simple_gwd,
)

__all__ = [
    "SimpleGwdParameters",
    "SimpleGwdState",
    "SimpleGwdTendencies",
    "brunt_vaisala_frequency",
    "orographic_source",
    "simple_gwd",
]
