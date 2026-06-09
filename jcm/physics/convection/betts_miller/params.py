"""Configuration for the Betts-Miller convective-adjustment scheme.

These are *static* configuration values: the ``shallow`` flavor and the
``do_envsat`` / ``do_taucape`` modifiers select genuinely different code paths
(notably ``do_shallower``'s data-dependent depth reduction), so they are chosen
at construction/trace time via Python branching rather than traced. The numeric
tunables (``tau_bm``, ``rhbm``, ...) are likewise construction-time constants.

Defaults reproduce Isca's ``betts_miller_nml`` defaults: the Frierson (2007)
"Simplified Betts-Miller" scheme (``do_simp`` with ``rhbm=0.8``,
``tau_bm=7200 s``), reference humidity relative to the *parcel* (``do_envsat``
off), and a fixed relaxation timescale (``do_taucape`` off).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ShallowScheme(enum.Enum):
    """How to treat columns whose deep adjustment would dry the column (precip < 0).

    This is the Betts-Miller "flavor" — every option shares the deep
    precipitating relaxation and differs only in the negative-precip branch.
    """

    #: Frierson (2007) Simplified Betts-Miller: shorten a relaxation timescale so
    #: the column-integrated heating and moistening stay energy-consistent and
    #: precipitation never goes negative (Isca ``do_simp=.true.``).
    SIMP = "simp"
    #: Iteratively raise the cloud top until the column-integrated precip is
    #: non-negative, then trim the topmost active layer (Isca ``do_shallower``).
    SHALLOWER = "shallower"
    #: Scale the reference humidity (and temperature) profiles so the
    #: column-integrated precip is exactly zero (Isca ``do_changeqref``).
    CHANGEQREF = "changeqref"
    #: No shallow scheme: zero the tendencies when the deep adjustment would dry
    #: the column (Isca with all shallow flags false).
    NONE = "none"


@dataclass(frozen=True)
class BettsMillerParameters:
    """Static configuration for :class:`BettsMillerConvection`.

    Attributes:
        tau_bm: Relaxation timescale toward the reference profile (s).
        rhbm: Target relative humidity of the reference profile (fraction).
        shallow: :class:`ShallowScheme` for the negative-precip / shallow case.
        do_envsat: If True, set reference humidity to ``rhbm`` × saturation wrt
            the *environment* temperature; if False (default), wrt the *parcel*.
        do_taucape: If True, scale ``tau_bm`` ∝ CAPE^(-1/2) (Isca ``do_taucape``).
        capetaubm: CAPE (J/kg) at which the scaled timescale equals ``tau_bm``.
        tau_min: Floor on the CAPE-scaled timescale (s).
        buoyancy_kick: Temperature perturbation (K) added to the surface parcel.

    """

    tau_bm: float = 7200.0
    rhbm: float = 0.8
    shallow: ShallowScheme = ShallowScheme.SIMP
    do_envsat: bool = False
    do_taucape: bool = False
    capetaubm: float = 900.0
    tau_min: float = 2400.0
    buoyancy_kick: float = 0.0

    @classmethod
    def default(cls) -> "BettsMillerParameters":
        """Return the default (Frierson-2007 Simplified Betts-Miller) configuration."""
        return cls()
