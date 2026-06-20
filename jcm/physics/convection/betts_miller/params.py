"""Configuration for the Betts-Miller convective-adjustment scheme.

:class:`BettsMillerParameters` is a JAX pytree (``flax.struct.dataclass``) so the
numeric tunables are differentiable leaves — gradients can be taken with respect
to them, exactly like the other physics parameter containers. The ``shallow``
flavor and the ``do_envsat`` / ``do_taucape`` modifiers are *static* fields
(``pytree_node=False``): they select genuinely different code paths (notably
``do_shallower``'s data-dependent depth reduction), so they are resolved by
Python branching at trace time rather than traced.

Defaults reproduce Isca's ``betts_miller_nml`` defaults: the Frierson (2007)
"Simplified Betts-Miller" scheme (``do_simp`` with ``rhbm=0.8``,
``tau_bm=7200 s``), reference humidity relative to the *parcel* (``do_envsat``
off), and a fixed relaxation timescale (``do_taucape`` off).
"""

from __future__ import annotations

import enum

import jax.numpy as jnp
from flax import struct


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


@struct.dataclass
class BettsMillerParameters:
    """Configuration for :class:`BettsMillerConvection`.

    The numeric fields are differentiable pytree leaves; ``shallow`` /
    ``do_envsat`` / ``do_taucape`` are static (``pytree_node=False``).

    Attributes:
        tau_bm: Relaxation timescale toward the reference profile (s).
        rhbm: Target relative humidity of the reference profile (fraction).
        capetaubm: CAPE (J/kg) at which the scaled timescale equals ``tau_bm``.
        tau_min: Floor on the CAPE-scaled timescale (s).
        buoyancy_kick: Temperature perturbation (K) added to the surface parcel.
        t_floor: Parcel temperature (K) below which the ascent is presumed
            CAPE-free (Isca's 173.16 K lookup-table floor).
        shallow: :class:`ShallowScheme` for the negative-precip / shallow case.
        do_envsat: If True, set reference humidity to ``rhbm`` × saturation wrt
            the *environment* temperature; if False (default), wrt the *parcel*.
        do_taucape: If True, scale ``tau_bm`` ∝ CAPE^(-1/2) (Isca ``do_taucape``).

    """

    # --- Differentiable tunables (pytree leaves) ----------------------------
    tau_bm: jnp.ndarray = 7200.0
    rhbm: jnp.ndarray = 0.8
    capetaubm: jnp.ndarray = 900.0
    tau_min: jnp.ndarray = 2400.0
    buoyancy_kick: jnp.ndarray = 0.0
    t_floor: jnp.ndarray = 173.16

    # --- Static configuration (selects code paths; not differentiated) ------
    shallow: ShallowScheme = struct.field(pytree_node=False, default=ShallowScheme.SIMP)
    do_envsat: bool = struct.field(pytree_node=False, default=False)
    do_taucape: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def default(cls) -> "BettsMillerParameters":
        """Return the default (Frierson-2007 Simplified Betts-Miller) configuration."""
        return cls()
