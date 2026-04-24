"""Numerical-dissipation terms (sponge layers, hyperdiffusion, etc.).

These are not physical parameterizations — they are stabilizers that damp
modes the coarse spectral dynamics cannot resolve. Keep them separate from
the physics schemes so it's clear what's physics and what's numerics.
"""

from jcm.physics.dissipation.upper_sponge import UpperSponge

__all__ = ["UpperSponge"]
