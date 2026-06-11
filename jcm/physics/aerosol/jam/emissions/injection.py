"""Smooth, differentiable vertical injection profiles for emissions (#498).

HAMMOZ injects emissions at discrete levels by type (``EM_SURFACE``,
``EM_LEVEL50M``, ``EM_VOLUME``, ``EM_FIRE`` in ``mo_hammoz_emissions.f90``). A
hard level-index pick has **zero/undefined gradient** w.r.t. the injection
height, which defeats the point of a differentiable GCM (injection height is a
large, calibratable uncertainty). So here the profile is a *smooth* normalised
Gaussian in geometric height, centred at ``injection_height`` with width
``injection_thickness``, evaluated on the model mid-layer heights — giving
well-defined ``d(emission)/d(height)`` and ``d(emission)/d(thickness)``. As the
thickness → 0 it collapses to a near-surface spike (the ``EM_SURFACE`` limit).

The returned per-level weights **sum to 1 over the column**, so distributing a
surface mass flux ``F`` [kg/m²/s] as ``F·w_k/(ρ_k·Δz_k)`` conserves the
column-integrated emitted mass exactly (``Σ ρ_k Δz_k · dq_k = F``).
"""

from __future__ import annotations

import jax.numpy as jnp

_MIN_THICKNESS = 1.0   # m — floor so the Gaussian never degenerates


def gaussian_injection_weights(
    height_full: jnp.ndarray,        # (nlev, ncols) mid-layer height [m]
    layer_thickness: jnp.ndarray,    # (nlev, ncols) [m]
    injection_height: jnp.ndarray,   # scalar [m]
    injection_thickness: jnp.ndarray,  # scalar [m]
) -> jnp.ndarray:
    """Per-level emission weights (sum to 1 over levels), smooth in the inputs.

    The continuous Gaussian ``g(z) = exp(-½((z−h)/σ)²)`` is discretised by
    weighting each level by its thickness, ``w_k ∝ g(z_k)·Δz_k``, then
    normalised. Differentiable w.r.t. ``injection_height`` (``h``) and
    ``injection_thickness`` (``σ``).
    """
    sigma = jnp.maximum(injection_thickness, _MIN_THICKNESS)
    z = (height_full - injection_height) / sigma
    g = jnp.exp(-0.5 * z * z) * layer_thickness
    total = jnp.sum(g, axis=0, keepdims=True)
    # Fall back to the lowest layer if the column has no overlap (numerically
    # the Gaussian underflowed everywhere) so weights still sum to 1.
    safe = total > 0.0
    weights = jnp.where(safe, g / jnp.where(safe, total, 1.0), 0.0)
    lowest = jnp.zeros_like(weights).at[-1].set(1.0)
    return jnp.where(safe, weights, lowest)
