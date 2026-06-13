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
    # Fallback when the Gaussian underflowed in *every* layer — i.e. the
    # injection height sits far outside the column (above the model top or below
    # the surface) or the width is so small no mid-layer height registers. Put
    # all the mass in the single layer whose height is nearest the injection
    # height, so an above-top height loads the top layer and a below-surface one
    # the bottom layer. (Previously this always picked the lowest layer, which
    # silently turned a calibrated high-altitude injection into a surface
    # emission — corrupting gradient-based tuning that explores large heights.)
    safe = total > 0.0
    weights = jnp.where(safe, g / jnp.where(safe, total, 1.0), 0.0)
    nlev = height_full.shape[0]
    nearest_idx = jnp.argmin(jnp.abs(height_full - injection_height), axis=0)
    nearest = (jnp.arange(nlev)[:, None] == nearest_idx[None, :]).astype(
        weights.dtype)
    return jnp.where(safe, weights, nearest)
