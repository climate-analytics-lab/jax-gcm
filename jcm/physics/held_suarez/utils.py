"""Held-Suarez physics utilities for coordinate system creation.

Date: 2026-01-26
"""
import jax.numpy as jnp
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.utils import get_coords as _get_coords


# Standard sigma boundaries used for Held-Suarez (same as SPEEDY 8-layer for consistency)
DEFAULT_SIGMA_BOUNDARIES = jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0])


def get_held_suarez_coords(layers=8, spectral_truncation=31, nodal_shape=None, sigma_boundaries=None, spmd_mesh=None) -> CoordinateSystem:
    """Create a CoordinateSystem for Held-Suarez physics.

    Held-Suarez can work with any sigma levels. By default, uses standard 8-layer boundaries
    for consistency with SPEEDY.

    Args:
        layers: Number of vertical levels (default 8)
        spectral_truncation: Spectral truncation number (default 31)
        nodal_shape: Optional nodal shape (ix, il) to infer spectral_truncation
        sigma_boundaries: Optional array of sigma boundaries. If None, uses DEFAULT_SIGMA_BOUNDARIES.
        spmd_mesh: Optional SPMD mesh for parallelization

    Returns:
        CoordinateSystem object

    """
    if sigma_boundaries is None:
        if layers != 8:
            raise ValueError(f"Default sigma boundaries only defined for 8 layers. Provide explicit sigma_boundaries for {layers} layers.")
        sigma_boundaries = DEFAULT_SIGMA_BOUNDARIES

    return _get_coords(
        sigma_boundaries=sigma_boundaries,
        spectral_truncation=spectral_truncation,
        nodal_shape=nodal_shape,
        spmd_mesh=spmd_mesh
    )
