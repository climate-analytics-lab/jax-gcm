"""Base class for composable physics parameterization terms.

A PhysicsTerm is a single physics parameterization (e.g. radiation, convection)
that can be composed with other terms to build a full physics package. Terms
communicate through a ``diagnostics`` dict that flows forward through the term
list, replacing the physics-package-specific PhysicsData structs.

See docs/design/composable_physics.md for the full design.

Date: 2026-04-12
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


class PhysicsTerm(nnx.Module):
    """Base class for a composable physics parameterization.

    Subclasses must:
      - Declare ``name``, ``category``, ``requires``, ``provides`` as
        ``ClassVar`` — static metadata, not pytree leaves.
      - Store tunable parameters as ``nnx.Param`` attributes so that
        gradients flow through them.
      - Store coordinate-dependent caches as ``nnx.Variable`` attributes
        (traced but not trainable by default).
      - Implement ``__call__`` and optionally ``cache_coords``.
    """

    name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def cache_coords(self, coords) -> None:
        """Populate coordinate-dependent cached state (in-place).

        Called once at Model construction time, outside any jitted region.
        Override in subclasses that need precomputed coordinate data.
        """

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict[str, jnp.ndarray],
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict[str, jnp.ndarray]]:
        """Compute tendencies and update diagnostics.

        Args:
            state: Current atmospheric state on the grid.
            diagnostics: Dict of intermediate diagnostic arrays produced
                by upstream terms. Must be updated functionally (return a
                new dict, no in-place mutation).
            forcing: Boundary condition forcing data.
            terrain: Terrain boundary conditions.

        Returns:
            A tuple of (tendencies, updated_diagnostics).

        """
        raise NotImplementedError

    def __add__(self, other):
        """Compose two terms (or a term and a ComposablePhysics).

        Returns a ComposablePhysics instance.
        """
        from jcm.physics.composable_physics import ComposablePhysics

        self_terms = [self]
        if hasattr(other, 'terms'):
            # ComposablePhysics or similar
            other_terms = list(other.terms)
        elif hasattr(other, 'category') and callable(other):
            # Another PhysicsTerm
            other_terms = [other]
        else:
            return NotImplemented
        return ComposablePhysics(terms=self_terms + other_terms)

    def __radd__(self, other):
        """Support sum() by handling 0 + PhysicsTerm."""
        if other == 0:
            # Support sum([term1, term2, ...]) which starts with 0 + term1
            from jcm.physics.composable_physics import ComposablePhysics
            return ComposablePhysics(terms=[self])
        return NotImplemented
