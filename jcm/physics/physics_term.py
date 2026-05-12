"""Base class for composable physics parameterization terms.

A PhysicsTerm is a single physics parameterization (e.g. radiation, convection)
that can be composed with other terms to build a full physics package. Terms
communicate through a ``diagnostics`` dict that flows forward through the term
list, replacing the physics-package-specific PhysicsData structs.

See docs/design/composable_physics.md for the full design.

Date: 2026-04-12
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData


@dataclasses.dataclass(frozen=True)
class TracerSpec:
    """Declares a tracer that a PhysicsTerm reads or writes.

    Model aggregates specs from every term at build time and seeds the
    initial state's tracer dict with ``initial_value`` for any tracer
    whose name is declared here and not already present.

    ``nondimensionalize=False`` means the state/tendency converters in
    physics_interface pass the tracer through untouched (no gram/kg
    scaling). Use this for tracers that already carry no unit expressible
    as a mixing ratio — e.g. number concentrations per kg of air.

    Attributes:
        name: key in ``state.tracers`` (also on the dynamics side).
        units: human-readable units, informational only.
        initial_value: fill value used when seeding the initial tracer dict.
        nondimensionalize: whether to apply the standard gram/kg
            nondimensionalization when converting between physics and
            dynamics representations.

    """

    name: str
    units: str = "kg/kg"
    initial_value: float = 0.0
    nondimensionalize: bool = True


class PhysicsTerm(nnx.Module):
    """Base class for a composable physics parameterization.

    Subclasses must:
      - Declare ``name``, ``category``, ``requires``, ``provides`` as
        ``ClassVar`` — static metadata, not pytree leaves.
      - Type-annotate any Parameters-dataclass kwargs on ``__init__``
        (e.g. ``params: ConvectionParameters | None = None``). The
        Hydra-driven ``runners.build_physics`` introspects these
        annotations to resolve YAML overrides into ``Parameters``
        instances before calling ``__init__``. Plain primitive kwargs
        (timescales, level counts, …) need no annotation — they are
        passed through verbatim from the YAML.
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

    # Declarative carry slots. Each entry maps a public ``physics_state``
    # key to a typed sub-struct class with a ``.zeros((ncols,), nlev)``
    # classmethod. The base ``initial_carry_state`` walks this dict and
    # zero-fills each slot — subclasses just declare what they own and
    # don't reimplement the shape extraction.
    #
    # Override ``initial_carry_state`` directly only when zero is the
    # wrong seed (e.g. ``TteTkeVerticalDiffusion`` floors TKE at ECHAM's
    # 0.01 m²/s² lower bound). Pure plumbing keys (``_date``,
    # ``_forcing_2d``, …) repopulate every step and must NOT appear here.
    carry_slots: ClassVar[dict[str, type]] = {}

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """Declare the tracers this term needs in ``state.tracers``.

        ``ComposablePhysics`` aggregates specs across all terms and
        ``Model`` uses the union to seed the initial state's tracer dict.
        Default is ``()`` — terms that only read ``specific_humidity``
        don't need to override.
        """
        return ()

    def initial_carry_state(self, coords) -> dict[str, Any]:
        """Return this term's initial cross-step carry-state slots.

        Operator-split physics threads a ``PhysicsCarryState`` (a dict
        of typed sub-structs, e.g. ``radiation``,
        ``vertical_diffusion``) from one ``dt`` to the next. Default:
        zero-fill every entry in :attr:`carry_slots`. Terms whose seed
        depends on integration history (e.g. TKE floored at ECHAM's
        0.01 m²/s² lower bound) override and either call ``super`` and
        edit the result, or build the dict directly.

        Subclasses that need cross-step state typically just declare
        :attr:`carry_slots` (a ``ClassVar[dict[str, type]]``); no
        override is needed for the zero case. ``coords`` flows in only
        so the typed sub-struct can size itself — the per-term
        shape-extraction boilerplate lives here once.

        IMPORTANT: do NOT probe ``__call__`` with a zero
        ``PhysicsState`` to discover shapes — zero temperature breaks
        downstream radiation (root cause of the averaged-mode NaN bug
        #470). Either rely on this default + declarative
        :attr:`carry_slots`, or build the slot directly from ``coords``.

        Args:
            coords: model :class:`dinosaur.coordinate_systems.CoordinateSystem`.

        Returns:
            Dict of carry slots this term contributes.

        """
        if not self.carry_slots:
            return {}
        col_shape = (
            coords.horizontal.nodal_shape[0] * coords.horizontal.nodal_shape[1],
        )
        nlev = coords.nodal_shape[0]
        return {
            key: cls.zeros(col_shape, nlev)
            for key, cls in self.carry_slots.items()
        }

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
