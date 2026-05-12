"""Diagnose physics tendencies from a prescribed atmospheric state series.

``PrescribedStateModel`` computes physics tendencies for each timestep
*independently* using ``vmap`` — there is no carry, no scan, and no tracer
evolution. Useful for offline diagnostics, validation against reference
data, and building lookup tables. For tracer evolution (where each step
depends on the previous one), use ``SingleColumnModel`` instead.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tree_math
from jax.tree_util import tree_map

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.forcing import ForcingData, default_forcing
from jcm.physics_interface import (
    Physics,
    PhysicsState,
    PhysicsTendency,
    verify_state,
)
from jcm.terrain import TerrainData


@tree_math.struct
class PrescribedStatePredictions:
    """Container for ``PrescribedStateModel.run`` outputs.

    Attributes:
        states: The prescribed atmospheric state time series.
        tendencies: Physics tendencies for each step.
        physics_data: Per-step diagnostics dict from the physics package.
        times: Times in days since the start of the run.

    """

    states: PhysicsState
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any


class PrescribedStateModel:
    """Compute physics tendencies for a prescribed state time series.

    Args:
        physics: Physics package whose ``compute_tendencies`` is called per step.
        coords: ``CoordinateSystem`` used for grids and ``physics.cache_coords``.
        terrain: Optional ``TerrainData`` boundary conditions; defaults to the
            aquaplanet derived from ``coords``.
        dt_seconds: Physics timestep in seconds (default 1800).

    """

    def __init__(
        self,
        physics: Physics,
        coords: CoordinateSystem,
        terrain: TerrainData | None = None,
        dt_seconds: float = 1800.0,
    ) -> None:
        """Initialise (see class docstring for argument descriptions)."""
        self.physics = physics
        self.coords = coords
        self.terrain = terrain if terrain is not None else TerrainData.aquaplanet(coords)
        self.dt_seconds = float(dt_seconds)
        self.physics.cache_coords(coords)
        # Hand the timestep down to the composable-physics container so its
        # terms read a single ``dt`` source — mirrors the wiring in ``Model``
        # and ``SingleColumnModel``.
        if hasattr(self.physics, "dt_seconds"):
            self.physics.dt_seconds = self.dt_seconds

    def run(
        self,
        states: list[PhysicsState] | PhysicsState,
        forcing: ForcingData | None = None,
        times: jnp.ndarray | None = None,
    ) -> PrescribedStatePredictions:
        """Compute physics tendencies for each prescribed state.

        Args:
            states: List of ``PhysicsState`` snapshots, or a single
                ``PhysicsState`` whose leading axis is time.
            forcing: Surface forcing; defaults to aquaplanet from ``coords``.
            times: Optional days-since-start array.

        Returns:
            ``PrescribedStatePredictions``.

        """
        if forcing is None:
            forcing = default_forcing(self.coords.horizontal)

        if isinstance(states, list):
            states = tree_map(lambda *a: jnp.stack(a, axis=0), *states)

        n_times = states.u_wind.shape[0]
        if times is None:
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)

        physics = self.physics
        terrain = self.terrain

        def step(state):
            clamped = verify_state(state)
            return physics.compute_tendencies(clamped, forcing, terrain)

        @jax.jit
        def vmapped():
            return jax.vmap(step)(states)

        tendencies, physics_data = vmapped()
        return PrescribedStatePredictions(
            states=states,
            tendencies=tendencies,
            physics_data=physics_data,
            times=times,
        )
