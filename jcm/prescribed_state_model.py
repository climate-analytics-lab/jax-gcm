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
import jax_datetime as jdt
import tree_math
from jax.tree_util import tree_map

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.date import DateData
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
        times: Times in days since ``start_date``.

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
        start_date: Starting date for time-series indexing (default 2000-01-01).
        dt_seconds: Physics timestep in seconds (default 1800).

    """

    def __init__(
        self,
        physics: Physics,
        coords: CoordinateSystem,
        terrain: TerrainData | None = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
        dt_seconds: float = 1800.0,
    ) -> None:
        """Initialise (see class docstring for argument descriptions)."""
        self.physics = physics
        self.coords = coords
        self.terrain = terrain if terrain is not None else TerrainData.aquaplanet(coords)
        self.start_date = start_date
        self.dt_seconds = float(dt_seconds)
        self.physics.cache_coords(coords)
        from jcm.physics.icon.icon_terms import ComposableIconPhysics
        if isinstance(self.physics, ComposableIconPhysics):
            self.physics.apply_timestep(self.dt_seconds)

    def _date_from_time_index(self, time_index) -> DateData:
        sim_time_seconds = time_index * self.dt_seconds
        seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(seconds=seconds_int),
            model_step=jnp.asarray(time_index).astype(jnp.int32),
            dt_seconds=self.dt_seconds,
        )

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
        start_date = self.start_date
        dt_seconds = self.dt_seconds

        def step(state, time_idx):
            sim_time_seconds = time_idx * dt_seconds
            seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
            date = DateData.set_date(
                model_time=start_date + jdt.Timedelta(seconds=seconds_int),
                model_step=jnp.asarray(time_idx).astype(jnp.int32),
                dt_seconds=dt_seconds,
            )
            clamped = verify_state(state)
            return physics.compute_tendencies(clamped, forcing, terrain, date)

        @jax.jit
        def vmapped():
            return jax.vmap(step)(states, jnp.arange(n_times))

        tendencies, physics_data = vmapped()
        return PrescribedStatePredictions(
            states=states,
            tendencies=tendencies,
            physics_data=physics_data,
            times=times,
        )
