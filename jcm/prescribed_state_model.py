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

from jcm.date import DEFAULT_CALENDAR, DateData
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

    def to_xarray(self):
        """Dump states + tendencies + physics_data into an ``xr.Dataset``.

        Minimal serialiser for offline diagnostics — does NOT use the
        coord-aware ``data_to_xarray`` path that ``ModelPredictions``
        uses (which would require threading ``coords`` + ``physics``
        through the predictions struct). Variables are emitted with raw
        positional dim names ``time, level, lon, lat`` and no units
        metadata; the intent is to support quick NaN-checks and column
        plots, not production climatology.

        Use ``ModelPredictions.to_xarray()`` when you need the full
        coord-aware serialisation (units, ``additional_coords``,
        flipped level axis, etc.).
        """
        import numpy as np
        import xarray as xr

        # Hardcoded positional dim names matching the prescribed-mode
        # vmap layout: (time, level, lon, lat) for column variables,
        # (time, lon, lat) for surface fields.
        def _dims_for(arr):
            shape = np.shape(arr)
            if len(shape) == 4:
                return ("time", "level", "lon", "lat")
            if len(shape) == 3:
                return ("time", "lon", "lat")
            if len(shape) == 2:
                return ("time", "level")
            if len(shape) == 1:
                return ("time",)
            return tuple(f"dim_{i}" for i in range(len(shape)))

        data_vars: dict[str, tuple] = {}

        def _add(prefix, struct):
            for k, v in struct.asdict().items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        data_vars[f"{prefix}{k}.{sk}"] = (_dims_for(sv), np.asarray(sv))
                else:
                    data_vars[f"{prefix}{k}"] = (_dims_for(v), np.asarray(v))

        _add("state.", self.states)
        _add("tend.", self.tendencies)

        if isinstance(self.physics_data, dict):
            for k, v in self.physics_data.items():
                if k.startswith("_"):
                    continue
                if hasattr(v, "asdict"):
                    _add(f"diag.{k.lstrip('_')}.", v)
                else:
                    arr = np.asarray(v)
                    data_vars[f"diag.{k}"] = (_dims_for(arr), arr)

        return xr.Dataset(
            data_vars=data_vars,
            coords={"time": np.asarray(self.times)},
        )


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
        start_date: jdt.Datetime | None = None,
        calendar: str = DEFAULT_CALENDAR,
    ) -> None:
        """Initialise (see class docstring for argument descriptions).

        ``start_date`` and ``calendar`` mirror :class:`jcm.model.Model` so
        each prescribed state can collapse ``TimeSeries`` forcing leaves
        (sea ice, SST, ozone climatology, ...) to the slice valid at that
        state's ``sim_time`` before physics is evaluated. Without this,
        from-file forcings stay as ``TimeSeries`` structs and physics
        terms that arithmetic-combine them with plain arrays raise
        ``TypeError: non-tree_math.VectorMixin argument is not a
        scalar``.
        """
        self.physics = physics
        self.coords = coords
        self.terrain = terrain if terrain is not None else TerrainData.aquaplanet(coords)
        self.dt_seconds = float(dt_seconds)
        self.start_date = start_date if start_date is not None else jdt.to_datetime("2000-01-01")
        self.calendar = calendar
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
        start_date = self.start_date
        calendar = self.calendar
        dt_seconds = self.dt_seconds

        # ``times`` is days-since-``start_date``; convert to sim_time
        # seconds so ``_date_for`` matches the wiring in
        # ``Model._step_fn``.
        sim_times = jnp.asarray(times) * 86400.0

        def _date_for(sim_time):
            sim_time = jax.lax.stop_gradient(sim_time)
            return DateData.set_date(
                model_time=start_date + jdt.Timedelta(
                    days=jnp.floor(sim_time / 86400).astype(jnp.int32),
                    seconds=jnp.round(sim_time % 86400).astype(jnp.int32),
                ),
                model_step=jnp.int32(sim_time / dt_seconds),
                dt_seconds=dt_seconds,
                calendar=calendar,
            )

        def step(state, sim_time):
            clamped = verify_state(state)
            # Collapse any TimeSeries forcing leaves to the slice valid
            # at this state's sim_time (same wiring as Model._step_fn).
            forcing_now = forcing.select(_date_for(sim_time), calendar=calendar)
            return physics.compute_tendencies(clamped, forcing_now, terrain)

        @jax.jit
        def vmapped():
            return jax.vmap(step)(states, sim_times)

        tendencies, physics_data = vmapped()
        return PrescribedStatePredictions(
            states=states,
            tendencies=tendencies,
            physics_data=physics_data,
            times=times,
        )
