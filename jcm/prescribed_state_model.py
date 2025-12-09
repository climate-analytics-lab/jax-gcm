"""
Prescribed State Model for diagnosing physics tendencies without dynamics.

This module provides a model class that computes physics tendencies for a
prescribed atmospheric state time series without running the dynamical core.
Useful for:
- Debugging physics parameterizations
- Offline physics diagnostics
- Validating physics implementations against reference data
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
import tree_math
from typing import Callable, Any, Optional, Union
import jax_datetime as jdt

from jcm.geometry import Geometry
from jcm.date import DateData
from jcm.forcing import ForcingData, default_forcing
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics, verify_state
from jcm.physics.speedy.speedy_physics import SpeedyPhysics


@tree_math.struct
class PrescribedStatePredictions:
    """Container for prescribed state model outputs.

    Attributes:
        states: The input PhysicsState time series (for reference).
        tendencies: Physics tendencies computed for each state.
        physics_data: Diagnostic physics data computed for each state.
        times: Timestamps of each state.
    """
    states: PhysicsState
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any

    def to_xarray(self, physics_module: Physics = None, coords=None):
        """Convert predictions to xarray.Dataset.

        Args:
            physics_module: Physics module used (for parsing physics fields).
            coords: Optional coordinate system.

        Returns:
            xarray.Dataset with states, tendencies, and physics diagnostics.
        """
        from dinosaur.xarray_utils import data_to_xarray
        from jcm.utils import get_coords
        from jcm.physics.icon.icon_physics import IconPhysics
        from numpy import timedelta64
        import pandas as pd
        from jcm.utils import DYNAMICS_UNITS_TABLE_CSV_PATH

        physics_module = physics_module or SpeedyPhysics()

        # Get coordinates from state shape
        nodal_shape = self.states.u_wind.shape[1:]  # (time, level, lat, lon) -> (level, lat, lon)
        coords = coords or get_coords(layers=nodal_shape[0], nodal_shape=nodal_shape[1:])

        # Convert states to dict
        states_dict = {f"state_{k}": v for k, v in self.states.asdict().items() if k != 'tracers'}
        if self.states.tracers:
            for k, v in self.states.tracers.items():
                states_dict[f"state_tracer_{k}"] = v

        # Convert tendencies to dict
        tendencies_dict = {f"tendency_{k}": v for k, v in self.tendencies.asdict().items() if k != 'tracers'}
        if self.tendencies.tracers:
            for k, v in self.tendencies.tracers.items():
                tendencies_dict[f"tendency_tracer_{k}"] = v

        # Convert physics data to dict
        physics_dict = physics_module.data_struct_to_dict(
            self.physics_data,
            geometry=Geometry.from_coords(coords, hybrid_vertical=isinstance(physics_module, IconPhysics))
        )

        # Combine all data
        all_data = states_dict | tendencies_dict | physics_dict

        times = jax.device_get(self.times)
        coords = jax.device_get(coords)

        pred_ds = data_to_xarray(
            all_data,
            coords=coords,
            serialize_coords_to_attrs=False,
            times=times - times[0]
        )

        # Import units
        units_df = pd.read_csv(DYNAMICS_UNITS_TABLE_CSV_PATH)
        if physics_module.UNITS_TABLE_CSV_PATH is not None:
            units_df = pd.concat([units_df, pd.read_csv(physics_module.UNITS_TABLE_CSV_PATH)], ignore_index=True)

        for var, unit, desc in zip(units_df["Variable"], units_df["Units"], units_df["Description"]):
            # Check for state_ and tendency_ prefixed versions
            for prefix in ["", "state_", "tendency_"]:
                prefixed_var = prefix + var
                if prefixed_var in pred_ds:
                    pred_ds[prefixed_var].attrs["units"] = unit
                    pred_ds[prefixed_var].attrs["description"] = desc

        # Flip vertical dimension
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # Convert time
        pred_ds['time'] = (
            times * (timedelta64(1, 'D') / timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')

        return pred_ds


class PrescribedStateModel:
    """Model for computing physics tendencies from prescribed atmospheric states.

    This model takes a time series of atmospheric states and computes physics
    tendencies without running the dynamical core. Useful for debugging physics, 
    and offline diagnostics.

    Example usage:
        >>> model = PrescribedStateModel(physics=IconPhysics(), geometry=geometry)
        >>> # Load states from netCDF
        >>> states = model.load_states_from_xarray(ds)
        >>> # Or create states programmatically
        >>> states = [PhysicsState(...), PhysicsState(...)]
        >>> # Compute physics tendencies
        >>> predictions = model.run(states, forcing=forcing)
    """

    def __init__(
        self,
        physics: Physics = None,
        geometry: Geometry = None,
        use_hybrid_coords: bool = None,
        start_date: jdt.Datetime = jdt.to_datetime('2000-01-01'),
        dt_seconds: float = 1800.0,
    ) -> None:
        """Initialize the prescribed state model.

        Args:
            physics: Physics module to use for computing tendencies.
            geometry: Geometry object describing the model grid.
            use_hybrid_coords: Whether to use hybrid vertical coordinates.
                If None, auto-detected from physics type.
            start_date: Start date for the simulation (for date-dependent physics).
            dt_seconds: Timestep in seconds for physics calculations (default 1800s = 30min).
        """
        self.physics = physics or SpeedyPhysics()

        # Auto-detect coordinate system based on physics type
        if use_hybrid_coords is None:
            from jcm.physics.icon import IconPhysics
            use_hybrid_coords = isinstance(self.physics, IconPhysics)
        self.use_hybrid_coords = use_hybrid_coords

        self.geometry = geometry
        self.start_date = start_date
        self.dt_seconds = dt_seconds

    def _date_from_time_index(self, time_index) -> DateData:
        """Create DateData for a given time index.

        Note: time_index can be a JAX array for traceability inside vmap/scan.
        """
        sim_time_seconds = time_index * self.dt_seconds
        # Use jnp.round().astype() to keep values traceable in JAX transformations
        seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(seconds=seconds_int),
            model_step=jnp.asarray(time_index).astype(jnp.int32),
            dt_seconds=self.dt_seconds
        )

    def _compute_tendencies_single(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        geometry: Geometry,
        date: DateData,
    ) -> tuple[PhysicsTendency, Any]:
        """Compute physics tendencies for a single state.

        Args:
            state: Atmospheric state.
            forcing: Surface forcing data.
            geometry: Model geometry.
            date: Date/time for the computation.

        Returns:
            Tuple of (tendencies, physics_data).
        """
        # Verify and clamp state to physical bounds
        clamped_state = verify_state(state)

        # Compute physics tendencies
        tendencies, physics_data = self.physics.compute_tendencies(
            clamped_state, forcing, geometry, date
        )

        return tendencies, physics_data

    def run(
        self,
        states: Union[list[PhysicsState], PhysicsState],
        forcing: ForcingData = None,
        times: jnp.ndarray = None,
    ) -> PrescribedStatePredictions:
        """Compute physics tendencies for a time series of prescribed states.

        Args:
            states: Either a list of PhysicsState objects, or a single PhysicsState
                with an extra leading time dimension (shape: [time, level, lat, lon]).
            forcing: Surface forcing data. If None, uses default aquaplanet forcing.
            times: Optional array of times in days. If None, uses integer indices
                multiplied by dt_seconds.

        Returns:
            PrescribedStatePredictions containing states, tendencies, and diagnostics.
        """
        if self.geometry is None:
            raise ValueError("Geometry must be set before calling run(). "
                           "Either pass geometry to __init__ or set model.geometry.")

        # Handle forcing
        if forcing is None:
            from jcm.utils import get_coords
            nlev, nlon, nlat = self.geometry.nodal_shape
            coords = get_coords(
                layers=nlev,
                nodal_shape=(nlon, nlat),
                hybrid_vertical=self.use_hybrid_coords
            )
            forcing = default_forcing(coords.horizontal)

        # Convert list of states to stacked PhysicsState if needed
        if isinstance(states, list):
            states = self._stack_states(states)

        # Get number of timesteps from the state
        n_times = states.u_wind.shape[0]

        # Generate times if not provided
        if times is None:
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)  # Convert to days

        # Compute tendencies for all states using vmap
        @jax.jit
        def compute_all_tendencies(states_stacked):
            def single_step(time_idx):
                # Extract single state from stacked states
                state = tree_map(lambda x: x[time_idx], states_stacked)
                date = self._date_from_time_index(time_idx)
                return self._compute_tendencies_single(state, forcing, self.geometry, date)

            # Use vmap over time indices
            all_tendencies, all_physics_data = jax.vmap(
                lambda idx: single_step(idx)
            )(jnp.arange(n_times))

            return all_tendencies, all_physics_data

        tendencies, physics_data = compute_all_tendencies(states)

        # Reshape ICON physics data if needed
        from jcm.physics.icon import IconPhysics
        if isinstance(self.physics, IconPhysics):
            # Reshape each timestep's physics data from column to 3D format
            physics_data = jax.vmap(
                lambda pd: self.physics.reshape_physics_data_to_3d(pd, self.geometry)
            )(physics_data)

        return PrescribedStatePredictions(
            states=states,
            tendencies=tendencies,
            physics_data=physics_data,
            times=times,
        )

    def run_single(
        self,
        state: PhysicsState,
        forcing: ForcingData = None,
        time_index: int = 0,
    ) -> tuple[PhysicsTendency, Any]:
        """Compute physics tendencies for a single state.

        Convenience method for computing tendencies for just one state.

        Args:
            state: Single atmospheric state.
            forcing: Surface forcing data.
            time_index: Time index for date calculation (default 0).

        Returns:
            Tuple of (tendencies, physics_data).
        """
        if self.geometry is None:
            raise ValueError("Geometry must be set before calling run_single().")

        if forcing is None:
            from jcm.utils import get_coords
            nlev, nlon, nlat = self.geometry.nodal_shape
            coords = get_coords(
                layers=nlev,
                nodal_shape=(nlon, nlat),
                hybrid_vertical=self.use_hybrid_coords
            )
            forcing = default_forcing(coords.horizontal)

        date = self._date_from_time_index(time_index)
        tendencies, physics_data = self._compute_tendencies_single(
            state, forcing, self.geometry, date
        )

        # Reshape ICON physics data if needed
        from jcm.physics.icon import IconPhysics
        if isinstance(self.physics, IconPhysics):
            physics_data = self.physics.reshape_physics_data_to_3d(physics_data, self.geometry)

        return tendencies, physics_data

    def _stack_states(self, states: list[PhysicsState]) -> PhysicsState:
        """Stack a list of PhysicsState objects into a single PhysicsState with time dimension."""
        return tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *states)

    # Backward-compatible static method aliases to shared utilities
    load_states_from_xarray = staticmethod(__import__('jcm.utils', fromlist=['load_states_from_xarray']).load_states_from_xarray)
    create_single_column_state = staticmethod(__import__('jcm.utils', fromlist=['create_single_column_state']).create_single_column_state)
