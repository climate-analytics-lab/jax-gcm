"""
Single Column Model for evolving physics tracers with prescribed atmospheric state.

This module provides a model class that evolves prognostic physics variables
(like cloud droplets, aerosol tracers) while keeping the large-scale atmospheric
state prescribed. Uses lax.scan for time-stepping since each step depends on
the previous physics state.

Useful for:
- True single-column model simulations
- Testing cloud/aerosol parameterizations
- Process-level studies where dynamics are prescribed from reanalysis
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


@tree_math.struct
class SCMState:
    """State container for Single Column Model.

    Separates prescribed atmospheric fields from prognostic tracers.

    Attributes:
        tracers: Dict of prognostic tracer fields that evolve with physics.
        physics_data: Physics diagnostic data that persists between steps.
    """
    tracers: dict  # Prognostic tracers (qc, qi, aerosols, etc.)
    physics_data: Any  # Physics data from previous step (for persistence)


@tree_math.struct
class SCMPredictions:
    """Container for Single Column Model outputs.

    Attributes:
        prescribed_states: The prescribed atmospheric states (for reference).
        tracer_states: Time series of evolved tracer fields.
        tendencies: Physics tendencies computed at each step.
        physics_data: Diagnostic physics data at each step.
        times: Timestamps of each state.
    """
    prescribed_states: PhysicsState
    tracer_states: dict  # Time series of tracer fields
    relaxed_states: dict  # Time series of relaxed prognostic variables
    tendencies: PhysicsTendency
    physics_data: Any
    times: Any


class SingleColumnModel:
    """Model for evolving physics tracers with prescribed atmospheric state.

    This model takes a time series of prescribed atmospheric states and evolves
    prognostic physics variables (tracers like cloud water, ice, aerosols) using
    lax.scan for proper time-stepping with state carry-over.

    Unlike PrescribedStateModel which uses vmap (independent timesteps),
    this model uses scan so that tracer fields can evolve step-by-step.

    Example usage:
        >>> model = SingleColumnModel(physics=IconPhysics(), geometry=geometry)
        >>> # Load prescribed states from netCDF
        >>> prescribed_states = model.load_states_from_xarray(ds)
        >>> # Initialize tracers
        >>> initial_tracers = {'qc': jnp.zeros(...), 'qi': jnp.zeros(...)}
        >>> # Run with evolving tracers
        >>> predictions = model.run(prescribed_states, initial_tracers=initial_tracers)
    """

    def __init__(
        self,
        physics: Physics,
        geometry: Geometry = None,
        use_hybrid_coords: bool = None,
        start_date: jdt.Datetime = jdt.to_datetime('2000-01-01'),
        dt_seconds: float = 1800.0,
        apply_tracer_tendencies: bool = True,
        relaxation_timescales: dict[str, float] = None,
    ) -> None:
        """Initialize the single column model.

        Args:
            physics: Physics module to use for computing tendencies.
            geometry: Geometry object describing the model grid.
            use_hybrid_coords: Whether to use hybrid vertical coordinates.
                If None, auto-detected from physics type.
            start_date: Start date for the simulation.
            dt_seconds: Timestep in seconds for physics calculations.
            apply_tracer_tendencies: Whether to apply tracer tendencies to
                update tracer fields. If False, tracers are diagnostic only.
            relaxation_timescales: Dictionary mapping variable names (u_wind, v_wind,
                temperature, specific_humidity) to relaxation timescales in seconds.
                Variables in this dict will be treated as prognostic and relaxed towards
                the prescribed state. proper prognostic evolution.
        """
        self.physics = physics
        self.apply_tracer_tendencies = apply_tracer_tendencies
        self.relaxation_timescales = relaxation_timescales or {}

        # Auto-detect coordinate system based on physics type
        if use_hybrid_coords is None:
            from jcm.physics.icon import IconPhysics
            use_hybrid_coords = isinstance(self.physics, IconPhysics)
        self.use_hybrid_coords = use_hybrid_coords

        self.geometry = geometry
        self.start_date = start_date
        self.dt_seconds = dt_seconds

    def _date_from_time_index(self, time_index) -> DateData:
        """Create DateData for a given time index."""
        sim_time_seconds = time_index * self.dt_seconds
        seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
        return DateData.set_date(
            model_time=self.start_date + jdt.Timedelta(seconds=seconds_int),
            model_step=jnp.asarray(time_index).astype(jnp.int32),
            dt_seconds=self.dt_seconds
        )

    def _make_step_fn(
        self,
        forcing: ForcingData,
        geometry: Geometry,
        apply_tendencies: bool,
        dt_seconds: float,
        tracer_names: tuple[str, ...] = None,
        relaxed_var_params: tuple[tuple[str, float], ...] = (),
    ) -> Callable:
        """Create a JIT-compatible step function.

        This separates the step logic from `self` to make it traceable.
        All Python control flow is resolved at function creation time.

        Args:
            forcing: Surface forcing data (static).
            geometry: Model geometry (static).
            apply_tendencies: Whether to apply tracer tendencies.
            dt_seconds: Timestep in seconds.
            tracer_names: Tuple of tracer names (must be provided for JIT).
            relaxed_var_params: Tuple of (name, tau) for relaxed variables.

        Returns:
            A pure function suitable for use in lax.scan.
        """
        physics = self.physics
        start_date = self.start_date

        # Capture tracer names at function creation time (static)
        if tracer_names is None:
            tracer_names = ()

        def compute_date(time_idx):
            """Compute date from time index."""
            sim_time_seconds = time_idx * dt_seconds
            seconds_int = jnp.round(sim_time_seconds).astype(jnp.int32)
            return DateData.set_date(
                model_time=start_date + jdt.Timedelta(seconds=seconds_int),
                model_step=jnp.asarray(time_idx).astype(jnp.int32),
                dt_seconds=dt_seconds
            )

        def step_fn(prescribed_state, tracers, relaxed_vars, physics_data, time_idx):
            """Pure step function for lax.scan."""
            # Combine prescribed state with current tracers and relaxed variables
            # Use relaxed variables where available, otherwise use prescribed state values
            full_state_args = prescribed_state.asdict()
            full_state_args.pop('tracers', None)  # Prepare dict for PhysicsState constructor

            # Override with relaxed variables
            for name, _ in relaxed_var_params:
                full_state_args[name] = relaxed_vars[name]
            
            # Add tracers
            full_state_args['tracers'] = tracers

            # Create the full state object (using the type of the prescribed state, usually PhysicsState)
            full_state = type(prescribed_state)(**full_state_args)

            # Verify and clamp state to physical bounds
            clamped_state = verify_state(full_state)

            # Compute date
            date = compute_date(time_idx)

            # Compute physics tendencies
            tendencies, new_physics_data = physics.compute_tendencies(
                clamped_state, forcing, geometry, date
            )

            # Update tracers - iterate over captured tracer_names (static)
            if apply_tendencies:
                # Build updated tracers dict using static keys
                updated_tracers = {}
                for name in tracer_names:
                    tracer = tracers[name]
                    # Get tendency, defaulting to zeros if not present
                    tracer_tend = tendencies.tracers.get(name, jnp.zeros_like(tracer))
                    # Forward Euler update with non-negativity
                    updated_tracers[name] = jnp.maximum(
                        tracer + dt_seconds * tracer_tend, 0.0
                    )
            else:
                updated_tracers = tracers

            # Update relaxed variables
            updated_relaxed_vars = {}
            for name, tau in relaxed_var_params:
                current_val = relaxed_vars[name]
                target_val = getattr(prescribed_state, name)
                phys_tend = getattr(tendencies, name)
                
                # Nudging tendency: relaxed towards prescribed state
                nudging_tend = (target_val - current_val) / tau
                
                # Forward Euler update
                # Note: Physics tendency + Nudging tendency
                updated_relaxed_vars[name] = current_val + dt_seconds * (phys_tend + nudging_tend)

            return tendencies, updated_tracers, updated_relaxed_vars, new_physics_data

        return step_fn

    def run(
        self,
        prescribed_states: Union[list[PhysicsState], PhysicsState],
        forcing: ForcingData = None,
        initial_tracers: dict = None,
        initial_physics_data: Any = None,
        times: jnp.ndarray = None,
        initial_relaxed_vars: dict = None,
    ) -> SCMPredictions:
        """Run the single column model with evolving tracers.

        Args:
            prescribed_states: Either a list of PhysicsState objects, or a single
                PhysicsState with an extra leading time dimension.
            forcing: Surface forcing data. If None, uses default aquaplanet forcing.
            initial_tracers: Initial tracer fields. If None, uses tracers from
                the first prescribed state (or empty dict).
            initial_physics_data: Initial physics data. If None, initialized to zeros.
            times: Optional array of times in days.
            initial_relaxed_vars: Initial values for relaxed prognostic variables.
                If None, initialized from the first prescribed state.

        Returns:
            SCMPredictions containing prescribed states, evolved tracers,
            tendencies, and diagnostics.
        """
        if self.geometry is None:
            raise ValueError("Geometry must be set before calling run().")

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
        if isinstance(prescribed_states, list):
            prescribed_states = self._stack_states(prescribed_states)

        # Get dimensions
        n_times = prescribed_states.u_wind.shape[0]

        # Initialize tracers
        if initial_tracers is None:
            # Try to get from first prescribed state
            first_tracers = tree_map(lambda x: x[0], prescribed_states.tracers)
            if first_tracers:
                initial_tracers = first_tracers
            else:
                initial_tracers = {}

        # Prepare relaxed variables configuration
        relaxed_var_params = tuple(sorted(self.relaxation_timescales.items()))
        
        # Initialize relaxed variables
        if initial_relaxed_vars is None:
            # Get first state to initialize relaxed variables
            first_state_slice = tree_map(lambda x: x[0], prescribed_states)
            initial_relaxed_vars = {}
            for name, _ in relaxed_var_params:
                # We assume the prescribed state has these fields (since it's PhysicsState)
                initial_relaxed_vars[name] = getattr(first_state_slice, name)

        # Initialize physics data
        # We need to run one step to get the proper PhysicsData structure
        # for lax.scan (pytree structure must match between input and output)
        if initial_physics_data is None:
            # Get first state to initialize physics data structure
            first_state = tree_map(lambda x: x[0], prescribed_states)
            
            # Update with initial relaxed variables if present
            # Note: We create a state that reflects the initial condition
            state_args = first_state.asdict()
            state_args.pop('tracers', None)
            for name, val in initial_relaxed_vars.items():
                state_args[name] = val
            state_args['tracers'] = initial_tracers
            
            # Reconstruct PhysicsState (or whatever type first_state is)
            first_state_combined = type(first_state)(**state_args)
            first_date = self._date_from_time_index(0)

            # Run one step to get physics data structure
            clamped_state = verify_state(first_state_combined)
            _, initial_physics_data = self.physics.compute_tendencies(
                clamped_state, forcing, self.geometry, first_date
            )

        # Generate times if not provided
        if times is None:
            times = jnp.arange(n_times) * (self.dt_seconds / 86400.0)

        # Create the step function (resolves Python control flow)
        step_fn = self._make_step_fn(
            forcing=forcing,
            geometry=self.geometry,
            apply_tendencies=self.apply_tracer_tendencies,
            dt_seconds=self.dt_seconds,
            tracer_names=tuple(initial_tracers.keys()),
            relaxed_var_params=relaxed_var_params,
        )

        # Define scan function using the pure step_fn
        def scan_step(carry, time_idx):
            tracers, relaxed_vars, physics_data = carry

            # Extract prescribed state for this timestep
            prescribed_state = tree_map(lambda x: x[time_idx], prescribed_states)
            # Remove tracers from prescribed state (we manage them separately)
            prescribed_state = prescribed_state.copy(tracers={})

            tendencies, new_tracers, new_relaxed_vars, new_physics_data = step_fn(
                prescribed_state, tracers, relaxed_vars, physics_data, time_idx
            )

            return (new_tracers, new_relaxed_vars, new_physics_data), (tendencies, new_tracers, new_relaxed_vars, new_physics_data)

        # Run scan with JIT
        initial_carry = (initial_tracers, initial_relaxed_vars, initial_physics_data)

        @jax.jit
        def run_scan():
            _, (all_tendencies, all_tracers, all_relaxed_vars, all_physics_data) = lax.scan(
                scan_step,
                initial_carry,
                jnp.arange(n_times)
            )
            return all_tendencies, all_tracers, all_relaxed_vars, all_physics_data

        tendencies, tracer_history, relaxed_vars_history, physics_data_history = run_scan()

        # Reshape ICON physics data if needed
        from jcm.physics.icon import IconPhysics
        if isinstance(self.physics, IconPhysics):
            physics_data_history = jax.vmap(
                lambda pd: self.physics.reshape_physics_data_to_3d(pd, self.geometry)
            )(physics_data_history)

        return SCMPredictions(
            prescribed_states=prescribed_states,
            tracer_states=tracer_history,
            relaxed_states=relaxed_vars_history,
            tendencies=tendencies,
            physics_data=physics_data_history,
            times=times,
        )

    def run_single(
        self,
        prescribed_state: PhysicsState,
        forcing: ForcingData = None,
        tracers: dict = None,
        physics_data: Any = None,
        time_index: int = 0,
    ) -> tuple[PhysicsTendency, dict, Any]:
        """Compute one physics step.

        Convenience method for single-step computation.

        Args:
            prescribed_state: Single prescribed atmospheric state.
            forcing: Surface forcing data.
            tracers: Current tracer fields (or None for empty).
            physics_data: Physics data from previous step (or None for zeros).
            time_index: Time index for date calculation.

        Returns:
            Tuple of (tendencies, updated_tracers, updated_physics_data).
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

        if tracers is None:
            tracers = prescribed_state.tracers if prescribed_state.tracers else {}

        if physics_data is None:
            # Initialize by running physics once to get proper structure
            temp_state = prescribed_state.copy(tracers=tracers)
            clamped_state = verify_state(temp_state)
            date = self._date_from_time_index(time_index)
            _, physics_data = self.physics.compute_tendencies(
                clamped_state, forcing, self.geometry, date
            )

        # Create step function and run single step
        step_fn = self._make_step_fn(
            forcing=forcing,
            geometry=self.geometry,
            apply_tendencies=self.apply_tracer_tendencies,
            dt_seconds=self.dt_seconds,
            tracer_names=tuple(tracers.keys()),
        )

        tendencies, updated_tracers, updated_physics_data = step_fn(
            prescribed_state.copy(tracers={}), tracers, physics_data, time_index
        )

        # Reshape ICON physics data if needed
        from jcm.physics.icon import IconPhysics
        if isinstance(self.physics, IconPhysics):
            updated_physics_data = self.physics.reshape_physics_data_to_3d(
                updated_physics_data, self.geometry
            )

        return tendencies, updated_tracers, updated_physics_data

    def _stack_states(self, states: list[PhysicsState]) -> PhysicsState:
        """Stack a list of PhysicsState objects into a single PhysicsState with time dimension."""
        return tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *states)

    # Backward-compatible static method aliases to shared utilities
    create_initial_tracers = staticmethod(__import__('jcm.utils', fromlist=['create_initial_tracers']).create_initial_tracers)
    load_states_from_xarray = staticmethod(__import__('jcm.utils', fromlist=['load_states_from_xarray']).load_states_from_xarray)
    create_single_column_state = staticmethod(__import__('jcm.utils', fromlist=['create_single_column_state']).create_single_column_state)
