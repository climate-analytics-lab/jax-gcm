import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import tree_math
from flax import nnx
from numpy import timedelta64
import dinosaur
from typing import Callable, Any
from dinosaur import typing
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.boundaries import BoundaryData, default_boundaries, update_boundaries_with_timestep
from jcm.date import DateData, Timestamp, Timedelta
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies, dynamics_state_to_physics_state
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.params import Parameters
from jcm.utils import stack_trees
from jcm.diffusion import DiffusionFilter
import pandas as pd
from datetime import datetime

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

@tree_math.struct
class Predictions:
    dynamics: PhysicsState
    physics: Any
Predictions.__doc__ = """Container for model prediction outputs from a single timestep.
Attributes:
    dynamics (PhysicsState): The physical state variables converted from the
        dynamical state.
    physics (Any): Diagnostic physics data computed by the physics package.
"""

class DiagnosticsCollector(nnx.Module):
    data: nnx.Variable
    i: nnx.Variable
    physical_step: nnx.Variable
    steps_to_average: int

    def __init__(self, steps_to_average):
        self.data = None
        self.i = nnx.Variable(0)
        self.physical_step = nnx.Variable(True)
        self.steps_to_average = steps_to_average

    def accumulate_if_physical_step(self, new_data):
        if self.physical_step.value:
            self.data.value = tree_map(
                lambda stacked_array, new_array: stacked_array.at[self.i.value].add(new_array/self.steps_to_average),
                self.data.value,
                new_data
            )
            self.physical_step.value = False

def averaged_trajectory_from_step(
    step_fn: typing.TimeStepFn,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: typing.PostProcessFn = lambda x: x,
    outer_step_post_process_fn: typing.PostProcessFn = lambda x: x,
) -> Callable[[typing.PyTreeState], tuple[typing.PyTreeState, Any]]:
    """Returns a function that accumulates repeated applications of `step_fn`.
    Compute a trajectory by repeatedly calling `step_fn()`
    `outer_steps * inner_steps` times.
    Args:
        step_fn: function that takes a state and returns state after one time step.
        outer_steps: number of steps to save in the generated trajectory.
        inner_steps: number of repeated calls to step_fn() between saved steps.
        start_with_input: unused, kept to match dinosaur.time_integration.trajectory_from_step API.
        post_process_fn: function to apply to trajectory outputs.
        outer_step_post_process_fn: function to apply to outer step outputs (for post processing that commutes with the time average).
    Returns:
        A function that takes an initial state and returns a tuple consisting of:
        (1) the final frame of the trajectory.
        (2) trajectory of length `outer_steps` representing time evolution (averaged over the inner steps between each outer step).
    """
    def integrate(x_initial, empty_data):
        diagnostics_collector = DiagnosticsCollector(steps_to_average=inner_steps)
        diagnostics_collector.data = nnx.Variable(stack_trees([empty_data] * outer_steps))
        graphdef, init_diag_state = nnx.split(diagnostics_collector)

        empty_sum = tree_map(jnp.zeros_like, x_initial)

        @nnx.scan(in_axes=(nnx.Carry,), out_axes=(nnx.Carry,), length=inner_steps)
        def inner_step(carry):
            x, x_sum, diag_state = carry
            x_sum += x  # include initial state, not final state
            temp_collector_inner = nnx.merge(graphdef, diag_state)
            temp_collector_inner.physical_step.value = True
            x_next = step_fn(temp_collector_inner)(x)
            _, updated_diag_state = nnx.split(temp_collector_inner)
            return (x_next, x_sum, updated_diag_state), (None,)

        @nnx.scan(in_axes=(nnx.Carry,), out_axes=(nnx.Carry,), length=outer_steps)
        def outer_step(carry):
            (x_final, x_sum, diag_state), _ = inner_step(carry)
            temp_collector_outer = nnx.merge(graphdef, diag_state)
            temp_collector_outer.i.value += 1
            _, updated_diag_state = nnx.split(temp_collector_outer)
            return (x_final, empty_sum, updated_diag_state), (x_sum / inner_steps,)
        
        carry = (x_initial, empty_sum, init_diag_state)
        (x_final, _, final_diag_state), (preds,) = outer_step(carry)
        return x_final, Predictions(
            dynamics=preds,
            physics=nnx.merge(graphdef, final_diag_state).data.value
        )

    return integrate

def get_coords(layers=8, horizontal_resolution=31) -> CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """
    try:
        horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f'T{horizontal_resolution}')
    except AttributeError:
        raise ValueError(f"Invalid horizontal resolution: {horizontal_resolution}. Must be one of: 21, 31, 42, 85, 106, 119, 170, 213, 340, or 425.")
    if layers not in sigma_layer_boundaries:
        raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {list(sigma_layer_boundaries.keys())}")

    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=PHYSICS_SPECS.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(sigma_layer_boundaries[layers])
    )

class Model:
    """
    Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet.

    #TODO: Factor out the geography and physics choices so you can choose independent of each other.
    """

    def __init__(self, time_step=30.0, save_interval=10.0, total_time=1200,
                 start_date=None, layers=8, horizontal_resolution=31,
                 coords: CoordinateSystem=None, boundaries: BoundaryData=None,
                 initial_state: PhysicsState=None, physics: Physics=None,
                 diffusion: DiffusionFilter=None, output_averages=False) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step:
                Model time step in minutes
            save_interval:
                Save interval in days
            total_time:
                Total integration time in days
            start_date:
                Start date of the simulation
            layers:
                Number of vertical layers
            horizontal_resolution:
                Horizontal resolution of the model (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425)
            coords:
                CoordinateSystem object describing model grid
            boundaries:
                BoundaryData object describing surface boundary conditions
            initial_state:
                Initial state of the model (PhysicsState object), optional
            physics:
                Physics object describing the model physics
            diffusion:
                DiffusionFilter object describing horizontal diffusion filter params
            output_averages:
                Whether to output time-averaged quantities
        """
        # Integration settings
        self.start_date = start_date or Timestamp.from_datetime(datetime(2000, 1, 1))
        self.save_interval = save_interval * units.day
        self.total_time = total_time * units.day
        self.time_step = time_step
        dt_si = self.time_step * units.minute

        self.physics_specs = PHYSICS_SPECS

        if coords is not None:
            self.coords = coords
            horizontal_resolution = coords.horizontal.total_wavenumbers - 2
        else:
            self.coords = get_coords(layers=layers, horizontal_resolution=horizontal_resolution)
        self.geometry = Geometry.from_coords(self.coords)

        self.inner_steps = int(self.save_interval.to(units.minute) / dt_si)
        self.outer_steps = int(self.total_time / self.save_interval)
        self.dt = self.physics_specs.nondimensionalize(dt_si)

        if initial_state is not None:
            self.initial_state = initial_state
        else:
            self.initial_state = None

        # Get the reference temperature and orography. This also returns the initial state function (if wanted to start from rest)
        self.default_state_fn, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self.physics_specs,
            p0=p0*units.pascal,
        )
        
        self.ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        
        self.physics = physics or SpeedyPhysics()

        self.diffusion = diffusion or DiffusionFilter.default()

        # TODO: make the truncation number a parameter consistent with the grid shape
        params_for_boundaries = (self.physics.parameters
                                 if (hasattr(self.physics, 'parameters') and isinstance(self.physics.parameters, Parameters))
                                 else Parameters.default())
        if boundaries is None:
            truncated_orography = primitive_equations.truncated_modal_orography(aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords, wavenumbers_to_clip=2)
            self.boundaries = default_boundaries(self.coords.horizontal, aux_features[dinosaur.xarray_utils.OROGRAPHY], params_for_boundaries)
        else:
            self.boundaries = update_boundaries_with_timestep(boundaries, params_for_boundaries, dt_si)
            truncated_orography = primitive_equations.truncated_modal_orography(self.boundaries.orog, self.coords, wavenumbers_to_clip=2)
        
        self.primitive = primitive_equations.PrimitiveEquations(
            reference_temperature=self.ref_temps,
            orography=truncated_orography, 
            coords=self.coords,
            physics_specs=self.physics_specs,
        )
        
        physics_forcing_eqn = lambda d: ExplicitODE.from_functions(lambda state:
            get_physical_tendencies(
                state=state,
                dynamics=self.primitive,
                time_step=time_step,
                physics=self.physics,
                boundaries=self.boundaries,
                geometry=self.geometry,
                diffusion=self.diffusion,
                date = DateData.set_date(
                    model_time = self.start_date + Timedelta(seconds=state.sim_time),
                    model_step = ((state.sim_time/60) / time_step).astype(jnp.int32),
                    dt_seconds = jnp.float32(time_step * 60.0)
                ),
                diagnostics_collector=d
            )
        )

        # Define trajectory times, expects start_with_input=False
        self.times = self.save_interval * jnp.arange(1, self.outer_steps+1)
        
        self.primitive_with_speedy = lambda d: dinosaur.time_integration.compose_equations([self.primitive, physics_forcing_eqn(d)])
        step_fn_unfiltered = lambda d: dinosaur.time_integration.imex_rk_sil3(self.primitive_with_speedy(d), self.dt)
        
        def conserve_global_mean_surface_pressure(u, u_next):
            return u_next.replace(
                # prevent global mean (0th spectral component) surface pressure drift by setting it to its value before timestep
                log_surface_pressure=u_next.log_surface_pressure.at[0, 0, 0].set(u.log_surface_pressure[0, 0, 0])
            )
        
        filters = [
            conserve_global_mean_surface_pressure,
            dinosaur.time_integration.horizontal_diffusion_step_filter(
                self.coords.horizontal, self.dt, tau=self.diffusion.state_diff_timescale, order=self.diffusion.state_diff_order),
        ]

        step_fn = lambda d=None: dinosaur.time_integration.step_with_filters(step_fn_unfiltered(d), filters)
        
        self.output_averages = output_averages
        self.step_fn = step_fn if self.output_averages else step_fn()
        
    def get_initial_state(self, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> primitive_equations.State:
        """Generates an initial state for a simulation.

        Args:
            random_seed: 
                Seed for the JAX random number generator.
            sim_time: 
                The starting simulation time for the state.
            humidity_perturbation: 
                If True and using the default state, adds a small amount of specific humidity.
        
        Returns:
            A `primitive_equations.State` object ready for integration.
        """
        from jcm.physics_interface import physics_state_to_dynamics_state

        # Either use the designated initial state, or generate one. The initial state to the dycore is a modal primitive_equations.State,
        # but the optional initial state from the user is a nodal PhysicsState
        if self.initial_state is not None:
            state = physics_state_to_dynamics_state(self.initial_state, self.primitive)
        else:
            state = self.default_state_fn(jax.random.PRNGKey(random_seed))
            # default state returns log surface pressure, we want it to be log(normalized_surface_pressure)
            # there are several ways to do this operation (in modal vs nodal space, with log vs absolute pressure), this one has the least error
            state.log_surface_pressure = self.coords.horizontal.to_modal(
                self.coords.horizontal.to_nodal(state.log_surface_pressure) - jnp.log(self.physics_specs.nondimensionalize(p0 * units.pascal)) # Makes this robust to different physics_specs, which will change default_state_fn behavior
            )

            # need to add specific humidity as a tracer
            state.tracers = {
                'specific_humidity': (1e-2 if humidity_perturbation else 0.0) * primitive_equations_states.gaussian_scalar(self.coords, self.physics_specs)
            }
        return primitive_equations.State(**state.asdict(), sim_time=sim_time)
    
    def _post_process(self, state: primitive_equations.State) -> Predictions:
        """Post-processes a single state from the simulation trajectory. This function is called by the integrator at each save point. It converts the dynamical state to a physical state and, if enabled, runs the physics package to compute diagnostic variables.
        
        Args:
            state: 
                A `primitive_equations.State` object from the simulation.
        
        Returns:
            A dictionary containing the `PhysicsState` ('dynamics') and the
            diagnostic physics variables (data structure determined by model.physics).
        """
        from jcm.physics_interface import dynamics_state_to_physics_state, verify_state
        from jcm.physics_interface import verify_state

        predictions = Predictions(
            dynamics=dynamics_state_to_physics_state(state, self.primitive),
            physics=None
        )
            
        if self.physics.write_output:
            date=DateData.set_date(
                model_time = self.start_date + Timedelta(seconds=state.sim_time),
                model_step = ((state.sim_time/60) / self.time_step).astype(jnp.int32),
                dt_seconds = jnp.float32(self.time_step * 60.0)
            )
            clamped_physics_state = verify_state(predictions.dynamics)
            _, physics_data = self.physics.compute_tendencies(clamped_physics_state, self.boundaries, self.geometry, date)
            predictions = predictions.replace(physics=physics_data)

        return predictions
    
    def _get_integrate_fn(self, *args, **kwargs):
        trajectory_fn = averaged_trajectory_from_step if self.output_averages else dinosaur.time_integration.trajectory_from_step

        def _integrate_fn(state):
            integrate_fn = jax.jit(trajectory_fn(
                *args,
                **kwargs,
                post_process_fn=self._post_process if self.physics.write_output else lambda x: x
            ))

            if self.output_averages: # If averaging is on, raw_predictions is already a Predictions with the physics populated but dynamics have not been post-processed
                empty_physics_data = self.physics.get_empty_data(self.geometry)
                final_state, raw_predictions = integrate_fn(state, empty_physics_data)
                predictions = raw_predictions.replace(
                    dynamics=dynamics_state_to_physics_state(raw_predictions.dynamics, self.primitive)
                )
            else:
                final_state, raw_predictions = integrate_fn(state)
                # If averaging is off, raw_predictions is already a post-processed Predictions if we wrote physics output
                # Otherwise it is a raw stack of dynamics states
                predictions = raw_predictions if self.physics.write_output else Predictions(
                    dynamics=dynamics_state_to_physics_state(raw_predictions, self.primitive),
                    physics=None
                )

            return final_state, predictions
        
        return _integrate_fn

    def unroll(self, state: primitive_equations.State) -> tuple[primitive_equations.State, Predictions]:
        """Runs the full simulation forward in time from a given state.
        
        Args:
            state: 
                The initial `primitive_equations.State` for the simulation.
        
        Returns:
            A tuple containing the trajectory of post-processed model states.
        """
        integrate = jax.jit(self._get_integrate_fn(
            self.step_fn, # FIXME: figure out where to put jax.checkpoint
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps,
            start_with_input=True,
        ))
        return integrate(state)

    def data_to_xarray(self, data):
        """Converts raw simulation data to an xarray.Dataset.
        
        Args:
            data: 
                A dictionary of raw simulation output arrays.
        
        Returns:
            An `xarray.Dataset` with labeled dimensions and coordinates.
        """
        from dinosaur.xarray_utils import data_to_xarray
        return data_to_xarray(data, coords=self.coords, times=self.times)

    def predictions_to_xarray(self, predictions):
        """Converts the full prediction trajectory to a final xarray.Dataset.
        This function unpacks the nested dictionary structure from the simulation
        output, formats the data, and converts the time coordinate to a
        datetime object.

        Args:
            predictions: 
                The raw output from the `unroll` method.

        Returns:
            A final `xarray.Dataset` ready for analysis and plotting.
        """
        # extract dynamics predictions (PhysicsState format)
        # and physics predictions from postprocessed output
        dynamics_predictions = predictions.dynamics
        physics_predictions = predictions.physics

        # prepare physics predictions for xarray conversion
        # (e.g. separate multi-channel fields so they are compatible with data_to_xarray)
        physics_preds_dict = self.physics.data_struct_to_dict(physics_predictions, self.geometry)
        
        pred_ds = self.data_to_xarray(dynamics_predictions.asdict() | physics_preds_dict)

        # Import units attribute associated with each xarray output from units_table.csv
        units_df = pd.read_csv("../jcm/physics/speedy/units_table.csv")
        units_from_csv = dict(zip(units_df["Variable"], units_df["Units"]))

        for var, unit in units_from_csv.items():
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
        
        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = (
            (self.start_date.delta.days + pred_ds.time - self.save_interval.m)*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')
        
        return pred_ds
