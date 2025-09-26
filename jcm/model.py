import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import tree_math
from packaging import version
from flax import __version__ as flax_version
from flax import nnx
from numpy import timedelta64
from typing import Any
from datetime import datetime
import dinosaur
from typing import Callable, Any
from dinosaur import typing
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.date import DateData, Timedelta, Timestamp
from jcm.boundaries import BoundaryData, default_boundaries
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies, dynamics_state_to_physics_state
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.params import Parameters
from jcm.utils import stack_trees
from jcm.diffusion import DiffusionFilter
import pandas as pd

_LEGACY_SCAN_API = version.parse(flax_version) < version.parse("0.10.0")

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

@tree_math.struct
class Predictions:
    """Container for model prediction outputs from a single timestep.

    Attributes:
        dynamics (PhysicsState): The physical state variables converted from
            the dynamical state.
        physics (Any): Diagnostic physics data computed by the physics package.
        times (Any): Timestamps of the predictions.
    """
    dynamics: PhysicsState
    physics: Any
    times: Any

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
    post_process_fn=lambda x: x,
    **kwargs
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

        out_axes = (nnx.Carry,) if _LEGACY_SCAN_API else (nnx.Carry, 0)
        empty_output = (None,) if _LEGACY_SCAN_API else None

        @nnx.scan(in_axes=(nnx.Carry,), out_axes=out_axes, length=inner_steps)
        @jax.checkpoint
        def inner_step(carry):
            x, x_sum, diag_state = carry
            x_sum += x  # include initial state, not final state
            temp_collector_inner = nnx.merge(graphdef, diag_state)
            temp_collector_inner.physical_step.value = True
            x_next = step_fn(temp_collector_inner)(x)
            _, updated_diag_state = nnx.split(temp_collector_inner)
            return (x_next, x_sum, updated_diag_state), empty_output

        @nnx.scan(in_axes=(nnx.Carry,), out_axes=out_axes, length=outer_steps)
        def outer_step(carry):
            (x_final, x_sum, diag_state), _ = inner_step(carry)
            temp_collector_outer = nnx.merge(graphdef, diag_state)
            temp_collector_outer.i.value += 1
            _, updated_diag_state = nnx.split(temp_collector_outer)
            return (x_final, empty_sum, updated_diag_state), (x_sum / inner_steps,)
        
        carry = (x_initial, empty_sum, init_diag_state)
        (x_final, _, final_diag_state), (preds,) = outer_step(carry)
        return x_final, post_process_fn(preds).replace(
            physics=nnx.merge(graphdef, final_diag_state).data.value,
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

    def __init__(self, time_step=30.0, layers=8, horizontal_resolution=31,
                 coords: CoordinateSystem=None, orography: jnp.ndarray=None,
                 physics: Physics=None, diffusion: DiffusionFilter=None) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step:
                Model time step in minutes
            layers: 
                Number of vertical layers
            horizontal_resolution:
                Horizontal resolution of the model (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425)
            coords:
                CoordinateSystem object describing model grid
            orography:
                Orography data (2D array)
            physics: 
                Physics object describing the model physics
            diffusion:
                DiffusionFilter object describing horizontal diffusion filter params
        """

        self.physics_specs = PHYSICS_SPECS
        self.dt_si = (time_step * units.minute).to(units.second)
        self.dt = self.physics_specs.nondimensionalize(self.dt_si)

        if coords is not None:
            self.coords = coords
            horizontal_resolution = coords.horizontal.total_wavenumbers - 2
        else:
            self.coords = get_coords(layers=layers, horizontal_resolution=horizontal_resolution)
        self.geometry = Geometry.from_coords(self.coords)

        # Get the reference temperature and orography. This also returns the initial state function (if wanted to start from rest)
        self.default_state_fn, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self.physics_specs,
            p0=p0*units.pascal,
        )
        
        self.ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        
        self.physics = physics or SpeedyPhysics()

        self.orography = orography if orography is not None else aux_features[dinosaur.xarray_utils.OROGRAPHY]
        self.diffusion = diffusion or DiffusionFilter.default()

        # TODO: make the truncation number a parameter consistent with the grid shape
        self.truncated_orography = primitive_equations.truncated_modal_orography(self.orography, self.coords, wavenumbers_to_clip=2)
        
        self.primitive = primitive_equations.PrimitiveEquations(
            reference_temperature=self.ref_temps,
            orography=self.truncated_orography,
            coords=self.coords,
            physics_specs=self.physics_specs,
        )
        
        def conserve_global_mean_surface_pressure(u, u_next):
            return u_next.replace(
                # prevent global mean (0th spectral component) surface pressure drift by setting it to its value before timestep
                log_surface_pressure=u_next.log_surface_pressure.at[0, 0, 0].set(u.log_surface_pressure[0, 0, 0])
            )
        
        self.filters = [
            conserve_global_mean_surface_pressure,
            dinosaur.time_integration.horizontal_diffusion_step_filter(
                self.coords.horizontal, self.dt, tau=self.diffusion.state_diff_timescale, order=self.diffusion.state_diff_order),
        ]

        # The following fields are set upon calling model.run
        self.initial_nodal_state = None
        self.start_date = None

        # spectral space primitive_equations.State updated by model.run and model.resume
        self._final_modal_state = None
        
    def _prepare_initial_modal_state(self, physics_state: PhysicsState=None, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> primitive_equations.State:
        """Prepares initial dinosaur.primitive_equations.State for a model run.

        Args:
            physics_state:
                Optional nodal PhysicsState from which to generate the modal state. If none provided, initial state will be isothermal atmosphere with random noise surface pressure perturbation.
            random_seed:
                Seed for pressure perturbation (default 0).
            sim_time:
                Optionally specify the sim_time attribute for the state (default 0.0).
            humidity_perturbation:
                If True and using the default state, adds a horizontally localized perturbation to specific humidity.

        Returns:
            A `primitive_equations.State` object ready for integration.
        """
        from jcm.physics_interface import physics_state_to_dynamics_state

        # Either use the designated initial state, or generate one. The initial state to the dycore is a modal primitive_equations.State,
        # but the optional initial state from the user is a nodal PhysicsState
        if physics_state is not None:
            state = physics_state_to_dynamics_state(physics_state, self.primitive)
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

    def _date_from_sim_time(self, sim_time) -> DateData:
        return DateData.set_date(
            model_time=self.start_date + Timedelta(seconds=sim_time),
            model_step=(sim_time / self.dt_si.m).astype(jnp.int32),
            dt_seconds=self.dt_si.m
        )

    def _create_step_fn(self, boundaries: BoundaryData, output_averages):
        physics_forcing_eqn = lambda d: ExplicitODE.from_functions(lambda state:
            get_physical_tendencies(
                state=state,
                dynamics=self.primitive,
                time_step=self.dt_si.m,
                physics=self.physics,
                boundaries=boundaries,
                diffusion=self.diffusion,
                geometry=self.geometry,
                date=self._date_from_sim_time(state.sim_time),
                diagnostics_collector=d
            )
        )
        primitive_with_speedy = lambda d: dinosaur.time_integration.compose_equations([self.primitive, physics_forcing_eqn(d)])
        unfiltered_step_fn = lambda d: dinosaur.time_integration.imex_rk_sil3(primitive_with_speedy(d), self.dt)
        step_fn = lambda d=None: dinosaur.time_integration.step_with_filters(unfiltered_step_fn(d), self.filters)
        return step_fn if output_averages else jax.checkpoint(step_fn())

    def _post_process(self, state: primitive_equations.State, boundaries: BoundaryData, output_averages: bool) -> Predictions:
        """Post-processes a single state from the simulation trajectory. This function is called by the integrator at each save point. It converts the dynamical state to a physical state and, if enabled, runs the physics package to compute diagnostic variables.
        
        Args:
            state: 
                A `primitive_equations.State` object from the simulation.
        
        Returns:
            A dictionary containing the `PhysicsState` ('dynamics') and the
            diagnostic physics variables (data structure determined by model.physics).
        """
        from jcm.physics_interface import verify_state

        predictions = Predictions(
            dynamics=dynamics_state_to_physics_state(state, self.primitive),
            physics=None,
            times=None
        )

        if not output_averages:
            date = self._date_from_sim_time(state.sim_time)
            clamped_physics_state = verify_state(predictions.dynamics)
            _, physics_data = self.physics.compute_tendencies(clamped_physics_state, boundaries, self.geometry, date)
            predictions = predictions.replace(physics=physics_data)

        return predictions
    
    def _get_integrate_fn(self, step_fn, outer_steps, inner_steps, post_process_fn, output_averages, **kwargs):
        trajectory_fn = averaged_trajectory_from_step if output_averages else dinosaur.time_integration.trajectory_from_step

        def _integrate_fn(state):
            integrate_fn = jax.jit(trajectory_fn(
                step_fn=step_fn,
                outer_steps=outer_steps,
                inner_steps=inner_steps,
                **kwargs,
                post_process_fn=post_process_fn
            ))
            
            # integrate_fn for avgs has different signature b/c empty physics data structure needed for DiagnosticsCollector initialization
            return integrate_fn(state, self.physics.get_empty_data(self.geometry)) if output_averages else integrate_fn(state)
        
        return _integrate_fn

    def resume(self,
               boundaries: BoundaryData=None,
               save_interval=10.0,
               total_time=120.0,
               output_averages=False
    ) -> Predictions:
        """Runs the full simulation forward in time starting from end of previous call to model.run or model.resume.
        
        Args:
            boundaries:
                BoundaryData containing boundary conditions for the run.
            save_interval:
                Interval at which to save model outputs (float).
            total_time:
                Total time to run the model (float).
            output_averages:
                Whether to output time-averaged quantities (default False).

        Returns:
            A Predictions object containing the trajectory of post-processed model states.
        """
        boundaries = boundaries or default_boundaries(self.coords.horizontal, self.orography)
        step_fn = self._create_step_fn(boundaries, output_averages)

        inner_steps = int(save_interval / self.dt_si.to(units.day).m)
        outer_steps = int(total_time / save_interval)
        start_time = self.start_date.delta.days + (self._final_modal_state.sim_time*units.second).to(units.day).m
        times = start_time + save_interval * jnp.arange(outer_steps)

        integrate = self._get_integrate_fn(
            step_fn,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=True,
            post_process_fn=lambda state: self._post_process(state, boundaries, output_averages),
            output_averages=output_averages
        )

        # starts from preexisting self._final_modal_state, then updates self._final_modal_state
        self._final_modal_state, predictions = integrate(self._final_modal_state)

        return predictions.replace(times=times)

    def run(self,
            initial_state: PhysicsState | primitive_equations.State = None,
            boundaries: BoundaryData=None,
            save_interval=10.0,
            total_time=120.0,
            start_date: Timestamp=Timestamp.from_datetime(datetime(2000, 1, 1)),
            output_averages=False
    ) -> tuple[primitive_equations.State, Predictions]:
        """Sets model.initial_nodal_state and model.start_date and runs the full simulation forward in time.
        
        Args:
            initial_state:
                PhysicsState or dinosaur.primitive_equations.State containing initial state of the model (default isothermal atmosphere).
            boundaries:
                BoundaryData containing boundary conditions for the run (default aquaplanet).
            save_interval:
                (float) interval at which to save model outputs in days (default 10.0).
            total_time:
                (float) total time to run the model in days (default 120.0).
            start_date:
                (Timestamp) start date for the model run (default January 1, 2000).
            output_averages:
                Whether to output time-averaged quantities (default False).

        Returns:
            A Predictions object containing the trajectory of post-processed model states.
        """
        if isinstance(initial_state, primitive_equations.State):
            self.initial_nodal_state = dynamics_state_to_physics_state(initial_state, self.primitive)
            self._final_modal_state = initial_state
        else:
            self.initial_nodal_state = initial_state
            self._final_modal_state = self._prepare_initial_modal_state(initial_state)

        self.start_date = start_date

        return self.resume(boundaries=boundaries, save_interval=save_interval, total_time=total_time, output_averages=output_averages)

    def predictions_to_xarray(self, predictions):
        """Converts the full prediction trajectory to a final xarray.Dataset.
        This function unpacks the nested dictionary structure from the simulation
        output, formats the data, and converts the time coordinate to a
        datetime object.

        Args:
            predictions: 
                The raw output from the `run` or `resume` method.

        Returns:
            A final `xarray.Dataset` ready for analysis and plotting.
        """
        from dinosaur.xarray_utils import data_to_xarray
        from pathlib import Path
        # extract dynamics predictions (PhysicsState format)
        # and physics predictions from postprocessed output
        dynamics_predictions = predictions.dynamics
        physics_predictions = predictions.physics

        # prepare physics predictions for xarray conversion
        # (e.g. separate multi-channel fields so they are compatible with data_to_xarray)
        physics_preds_dict = self.physics.data_struct_to_dict(physics_predictions, self.geometry)

        times = jax.device_get(predictions.times)

        pred_ds = data_to_xarray(dynamics_predictions.asdict() | physics_preds_dict, coords=self.coords, times=times - times[0])

        # Import units attribute associated with each xarray output from units_table.csv
        units_df = pd.read_csv(Path(__file__).parent.parent / "jcm" / "physics" / "speedy" / "units_table.csv")
        units_from_csv = dict(zip(units_df["Variable"], units_df["Units"]))

        for var, unit in units_from_csv.items():
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
        
        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = (
            times*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')
        
        return pred_ds
