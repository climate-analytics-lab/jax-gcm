import jax
import jax.numpy as jnp
import tree_math
from numpy import timedelta64
from typing import Any
from datetime import datetime
import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.date import DateData, Timedelta, Timestamp
from jcm.boundaries import BoundaryData, default_boundaries
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies, dynamics_state_to_physics_state
from jcm.diffusion import DiffusionFilter
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
import pandas as pd
from functools import partial

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
                 physics: Physics=None, diffusion: DiffusionFilter=None,
                 start_date: Timestamp=Timestamp.from_datetime(datetime(2000, 1, 1))) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step: 
                Model time step in minutes
            layers: 
                Number of vertical layers
            horizontal_resolution: 
                Horizontal resolution of the model (31, 42, 85, or 213)
            coords: 
                CoordinateSystem object describing model grid
            orography:
                Orography data (2D array)
            physics: 
                Physics object describing the model physics
            diffusion:
                DiffusionFilter object describing horizontal diffusion filter params
            start_date: 
                Timestamp object containing start date of the simulation (default January 1, 2000)
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

        self.start_date = start_date

        # grid space PhysicsState set upon calling model.run
        self.initial_nodal_state = None

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

    def _create_step_fn(self, boundaries: BoundaryData):
        physics_forcing_eqn = ExplicitODE.from_functions(lambda state:
            get_physical_tendencies(
                state=state,
                dynamics=self.primitive,
                time_step=self.dt_si.m,
                physics=self.physics,
                boundaries=boundaries,
                diffusion=self.diffusion,
                geometry=self.geometry,
                date=self._date_from_sim_time(state.sim_time)
            )
        )

        primitive_with_speedy = dinosaur.time_integration.compose_equations([self.primitive, physics_forcing_eqn])
        unfiltered_step_fn = dinosaur.time_integration.imex_rk_sil3(primitive_with_speedy, self.dt)
        return dinosaur.time_integration.step_with_filters(unfiltered_step_fn, self.filters)

    def _post_process(self, state: primitive_equations.State, boundaries: BoundaryData) -> Predictions:
        """Post-processes a single state from the simulation trajectory. This function is called by the integrator at each save point. It converts the dynamical state to a physical state and, if enabled, runs the physics package to compute diagnostic variables.
        
        Args:
            state: 
                A `primitive_equations.State` object from the simulation.
        
        Returns:
            A dictionary containing the `PhysicsState` ('dynamics') and the
            diagnostic `PhysicsData` ('physics').
        """
        from jcm.physics_interface import verify_state

        physics_state = dynamics_state_to_physics_state(state, self.primitive)
        
        physics_data = None
        if self.physics.write_output:
            date = self._date_from_sim_time(state.sim_time)
            clamped_physics_state = verify_state(physics_state)
            _, physics_data = self.physics.compute_tendencies(clamped_physics_state, boundaries, self.geometry, date)

        return Predictions(dynamics=physics_state, physics=physics_data, times=None)

    @partial(jax.jit, static_argnums=(0, 1, 2)) # Note: will not recompile if model fields change
    def run_from_state(self,
                       initial_state: primitive_equations.State,
                       boundaries: BoundaryData,
                       save_interval=10.0,
                       total_time=120.0,
    ) -> tuple[primitive_equations.State, Predictions]:
        """Runs the full simulation forward in time starting from given initial state.
        Alternative to model.run / model.resume which does not read/write model's internal current state.
        
        Args:
            initial_state:
                dinosaur.primitive_equations.State containing initial state of the run.
            boundaries:
                BoundaryData containing boundary conditions for the run.
            save_interval:
                (float) interval at which to save model outputs in days (default 10.0).
            total_time:
                (float) total time to run the model in days (default 120.0).
    
        Returns:
            A tuple containing (final dinosaur.primitive_equations.State, Predictions object containing trajectory of post-processed model states).
        """
        step_fn = self._create_step_fn(boundaries)
        
        inner_steps = int(save_interval / self.dt_si.to(units.day).m)
        outer_steps = int(total_time / save_interval)
        times = self.start_date.delta.days \
                + (initial_state.sim_time*units.second).to(units.day).m \
                + save_interval * jnp.arange(outer_steps)

        integrate_fn = dinosaur.time_integration.trajectory_from_step(
            jax.checkpoint(step_fn),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=True,
            post_process_fn=lambda state: self._post_process(state, boundaries)
        )
        
        final_modal_state, predictions = integrate_fn(initial_state)
        return final_modal_state, predictions.replace(times=times)

    def resume(self,
               boundaries: BoundaryData=None,
               save_interval=10.0,
               total_time=120.0
    ) -> Predictions:
        """Runs the full simulation forward in time starting from end of previous call to model.run or model.resume.
        
        Args:
            boundaries:
                BoundaryData containing boundary conditions for the run.
            save_interval:
                Interval at which to save model outputs (float).
            total_time:
                Total time to run the model (float).
            
        Returns:
            A Predictions object containing the trajectory of post-processed model states.
        """
        # starts from preexisting self._final_modal_state, then updates self._final_modal_state
        final_modal_state, predictions = self.run_from_state(
            initial_state=self._final_modal_state,
            boundaries=boundaries,
            save_interval=save_interval,
            total_time=total_time
        )
        
        self._final_modal_state = final_modal_state
        return predictions

    def run(self,
            initial_state: PhysicsState | primitive_equations.State = None,
            boundaries: BoundaryData=None,
            save_interval=10.0,
            total_time=120.0,
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

        Returns:
            A Predictions object containing the trajectory of post-processed model states.
        """
        if isinstance(initial_state, primitive_equations.State):
            self.initial_nodal_state = dynamics_state_to_physics_state(initial_state, self.primitive)
            self._final_modal_state = initial_state
        else:
            self.initial_nodal_state = initial_state
            self._final_modal_state = self._prepare_initial_modal_state(initial_state)

        return self.resume(boundaries=boundaries, save_interval=save_interval, total_time=total_time)

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
        # and physics predictions (PhysicsData format) from postprocessed output
        dynamics_predictions = predictions.dynamics
        physics_predictions = predictions.physics

        # prepare physics predictions for xarray conversion
        # (e.g. separate multi-channel fields so they are compatible with data_to_xarray)
        physics_preds_dict = self.physics.data_struct_to_dict(physics_predictions, self.geometry)

        times = jax.device_get(predictions.times)

        pred_ds = data_to_xarray(dynamics_predictions.asdict() | physics_preds_dict, coords=self.coords, times=times - times[0])

        # Import units attribute associated with each xarray output from units_table.csv
        units_df = pd.read_csv(Path(__file__).parent.parent / "jcm" / "physics" / "speedy" / "units_table.csv")
        for var, unit, desc in zip(units_df["Variable"], units_df["Units"], units_df["Description"]):
            if var in pred_ds:
                pred_ds[var].attrs["units"] = unit
                pred_ds[var].attrs["description"] = desc
        
        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = (
            times*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')
        
        return pred_ds
