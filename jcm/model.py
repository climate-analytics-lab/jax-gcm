import jax
import jax.numpy as jnp
import tree_math
from numpy import timedelta64
from typing import Any
import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0, state_diff_timescale, state_diff_order
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.boundaries import BoundaryData, default_boundaries, update_boundaries_with_timestep
from jcm.date import DateData, Timestamp, Timedelta
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.params import Parameters
from jcm.diffusion import DiffusionFilter
import pandas as pd

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
                 initial_state: PhysicsState=None, physics: Physics=None, diffusion: DiffusionFilter=None) -> None:
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
                Horizontal resolution of the model (31, 42, 85, or 213)
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
        """
        from datetime import datetime

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
        
        physics_forcing_eqn = ExplicitODE.from_functions(lambda state:
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
                )
            )
        )

        # Define trajectory times, expects start_with_input=False
        self.times = self.save_interval * jnp.arange(1, self.outer_steps+1)
        
        self.primitive_with_speedy = dinosaur.time_integration.compose_equations([self.primitive, physics_forcing_eqn])
        step_fn = dinosaur.time_integration.imex_rk_sil3(self.primitive_with_speedy, self.dt)
        
        def conserve_global_mean_surface_pressure(u, u_next):
            return u_next.replace(
                # prevent global mean (0th spectral component) surface pressure drift by setting it to its value before timestep
                log_surface_pressure=u_next.log_surface_pressure.at[0, 0, 0].set(u.log_surface_pressure[0, 0, 0])
            )
        
        filters = [
            conserve_global_mean_surface_pressure,
            dinosaur.time_integration.horizontal_diffusion_step_filter(
                self.coords.horizontal, self.dt, diffusion.state_diff_timesacle, order=diffusion.state_diff_order),
        ]
        self.step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)

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

    def post_process(self, state: primitive_equations.State) -> Predictions:
        """Post-processes a single state from the simulation trajectory. This function is called by the integrator at each save point. It converts the dynamical state to a physical state and, if enabled, runs the physics package to compute diagnostic variables.
        
        Args:
            state: 
                A `primitive_equations.State` object from the simulation.
        
        Returns:
            A dictionary containing the `PhysicsState` ('dynamics') and the
            diagnostic `PhysicsData` ('physics').
        """
        from jcm.date import DateData
        from jcm.physics_interface import dynamics_state_to_physics_state, verify_state

        physics_state = dynamics_state_to_physics_state(state, self.primitive)
        
        physics_data = None
        if self.physics.write_output:
            date=DateData.set_date(
                model_time = self.start_date + Timedelta(seconds=state.sim_time),
                model_step = ((state.sim_time/60) / self.time_step).astype(jnp.int32),
                dt_seconds = jnp.float32(self.time_step * 60.0)
            )
            clamped_physics_state = verify_state(physics_state)
            _, physics_data = self.physics.compute_tendencies(clamped_physics_state, self.boundaries, self.geometry, date)

        return Predictions(dynamics=physics_state, physics=physics_data)

    def unroll(self, state: primitive_equations.State) -> tuple[primitive_equations.State, Predictions]:
        """Runs the full simulation forward in time from a given state.
        
        Args:
            state: 
                The initial `primitive_equations.State` for the simulation.
        
        Returns:
            A tuple containing the trajectory of post-processed model states.
        """
        integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
            jax.checkpoint(self.step_fn),
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps,
            start_with_input=True,
            post_process_fn=self.post_process,
        ))
        return integrate_fn(state)

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
        # and physics predictions (PhysicsData format) from postprocessed output
        dynamics_predictions = predictions.dynamics
        physics_predictions = predictions.physics

        # prepare physics predictions for xarray conversion
        # (e.g. separate multi-channel fields so they are compatible with data_to_xarray)
        physics_preds_dict = self.physics.data_struct_to_dict(physics_predictions, self.geometry)
        
        pred_ds = self.data_to_xarray(dynamics_predictions.asdict() | physics_preds_dict)

        # Import units attribute associated with each xarray output from units_table.csv
        units_df = pd.read_csv("units_table.csv")
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
