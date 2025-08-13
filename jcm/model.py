import jax
import jax.numpy as jnp
import tree_math
from numpy import timedelta64
from typing import Any, Callable
from datetime import datetime
from xarray import Dataset
import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.constants import p0
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.boundaries import BoundaryData, default_boundaries, update_boundaries_with_timestep
from jcm.date import DateData, Timestamp, Timedelta
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.params import Parameters

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

@tree_math.struct
class Predictions:
    dynamics: PhysicsState
    physics: Any
    _data_to_xarray: Callable[[dict[str, Any]], Dataset]
    _data_struct_to_dict: Callable[[Any], dict[str, Any]]

    # Moving this here allows for future refactor making physics an argument of unroll
    def to_xarray(self) -> Dataset:
        # extract dynamics predictions (PhysicsState format)
        # and physics predictions (PhysicsData format) from postprocessed output
        # and convert to dict for conversion to xarray
        dynamics_preds_dict = self.dynamics.asdict()
        # ensure physics dict is flat (e.g. separate multi-channel fields) for data_to_xarray
        physics_preds_dict = self._data_struct_to_dict(self.physics)

        pred_ds = self._data_to_xarray(dynamics_preds_dict | physics_preds_dict)

        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        return pred_ds.isel(level=slice(None, None, -1))
        
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
                 physics: Physics=None) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step: Model time step in minutes
            layers: Number of vertical layers
            horizontal_resolution: Horizontal resolution of the model (31, 42, 85, or 213)
            coords: CoordinateSystem object describing model grid
            orography: 2D array describing surface orography
            physics: Physics object describing the model physics
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
        # TODO: make the truncation number a parameter consistent with the grid shape
        truncated_orography = primitive_equations.truncated_modal_orography(self.orography, self.coords, wavenumbers_to_clip=2)

        self.primitive = primitive_equations.PrimitiveEquations(
            reference_temperature=self.ref_temps,
            orography=truncated_orography, 
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
            dinosaur.time_integration.exponential_step_filter(
                self.coords.horizontal, self.dt, tau=0.0087504, order=1.5, cutoff=0.8
            ),
        ]

    def _prepare_boundaries(self, boundaries: BoundaryData=None) -> BoundaryData:
        params_for_boundaries = (self.physics.parameters 
                                 if (hasattr(self.physics, 'parameters') and isinstance(self.physics.parameters, Parameters))
                                 else Parameters.default())
        if boundaries is None:
            return default_boundaries(self.coords.horizontal, self.orography, params_for_boundaries)
        return update_boundaries_with_timestep(boundaries, params_for_boundaries, self.dt_si.m)

    def _create_step_fn(self, boundaries: BoundaryData, start_date: Timestamp):
        boundaries = self._prepare_boundaries(boundaries)
        physics_forcing_eqn = ExplicitODE.from_functions(lambda state:
            get_physical_tendencies(
                state=state,
                dynamics=self.primitive,
                time_step=self.dt_si.m,
                physics=self.physics,
                boundaries=boundaries,
                geometry=self.geometry,
                date = DateData.set_date(
                    model_time = start_date + Timedelta(seconds=state.sim_time),
                    model_step = (state.sim_time / self.dt_si.m).astype(jnp.int32),
                    dt_seconds = jnp.float32(self.dt_si.m)
                )
            )
        )
        
        primitive_with_speedy = dinosaur.time_integration.compose_equations([self.primitive, physics_forcing_eqn])
        unfiltered_step_fn = dinosaur.time_integration.imex_rk_sil3(primitive_with_speedy, self.dt)
        return dinosaur.time_integration.step_with_filters(unfiltered_step_fn, self.filters)

    def prepare_initial_state(self, initial_state: PhysicsState=None, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> primitive_equations.State:
        from jcm.physics_interface import physics_state_to_dynamics_state

        # Either use the designated initial state, or generate one. The initial state to the dycore is a modal primitive_equations.State,
        # but the optional initial state from the user is a nodal PhysicsState
        if initial_state is not None:
            state = physics_state_to_dynamics_state(initial_state, self.primitive)
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

    def post_process(self, state: primitive_equations.State, boundaries: BoundaryData, start_date: Timestamp) -> Predictions:
        from jcm.date import DateData
        from jcm.physics_interface import dynamics_state_to_physics_state, verify_state

        physics_state = dynamics_state_to_physics_state(state, self.primitive)
        
        physics_data = None
        if self.physics.write_output:
            date=DateData.set_date(
                model_time = start_date + Timedelta(seconds=state.sim_time),
                model_step = (state.sim_time / self.dt_si.m).astype(jnp.int32),
                dt_seconds = jnp.float32(self.dt_si.m)
            )
            clamped_physics_state = verify_state(physics_state)
            _, physics_data = self.physics.compute_tendencies(clamped_physics_state, boundaries, self.geometry, date)

        return Predictions(dynamics=physics_state, physics=physics_data,
                           _data_to_xarray=None, _data_struct_to_dict=None)

    def _data_to_xarray(self, data, times) -> Dataset:
        from dinosaur.xarray_utils import data_to_xarray
        ds = data_to_xarray(data, coords=self.coords, times=(times - times[0]).m)
        ds['time'] = (
            (times[0].m + ds.time)*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')
        return ds

    def unroll(self,
               state: primitive_equations.State=None,
               boundaries: BoundaryData=None,
               save_interval=10.0,
               total_time=120.0,
               start_date: Timestamp=Timestamp.from_datetime(datetime(2000, 1, 1))
    ) -> tuple[primitive_equations.State, Predictions]:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            state: Initial state of the model
            boundaries: Boundary conditions for the model
            save_interval: Save interval in days
            total_time: Total integration time in days
            start_date: Start date of the simulation
        """
        inner_steps = int(save_interval / self.dt_si.to(units.day).m)
        outer_steps = int(total_time / save_interval)
        times = (start_date.delta.days + save_interval * jnp.arange(outer_steps)) * units.day

        integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
            jax.checkpoint(self._create_step_fn(boundaries, start_date)),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=True,
            post_process_fn=lambda state: self.post_process(state, boundaries, start_date),
        ))
        final_state, predictions = integrate_fn(state or self.prepare_initial_state())

        return final_state, predictions.replace(
            _data_to_xarray=lambda data: self._data_to_xarray(data, times),
            _data_struct_to_dict=lambda data: self.physics.data_struct_to_dict(data, self.geometry),
        )
