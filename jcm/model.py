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
from dinosaur.vertical_interpolation import HybridCoordinates
from jcm.boundaries import BoundaryData, default_boundaries, update_boundaries_with_timestep
from jcm.date import DateData, Timestamp, Timedelta
from jcm.physics.speedy.params import Parameters
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.physics.speedy.physical_constants import p0
from jcm.physics_interface import PhysicsState, Physics, get_physical_tendencies
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
import dataclasses

class HybridCoordinatesWithCenters(HybridCoordinates):
    """
    Extension of Dinosaur's HybridCoordinates that adds sigma-like attributes.
    
    This enables compatibility with isothermal_rest_atmosphere calculations
    that expect sigma-like coordinate systems with centers and layer_thickness.
    """
    
    def __init__(self, a_boundaries, b_boundaries):
        super().__init__(a_boundaries, b_boundaries)
        # Pre-compute centers and layer_thickness as concrete numpy arrays to avoid tracer issues
        import numpy as np
        
        # For hybrid coordinates, we need to handle pure pressure levels (b=0) and hybrid levels (b>0)
        # Create a monotonic sigma-like coordinate that avoids zeros
        n_levels = len(b_boundaries) - 1
        centers = np.zeros(n_levels)
        
        # Find the first hybrid level (where b > 0)
        first_hybrid_idx = np.where(b_boundaries > 0)[0][0] - 1  # -1 for level index
        
        # For pure pressure levels, assign small increasing values
        if first_hybrid_idx > 0:
            centers[:first_hybrid_idx] = np.linspace(1e-6, 1e-4, first_hybrid_idx)
        
        # For hybrid levels, use b-coordinate centers
        for i in range(first_hybrid_idx, n_levels):
            centers[i] = (b_boundaries[i] + b_boundaries[i+1]) / 2
        
        self.centers = np.asarray(centers)
        
        # For layer thickness, we need to handle pure pressure levels differently
        # Use a small but non-zero thickness for pure pressure levels
        layer_thickness = np.diff(b_boundaries)
        
        # For pure pressure levels (where diff is 0), assign small thicknesses
        zero_thickness_mask = layer_thickness == 0
        if np.any(zero_thickness_mask):
            # Assign small, uniform thickness for pure pressure levels
            layer_thickness[zero_thickness_mask] = 1e-6
        
        self.layer_thickness = np.asarray(layer_thickness)

    # layer_thickness is now a pre-computed attribute, not a property
    
    @property
    def boundaries(self) -> jnp.ndarray:
        """
        Return boundaries for sigma-like interface compatibility.
        
        Maps to b_boundaries which represent the sigma component of hybrid coordinates.
        """
        return self.b_boundaries
    
    @property
    def center_to_center(self) -> jnp.ndarray:
        """
        Return distances between consecutive level centers.
        
        This is the difference between consecutive centers, used in some
        vertical integration calculations.
        """
        import numpy as np
        return np.diff(self.centers)

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

@tree_math.struct
class Predictions:
    dynamics: PhysicsState
    physics: Any

def _get_horizontal_grid(horizontal_resolution: int):
    """Get horizontal grid for given resolution with validation."""
    try:
        return getattr(dinosaur.spherical_harmonic.Grid, f'T{horizontal_resolution}')
    except AttributeError:
        raise ValueError(f"Invalid horizontal resolution: {horizontal_resolution}. Must be one of: 21, 31, 42, 85, 106, 119, 170, 213, 340, or 425.")


def get_coords(layers: int = 8, horizontal_resolution: int = 31, hybrid: bool = False) -> CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given configuration.
    
    Args:
        layers: Number of vertical layers
        horizontal_resolution: Horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425)
        hybrid: If True, use hybrid sigma-pressure coordinates; if False, use sigma coordinates
        
    Returns:
        CoordinateSystem with specified coordinates
    """
    horizontal_grid = _get_horizontal_grid(horizontal_resolution)
    
    if hybrid:
        from jcm.vertical.icon_levels import ICONLevels
        hybrid_levels = ICONLevels.get_levels(layers)
        vertical_coords = HybridCoordinatesWithCenters(
            hybrid_levels.a_boundaries,
            hybrid_levels.b_boundaries
        )
    else:
        if layers not in sigma_layer_boundaries:
            raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {list(sigma_layer_boundaries.keys())}")
        vertical_coords = dinosaur.sigma_coordinates.SigmaCoordinates(sigma_layer_boundaries[layers])

    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=PHYSICS_SPECS.radius),
        vertical=vertical_coords
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
                 use_hybrid_coords: bool = None) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step: Model time step in minutes
            save_interval: Save interval in days
            total_time: Total integration time in days
            start_date: Start date of the simulation
            layers: Number of vertical layers
            horizontal_resolution: Horizontal resolution of the model (31, 42, 85, or 213)
            coords: CoordinateSystem object describing model grid
            boundaries: BoundaryData object describing surface boundary conditions
            initial_state: Initial state of the model (PhysicsState object), optional
            physics: Physics object describing the model physics
            use_hybrid_coords: If True, use hybrid sigma-pressure coordinates; if False, use sigma; 
                             if None, auto-detect based on physics type
        """
        from datetime import datetime

        # Integration settings
        self.start_date = start_date or Timestamp.from_datetime(datetime(2000, 1, 1))
        self.save_interval = save_interval * units.day
        self.total_time = total_time * units.day
        self.time_step = time_step
        dt_si = self.time_step * units.minute

        self.physics_specs = PHYSICS_SPECS

        # Auto-detect coordinate system based on physics type
        if use_hybrid_coords is None:
            from jcm.physics.icon import IconPhysics
            use_hybrid_coords = isinstance(physics, IconPhysics)
        
        if coords is not None:
            self.coords = coords
            horizontal_resolution = coords.horizontal.total_wavenumbers - 2
        else:
            self.coords = get_coords(layers=layers, horizontal_resolution=horizontal_resolution, hybrid=use_hybrid_coords)
        
        # Set up geometry with appropriate coordinate system
        self.geometry = Geometry.from_coords(self.coords, hybrid=use_hybrid_coords)

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
            p0=p0 * units.pascal,
            p1=.05 * p0 * units.pascal,
        )
        
        self.ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        
        self.physics = physics or SpeedyPhysics()

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
                date = DateData.set_date(
                    model_time = self.start_date + Timedelta(seconds=state.sim_time),
                    model_step = jnp.int32((state.sim_time/60) / time_step)
                )
            )
        )

        # Define trajectory times, expects start_with_input=False
        self.times = self.save_interval * jnp.arange(1, self.outer_steps+1)
        
        self.primitive_with_speedy = dinosaur.time_integration.compose_equations([self.primitive, physics_forcing_eqn])
        step_fn = dinosaur.time_integration.imex_rk_sil3(self.primitive_with_speedy, self.dt)
        filters = [
            dinosaur.time_integration.exponential_step_filter(
                self.coords.horizontal, self.dt, tau=0.0087504, order=1.5, cutoff=0.8
            ),
        ]
        self.step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)

    def get_initial_state(self, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> primitive_equations.State:
        from jcm.physics_interface import physics_state_to_dynamics_state

        # Either use the designated initial state, or generate one. The initial state to the model is in dynamics form, but the
        # optional initial state from the user is in physics form
        if self.initial_state is not None:
            self.initial_state.surface_pressure = self.initial_state.surface_pressure / p0 # convert to normalized surface pressure
            state = physics_state_to_dynamics_state(self.initial_state, self.primitive)
        else:
            state = self.default_state_fn(jax.random.PRNGKey(random_seed))
            # default state returns log surface pressure, we want it to be log(normalized_surface_pressure)
            # there are several ways to do this operation (in modal vs nodal space, with log vs absolute pressure), this one has the least error
            state.log_surface_pressure = self.coords.horizontal.to_modal(
                self.coords.horizontal.to_nodal(state.log_surface_pressure) - jnp.log(p0)
            )

            # need to add specific humidity as a tracer
            state.tracers = {
                'specific_humidity': (1e-2 if humidity_perturbation else 0.0) * primitive_equations_states.gaussian_scalar(self.coords, self.physics_specs)
            }
        return primitive_equations.State(**state.asdict(), sim_time=sim_time)

    def post_process(self, state: primitive_equations.State) -> Predictions:
        from jcm.date import DateData
        from jcm.physics_interface import dynamics_state_to_physics_state, verify_state

        physics_state = dynamics_state_to_physics_state(state, self.primitive)
        
        physics_data = None
        if self.physics.write_output:
            date=DateData.set_date(
                model_time = self.start_date + Timedelta(seconds=state.sim_time),
                model_step = jnp.int32((state.sim_time/60) / self.time_step)
            )
            clamped_physics_state = verify_state(physics_state)
            _, physics_data = self.physics.compute_tendencies(clamped_physics_state, self.boundaries, self.geometry, date)
        
        # convert back to SI to match convention for user-defined initial PhysicsStates
        physics_state.surface_pressure = physics_state.surface_pressure * p0
        
        return Predictions(dynamics=physics_state, physics=physics_data)

    def unroll(self, state: primitive_equations.State) -> tuple[primitive_equations.State, Predictions]:
        # integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
        integrate_fn = (dinosaur.time_integration.trajectory_from_step(
            jax.checkpoint(self.step_fn),
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps,
            start_with_input=True,
            post_process_fn=self.post_process,
        ))
        return integrate_fn(state)

    def data_to_xarray(self, data):
        from dinosaur.xarray_utils import data_to_xarray
        return data_to_xarray(data, coords=self.coords, times=self.times)

    def predictions_to_xarray(self, predictions):
        # extract dynamics predictions (PhysicsState format)
        # and physics predictions (PhysicsData format) from postprocessed output
        dynamics_predictions = predictions.dynamics
        physics_predictions = predictions.physics

        # prepare physics predictions for xarray conversion
        # (e.g. separate multi-channel fields so they are compatible with data_to_xarray)
        physics_preds_dict = self.physics.data_struct_to_dict(physics_predictions, self.geometry)
        
        # Add time dimension to dynamics predictions
        dynamics_dict = dynamics_predictions.asdict()
        # for key, value in dynamics_dict.items():
            # if hasattr(value, 'shape') and value.ndim >= 1:
                # Add time dimension if not already present
                # if value.shape[0] != 1:
                #     dynamics_dict[key] = value.reshape(1, *value.shape)
        
        pred_ds = self.data_to_xarray(dynamics_dict | physics_preds_dict)
        
        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = (
            (self.start_date.delta.days + pred_ds.time - self.save_interval.m)*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')
        
        return pred_ds
