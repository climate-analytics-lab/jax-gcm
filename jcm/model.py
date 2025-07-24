from dinosaur.spherical_harmonic import vor_div_to_uv_nodal
import jax
import jax.numpy as jnp
from numpy import timedelta64
import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.boundaries import BoundaryData, default_boundaries, update_boundaries_with_timestep, initialize_boundaries
from jcm.date import Timestamp, Timedelta
from jcm.params import Parameters
from jcm.geometry import sigma_layer_boundaries, Geometry
from jcm.physical_constants import p0
from jcm.physics import PhysicsState, PhysicsTendency, PhysicsData

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

def set_physics_flags(state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData=None,
    geometry: Geometry=None
) -> tuple[PhysicsTendency, PhysicsData]:
    from jcm.physical_constants import nstrad
    '''
        Sets flags that indicate whether a tendency function should be run.
        clouds, get_shortwave_rad_fluxes are the only functions that currently depend on this. 
        This could also apply to forcing and coupling.
    '''
    model_step = physics_data.date.model_step
    compute_shortwave = (jnp.mod(model_step, nstrad) == 1)
    shortwave_data = physics_data.shortwave_rad.copy(compute_shortwave=compute_shortwave)
    physics_data = physics_data.copy(shortwave_rad=shortwave_data)

    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return physics_tendencies, physics_data

def get_speedy_physics_terms(checkpoint_terms=True):
    """
    Returns a list of functions that compute physical tendencies for the model.
    """
    
    from jcm.humidity import spec_hum_to_rel_hum
    from jcm.convection import get_convection_tendencies
    from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
    from jcm.shortwave_radiation import get_shortwave_rad_fluxes, get_clouds
    from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
    from jcm.surface_flux import get_surface_fluxes
    from jcm.vertical_diffusion import get_vertical_diffusion_tend
    from jcm.land_model import couple_land_atm
    from jcm.slabocean_model import couple_sea_atm
    from jcm.forcing import set_forcing

    physics_terms = [
        set_physics_flags,
        set_forcing,
        spec_hum_to_rel_hum,
        get_convection_tendencies,
        get_large_scale_condensation_tendencies,
        get_clouds,
        get_shortwave_rad_fluxes,
        get_downward_longwave_rad_fluxes,
        get_surface_fluxes,
        get_upward_longwave_rad_fluxes,
        get_vertical_diffusion_tend,
        couple_land_atm, # eventually couple sea model and ice model here
        couple_sea_atm,  # order is arbitrary. Need further discussion.
    ]

    #if sea_coupling_flag > 0:
    #    physics_terms.insert(-4, get_surface_fluxes)
    physics_terms.insert(-4, get_surface_fluxes)

    if not checkpoint_terms:
        return physics_terms

    static_argnums = {
        set_forcing: (2,),
        couple_land_atm: (3,),
    }
    
    # Static argnum 4 is the Geometry object
    return [jax.checkpoint(term, static_argnums=static_argnums.get(term, ()) + (4,)) for term in physics_terms]

def convert_tendencies_to_equation(
        dynamics: primitive_equations.PrimitiveEquations, 
        time_step,
        physics_terms,
        reference_date,
        boundaries: BoundaryData,
        parameters: Parameters,
        geometry: Geometry
    ) -> ExplicitODE:

    from jcm.physics_data import PhysicsData
    from jcm.physics import get_physical_tendencies
    from jcm.date import DateData

    def physical_tendencies(state):
        '''
            Sets model date and step, initializes an empty PhysicsData instance, and returns a 
            function to get physical tendencies.
            state - dynamics state from Dinosaur
            state.sim_time - simulation time in seconds (Dinosaur object)
            time_step - model time step in minutes 
            reference_date - start date of the simulation
        '''

        # Set the model time (in datetime format) and model step (number of steps since start time)
        date = DateData.set_date(
            model_time = reference_date + Timedelta(seconds=state.sim_time),
            model_step = ((state.sim_time/60) / time_step).astype(jnp.int32)
        )

        data = PhysicsData.zeros(
            dynamics.coords.nodal_shape[1:],
            dynamics.coords.nodal_shape[0],
            date=date
        )

        return get_physical_tendencies(
            state=state,
            dynamics=dynamics,
            time_step=time_step,
            physics_terms=physics_terms,
            boundaries=boundaries,
            parameters=parameters,
            geometry=geometry,
            data=data
        )
    return ExplicitODE.from_functions(physical_tendencies)

def get_coords(layers=8, horizontal_resolution=31) -> CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (31, 42, 85, or 213).
    """
    resolution_map = {
        31: dinosaur.spherical_harmonic.Grid.T31,
        42: dinosaur.spherical_harmonic.Grid.T42,
        85: dinosaur.spherical_harmonic.Grid.T85,
        213: dinosaur.spherical_harmonic.Grid.T213,
    }

    if horizontal_resolution not in resolution_map:
        raise ValueError(f"Invalid resolution: {horizontal_resolution}. Must be one of: {list(resolution_map.keys())}")

    if layers not in sigma_layer_boundaries:
        raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {list(sigma_layer_boundaries.keys())}")

    # Define the coordinate system
    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=resolution_map[horizontal_resolution](radius=PHYSICS_SPECS.radius), # truncation
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(sigma_layer_boundaries[layers])
    )

class SpeedyModel:
    """
    Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet.

    #TODO: Factor out the geography and physics choices so you can choose independent of each other.
    """

    def __init__(self, time_step=30.0, save_interval=10.0, total_time=1200, start_date=None,
                 layers=8, horizontal_resolution=31, coords: CoordinateSystem=None,
                 boundaries: BoundaryData=None, initial_state: PhysicsState=None, parameters: Parameters=None,
                 post_process=True, checkpoint_terms=True) -> None:
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
            parameters: Parameters object describing model parameters
            physics_specs: PrimitiveEquationsSpecs object describing the model physics
            post_process: Whether to post-process the model output
            checkpoint_terms: Whether to jax.checkpoint each physics term
        """
        from datetime import datetime

        # Integration settings
        self.start_date = start_date or Timestamp.from_datetime(datetime(2000, 1, 1))
        self.save_interval = save_interval * units.day
        self.total_time = total_time * units.day
        dt_si = time_step * units.minute

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

        self.parameters = parameters or Parameters.default()
        self.post_process_physics = post_process

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
        
        # this implicitly calls initialize_modules, must be before we set boundaries
        self.physics_terms = get_speedy_physics_terms(checkpoint_terms=checkpoint_terms)
        
        print("I am here?")
        # TODO: make the truncation number a parameter consistent with the grid shape
        if boundaries is None:
            print("boundaries is None")
            truncated_orography = primitive_equations.truncated_modal_orography(aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords, wavenumbers_to_clip=2)
            self.boundaries = default_boundaries(self.coords.horizontal, aux_features[dinosaur.xarray_utils.OROGRAPHY], self.parameters)
        elif boundaries == "Tien-Yiao_test":

            truncated_orography = primitive_equations.truncated_modal_orography(
                aux_features[dinosaur.xarray_utils.OROGRAPHY],
                self.coords,
                wavenumbers_to_clip=2,
            )

            import os.path

            script_dir = os.path.dirname(__file__)

            self.boundaries = initialize_boundaries(
                filename          = "%s/data/bc/t30/clim/boundaries_daily.nc" % script_dir,
                grid              = self.coords.horizontal,
                parameters        = self.parameters,
                truncation_number = 2,
                time_step = dt_si,
            )

        else:

            self.boundaries = update_boundaries_with_timestep(boundaries, self.parameters, dt_si)
            truncated_orography = primitive_equations.truncated_modal_orography(self.boundaries.orog, self.coords, wavenumbers_to_clip=2)
        
        self.primitive = primitive_equations.PrimitiveEquations(
            self.ref_temps,
            truncated_orography, 
            self.coords,
            self.physics_specs)

        speedy_forcing = convert_tendencies_to_equation(self.primitive,
                                                        time_step,
                                                        self.physics_terms,
                                                        self.start_date,
                                                        self.boundaries,
                                                        self.parameters,
                                                        self.geometry)
        
        # Define trajectory times, expects start_with_input=False
        self.times = self.save_interval * jnp.arange(1, self.outer_steps+1)
        
        self.primitive_with_speedy = dinosaur.time_integration.compose_equations([self.primitive, speedy_forcing])
        step_fn = dinosaur.time_integration.imex_rk_sil3(self.primitive_with_speedy, self.dt)
        filters = [
            dinosaur.time_integration.exponential_step_filter(
                self.coords.horizontal, self.dt, tau=0.0087504, order=1.5, cutoff=0.8
            ),
        ]
        self.step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)

    def get_initial_state(self, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> primitive_equations.State:
        from jcm.physics import physics_state_to_dynamics_state

        #Either use the designated initial state, or generate one. The initial state to the model is in dynamics form, but the
        # optional initial state from the user is in physics form
        if self.initial_state is not None:
            self.initial_state.surface_pressure = self.initial_state.surface_pressure / p0 # convert to normalized surface pressure 
            state = physics_state_to_dynamics_state(self.initial_state, self.primitive)
            return primitive_equations.State(**state.asdict(), sim_time=sim_time)
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

    def post_process(self, state):
        from jcm.date import DateData
        from jcm.physics_data import PhysicsData
        from jcm.physics import dynamics_state_to_physics_state

        physics_state = dynamics_state_to_physics_state(state, self.primitive)
        
        physics_data = None
        if self.post_process_physics:
            physics_data = PhysicsData.zeros(
                self.coords.nodal_shape[1:],
                self.coords.nodal_shape[0],
                date=DateData.set_date(model_time = self.start_date + Timedelta(seconds=state.sim_time))
            )

            for term in self.physics_terms:
                _, physics_data = term(physics_state, physics_data, self.parameters, self.boundaries, self.geometry)

        # convert back to SI to match convention for user-defined initial PhysicsStates
        physics_state.surface_pressure = physics_state.surface_pressure * p0
        
        return {
            'dynamics': physics_state,
            'physics': physics_data
        }

    def unroll(self, state: primitive_equations.State) -> tuple[primitive_equations.State, primitive_equations.State]:
        integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
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
        dynamics_predictions = predictions['dynamics']
        physics_predictions = predictions['physics']

        physics_preds_dict = {}
        if physics_predictions is not None:
            physics_preds_dict = {
                f"{module}.{field}": value # avoids name conflicts between fields of different modules
                for module, module_dict in physics_predictions.asdict().items()
                for field, value in module_dict.asdict().items()
            }
            # replace multi-channel fields with a field for each channel
            _original_keys = list(physics_preds_dict.keys())
            for k in _original_keys:
                v = physics_preds_dict[k]
                if len(v.shape) == 5 or (len(v.shape) == 4 and v.shape[1] != self.coords.nodal_shape[0]):
                    physics_preds_dict.update(
                        {f"{k}.{i}": v[..., i] for i in range(v.shape[-1])}
                    )
                    del physics_preds_dict[k]

        pred_ds = self.data_to_xarray(dynamics_predictions.asdict() | physics_preds_dict)
        
        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = (
            (self.start_date.delta.days + pred_ds.time - self.save_interval.m)*(timedelta64(1, 'D')/timedelta64(1, 'ns'))
        ).astype('datetime64[ns]')
        
        return pred_ds
