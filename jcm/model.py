import dinosaur.primitive_equations_states
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal
import jax
import jax.numpy as jnp
import numpy as np
import dinosaur
from dinosaur.scales import SI_SCALE, units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from jcm.boundaries import initialize_boundaries, default_boundaries, update_boundaries_with_timestep
from jcm.date import Timestamp, Timedelta
from jcm.params import Parameters
from jcm.geometry import sigma_layer_boundaries, Geometry

PHYSICS_SPECS = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

def get_speedy_physics_terms(sea_coupling_flag=0, checkpoint_terms=True):
    """
    Returns a list of functions that compute physical tendencies for the model.
    """
    
    from jcm.humidity import spec_hum_to_rel_hum
    from jcm.convection import get_convection_tendencies
    from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
    from jcm.shortwave_radiation import get_shortwave_rad_fluxes, clouds
    from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
    from jcm.surface_flux import get_surface_fluxes
    from jcm.vertical_diffusion import get_vertical_diffusion_tend
    from jcm.land_model import couple_land_atm
    from jcm.forcing import set_forcing

    physics_terms = [
        set_forcing,
        spec_hum_to_rel_hum,
        get_convection_tendencies,
        get_large_scale_condensation_tendencies,
        clouds,
        get_shortwave_rad_fluxes,
        get_downward_longwave_rad_fluxes,
        get_surface_fluxes,
        get_upward_longwave_rad_fluxes,
        get_vertical_diffusion_tend,
        couple_land_atm # eventually couple sea model and ice model here
    ]
    if sea_coupling_flag > 0:
        physics_terms.insert(-3, get_surface_fluxes)

    if not checkpoint_terms:
        return physics_terms

    static_argnums = {
        set_forcing: (2,),
        couple_land_atm: (3,),
    }
    
    return [jax.checkpoint(term, static_argnums=static_argnums.get(term, ())) for term in physics_terms]

def convert_tendencies_to_equation(dynamics, time_step, physics_terms, reference_date, boundaries, parameters, geometry):
    from jcm.physics_data import PhysicsData
    from jcm.physics import get_physical_tendencies
    from jcm.date import DateData

    def physical_tendencies(state):
        
        date = DateData.set_date(
            model_time = reference_date + Timedelta(
                seconds=state.sim_time
            )
        )

        data = PhysicsData.zeros(
            dynamics.coords.nodal_shape[1:],
            dynamics.coords.nodal_shape[0],
            date=date
        )

        return get_physical_tendencies(state=state, dynamics=dynamics, time_step=time_step, physics_terms=physics_terms, boundaries=boundaries, parameters=parameters, geometry=geometry, data=data)
    return ExplicitODE.from_functions(physical_tendencies)

def get_coords(layers=8, horizontal_resolution=31, physics_specs=PHYSICS_SPECS):
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
        horizontal=resolution_map[horizontal_resolution](radius=physics_specs.radius), # truncation 
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(sigma_layer_boundaries[layers])
    )

class SpeedyModel:
    """
    Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet.

    #TODO: Factor out the geography and physics choices so you can choose independent of each other.
    """

    def __init__(self, time_step=10, save_interval=10, total_time=1200, start_date=None,
                 layers=8, horizontal_resolution=31, coords=None,
                 boundary_data=None, physics_specs=None,
                 parameters=None, post_process=True, checkpoint_terms=True) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
                
        Args:
            time_step: Model time step in minutes
            save_interval: Save interval in days
            total_time: Total integration time in days
            layers: Number of vertical layers
            start_date: Start date of the simulation
            boundary_file: Path to the boundary conditions file including land-sea mask and albedo
            horizontal_resolution: Horizontal resolution of the model (31, 42, 85, 213)
            parameters: Model parameters
            post_process: Whether to post-process the model output

        """
        from datetime import datetime


        # Integration settings
        self.start_date = start_date or Timestamp.from_datetime(datetime(2000, 1, 1))
        self.save_interval = save_interval * units.day
        self.total_time = total_time * units.day
        dt_si = time_step * units.minute

        self.physics_specs = physics_specs or PHYSICS_SPECS
        if coords is not None:
            self.coords = coords
            horizontal_resolution = coords.horizontal.total_wavenumbers - 2
        else:
            self.coords = get_coords(layers=layers, horizontal_resolution=horizontal_resolution, physics_specs=self.physics_specs)
        self.geometry = Geometry.from_coords(self.coords)

        self.inner_steps = int(self.save_interval / dt_si)
        self.outer_steps = int(self.total_time / self.save_interval)
        self.dt = self.physics_specs.nondimensionalize(dt_si)

        self.parameters = parameters or Parameters.default()
        self.post_process_physics = post_process

        # Get the reference temperature and orography. This also returns the initial state function (if wanted to start from rest)
        p0 = 100e3 * units.pascal
        p1 = 5e3 * units.pascal

        self.initial_state_fn, aux_features = dinosaur.primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self.physics_specs,
            p0=p0,
            p1=p1
        )
        
        self.ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        
        # this implicitly calls initialize_modules, must be before we set boundaries
        self.physics_terms = get_speedy_physics_terms(checkpoint_terms=checkpoint_terms)
        
        # TODO: make the truncation number a parameter consistent with the grid shape
        if boundary_data is None:
            truncated_orography = dinosaur.primitive_equations.truncated_modal_orography(aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords)
            self.boundaries = default_boundaries(self.coords.horizontal, truncated_orography, self.parameters, time_step=dt_si)
        else:
            self.boundaries = update_boundaries_with_timestep(boundary_data, self.parameters, dt_si)
            truncated_orography = self.coords.horizontal.to_modal(
                self.physics_specs.nondimensionalize(
                    self.boundaries.phis0 * units.meter ** 2 / units.second ** 2
                )
            )
        
        self.primitive = dinosaur.primitive_equations.PrimitiveEquationsWithTime(
            self.ref_temps,
            truncated_orography * 1e-3, #FIXME: currently prevents blowup when using 'realistic' boundary conditions
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

    def get_initial_state(self, random_seed=0, sim_time=0.0, humidity_perturbation=False) -> dinosaur.primitive_equations.State:
        state = self.initial_state_fn(jax.random.PRNGKey(random_seed))
        state.log_surface_pressure = state.log_surface_pressure * 1e-3
        state.tracers = {
            'specific_humidity': (1e-2 if humidity_perturbation else 0) * primitive_equations_states.gaussian_scalar(self.coords, self.physics_specs)
        }
        return dinosaur.primitive_equations.State(**state.asdict(), sim_time=sim_time)
    
    def post_process(self, state):
        from jcm.date import DateData
        from jcm.physics_data import PhysicsData
        from jcm.physics import dynamics_state_to_physics_state

        date = DateData.set_date(
            model_time = self.start_date + Timedelta(
                seconds=state.sim_time
                )
        )

        data = PhysicsData.zeros(
            self.coords.nodal_shape[1:],
            self.coords.nodal_shape[0],
            date=date
        )

        if self.post_process_physics:
            physics_state = dynamics_state_to_physics_state(state, self.primitive)
            for term in self.physics_terms:
                _, data = term(physics_state, data, self.parameters, self.boundaries, self.geometry)
        else:
            pass # Return an empty physics data object

        return {
            'dynamics': state,
            'physics': data,
        }
    
    def unroll(self, state: dinosaur.primitive_equations.State) -> tuple[dinosaur.primitive_equations.State, dinosaur.primitive_equations.State]:
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
        # extract dynamics predictions (State format)
        # and physics predictions (PhysicsData format) from postprocessed output
        dynamics_predictions = predictions['dynamics']
        physics_predictions = predictions['physics']

        # prepare dynamics predictions for xarray conversion:
        # convert from modal to nodal, and dimensionalize
        u_nodal, v_nodal = vor_div_to_uv_nodal(self.coords.horizontal,
                                                dynamics_predictions.vorticity,
                                                dynamics_predictions.divergence)
        # TODO: compute w_nodal and add to dataset - vertical velocity function only accepts a State rather than predictions (set of States at multiple times) so this doesn't work
        # w_nodal = -primitive_equations.compute_vertical_velocity(dynamics_predictions, self.coords)
        log_surface_pressure_nodal = jnp.squeeze(self.coords.horizontal.to_nodal(dynamics_predictions.log_surface_pressure), axis=1)
        surface_pressure_nodal = jnp.exp(log_surface_pressure_nodal) * 1e5
        diagnostic_state_preds = primitive_equations.compute_diagnostic_state(dynamics_predictions, self.coords)

        # dimensionalize
        diagnostic_state_preds.temperature_variation += self.ref_temps[:, jnp.newaxis, jnp.newaxis]
        diagnostic_state_preds.tracers['specific_humidity'] = self.physics_specs.dimensionalize(diagnostic_state_preds.tracers['specific_humidity'], units.gram / units.kilogram).m

        # prepare physics predictions for xarray conversion:
        # unpack into single dictionary, and unpack individual fields
        # unpack PhysicsData struct
        physics_state_preds = {
            f"{module}.{field}": value # avoids name conflicts between fields of different modules
            for module, module_dict in physics_predictions.asdict().items()
            for field, value in module_dict.asdict().items()
        }
        # replace multi-channel fields with a field for each channel
        _original_keys = list(physics_state_preds.keys())
        for k in _original_keys:
            v = physics_state_preds[k]
            if len(v.shape) == 5 or (len(v.shape) == 4 and v.shape[1] != self.coords.nodal_shape[0]):
                physics_state_preds.update(
                    {f"{k}.{i}": v[..., i] for i in range(v.shape[-1])}
                )
                del physics_state_preds[k]

        # create xarray dataset
        nodal_predictions = {
            **diagnostic_state_preds.asdict(),
            **physics_state_preds
        }
        broken_keys = ['cos_lat_u', 'cos_lat_grad_log_sp', 'cos_lat_grad_log_sp', ] # These are tuples which are not supported by xarray
        broken_keys += ['sigma_dot_explicit', 'sigma_dot_full'] # These have one less time step for some reason...
        pred_ds = self.data_to_xarray({k: v for k, v in nodal_predictions.items() if k not in broken_keys})
        pred_ds = pred_ds.rename_vars({'temperature_variation': 'temperature'})
        pred_ds['u'] = self.data_to_xarray({'u': u_nodal})['u']
        pred_ds['v'] = self.data_to_xarray({'v': v_nodal})['v']
        pred_ds['surface_pressure'] = self.data_to_xarray({'surface_pressure': surface_pressure_nodal})['surface_pressure']
        
        # Flip the vertical dimension so that it goes from the surface to the top of the atmosphere
        pred_ds = pred_ds.isel(level=slice(None, None, -1))

        # convert time in days to datetime
        pred_ds['time'] = ((self.start_date.delta.days + pred_ds.time - self.save_interval.m)*(np.timedelta64(1, 'D')/np.timedelta64(1, 'ns'))).astype('datetime64[ns]')
        
        return pred_ds
