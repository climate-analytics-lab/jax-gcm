import dinosaur.primitive_equations_states
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal
import jax
import jax.numpy as jnp
import dinosaur
from dinosaur.scales import units
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from jcm.date import Timestamp, Timedelta
from jcm.params import Parameters

def initialize_modules(kx=8, il=64):
    from jcm.geometry import initialize_geometry
    initialize_geometry(kx=kx, il=il)
    from jcm.physics import initialize_physics
    initialize_physics()

def get_speedy_physics_terms(grid_shape, sea_coupling_flag=0):
    """
    Returns a list of functions that compute physical tendencies for the model.
    """
    initialize_modules(kx = grid_shape[0], il = grid_shape[2])
    
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
    return physics_terms

def fixed_ssts(ix):
    """
    Returns an array of SSTs with simple cos^2 profile from 300K at the equator to 273K at 60 degrees latitude.
    Obtained from Neale, R.B. and Hoskins, B.J. (2000), "A standard test for AGCMs including their physical parametrizations: I: the proposal." Atmosph. Sci. Lett., 1: 101-107. https://doi.org/10.1006/asle.2000.0022
    """
    from jcm.geometry import radang
    sst_profile = jnp.where(jnp.abs(radang) < jnp.pi/3, 27*jnp.cos(3*radang/2)**2, 0) + 273.15
    return jnp.tile(sst_profile[jnp.newaxis, :], (ix, 1))

#  add boundaries argument
def convert_tendencies_to_equation(dynamics, time_step, physics_terms, reference_date, boundaries, parameters):
    from jcm.physics_data import PhysicsData, SeaModelData
    from jcm.physics import get_physical_tendencies
    from jcm.date import DateData

    def physical_tendencies(state):
        
        date = DateData.set_date(
            model_time = reference_date + Timedelta(
                seconds=dynamics.physics_specs.dimensionalize(state.sim_time, units.second).m
            )
        )

        sea_model = SeaModelData.zeros(
            dynamics.coords.nodal_shape[1:],
            tsea = fixed_ssts(dynamics.coords.nodal_shape[1])
        )

        data = PhysicsData.zeros(
            dynamics.coords.nodal_shape[1:],
            dynamics.coords.nodal_shape[0],
            date=date,
            sea_model=sea_model
        )

        return get_physical_tendencies(state, dynamics, time_step, physics_terms, data, boundaries, parameters)
    return ExplicitODE.from_functions(physical_tendencies)

class SpeedyModel:
    """
    Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet.

    #TODO: Factor out the geography and physics choices so you can choose independent of each other.
    """

    def __init__(self, time_step=10, save_interval=10, total_time=1200, layers=8, 
                 start_date=None, boundary_file='boundaries.nc', parameters=None) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
                
        Args:
            time_step: Model time step in minutes
            save_interval: Save interval in days
            total_time: Total integration time in days
            layers: Number of vertical layers
            start_date: Start date of the simulation
            boundary_file: Path to the boundary conditions file including land-sea mask and albedo

        """
        from datetime import datetime
        from jcm.boundaries import initialize_boundaries

        # Integration settings
        self.start_date = start_date or Timestamp.from_datetime(datetime(2000, 1, 1))
        dt_si = time_step * units.minute
        save_every = save_interval * units.day
        total_time = total_time * units.day

        # Define the coordinate system
        self.coords = dinosaur.coordinate_systems.CoordinateSystem(
            horizontal=dinosaur.spherical_harmonic.Grid.T42(), # truncation 
            vertical=dinosaur.sigma_coordinates.SigmaCoordinates.equidistant(layers))
        
        # Not sure why we need this...
        self.physics_specs = dinosaur.primitive_equations.PrimitiveEquationsSpecs.from_si()

        self.inner_steps = int(save_every / dt_si)
        self.outer_steps = int(total_time / save_every)
        self.dt = self.physics_specs.nondimensionalize(dt_si)

        # Get the reference temerature and orography. This also returns the initial state function (if wanted to start from rest)
        p0 = 100e3 * units.pascal
        p1 = 5e3 * units.pascal

        self.initial_state_fn, aux_features = dinosaur.primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self.physics_specs,
            p0=p0,
            p1=p1
        )
        
        self.ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        orography = dinosaur.primitive_equations.truncated_modal_orography(
            aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords)

        # Governing equations
        self.primitive = dinosaur.primitive_equations.PrimitiveEquationsWithTime(
            self.ref_temps,
            orography,
            self.coords,
            self.physics_specs)
        
        self.physics_terms = get_speedy_physics_terms(self.coords.nodal_shape)

        # TODO: make the truncation number a parameter consistent with the grid shape        
        boundaries = initialize_boundaries(boundary_file, self.primitive, 42)

        speedy_forcing = convert_tendencies_to_equation(self.primitive, time_step, 
                                                        self.physics_terms, self.start_date, 
                                                        boundaries, parameters)

        self.primitive_with_speedy = dinosaur.time_integration.compose_equations([self.primitive, speedy_forcing])

        # Define trajectory times, expects start_with_input=False
        self.times = save_every * jnp.arange(1, self.outer_steps+1)

        step_fn = dinosaur.time_integration.imex_rk_sil3(self.primitive_with_speedy, self.dt)
        filters = [
            dinosaur.time_integration.exponential_step_filter(
                self.coords.horizontal, self.dt, tau=0.0087504, order=1.5, cutoff=0.8),
        ]

        self.step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)

    def get_initial_state(self, random_seed=0, sim_time=0.0) -> dinosaur.primitive_equations.StateWithTime:
        state = self.initial_state_fn(jax.random.PRNGKey(random_seed))
        state.log_surface_pressure = state.log_surface_pressure * 1e-3
        state.tracers = {
            'specific_humidity': 1e-2 * primitive_equations_states.gaussian_scalar(self.coords, self.physics_specs)
        }
        return dinosaur.primitive_equations.StateWithTime(**state.asdict(), sim_time=sim_time)

    def advance(self, state: dinosaur.primitive_equations.StateWithTime) -> dinosaur.primitive_equations.StateWithTime:
        return self.step_fn(state)
    
    def post_process(self, state):
        from jcm.date import DateData
        from jcm.physics_data import PhysicsData, SeaModelData
        from jcm.physics import dynamics_state_to_physics_state

        date = DateData.set_date(
            model_time = self.start_date + Timedelta(
                seconds=self.physics_specs.dimensionalize(state.sim_time, units.second).m
                )
        )

        # TODO: factor this out into boundary conditions object
        sea_model = SeaModelData.zeros(
            self.coords.nodal_shape[1:],
            tsea = fixed_ssts(self.coords.nodal_shape[1])
        )

        data = PhysicsData.zeros(
            self.coords.nodal_shape[1:],
            self.coords.nodal_shape[0],
            date=date,
            sea_model=sea_model
        )

        physics_state = dynamics_state_to_physics_state(state, self.primitive)
        for term in self.physics_terms:
            _, data = term(physics_state, data)
        
        return {
            'dynamics': state,
            'physics': data,
        }
    
    def unroll(self, state: dinosaur.primitive_equations.StateWithTime) -> tuple[dinosaur.primitive_equations.StateWithTime, dinosaur.primitive_equations.StateWithTime]:
        integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
            self.step_fn,
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
        from dinosaur.xarray_utils import data_to_xarray

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
        log_surface_pressure_nodal = jnp.squeeze(self.coords.horizontal.to_nodal(dynamics_predictions.log_surface_pressure))
        surface_pressure_nodal = jnp.exp(log_surface_pressure_nodal) * 1e5
        diagnostic_state_preds = primitive_equations.compute_diagnostic_state(dynamics_predictions, self.coords)

        # dimensionalize
        u_nodal = self.physics_specs.dimensionalize(jnp.asarray(u_nodal), units.meter / units.second).m
        v_nodal = self.physics_specs.dimensionalize(jnp.asarray(v_nodal), units.meter / units.second).m
        diagnostic_state_preds.temperature_variation = (288 * units.kelvin + self.physics_specs.dimensionalize(diagnostic_state_preds.temperature_variation, units.kelvin)).m
        diagnostic_state_preds.tracers['specific_humidity'] = self.physics_specs.dimensionalize(diagnostic_state_preds.tracers['specific_humidity'], units.gram / units.kilogram).m

        # prepare physics predictions for xarray conversion:
        # unpack into single dictionary, and unpack/transpose individual fields
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
            if len(v.shape) == 5 or (len(v.shape) == 4 and v.shape[-1] != self.coords.nodal_shape[0]):
                physics_state_preds.update(
                    {f"{k}.{i}": v[..., i] for i in range(v.shape[-1])}
                )
                del physics_state_preds[k]
        # convert x, y, z to z, x, y to match dinosaur dimension ordering
        for k, v in physics_state_preds.items():
            if v.shape[1:4] == self.coords.nodal_shape[1:] + (self.coords.nodal_shape[0],):
                physics_state_preds[k] = jnp.moveaxis(v, (0, 1, 2, 3), (0, 2, 3, 1))

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
        # convert integer time (in days) to datetime
        pred_ds['time'] = (self.start_date.delta.days + pred_ds.time).astype('datetime64[D]')
        
        return pred_ds
