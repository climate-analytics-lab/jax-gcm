import jax
import jax.numpy as jnp
import dinosaur
from dinosaur.scales import units
from dinosaur import primitive_equations_states
from dinosaur.time_integration import ExplicitODE
from dinosaur.primitive_equations import State
from datetime import datetime

def convert_tendencies_to_equation(dynamics, physics_terms, reference_date, dt):
    from jcm.physics_data import PhysicsData
    from jcm.physics import get_physical_tendencies
    def physical_tendencies(state):            
        from jcm.date import DateData
        model_time = reference_date #+ state.sim_time * units.second
        model_steps = jnp.round(state.sim_time * units.second / dt)
        data = PhysicsData(dynamics.coords.nodal_shape[1:],
                    dynamics.coords.nodal_shape[0],
                    date_data=DateData(model_time, model_steps))
        
        # Remove the sim_time and convert to a plain State object
        _state = state.asdict()
        _state.pop('sim_time')
        state = State(**_state)

        return get_physical_tendencies(state, dynamics, physics_terms, data)
    return ExplicitODE.from_functions(physical_tendencies)

def initialize_modules(kx=8, il=64):
    from jcm.geometry import initialize_geometry
    initialize_geometry(kx=kx, il=il)
    from jcm.physics import initialize_physics
    initialize_physics()

class SpeedyModel:
    """
    Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet.

    #TODO: Factor out the geography and physics choices so you can choose independent of each other.
    """

    def __init__(self, time_step=10, save_interval=10, total_time=1200, layers=8, start_date=None) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
                
        Args:
            time_step: Model time step in minutes
            save_interval: Save interval in days
            total_time: Total integration time in days
            layers: Number of vertical layers
            start_date: Start date of the simulation

        """

        # Integration settings
        start_date = start_date or datetime(2000, 1, 1)
        dt_si = time_step * units.minute
        save_every = save_interval * units.day
        total_time = total_time * units.day

        # Define the coordinate system
        self.coords = dinosaur.coordinate_systems.CoordinateSystem(
            horizontal=dinosaur.spherical_harmonic.Grid.T42(),
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
            p1=p1)
        
        ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        orography = dinosaur.primitive_equations.truncated_modal_orography(
            aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords)

        # Governing equations
        primitive = dinosaur.primitive_equations.PrimitiveEquationsWithTime(
            ref_temps,
            orography,
            self.coords,
            self.physics_specs)
        
        initialize_modules(kx = self.coords.nodal_shape[0],
                           il = self.coords.nodal_shape[2])

        from jcm.humidity import spec_hum_to_rel_hum
        from jcm.convection import get_convection_tendencies
        from jcm.large_scale_condensation import get_large_scale_condensation_tendencies
        from jcm.shortwave_radiation import get_shortwave_rad_fluxes, clouds
        from jcm.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
        from jcm.surface_flux import get_surface_fluxes
        from jcm.vertical_diffusion import get_vertical_diffusion_tend

        physics_terms = [
            spec_hum_to_rel_hum, # this could get called in get_physics_tendencies before looping over the physics terms
            get_convection_tendencies,
            get_large_scale_condensation_tendencies,
            clouds,
            get_shortwave_rad_fluxes,
            get_downward_longwave_rad_fluxes,
            get_surface_fluxes, # In speedy this gets called again if air-sea coupling is on
            get_upward_longwave_rad_fluxes,
            get_vertical_diffusion_tend
        ]
        speedy_forcing = convert_tendencies_to_equation(primitive, physics_terms, start_date, dt_si)

        self.primitive_with_hs = dinosaur.time_integration.compose_equations([primitive, speedy_forcing])

        # Define trajectory times, expects start_with_input=False
        self.times = save_every * jnp.arange(1, self.outer_steps+1)

        step_fn = dinosaur.time_integration.imex_rk_sil3(self.primitive_with_hs, self.dt)
        filters = [
            dinosaur.time_integration.exponential_step_filter(
                self.coords.horizontal, self.dt, tau=0.0087504, order=1.5, cutoff=0.8),
        ]

        self.step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)
        
    def get_initial_state(self, random_seed=0, sim_time=0.0) -> dinosaur.primitive_equations.StateWithTime:
        state = self.initial_state_fn(jax.random.PRNGKey(random_seed))
        return dinosaur.primitive_equations.StateWithTime(**state.asdict(), sim_time=sim_time)

    def advance(self, state: dinosaur.primitive_equations.StateWithTime) -> dinosaur.primitive_equations.State:
        return self.step_fn(state)
                                 
    def unroll(self, state: dinosaur.primitive_equations.StateWithTime) -> tuple[dinosaur.primitive_equations.StateWithTime, dinosaur.primitive_equations.StateWithTime]:
        integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
            self.step_fn,
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps))
        return integrate_fn(state)

