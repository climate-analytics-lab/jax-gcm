
import dinosaur
from dinosaur.scales import units, SI_SCALE
import jax
import jax.numpy as jnp
from jcm.physics import get_physical_tendencies
from dinosaur.time_integration import ExplicitODE
from dinosaur import primitive_equations, primitive_equations_states
from jcm.held_suarez import HeldSuarezForcing
from jcm.boundaries import BoundaryData
from jcm.params import Parameters
from jcm.geometry import Geometry

PHYSICS_SPECS = primitive_equations.PrimitiveEquationsSpecs.from_si(scale = SI_SCALE)

def convert_tendencies_to_equation(dynamics, time_step, physics_terms):
    def physical_tendencies(state):

        parameters = Parameters.default()

        # Held Suarez doesn't use these boundaries, so it doesn't matter what they are
        boundaries = BoundaryData.zeros((1,1))

        geometry = Geometry.from_grid_shape((1,1), 8)

        return get_physical_tendencies(state, dynamics, time_step, physics_terms,
                                       boundaries, parameters, geometry)
    return ExplicitODE.from_functions(physical_tendencies)

class HeldSuarezModel:
    """
    Top level class for a JAX-GCM configuration using the Speedy physics on an aquaplanet.

    #TODO: Factor out the geography and physics choices so you can choose independent of each other.
    """

    def __init__(self, time_step=10, save_interval=10, total_time=1200, layers=8) -> None:
        """
        Initialize the model with the given time step, save interval, and total time.
        
        Args:
            time_step: Model time step in minutes
            save_interval: Save interval in days
            total_time: Total integration time in days
            layers: Number of vertical layers
        """

        # Integration settings
        dt_si = time_step * units.minute
        save_every = save_interval * units.day
        total_time = total_time * units.day

        self.physics_specs = PHYSICS_SPECS

        # Define the coordinate system
        self.coords = dinosaur.coordinate_systems.CoordinateSystem(
            horizontal=dinosaur.spherical_harmonic.Grid.T42(radius=self.physics_specs.radius),
            vertical=dinosaur.sigma_coordinates.SigmaCoordinates.equidistant(layers))
        
        self.inner_steps = int(save_every / dt_si)
        self.outer_steps = int(total_time / save_every)
        self.dt = self.physics_specs.nondimensionalize(dt_si)

        # Get the reference temerature and orography.
        # This also returns the initial state function (if wanted to start from rest)
        p0 = 100e3 * units.pascal
        p1 = 5e3 * units.pascal

        self.initial_state_fn, aux_features = primitive_equations_states.isothermal_rest_atmosphere(
            coords=self.coords,
            physics_specs=self.physics_specs,
            p0=p0,
            p1=p1
        )
        
        self.ref_temps = aux_features[dinosaur.xarray_utils.REF_TEMP_KEY]
        self.orography = primitive_equations.truncated_modal_orography(
            aux_features[dinosaur.xarray_utils.OROGRAPHY], self.coords)

        # Governing equations
        primitive = primitive_equations.PrimitiveEquations(
            self.ref_temps,
            self.orography,
            self.coords,
            self.physics_specs
        )

        hsf = HeldSuarezForcing(self.coords, self.physics_specs, self.ref_temps)

        physics_terms = [ hsf.held_suarez_forcings ]

        speedy_forcing = convert_tendencies_to_equation(primitive, time_step, physics_terms)

        self.primitive_with_hs = dinosaur.time_integration.compose_equations([primitive, speedy_forcing])

        # Define trajectory times, expects start_with_input=False
        self.times = save_every * jnp.arange(1, self.outer_steps+1)

        step_fn = dinosaur.time_integration.imex_rk_sil3(self.primitive_with_hs, self.dt)
        filters = [
            dinosaur.time_integration.exponential_step_filter(
                self.coords.horizontal, self.dt, tau=0.0087504, order=1.5, cutoff=0.8),
        ]

        self.step_fn = dinosaur.time_integration.step_with_filters(step_fn, filters)
        
    def get_initial_state(self, random_seed=0, sim_time=0.0) -> primitive_equations.State:
        state =  self.initial_state_fn(jax.random.PRNGKey(random_seed))
        return primitive_equations.State(**state.asdict(), sim_time=sim_time)

    def unroll(self, state: primitive_equations.State) -> tuple[primitive_equations.State, primitive_equations.State]:
        integrate_fn = jax.jit(dinosaur.time_integration.trajectory_from_step(
            self.step_fn,
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps))
        return integrate_fn(state)

    def data_to_xarray(self, data):
        from dinosaur.xarray_utils import data_to_xarray
        
        return data_to_xarray(data, coords=self.coords, times=self.times)