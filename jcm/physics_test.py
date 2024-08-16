import unittest
from jcm.physics import get_physical_tendencies
from jcm.held_suarez_forcing import held_suarez_forcings
from collections import abc
import jax
import jax.numpy as jnp
import tree_math
from typing import Callable

from dinosaur import scales
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.spherical_harmonic import vor_div_to_uv_nodal, uv_nodal_to_vor_div_modal
from dinosaur.primitive_equations import get_geopotential, State, PrimitiveEquations, PrimitiveEquationsSpecs
from dinosaur.sigma_coordinates import centered_vertical_advection

class TestPhysicsUnit(unittest.TestCase):

    def test_get_physical_tendencies(self):
        #define some dimensions 
        ix = 5

        # Creating a state with sample data
        vorticity = jnp.zeros(ix)  # fill with appropriate data
        divergence = jnp.zeros(ix) 
        temperature_variation = jnp.zeros(ix) 
        log_surface_pressure = jnp.zeros(ix) 
        tracers = {'humidity': jnp.zeros(ix) , 'pollutant': jnp.zeros(ix) }

        state = State(vorticity=vorticity,
                    divergence=divergence,
                    temperature_variation=temperature_variation,
                    log_surface_pressure=log_surface_pressure,tracers=tracers) #State
        ##############################################################

        reference_temperature = jnp.zeros(ix)
        orography = jnp.zeros(ix)
        # Create an instance of CoordinateSystem (example)
        coords = CoordinateSystem(horizontal_shape=(10, 10), vertical_layers=ix)
        # Create an instance of PrimitiveEquationsSpecs (example)
        physics_specs = PrimitiveEquationsSpecs(radius=1.0, angular_velocity=1.0,
            gravity_acceleration=9.81, ideal_gas_constant=287.05,
            water_vapor_gas_constant=461.5,
            water_vapor_isobaric_heat_capacity=1005,
            kappa=0.286,
            scale=scales.Scale)

        # Choose a vertical multiplication method
        vertical_matmul_method = 'dense'  # or 'sparse'

        # Define a vertical advection function (optional, using default here)
        vertical_advection = centered_vertical_advection

        # Include vertical advection (optional)
        include_vertical_advection = True

        # Instantiate the PrimitiveEquations object
        dynamics = PrimitiveEquations(
            reference_temperature=reference_temperature,
            orography=orography,
            coords=coords,
            physics_specs=physics_specs,
            vertical_matmul_method=vertical_matmul_method,
            vertical_advection=vertical_advection,
            include_vertical_advection=include_vertical_advection)

        physics_terms = [ held_suarez_forcings ] #abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]

        dynamics_tendency = get_physical_tendencies(state,dynamics,physics_terms)

        self.assertIsNotNone(dynamics_tendency)



    # def test_held_suarez_forcing(self):
    #     units = scales.units
    #     layers = 26
    #     coords = coordinate_systems.CoordinateSystem(
    #         horizontal=spherical_harmonic.Grid.T42(),
    #         vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
    #     physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()

    #     initial_state_fn, aux_features = (
    #         primitive_equations_states.isothermal_rest_atmosphere(
    #             coords, physics_specs, p0=1e5 * units.pascal,
    #             p1=5e3 * units.pascal,))
    #     ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    #     state = initial_state_fn(rng_key=jax.random.PRNGKey(0))

    #     hs = held_suarez.HeldSuarezForcing(
    #         coords=coords,
    #         physics_specs=physics_specs,
    #         reference_temperature=ref_temps)

    #     self.assertEqual(hs.kv().shape, (coords.vertical.layers, 1, 1))
    #     self.assertEqual(hs.kt().shape, coords.nodal_shape)

    #     surface_pressure = np.ones(coords.nodal_shape)
    #     self.assertEqual(hs.equilibrium_temperature(surface_pressure).shape,
    #                     coords.nodal_shape)

    #     explicit_terms = hs.explicit_terms(state)
    #     np.testing.assert_allclose(explicit_terms.log_surface_pressure, 0)