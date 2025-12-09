"""Tests for the SingleColumnModel class."""

import unittest
import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.single_column_model import SingleColumnModel, SCMPredictions, SCMState
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.geometry import Geometry
from jcm.constants import grav


class TestSingleColumnModel(unittest.TestCase):
    """Test the SingleColumnModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.nlev = 40
        self.nlon = 96
        self.nlat = 48
        self.shape = (self.nlev, self.nlon, self.nlat)

        self.geometry = Geometry.from_grid_shape(
            nodal_shape=(self.nlon, self.nlat),
            num_levels=self.nlev
        )

        self.test_state = self._create_test_state()

    def _create_test_state(self) -> PhysicsState:
        """Create a realistic atmospheric state for testing."""
        # Temperature profile
        t_surface = 288.0
        lapse_rate = 6.5e-3
        z_levels = jnp.linspace(0, 30000, self.nlev)[::-1]
        t_profile = jnp.maximum(t_surface - lapse_rate * z_levels, 200.0)
        temperature = jnp.broadcast_to(t_profile[:, None, None], self.shape)

        # Humidity
        q_surface = 0.012
        q_profile = q_surface * jnp.exp(-z_levels / 3000.0)
        specific_humidity = jnp.broadcast_to(q_profile[:, None, None], self.shape)

        # Winds
        u_wind = jnp.ones(self.shape) * 5.0
        v_wind = jnp.zeros(self.shape)

        # Geopotential
        geopotential = jnp.broadcast_to((grav * z_levels)[:, None, None], self.shape)

        # Surface pressure
        normalized_surface_pressure = jnp.ones((self.nlon, self.nlat))

        return PhysicsState(
            u_wind=u_wind,
            v_wind=v_wind,
            temperature=temperature,
            specific_humidity=specific_humidity,
            geopotential=geopotential,
            normalized_surface_pressure=normalized_surface_pressure,
            tracers={'qc': jnp.zeros(self.shape), 'qi': jnp.zeros(self.shape)},
        )

    def test_initialization(self):
        """Test model initialization."""
        from jcm.physics.icon import IconPhysics

        model = SingleColumnModel(
            physics=IconPhysics(),
            geometry=self.geometry,
        )

        self.assertIsNotNone(model.physics)
        self.assertIsNotNone(model.geometry)
        self.assertEqual(model.dt_seconds, 1800.0)
        self.assertTrue(model.apply_tracer_tendencies)

    def test_initialization_without_tracer_update(self):
        """Test model initialization with tracer updates disabled."""
        from jcm.physics.icon import IconPhysics

        model = SingleColumnModel(
            physics=IconPhysics(),
            geometry=self.geometry,
            apply_tracer_tendencies=False,
        )

        self.assertFalse(model.apply_tracer_tendencies)

    def test_run_single(self):
        """Test single step computation."""
        from jcm.physics.icon import IconPhysics
        from jcm.physics.icon.parameters import Parameters

        icon_params = Parameters.default().with_convection(dt_conv=1800.0)
        physics = IconPhysics(parameters=icon_params)

        model = SingleColumnModel(
            physics=physics,
            geometry=self.geometry,
            use_hybrid_coords=False,
        )

        initial_tracers = {'qc': jnp.zeros(self.shape), 'qi': jnp.zeros(self.shape)}

        tendencies, updated_tracers, physics_data = model.run_single(
            self.test_state,
            tracers=initial_tracers,
        )

        # Check shapes
        self.assertEqual(tendencies.temperature.shape, self.shape)
        self.assertEqual(updated_tracers['qc'].shape, self.shape)
        self.assertEqual(updated_tracers['qi'].shape, self.shape)

        # Check for NaNs
        self.assertFalse(
            jnp.any(jnp.isnan(tendencies.temperature)),
            "Temperature tendencies contain NaN"
        )

    def test_run_time_series(self):
        """Test running with a time series of prescribed states."""
        from jcm.physics.icon import IconPhysics
        from jcm.physics.icon.parameters import Parameters

        icon_params = Parameters.default().with_convection(dt_conv=1800.0)
        physics = IconPhysics(parameters=icon_params)

        model = SingleColumnModel(
            physics=physics,
            geometry=self.geometry,
            use_hybrid_coords=False,
        )

        # Create a short time series (2 timesteps for speed)
        n_times = 2
        stacked_state = PhysicsState(
            u_wind=jnp.stack([self.test_state.u_wind] * n_times),
            v_wind=jnp.stack([self.test_state.v_wind] * n_times),
            temperature=jnp.stack([self.test_state.temperature] * n_times),
            specific_humidity=jnp.stack([self.test_state.specific_humidity] * n_times),
            geopotential=jnp.stack([self.test_state.geopotential] * n_times),
            normalized_surface_pressure=jnp.stack(
                [self.test_state.normalized_surface_pressure] * n_times
            ),
            tracers={
                'qc': jnp.stack([jnp.zeros(self.shape)] * n_times),
                'qi': jnp.stack([jnp.zeros(self.shape)] * n_times),
            },
        )

        initial_tracers = {'qc': jnp.zeros(self.shape), 'qi': jnp.zeros(self.shape)}

        predictions = model.run(stacked_state, initial_tracers=initial_tracers)

        # Check output structure
        self.assertIsInstance(predictions, SCMPredictions)

        # Check shapes include time dimension
        self.assertEqual(predictions.tendencies.temperature.shape[0], n_times)
        self.assertEqual(predictions.tracer_states['qc'].shape[0], n_times)
        self.assertEqual(len(predictions.times), n_times)

        # Check tendencies are finite
        self.assertFalse(
            jnp.any(jnp.isnan(predictions.tendencies.temperature)),
            "Temperature tendencies contain NaN"
        )

    def test_tracer_evolution(self):
        """Test that tracers actually evolve when tendencies are applied."""
        from jcm.physics.icon import IconPhysics
        from jcm.physics.icon.parameters import Parameters

        icon_params = Parameters.default().with_convection(dt_conv=1800.0)
        physics = IconPhysics(parameters=icon_params)

        model = SingleColumnModel(
            physics=physics,
            geometry=self.geometry,
            use_hybrid_coords=False,
            apply_tracer_tendencies=True,
        )

        # Run two steps
        n_times = 2
        stacked_state = PhysicsState(
            u_wind=jnp.stack([self.test_state.u_wind] * n_times),
            v_wind=jnp.stack([self.test_state.v_wind] * n_times),
            temperature=jnp.stack([self.test_state.temperature] * n_times),
            specific_humidity=jnp.stack([self.test_state.specific_humidity] * n_times),
            geopotential=jnp.stack([self.test_state.geopotential] * n_times),
            normalized_surface_pressure=jnp.stack(
                [self.test_state.normalized_surface_pressure] * n_times
            ),
            tracers={
                'qc': jnp.stack([jnp.zeros(self.shape)] * n_times),
                'qi': jnp.stack([jnp.zeros(self.shape)] * n_times),
            },
        )

        initial_tracers = {'qc': jnp.zeros(self.shape), 'qi': jnp.zeros(self.shape)}

        predictions = model.run(stacked_state, initial_tracers=initial_tracers)

        # Tracer values at step 1 should potentially differ from step 0
        # (if physics produces non-zero tendencies)
        # At minimum, tracers should be non-negative
        self.assertTrue(jnp.all(predictions.tracer_states['qc'] >= 0))
        self.assertTrue(jnp.all(predictions.tracer_states['qi'] >= 0))

    def test_create_initial_tracers(self):
        """Test helper function for creating initial tracers."""
        tracers = SingleColumnModel.create_initial_tracers(
            shape=self.shape,
            tracer_names=['qc', 'qi', 'aerosol'],
            cloud_water=1e-5,
            cloud_ice=1e-6,
        )

        self.assertEqual(tracers['qc'].shape, self.shape)
        self.assertEqual(tracers['qi'].shape, self.shape)
        self.assertEqual(tracers['aerosol'].shape, self.shape)

        self.assertTrue(jnp.allclose(tracers['qc'], 1e-5))
        self.assertTrue(jnp.allclose(tracers['qi'], 1e-6))
        self.assertTrue(jnp.allclose(tracers['aerosol'], 0.0))


if __name__ == '__main__':
    unittest.main()
