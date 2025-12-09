"""Tests for the PrescribedStateModel class."""

import unittest
import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.prescribed_state_model import PrescribedStateModel, PrescribedStatePredictions
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.geometry import Geometry
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.forcing import ForcingData


class TestPrescribedStateModel(unittest.TestCase):
    """Test the PrescribedStateModel class."""

    def setUp(self):
        """Set up test fixtures.

        Note: We use single-column geometry (1,1) to avoid tree_math pytree
        registration conflicts with other tests that use different grid sizes.
        """
        self.nlev = 8
        self.nlon = 1
        self.nlat = 1
        self.shape = (self.nlev, self.nlon, self.nlat)

        # Create a single-column geometry (compatible with SPEEDY physics tests)
        self.geometry = Geometry.single_column_geometry(num_levels=self.nlev)

        # Create forcing data for single-column (can't use default_forcing for (1,1) grid)
        self.forcing = ForcingData.ones((self.nlon, self.nlat))

        # Create a realistic test state
        self.test_state = self._create_test_state()

    def _create_test_state(self) -> PhysicsState:
        """Create a realistic atmospheric state for testing."""
        # Temperature profile (decreasing with height)
        t_surface = 288.0  # K
        lapse_rate = 6.5e-3  # K/m
        z_levels = jnp.linspace(0, 15000, self.nlev)[::-1]  # Top to bottom
        t_profile = t_surface - lapse_rate * z_levels

        # Broadcast to full shape
        temperature = jnp.broadcast_to(
            t_profile[:, None, None], self.shape
        )

        # Humidity (decreasing exponentially with height)
        q_surface = 0.015  # kg/kg
        scale_height = 2500.0  # m
        q_profile = q_surface * jnp.exp(-z_levels / scale_height)
        specific_humidity = jnp.broadcast_to(
            q_profile[:, None, None], self.shape
        )

        # Winds (simple zonal flow)
        u_wind = jnp.ones(self.shape) * 5.0  # m/s
        v_wind = jnp.zeros(self.shape)

        # Geopotential
        from jcm.constants import grav
        geopotential = jnp.broadcast_to(
            (grav * z_levels)[:, None, None], self.shape
        )

        # Surface pressure
        normalized_surface_pressure = jnp.ones((self.nlon, self.nlat))

        return PhysicsState(
            u_wind=u_wind,
            v_wind=v_wind,
            temperature=temperature,
            specific_humidity=specific_humidity,
            geopotential=geopotential,
            normalized_surface_pressure=normalized_surface_pressure,
            tracers={},
        )

    def test_initialization(self):
        """Test model initialization."""
        model = PrescribedStateModel(
            physics=SpeedyPhysics(),
            geometry=self.geometry,
        )

        self.assertIsNotNone(model.physics)
        self.assertIsNotNone(model.geometry)
        self.assertEqual(model.dt_seconds, 1800.0)

    def test_initialization_without_geometry(self):
        """Test model initialization without geometry."""
        model = PrescribedStateModel(physics=SpeedyPhysics())
        self.assertIsNone(model.geometry)

    @unittest.skip("SPEEDY physics has tree_math pytree registration issues in pytest - see test_icon_physics for full integration test")
    def test_run_single(self):
        """Test computing tendencies for a single state."""
        model = PrescribedStateModel(
            physics=SpeedyPhysics(),
            geometry=self.geometry,
        )

        tendencies, physics_data = model.run_single(self.test_state, forcing=self.forcing)

        # Check that tendencies have the right shape
        self.assertEqual(tendencies.temperature.shape, self.shape)
        self.assertEqual(tendencies.u_wind.shape, self.shape)
        self.assertEqual(tendencies.v_wind.shape, self.shape)
        self.assertEqual(tendencies.specific_humidity.shape, self.shape)

    def test_run_single_requires_geometry(self):
        """Test that run_single raises error without geometry."""
        model = PrescribedStateModel(physics=SpeedyPhysics())

        with self.assertRaises(ValueError):
            model.run_single(self.test_state)

    @unittest.skip("SPEEDY physics has tree_math pytree registration issues in pytest - see test_icon_physics for full integration test")
    def test_run_time_series(self):
        """Test computing tendencies for a time series of states."""
        model = PrescribedStateModel(
            physics=SpeedyPhysics(),
            geometry=self.geometry,
        )

        # Create a time series of states (3 timesteps)
        n_times = 3
        states = [self.test_state for _ in range(n_times)]

        predictions = model.run(states, forcing=self.forcing)

        # Check output structure
        self.assertIsInstance(predictions, PrescribedStatePredictions)

        # Check shapes include time dimension
        self.assertEqual(predictions.states.temperature.shape[0], n_times)
        self.assertEqual(predictions.tendencies.temperature.shape[0], n_times)
        self.assertEqual(len(predictions.times), n_times)

    @unittest.skip("SPEEDY physics has tree_math pytree registration issues in pytest - see test_icon_physics for full integration test")
    def test_run_with_stacked_state(self):
        """Test run with pre-stacked state array."""
        model = PrescribedStateModel(
            physics=SpeedyPhysics(),
            geometry=self.geometry,
        )

        # Stack states manually
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
            tracers={},
        )

        predictions = model.run(stacked_state, forcing=self.forcing)

        self.assertEqual(predictions.tendencies.temperature.shape[0], n_times)

    def test_create_single_column_state(self):
        """Test creating a single-column state."""
        nlev = 20
        temperature = jnp.linspace(300, 220, nlev)
        specific_humidity = jnp.linspace(0.02, 0.0001, nlev)

        state = PrescribedStateModel.create_single_column_state(
            temperature=temperature,
            specific_humidity=specific_humidity,
            nlev=nlev,
        )

        # Check shapes
        self.assertEqual(state.temperature.shape, (nlev, 1, 1))
        self.assertEqual(state.u_wind.shape, (nlev, 1, 1))
        self.assertEqual(state.geopotential.shape, (nlev, 1, 1))
        self.assertEqual(state.normalized_surface_pressure.shape, (1, 1))

        # Check values
        self.assertTrue(jnp.allclose(state.temperature[:, 0, 0], temperature))

    def test_date_calculation(self):
        """Test that dates are correctly computed for each timestep."""
        start_date = jdt.to_datetime('2020-06-15')
        dt_seconds = 3600.0  # 1 hour

        model = PrescribedStateModel(
            physics=SpeedyPhysics(),
            geometry=self.geometry,
            start_date=start_date,
            dt_seconds=dt_seconds,
        )

        date0 = model._date_from_time_index(0)
        date1 = model._date_from_time_index(1)

        # Check that dates advance correctly
        self.assertEqual(date0.model_step, 0)
        self.assertEqual(date1.model_step, 1)


class TestPrescribedStateModelWithICON(unittest.TestCase):
    """Test PrescribedStateModel with ICON physics."""

    def setUp(self):
        """Set up test fixtures for ICON tests."""
        self.nlev = 40
        self.nlon = 96
        self.nlat = 48
        self.shape = (self.nlev, self.nlon, self.nlat)

        self.geometry = Geometry.from_grid_shape(
            nodal_shape=(self.nlon, self.nlat),
            num_levels=self.nlev
        )

    def _create_icon_test_state(self) -> PhysicsState:
        """Create a test state suitable for ICON physics."""
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
        from jcm.constants import grav
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

    def test_icon_physics(self):
        """Test with ICON physics module."""
        from jcm.physics.icon import IconPhysics
        from jcm.physics.icon.parameters import Parameters

        # Create ICON physics with matching timestep
        icon_params = Parameters.default().with_convection(dt_conv=1800.0)
        physics = IconPhysics(parameters=icon_params)

        model = PrescribedStateModel(
            physics=physics,
            geometry=self.geometry,
            use_hybrid_coords=False,  # Use sigma for stability
        )

        state = self._create_icon_test_state()
        tendencies, physics_data = model.run_single(state)

        # Check tendencies are computed
        self.assertEqual(tendencies.temperature.shape, self.shape)

        # Check for NaNs
        self.assertFalse(
            jnp.any(jnp.isnan(tendencies.temperature)),
            "Temperature tendencies contain NaN"
        )


if __name__ == '__main__':
    unittest.main()
