"""Tests for ``jcm.single_column_model.SingleColumnModel``."""

import unittest

import jax.numpy as jnp
import pytest
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.constants import grav
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.icon.icon_terms import icon_physics
from jcm.physics_interface import PhysicsState
from jcm.single_column_model import SCMPredictions, SingleColumnModel
from jcm.utils import create_initial_tracers, create_single_column_state


def _make_column_state(nlev: int) -> PhysicsState:
    """Build a vertically stratified 1-D column state."""
    z = jnp.linspace(0, 30000, nlev)[::-1]
    t_profile = jnp.maximum(288.0 - 6.5e-3 * z, 200.0)
    q_profile = 0.012 * jnp.exp(-z / 3000.0)
    return PhysicsState(
        u_wind=jnp.full(nlev, 5.0),
        v_wind=jnp.zeros(nlev),
        temperature=t_profile,
        specific_humidity=q_profile,
        geopotential=grav * z,
        normalized_surface_pressure=jnp.asarray(1.0),
        tracers={'qc': jnp.zeros(nlev), 'qi': jnp.zeros(nlev)},
    )


class TestSCMConstruction(unittest.TestCase):
    """Cheap tests for the SCM's coord-stub bookkeeping."""

    def test_init_builds_one_one_grid_at_lat_lon(self):
        scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=30.0,
            lon_deg=180.0,
        )
        self.assertEqual(scm.coords.horizontal.nodal_shape, (1, 1))
        self.assertEqual(scm.coords.nodal_shape, (8, 1, 1))
        self.assertAlmostEqual(
            float(scm.coords.horizontal.latitudes[0]),
            float(jnp.deg2rad(30.0)),
        )
        self.assertAlmostEqual(
            float(scm.coords.horizontal.longitudes[0]),
            float(jnp.deg2rad(180.0)),
        )

    def test_init_defaults_to_single_column_terrain_and_forcing(self):
        scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
        )
        self.assertEqual(scm.terrain.orog.shape, (1, 1))
        self.assertEqual(scm.forcing.sea_surface_temperature.shape, (1, 1))


class TestSCMHeldSuarez(unittest.TestCase):
    """Held-Suarez SCM run on a small column."""

    def setUp(self):
        self.column_state = _make_column_state(nlev=8)
        self.scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0,
            lon_deg=0.0,
        )

    def test_run_smoke(self):
        states = [self.column_state, self.column_state, self.column_state]
        predictions = self.scm.run(states)
        self.assertIsInstance(predictions, SCMPredictions)
        # Tendencies should be 1-D in level with a leading time axis.
        self.assertEqual(predictions.tendencies.temperature.shape, (3, 8))
        self.assertIn('qc', predictions.tracer_states)
        self.assertEqual(predictions.tracer_states['qc'].shape, (3, 8))

    def test_disable_tracer_update(self):
        scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            apply_tracer_tendencies=False,
        )
        states = [self.column_state, self.column_state]
        predictions = scm.run(states)
        self.assertEqual(predictions.tendencies.temperature.shape, (2, 8))


class TestSCMICON(unittest.TestCase):
    """ICON-grey SCM run — exercises tracer evolution."""

    def test_icon_run_smoke(self):
        column_state = _make_column_state(nlev=8)
        scm = SingleColumnModel(
            physics=icon_physics(radiation_scheme='grey'),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0,
            lon_deg=0.0,
        )
        predictions = scm.run([column_state, column_state])
        self.assertEqual(predictions.tendencies.temperature.shape, (2, 8))
        self.assertIn('qc', predictions.tracer_states)
        self.assertIn('qi', predictions.tracer_states)


class TestSCMHelpers(unittest.TestCase):
    """The SCM-oriented helpers in ``jcm.utils``."""

    def test_create_single_column_state_is_one_dimensional(self):
        nlev = 8
        T = jnp.linspace(280, 220, nlev)
        q = jnp.full((nlev,), 0.005)
        state = create_single_column_state(T, q)
        self.assertEqual(state.temperature.shape, (nlev,))
        self.assertEqual(state.normalized_surface_pressure.shape, ())

    def test_create_initial_tracers(self):
        tracers = create_initial_tracers(4, cloud_water=1e-4)
        self.assertEqual(set(tracers), {'qc', 'qi'})
        self.assertEqual(tracers['qc'].shape, (4,))
        self.assertAlmostEqual(float(tracers['qc'][0]), 1e-4)


# Slow-marked companions — see jcm/runners_test.py for rationale.

@pytest.mark.slow
class TestSCMHeldSuarezSlow(TestSCMHeldSuarez):
    pass


@pytest.mark.slow
class TestSCMICONSlow(TestSCMICON):
    pass
