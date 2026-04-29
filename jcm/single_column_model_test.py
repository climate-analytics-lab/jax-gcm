"""Tests for ``jcm.single_column_model.SingleColumnModel``."""

import unittest

import jax.numpy as jnp
import pytest

from jcm.constants import grav
from jcm.physics_interface import PhysicsState
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.physics.icon.icon_terms import icon_physics
from jcm.single_column_model import SCMPredictions, SingleColumnModel
from jcm.terrain import TerrainData
from jcm.utils import (
    create_initial_tracers,
    create_single_column_state,
    get_coords,
)


def _make_test_state(coords) -> PhysicsState:
    """Build a vertically stratified atmospheric state on the test grid."""
    nlev = coords.nodal_shape[0]
    nlon, nlat = coords.horizontal.nodal_shape
    shape = (nlev, nlon, nlat)

    z = jnp.linspace(0, 30000, nlev)[::-1]
    t_profile = jnp.maximum(288.0 - 6.5e-3 * z, 200.0)
    q_profile = 0.012 * jnp.exp(-z / 3000.0)

    return PhysicsState(
        u_wind=jnp.full(shape, 5.0),
        v_wind=jnp.zeros(shape),
        temperature=jnp.broadcast_to(t_profile[:, None, None], shape),
        specific_humidity=jnp.broadcast_to(q_profile[:, None, None], shape),
        geopotential=jnp.broadcast_to((grav * z)[:, None, None], shape),
        normalized_surface_pressure=jnp.ones((nlon, nlat)),
        tracers={'qc': jnp.zeros(shape), 'qi': jnp.zeros(shape)},
    )


@pytest.mark.slow
class TestSingleColumnModel(unittest.TestCase):
    """Held-Suarez SCM tests — cheap and fully aquaplanet."""

    def setUp(self):
        self.coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        self.terrain = TerrainData.aquaplanet(self.coords)
        self.state = _make_test_state(self.coords)
        self.physics = held_suarez_physics()

    def test_initialization_caches_coords(self):
        model = SingleColumnModel(physics=self.physics, coords=self.coords)
        self.assertIsNotNone(model.coords)
        self.assertIsNotNone(model.terrain)
        self.assertEqual(model.dt_seconds, 1800.0)
        self.assertTrue(model.apply_tracer_tendencies)

    def test_run_held_suarez_smoke(self):
        model = SingleColumnModel(physics=self.physics, coords=self.coords)
        states = [self.state, self.state, self.state]
        predictions = model.run(states)
        self.assertIsInstance(predictions, SCMPredictions)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 3)
        # qc/qi were carried in via the prescribed state; Held-Suarez writes
        # no tracer tendency, so they should remain at their initial zeros.
        self.assertIn('qc', predictions.tracer_states)
        self.assertEqual(predictions.tracer_states['qc'].shape[0], 3)

    def test_disable_tracer_update(self):
        scm = SingleColumnModel(
            physics=self.physics,
            coords=self.coords,
            apply_tracer_tendencies=False,
        )
        states = [self.state, self.state]
        predictions = scm.run(states)
        # Held-Suarez has no tracer tendencies, so this just checks the path runs.
        self.assertEqual(predictions.tendencies.temperature.shape[0], 2)


@pytest.mark.slow
class TestSingleColumnModelICON(unittest.TestCase):
    """ICON-physics SCM test on a small grid — exercises tracer evolution."""

    def setUp(self):
        # ICON only ships hybrid level definitions for 40/47; use sigma here
        # to keep the test cheap.
        self.coords = get_coords(
            SigmaCoordinates.equidistant(8), spectral_truncation=21,
        )
        self.terrain = TerrainData.aquaplanet(self.coords)
        self.state = _make_test_state(self.coords)

    def test_icon_run_smoke(self):
        physics = icon_physics(radiation_scheme='grey')
        scm = SingleColumnModel(
            physics=physics, coords=self.coords, terrain=self.terrain,
        )
        # Two timesteps is enough for tracer-evolution wiring.
        states = [self.state, self.state]
        predictions = scm.run(states)
        self.assertIsInstance(predictions, SCMPredictions)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 2)
        # qc/qi are required tracers; they should evolve through scan.
        self.assertIn('qc', predictions.tracer_states)
        self.assertIn('qi', predictions.tracer_states)


class TestSCMHelpers(unittest.TestCase):
    """The SCM-oriented helpers in ``jcm.utils``."""

    def test_create_single_column_state(self):
        nlev = 8
        T = jnp.linspace(280, 220, nlev)
        q = jnp.full((nlev,), 0.005)
        state = create_single_column_state(T, q)
        self.assertEqual(state.temperature.shape, (nlev, 1, 1))
        self.assertEqual(state.normalized_surface_pressure.shape, (1, 1))

    def test_create_initial_tracers(self):
        tracers = create_initial_tracers((4, 1, 1), cloud_water=1e-4)
        self.assertEqual(set(tracers), {'qc', 'qi'})
        self.assertAlmostEqual(float(tracers['qc'][0, 0, 0]), 1e-4)
        self.assertAlmostEqual(float(tracers['qi'][0, 0, 0]), 0.0)
