"""Tests for ``jcm.prescribed_state_model.PrescribedStateModel``."""

import unittest

import jax.numpy as jnp
import pytest

from jcm.constants import grav
from jcm.physics_interface import PhysicsState
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from jcm.prescribed_state_model import (
    PrescribedStateModel,
    PrescribedStatePredictions,
)


def _make_test_state(coords) -> PhysicsState:
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
        tracers={},
    )


class TestPrescribedStateModel(unittest.TestCase):
    def setUp(self):
        self.coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        self.state = _make_test_state(self.coords)

    def test_run_smoke(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        states = [self.state] * 3
        predictions = model.run(states)
        self.assertIsInstance(predictions, PrescribedStatePredictions)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 3)
        self.assertEqual(predictions.times.shape[0], 3)


# Slow-marked companion — see jcm/runners_test.py for rationale.

@pytest.mark.slow
class TestPrescribedStateModelSlow(TestPrescribedStateModel):
    pass
