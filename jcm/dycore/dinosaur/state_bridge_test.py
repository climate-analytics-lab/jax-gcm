"""Round-trip tests for :mod:`jcm.dycore.dinosaur.state_bridge`.

The fast-test :mod:`jcm.physics_interface_test` already exercises both
directions of the conversion at T31L8 sigma. This file marks an end-to-end
round-trip as ``@pytest.mark.slow`` so the PR-time slow-coverage run also
walks ``physics_state_to_dynamics_state`` — without it the inverse direction
shows up as uncovered against the .coveragerc-pr threshold even though every
fast test that takes a ``PhysicsState`` initial state goes through it.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import pytest

from jcm.dycore.dinosaur.state_bridge import (
    dynamics_state_to_physics_state,
    physics_state_to_dynamics_state,
)
from jcm.model import Model
from jcm.physics.physics_term import TracerSpec
from jcm.physics.speedy.speedy_coords import get_speedy_coords


@pytest.mark.slow
class TestStateBridgeRoundTripSlow(unittest.TestCase):
    """End-to-end gridpoint↔modal round-trip on the default SPEEDY coords."""

    def test_round_trip_preserves_scalars_and_tracer_specs(self):
        # A real Model gives us a fully-wired ``primitive`` operator we can
        # hand to the standalone conversion helpers; the same code paths run
        # inside ``Model.run`` every step. Tolerance is set against spectral
        # round-trip noise (T31 nodes ≠ modal truncation; the inverse loses
        # the high-wavenumber tail). Winds are NOT round-trip-stable on the
        # sphere — a uniform u≠0 has zero vorticity and non-zero divergence,
        # but the spectral path goes through vor/div decomposition and loses
        # the constant component, so we don't assert on them here.
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        model = Model(coords=coords, time_step=720)
        primitive = model.dycore.primitive

        nodal_shape = coords.horizontal.nodal_shape
        kx = coords.vertical.layers
        state = model.dycore.to_physics_state(model._prepare_initial_dycore_state())
        seeded = state.copy(
            temperature=state.temperature + 1.0,
            tracers={
                "co2_vmr": jnp.full((kx,) + nodal_shape, 4.2e-4),
            },
        )
        tracer_specs = {"co2_vmr": TracerSpec(
            "co2_vmr", initial_value=4.2e-4, nondimensionalize=False,
        )}
        modal = physics_state_to_dynamics_state(seeded, primitive, tracer_specs=tracer_specs)
        recovered = dynamics_state_to_physics_state(modal, primitive, tracer_specs=tracer_specs)

        self.assertTrue(jnp.allclose(recovered.temperature, seeded.temperature, rtol=1e-4))
        # ``nondimensionalize=False`` tracers must pass through the round-trip
        # without the gram/kg scaling — this is the load-bearing branch for
        # GHG / number-concentration tracers under the v2 dycore protocol.
        self.assertTrue(jnp.allclose(
            recovered.tracers["co2_vmr"], seeded.tracers["co2_vmr"], rtol=1e-4,
        ))


if __name__ == "__main__":
    unittest.main()
