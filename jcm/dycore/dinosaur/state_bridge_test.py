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


@pytest.mark.slow
class TestHybridSurfacePressureRoundTrip(unittest.TestCase):
    """Hybrid-coordinate surface pressure must survive a PhysicsState round-trip.

    Regression for the asymmetry where ``dynamics_state_to_physics_state``
    divides hybrid ``sp`` by ``p0`` (exposing ``P_s/p0``) but the inverse logged
    that normalized value directly. For hybrid coords dinosaur stores
    ``log(P_s)`` (nondim Pa), so the inverse must multiply by ``p0`` first;
    without it surface pressure collapses by a factor of ~p0.
    """

    def test_hybrid_round_trip_preserves_surface_pressure(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.utils import get_coords

        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        model = Model(
            coords=coords,
            physics=echam_physics(radiation_scheme="grey", checkpoint_terms=False),
            time_step=180.0,
        )
        primitive = model.dycore.primitive
        tracer_specs = {spec.name: spec for spec in model.physics.required_tracers()}

        state = model.dycore.to_physics_state(model._prepare_initial_dycore_state())
        # A clearly non-trivial normalized surface pressure (P_s/p0 ≈ 0.97).
        seeded = state.copy(
            normalized_surface_pressure=state.normalized_surface_pressure * 0.97,
        )

        modal = physics_state_to_dynamics_state(seeded, primitive, tracer_specs=tracer_specs)
        recovered = dynamics_state_to_physics_state(modal, primitive, tracer_specs=tracer_specs)

        # The load-bearing assertion: surface pressure survives the round-trip.
        # Pre-fix, ``recovered`` is smaller than ``seeded`` by ~p0 (~1e5).
        self.assertTrue(jnp.allclose(
            recovered.normalized_surface_pressure,
            seeded.normalized_surface_pressure,
            rtol=1e-4,
        ))


if __name__ == "__main__":
    unittest.main()
