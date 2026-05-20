"""Protocol-canary tests on a non-lat/lon horizontal layout.

This is the Phase-1 design spike: a deliberately minimal cubed-sphere fake
exercises the :class:`DynamicalCore` protocol end-to-end (state init →
gridpoint projection → forward-Euler add of a physics tendency → next state),
on arrays shaped ``(nlev, nelem, gll, gll)``. If anything in the rest of
jax-gcm silently assumed lat/lon layout, the round-trip here would shape-mismatch.

The fake :class:`FakeCubedSphereDycore` doesn't run real dynamics — it's
identity-with-physics-Euler-add — so the assertions are about plumbing, not
about science. The Phase-2 :class:`PyscesDycore` will reuse the same test
harness once it lands.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from jcm.dycore._fake_cubed_sphere import FakeCubedSphereDycore
from jcm.physics_interface import (
    PhysicsState, PhysicsTendency, compute_physics_step_gridpoint, verify_state,
)
from jcm.physics.physics_term import TracerSpec


class TestFakeCubedSphereDycoreProtocol(unittest.TestCase):
    """The protocol must work on a horizontal layout that is not (nlon, nlat)."""

    def setUp(self):
        self.nelem, self.gll, self.nlev = 4, 3, 5
        self.dycore = FakeCubedSphereDycore(
            nelem=self.nelem, gll=self.gll, nlev=self.nlev,
        )
        self.shape_3d = (self.nlev, self.nelem, self.gll, self.gll)
        self.shape_2d = (self.nelem, self.gll, self.gll)

    def test_initial_state_shapes_match_cubed_sphere_layout(self):
        state = self.dycore.initial_state(None)
        self.assertEqual(state.temperature.shape, self.shape_3d)
        self.assertEqual(state.u_wind.shape, self.shape_3d)
        self.assertEqual(state.normalized_surface_pressure.shape, self.shape_2d)

    def test_to_physics_state_preserves_layout(self):
        state = self.dycore.initial_state(None)
        ps = self.dycore.to_physics_state(state)
        # The protocol contract: PhysicsState arrays are
        # (nlev, *horizontal_shape) on the dycore's native layout.
        self.assertEqual(ps.temperature.shape, self.shape_3d)
        self.assertEqual(ps.geopotential.shape, self.shape_3d)
        self.assertEqual(ps.normalized_surface_pressure.shape, self.shape_2d)

    def test_round_trip_identity(self):
        """Building an initial state from a custom PhysicsState round-trips."""
        target = PhysicsState(
            u_wind=jnp.full(self.shape_3d, 1.5),
            v_wind=jnp.full(self.shape_3d, -2.5),
            temperature=jnp.full(self.shape_3d, 271.0),
            specific_humidity=jnp.full(self.shape_3d, 1e-3),
            geopotential=jnp.zeros(self.shape_3d),
            normalized_surface_pressure=jnp.full(self.shape_2d, 0.98),
            tracers={},
        )
        state = self.dycore.initial_state(target)
        recovered = self.dycore.to_physics_state(state)
        self.assertTrue(jnp.allclose(recovered.u_wind, target.u_wind))
        self.assertTrue(jnp.allclose(recovered.temperature, target.temperature))
        self.assertTrue(jnp.allclose(
            recovered.normalized_surface_pressure,
            target.normalized_surface_pressure,
        ))

    def test_step_forward_euler_adds_tendency(self):
        state = self.dycore.initial_state(None)
        dt = self.dycore.dt_seconds
        tend = PhysicsTendency(
            u_wind=jnp.full(self.shape_3d, 1.0 / dt),  # 1 m/s per step
            v_wind=jnp.zeros(self.shape_3d),
            temperature=jnp.full(self.shape_3d, 0.5 / dt),  # 0.5 K per step
            specific_humidity=jnp.zeros(self.shape_3d),
            tracers={},
        )
        next_state = self.dycore.step(state, tend)
        self.assertTrue(jnp.allclose(next_state.u_wind, state.u_wind + 1.0))
        self.assertTrue(jnp.allclose(next_state.temperature, state.temperature + 0.5))

    def test_tracer_specs_seed_state(self):
        """Declared tracers appear in the initial state with their initial_value."""
        specs = {
            "co2_vmr": TracerSpec("co2_vmr", initial_value=4.2e-4, nondimensionalize=False),
            "qc": TracerSpec("qc", initial_value=0.0),
        }
        state = self.dycore.initial_state(None, tracer_specs=specs)
        self.assertIn("co2_vmr", state.tracers)
        self.assertIn("qc", state.tracers)
        self.assertTrue(jnp.allclose(state.tracers["co2_vmr"], 4.2e-4))
        self.assertEqual(state.tracers["co2_vmr"].shape, self.shape_3d)

    def test_compute_physics_step_gridpoint_works_on_cubed_sphere(self):
        """The dycore-agnostic physics-step helper accepts our cubed-sphere PhysicsState."""

        class _NullPhysics:
            UNITS_TABLE_CSV_PATH = None
            cached_coords = None

            def required_tracers(self):
                return ()

            def compute_tendencies(self, state, forcing, terrain,
                                   prev_physics_data=None):
                tend = PhysicsTendency(
                    u_wind=jnp.zeros_like(state.u_wind),
                    v_wind=jnp.zeros_like(state.v_wind),
                    temperature=jnp.zeros_like(state.temperature),
                    specific_humidity=jnp.zeros_like(state.specific_humidity),
                    tracers={},
                )
                return tend, prev_physics_data

        state = self.dycore.initial_state(None)
        ps = self.dycore.to_physics_state(state)
        tend, _ = compute_physics_step_gridpoint(
            ps, forcing=None, terrain=None, physics_state_carry={},
            physics=_NullPhysics(), time_step=self.dycore.dt_seconds,
        )
        # Tendency shapes match the (nlev, *horizontal_shape) layout — no
        # silent broadcasting onto a hidden lat/lon shape.
        self.assertEqual(tend.temperature.shape, self.shape_3d)

    def test_state_is_a_jax_pytree(self):
        """The state must scan/tree_map cleanly (required for lax.scan)."""
        state = self.dycore.initial_state(None)
        scaled = jax.tree_util.tree_map(lambda x: x * 0.0, state)
        self.assertTrue(jnp.allclose(scaled.u_wind, 0.0))
        self.assertTrue(jnp.allclose(scaled.temperature, 0.0))


class TestVerifyStatePreservesCubedSphereShape(unittest.TestCase):
    """``verify_state`` is dycore-agnostic; it must not reshape on output."""

    def test_verify_state_round_trips_layout(self):
        dycore = FakeCubedSphereDycore(nelem=3, gll=2, nlev=4)
        state = dycore.initial_state(None)
        ps = dycore.to_physics_state(state)
        clamped = verify_state(ps)
        self.assertEqual(clamped.temperature.shape, ps.temperature.shape)
        self.assertEqual(
            clamped.normalized_surface_pressure.shape,
            ps.normalized_surface_pressure.shape,
        )


if __name__ == "__main__":
    unittest.main()
