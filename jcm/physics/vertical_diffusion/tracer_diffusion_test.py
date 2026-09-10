"""Tests for the implicit tracer vertical diffusion (#602 item 2)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.vertical_diffusion.tracer_diffusion import (
    TracerDiffusionParameters,
    TracerVerticalDiffusion,
    diffuse_tracers_implicit,
)
from jcm.physics_interface import PhysicsState


class DiffuseImplicitTest(unittest.TestCase):
    def _grid(self, nlev=12, ncols=2):
        rho = jnp.linspace(0.3, 1.2, nlev)[:, None] * jnp.ones((1, ncols))
        dz = jnp.linspace(900.0, 150.0, nlev)[:, None] * jnp.ones((1, ncols))
        kh = jnp.linspace(1.0, 40.0, nlev)[:, None] * jnp.ones((1, ncols))
        return rho, dz, kh

    def test_conserves_column_mass_exactly(self):
        rho, dz, kh = self._grid()
        q = jnp.stack([
            jnp.linspace(0.0, 1.0e-8, 12)[:, None] * jnp.ones((1, 2)),
            jnp.zeros((12, 2)).at[-1].set(5.0e-9),
        ])
        q_new = diffuse_tracers_implicit(q, kh, rho, dz, 1800.0)
        dm = rho * dz
        before = jnp.sum(q * dm[jnp.newaxis], axis=1)
        after = jnp.sum(q_new * dm[jnp.newaxis], axis=1)
        np.testing.assert_allclose(
            np.asarray(after), np.asarray(before), rtol=1e-6,
        )

    def test_mixes_surface_load_upward(self):
        rho, dz, kh = self._grid()
        q = jnp.zeros((1, 12, 2)).at[0, -1].set(1.0e-8)
        q_new = diffuse_tracers_implicit(q, kh, rho, dz, 1800.0)
        # The layer above the loaded one gains; the loaded one loses.
        self.assertGreater(float(q_new[0, -2, 0]), 0.0)
        self.assertLess(float(q_new[0, -1, 0]), 1.0e-8)

    def test_unconditionally_stable_and_bounded(self):
        # Backward Euler with an M-matrix obeys the maximum principle for
        # ANY K·dt: no overshoot, no negatives, even at K·dt/dz² >> 1.
        rho, dz, kh = self._grid()
        q = jnp.zeros((1, 12, 2)).at[0, 5].set(1.0e-6)
        q_new = diffuse_tracers_implicit(
            q, 1.0e5 * kh, rho, dz, 36000.0,
        )
        self.assertGreaterEqual(float(q_new.min()), 0.0)
        self.assertLessEqual(float(q_new.max()), 1.0e-6 * (1.0 + 1e-6))
        self.assertTrue(bool(jnp.all(jnp.isfinite(q_new))))

    def test_two_layer_exchange_matches_closed_form(self):
        # Analytic pin (a magnitude error in the conductance — dropped
        # rho_int, dz_int or dm — is invisible to conservation and
        # direction checks). Two equal layers exchanging through one
        # interface, backward Euler:
        #   d' = d / (1 + 2·dt·g/dm),  g = rho_int·K_int/dz_int.
        rho = jnp.full((2, 1), 0.8)
        dz = jnp.full((2, 1), 250.0)
        kh = jnp.array([[10.0], [30.0]])
        q = jnp.array([[[4.0e-9], [1.0e-9]]])
        dt = 900.0
        g = 0.8 * 20.0 / 250.0
        dm = 0.8 * 250.0
        expected_diff = (4.0e-9 - 1.0e-9) / (1.0 + 2.0 * dt * g / dm)
        q_new = diffuse_tracers_implicit(q, kh, rho, dz, dt)
        got_diff = float(q_new[0, 0, 0] - q_new[0, 1, 0])
        np.testing.assert_allclose(got_diff, expected_diff, rtol=1e-6)
        np.testing.assert_allclose(
            float(jnp.sum(q_new)), 5.0e-9, rtol=1e-6,
        )

    def test_zero_k_is_identity(self):
        rho, dz, _ = self._grid()
        q = jnp.ones((1, 12, 2)) * 3.0e-9
        q_new = diffuse_tracers_implicit(
            q, jnp.zeros((12, 2)), rho, dz, 1800.0,
        )
        np.testing.assert_allclose(np.asarray(q_new), np.asarray(q))


class _VDiff:
    def __init__(self, kh):
        self.kh = kh


class TracerVerticalDiffusionTermTest(unittest.TestCase):
    def _setup(self, nlev=8, ncols=2, with_vdiff=True):
        shape = (nlev, ncols)
        tracers = {
            "m_so4_acc": jnp.zeros(shape).at[-1].set(1.0e-9),
            "n_acc": jnp.zeros(shape).at[-1].set(1.0e7),
        }
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 280.0), tracers=tracers,
        )
        diagnostics = {
            "air_density": jnp.full(shape, 1.0),
            "layer_thickness": jnp.full(shape, 300.0),
            "_dt_seconds": 1800.0,
        }
        if with_vdiff:
            diagnostics["vertical_diffusion"] = _VDiff(
                jnp.full(shape, 30.0)
            )
        return state, diagnostics

    def test_mixes_and_conserves(self):
        state, diagnostics = self._setup()
        term = TracerVerticalDiffusion(("m_so4_acc", "n_acc"))
        tend, _ = term(state, diagnostics, None, None)
        dm = 300.0
        for nm in ("m_so4_acc", "n_acc"):
            dq = np.asarray(tend.tracers[nm])
            self.assertLess(dq[-1, 0], 0.0)          # surface layer loses
            self.assertGreater(dq[-2, 0], 0.0)       # layer above gains
            total = float(np.sum(dq) * dm)
            scale = float(np.sum(np.abs(dq)) * dm)
            self.assertLessEqual(abs(total), 1e-6 * max(scale, 1e-30), nm)

    def test_noop_without_vdiff_diagnostic(self):
        state, diagnostics = self._setup(with_vdiff=False)
        term = TracerVerticalDiffusion(("m_so4_acc",))
        tend, _ = term(state, diagnostics, None, None)
        np.testing.assert_array_equal(
            np.asarray(tend.tracers["m_so4_acc"]), 0.0,
        )

    def test_empty_probe_state_is_safe(self):
        state, diagnostics = self._setup()
        state = state.copy(tracers={})
        term = TracerVerticalDiffusion(("m_so4_acc",))
        tend, _ = term(state, diagnostics, None, None)
        np.testing.assert_array_equal(
            np.asarray(tend.tracers["m_so4_acc"]), 0.0,
        )

    def test_grad_through_diffusion_scale(self):
        state, diagnostics = self._setup()

        def loss(scale):
            term = TracerVerticalDiffusion(
                ("m_so4_acc",),
                params=TracerDiffusionParameters(diffusion_scale=scale),
            )
            tend, _ = term(state, diagnostics, None, None)
            return jnp.sum(tend.tracers["m_so4_acc"] ** 2)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)

    def test_empty_tracer_list_rejected(self):
        with self.assertRaises(ValueError):
            TracerVerticalDiffusion(())


if __name__ == "__main__":
    unittest.main()
