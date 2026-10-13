"""Tests for the #713 per-species aerosol mass-budget gauge."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.budget_gauge import CARRY_KEY, gauge_aerosol_budget
from jcm.physics_interface import PhysicsState


def _make(nlev=4, ncols=3, q=1e-9):
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 270.0),
        tracers={
            "m_du_acc": jnp.full((nlev, ncols), q),
            "m_du_cor": jnp.full((nlev, ncols), 2 * q),
            "n_acc": jnp.full((nlev, ncols), 1e8),   # number: not gauged
            "qc": jnp.zeros((nlev, ncols)),          # non-aerosol: not gauged
        },
    )
    diagnostics = {
        "air_density": jnp.ones((nlev, ncols)),
        "layer_thickness": jnp.full((nlev, ncols), 100.0),
    }
    return state, diagnostics


class GaugeTest(unittest.TestCase):
    def test_mass_and_ptend_are_species_sums(self):
        state, diag = _make()
        tends = {"m_du_acc": jnp.full((4, 3), 1e-13),
                 "m_du_cor": jnp.full((4, 3), -2e-13)}
        out = gauge_aerosol_budget(diag, state, tends, 600.0)
        # dm = 100 kg/m² per level, 4 levels: mass = (1+2)e-9 * 400.
        np.testing.assert_allclose(out["budget_mass_du"], 3e-9 * 400.0)
        np.testing.assert_allclose(out["budget_ptend_du"],
                                   (1e-13 - 2e-13) * 400.0)
        # First call: no expectation yet -> dynamics residual is zero.
        np.testing.assert_array_equal(out["budget_dyn_du"], 0.0)
        self.assertIn(CARRY_KEY, out)

    def test_dynamics_residual_closes_and_detects_a_leak(self):
        dt = 600.0
        state, diag = _make()
        tends = {"m_du_acc": jnp.full((4, 3), 1e-13),
                 "m_du_cor": jnp.zeros((4, 3))}
        out = gauge_aerosol_budget(diag, state, tends, dt)

        # A CONSERVATIVE host applies exactly q += dt*tend: residual 0.
        conserved = state.copy(tracers={
            **state.tracers,
            "m_du_acc": state.tracers["m_du_acc"] + dt * tends["m_du_acc"],
        })
        out2 = gauge_aerosol_budget(dict(out), conserved, tends, dt)
        np.testing.assert_allclose(np.asarray(out2["budget_dyn_du"]), 0.0,
                                   atol=1e-22)

        # A LEAKY host (transport created 10% extra acc mass): residual
        # equals exactly the created column mass per unit time.
        leaky = state.copy(tracers={
            **state.tracers,
            "m_du_acc": 1.1 * (state.tracers["m_du_acc"]
                               + dt * tends["m_du_acc"]),
        })
        out3 = gauge_aerosol_budget(dict(out), leaky, tends, dt)
        created = 0.1 * float(
            (state.tracers["m_du_acc"][0, 0] + dt * 1e-13) * 400.0)
        np.testing.assert_allclose(np.asarray(out3["budget_dyn_du"]),
                                   created / dt, rtol=1e-5)

    def test_zero_filled_template_expectation_reads_as_invalid(self):
        # get_empty_data traces the gauge and ZERO-FILLS the carry, and
        # the initial physics carry hands that template back on step 1 —
        # a warm start (nonzero burden) must NOT report its entire
        # initial mass as a fictitious dynamics source (Codex P2 on
        # #720). The zero-filled _valid flag marks the seed structural.
        import jax

        dt = 600.0
        state, diag = _make(q=5e-9)     # warm start: real burden
        tends = {"m_du_acc": jnp.zeros((4, 3)), "m_du_cor": jnp.zeros((4, 3))}
        real = gauge_aerosol_budget(dict(diag), state, tends, dt)
        # Simulate the template: zero-fill the carried expectation.
        template = jax.tree_util.tree_map(
            jnp.zeros_like, real[CARRY_KEY])
        seeded = {**diag, CARRY_KEY: template}
        out = gauge_aerosol_budget(seeded, state, tends, dt)
        np.testing.assert_array_equal(np.asarray(out["budget_dyn_du"]), 0.0)
        # The step it emits is valid, so the NEXT step measures normally.
        out2 = gauge_aerosol_budget(dict(out), state, tends, dt)
        np.testing.assert_allclose(np.asarray(out2["budget_dyn_du"]), 0.0,
                                   atol=1e-22)

    def test_noop_without_airmass_or_aerosols(self):
        state, diag = _make()
        # No air-mass diagnostics -> untouched dict.
        out = gauge_aerosol_budget({}, state, {}, 600.0)
        self.assertEqual(out, {})
        # No aerosol tracers -> untouched dict.
        bare = PhysicsState.zeros((4, 3)).copy(
            temperature=jnp.full((4, 3), 270.0),
            tracers={"qc": jnp.zeros((4, 3))},
        )
        out = gauge_aerosol_budget(dict(diag), bare, {}, 600.0)
        self.assertNotIn("budget_mass_du", out)


if __name__ == "__main__":
    unittest.main()
