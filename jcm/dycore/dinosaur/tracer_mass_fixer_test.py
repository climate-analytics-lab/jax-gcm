"""Tests for the local (Bermejo–Conde) semi-Lagrangian tracer mass fixer."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.dycore.dinosaur.tracer_mass_fixer import bermejo_conde_correction


def _fields(seed=0, shape=(4, 6, 5)):
    # High-order values a clear margin above or below the low-order ones, so
    # the weights' kink at q_high == q_low is never within a finite-difference
    # step of a point.
    rng = np.random.default_rng(seed)
    q_low = rng.uniform(0.3, 1.0, shape)
    side = rng.choice([-1.0, 1.0], shape)
    q_high = q_low + side * rng.uniform(0.05, 0.25, shape)
    mass = rng.uniform(0.5, 1.5, shape)
    return jnp.asarray(q_high), jnp.asarray(q_low), jnp.asarray(mass)


class BermejoCondeCorrectionTest(unittest.TestCase):
    """Diamantakis & Flemming (2014) Eqs. (5)-(6), β = 1."""

    def test_restores_the_target_mass(self):
        q_high, q_low, m = _fields()
        for frac in (0.98, 1.02):                 # deficit and surplus
            target = frac * float(jnp.sum(q_high * m))
            q1 = bermejo_conde_correction(q_high, q_low, target, m)
            np.testing.assert_allclose(float(jnp.sum(q1 * m)), target,
                                       rtol=1e-6)

    def test_surplus_comes_off_where_the_cubic_sits_above_the_linear(self):
        q_high, q_low, m = _fields(1)
        target = 0.99 * float(jnp.sum(q_high * m))
        q1 = np.asarray(bermejo_conde_correction(q_high, q_low, target, m))
        q_high, q_low = np.asarray(q_high), np.asarray(q_low)
        above = q_high > q_low
        # Lowered only where the high-order value exceeds the low-order one,
        # untouched elsewhere, and never below the low-order value.
        self.assertTrue((q1[above] <= q_high[above]).all())
        np.testing.assert_array_equal(q1[~above], q_high[~above])
        self.assertTrue((q1 >= np.minimum(q_high, q_low) - 1e-6).all())
        self.assertTrue((q1 >= 0.0).all())

    def test_deficit_is_restored_where_the_cubic_sits_below_the_linear(self):
        q_high, q_low, m = _fields(2)
        target = 1.01 * float(jnp.sum(q_high * m))
        q1 = np.asarray(bermejo_conde_correction(q_high, q_low, target, m))
        q_high, q_low = np.asarray(q_high), np.asarray(q_low)
        below = q_high < q_low
        self.assertTrue((q1[below] >= q_high[below]).all())
        np.testing.assert_array_equal(q1[~below], q_high[~below])
        self.assertTrue((q1 <= np.maximum(q_high, q_low) + 1e-6).all())

    def test_an_error_beyond_the_weights_capacity_is_left_over(self):
        # lambda is held at 1: every weighted point goes to the linear value
        # and the rest of the error stays (for the global fixer).
        q_high, q_low, m = _fields(3)
        target = 0.5 * float(jnp.sum(q_high * m))
        q1 = np.asarray(bermejo_conde_correction(q_high, q_low, target, m))
        expected = np.minimum(np.asarray(q_high), np.asarray(q_low))
        np.testing.assert_allclose(q1, expected, rtol=1e-6)

    def test_no_error_and_no_weights_are_no_ops(self):
        q_high, q_low, m = _fields(4)
        exact = float(jnp.sum(q_high * m))
        np.testing.assert_allclose(
            bermejo_conde_correction(q_high, q_low, exact, m), q_high,
            rtol=1e-6)
        # Identical high- and low-order fields: nothing to weight by.
        np.testing.assert_array_equal(
            bermejo_conde_correction(q_high, q_high, 0.9 * exact, m), q_high)

    def test_beta_other_than_one_concentrates_the_weight(self):
        # beta != 1 exercises the ``w ** beta`` branch bermejo_conde_correction
        # takes only off the beta == 1.0 fast path; the IFS default stays 1
        # (BC_BETA), so this checks the general form against a direct
        # evaluation of Eqs. (5)-(6) rather than against a reference number.
        # A smaller excess than the beta=1 tests: squaring the already-small
        # weights shrinks the correction capacity, so a 3% excess (as used
        # elsewhere in this file) would exceed it and clip lambda at 1 --
        # exercising test_an_error_beyond_the_weights_capacity_is_left_over's
        # case instead of this one's. 0.3% stays within capacity here.
        q_high, q_low, m = _fields(6)
        target = 0.997 * float(jnp.sum(q_high * m))
        q1 = bermejo_conde_correction(q_high, q_low, target, m, beta=2.0)
        np.testing.assert_allclose(float(jnp.sum(q1 * m)), target, rtol=1e-6)
        excess = float(jnp.sum(q_high * m) - target)
        sign = 1.0 if excess >= 0.0 else -1.0
        w = jnp.maximum(sign * (q_high - q_low), 0.0) ** 2.0
        lam = min(max(excess * sign / float(jnp.sum(w * m)), 0.0), 1.0)
        self.assertLess(lam, 1.0)          # within capacity, as intended
        np.testing.assert_allclose(q1, q_high - sign * lam * w, rtol=1e-5)

    def test_gradients(self):
        from jcm.testing import check_gradients

        q_high, q_low, m = _fields(5)
        target = jnp.asarray(0.99 * float(jnp.sum(q_high * m)))
        check_gradients(
            lambda qh, ql, t: bermejo_conde_correction(qh, ql, t, m),
            (q_high, q_low, target), rtol=1e-3)
        # A uniform field has zero weights everywhere: the guarded quotient
        # must still give finite derivatives.
        flat = jnp.ones((3, 4, 5))
        g = jax.grad(lambda t: jnp.sum(
            bermejo_conde_correction(flat, flat, t, flat)))(jnp.asarray(50.0))
        self.assertTrue(np.isfinite(float(g)))


def _sl_dycore(local, **sl_options):
    from jcm.dycore.dinosaur.dycore import DinosaurDycore
    from jcm.physics.physics_term import TracerSpec
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import TerrainData

    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    return DinosaurDycore(
        coords=coords, terrain=TerrainData.aquaplanet(coords), dt_seconds=2400.0,
        advection="semi_lagrangian",
        tracer_specs={"dust": TracerSpec(name="dust")},
        sl_options={"local_mass_fixer": local, **sl_options},
    )


class SemiLagrangianTransportTest(unittest.TestCase):
    """The fixer inside the dinosaur SL step, on a sharp, patchy tracer.

    ``_dynamics_step_fn`` is the whole two-stage Crank-Nicolson RK2 step, not
    just the transport the fixer corrects: the corrector's semi-Lagrangian
    result is then combined with an explicit term and run through the
    implicit gravity-wave solve, which also updates ``log_surface_pressure``
    (``_fix_nodal_tracer_mass``'s own docstring calls this "the pressure
    change between the transport and the end of the step"). So a global-mass
    check against the *final* state is not exactly round-off even with the
    local fixer on -- it is round-off plus that pressure-change residual,
    which is what the global proportional fixer in ``step()`` (not exercised
    here) closes. The tolerances below bound that residual, not zero.
    """

    def _patchy_state(self, dycore):
        # A patch of a few cells on one level in otherwise clean air: the
        # regime where the quasi-monotone limiter's clipping of the cubic
        # undershoots creates mass at the edges (#1062).
        state = dycore.initial_state(None, random_seed=0)
        blob = np.zeros(state.tracers["dust"].shape)
        blob[3, 10:13, 20:23] = 1.0
        q0 = jnp.asarray(1e-8 * blob, dtype=state.temperature_variation.dtype)
        return state.replace(tracers={**state.tracers, "dust": q0})

    def _step(self, local):
        dycore = _sl_dycore(local=local)
        state = self._patchy_state(dycore)
        # Transport alone, before the global proportional fixer.
        moved = dycore._dynamics_step_fn(state)
        return dycore, state, moved

    def _mass(self, dycore, state):
        w = np.asarray(dycore.coords.horizontal.quadrature_weights)
        dp = np.asarray(dycore._nodal_tracer_column_weight(state))
        return float((np.asarray(state.tracers["dust"]) * dp * w).sum())

    def test_transport_keeps_the_mass_without_the_global_factor(self):
        dycore, state, moved = self._step(local=True)
        rel = self._mass(dycore, moved) / self._mass(dycore, state) - 1.0
        self.assertLess(abs(rel), 1e-4)
        self.assertGreaterEqual(float(jnp.min(moved.tracers["dust"])), 0.0)

    def test_without_it_the_limiter_creates_mass(self):
        # The control: the quasi-monotone limiter's creation the local fixer
        # removes (and the global proportional fixer would otherwise take
        # back from wherever the tracer is). Measured ~2.1e-5 at this grid's
        # resolution and blob size; the threshold is an order of magnitude
        # below that, not the measured value itself, so the test isn't
        # fragile to the exact random seed or grid.
        dycore, state, moved = self._step(local=False)
        rel = self._mass(dycore, moved) / self._mass(dycore, state) - 1.0
        self.assertGreater(rel, 1e-6)

    def test_locality_a_patchy_region_does_not_drain_a_distant_smooth_one(self):
        """A single sharp feature's correction stays near it.

        Reproduces, at unit-test scale, the mechanism #1062 reports: with
        only the global proportional fixer, the limiter's creation at the
        patch is returned in proportion to the *whole* field, so a distant,
        perfectly smooth reservoir loses mass even though nothing happened
        to it locally. With the local fixer the correction is confined to
        the departure-point neighbourhood of the patch itself.
        """
        dycore = _sl_dycore(local=True)
        state = self._patchy_state(dycore)
        # A spatially uniform background reservoir, far from the patch and
        # far larger than it, with no structure for the limiter to clip.
        background = 1e-6 * np.ones(state.tracers["dust"].shape)
        q0 = jnp.asarray(background, dtype=state.temperature_variation.dtype)
        patch = np.asarray(state.tracers["dust"])
        q0 = jnp.asarray(np.where(patch > 0, patch, q0),
                         dtype=state.temperature_variation.dtype)
        state = state.replace(tracers={**state.tracers, "dust": q0})

        moved = dycore._dynamics_step_fn(state)

        # The far side of the grid from the patch (patch sits at lon index
        # 10:13 of a length ~2*21+? nodal grid; take an antipodal longitude
        # band well away from it and away from the poles).
        far = np.asarray(moved.tracers["dust"])[3, -13:-10, 20:23]
        rel_far = far / background[3, -13:-10, 20:23] - 1.0
        self.assertLess(np.max(np.abs(rel_far)), 1e-6)


if __name__ == "__main__":
    unittest.main()
