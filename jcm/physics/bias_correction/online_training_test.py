"""Tests for the online (backprop-through-rollout) bias-correction trainer.

The fast tests drive the trainer with the fake cubed-sphere dycore (identity
dynamics, forward-Euler physics add): they prove the plumbing, the gradient
path through ``Model.run_from_state``, and the guard/curriculum logic, in
seconds and with no SPEEDY in the loop. A single ``@pytest.mark.slow`` test
repeats the gradient check on real SPEEDY at T21.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.dycore._fake_cubed_sphere import FakeCubedSphereDycore
from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.physics.bias_correction.nn_bias_correction import init_mlp
from jcm.physics.bias_correction.offline_training import build_term
from jcm.physics.bias_correction.online_training import (
    CurriculumStage,
    default_curriculum,
    lat_weights,
    level_weights,
    make_climatology_loss,
    make_combined_loss,
    make_rollout_loss,
    make_train_step,
    state_error,
    stds_from_stats,
    train_online,
)
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.bias_correction.optim import adam_init


NELEM, GLL, NLEV = 2, 2, 4
N_IO = 4 * NLEV
DT_SECONDS = 1800.0
TIME_STEP_MINUTES = 30.0


def _dycore():
    return FakeCubedSphereDycore(nelem=NELEM, gll=GLL, nlev=NLEV,
                                 dt_seconds=DT_SECONDS)


def _buffers():
    """Identity normalisation, per-second output scale of one unit/day."""
    in_mean = jnp.zeros(N_IO)
    in_std = jnp.ones(N_IO)
    out_scale = jnp.ones(N_IO) * (1.0 / 86400.0)
    return in_mean, in_std, out_scale


def _make_build_model(dycore):
    """``layers -> Model`` with only the bias term as physics.

    Identity dynamics plus a single term keeps every degree of freedom in
    the network, so the rollout is exactly (state + sum of corrections) and
    the tests see the gradient path through ``run_from_state`` undiluted.
    """
    in_mean, in_std, out_scale = _buffers()

    def build_model(layers):
        term = build_term(layers, in_mean, in_std, out_scale,
                          ("temperature", "specific_humidity"))
        return Model(dycore=dycore, physics=ComposablePhysics([term]),
                     time_step=TIME_STEP_MINUTES)

    return build_model


def _loss_ingredients():
    T_std = jnp.ones(NLEV)
    q_std = jnp.ones(NLEV)
    weights = jnp.ones(())
    return T_std, q_std, weights


def _forcing(dycore):
    return ForcingData.zeros(dycore.coords.horizontal.nodal_shape)


def _init_and_targets(dycore, dT=0.5, dq=0.1):
    """Build an initial state plus targets offset from it by a constant."""
    init = dycore.initial_state(None)
    ps = dycore.to_physics_state(init)
    return init, ps.temperature + dT, ps.specific_humidity + dq


class TestLossIngredients(unittest.TestCase):
    """lat_weights, stds_from_stats, and state_error building blocks."""

    def test_lat_weights_fallback_is_scalar_one(self):
        class _H:
            pass  # no latitudes attribute

        class _C:
            horizontal = _H()

        w = lat_weights(_C())
        self.assertEqual(float(w), 1.0)

    def test_lat_weights_per_point_latitudes_broadcast(self):
        # The fake cubed-sphere coords carry per-gridpoint latitudes; the
        # weights then cover the whole horizontal shape with mean one.
        coords = _dycore().coords
        w = lat_weights(coords)
        self.assertEqual(w.shape, coords.horizontal.nodal_shape)
        self.assertAlmostEqual(float(jnp.mean(w)), 1.0, places=6)

    def test_lat_weights_cosine_mean_one(self):
        class _H:
            latitudes = np.linspace(-1.2, 1.2, 8)

        class _C:
            horizontal = _H()

        w = lat_weights(_C())
        self.assertEqual(w.shape, (8,))
        self.assertAlmostEqual(float(jnp.mean(w)), 1.0, places=6)

    def test_lat_weights_floor_lifts_poles(self):
        # cos(lat) ~ 0 at the poles; a floor lifts those rows into the loss.
        class _H:
            latitudes = np.array([-np.pi / 2, -np.pi / 4, 0.0,
                                  np.pi / 4, np.pi / 2])

        class _C:
            horizontal = _H()

        # Floor 0.0 reproduces today's pure-cosine weighting exactly.
        w_none = lat_weights(_C(), pole_weight_floor=0.0)
        np.testing.assert_allclose(np.asarray(w_none),
                                   np.asarray(lat_weights(_C())), rtol=1e-6)
        self.assertAlmostEqual(float(jnp.min(w_none)), 0.0, places=6)

        # A positive floor lifts the polar weight off zero, still mean 1.
        w_floor = lat_weights(_C(), pole_weight_floor=0.3)
        self.assertAlmostEqual(float(jnp.mean(w_floor)), 1.0, places=6)
        self.assertGreater(float(jnp.min(w_floor)), 0.0)
        self.assertGreater(float(jnp.min(w_floor)), float(jnp.min(w_none)))

    def test_stds_floor_protects_tiny_levels(self):
        in_std = jnp.ones(N_IO)
        # Zero out one humidity level; the floor must lift it off zero.
        iq = 1  # FIELD_ORDER[1] == specific_humidity
        in_std = in_std.at[iq * NLEV].set(1e-12)
        T_std, q_std = stds_from_stats(in_std, NLEV, floor_frac=0.05)
        self.assertGreaterEqual(float(jnp.min(q_std)), 0.05)
        np.testing.assert_allclose(np.asarray(T_std), np.ones(NLEV))

    def test_state_error_zero_for_exact_prediction(self):
        dycore = _dycore()
        init = dycore.initial_state(None)
        ps = dycore.to_physics_state(init)
        T_std, q_std, w = _loss_ingredients()
        err = state_error(ps, ps.temperature, ps.specific_humidity,
                          T_std, q_std, w)
        self.assertEqual(float(err), 0.0)


class TestLevelWeights(unittest.TestCase):
    """Vertical loss weighting: the one axis the loss never weighted.

    Latitude weighting has always existed, so every configuration traded
    near-surface accuracy against the mid-troposphere with no way to say which
    it preferred. These cover the dial and, most importantly, that leaving it
    unset reproduces the old unweighted loss exactly.
    """

    def _one_level_error(self, k):
        """Prediction equal to target except for a 1 K offset at level ``k``."""
        dycore = _dycore()
        ps = dycore.to_physics_state(dycore.initial_state(None))
        target_T = ps.temperature.at[k].add(1.0)
        T_std, q_std, w = _loss_ingredients()

        def err(level_w):
            return float(state_error(ps, target_T, ps.specific_humidity,
                                     T_std, q_std, w, level_w))

        return err

    def test_none_is_scalar_one(self):
        self.assertEqual(float(level_weights(None, NLEV)), 1.0)

    def test_uniform_spec_normalizes_to_all_ones(self):
        # Any constant spec is the same dial setting: no vertical preference.
        for constant in (1.0, 7.5):
            w = level_weights([constant] * NLEV, NLEV)
            np.testing.assert_allclose(np.asarray(w), np.ones(NLEV), rtol=1e-6)

    def test_mean_one_and_ratios_preserved(self):
        spec = [1.0, 3.0] + [1.0] * (NLEV - 2)
        w = level_weights(spec, NLEV)
        self.assertAlmostEqual(float(jnp.mean(w)), 1.0, places=6)
        self.assertAlmostEqual(float(w[1]) / float(w[0]), 3.0, places=6)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            level_weights([1.0] * (NLEV + 1), NLEV)

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            level_weights([-1.0] + [1.0] * (NLEV - 1), NLEV)

    def test_rejects_all_zero(self):
        with self.assertRaises(ValueError):
            level_weights([0.0] * NLEV, NLEV)

    def test_uniform_reproduces_the_unweighted_loss(self):
        # The regression that matters: existing runs must not move.
        err = self._one_level_error(k=1)
        self.assertAlmostEqual(err(level_weights([1.0] * NLEV, NLEV)),
                               err(None), places=10)

    def test_boosting_a_level_raises_the_loss_it_contributes(self):
        k = 1
        err = self._one_level_error(k)
        baseline = err(None)
        boosted = [1.0] * NLEV
        boosted[k] = 4.0
        suppressed = [1.0] * NLEV
        suppressed[k] = 0.25
        self.assertGreater(err(level_weights(boosted, NLEV)), baseline)
        self.assertLess(err(level_weights(suppressed, NLEV)), baseline)

    def test_zero_weight_removes_a_level_from_the_loss(self):
        # Error lives only at level k, so zeroing k must zero the loss.
        k = 2
        err = self._one_level_error(k)
        spec = [1.0] * NLEV
        spec[k] = 0.0
        self.assertGreater(err(None), 0.0)
        self.assertAlmostEqual(err(level_weights(spec, NLEV)), 0.0, places=10)

    def test_exact_prediction_stays_zero_under_weighting(self):
        dycore = _dycore()
        ps = dycore.to_physics_state(dycore.initial_state(None))
        T_std, q_std, w = _loss_ingredients()
        level_w = level_weights([1.0, 5.0] + [1.0] * (NLEV - 2), NLEV)
        err = state_error(ps, ps.temperature, ps.specific_humidity,
                          T_std, q_std, w, level_w)
        self.assertEqual(float(err), 0.0)


class TestRolloutLoss(unittest.TestCase):
    """Gradient flow and baselines through Model.run_from_state."""

    def setUp(self):
        self.dycore = _dycore()
        self.build_model = _make_build_model(self.dycore)
        self.forcing = _forcing(self.dycore)
        self.T_std, self.q_std, self.w = _loss_ingredients()

    def _loss_fn(self, n_steps):
        return make_rollout_loss(
            self.build_model, self.forcing, n_steps=n_steps,
            time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)

    def test_gradient_is_finite_and_nonzero(self):
        layers = init_mlp(jax.random.key(0), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        grads = jax.grad(self._loss_fn(n_steps=2))(layers, init, tT, tq)
        total = 0.0
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
            total += float(jnp.sum(jnp.abs(leaf)))
        self.assertGreater(total, 0.0)

    def test_zero_init_term_matches_no_term_baseline(self):
        # The no-op guarantee must survive the rollout: a zero-init term
        # gives exactly the loss of a model with no correction at all.
        zero_layers = init_mlp(jax.random.key(1), (N_IO, 8, N_IO),
                               zero_last_layer=True)
        init, tT, tq = _init_and_targets(self.dycore)
        loss_with_term = self._loss_fn(n_steps=2)(zero_layers, init, tT, tq)

        bare_model = Model(dycore=self.dycore,
                           physics=ComposablePhysics([]),
                           time_step=TIME_STEP_MINUTES)
        dt_days = TIME_STEP_MINUTES / (60.0 * 24.0)
        final, _ = bare_model.run_from_state(
            init, self.forcing,
            save_interval=2 * dt_days * (1 + 1e-9),
            total_time=2 * dt_days * (1 + 1e-9))
        pred = self.dycore.to_physics_state(final)
        baseline = state_error(pred, tT, tq, self.T_std, self.q_std, self.w)
        self.assertAlmostEqual(float(loss_with_term), float(baseline),
                               places=6)

    def test_rollout_advances_expected_steps(self):
        # Guards the int(save_interval / dt) truncation: sim_time must
        # advance by exactly n_steps * dt.
        layers = init_mlp(jax.random.key(2), (N_IO, 8, N_IO))
        model = self.build_model(layers)
        init = self.dycore.initial_state(None)
        for n_steps in (1, 2, 12):
            dt_days = TIME_STEP_MINUTES / (60.0 * 24.0)
            span = n_steps * dt_days * (1 + 1e-9)
            final, _ = model.run_from_state(init, self.forcing,
                                            save_interval=span,
                                            total_time=span)
            advance = float(self.dycore.sim_time(final)
                            - self.dycore.sim_time(init))
            self.assertAlmostEqual(advance, n_steps * DT_SECONDS, delta=1e-3)


class TestClimatologyLoss(unittest.TestCase):
    """The free-running time-mean loss: gradient flow and the zero-term baseline."""

    def setUp(self):
        self.dycore = _dycore()
        self.build_model = _make_build_model(self.dycore)
        self.forcing = _forcing(self.dycore)
        self.T_std, self.q_std, self.w = _loss_ingredients()

    def _loss_fn(self, n_steps):
        return make_climatology_loss(
            self.build_model, self.forcing, n_steps=n_steps,
            time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)

    def _bare_mean_state(self, init, n_steps):
        bare_model = Model(dycore=self.dycore,
                           physics=ComposablePhysics([]),
                           time_step=TIME_STEP_MINUTES)
        dt_days = TIME_STEP_MINUTES / (60.0 * 24.0)
        span = n_steps * dt_days * (1 + 1e-9)
        _final, preds = bare_model.run_from_state(
            init, self.forcing, save_interval=span, total_time=span,
            output_averages=True)
        return jax.tree_util.tree_map(lambda a: a[0], preds.dynamics)

    def test_gradient_is_finite_and_nonzero(self):
        layers = init_mlp(jax.random.key(0), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        grads = jax.grad(self._loss_fn(n_steps=4))(layers, init, tT, tq)
        total = 0.0
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
            total += float(jnp.sum(jnp.abs(leaf)))
        self.assertGreater(total, 0.0)

    def test_zero_init_term_matches_no_term_baseline(self):
        # The no-op guarantee must survive the averaged path too: a zero-init
        # term gives exactly the time-mean loss of a model with no correction.
        zero_layers = init_mlp(jax.random.key(1), (N_IO, 8, N_IO),
                               zero_last_layer=True)
        init, tT, tq = _init_and_targets(self.dycore)
        loss_with_term = self._loss_fn(n_steps=4)(zero_layers, init, tT, tq)
        baseline = state_error(self._bare_mean_state(init, 4), tT, tq,
                               self.T_std, self.q_std, self.w)
        self.assertAlmostEqual(float(loss_with_term), float(baseline),
                               places=6)


class TestCombinedLoss(unittest.TestCase):
    """rollout + lam*clim: exact decomposition, gradient flow, packed targets."""

    def setUp(self):
        self.dycore = _dycore()
        self.build_model = _make_build_model(self.dycore)
        self.forcing = _forcing(self.dycore)
        self.T_std, self.q_std, self.w = _loss_ingredients()

    def _combined(self, lam=1.0, n_roll=2, n_clim=4):
        return make_combined_loss(
            self.build_model, self.forcing, n_rollout_steps=n_roll,
            n_clim_steps=n_clim, lam=lam, time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)

    def test_combined_equals_rollout_plus_lam_clim(self):
        # The whole point of the combined loss is that it is exactly the sum;
        # distinct rollout vs climatology targets make the decomposition a real
        # check rather than a tautology.
        n_roll, n_clim, lam = 2, 4, 0.5
        layers = init_mlp(jax.random.key(0), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        rollT, climT = tT, tT + 0.2
        rollq, climq = tq, tq + 0.05

        roll = make_rollout_loss(
            self.build_model, self.forcing, n_steps=n_roll,
            time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)
        clim = make_climatology_loss(
            self.build_model, self.forcing, n_steps=n_clim,
            time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)
        comb = self._combined(lam=lam, n_roll=n_roll, n_clim=n_clim)

        expected = (float(roll(layers, init, rollT, rollq))
                    + lam * float(clim(layers, init, climT, climq)))
        got = float(comb(layers, init, (rollT, climT), (rollq, climq)))
        self.assertAlmostEqual(got, expected, places=5)

    def test_gradient_is_finite_and_nonzero(self):
        comb = self._combined()
        layers = init_mlp(jax.random.key(1), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        grads = jax.grad(comb)(layers, init, (tT, tT + 0.2), (tq, tq + 0.05))
        total = 0.0
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
            total += float(jnp.sum(jnp.abs(leaf)))
        self.assertGreater(total, 0.0)

    def test_packed_targets_run_through_train_step(self):
        # The packed-target trick must survive make_train_step untouched.
        comb = self._combined()
        layers = init_mlp(jax.random.key(2), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        step = make_train_step(comb, lr=1e-3, clip_norm=10.0)
        m, v = adam_init(layers)
        _new_layers, _, _, loss, _gn, ok = step(
            layers, m, v, jnp.asarray(1.0), init,
            (tT, tT + 0.2), (tq, tq + 0.05))
        self.assertTrue(bool(ok))
        self.assertTrue(bool(jnp.isfinite(loss)))


class TestTrainStep(unittest.TestCase):
    """The jitted step optimises and the NaN guard holds."""

    def setUp(self):
        self.dycore = _dycore()
        self.build_model = _make_build_model(self.dycore)
        self.forcing = _forcing(self.dycore)
        self.T_std, self.q_std, self.w = _loss_ingredients()
        self.loss_fn = make_rollout_loss(
            self.build_model, self.forcing, n_steps=2,
            time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)

    def test_loss_decreases_over_a_few_updates(self):
        layers = init_mlp(jax.random.key(3), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        step = make_train_step(self.loss_fn, lr=1e-2, clip_norm=10.0)
        m, v = adam_init(layers)
        first = float(self.loss_fn(layers, init, tT, tq))
        for t in range(1, 30):
            layers, m, v, loss, _, ok = step(
                layers, m, v, jnp.asarray(float(t)), init, tT, tq)
            self.assertTrue(bool(ok))
        last = float(self.loss_fn(layers, init, tT, tq))
        self.assertLess(last, first)

    def test_nan_guard_keeps_weights(self):
        layers = init_mlp(jax.random.key(4), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        step = make_train_step(self.loss_fn, lr=1e-2, clip_norm=10.0)
        m, v = adam_init(layers)
        bad_T = tT.at[0].set(jnp.nan)
        new_layers, _, _, loss, _, ok = step(
            layers, m, v, jnp.asarray(1.0), init, bad_T, tq)
        self.assertFalse(bool(ok))
        self.assertFalse(bool(jnp.isfinite(loss)))
        for a, b in zip(jax.tree_util.tree_leaves(new_layers),
                        jax.tree_util.tree_leaves(layers)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestTrainOnline(unittest.TestCase):
    """Curriculum driver: self-consistent learning, schedule, abort."""

    def setUp(self):
        self.dycore = _dycore()
        self.build_model = _make_build_model(self.dycore)
        self.forcing = _forcing(self.dycore)
        self.T_std, self.q_std, self.w = _loss_ingredients()

    def _stack(self, trees):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)

    def test_default_curriculum_is_monotone(self):
        stages = default_curriculum()
        lengths = [s.n_steps for s in stages]
        self.assertEqual(lengths, sorted(lengths))
        self.assertTrue(all(s.updates > 0 for s in stages))

    def test_learns_a_self_consistent_target(self):
        # Targets generated by a rollout with "truth" weights; training a
        # perturbed copy toward them must reduce the loss.
        truth = init_mlp(jax.random.key(5), (N_IO, 8, N_IO),
                         zero_last_layer=False)
        n_steps = 2
        loss_fn = make_rollout_loss(
            self.build_model, self.forcing, n_steps=n_steps,
            time_step_minutes=TIME_STEP_MINUTES,
            T_std=self.T_std, q_std=self.q_std, weights=self.w)

        inits = [self.dycore.initial_state(None, sim_time=0.0),
                 self.dycore.initial_state(None, sim_time=DT_SECONDS)]
        dt_days = TIME_STEP_MINUTES / (60.0 * 24.0)
        span = n_steps * dt_days * (1 + 1e-9)
        truth_model = self.build_model(truth)
        targets_T, targets_q = [], []
        for s in inits:
            final, _ = truth_model.run_from_state(
                s, self.forcing, save_interval=span, total_time=span)
            ps = self.dycore.to_physics_state(final)
            targets_T.append(ps.temperature)
            targets_q.append(ps.specific_humidity)

        samples = {n_steps: (self._stack(inits),
                             jnp.stack(targets_T), jnp.stack(targets_q))}
        start = jax.tree_util.tree_map(
            lambda a: a + 0.3 * jax.random.normal(jax.random.key(6), a.shape),
            truth)
        before = float(loss_fn(start, inits[0], targets_T[0], targets_q[0]))
        trained, history = train_online(
            start, self.build_model, self.forcing, samples,
            (CurriculumStage(n_steps=n_steps, updates=20, lr=1e-2),),
            T_std=self.T_std, q_std=self.q_std, weights=self.w,
            time_step_minutes=TIME_STEP_MINUTES, key=jax.random.key(7))
        after = float(loss_fn(trained, inits[0], targets_T[0], targets_q[0]))
        self.assertLess(after, before)
        self.assertEqual(len(history), 20)
        self.assertTrue(all(h["ok"] for h in history))

    def test_stage_end_hook_fires_once_per_stage(self):
        layers = init_mlp(jax.random.key(10), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        samples = {1: (self._stack([init]), jnp.stack([tT]), jnp.stack([tq])),
                   2: (self._stack([init]), jnp.stack([tT]), jnp.stack([tq]))}
        stages = (CurriculumStage(n_steps=1, updates=2, lr=1e-3),
                  CurriculumStage(n_steps=2, updates=2, lr=1e-3))
        calls = []
        train_online(
            layers, self.build_model, self.forcing, samples, stages,
            T_std=self.T_std, q_std=self.q_std, weights=self.w,
            time_step_minutes=TIME_STEP_MINUTES, key=jax.random.key(11),
            stage_end_hook=lambda i, s, ly: calls.append((i, s.n_steps)))
        self.assertEqual(calls, [(0, 1), (1, 2)])

    def test_stage_aborts_after_max_bad_steps(self):
        layers = init_mlp(jax.random.key(8), (N_IO, 8, N_IO),
                          zero_last_layer=False)
        init, tT, tq = _init_and_targets(self.dycore)
        bad_T = jnp.full_like(tT, jnp.nan)
        samples = {2: (self._stack([init]),
                       jnp.stack([bad_T]), jnp.stack([tq]))}
        trained, history = train_online(
            layers, self.build_model, self.forcing, samples,
            (CurriculumStage(n_steps=2, updates=10, lr=1e-2),),
            T_std=self.T_std, q_std=self.q_std, weights=self.w,
            time_step_minutes=TIME_STEP_MINUTES, key=jax.random.key(9),
            max_bad_steps=3)
        self.assertEqual(len(history), 3)          # aborted, not 10
        self.assertTrue(not any(h["ok"] for h in history))
        for a, b in zip(jax.tree_util.tree_leaves(trained),
                        jax.tree_util.tree_leaves(layers)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSpeedyRolloutGradient(unittest.TestCase):
    """The gradient path holds on real SPEEDY dynamics (reduced resolution)."""

    @pytest.mark.slow
    def test_speedy_t21_rollout_gradient_finite(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.terrain import TerrainData

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        forcing = ForcingData.zeros(coords.horizontal.nodal_shape)
        nlev = coords.vertical.layers
        n_io = 4 * nlev
        in_mean = jnp.zeros(n_io)
        in_std = jnp.ones(n_io)
        out_scale = jnp.ones(n_io) * (1.0 / 86400.0)

        def build_model(layers):
            term = build_term(layers, in_mean, in_std, out_scale,
                              ("temperature", "specific_humidity"))
            return Model(coords=coords, terrain=terrain,
                         physics=speedy_physics() + term)

        probe = build_model(init_mlp(jax.random.key(0), (n_io, 8, n_io)))
        probe.bootstrap_state()          # populates, does not return
        init = probe._final_dycore_state

        T_std = jnp.ones(nlev)
        q_std = jnp.ones(nlev)
        weights = lat_weights(coords)
        loss_fn = make_rollout_loss(
            build_model, forcing, n_steps=2,
            T_std=T_std, q_std=q_std, weights=weights)

        layers = init_mlp(jax.random.key(1), (n_io, 16, n_io),
                          zero_last_layer=False)
        ps = probe.dycore.to_physics_state(init)
        loss, grads = jax.value_and_grad(loss_fn)(
            layers, init, ps.temperature, ps.specific_humidity)
        self.assertTrue(bool(jnp.isfinite(loss)))
        for leaf in jax.tree_util.tree_leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))


if __name__ == "__main__":
    unittest.main()
