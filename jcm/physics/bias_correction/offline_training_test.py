"""Tests for the offline warm-up pipeline (issue #356).

Synthetic and cloud-free: no ERA5, no Model run. Covers the target recompute
(matches ``nudging_tendency``), the column assembly, the normalisation stats,
that training drives the loss down, and that a trained term round-trips through
``save``/``load_bias_correction`` and is no longer a no-op.
"""

import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.nudging import NudgingConfig, NudgingTarget, nudging_tendency
from jcm.physics_interface import PhysicsState
from jcm.physics.bias_correction.nn_bias_correction import (
    init_mlp,
    make_bias_correction,
    load_bias_correction,
)
from jcm.physics.bias_correction.offline_training import (
    recompute_target_tendencies,
    assemble_training_arrays,
    compute_norm_stats,
    corrected_feature_mask,
    normalized_mse,
    train,
    build_term,
)

_CORRECT = ("temperature", "specific_humidity")


def _column_state(shape, key):
    """Build a realistic PhysicsState of shape ``(nlev, *horiz)``."""
    k = jax.random.split(key, 4)
    horiz = shape[1:]
    return PhysicsState(
        u_wind=jax.random.normal(k[0], shape),
        v_wind=jax.random.normal(k[1], shape),
        temperature=250.0 + jax.random.normal(k[2], shape),
        specific_humidity=1e-3 * (1.0 + jax.random.uniform(k[3], shape)),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones(horiz),
        tracers={},
    )


def _trajectory(ntime, nlev, nlon, nlat, key):
    """Build a (ntime, nlev, nlon, nlat) trajectory and matching ERA5 target."""
    ks = jax.random.split(key, 8)
    shape = (ntime, nlev, nlon, nlat)
    states = PhysicsState(
        u_wind=jax.random.normal(ks[0], shape),
        v_wind=jax.random.normal(ks[1], shape),
        temperature=250.0 + jax.random.normal(ks[2], shape),
        specific_humidity=1e-3 * (1.0 + jax.random.uniform(ks[3], shape)),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones((ntime, nlon, nlat)),
        tracers={},
    )
    target = NudgingTarget(
        u_wind=jax.random.normal(ks[4], shape),
        v_wind=jax.random.normal(ks[5], shape),
        temperature=250.0 + jax.random.normal(ks[6], shape),
        specific_humidity=1e-3 * (1.0 + jax.random.uniform(ks[7], shape)),
    )
    return states, target


class TestRecomputeTarget(unittest.TestCase):
    """recompute_target_tendencies reproduces the per-step nudge exactly."""

    def test_matches_single_step(self):
        ntime, nlev, nlon, nlat = 3, 4, 5, 6
        states, target = _trajectory(ntime, nlev, nlon, nlat, jax.random.key(0))
        config = NudgingConfig.temp_humidity(nlev, tau_seconds=86400.0)

        tends = recompute_target_tendencies(states, target, config)

        # Step 1 recomputed directly from the pure function.
        s1 = jax.tree.map(lambda a: a[1], states)
        t1 = jax.tree.map(lambda a: a[1], target)
        ref = nudging_tendency(s1, t1, config)
        np.testing.assert_allclose(
            np.asarray(tends.temperature[1]), np.asarray(ref.temperature), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(tends.specific_humidity[1]), np.asarray(ref.specific_humidity),
            rtol=1e-6)

    def test_winds_untouched_under_temp_humidity_config(self):
        ntime, nlev, nlon, nlat = 2, 4, 5, 6
        states, target = _trajectory(ntime, nlev, nlon, nlat, jax.random.key(1))
        config = NudgingConfig.temp_humidity(nlev, tau_seconds=86400.0)
        tends = recompute_target_tendencies(states, target, config)
        np.testing.assert_array_equal(
            np.asarray(tends.u_wind), np.zeros((ntime, nlev, nlon, nlat)))
        np.testing.assert_array_equal(
            np.asarray(tends.v_wind), np.zeros((ntime, nlev, nlon, nlat)))


class TestAssembleAndStats(unittest.TestCase):

    def test_assemble_shapes(self):
        ntime, nlev, nlon, nlat = 3, 4, 5, 6
        states, target = _trajectory(ntime, nlev, nlon, nlat, jax.random.key(2))
        config = NudgingConfig.temp_humidity(nlev)
        tends = recompute_target_tendencies(states, target, config)
        feats, targets = assemble_training_arrays(states, tends)
        n = ntime * nlon * nlat
        self.assertEqual(feats.shape, (n, 4 * nlev))
        self.assertEqual(targets.shape, (n, 4 * nlev))

    def test_norm_stats_shapes_and_uncorrected_scale(self):
        ntime, nlev, nlon, nlat = 3, 4, 5, 6
        states, target = _trajectory(ntime, nlev, nlon, nlat, jax.random.key(3))
        config = NudgingConfig.temp_humidity(nlev)
        tends = recompute_target_tendencies(states, target, config)
        feats, targets = assemble_training_arrays(states, tends)
        in_mean, in_std, out_scale = compute_norm_stats(feats, targets, _CORRECT)

        self.assertEqual(in_mean.shape, (4 * nlev,))
        self.assertEqual(out_scale.shape, (4 * nlev,))
        self.assertTrue(jnp.all(in_std > 0.0))
        # u and v slots are not corrected, so their scale stays exactly 1.
        mask = corrected_feature_mask(nlev, _CORRECT)
        np.testing.assert_array_equal(np.asarray(out_scale)[~mask], 1.0)
        # corrected slots carry the std of the recorded tendency (non-trivial).
        self.assertTrue(jnp.all(out_scale[mask] > 0.0))

    def test_out_scale_has_no_variance_floor(self):
        # July 10 lesson: out_scale is dual-use (it also scales the term's
        # live output), so compute_norm_stats deliberately does NOT floor
        # near-zero-variance slots. A squashed stratospheric-humidity level
        # keeps its tiny scale -> tiny correction authority, a safety
        # property; flooring it gave the net ~150x too much authority and the
        # free run blew up +30 K. (The online loss floors its own separate
        # normalisation in stds_from_stats; that one never touches the output.)
        nlev = 4
        n_io = 4 * nlev
        rng = np.random.default_rng(0)
        feats = jnp.asarray(rng.normal(size=(100, n_io)))
        targets_np = rng.normal(size=(100, n_io))
        iq = 1  # FIELD_ORDER[1] == specific_humidity
        targets_np[:, iq * nlev] *= 1e-9   # squash one humidity level
        _, _, out_scale = compute_norm_stats(feats, jnp.asarray(targets_np),
                                             _CORRECT)
        q_block = np.asarray(out_scale)[iq * nlev:(iq + 1) * nlev]
        # unfloored: the squashed slot stays tiny, far below floor_frac of the
        # block max the old floor would have lifted it to ...
        self.assertLess(float(q_block[0]), 1e-6)
        self.assertLess(float(q_block[0]), 0.05 * float(q_block.max()))
        # ... but still positive, so the normalisation stays well posed.
        self.assertGreater(float(q_block[0]), 0.0)


class TestTrainReducesLoss(unittest.TestCase):
    """Training a non-zero MLP on a learnable target drives the loss down."""

    def test_loss_drops(self):
        nlev, n = 4, 2000
        n_io = 4 * nlev
        key = jax.random.key(0)
        k_f, k_w, k_init = jax.random.split(key, 3)

        feats = jax.random.normal(k_f, (n, n_io))
        # Linear, learnable target on the T and q slots; winds left at zero.
        w = 0.5 * jax.random.normal(k_w, (n_io, n_io))
        mask = jnp.asarray(corrected_feature_mask(nlev, _CORRECT), dtype=feats.dtype)
        targets = (feats @ w) * mask * 1e-5

        in_mean, in_std, out_scale = compute_norm_stats(feats, targets, _CORRECT)

        zero_layers = init_mlp(k_init, (n_io, 32, n_io), zero_last_layer=True)
        before = normalized_mse(zero_layers, feats, targets,
                                in_mean, in_std, out_scale, _CORRECT)

        start = init_mlp(k_init, (n_io, 32, n_io), zero_last_layer=False)
        trained, history = train(start, feats, targets, in_mean, in_std, out_scale,
                                 _CORRECT, steps=400, batch_size=256, lr=1e-2,
                                 key=jax.random.key(1))
        after = normalized_mse(trained, feats, targets,
                               in_mean, in_std, out_scale, _CORRECT)

        self.assertGreater(before, 0.0)
        self.assertLess(after, 0.3 * before)
        self.assertLess(history[-1][1], history[0][1])


class TestSaveLoadRoundTrip(unittest.TestCase):
    """A trained term survives save/load and is no longer a no-op."""

    def test_roundtrip_matches_and_is_nonzero(self):
        term = make_bias_correction(nlev=6, key=jax.random.key(9),
                                    zero_last_layer=False)
        state = _column_state((6, 5), jax.random.key(0))
        tend, _ = term(state, {}, None, None)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "offline_term.npz")
            term.save(path)
            loaded = load_bias_correction(path)

        loaded_tend, _ = loaded(state, {}, None, None)
        np.testing.assert_allclose(np.asarray(loaded_tend.temperature),
                                   np.asarray(tend.temperature), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(loaded_tend.specific_humidity),
                                   np.asarray(tend.specific_humidity), rtol=1e-6)
        self.assertEqual(loaded.correct, term.correct)
        # A non-zero-init term is not a no-op.
        self.assertGreater(float(jnp.sum(jnp.abs(loaded_tend.temperature))), 0.0)

    def test_build_term_then_save(self):
        nlev = 4
        n_io = 4 * nlev
        layers = init_mlp(jax.random.key(2), (n_io, 16, n_io), zero_last_layer=False)
        in_mean = jnp.zeros(n_io)
        in_std = jnp.ones(n_io)
        out_scale = jnp.ones(n_io)
        term = build_term(layers, in_mean, in_std, out_scale, _CORRECT)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "term.npz")
            term.save(path)
            loaded = load_bias_correction(path)
        self.assertEqual(loaded.correct, _CORRECT)


if __name__ == "__main__":
    unittest.main()
