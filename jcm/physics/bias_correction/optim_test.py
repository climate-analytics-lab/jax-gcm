"""Tests for the shared hand-rolled optimizer helpers."""

import unittest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jcm.physics.bias_correction.optim import (
    adam_init,
    adam_update,
    clip_by_global_norm,
    global_norm,
)


class TestGlobalNorm(unittest.TestCase):
    """Norm and clipping across a multi-leaf pytree."""

    def test_global_norm_spans_leaves(self):
        tree = {"a": jnp.array([3.0]), "b": jnp.array([4.0])}
        self.assertAlmostEqual(float(global_norm(tree)), 5.0, places=6)

    def test_clip_scales_large_tree_to_max_norm(self):
        tree = {"a": jnp.array([3.0]), "b": jnp.array([4.0])}   # norm 5
        clipped, norm = clip_by_global_norm(tree, 1.0)
        self.assertAlmostEqual(float(norm), 5.0, places=6)
        self.assertAlmostEqual(float(global_norm(clipped)), 1.0, places=5)
        # Direction is preserved, only the magnitude shrinks.
        np.testing.assert_allclose(np.asarray(clipped["a"]) * 5.0,
                                   np.asarray(tree["a"]), rtol=1e-5)

    def test_clip_leaves_small_tree_untouched(self):
        tree = (jnp.array([0.3, 0.4]),)                          # norm 0.5
        clipped, norm = clip_by_global_norm(tree, 1.0)
        self.assertAlmostEqual(float(norm), 0.5, places=6)
        np.testing.assert_array_equal(np.asarray(clipped[0]),
                                      np.asarray(tree[0]))

    def test_clip_zero_tree_is_finite(self):
        clipped, norm = clip_by_global_norm((jnp.zeros(3),), 1.0)
        self.assertEqual(float(norm), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(clipped[0]))))


class TestAdam(unittest.TestCase):
    """The extracted Adam reproduces the standard update."""

    def test_first_step_matches_hand_computation(self):
        # With zero moments and t=1 the bias-corrected step is exactly
        # lr * g / (|g| * sqrt(1-b2)/sqrt(1-b2) + eps) ~= lr * sign(g).
        params = (jnp.array([1.0]),)
        grads = (jnp.array([2.0]),)
        m, v = adam_init(params)
        lr = 1e-3
        new_params, m, v = adam_update(params, grads, m, v,
                                       jnp.asarray(1.0), lr)
        b1, b2, eps = 0.9, 0.999, 1e-8
        mm = (1 - b1) * 2.0
        vv = (1 - b2) * 4.0
        scale = lr * jnp.sqrt(1 - b2) / (1 - b1)
        expected = 1.0 - scale * mm / (jnp.sqrt(vv) + eps)
        self.assertAlmostEqual(float(new_params[0][0]), float(expected),
                               places=7)

    def test_converges_on_quadratic(self):
        # Minimise |x - 3|^2; Adam should get close in a few hundred steps.
        params = (jnp.array([0.0]),)
        m, v = adam_init(params)
        loss = lambda p: jnp.sum((p[0] - 3.0) ** 2)
        for t in range(1, 400):
            grads = jax.grad(loss)(params)
            params, m, v = adam_update(params, grads, m, v,
                                       jnp.asarray(float(t)), lr=5e-2)
        self.assertLess(abs(float(params[0][0]) - 3.0), 0.05)

    def test_jit_compatible(self):
        params = (jnp.ones(4),)
        grads = (jnp.ones(4),)
        m, v = adam_init(params)
        step = jax.jit(lambda p, g, m, v, t: adam_update(p, g, m, v, t, 1e-3))
        out, _, _ = step(params, grads, m, v, jnp.asarray(1.0))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out[0]))))


class TestAdamPerParameterLR(unittest.TestCase):
    """A pytree ``lr`` gives each leaf its own rate.

    This exists for the widened-context problem: Adam's step is bounded by
    ``lr`` whatever the gradient, so a zero-initialised row added at the
    climatology stage cannot move further than ``updates * lr``. The shipped
    budget caps it at ~6% of Glorot scale and the real terms reached 0.4%,
    which makes the new input numerically inert. A per-leaf rate lets those
    rows train without touching the converged ones.
    """

    def test_uniform_pytree_lr_matches_scalar_lr(self):
        # The per-leaf path must be a strict generalisation: filling the tree
        # with one value has to reproduce the scalar path exactly, or existing
        # runs would silently change.
        params = (jnp.array([1.0, -2.0]), jnp.array([0.5]))
        grads = (jnp.array([0.3, 0.7]), jnp.array([-0.2]))
        m, v = adam_init(params)
        t = jnp.asarray(3.0)

        scalar, _, _ = adam_update(params, grads, m, v, t, 1e-3)
        tree_lr = jtu.tree_map(lambda p: jnp.full_like(p, 1e-3), params)
        per_leaf, _, _ = adam_update(params, grads, m, v, t, tree_lr)

        for a, b in zip(scalar, per_leaf, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)

    def test_zero_lr_leaf_is_frozen(self):
        # The mask this enables: rate 0 on converged weights, non-zero on the
        # new rows, so a warm start can train an added input in place.
        params = (jnp.array([1.0]), jnp.array([1.0]))
        grads = (jnp.array([1.0]), jnp.array([1.0]))
        m, v = adam_init(params)
        tree_lr = (jnp.zeros_like(params[0]), jnp.full_like(params[1], 1e-2))

        out, _, _ = adam_update(params, grads, m, v, jnp.asarray(1.0), tree_lr)

        np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(params[0]))
        self.assertLess(float(out[1][0]), 1.0)

    def test_within_leaf_mask_freezes_selected_rows(self):
        # The real use is a mask INSIDE one kernel: old rows frozen, appended
        # rows live. Broadcasting within a leaf has to work, not just per-leaf.
        params = (jnp.ones((3, 2)),)
        grads = (jnp.ones((3, 2)),)
        m, v = adam_init(params)
        mask = jnp.array([[0.0, 0.0], [0.0, 0.0], [1e-2, 1e-2]])

        out, _, _ = adam_update(params, grads, m, v, jnp.asarray(1.0), (mask,))

        np.testing.assert_array_equal(np.asarray(out[0][:2]), np.ones((2, 2)))
        self.assertLess(float(out[0][2, 0]), 1.0)

    def test_per_parameter_lr_is_jit_compatible(self):
        params = (jnp.ones(4),)
        grads = (jnp.ones(4),)
        m, v = adam_init(params)
        step = jax.jit(adam_update)
        out, _, _ = step(params, grads, m, v, jnp.asarray(1.0),
                         (jnp.full(4, 1e-3),))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out[0]))))


if __name__ == "__main__":
    unittest.main()
