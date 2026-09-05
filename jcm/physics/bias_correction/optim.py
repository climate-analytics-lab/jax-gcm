"""Hand-rolled Adam and gradient clipping shared by the bias-correction trainers.

The repo deliberately carries no optax dependency, so the offline trainer
rolled its own Adam. The online trainer needs the identical update plus
global-norm clipping (backprop through a rollout produces occasional gradient
spikes the offline problem never sees), so the optimizer math lives here once
instead of twice. Everything is a pure function over pytrees, safe under
``jax.jit``.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.tree_util as jtu

ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8


def adam_init(params):
    """Zero first/second-moment pytrees matching ``params``.

    Returns:
        ``(m, v)`` with the structure of ``params``.

    """
    return (jtu.tree_map(jnp.zeros_like, params),
            jtu.tree_map(jnp.zeros_like, params))


def adam_update(params, grads, m, v, t, lr, *,
                b1: float = ADAM_B1, b2: float = ADAM_B2,
                eps: float = ADAM_EPS):
    """One Adam step with bias correction.

    Args:
        params, grads: parameter pytree and its gradient.
        m, v: first/second moments from :func:`adam_init` (or a prior step).
        t: 1-based step counter (float array so it traces under jit).
        lr: learning rate. Either a scalar, or a pytree with the same
            structure as ``params`` giving a PER-PARAMETER rate. The
            per-parameter form exists because Adam's step size is bounded by
            ``lr`` regardless of gradient magnitude, so a freshly widened
            zero row cannot travel further than ``updates * lr`` no matter how
            informative its input is. At the shipped climatology budget
            (210 updates, lr 2.5e-5) that ceiling is 5.3e-3 against a Glorot
            scale of 8.3e-2, and the two shipped context terms landed at
            3.4e-4 and 3.0e-4: about 0.4% of Glorot, small enough that the
            input is numerically inert. A masked rate lets new rows train at
            their own speed without disturbing the converged weights.
        b1, b2, eps: standard Adam constants.

    Returns:
        ``(params, m, v)`` updated.

    """
    m = jtu.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
    v = jtu.tree_map(lambda a, g: b2 * a + (1 - b2) * g ** 2, v, grads)
    bias_corr = jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    if jtu.tree_structure(lr) == jtu.tree_structure(params):
        params = jtu.tree_map(
            lambda p, mm, vv, ll: p - ll * bias_corr * mm / (jnp.sqrt(vv) + eps),
            params, m, v, lr)
    else:
        # Scalar path, kept exactly as it was so existing runs are unchanged.
        scale = lr * bias_corr
        params = jtu.tree_map(
            lambda p, mm, vv: p - scale * mm / (jnp.sqrt(vv) + eps), params, m, v)
    return params, m, v


def global_norm(tree) -> jnp.ndarray:
    """L2 norm over every leaf of ``tree`` taken together."""
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in jtu.tree_leaves(tree)))


def clip_by_global_norm(tree, max_norm: float):
    """Scale ``tree`` so its global norm is at most ``max_norm``.

    Returns:
        ``(clipped_tree, pre_clip_norm)``. The pre-clip norm is returned so
        the caller can log it and use it in finiteness guards.

    """
    norm = global_norm(tree)
    # The epsilon keeps the scale finite when the norm is exactly zero.
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return jtu.tree_map(lambda a: a * scale, tree), norm
