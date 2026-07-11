"""Width-parameterized smooth replacements for hard branches in SPEEDY physics.

The SPEEDY schemes gate fluxes and diagnostics with hard comparisons
(``drh > drh0``, ``clip(x, 0, 1)``, ``max(x, 0)``). The comparisons whose
gated quantity does not vanish at the threshold put value jumps in the
model's parameter and state dependence, and every hinge zeroes a gradient
on one side. These helpers replace them with sigmoid gates, softplus
hinges, and hyperbolic min/max of a caller-chosen half-width.

All helpers accept ``width = 0`` and then reproduce the hard operation
exactly (bit-identical forward), so the default parameters leave SPEEDY
untouched and the pinned regression references stay valid. The width-0
branch is guarded with the double-where pattern so it cannot leak a
division-by-zero cotangent (see JAX_gotchas.md).

The width is a physical scale in the units of the gated variable (an RH
fraction, an energy in J/kg, a humidity in g/kg): roughly the range over
which the hard switch is smeared. Choose it small against the natural
variability of the argument so the forward change stays a perturbation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _safe_width(width):
    """Width guarded for use as a divisor when it may be exactly zero."""
    on = width > 0.0
    return on, jnp.where(on, width, 1.0)


def smooth_gate(x, threshold, width):
    """Sigmoid gate in [0, 1]: 1 well above ``threshold``, 0 well below.

    ``width = 0`` gives the hard indicator ``(x > threshold)`` as a float.
    """
    on, w = _safe_width(width)
    return jnp.where(
        on,
        jax.nn.sigmoid((x - threshold) / w),
        (x > threshold).astype(jnp.result_type(x)),
    )


def smooth_pos(x, width):
    """Softplus positive part: ``max(x, 0)`` smeared over ``width``.

    Exact ``jnp.maximum(x, 0)`` at ``width = 0``. Overshoots the hard
    hinge by ``width·log(2)`` at the corner and decays exponentially into
    the clipped side.
    """
    on, w = _safe_width(width)
    return jnp.where(on, w * jax.nn.softplus(x / w), jnp.maximum(x, 0.0))


def smooth_min(a, b, width):
    """Hyperbolic smooth minimum; exact ``jnp.minimum`` at ``width = 0``.

    Undershoots the hard minimum by at most ``width/2`` where ``a == b``.
    The width inside the sqrt is floored to 1 when the gate is off so the
    unselected branch cannot produce a ``sqrt(0)`` NaN cotangent at
    ``a == b``.
    """
    on, w = _safe_width(width)
    gap = a - b
    soft = 0.5 * (a + b - jnp.sqrt(gap * gap + w * w))
    return jnp.where(on, soft, jnp.minimum(a, b))


def smooth_max(a, b, width):
    """Hyperbolic smooth maximum; exact ``jnp.maximum`` at ``width = 0``."""
    on, w = _safe_width(width)
    gap = a - b
    soft = 0.5 * (a + b + jnp.sqrt(gap * gap + w * w))
    return jnp.where(on, soft, jnp.maximum(a, b))


def smooth_clip01(x, width):
    """Softplus-pair soft clip to [0, 1]; exact ``jnp.clip(x, 0, 1)`` at 0.

    Identity in the interior, exponential (never exactly flat) tails at
    the edges, so gradients survive saturation. Same construction as the
    Sundqvist cover ``smooth_b0`` (mo_cover review B.2.4).
    """
    on, w = _safe_width(width)
    soft = w * jax.nn.softplus(x / w) - w * jax.nn.softplus((x - 1.0) / w)
    return jnp.where(on, soft, jnp.clip(x, 0.0, 1.0))
