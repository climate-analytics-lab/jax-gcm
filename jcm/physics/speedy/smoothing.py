"""Smooth surrogates for the hard branches in SPEEDY physics.

The SPEEDY schemes gate fluxes and diagnostics with hard comparisons
(``drh > drh0``, ``clip(x, 0, 1)``, ``max(x, 0)``) and take a square root
of the precipitation rate in the cloud cover. Each is part of the reference
formulation, and the model's value stays exactly that. What is useless for
gradient work is their derivative: zero on a plateau or on either side of a
switch, unbounded at the square root's corner.

Two families of functions live here.

- ``smooth_gate`` / ``smooth_pos`` / ``smooth_min`` / ``smooth_max`` /
  ``smooth_clip01`` are the smooth functions themselves: sigmoid gates,
  softplus hinges, hyperbolic min/max and a softplus-pair clip of a
  caller-chosen half-width. At ``width = 0`` each is exactly the hard
  operation (guarded with the double-where pattern so the width-0 branch
  cannot leak a division-by-zero cotangent; see JAX_gotchas.md).
- ``surrogate_gate`` / ``surrogate_pos`` / ``surrogate_min`` /
  ``surrogate_max`` / ``surrogate_clip01`` / ``surrogate_sqrt`` are what the
  schemes call. Each returns the hard operation's value, bit for bit, and
  the derivatives of the matching smooth function at the given width -- the
  construction of :func:`jcm.physics.surrogate_gradient.with_surrogate_gradient`
  (``docs/source/design/surrogate_gradients.md``). A width of zero gives the
  reference derivative; the forward model never depends on the width.

The width is a scale in the units of the gated variable (an RH fraction, an
energy in J/kg, a humidity in g/kg): roughly the range over which the
derivative sees the switch as a ramp. It is a field of the scheme's
parameters, and so a pytree leaf, but it only shapes the derivative: every
surrogate drops its tangent and holds it fixed at higher orders, so any
derivative with respect to it is zero, as it must be for a quantity the
value does not depend on.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _safe_width(width):
    """Width guarded for use as a divisor when it may be exactly zero.

    A width so small that its square underflows (below ``sqrt(tiny)`` of its
    dtype, ~1e-19 in float32) counts as zero: the hyperbolic min/max would
    otherwise take ``sqrt(0)`` at the corner, whose derivative is NaN.
    """
    width = jnp.asarray(width)
    dtype = width.dtype if jnp.issubdtype(width.dtype, jnp.inexact) else jnp.float32
    on = width > jnp.sqrt(jnp.finfo(dtype).tiny)
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
    the edges, so gradients survive saturation. The same construction is
    the surrogate that defines the derivative of the ECHAM cloud cover's
    clip (``sundqvist._cover_surrogate``, width ``smooth_b0``).
    """
    on, w = _safe_width(width)
    soft = w * jax.nn.softplus(x / w) - w * jax.nn.softplus((x - 1.0) / w)
    return jnp.where(on, soft, jnp.clip(x, 0.0, 1.0))


def _surrogate(smooth):
    """Wrap ``smooth(*operands, width)`` as hard value + smooth derivative.

    The value is ``smooth(*operands, 0.0)`` -- the hard operation, exactly --
    and the derivatives are those of ``smooth(*operands, width)``, taken by
    differentiating it, with the width held fixed: the construction of
    :func:`jcm.physics.surrogate_gradient.with_surrogate_gradient`. The width
    is a primal argument of the ``custom_jvp`` whose tangent the rule
    ignores, rather than a closure: it is a traced parameter leaf, and a
    ``custom_jvp`` re-traces its rule outside the jit trace that produced a
    closed-over tracer (``nnx.grad`` around a jitted scheme), which leaks it.
    """

    @jax.custom_jvp
    def surrogate(*operands_and_width):
        *operands, _ = operands_and_width
        return smooth(*operands, 0.0)

    @surrogate.defjvp
    def surrogate_jvp(primals, tangents):
        *operands, width = primals
        *operand_tangents, _ = tangents
        # The width is held fixed for every order of differentiation: its
        # tangent is dropped here, and stop_gradient keeps a second
        # differentiation of this rule from reaching it either, so mixed
        # derivatives with respect to the width are zero both ways round.
        fixed_width = jax.lax.stop_gradient(width)
        _, tangent_out = jax.jvp(
            lambda *xs: smooth(*xs, fixed_width), tuple(operands), tuple(operand_tangents))
        # The value from ``surrogate`` itself, so a second differentiation
        # meets this rule again rather than the hard operation's kinks.
        primal_out = surrogate(*primals)
        # A width of another precision (a float64 width read from a file or
        # produced by an optimiser, float32 operands) must not change the
        # tangent's dtype: custom_jvp requires it to match the primal's.
        return primal_out, tangent_out.astype(primal_out.dtype)

    surrogate.__name__ = "surrogate_" + smooth.__name__.removeprefix("smooth_")
    surrogate.__doc__ = (
        f"``{smooth.__name__}``'s hard value with its derivatives at ``width``."
    )
    return surrogate


surrogate_gate = _surrogate(smooth_gate)
surrogate_pos = _surrogate(smooth_pos)
surrogate_min = _surrogate(smooth_min)
surrogate_max = _surrogate(smooth_max)
surrogate_clip01 = _surrogate(smooth_clip01)


def _floored_sqrt(x, floor, offset):
    """``sqrt(max(x, floor))``, or with ``offset > 0`` the regularised root.

    The regularised form is ``sqrt(smooth_pos(x, sqrt(offset)) + offset)``.
    ``offset = 0`` is exactly the floored root (double-where guarded).
    """
    on, safe = _safe_width(offset)
    regularised = jnp.sqrt(smooth_pos(x, jnp.sqrt(safe)) + safe)
    return jnp.where(on, regularised, jnp.sqrt(jnp.maximum(x, floor)))


_surrogate_floored_sqrt = _surrogate(_floored_sqrt)


def surrogate_sqrt(x, floor, offset):
    """``sqrt(max(x, floor))`` with the derivative of ``sqrt(x+ + offset)``.

    The value is the reference's floored square root. Its slope,
    ``1/(2 sqrt(x))``, is unbounded as ``x`` goes to the floor -- a singular
    point the model visits wherever it barely rains (the cloud cover's
    precipitation term). The derivative is instead that of
    ``sqrt(smooth_pos(x, sqrt(offset)) + offset)``, bounded by
    ``1/(2 sqrt(offset))`` and within ``offset/x`` of the reference slope
    where ``x`` is large against ``offset``. ``offset = 0`` gives the
    reference derivative.
    """
    return _surrogate_floored_sqrt(x, floor, offset)

