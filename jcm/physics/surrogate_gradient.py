"""Reference-exact values with the derivatives of a smooth surrogate.

A reference formulation (ECHAM, SPEEDY, ...) is full of clips, thresholds and
switches that sit inside the range the model actually visits: a cloud cover
clipped to ``[0, 1]``, a phase chosen by a temperature threshold, a power law
whose slope is infinite at zero. Replacing such a function by a smooth one
changes the forward model, and with it the climate that is validated and
tuned against the reference. Keeping it as written leaves automatic
differentiation with a derivative that is identically zero on a plateau,
infinite at a singular point, or blind to a switch.

:func:`with_surrogate_gradient` separates the two. The value is the reference
function's, bit for bit. The derivatives are those of a smooth function named
at the call site, taken by automatic differentiation of that function. See
``docs/source/design/surrogate_gradients.md`` for when this is the right tool
and when it is not.
"""

from typing import Callable

import jax

__all__ = ["with_surrogate_gradient"]


def with_surrogate_gradient(exact: Callable, surrogate: Callable) -> Callable:
    """Return ``f`` with ``exact``'s value and ``surrogate``'s derivatives.

    ``f(*args)`` evaluates ``exact(*args)`` and nothing else, so the forward
    model is the reference formulation bit for bit. Under ``jax.jvp``,
    ``jax.grad`` and every transformation built on them, the tangent of ``f``
    is the tangent of ``surrogate`` at the same arguments, obtained by
    differentiating ``surrogate`` itself. The derivative is therefore the
    exact derivative of a function that can be read, plotted and tested, not a
    hand-written slope. Forward and reverse mode are adjoint by construction,
    because the reverse rule is the transpose of the forward one.

    Rules for the two functions:

    - Both take the same positional arguments and return outputs of the same
      structure, shape and dtype.
    - Every quantity a caller may differentiate with respect to must be an
      argument. A traced value reached through a closure is invisible to the
      derivative rule.
    - A smoothing width is configuration of the derivative, not a physical
      parameter: the value does not depend on it, so a gradient with respect
      to it means nothing. Close over it as a static Python number (a
      ``pytree_node=False`` field of the scheme's parameters). A width of
      zero means "use the true derivative", which the caller selects by
      calling ``exact`` directly instead of this wrapper.
    - Integer and boolean arguments (masks, level indices) are allowed. They
      carry no tangent.

    Args:
        exact: The reference formulation. Defines the value.
        surrogate: A smooth function close to ``exact``. Defines every
            derivative.

    Returns:
        A function of the same arguments as ``exact``.

    """

    @jax.custom_jvp
    def f(*args):
        return exact(*args)

    @f.defjvp
    def f_jvp(primals, tangents):
        # The value comes from ``f`` rather than ``exact`` so that a second
        # differentiation meets this rule again instead of differentiating
        # the reference function's kinks.
        primal_out = f(*primals)
        _, tangent_out = jax.jvp(surrogate, primals, tangents)
        return primal_out, tangent_out

    return f
