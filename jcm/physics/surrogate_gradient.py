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

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

__all__ = ["with_surrogate_gradient"]


def with_surrogate_gradient(
    exact: Callable, surrogate: Callable | Sequence[Callable | None],
) -> Callable:
    """Return ``f`` with ``exact``'s value and ``surrogate``'s derivatives.

    ``f(*args)`` evaluates ``exact(*args)`` and nothing else, so the forward
    model is the reference formulation bit for bit. Under ``jax.jvp``,
    ``jax.grad`` and every transformation built on them, the tangent of ``f``
    is the tangent of ``surrogate`` at the same arguments, obtained by
    differentiating ``surrogate`` itself. The derivative is therefore the
    exact derivative of a function that can be read, plotted and tested, not a
    hand-written slope. Forward and reverse mode are adjoint by construction,
    because the reverse rule is the transpose of the forward one.

    ``surrogate`` is usually one function sharing ``exact``'s whole argument
    list. Pass a SEQUENCE instead -- one entry per positional argument of
    ``exact`` -- when different arguments need different treatment: one
    argument's true derivative is singular or a plateau (it needs smoothing)
    while another's is already fine (it needs the exact derivative, which may
    still need a kink in some OTHER argument guarded away -- see
    ``jcm.physics.clouds.lohmann_2m.deposition_freezing._demott_n_aer_power``
    for a worked example: a fractional power singular in one argument but
    perfectly differentiable, away from that argument's own kink, in the
    other). Entry ``i`` computes argument ``i``'s own contribution to the
    tangent, by differentiating entry ``i`` in direction ``i`` ONLY (every
    other argument's incoming tangent is zeroed for that call) and summing
    the per-argument contributions -- ordinary multivariable differentiation,
    just allowing each partial derivative to come from its own named
    function. ``None`` for an entry means "this argument's contribution comes
    from ``exact`` itself" (its true derivative needs no smoothing). Passing
    the same function for every entry is equivalent to the single-function
    form above (summing the per-argument directional derivatives of one
    function recovers its ordinary one-call JVP, by linearity), which is why
    that form is kept as the common, cheaper, single-``jax.jvp`` path rather
    than being special-cased out of this one.

    Rules for the functions:

    - All take the same positional arguments and return outputs of the same
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
        surrogate: A smooth function close to ``exact`` (defines every
            derivative), or a sequence of one such function per positional
            argument of ``exact`` (``None`` for an argument whose own exact
            derivative needs no smoothing).

    Returns:
        A function of the same arguments as ``exact``.

    """
    if isinstance(surrogate, (list, tuple)):
        return _with_per_argument_surrogate_gradient(exact, surrogate)

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


def _with_per_argument_surrogate_gradient(
    exact: Callable, surrogates: Sequence[Callable | None],
) -> Callable:
    surrogates = tuple(s if s is not None else exact for s in surrogates)

    @jax.custom_jvp
    def f(*args):
        return exact(*args)

    @f.defjvp
    def f_jvp(primals, tangents):
        primal_out = f(*primals)
        tangent_out = None
        for i, (surrogate_i, tangent_i) in enumerate(zip(surrogates, tangents)):
            # Zero every OTHER argument's tangent for this call, so the
            # resulting directional derivative is argument i's contribution
            # alone -- whatever surrogate_i's own sensitivity to the other
            # arguments happens to be plays no part, by construction. Any
            # argument (not just ``n_aer_coarse``-style scalars) may be a
            # pytree -- e.g. a dict-valued field -- so the zero tangent and
            # the running sum are built leaf by leaf, exactly as a single
            # ``jax.jvp`` call already does internally for the one-surrogate
            # path above.
            only_i = tuple(
                tangent_i if j == i else jax.tree_util.tree_map(jnp.zeros_like, other)
                for j, other in enumerate(tangents)
            )
            _, contribution = jax.jvp(surrogate_i, primals, only_i)
            tangent_out = (contribution if tangent_out is None
                            else jax.tree_util.tree_map(
                                lambda a, b: a + b, tangent_out, contribution))
        return primal_out, tangent_out

    return f
