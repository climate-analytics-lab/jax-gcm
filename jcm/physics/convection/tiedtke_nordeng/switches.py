"""ECHAM's convective decisions: exact values, surrogate derivatives.

The Tiedtke-Nordeng scheme decides, at every call, whether a column convects
(``cumastr``'s ``zlo1`` gate on the cloud-base moisture budget), which type of
plume it carries (the moisture-convergence test), how far the plume rises (the
ascent test at each interface) and where it rains (the ``zdnoprc`` onset
depth). Each decision compares a continuous quantity with a threshold inside
the range the model visits every step, so each is a step function: its
reference derivative is zero with respect to the quantity it switches on.

Every decision here keeps ECHAM's value exactly and carries the derivative of
a named logistic surrogate through
:func:`jcm.physics.surrogate_gradient.with_surrogate_gradient` (see
``docs/source/design/surrogate_gradients.md``). Each helper returns 1.0 or 0.0
in the value. The widths are static (``pytree_node=False`` fields of
:class:`~.types.ConvectionParameters`); a width of zero selects the reference
derivative, which the helpers then return by calling the exact function
directly.

This module is a leaf: it imports nothing from the scheme, so the trigger
(``tiedtke_nordeng.py``) and the ascent (``updraft.py``) share it.
"""

import jax
import jax.numpy as jnp

from jcm.physics.surrogate_gradient import with_surrogate_gradient


def _step(distance, inclusive):
    """1.0 where ``distance > 0`` (``>= 0`` if ``inclusive``), else 0.0."""
    on = distance >= 0.0 if inclusive else distance > 0.0
    return jnp.where(on, jnp.ones_like(distance), jnp.zeros_like(distance))


def threshold_pair(width, inclusive=False):
    """``(exact, surrogate)`` of a 0/1 switch on a signed distance.

    ``exact(d)`` is 1 past the threshold (``d > 0``, or ``d >= 0`` if
    ``inclusive``); ``surrogate(d) = sigmoid(d/width)``, which differs from
    it by less than ``sigmoid(-n)`` beyond ``n`` widths of the threshold.
    """
    def exact(d):
        return _step(d, inclusive)

    def surrogate(d):
        return jax.nn.sigmoid(d / width)

    return exact, surrogate


def threshold_switch(distance, width, inclusive=False):
    """ECHAM's 0/1 switch on ``distance``, with the logistic's derivative.

    See :func:`threshold_pair`. ``width = 0`` returns the step with its
    reference (zero) derivative.
    """
    exact, surrogate = threshold_pair(width, inclusive)
    if width == 0:
        return exact(distance)
    return with_surrogate_gradient(exact, surrogate)(distance)


def rescaled_sigmoid(x, width):
    """Smooth gate that is exactly 0 for ``x <= 0`` and tends to 1 above.

    ``(sigmoid((x - w)/w) - sigmoid(-1))/(1 - sigmoid(-1))``, clipped at zero:
    0 at and below zero, 0.32 at one width, 0.9998 at ten.
    """
    s0 = jax.nn.sigmoid(-1.0)
    return jnp.maximum(
        (jax.nn.sigmoid((x - width) / width) - s0) / (1.0 - s0), 0.0)


#: ``cuasc``'s mass-flux floor of the ascent test: the plume continues only
#: while ``pmfu >= 0.01·pmfub`` (mo_cuascent.f90:450).
ASCENT_MIN_FLUX_FRACTION = 0.01


def ascent_test_pair(condensate_width, buoyancy_width, mass_flux_width):
    """``(exact, surrogate)`` of ``cuasc``'s ascent test at one interface.

    Both take ``(cond, zbuo, mfu, mfub)``: the vapour the level's saturation
    adjustment condensed [kg/kg] (positive exactly where ECHAM's
    ``pqu < zqold`` holds), ECHAM's buoyancy ``zbuo`` [K] (the plume's
    condensate-loaded virtual temperature less the half-level environment's,
    plus ``zlift`` where the interface below is still ``klab == 1``), the
    plume's mass flux there and the cloud-base flux [kg/m²/s].

    ``exact`` is ECHAM's conjunction (mo_cuascent.f90:436-451)
    ``pqu < zqold .AND. zbuo > 0 .AND. pmfu >= 0.01·pmfub`` as 1.0/0.0.
    ``surrogate`` replaces each comparison by a smooth gate and multiplies
    them: :func:`rescaled_sigmoid` of the condensate (exactly zero where
    nothing condenses, so a plume that stops condensing carries no
    derivative through this factor), a logistic of ``zbuo/buoyancy_width``
    and a logistic of ``(mfu/mfub - 0.01)/mass_flux_width``. A width of zero
    replaces that factor by its exact step.
    """
    def exact(cond, zbuo, mfu, mfub):
        passed = ((cond > 0.0) & (zbuo > 0.0)
                  & (mfu >= ASCENT_MIN_FLUX_FRACTION * mfub))
        return jnp.where(passed, jnp.ones_like(zbuo), jnp.zeros_like(zbuo))

    def surrogate(cond, zbuo, mfu, mfub):
        ratio = mfu / jnp.maximum(mfub, 1.0e-10)
        f_cond = (_step(cond, False) if condensate_width == 0
                  else rescaled_sigmoid(cond, condensate_width))
        f_buoy = (_step(zbuo, False) if buoyancy_width == 0
                  else jax.nn.sigmoid(zbuo / buoyancy_width))
        f_flux = (_step(ratio - ASCENT_MIN_FLUX_FRACTION, True)
                  if mass_flux_width == 0
                  else jax.nn.sigmoid(
                      (ratio - ASCENT_MIN_FLUX_FRACTION) / mass_flux_width))
        return f_cond * f_buoy * f_flux

    return exact, surrogate


def ascent_test(cond, zbuo, mfu, mfub, condensate_width, buoyancy_width,
                mass_flux_width):
    """ECHAM's ascent test as 1.0/0.0, with the surrogate's derivative.

    See :func:`ascent_test_pair`. All three widths zero returns the exact
    test with its reference (zero) derivative.
    """
    exact, surrogate = ascent_test_pair(
        condensate_width, buoyancy_width, mass_flux_width)
    if condensate_width == 0 and buoyancy_width == 0 and mass_flux_width == 0:
        return exact(cond, zbuo, mfu, mfub)
    dtype = jnp.result_type(zbuo)
    args = [jnp.asarray(a, dtype=dtype) for a in (cond, zbuo, mfu, mfub)]
    return with_surrogate_gradient(exact, surrogate)(*args)


def relative_threshold_pair(width):
    """``(exact, surrogate)`` of ``value > floor`` for a positive ``floor``.

    Both take ``(value, floor)``. ``exact`` is 1.0 where ``value > floor``;
    ``surrogate`` is ``sigmoid((value - floor)/(width·floor))``, a logistic
    whose width is the fraction ``width`` of the floor itself — the scale on
    which ``cumastr``'s ``zdqmin = max(0.01·pqenh, 1e-10)`` varies with the
    cloud-base humidity.
    """
    def exact(value, floor):
        return _step(value - floor, False)

    def surrogate(value, floor):
        return jax.nn.sigmoid((value - floor) / (width * floor))

    return exact, surrogate


def relative_threshold_switch(value, floor, width):
    """ECHAM's ``value > floor`` as 1.0/0.0, with the surrogate's derivative.

    See :func:`relative_threshold_pair`; ``width = 0`` keeps the reference
    derivative.
    """
    exact, surrogate = relative_threshold_pair(width)
    if width == 0:
        return exact(value, floor)
    dtype = jnp.result_type(value)
    return with_surrogate_gradient(exact, surrogate)(
        jnp.asarray(value, dtype=dtype), jnp.asarray(floor, dtype=dtype))
