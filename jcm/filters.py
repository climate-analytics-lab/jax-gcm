"""Gridpoint tracer filters applied at the dynamics→physics boundary.

These are *dycore-side* operations: a dynamical core projects its native state
to the gridpoint :class:`~jcm.physics_interface.PhysicsState` the physics
consume, and may clean the tracers as it does so. Spectral cores in particular
project a sharp, near-zero tracer source with Gibbs ringing — negative
overshoots that are unphysical for any downstream term (aerosol microphysics
size/condensation, activation, radiation optics). A grid-point core has no such
problem, so the filter is opt-in per dycore (see ``tracer_filter`` on
:class:`jcm.dycore.dinosaur.dycore.DinosaurDycore`) and a no-op by default.

A tracer filter is any callable ``(tracers, dp) -> tracers`` where ``tracers``
maps name → ``(nlev, *horiz)`` gridpoint field and ``dp`` is the per-layer air
mass ``∝ Δp`` (same shape), supplied by the dycore from its own vertical
coordinate and surface pressure.

The module also holds :func:`stable_quotient`, the division the dycore's
mass fixer and :func:`mass_conserving_positivity` share: both rescale a tracer
by a ratio of two mass totals, and those totals can be arbitrarily small.
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp


@jax.custom_jvp
def stable_quotient(numerator, denominator):
    """``numerator / denominator`` with the derivative evaluated without ``denominator**-2``.

    The value is the plain quotient, bit for bit. The derivative is the true
    derivative of that quotient, not a surrogate:
    ``d(n/d) = (dn - (n/d)·dd) / d``. Reverse-mode differentiation of a bare
    ``n / d`` forms ``-n · d**-2`` instead, and in float32 ``d**-2`` is
    ``inf`` for every ``d`` below ``2**-63 ≈ 1.08e-19`` (the square underflows)
    although ``n / d`` itself is representable; a zero cotangent multiplying
    that ``inf`` is ``nan``. This form needs only ``1/d``, which is finite for
    any ``d`` a mass total can hold above the guard that keeps the
    denominator positive.

    The caller still guards the denominator against zero (a double
    ``jnp.where``); this function does not mask anything, so its value and
    derivative are those of ``n / d`` wherever ``d != 0``.

    Args:
        numerator: Array or scalar.
        denominator: Array or scalar, broadcast against ``numerator``; nonzero.

    Returns:
        ``numerator / denominator``.

    """
    return numerator / denominator


@stable_quotient.defjvp
def _stable_quotient_jvp(primals, tangents):
    numerator, denominator = primals
    d_numerator, d_denominator = tangents
    # The value comes from ``stable_quotient`` so that differentiating twice
    # meets this rule again.
    ratio = stable_quotient(numerator, denominator)
    return ratio, (d_numerator - ratio * d_denominator) / denominator


def mass_conserving_positivity(q: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """Clip ``q`` to non-negative while conserving its column-integrated mass.

    A naïve floor at zero would *add* mass, so instead clip the negatives and
    rescale the surviving positive part of each column so the column mass is
    unchanged ("hole-filling"):

        q' = max(0, q) · max(M, 0) / M_clip,
        M = Σ_k m_k q_k,   M_clip = Σ_k m_k max(0, q_k),

    with ``m_k`` the per-layer air mass (∝ Δp). Non-negative by construction and
    column-mass-conserving when ``M > 0``; a column whose mass is spuriously
    net-negative is zeroed (the only, unavoidable, non-conservation). ``q`` and
    ``m`` are ``(nlev, *horiz)``; the reduction is over the leading level axis.
    """
    q_clip = jnp.maximum(0.0, q)
    col_mass = jnp.sum(m * q, axis=0)
    col_mass_clip = jnp.sum(m * q_clip, axis=0)
    # A column with no positive mass (an empty tracer column is the ordinary
    # case) has ``col_mass_clip == 0``. The value there is 0 by the mask, but
    # reverse mode differentiates the masked quotient too, and ``0/0`` in it
    # reaches the gradient as ``nan``. The division therefore sees a benign
    # denominator in those columns (double ``where``), and
    # :func:`stable_quotient` keeps its derivative finite for a positive but
    # tiny column mass, where the bare quotient's ``denominator**-2`` overflows.
    has_mass = col_mass_clip > 0.0
    safe_mass_clip = jnp.where(has_mass, col_mass_clip, 1.0)
    scale = jnp.where(
        has_mass,
        stable_quotient(jnp.maximum(col_mass, 0.0), safe_mass_clip),
        0.0)
    return q_clip * scale[jnp.newaxis, ...]


class MassConservingPositivity:
    """Mass-conserving positivity filter for all gridpoint tracers.

    Applies :func:`mass_conserving_positivity` to every tracer field using the
    per-layer air mass ``dp`` so the rescale conserves the same column mass the
    model integrates. Parameter-free and dycore-agnostic.

    Intended use: pass an instance as ``tracer_filter`` to a spectral dynamical
    core so the gridpoint state handed to the physics has no ringing-induced
    negative tracer masses/numbers. It is a *guard*, not a cure for the deeper
    instability — it cannot restore modal mass/number consistency, which needs a
    positivity-preserving tracer transport (see issue #521).
    """

    def __call__(
        self, tracers: Mapping[str, jnp.ndarray], dp: jnp.ndarray
    ) -> dict[str, jnp.ndarray]:
        """Return ``tracers`` with each field floored mass-conservingly."""
        return {k: mass_conserving_positivity(q, dp) for k, q in tracers.items()}
