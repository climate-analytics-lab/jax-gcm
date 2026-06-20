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
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp


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
    scale = jnp.where(col_mass_clip > 0.0,
                      jnp.maximum(col_mass, 0.0) / col_mass_clip, 0.0)
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
