"""Convective adjustment for Tiedtke-Nordeng scheme.

Faithful port of ECHAM ``mo_cuadjust.f90`` ``cuadjtq``: a linearised
Newton-Raphson saturation adjustment that handles the temperature-q_sat
feedback in a single (or two) iteration with proper convergence
behaviour.

The Newton step is::

    Δq = (q - q_sat(T)) / (1 + (L/cp) · dq_sat/dT)

which is the linearisation around T of the implicit equation
``q - Δq = q_sat(T + (L/cp)·Δq)``. The denominator damps the step by
the warming feedback (a hotter parcel can hold more vapour, so less
condensation is needed than ``q - q_sat(T)`` would suggest). Without
that denominator a simple ``cond = max(q - q_sat, 0)`` over-condenses,
over-warms, and either oscillates or needs many iterations to settle.

ECHAM's ``cuadjtq`` runs the Newton step once with a sign clip
(``kcall``-dependent), then optionally a second refinement pass on
columns that actually condensed. We expose the same three modes so the
existing call sites (cubase / cuasc / cudlfs) can pick the right one:

* ``kcall=0`` — environmental q_sat (cuini): both signs allowed.
* ``kcall=1`` — condensation only (cubase, cuasc): ``Δq >= 0``.
* ``kcall=2`` — evaporation only (cudlfs, cuddraf): ``Δq <= 0``.

"""

import jax.numpy as jnp
from jax import lax
from typing import Tuple

import jcm.constants as c
# Analytic (qs, dqs/dT) for the cuadjtq Newton step; shared with the updraft
# module via jcm.physics.convection.saturation.
from jcm.physics.convection.saturation import (
    saturation_specific_humidity_and_derivative as _qsat_and_dqsat_dt,
)


def cuadjtq(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    kcall: int = 1,
    refine: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ECHAM-style linearised saturation adjustment.

    Direct port of ``mo_cuadjust.f90`` ``cuadjtq``. Returns
    ``(T_adj, q_adj, condensate)`` where ``condensate >= 0`` for
    ``kcall=1`` (condensation in updrafts) and ``condensate <= 0`` for
    ``kcall=2`` (evaporation in downdrafts). The caller decides whether
    to allocate the condensate to liquid, ice, or precipitation.

    Args:
        temperature: Temperature [K].
        specific_humidity: Vapour mixing ratio [kg/kg].
        pressure: Pressure [Pa].
        kcall: 0 = both directions (cuini env q_sat), 1 = condensation
            only (cubase, cuasc), 2 = evaporation only (cudlfs).
        refine: Run a second Newton iteration on columns that condensed
            (matches ECHAM's two-pass behaviour). Disable for the rare
            cases where one pass is enough and you want bit-exact
            equivalence with cuadjtq's first-pass output.

    Returns:
        ``(T_adj, q_adj, condensate)``: adjusted temperature and vapour
        with ``condensate = q - q_adj`` reflecting the moist exchange.

    """
    def _newton(T, q):
        # Phase-consistent latent heat (ECHAM cuadjtq pairs the ice
        # saturation table with L_s below the melting point — review
        # finding 2.7; a fixed L_v under-releases mixed-phase latent heat
        # by ~13 %). The es switch in the shared saturation module flips
        # at tmelt, so L flips with it. DRY ``cpd`` is the reference here:
        # cuadjtq reads ``L/cp`` from the ``tlucub``/``tlucuc`` tables built
        # with ``zalvdcp = alv/cpd``, ``zalsdcp = als/cpd``
        # (mo_echam_convect_tables.f90:214-215, 254-258) — unlike the
        # cumastr static-energy ledger, which uses the moist ``zcpq``.
        L_cp = jnp.where(T >= c.tmelt, c.alhc, c.alhs) / c.cpd
        qs, dqs_dT = _qsat_and_dqsat_dt(T, pressure)
        cond = (q - qs) / (1.0 + L_cp * dqs_dT)
        # Apply the kcall sign clip exactly as ECHAM does.
        cond = lax.cond(
            kcall == 1,
            lambda c: jnp.maximum(c, 0.0),
            lambda c: lax.cond(
                kcall == 2,
                lambda cc: jnp.minimum(cc, 0.0),
                lambda cc: cc,  # kcall=0: both directions
                c,
            ),
            cond,
        )
        return T + L_cp * cond, q - cond, cond

    T1, q1, cond1 = _newton(temperature, specific_humidity)
    if not refine:
        return T1, q1, cond1
    # Second iteration only fires on cells that condensed in pass 1.
    # We always run it (jit-friendly), but multiply the second-pass
    # condensate by a mask so unchanged cells stay unchanged.
    pass1_active = jnp.abs(cond1) > 0.0
    T2, q2, cond2 = _newton(T1, q1)
    cond2 = jnp.where(pass1_active, cond2, 0.0)
    T_final = jnp.where(pass1_active, T2, T1)
    q_final = jnp.where(pass1_active, q2, q1)
    return T_final, q_final, cond1 + cond2
