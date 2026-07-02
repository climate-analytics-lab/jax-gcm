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

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
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
        # at tmelt, so L flips with it.
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


@jax.jit
def saturation_adjustment(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Saturation adjustment with liquid / ice partitioning.

    Wraps :func:`cuadjtq` (``kcall=1``, condensation-only) with a
    temperature-based liquid / ice split for the resulting condensate.
    The split mirrors what ECHAM's cloud scheme does outside cuadjtq.

    Args:
        temperature: Temperature after convective tendencies [K].
        specific_humidity: Specific humidity after tendencies [kg/kg].
        pressure: Pressure [Pa].
        cloud_water: Cloud liquid water before adjustment [kg/kg].
        cloud_ice: Cloud ice before adjustment [kg/kg].

    Returns:
        Adjusted ``(T, q, qc, qi)``.

    """
    # The proper Newton step uses specific humidity directly (matches
    # what cuadjtq does); the previous mixing-ratio detour was unnecessary.
    t_adj, q_adj, condensate = cuadjtq(
        temperature, specific_humidity, pressure, kcall=1, refine=True,
    )

    # Liquid / ice split — mirrors what ECHAM cuasc does outside cuadjtq.
    t_freeze = c.tmelt
    t_ice = c.tmelt - 23.0
    frac_liquid = jnp.clip((t_adj - t_ice) / (t_freeze - t_ice), 0, 1)
    frac_ice = 1.0 - frac_liquid

    # Add condensate to existing cloud water/ice. The latent heat
    # adjustment in cuadjtq used L_water; correct for the ice-fraction
    # difference (L_sub - L_water) so the ice condensate releases the
    # full sublimation latent heat.
    qc_adj = cloud_water + condensate * frac_liquid
    qi_adj = cloud_ice + condensate * frac_ice
    t_adj = t_adj + condensate * frac_ice * (c.alhs - c.alhc) / c.cpd

    # Belt-and-braces clip to non-negative (cuadjtq guarantees this for
    # ``kcall=1`` but downstream consumers expect it from the wrapper).
    q_adj = jnp.maximum(q_adj, 0.0)
    qc_adj = jnp.maximum(qc_adj, 0.0)
    qi_adj = jnp.maximum(qi_adj, 0.0)

    return t_adj, q_adj, qc_adj, qi_adj


def energy_conservation_check(
    temperature_old: jnp.ndarray,
    specific_humidity_old: jnp.ndarray,
    cloud_water_old: jnp.ndarray,
    cloud_ice_old: jnp.ndarray,
    temperature_new: jnp.ndarray,
    specific_humidity_new: jnp.ndarray,
    cloud_water_new: jnp.ndarray,
    cloud_ice_new: jnp.ndarray,
    precipitation: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """Check energy conservation in convective adjustment
    
    Args:
        *_old: State before adjustment
        *_new: State after adjustment
        precipitation: Precipitation rate (kg/m²/s)
        dt: Time step (s)
        
    Returns:
        Energy imbalance (W/m²)

    """
    # Sensible heat change
    dT = temperature_new - temperature_old
    sensible = c.cpd * dT / dt
    
    # Latent heat changes
    dq = specific_humidity_new - specific_humidity_old
    dqc = cloud_water_new - cloud_water_old
    dqi = cloud_ice_new - cloud_ice_old
    
    # Latent heat (vapor uses L at current temperature)
    t_avg = 0.5 * (temperature_old + temperature_new)
    lv = c.alhc + (c.alhs - c.alhc) * jnp.clip((c.tmelt - t_avg) / 23.0, 0, 1)
    
    latent_vapor = lv * dq / dt
    latent_liquid = c.alhc * dqc / dt
    latent_ice = c.alhs * dqi / dt
    
    # Precipitation removes energy
    # Assume precipitation temperature is cloud temperature
    precip_energy = precipitation * c.cpd * (t_avg - c.tmelt)
    
    # Total energy change
    total_energy = sensible + latent_vapor + latent_liquid + latent_ice + precip_energy
    
    return total_energy


@jax.jit
def convective_adjustment(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    convective_tendency_t: jnp.ndarray,
    convective_tendency_q: jnp.ndarray,
    convective_tendency_qc: jnp.ndarray,
    convective_tendency_qi: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply convective tendencies and perform saturation adjustment
    
    This is the main interface for applying convection results to the
    model state, ensuring thermodynamic consistency.
    
    Args:
        temperature: Temperature before convection (K)
        specific_humidity: Specific humidity before (kg/kg)
        pressure: Pressure (Pa)
        cloud_water: Cloud water before (kg/kg)
        cloud_ice: Cloud ice before (kg/kg)
        convective_tendency_*: Tendencies from convection scheme
        dt: Time step (s)
        
    Returns:
        Tuple of adjusted (temperature, specific_humidity, cloud_water, cloud_ice)

    """
    # Apply convective tendencies
    t_conv = temperature + convective_tendency_t * dt
    q_conv = specific_humidity + convective_tendency_q * dt
    qc_conv = cloud_water + convective_tendency_qc * dt
    qi_conv = cloud_ice + convective_tendency_qi * dt
    
    # Ensure positive values before adjustment
    q_conv = jnp.maximum(q_conv, 0.0)
    qc_conv = jnp.maximum(qc_conv, 0.0)
    qi_conv = jnp.maximum(qi_conv, 0.0)
    
    # Perform saturation adjustment
    t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
        t_conv, q_conv, pressure, qc_conv, qi_conv
    )
    
    return t_adj, q_adj, qc_adj, qi_adj
