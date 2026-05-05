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

from jcm.constants import (
    cp, alhc, alhs, tmelt, eps
)
from .tiedtke_nordeng import (
    saturation_mixing_ratio, saturation_vapor_pressure
)


# Tetens coefficients matching ``saturation_vapor_pressure`` (water above
# 0 °C, ice below). Same as updraft.py but kept here so this module can
# stand alone.
_TETENS_A_WATER = 17.27
_TETENS_C_WATER = 237.3
_TETENS_A_ICE = 35.86
_TETENS_C_ICE = 265.5


def _qsat_and_dqsat_dt(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Saturation specific humidity and its temperature derivative.

    Closed-form derivative of ``saturation_mixing_ratio`` for the Tetens
    formulation, so the Newton step is bit-reproducible under JIT
    without relying on autodiff through a saturation lookup table.
    Mirrors what the ECHAM lookup tables ``ua/dua`` provide.
    """
    es = saturation_vapor_pressure(temperature)
    p_safe = jnp.maximum(pressure, 1.0)
    es_safe = jnp.minimum(es, 0.99 * p_safe)
    denom = jnp.maximum(p_safe - es_safe * (1.0 - eps), 1.0)
    qs = eps * es_safe / denom

    tc = temperature - tmelt
    des_dT_water = es * _TETENS_A_WATER * _TETENS_C_WATER / jnp.maximum(
        (tc + _TETENS_C_WATER) ** 2, 1e-3,
    )
    des_dT_ice = es * _TETENS_A_ICE * _TETENS_C_ICE / jnp.maximum(
        (tc + _TETENS_C_ICE) ** 2, 1e-3,
    )
    des_dT = jnp.where(temperature > tmelt, des_dT_water, des_dT_ice)
    dqs_dT = eps * p_safe * des_dT / denom ** 2
    return qs, dqs_dT


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
    L_cp = alhc / cp

    def _newton(T, q):
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
    t_freeze = tmelt
    t_ice = tmelt - 23.0
    frac_liquid = jnp.clip((t_adj - t_ice) / (t_freeze - t_ice), 0, 1)
    frac_ice = 1.0 - frac_liquid

    # Add condensate to existing cloud water/ice. The latent heat
    # adjustment in cuadjtq used L_water; correct for the ice-fraction
    # difference (L_sub - L_water) so the ice condensate releases the
    # full sublimation latent heat.
    qc_adj = cloud_water + condensate * frac_liquid
    qi_adj = cloud_ice + condensate * frac_ice
    t_adj = t_adj + condensate * frac_ice * (alhs - alhc) / cp

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
    sensible = cp * dT / dt
    
    # Latent heat changes
    dq = specific_humidity_new - specific_humidity_old
    dqc = cloud_water_new - cloud_water_old
    dqi = cloud_ice_new - cloud_ice_old
    
    # Latent heat (vapor uses L at current temperature)
    t_avg = 0.5 * (temperature_old + temperature_new)
    lv = alhc + (alhs - alhc) * jnp.clip((tmelt - t_avg) / 23.0, 0, 1)
    
    latent_vapor = lv * dq / dt
    latent_liquid = alhc * dqc / dt
    latent_ice = alhs * dqi / dt
    
    # Precipitation removes energy
    # Assume precipitation temperature is cloud temperature
    precip_energy = precipitation * cp * (t_avg - tmelt)
    
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


def test_saturation_adjustment():
    """Test the saturation adjustment"""
    # Create supersaturated conditions
    temperature = jnp.array(280.0)  # K
    pressure = jnp.array(90000.0)    # Pa
    
    # Get saturation mixing ratio
    rs = saturation_mixing_ratio(pressure, temperature)
    qs = rs / (1 + rs)  # Convert to specific humidity
    
    # Create supersaturated state (120% RH)
    specific_humidity = 1.2 * qs
    cloud_water = jnp.array(0.0)
    cloud_ice = jnp.array(0.0)
    
    # Perform adjustment
    t_adj, q_adj, qc_adj, qi_adj = saturation_adjustment(
        temperature, specific_humidity, pressure,
        cloud_water, cloud_ice
    )
    
    # Check results
    print(f"Initial T: {temperature:.2f} K, q: {specific_humidity*1000:.2f} g/kg")
    print(f"Adjusted T: {t_adj:.2f} K, q: {q_adj*1000:.2f} g/kg")
    print(f"Cloud water: {qc_adj*1000:.2f} g/kg")
    print(f"Temperature increase: {t_adj - temperature:.2f} K")
    
    # Should have condensation and warming
    assert t_adj > temperature  # Latent heat release
    assert q_adj < specific_humidity  # Vapor removed
    assert qc_adj > cloud_water  # Cloud water increased
    
    # Should be approximately saturated after adjustment
    rs_adj = saturation_mixing_ratio(pressure, t_adj)
    qs_adj = rs_adj / (1 + rs_adj)
    rh_adj = q_adj / qs_adj
    print(f"Final RH: {rh_adj*100:.1f}%")
    # The adjustment reduces supersaturation significantly
    assert 0.75 < rh_adj < 1.05  # Should be closer to saturation
    
    print("Saturation adjustment test passed!")


def test_energy_conservation():
    """Test energy conservation check"""
    # Create a simple state change
    t_old = jnp.array(280.0)
    q_old = jnp.array(0.010)
    qc_old = jnp.array(0.001)
    qi_old = jnp.array(0.0)
    
    # Warming and drying (condensation)
    t_new = jnp.array(281.0)
    q_new = jnp.array(0.008)
    qc_new = jnp.array(0.003)
    qi_new = jnp.array(0.0)
    
    precip = jnp.array(0.0)
    dt = 3600.0
    
    imbalance = energy_conservation_check(
        t_old, q_old, qc_old, qi_old,
        t_new, q_new, qc_new, qi_new,
        precip, dt
    )
    
    print(f"Energy imbalance: {imbalance:.2f} W/m²")
    
    # The imbalance should be small if energy is conserved
    # (some imbalance is expected due to approximations)
    assert jnp.abs(imbalance) < 10.0  # Less than 10 W/m²
    
    print("Energy conservation test passed!")


if __name__ == "__main__":
    test_saturation_adjustment()
    print()
    test_energy_conservation()