"""Shared saturation thermodynamics (Tetens) for the convection schemes.

Centralises the Tetens saturation-vapour-pressure coefficients and the
saturation specific-humidity / mixing-ratio formulas (and the analytic
``dqs/dT`` used by the Newton adjustment steps) that were previously duplicated
across ``tiedtke_nordeng``, ``updraft``, ``adjustment`` and ``betts_miller``.

Scheme-specific choices are parameters rather than separate copies:

* ``phase`` selects the saturation surface — ``"auto"`` (water above the melting
  point, ice below; the ECHAM/Tiedtke convention), ``"water"`` (Betts-Miller,
  following Isca), or ``"ice"``.
* ``clip`` optionally bounds the returned specific humidity (Tiedtke clips to
  ``[0, 0.5]`` via :func:`saturation_mixing_ratio`).

Physical constants (``eps``, ``tmelt``) are read by attribute access from
:mod:`jcm.constants`, so a runtime ``set_constants`` override is honoured.
"""

import jax.numpy as jnp

import jcm.constants as c

# Tetens coefficients: es = ES0 * exp(A * tc / (tc + B)), with tc = T - tmelt.
# Water is used above the melting point, ice below. The ice ``A`` (35.86) is
# this package's historical value; what matters for the Newton steps is that the
# saturation target and its derivative use the *same* coefficients.
ES0 = 610.78          # Pa (saturation vapour pressure at the melting point)
A_WATER = 17.27
B_WATER = 237.3
A_ICE = 35.86
B_ICE = 265.5

# Math-safety temperature clip: the Tetens denominators (tc + B) vanish near
# 8-36 K, far below any physical temperature. A loose bound avoids NaNs without
# masking genuine upstream bugs.
_T_MIN = 50.0
_T_MAX = 500.0


def saturation_vapor_pressure(temperature: jnp.ndarray,
                              phase: str = "auto") -> jnp.ndarray:
    """Tetens saturation vapour pressure (Pa).

    Args:
        temperature: Temperature (K).
        phase: ``"auto"`` (water above ``tmelt``, ice below), ``"water"`` or
            ``"ice"``.

    """
    temperature = jnp.clip(temperature, _T_MIN, _T_MAX)
    tc = temperature - c.tmelt
    es_water = ES0 * jnp.exp(A_WATER * tc / (tc + B_WATER))
    if phase == "water":
        return es_water
    es_ice = ES0 * jnp.exp(A_ICE * tc / (tc + B_ICE))
    if phase == "ice":
        return es_ice
    return jnp.where(temperature > c.tmelt, es_water, es_ice)


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 phase: str = "auto",
                                 clip: tuple[float, float] | None = None) -> jnp.ndarray:
    """Saturation specific humidity [kg/kg].

    Args:
        temperature: Temperature (K).
        pressure: Pressure (Pa).
        phase: Saturation surface, see :func:`saturation_vapor_pressure`.
        clip: Optional ``(lo, hi)`` bound on the result.

    """
    es = saturation_vapor_pressure(temperature, phase=phase)
    # Cap es below the pressure so the denominator stays positive at low p.
    es = jnp.minimum(es, 0.99 * jnp.maximum(pressure, 1.0))
    qs = c.eps * es / jnp.maximum(pressure - es * (1.0 - c.eps), 1.0)
    if clip is not None:
        qs = jnp.clip(qs, clip[0], clip[1])
    return qs


def saturation_mixing_ratio(pressure: jnp.ndarray,
                            temperature: jnp.ndarray,
                            phase: str = "auto") -> jnp.ndarray:
    """Saturation specific humidity [kg/kg] clipped to ``[0, 0.5]``.

    The argument order ``(pressure, temperature)`` and the clip match the
    historical Tiedtke-Nordeng helper this replaces.
    """
    return saturation_specific_humidity(temperature, pressure, phase=phase,
                                        clip=(0.0, 0.5))


def saturation_specific_humidity_and_derivative(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    phase: str = "auto",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(qs, dqs/dT)`` analytically for the cuadjtq / updraft Newton steps.

    Computing the derivative in closed form keeps the Newton iteration
    bit-reproducible under JIT without differentiating through a lookup table.
    """
    es = saturation_vapor_pressure(temperature, phase=phase)
    p_safe = jnp.maximum(pressure, 1.0)
    es_safe = jnp.minimum(es, 0.99 * p_safe)
    denom = jnp.maximum(p_safe - es_safe * (1.0 - c.eps), 1.0)
    qs = c.eps * es_safe / denom

    tc = temperature - c.tmelt
    des_dT_water = es * A_WATER * B_WATER / jnp.maximum((tc + B_WATER) ** 2, 1e-3)
    des_dT_ice = es * A_ICE * B_ICE / jnp.maximum((tc + B_ICE) ** 2, 1e-3)
    des_dT = jnp.where(temperature > c.tmelt, des_dT_water, des_dT_ice)
    dqs_dT = c.eps * p_safe * des_dT / denom ** 2
    return qs, dqs_dT
