"""Tetens saturation for Betts-Miller and the JAM aerosol modules.

This module serves the schemes that follow their own references rather than
ECHAM's saturation tables:

* Betts-Miller (:mod:`jcm.physics.convection.betts_miller`) follows Isca's
  ``betts_miller.f90`` and saturates over liquid water everywhere
  (``phase="water"``).
* JAM's MAM4 microphysics RH
  (:mod:`~jcm.physics.aerosol.jam.microphysics.mam4_jax`) and its
  ice-nucleation ice supersaturation
  (:mod:`~jcm.physics.aerosol.jam.ice_nucleation.ice_term`).

ECHAM physics (Tiedtke-Nordeng convection, the cloud cover, the 1M and 2M
cloud schemes, the vertical diffusion, the surface tiles) does **not** use this
module: it takes Sonntag (1990), the formula ECHAM's lookup tables hold, from
:mod:`jcm.physics.thermodynamics`.

The formula is the Tetens/Magnus form

    es(T) = c1es · exp(c3 · (T − tmelt) / (T − c4))

with ``c1es = 610.78 Pa`` and, over liquid water, ``c3les = 17.269`` /
``c4les = 35.86 K``; over ice, ``c3ies = 21.875`` / ``c4ies = 7.66 K`` — the
constants of ECHAM's ``mo_physical_constants.f90`` (l.172-177), which ECHAM
itself uses only to invert the 2 m dew point in its surface post-processing.
``phase="auto"`` saturates over water at ``T >= tmelt`` and over ice below.

The saturation specific humidity is ``eps·es/(p − (1 − eps)·es)`` with
``c.eps`` (0.622), the denominator guarded against non-positivity and the
result capped at 0.5.

These functions are intentionally *not* ``@jit``-ed: they always run inside a
caller's compiled graph. ``phase`` is a static Python string resolved at trace
time.
"""

import jax.numpy as jnp

import jcm.constants as c

#: Saturation vapour pressure at the melting point [Pa] (``c1es``).
ES0 = 610.78
_C3LES = 17.269   # over liquid water
_C4LES = 35.86    # K
_C3IES = 21.875   # over ice
_C4IES = 7.66     # K

# Math-safety temperature clip: the exponent denominator (T − c4) vanishes near
# 8-36 K, far below any physical temperature. The clip also zeroes the
# temperature gradient outside these bounds.
_T_MIN = 50.0
_T_MAX = 500.0

# Cap on qs: at very high T / low p the denominator p − (1−eps)·es shrinks
# toward zero; 0.5 is far above any physical qs.
_QS_MAX = 0.5


def _validate_phase(phase: str) -> None:
    if phase not in ("auto", "water", "ice"):
        raise ValueError(
            f"phase must be 'auto', 'water' or 'ice', got {phase!r}")


def saturation_vapor_pressure(temperature: jnp.ndarray,
                              phase: str = "auto") -> jnp.ndarray:
    """Tetens saturation vapour pressure (Pa).

    Args:
        temperature: Temperature (K), clipped to [50, 500] K.
        phase: ``"auto"`` (water at/above ``tmelt``, ice below), ``"water"``
            or ``"ice"``.

    """
    _validate_phase(phase)
    temperature = jnp.clip(temperature, _T_MIN, _T_MAX)
    tc = temperature - c.tmelt
    if phase == "water":
        return ES0 * jnp.exp(_C3LES * tc / (temperature - _C4LES))
    if phase == "ice":
        return ES0 * jnp.exp(_C3IES * tc / (temperature - _C4IES))
    es_water = ES0 * jnp.exp(_C3LES * tc / (temperature - _C4LES))
    es_ice = ES0 * jnp.exp(_C3IES * tc / (temperature - _C4IES))
    return jnp.where(temperature >= c.tmelt, es_water, es_ice)


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 phase: str = "auto",
                                 clip: tuple[float, float] | None = None) -> jnp.ndarray:
    """Tetens saturation specific humidity [kg/kg].

    Args:
        temperature: Temperature (K).
        pressure: Pressure (Pa).
        phase: Saturation surface, see :func:`saturation_vapor_pressure`.
        clip: Optional ``(lo, hi)`` bound on the result (applied on top of
            the 0.5 cap).

    """
    es = saturation_vapor_pressure(temperature, phase=phase)
    denom = jnp.maximum(pressure - (1.0 - c.eps) * es, c.epsilon)
    qs = jnp.minimum(c.eps * es / denom, _QS_MAX)
    if clip is not None:
        qs = jnp.clip(qs, clip[0], clip[1])
    return qs
