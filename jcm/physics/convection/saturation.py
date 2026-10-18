"""Tetens saturation for Betts-Miller and two JAM aerosol modules.

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
module for saturation itself: it takes Sonntag (1990), the formula ECHAM's
lookup tables hold, from :mod:`jcm.physics.thermodynamics`. The one
documented exception is :func:`dewpoint_temperature`, used by
``jcm.physics.surface.echam.turbulent_fluxes.compute_surface_diagnostics``
for the EchamSurface 2 m dew point (jax-gcm#1140) — ECHAM itself inverts its
own 2 m dew point with exactly these Tetens constants (see below), so that
one diagnostic is the anticipated use of this module from ECHAM physics.

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


def dewpoint_temperature(specific_humidity: jnp.ndarray,
                         pressure: jnp.ndarray,
                         phase: str = "auto") -> jnp.ndarray:
    """Dew point: the exact analytic inverse of :func:`saturation_specific_humidity`.

    Solves ``qs(T_d, p) = q`` for ``T_d`` in closed form — invert
    ``q = eps·es/(p − (1−eps)·es)`` for ``es``, then invert the Tetens law
    ``es = ES0·exp(c3·(T−tmelt)/(T−c4))`` for ``T``::

        es = p·q / (eps + q·(1−eps))
        T_d = (c3·tmelt − c4·ln(es/ES0)) / (c3 − ln(es/ES0))

    Used by ``EchamSurface`` / ``compute_surface_diagnostics`` (jax-gcm#1140)
    for the 2 m dew point: ``jcm.physics.surface.echam.turbulent_fluxes``
    calls this on the already-interpolated 2 m humidity, consistent with
    this module's documented role as "the constants ECHAM itself uses only
    to invert the 2 m dew point" (module docstring above) — ECHAM's own
    postproc_land/_ocean/_ice use exactly these ``c1es``/``c3les``/``c4les``/
    ``c3ies``/``c4ies`` constants for this one purpose.

    This is *not* a port of ECHAM's own ``zdew2w``/``zdewl``/``zdewi`` lines
    (``mo_surface_ocean.f90::postproc_ocean`` and the land/ice analogues):
    as transcribed, that formula's Magnus argument is ``p·q/(1+vtmpc1·q)``
    with no explicit division by ``eps`` before the ``LOG(·/c2es)``, which
    is ``eps`` too small relative to the ``es`` the Tetens law defines (it
    is consistent with the *raw* ``ua(T) = eps·es(T)`` table value that
    ECHAM's own ``qsat`` lookup divides out one step earlier, not with
    ``es`` itself) — feeding that line a ``q`` taken at EXACT saturation
    over a temperature ``T`` does not return ``T``: at ``T=290 K``,
    ``p=101325 Pa`` it recovers 282.7 K, a ~7 K low bias, by direct
    calculation against this reference. Reproducing that specific offset
    would fail this function's own round-trip requirement (saturation in
    gives the same temperature back out), so this implementation is the
    exact inverse of jcm's *own* forward law instead of a literal
    transcription of that one Fortran line; see the design doc
    cross-referenced from the call site for the measured offset.

    ``phase="auto"`` inverts assuming liquid water first; where that
    estimate falls below ``tmelt`` (self-inconsistent with the water
    branch's own saturation curve), it is redone over ice — the same
    water-above/ice-below-``tmelt`` rule :func:`saturation_vapor_pressure`
    uses in the forward direction, so a round trip through ``phase="auto"``
    on each side picks the same branch on both legs.

    Args:
        specific_humidity: Specific humidity [kg/kg] (floored above 0 to
            keep the log finite; zero humidity has no finite dew point).
        pressure: Pressure (Pa), the same pressure ``specific_humidity``
            saturates at.
        phase: ``"auto"`` (default), ``"water"`` or ``"ice"``.

    Returns:
        Dew-point temperature [K].

    """
    _validate_phase(phase)
    q = jnp.maximum(specific_humidity, 1.0e-12)
    # Exact inverse of ``saturation_specific_humidity``'s
    # ``qs = eps·es/(p - (1-eps)·es)``.
    es = pressure * q / (c.eps + q * (1.0 - c.eps))
    log_ratio = jnp.log(jnp.maximum(es / ES0, 1.0e-12))

    def _invert(c3, c4):
        return (c3 * c.tmelt - c4 * log_ratio) / (c3 - log_ratio)

    if phase == "water":
        return _invert(_C3LES, _C4LES)
    if phase == "ice":
        return _invert(_C3IES, _C4IES)
    t_water = _invert(_C3LES, _C4LES)
    t_ice = _invert(_C3IES, _C4IES)
    return jnp.where(t_water >= c.tmelt, t_water, t_ice)
