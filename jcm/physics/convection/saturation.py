"""Saturation thermodynamics for the convection schemes.

Thin wrappers around :mod:`jcm.physics.thermodynamics` — the single shared
ECHAM 6.3 saturation implementation — preserving the signatures that
``tiedtke_nordeng`` (updraft, downdraft, cuadjtq Newton steps, CAPE) and
``betts_miller`` call.

Scheme-specific choices are parameters rather than separate copies:

* ``phase`` selects the saturation surface — ``"auto"`` (water at/above the
  melting point, ice below; the ECHAM/Tiedtke convention), ``"water"``
  (Betts-Miller, following Isca), or ``"ice"``.
* ``clip`` optionally bounds the returned specific humidity (Tiedtke clips to
  ``[0, 0.5]`` via :func:`saturation_mixing_ratio`).

The coefficients are ECHAM's ``mo_convect_tables`` c3/c4 pairs (water:
``c3les = 17.269`` / ``c4les = 35.86 K``; ice: ``c3ies = 21.875`` /
``c4ies = 7.66 K``). This module previously carried its own Tetens constants
whose ice pair reused the *water* ``c4`` (35.86) as the ice ``A`` coefficient
— a transcription error that made sub-freezing saturation ~3× too low at
−20 °C and ~60× too low at −60 °C. Delegating to the shared module is what
fixed that; the water branch changed only microscopically (17.27/237.3 →
17.269 / (T − 35.86), equivalent to ~0.01 %).

These functions are intentionally *not* ``@jit``-ed: they are always called
inside the model's outer jit (the ``_run_from_state`` step), so they are
inlined and fused into that compiled graph. ``phase`` is a static Python
string resolved at trace time.
"""

import jax.numpy as jnp

from jcm.physics import thermodynamics

# Saturation vapour pressure at the melting point [Pa] — re-exported because
# callers/tests use it as the phase-independent anchor value es(tmelt).
ES0 = thermodynamics.C1ES


def saturation_vapor_pressure(temperature: jnp.ndarray,
                              phase: str = "auto") -> jnp.ndarray:
    """Saturation vapour pressure (Pa).

    Args:
        temperature: Temperature (K).
        phase: ``"auto"`` (water at/above ``tmelt``, ice below), ``"water"``
            or ``"ice"``.

    """
    return thermodynamics.saturation_vapor_pressure(temperature, phase=phase)


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 phase: str = "auto",
                                 clip: tuple[float, float] | None = None) -> jnp.ndarray:
    """Saturation specific humidity [kg/kg].

    Args:
        temperature: Temperature (K).
        pressure: Pressure (Pa).
        phase: Saturation surface, see :func:`saturation_vapor_pressure`.
        clip: Optional ``(lo, hi)`` bound on the result (applied on top of
            the shared implementation's ECHAM ``MIN(..., 0.5)`` cap).

    """
    qs = thermodynamics.saturation_specific_humidity(
        temperature, pressure, phase=phase)
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
    bit-reproducible under JIT without differentiating through the guard
    logic; the derivative uses the same c3/c4 coefficients as the value.
    """
    return thermodynamics.saturation_specific_humidity_and_derivative(
        temperature, pressure, phase=phase)
