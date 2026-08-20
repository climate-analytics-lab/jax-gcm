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
from jax import lax

import jcm.constants as c
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


def cuadjtq_newton(
    temperature: jnp.ndarray,
    total_water: jnp.ndarray,
    pressure: jnp.ndarray,
    n_refine: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Newton-Raphson saturation adjustment (``cuadjtq``, kcall=1 flavour).

    Matches ECHAM/ICON ``mo_cuadjust.f90`` ``cuadjtq`` for the
    "condensation-only" mode used inside updrafts. The first iteration
    clips the Newton step to be non-negative (only condensation, never
    evaporation of pre-existing liquid). Subsequent refinement iterations
    allow both directions so Newton overshoot in one direction can be
    corrected.

    The Newton step:

        Δq = (q - qs(T)) / (1 + (L/cp) * dqs/dT)

    is the linearised solution to ``q - Δq = qs(T + L·Δq/cp)``
    (ECHAM ``mo_cuadjust.f90:139-142``, ``zcond = (pq-zqsat)/(1+zlcdqsdt)``).
    The ``1 + (L/cp)·dqs/dT`` denominator is what makes this correct: the
    naive ``q - qs(T)`` over-condenses, because condensing warms the parcel
    and so *raises* the saturation value the parcel has to meet. With one
    refinement the residual ``q - qs(T_adj)`` typically drops to <~0.5%
    even for strong supersaturation; a single undamped pass leaves parcels
    3-30% off, under-releasing latent heat and cooling the mid-troposphere
    in RCE.

    Total water is conserved by construction — every step moves the same
    ``cond`` from vapour to liquid — and a subsaturated parcel is returned
    unchanged rather than being moistened up to saturation.

    Lives here rather than in ``tiedtke_nordeng/updraft.py`` so that
    ``calculate_cape_cin`` (in ``tiedtke_nordeng.py``, which ``updraft``
    imports from) can call the same routine without an import cycle.

    Args:
        temperature: Temperature (K)
        total_water: Total water mixing ratio (kg/kg)
        pressure: Pressure (Pa)
        n_refine: Number of refinement iterations after the first
            condensation-only pass (Fortran cuadjtq uses 1 refinement).

    Returns:
        Tuple of (T_adj, vapour, liquid) with ``vapour + liquid == total_water``
        and ``vapour ≈ qs(T_adj)`` to within a fraction of a percent.

    """
    def _lcp(T):
        # Phase-consistent latent heat: L_s pairs with the ice saturation
        # branch of ``phase="auto"`` below tmelt (review finding 2.7).
        return jnp.where(T >= c.tmelt, c.alhc, c.alhs) / c.cpd

    def _first_pass(T, q_vap, liq):
        """Condensation-only Newton step (kcall=1)."""
        L_cp = _lcp(T)
        qs, dqs_dT = saturation_specific_humidity_and_derivative(T, pressure)
        cond = (q_vap - qs) / (1.0 + L_cp * dqs_dT)
        cond = jnp.maximum(cond, 0.0)
        return T + L_cp * cond, q_vap - cond, liq + cond

    def _refine_body(carry, _):
        """Refinement: allow both directions (kcall=0) to correct Newton
        overshoot, but only while there's liquid available to re-evaporate.
        """
        T, q_vap, liq = carry
        L_cp = _lcp(T)
        qs, dqs_dT = saturation_specific_humidity_and_derivative(T, pressure)
        cond = (q_vap - qs) / (1.0 + L_cp * dqs_dT)
        # Don't evaporate more than available liquid
        cond = jnp.maximum(cond, -liq)
        return (T + L_cp * cond, q_vap - cond, liq + cond), None

    T1, q1, liq1 = _first_pass(temperature,
                               total_water,
                               jnp.zeros_like(total_water))
    (T_adj, vapor, liquid), _ = lax.scan(
        _refine_body, (T1, q1, liq1), None, length=n_refine
    )
    return T_adj, vapor, liquid


def cuadjtq_newton_evap(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    n_refine: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Newton-Raphson wet-bulb adjustment (``cuadjtq``, kcall=2 flavour).

    ECHAM's evaporation-only mode, used wherever descending or mixed air
    evaporates precipitation toward saturation (``cudlfs`` / ``cuddraf``,
    mo_cuadjust.f90:149-166): the same damped Newton step as
    :func:`cuadjtq_newton`,

        Δq = (q − qs(T)) / (1 + (L/cp)·dqs/dT),

    but clipped ``MIN(Δq, 0)`` in every pass — only evaporation, never
    condensation, so already-saturated air is returned unchanged. The fixed
    point is the isobaric wet bulb: ``cp·ΔT + L·Δq = 0`` by construction,
    so moist static energy is conserved exactly, and the state-dependent
    damper is what a hand-rolled ``T − 0.3·(L/cp)(qs−q)`` (the pre-#694
    downdraft code) replaced with a constant — off by −5.2 kJ/kg of MSE at
    300 K/900 hPa and +3.2 kJ/kg at 280 K/700 hPa, with the sign flipping
    in the mid-troposphere where the LFS sits.

    Args:
        temperature: Temperature (K).
        humidity: Specific humidity (kg/kg).
        pressure: Pressure (Pa).
        n_refine: Refinement passes after the first (ECHAM uses 1).

    Returns:
        Tuple of ``(T_wb, q_wb)`` with ``cp·(T_wb − T) + L·(q_wb − q) = 0``.

    """
    def _lcp(T):
        return jnp.where(T >= c.tmelt, c.alhc, c.alhs) / c.cpd

    def _pass(carry, _):
        T, q = carry
        L_cp = _lcp(T)
        qs, dqs_dT = saturation_specific_humidity_and_derivative(T, pressure)
        cond = (q - qs) / (1.0 + L_cp * dqs_dT)
        cond = jnp.minimum(cond, 0.0)          # kcall=2: evaporation only
        return (T + L_cp * cond, q - cond), None

    (T_wb, q_wb), _ = lax.scan(
        _pass, (temperature, humidity), None, length=1 + n_refine
    )
    return T_wb, q_wb
