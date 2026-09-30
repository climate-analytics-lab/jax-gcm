"""Saturation of the Tiedtke-Nordeng scheme: ECHAM's ``ua`` table and ``cuadjtq``.

ECHAM's convection reads saturation from the ``ua`` lookup table everywhere
(``cumastr``, ``cuini``, ``cubase``, ``cuasc``, ``cudlfs``, ``cuddraf`` and
``cuadjtq``: ``lookup_ua_spline`` / ``lookup_ua_list_spline``, e.g.
``mo_cumastr.f90`` l.220-221 and 601-603, ``mo_cuadjust.f90`` l.85-86): Sonntag
(1990) over ice at and below the melting point and over water above
(:func:`jcm.physics.thermodynamics.es_ua`). ``cuadjtq`` takes ``L/cp`` from
``lookup_ubc`` with the same switch (``als/cpd`` at and below ``tmelt``,
``alv/cpd`` above; ``mo_echam_convect_tables.f90`` l.328-332).

The functions live in this leaf module, which imports only
:mod:`jcm.physics.thermodynamics`, so that ``tiedtke_nordeng.py`` (trigger,
CAPE), ``updraft.py``, ``downdraft.py`` and ``half_levels.py`` can all call them
without an import cycle.

They are intentionally not ``@jit``-ed: they run inside the model's outer jit
and inline there.
"""

import jax.numpy as jnp
from jax import lax

import jcm.constants as c
from jcm.physics import thermodynamics


def saturation_mixing_ratio(pressure: jnp.ndarray,
                            temperature: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity [kg/kg] from ECHAM's ``ua`` table.

    ``x = MIN(ua/p, 0.5)``, ``qs = x/(1 − vtmpc1·x)`` (``mo_cumastr.f90``
    l.220-226, ``mo_cuadjust.f90`` l.107-110). The argument order
    ``(pressure, temperature)`` is the scheme's historical one.
    """
    return thermodynamics.saturation_specific_humidity(
        temperature, pressure, phase="auto")


def lcp_ua(temperature):
    """``L/cp`` of ``lookup_ubc``: ``als/cpd`` at and below ``tmelt``, else ``alv/cpd``.

    DRY ``cpd`` is the reference: ``cuadjtq`` reads ``L/cp`` from ``uc``,
    built with ``zalvdcp = alv/cpd``, ``zalsdcp = als/cpd``
    (``mo_echam_convect_tables.f90`` l.214-215, 322-323), not the moist
    ``zcpq`` of the ``cumastr`` static-energy ledger. The switch is the ``ua``
    table's, so the latent heat always pairs with the saturation surface.
    """
    return jnp.where(thermodynamics.ua_ice_phase(temperature),
                     c.alhs, c.alhc) / c.cpd


def cuadjtq_newton(
    temperature: jnp.ndarray,
    total_water: jnp.ndarray,
    pressure: jnp.ndarray,
    n_refine: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Newton-Raphson saturation adjustment (``cuadjtq``, kcall=1 flavour).

    ECHAM ``mo_cuadjust.f90`` ``cuadjtq`` in the "condensation-only" mode
    used inside updrafts. The first iteration clips the Newton step to be
    non-negative (only condensation, never evaporation of pre-existing
    liquid). The refinement iterations allow both directions, bounded by the
    liquid, so Newton overshoot can be corrected; ECHAM takes one unclipped
    refinement where this takes ``n_refine`` (#957).

    The Newton step:

        Δq = (q - qs(T)) / (1 + (L/cp) * dqs/dT)

    is the linearised solution to ``q - Δq = qs(T + L·Δq/cp)``
    (ECHAM ``mo_cuadjust.f90`` l.107-117, ``zcond = (pq-zqsat)/(1+zlcdqsdt)``).
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
    def _first_pass(T, q_vap, liq):
        """Condensation-only Newton step (kcall=1)."""
        L_cp = lcp_ua(T)
        qs, dqs_dT = thermodynamics.saturation_specific_humidity_and_derivative(
            T, pressure)
        cond = (q_vap - qs) / (1.0 + L_cp * dqs_dT)
        cond = jnp.maximum(cond, 0.0)
        return T + L_cp * cond, q_vap - cond, liq + cond

    def _refine_body(carry, _):
        """Refinement: allow both directions (kcall=0) to correct Newton
        overshoot, but only while there's liquid available to re-evaporate.
        """
        T, q_vap, liq = carry
        L_cp = lcp_ua(T)
        qs, dqs_dT = thermodynamics.saturation_specific_humidity_and_derivative(
            T, pressure)
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
    mo_cuadjust.f90 l.149-166): the same damped Newton step as
    :func:`cuadjtq_newton`,

        Δq = (q − qs(T)) / (1 + (L/cp)·dqs/dT),

    but clipped ``MIN(Δq, 0)`` in every pass — only evaporation, never
    condensation, so already-saturated air is returned unchanged (ECHAM's
    refinement pass is unclipped, #957). The fixed
    point is the isobaric wet bulb: ``cp·ΔT + L·Δq = 0`` by construction,
    so moist static energy is conserved exactly; the state-dependent damper
    is what makes the evaporated amount the wet-bulb deficit rather than a
    fixed fraction of the saturation deficit.

    Args:
        temperature: Temperature (K).
        humidity: Specific humidity (kg/kg).
        pressure: Pressure (Pa).
        n_refine: Refinement passes after the first (ECHAM uses 1).

    Returns:
        Tuple of ``(T_wb, q_wb)`` with ``cp·(T_wb − T) + L·(q_wb − q) = 0``.

    """
    def _pass(carry, _):
        T, q = carry
        L_cp = lcp_ua(T)
        qs, dqs_dT = thermodynamics.saturation_specific_humidity_and_derivative(
            T, pressure)
        cond = (q - qs) / (1.0 + L_cp * dqs_dT)
        cond = jnp.minimum(cond, 0.0)          # kcall=2: evaporation only
        return (T + L_cp * cond, q - cond), None

    (T_wb, q_wb), _ = lax.scan(
        _pass, (temperature, humidity), None, length=1 + n_refine
    )
    return T_wb, q_wb
