"""The saturation vapour pressure ECHAM6.3's cloud routines look up.

ECHAM6.3 (r7492) does not use a Tetens formula in ``cover`` and ``cloud``. It
tabulates, in ``mo_echam_convect_tables.f90::init_convect_tables``
(l.262-309), the five-term fit

    ln es(T) = a1/T + a2 + a3·0.01·T + a4·1e-5·T² + a5·ln T        [es in Pa]

with separate coefficients over liquid water (``cavl1..5``, l.42-46) and over
ice (``cavi1..5``, l.48-52), and interpolates the table with a cubic Hermite
spline on 0.025 K knots whose knot derivatives are the analytic ones
(``fetch_ua_spline``, l.394-437). The tables store ``es·rd/rv``: the ``ua``
table uses the ice fit at and below ``tmelt`` and the water fit above it, the
``uaw`` table the water fit everywhere.

The functions here evaluate the fit itself rather than its spline. The
spline's interpolation error on 0.025 K knots is below 1e-10 of the value
across the tabulated range (``echam_saturation_test.py`` pins it), far below
float32 resolution.

Phase: ``cover`` and ``cloud`` choose the table per cell with ECHAM's ``lo2``
switch, ice saturation where ``T < cthomi`` or where ``T < tmelt`` and the
cloud ice exceeds ``csecfrl`` (``prepare_ua_index_spline``, l.664-667).

Saturation specific humidity is formed as ECHAM forms it
(``mo_cover.f90`` l.221-223, ``mo_radiation.f90`` l.422-425)::

    x  = MIN(es·rd/rv / p, 0.5)
    qs = x / (1 - vtmpc1·x)

which equals ``eps·es/(p - (1 - eps)·es)`` with ``eps = rd/rv`` below the
cap. The constants are read from :mod:`jcm.constants` at call time, so
``set_constants(rv=461.51, grav=9.80665, ...)`` reproduces ECHAM's own set.

This module serves the ECHAM cloud schemes. It does not replace the shared
Tetens helpers in :mod:`jcm.physics.thermodynamics` that other schemes use.
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c

__all__ = [
    "ECHAM_TABLE_T_MAX",
    "ECHAM_TABLE_T_MIN",
    "ICE_COEFFICIENTS",
    "WATER_COEFFICIENTS",
    "dlnes_dT_ice",
    "dlnes_dT_water",
    "es_ice",
    "es_water",
    "lo2_ice_phase",
    "qsat_from_es",
]

#: ``cavl1..cavl5``, ``mo_echam_convect_tables.f90`` l.42-46.
WATER_COEFFICIENTS = (-6096.9385, 21.2409642, -2.711193, 1.673952, 2.433502)
#: ``cavi1..cavi5``, ``mo_echam_convect_tables.f90`` l.48-52.
ICE_COEFFICIENTS = (-6024.5282, 29.32707, 1.0613868, -1.3198825, -0.49382577)

#: Bounds of ECHAM's table (``tlbound``/``tubound``, l.107-108). ECHAM stops
#: with a lookup error outside them; here the temperature is clipped to them,
#: which changes nothing inside.
ECHAM_TABLE_T_MIN = 50.0
ECHAM_TABLE_T_MAX = 400.0

#: ECHAM's cap on ``es·rd/rv/p`` (``mo_cover.f90`` l.221).
_X_MAX = 0.5


def _ln_es(temperature, coefficients):
    a1, a2, a3, a4, a5 = coefficients
    t = jnp.clip(temperature, ECHAM_TABLE_T_MIN, ECHAM_TABLE_T_MAX)
    return a1 / t + a2 + a3 * 0.01 * t + a4 * 1.0e-5 * t * t + a5 * jnp.log(t)


def _dln_es_dT(temperature, coefficients):
    a1, _, a3, a4, a5 = coefficients
    t = jnp.clip(temperature, ECHAM_TABLE_T_MIN, ECHAM_TABLE_T_MAX)
    return -a1 / (t * t) + a3 * 0.01 + a4 * 2.0e-5 * t + a5 / t


def es_water(temperature):
    """Saturation vapour pressure over liquid water [Pa] (``cavl`` fit)."""
    return jnp.exp(_ln_es(temperature, WATER_COEFFICIENTS))


def es_ice(temperature):
    """Saturation vapour pressure over ice [Pa] (``cavi`` fit)."""
    return jnp.exp(_ln_es(temperature, ICE_COEFFICIENTS))


def dlnes_dT_water(temperature):
    """``d ln(es_water)/dT`` [1/K], the analytic slope ECHAM tabulates."""
    return _dln_es_dT(temperature, WATER_COEFFICIENTS)


def dlnes_dT_ice(temperature):
    """``d ln(es_ice)/dT`` [1/K], the analytic slope ECHAM tabulates."""
    return _dln_es_dT(temperature, ICE_COEFFICIENTS)


def lo2_ice_phase(temperature, cloud_ice, csecfrl, cthomi):
    """ECHAM's ``lo2`` phase switch: ``True`` selects ice saturation.

    ``lo2 = (T < cthomi) .OR. (T < tmelt .AND. xi > csecfrl)``, as coded with
    ``FSEL`` in ``prepare_ua_index_spline`` (l.664-667): the comparisons are
    strict, so ``T == tmelt`` and ``xi == csecfrl`` select water.
    """
    return (temperature < cthomi) | (
        (temperature < c.tmelt) & (cloud_ice > csecfrl))


def qsat_from_es(es, pressure):
    """Saturation specific humidity [kg/kg] from ``es``, as ECHAM forms it.

    ``x = MIN(es·rd/rv/p, 0.5)``, ``qs = x/(1 - vtmpc1·x)``
    (``mo_cover.f90`` l.221-223). With ``vtmpc1 = rv/rd - 1`` the denominator
    is at least ``1 - 0.5·vtmpc1 > 0``, so no further guard is needed.
    """
    x = jnp.minimum(es * (c.rd / c.rv) / pressure, _X_MAX)
    return x / (1.0 - c.vtmpc1 * x)
