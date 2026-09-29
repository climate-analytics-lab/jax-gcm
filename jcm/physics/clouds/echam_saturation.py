"""Saturation vapour pressure and phase rule of the ECHAM cloud schemes.

**The formula switch.** :func:`es_water`, :func:`es_ice` and their log
derivatives evaluate the formula named by :data:`SATURATION_FORMULA`, the one
setting the cloud cover and the 1M scheme both read:

* ``"tetens"`` (the default): jcm's Tetens pair
  ``es = 610.78·exp(a·Tc/(Tc + b))`` with ``(a, b) = (17.27, 237.3)`` over
  water and ``(21.87, 265.5)`` over ice, which the cloud schemes used before
  this switch existed.
* ``"sonntag"``: the formula ECHAM6.3's cloud routines actually use.

Which one is the default is decided for all of jcm's ECHAM physics at once:
convection and the two-moment scheme take their vapour pressure from
:mod:`jcm.physics.thermodynamics`, and a cloud scheme on a different curve
from convection would judge detrained condensate against another
saturation. Set it with :func:`set_saturation_formula` before building the
model, as with :func:`jcm.constants.set_constants`: it is read when a
function is traced.

**ECHAM's formula.** ECHAM6.3 (r7492) tabulates, in
``mo_echam_convect_tables.f90::init_convect_tables`` (l.262-309), the
five-term fit of Sonntag (1990)

    ln es(T) = a1/T + a2 + a3·0.01·T + a4·1e-5·T² + a5·ln T        [es in Pa]

with separate coefficients over liquid water (``cavl1..5``, l.42-46) and
over ice (``cavi1..5``, l.48-52), and interpolates the table with a cubic
Hermite spline on 0.025 K knots whose knot derivatives are the analytic ones
(``fetch_ua_spline``, l.394-437). The tables store ``es·rd/rv``: the ``ua``
table uses the ice fit at and below ``tmelt`` and the water fit above it,
the ``uaw`` table the water fit everywhere. :func:`sonntag_es_water` and
:func:`sonntag_es_ice` evaluate the fit itself rather than its spline; the
spline's interpolation error is below 1e-10 of the value across the
tabulated range (``echam_saturation_test.py`` pins it). Tetens differs from
it by up to 0.14 % above 273 K, 2.4 % over water and 1.2 % over ice between
238 and 273 K, and 16 % and 8 % below.

**Phase.** ``cover`` and ``cloud`` choose the surface per cell with ECHAM's
``lo2`` switch, ice where ``T < cthomi`` or where ``T < tmelt`` and the cloud
ice exceeds ``csecfrl`` (``prepare_ua_index_spline``, l.664-667):
:func:`lo2_ice_phase`.

**Saturation specific humidity** is formed as ECHAM forms it
(``mo_cover.f90`` l.221-223, ``mo_radiation.f90`` l.422-425)::

    x  = MIN(es·rd/rv / p, 0.5)
    qs = x / (1 - vtmpc1·x)

which equals ``eps·es/(p - (1 - eps)·es)`` with ``eps = rd/rv`` below the
cap: :func:`qsat_from_es`. The constants are read from :mod:`jcm.constants`
at call time, so ``set_constants(rv=461.51, grav=9.80665, ...)`` reproduces
ECHAM's own set.
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c

__all__ = [
    "ECHAM_TABLE_T_MAX",
    "ECHAM_TABLE_T_MIN",
    "ICE_COEFFICIENTS",
    "SATURATION_FORMULA",
    "SATURATION_FORMULAS",
    "TETENS_ICE",
    "TETENS_WATER",
    "WATER_COEFFICIENTS",
    "dlnes_dT_ice",
    "dlnes_dT_water",
    "es_ice",
    "es_water",
    "lo2_ice_phase",
    "qsat_from_es",
    "set_saturation_formula",
    "sonntag_dlnes_dT_ice",
    "sonntag_dlnes_dT_water",
    "sonntag_es_ice",
    "sonntag_es_water",
    "tetens_dlnes_dT_ice",
    "tetens_dlnes_dT_water",
    "tetens_es_ice",
    "tetens_es_water",
]

SATURATION_FORMULAS = ("tetens", "sonntag")

#: The formula :func:`es_water` / :func:`es_ice` evaluate. See the module
#: docstring; change it with :func:`set_saturation_formula`.
SATURATION_FORMULA = "tetens"

#: ``cavl1..cavl5``, ``mo_echam_convect_tables.f90`` l.42-46.
WATER_COEFFICIENTS = (-6096.9385, 21.2409642, -2.711193, 1.673952, 2.433502)
#: ``cavi1..cavi5``, ``mo_echam_convect_tables.f90`` l.48-52.
ICE_COEFFICIENTS = (-6024.5282, 29.32707, 1.0613868, -1.3198825, -0.49382577)

#: Tetens ``(a, b)`` of ``es = 610.78·exp(a·Tc/(Tc + b))``, jcm's pair.
TETENS_WATER = (17.27, 237.3)
TETENS_ICE = (21.87, 265.5)
_TETENS_E0 = 610.78

#: Bounds of ECHAM's table (``tlbound``/``tubound``, l.107-108). ECHAM stops
#: with a lookup error outside them; the Sonntag fit clips the temperature to
#: them, which changes nothing inside.
ECHAM_TABLE_T_MIN = 50.0
ECHAM_TABLE_T_MAX = 400.0

#: ECHAM's cap on ``es·rd/rv/p`` (``mo_cover.f90`` l.221).
_X_MAX = 0.5


def set_saturation_formula(name: str) -> None:
    """Choose the formula of :func:`es_water` / :func:`es_ice`.

    Args:
        name: ``"tetens"`` or ``"sonntag"``.

    """
    global SATURATION_FORMULA
    if name not in SATURATION_FORMULAS:
        raise ValueError(f"saturation formula must be one of "
                         f"{SATURATION_FORMULAS}, got {name!r}")
    SATURATION_FORMULA = name


# --- Sonntag (1990), the fit ECHAM's tables hold ---------------------------

def _ln_es(temperature, coefficients):
    a1, a2, a3, a4, a5 = coefficients
    t = jnp.clip(temperature, ECHAM_TABLE_T_MIN, ECHAM_TABLE_T_MAX)
    return a1 / t + a2 + a3 * 0.01 * t + a4 * 1.0e-5 * t * t + a5 * jnp.log(t)


def _dln_es_dT(temperature, coefficients):
    a1, _, a3, a4, a5 = coefficients
    t = jnp.clip(temperature, ECHAM_TABLE_T_MIN, ECHAM_TABLE_T_MAX)
    return -a1 / (t * t) + a3 * 0.01 + a4 * 2.0e-5 * t + a5 / t


def sonntag_es_water(temperature):
    """ECHAM's saturation vapour pressure over liquid water [Pa]."""
    return jnp.exp(_ln_es(temperature, WATER_COEFFICIENTS))


def sonntag_es_ice(temperature):
    """ECHAM's saturation vapour pressure over ice [Pa]."""
    return jnp.exp(_ln_es(temperature, ICE_COEFFICIENTS))


def sonntag_dlnes_dT_water(temperature):
    """``d ln(es)/dT`` over water [1/K], the analytic slope ECHAM tabulates."""
    return _dln_es_dT(temperature, WATER_COEFFICIENTS)


def sonntag_dlnes_dT_ice(temperature):
    """``d ln(es)/dT`` over ice [1/K], the analytic slope ECHAM tabulates."""
    return _dln_es_dT(temperature, ICE_COEFFICIENTS)


# --- Tetens, jcm's pair -----------------------------------------------------

def _tetens(temperature, coefficients):
    a, b = coefficients
    tc = temperature - c.tmelt
    return _TETENS_E0 * jnp.exp(a * tc / (tc + b))


def _tetens_dln(temperature, coefficients):
    a, b = coefficients
    tc = temperature - c.tmelt
    return a * b / (tc + b) ** 2


def tetens_es_water(temperature):
    """Return jcm's Tetens saturation vapour pressure over water [Pa]."""
    return _tetens(temperature, TETENS_WATER)


def tetens_es_ice(temperature):
    """Return jcm's Tetens saturation vapour pressure over ice [Pa]."""
    return _tetens(temperature, TETENS_ICE)


def tetens_dlnes_dT_water(temperature):
    """``d ln(es)/dT`` of :func:`tetens_es_water` [1/K]."""
    return _tetens_dln(temperature, TETENS_WATER)


def tetens_dlnes_dT_ice(temperature):
    """``d ln(es)/dT`` of :func:`tetens_es_ice` [1/K]."""
    return _tetens_dln(temperature, TETENS_ICE)


# --- The selected formula ---------------------------------------------------

def es_water(temperature):
    """Saturation vapour pressure over water [Pa], selected formula."""
    if SATURATION_FORMULA == "sonntag":
        return sonntag_es_water(temperature)
    return tetens_es_water(temperature)


def es_ice(temperature):
    """Saturation vapour pressure over ice [Pa], selected formula."""
    if SATURATION_FORMULA == "sonntag":
        return sonntag_es_ice(temperature)
    return tetens_es_ice(temperature)


def dlnes_dT_water(temperature):
    """``d ln(es_water)/dT`` [1/K] of the selected formula."""
    if SATURATION_FORMULA == "sonntag":
        return sonntag_dlnes_dT_water(temperature)
    return tetens_dlnes_dT_water(temperature)


def dlnes_dT_ice(temperature):
    """``d ln(es_ice)/dT`` [1/K] of the selected formula."""
    if SATURATION_FORMULA == "sonntag":
        return sonntag_dlnes_dT_ice(temperature)
    return tetens_dlnes_dT_ice(temperature)


# --- Phase and humidity -----------------------------------------------------

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
