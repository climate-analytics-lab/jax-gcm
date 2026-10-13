"""Saturation and phase rule of the ECHAM cloud schemes.

The cloud cover (``sundqvist.py``) and the 1M scheme (``echam_1m.py``) take
their saturation vapour pressure, its log derivative and ``qs`` from here.
Under the default :data:`SATURATION_FORMULA` (``"sonntag"``) every one of them
is :mod:`jcm.physics.thermodynamics`' own function: ECHAM's Sonntag (1990)
fit, the one saturation implementation all ECHAM physics in jcm reads.
``WATER_COEFFICIENTS``, ``ICE_COEFFICIENTS``, ``ECHAM_TABLE_T_MIN``/``_MAX``
and :func:`qsat_from_es` are re-exported from it under the same names, and
``sonntag_es_water``, ``sonntag_es_ice``, ``sonntag_dlnes_dT_water`` and
``sonntag_dlnes_dT_ice`` are its ``es_water``, ``es_ice``, ``dlnes_dT_water``
and ``dlnes_dT_ice``.

The module adds two things to it:

- :func:`lo2_ice_phase`, ECHAM's ``lo2`` switch between the ice and the water
  surface, ice where ``T < cthomi`` or where ``T < tmelt`` and the cloud ice
  exceeds ``csecfrl`` (``prepare_ua_index_spline``,
  ``mo_echam_convect_tables.f90`` l.664-667). It depends on the cloud ice, so
  :mod:`~jcm.physics.thermodynamics` leaves it to the cloud schemes, whose
  cover and condensation choose the surface per cell with it. It is not
  ECHAM's ``ua`` table rule (:func:`jcm.physics.thermodynamics.ua_ice_phase`,
  ice at and below ``tmelt``).
- jcm's former Tetens pair (:func:`tetens_es_water`, :func:`tetens_es_ice`),
  which :data:`SATURATION_FORMULA` = ``"tetens"`` routes :func:`es_water`,
  :func:`es_ice` and their log derivatives through. That is a test utility,
  not a model option: the ECHAM Fortran reference data include a ``tetens``
  variant (ECHAM's routines run with this pair), and switching lets a test
  separate a formulation error from the vapour-pressure formula when
  localising a disagreement. Nothing in the model configuration sets it.
  Tetens differs from Sonntag by up to 0.14 % above 273 K, 2.4 % over water
  and 1.2 % over ice between 238 and 273 K, and 16 % and 8 % below.
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c
from jcm.physics import thermodynamics as _thermodynamics
from jcm.physics.thermodynamics import (
    ECHAM_TABLE_T_MAX,
    ECHAM_TABLE_T_MIN,
    ICE_COEFFICIENTS,
    WATER_COEFFICIENTS,
    qsat_from_es,
)

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

#: The formula :func:`es_water` / :func:`es_ice` evaluate: ECHAM's
#: ``"sonntag"``. ``"tetens"`` exists for tests only (module docstring).
SATURATION_FORMULA = "sonntag"

# --- Sonntag (1990): thermodynamics' functions, under the names the Tetens
# --- switch and its tests address them by -----------------------------------

sonntag_es_water = _thermodynamics.es_water
sonntag_es_ice = _thermodynamics.es_ice
sonntag_dlnes_dT_water = _thermodynamics.dlnes_dT_water
sonntag_dlnes_dT_ice = _thermodynamics.dlnes_dT_ice


# --- Tetens, jcm's former pair (test utility) --------------------------------

#: Tetens ``(a, b)`` of ``es = 610.78·exp(a·Tc/(Tc + b))``, jcm's pair.
TETENS_WATER = (17.27, 237.3)
TETENS_ICE = (21.87, 265.5)
_TETENS_E0 = 610.78


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


# --- The formula the ECHAM cloud schemes use ---------------------------------

def es_water(temperature):
    """Saturation vapour pressure over water [Pa].

    :func:`jcm.physics.thermodynamics.es_water` (ECHAM's ``uaw``·rv/rd).
    """
    if SATURATION_FORMULA == "tetens":
        return tetens_es_water(temperature)
    return _thermodynamics.es_water(temperature)


def es_ice(temperature):
    """Saturation vapour pressure over ice [Pa].

    :func:`jcm.physics.thermodynamics.es_ice`, the ice fit at every
    temperature.
    """
    if SATURATION_FORMULA == "tetens":
        return tetens_es_ice(temperature)
    return _thermodynamics.es_ice(temperature)


def dlnes_dT_water(temperature):
    """``d ln(es_water)/dT`` [1/K], :func:`jcm.physics.thermodynamics.dlnes_dT_water`."""
    if SATURATION_FORMULA == "tetens":
        return tetens_dlnes_dT_water(temperature)
    return _thermodynamics.dlnes_dT_water(temperature)


def dlnes_dT_ice(temperature):
    """``d ln(es_ice)/dT`` [1/K], :func:`jcm.physics.thermodynamics.dlnes_dT_ice`."""
    if SATURATION_FORMULA == "tetens":
        return tetens_dlnes_dT_ice(temperature)
    return _thermodynamics.dlnes_dT_ice(temperature)


# --- Phase ---------------------------------------------------------------------

def lo2_ice_phase(temperature, cloud_ice, csecfrl, cthomi):
    """ECHAM's ``lo2`` phase switch: ``True`` selects ice saturation.

    ``lo2 = (T < cthomi) .OR. (T < tmelt .AND. xi > csecfrl)``, as coded with
    ``FSEL`` in ``prepare_ua_index_spline`` (l.664-667): the comparisons are
    strict, so ``T == tmelt`` and ``xi == csecfrl`` select water.
    """
    return (temperature < cthomi) | (
        (temperature < c.tmelt) & (cloud_ice > csecfrl))
