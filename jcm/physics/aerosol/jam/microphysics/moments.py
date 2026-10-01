r"""Log-normal moment algebra for a modal aerosol population.

Every modal microphysics process is an integral of some kernel over a
mode's size distribution, and for a log-normal mode those integrals reduce
to moments of the distribution. Condensation needs the integral of the
transition-regime-corrected diameter; coagulation needs the Brownian kernel
integrals; ageing needs the mode's surface area to count monolayers;
renaming needs the mass and number above a cut diameter. All of them are
built from one expression, so it is written once here.

For a log-normal mode with number ``N``, number-median DIAMETER ``Dg`` and
geometric standard deviation ``sigma_g``, the k-th moment of the diameter
distribution is

.. math::

    M_k = N \\, D_g^k \\, \\exp\\!\\left(\\tfrac{1}{2} k^2 \\ln^2 \\sigma_g\\right)

so ``M_0 = N``, the total surface area is ``pi * M_2`` and the total volume
is ``(pi/6) * M_3``. Two consequences are used throughout:

* the moments of a log-normal are themselves log-normal in ``Dg``, which is
  why ``diameter_from_volume`` below is a closed-form inversion rather than
  an iteration;
* ``sigma_g`` is a FIXED property of each mode in MAM — modes are defined by
  their width and particles move between them rather than the widths
  changing — so ``exp(k^2 ln^2 sigma_g / 2)`` is a compile-time constant per
  (mode, k) and never a traced value.

Diameters, not radii. ``AerosolMode.dgnum`` and its bounds are diameters,
and the MAM/CMAQ literature these processes come from is written in
diameters; ``JamAerosolState`` publishes radii because the deposition and
optics terms downstream want them. That conversion happens at the boundary
in :mod:`~jcm.physics.aerosol.jam.microphysics.koehler`, not here.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from jcm.physics.aerosol.jam.population import AerosolMode, ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name

#: Number floor [kg^-1]. An empty mode must give a finite diameter and a
#: finite *gradient*: ``volume / number`` with number exactly zero is inf in
#: the forward pass and NaN in the reverse pass, and a NaN gradient reaches
#: every upstream parameter. Identical to the floor ``koehler`` used before
#: it delegated here, so the refactor is bitwise-neutral.
TINY_NUMBER = 1.0e-30

#: Volume floor [m^3/kg], likewise.
TINY_VOLUME = 1.0e-40


def moment_factor(mode: AerosolMode, k: float) -> float:
    """``exp(k^2 ln^2(sigma_g) / 2)``, the log-normal moment factor.

    A plain Python float: ``sigma_g`` is static per mode, so this folds at
    trace time rather than being recomputed per cell.
    """
    ln_sigma = math.log(mode.geom_std_dev)
    return math.exp(0.5 * k * k * ln_sigma * ln_sigma)


def moment(number, diameter, mode: AerosolMode, k: float):
    """Return the k-th diameter moment ``M_k`` of one mode.

    Args:
        number: number concentration ``N`` [kg^-1], ``(nlev, ncols)``.
        diameter: number-median dry diameter ``Dg`` [m], same shape.
        mode: supplies the static ``geom_std_dev``.
        k: moment order. Need not be an integer: the transition-regime
            condensation kernel uses fractional orders.

    Returns:
        ``M_k`` in [kg^-1 m^k].

    """
    return number * diameter ** k * moment_factor(mode, k)


def volume_factor(mode: AerosolMode) -> float:
    """``(pi/6) exp(9/2 ln^2 sigma_g)``: total volume per ``N * Dg^3``.

    Named because it is needed in both directions, volume from diameter and
    diameter from volume, and writing it out twice is how the two drift.
    """
    return (math.pi / 6.0) * moment_factor(mode, 3.0)


def volume_from_diameter(number, diameter, mode: AerosolMode):
    """Total dry volume [m^3/kg] of a mode from its number and ``Dg``."""
    return volume_factor(mode) * number * diameter ** 3


def diameter_from_volume(volume, number, mode: AerosolMode):
    """Number-median dry diameter [m] from total volume and number.

    The closed-form inversion of :func:`volume_from_diameter`, clipped to
    the mode's ``[dgnum_lo, dgnum_hi]``. The clip is what makes a mode a
    mode: MAM defines each by a fixed size window and moves particles
    BETWEEN modes (renaming, ageing) rather than letting one grow without
    limit, so an out-of-window diameter here is a signal for those
    processes, not a value to carry.

    **Why two ``where`` calls and not a floor.** ``volume / max(number,
    tiny)`` is finite forward but its reverse-mode derivative with respect
    to ``number`` is ``-volume / max(number, tiny)^2``, and ``tiny^2``
    underflows to zero in float32 (``1e-60`` is below the smallest
    subnormal). That is ``0/0`` in the cotangent, and ``where`` does not
    rescue it: ``jnp.where(ok, nan, 0)`` is fine forward and NaN in reverse,
    because the unselected branch's cotangent is ``NaN * 0``. Routing the
    unselected branch through a harmless divisor (``1.0``) and masking the
    *result* is the standard fix (JAX_gotchas.md). Measured 2026-10-01: the
    floor form gave a NaN gradient at ``(volume=0, number=0)`` in float32
    and a finite one in float64.

    An empty mode (number below the floor) reports its lower bound, which
    is also where ``cbrt`` of the floored argument lands, so the forward
    value at the boundary is continuous.
    """
    populated = number > TINY_NUMBER
    safe_number = jnp.where(populated, number, 1.0)
    cubed = jnp.where(
        populated, volume / (safe_number * volume_factor(mode)), 0.0)
    diameter = jnp.cbrt(jnp.maximum(cubed, mode.dgnum_lo ** 3))
    return jnp.clip(diameter, mode.dgnum_lo, mode.dgnum_hi)


def surface_area(number, diameter, mode: AerosolMode):
    """Total surface area [m^2/kg] of a mode, ``pi * M_2``.

    What ageing counts monolayers against and condensation partitions mass
    by, so AER-03 and AER-06 both read it.
    """
    return math.pi * moment(number, diameter, mode, 2.0)


def mode_volume(masses: dict, mode: AerosolMode, spec: ModalAerosolSpec):
    """Total dry volume [m^3/kg] of one mode: ``sum_i m_i / rho_i``.

    Args:
        masses: flat ``{m_<species>_<mode>: array}`` dry masses [kg/kg].
        mode: the mode to total.
        spec: supplies each species' material density.

    """
    total = None
    for species in mode.species:
        contribution = (masses[mass_name(species, mode.short)]
                        / spec.species_props(species).density)
        total = contribution if total is None else total + contribution
    return total


def mode_mass(masses: dict, mode: AerosolMode):
    """Total dry mass [kg/kg] of one mode, summed over its species."""
    total = None
    for species in mode.species:
        contribution = masses[mass_name(species, mode.short)]
        total = contribution if total is None else total + contribution
    return total
