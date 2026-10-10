"""In-cloud impaction scavenging: ECHAM-HAM's ``ic_scav_imp``.

The port of HAM2.3's size-dependent in-cloud impaction (``nwetdep = 3``,
``mo_ham_wetdep.f90::ic_scav_imp``; Croft et al. 2010): the fraction of a
mode's interstitial aerosol that cloud droplets and ice crystals collect,
per phase and per moment.

* **Droplets.** ``SCAVDROPN`` / ``SCAVDROPM`` are interpolated at the
  droplet effective radius ``reffl`` and the aerosol radius; the value is
  the scavenged fraction itself.
* **Ice.** ``SCAVICEPLATE`` is interpolated at the crystal effective
  radius ``reffi`` and the aerosol radius. The value is a collection kernel
  ``K`` of a plate, and the fraction is ``1 - exp(-K * 1e-6 * ICNC * dt)``
  with ICNC the in-cloud crystal number in m-3.
* **Aerosol radius.** The mode's wet count-median radius for the number
  moment and the wet mass-median radius, ``r * exp(3 ln^2 sigma)``, for the
  mass moment (HAM's ``cmedr2mmedr``), capped at 50 um.
* **Interpolation.** HAM's ``scavcoef_bilinterp``
  (``mo_ham_tools.f90``): bilinear in the two radii between the bracketing
  table nodes, with its own branches for a collapsed axis.

``get_icscavfrac`` clips the fraction to [0, 1]; the functions here return
it clipped the same way.

Three evident defects of r7492 are corrected by default (variant
``"ham"``). Each is reproducible with ``variant="ham_r7492"``, which
matches the compiled r7492 routines to round-off:

* **Droplet axis node 6.** ``cdroprad(6)`` reads 0.0 where the 5 um spacing
  gives 30.0. Droplets of 25-35 um then interpolate against a 0 um node.
* **Corner order.** ``ic_scav_imp`` fills ``Q12`` with the (x2, y1) corner
  and ``Q21`` with (x1, y2), but ``scavcoef_bilinterp`` reads ``Q21`` as
  (x2, y1) and ``Q12`` as (x1, y2). The off-diagonal corners are swapped,
  so the result is not an interpolation of the table: at the (x2, y1)
  node it returns the (x1, y2) entry.
* **Plate index above 50 um.** The plate axis is 25 um apart above 50 um,
  but the node pair is ``8 + floor(reffi/50)``. For ``reffi`` in
  [50, 100) the bracket is (45, 50) um, and in [100, 150) it is
  (50, 75) um, so the interpolation extrapolates up to about ten bracket
  widths. ``8 + floor(reffi/25)`` brackets every radius.

With the three corrections the result is the piecewise-bilinear
interpolant of the tables, continuous in both radii except at HAM's 1 um
crystal gate, and clamped at the table ends: constant beyond the last
droplet node and the largest plate, and zero below a 1 um crystal, where
HAM's plate index returns the zero row (the two-moment scheme's crystals
are at least ``ceffmin`` = 10 um).

All functions broadcast over any input shape and are differentiable in
the radii, the crystal number and the timestep. The table node pair is
selected from the inputs under ``stop_gradient`` (an integer choice has no
derivative, and the ``log`` of a vanishing radius would otherwise poison
the reverse pass); the derivative flows through the interpolation weights.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.wetdep.incloud_impaction_tables import (
    CAERORAD_UM,
    CDROPRAD_UM,
    CDROPRAD_UM_R7492,
    CPLATERAD_UM,
    SCAVDROPM,
    SCAVDROPN,
    SCAVICEPLATE,
)

#: The variants: ``"ham"`` corrects the three r7492 defects (the default),
#: ``"ham_r7492"`` reproduces the compiled r7492 routines.
VARIANTS = ("ham", "ham_r7492")

#: HAM's ``zeps = EPSILON(1._dp)`` (mo_ham_wetdep.f90), below which a radius
#: or a crystal number counts as absent.
_ZEPS = float(np.finfo(np.float64).eps)

#: HAM caps the aerosol radius at 50 um before the lookup (``mr``).
_MR_CAP_UM = 50.0


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(
            f"in-cloud impaction variant {variant!r} is not one of {VARIANTS}")


def _check_moment(moment: str) -> None:
    if moment not in ("number", "mass"):
        raise ValueError(f"moment must be 'number' or 'mass', got {moment!r}")


def _const(table, like):
    return jnp.asarray(table, dtype=jnp.result_type(like, jnp.float32))


def count_to_mass_median(geom_std_dev: float) -> float:
    """HAM's ``cmedr2mmedr``: mass-median over count-median radius, ``exp(3 ln^2 sigma)``."""
    return math.exp(3.0 * math.log(geom_std_dev) ** 2)


def impaction_radius_um(r_wet_m, geom_std_dev: float, moment: str):
    """Aerosol radius HAM looks the tables up at [um] (``mr``).

    The wet count-median radius for the number moment, the wet mass-median
    radius for the mass moment, capped at 50 um (``ham_wetdep``:
    ``mr = MIN(rwet*zrad_fac, 50e-6)*1e6``).

    Args:
        r_wet_m: wet count-median radius of the mode [m].
        geom_std_dev: the mode's geometric standard deviation.
        moment: ``"number"`` or ``"mass"``.

    Returns:
        The lookup radius in micrometres, shaped like ``r_wet_m``.

    """
    _check_moment(moment)
    factor = 1.0 if moment == "number" else count_to_mass_median(geom_std_dev)
    return jnp.minimum(r_wet_m * factor, _MR_CAP_UM * 1.0e-6) * 1.0e6


def _aerosol_nodes(mr_um):
    """Aerosol-radius node pair (``ham_wetdep``'s ``indexy1``/``indexy2``)."""
    mr = jax.lax.stop_gradient(mr_um)
    live = mr > _ZEPS
    z = jnp.floor(3.0 * (jnp.log(1.0e4 * jnp.where(live, mr, 1.0))
                         / math.log(2.0)) + 1.0)
    i1 = jnp.where(live, jnp.clip(z, 0, 60), 0).astype(jnp.int32)
    i2 = jnp.where(live, jnp.clip(z + 1.0, 0, 60), 0).astype(jnp.int32)
    return i1, i2


def _droplet_nodes(reffl_um):
    """Droplet-radius node pair (``ic_scav_imp``, water phase)."""
    r = jax.lax.stop_gradient(reffl_um)
    live = r > _ZEPS
    f = jnp.floor(r / 5.0)
    i1 = jnp.where(live, jnp.clip(f, 0, 9), 0).astype(jnp.int32)
    i2 = jnp.where(live, jnp.clip(f + 1.0, 0, 9), 0).astype(jnp.int32)
    return i1, i2


def _plate_nodes(reffi_um, icnc_m3, coarse_step_um: float):
    """Plate-radius node pair (``ic_scav_imp``, ice phase).

    Below 50 um the 5 um part of the axis; from 50 um the 25 um part,
    entered at ``8 + floor(reffi/step)``: r7492 uses ``step = 50``, the
    corrected variant ``step = 25``. Without crystals, or below 1 um, the
    pair is (0, 0): the zero row.
    """
    r = jax.lax.stop_gradient(reffi_um)
    have = jax.lax.stop_gradient(icnc_m3) >= _ZEPS
    fine = have & (r < 50.0) & (r >= 1.0)
    coarse = have & (r >= 50.0)
    f = jnp.floor(r / 5.0)
    g = jnp.floor(r / coarse_step_um)
    i1 = jnp.where(coarse, jnp.clip(8.0 + g, 0, 34),
                   jnp.where(fine, jnp.clip(f, 0, 10), 0))
    i2 = jnp.where(coarse, jnp.clip(9.0 + g, 0, 34),
                   jnp.where(fine, jnp.clip(f + 1.0, 0, 10), 0))
    return i1.astype(jnp.int32), i2.astype(jnp.int32)


def scavcoef_bilinterp(x, y, x1, x2, y1, y2, q11, q12, q21, q22):
    """HAM's ``scavcoef_bilinterp`` (``mo_ham_tools.f90``), branch for branch.

    Bilinear in ``(x, y)`` with ``q11 = f(x1, y1)``, ``q21 = f(x2, y1)``,
    ``q12 = f(x1, y2)`` and ``q22 = f(x2, y2)``. Where one axis has
    collapsed (equal node values) it interpolates along the other only,
    from ``(q11, q21)`` along x and from ``(q21, q22)`` along y; where both
    have, it returns ``q11``. The unused denominators are 1, as in HAM, so
    every branch stays finite.
    """
    same_x = x1 == x2
    same_y = y1 == y2
    dx = jnp.where(same_x, 1.0, x2 - x1)
    dy = jnp.where(same_y, 1.0, y2 - y1)
    wx1 = (x2 - x) / dx
    wx2 = (x - x1) / dx
    wy1 = (y2 - y) / dy
    wy2 = (y - y1) / dy
    x_only = wx1 * q11 + wx2 * q21
    y_only = wy1 * q21 + wy2 * q22
    both = wy1 * (wx1 * q11 + wx2 * q21) + wy2 * (wx1 * q12 + wx2 * q22)
    return jnp.where(
        same_x,
        jnp.where(same_y, q11, y_only),
        jnp.where(same_y, x_only, both),
    )


def _interpolate(table, x_axis, x, i1, i2, mr_um, variant):
    """Gather the four corners as the variant fills them and interpolate."""
    tab = _const(table, x)
    xa = _const(x_axis, x)
    ya = _const(CAERORAD_UM, x)
    j1, j2 = _aerosol_nodes(mr_um)
    q11 = tab[i1, j1]
    q22 = tab[i2, j2]
    if variant == "ham_r7492":
        # ic_scav_imp's own filling: Q12 <- (x2, y1), Q21 <- (x1, y2).
        q12, q21 = tab[i2, j1], tab[i1, j2]
    else:
        q12, q21 = tab[i1, j2], tab[i2, j1]
    return scavcoef_bilinterp(x, mr_um, xa[i1], xa[i2], ya[j1], ya[j2],
                              q11, q12, q21, q22)


def droplet_impaction_fraction(reffl_um, mr_um, moment: str,
                               variant: str = "ham"):
    """Fraction of a mode's interstitial aerosol cloud droplets collect [-].

    ``ic_scav_imp``'s water phase: ``SCAVDROPN`` (number) or ``SCAVDROPM``
    (mass) at the droplet effective radius and the aerosol radius, clipped
    to [0, 1] as ``get_icscavfrac`` clips it.

    Args:
        reffl_um: in-cloud droplet effective radius [um] (0 without liquid).
        mr_um: aerosol lookup radius [um], see :func:`impaction_radius_um`.
        moment: ``"number"`` or ``"mass"``, selecting the table.
        variant: ``"ham"`` (default) or ``"ham_r7492"``.

    Returns:
        The impaction-scavenged fraction, broadcast over the inputs.

    """
    _check_variant(variant)
    _check_moment(moment)
    reffl_um, mr_um = jnp.broadcast_arrays(reffl_um, mr_um)
    table = SCAVDROPN if moment == "number" else SCAVDROPM
    axis = CDROPRAD_UM_R7492 if variant == "ham_r7492" else CDROPRAD_UM
    i1, i2 = _droplet_nodes(reffl_um)
    frac = _interpolate(table, axis, reffl_um, i1, i2, mr_um, variant)
    return jnp.clip(frac, 0.0, 1.0)


def crystal_impaction_fraction(reffi_um, icnc_m3, mr_um, dt,
                               variant: str = "ham"):
    """Fraction of a mode's interstitial aerosol ice crystals collect in ``dt`` [-].

    ``ic_scav_imp``'s ice phase: the plate kernel ``K`` from
    ``SCAVICEPLATE`` at the crystal effective radius and the aerosol radius,
    and the fraction ``1 - exp(-K * 1e-6 * ICNC * dt)``, clipped to [0, 1]
    as ``get_icscavfrac`` clips it. The same kernel serves number and mass;
    the moment enters through ``mr_um``.

    Args:
        reffi_um: in-cloud crystal effective radius [um] (0 without ice).
        icnc_m3: in-cloud crystal number concentration [m-3].
        mr_um: aerosol lookup radius [um], see :func:`impaction_radius_um`.
        dt: the timestep [s].
        variant: ``"ham"`` (default) or ``"ham_r7492"``.

    Returns:
        The impaction-scavenged fraction, broadcast over the inputs.

    """
    _check_variant(variant)
    reffi_um, icnc_m3, mr_um = jnp.broadcast_arrays(reffi_um, icnc_m3, mr_um)
    step = 50.0 if variant == "ham_r7492" else 25.0
    i1, i2 = _plate_nodes(reffi_um, icnc_m3, step)
    kernel = _interpolate(SCAVICEPLATE, CPLATERAD_UM, reffi_um, i1, i2,
                          mr_um, variant)
    # r7492's plate index extrapolates, and the kernel can go negative; HAM's
    # clip then zeroes the fraction. Flooring the kernel first gives the same
    # value and keeps exp(+x) from overflowing, which would turn the clip's
    # zero derivative into 0 * inf = NaN. The corrected lookup interpolates
    # non-negative entries, so the floor never binds there.
    kernel = jnp.maximum(kernel, 0.0)
    # -expm1 is HAM's 1 - exp(-x), conditioned for small x.
    frac = -jnp.expm1(-kernel * 1.0e-6 * icnc_m3 * dt)
    return jnp.clip(frac, 0.0, 1.0)
