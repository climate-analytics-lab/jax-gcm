"""ECHAM6.3's grid-dependent cloud defaults (``mo_echam_cloud_params.f90``).

ECHAM sets part of its cloud-cover and cloud-microphysics tunables per grid, in
``mo_echam_cloud_params.f90::sucloud`` (ECHAM6.3-HAM2.3 r7492, l.112-243):

* **by spectral truncation** (l.198-240): ``crs``, ``crt``, ``nex``, ``nadd``,
  ``csatsc``, ``cinv`` (read by ``mo_cover.f90::cover``) and ``cvtfall``,
  ``csecfrl``, ``clwprat`` (read by ``mo_cloud.f90::cloud``; ``csecfrl`` also
  by the saturation lookup that ``cover`` calls). ECHAM defines rows for T31,
  T63, T127 and T255 and stops with "Truncation not supported" at any other.
* **by vertical grid** (l.132-162): the inversion-search bounds ``jbmin`` and
  ``jbmax``, the first full levels (counting from the top) below 2000 m and
  below 500 m, with heights estimated as ``(p_s - p) / (g · 1.25 kg m-3)``
  from the level pressures at a fixed 101320 Pa surface pressure. These are
  grid geometry, not tunables: :func:`inversion_levels` computes them from the
  model's own levels, which gives ECHAM's 40/45 at L47 and 88/93 at L95.

:func:`echam_cloud_defaults` gives the per-truncation tunables through the
generic :func:`jcm.physics.resolution_defaults.resolution_defaults`: ECHAM's
values at T31, T63, T127 and T255, linear interpolation in the truncation
number between them (T106 lies 43/64 of the way from T63 to T127), and the
nearer ECHAM truncation's value for ``nex`` and ``nadd``, which are integers in
ECHAM. ECHAM itself has no configuration between its four truncations; the
interpolated values are jcm's choice and are untuned.
"""

from __future__ import annotations

import numpy as np

import jcm.constants as c
from jcm.physics.resolution_defaults import resolution_defaults

__all__ = [
    "CTHOMI_BELOW_TMELT",
    "ECHAM_CLOUD_DEFAULTS",
    "ECHAM_CLOUD_NEAREST_FIELDS",
    "INVERSION_REFERENCE_DENSITY",
    "INVERSION_REFERENCE_SURFACE_PRESSURE",
    "INVERSION_SEARCH_BOTTOM_M",
    "INVERSION_SEARCH_TOP_M",
    "echam_cloud_defaults",
    "inversion_levels",
    "inversion_levels_from_interfaces",
]

#: ``mo_echam_cloud_params.f90`` l.198-237: ECHAM's four truncations.
#:
#: ``crs``/``crt``: critical relative humidity at the surface / aloft;
#: ``nex``: exponent of the critical-RH profile (INTEGER in ECHAM);
#: ``nadd``: extra level below the inversion that is enhanced (INTEGER);
#: ``csatsc``: stratocumulus saturation factor at an inversion;
#: ``cinv``: stability threshold, fraction of the dry adiabatic lapse rate;
#: ``cvtfall``: ice fall-speed prefactor;
#: ``csecfrl``: cloud ice [kg/kg] above which ice saturation applies;
#: ``clwprat``: liquid-water-path ratio of the ``ktype`` 2 -> 4 re-typing.
ECHAM_CLOUD_DEFAULTS: dict[int, dict[str, float | int]] = {
    31: dict(crs=0.95, crt=0.85, nex=1, nadd=1, csatsc=0.1, cinv=0.5,
             cvtfall=3.0, csecfrl=5.0e-7, clwprat=0.0),
    63: dict(crs=0.975, crt=0.75, nex=2, nadd=0, csatsc=0.7, cinv=0.25,
             cvtfall=2.5, csecfrl=5.0e-6, clwprat=4.0),
    127: dict(crs=0.994, crt=0.75, nex=2, nadd=0, csatsc=0.7, cinv=0.25,
              cvtfall=3.0, csecfrl=1.0e-5, clwprat=4.0),
    255: dict(crs=0.994, crt=0.75, nex=2, nadd=0, csatsc=0.7, cinv=0.25,
              cvtfall=3.0, csecfrl=1.0e-5, clwprat=4.0),
}

#: Fields that are integers in ECHAM and are therefore never interpolated.
ECHAM_CLOUD_NEAREST_FIELDS = ("nex", "nadd")

#: ``cthomi = tmelt - 35`` (``mo_echam_cloud_params.f90`` l.54), the
#: homogeneous-freezing temperature, kept as an offset from ``tmelt``.
CTHOMI_BELOW_TMELT = 35.0

#: Constants of ECHAM's inversion-level rule (``sucloud`` l.137, 147, 154, 161).
INVERSION_REFERENCE_SURFACE_PRESSURE = 101320.0   # Pa
INVERSION_REFERENCE_DENSITY = 1.25                # kg m-3
INVERSION_SEARCH_TOP_M = 2000.0                   # jbmin: first level below
INVERSION_SEARCH_BOTTOM_M = 500.0                 # jbmax: first level below


def echam_cloud_defaults(truncation: int | None) -> dict[str, float | int]:
    """ECHAM's cloud tunables for a truncation, interpolated between its rows.

    Args:
        truncation: the triangular truncation (63 for T63), or ``None`` for a
            grid that is not spectral (which gets the T63 row, with a warning).

    Returns:
        ``{field: value}`` for ``crs``, ``crt``, ``nex``, ``nadd``, ``csatsc``,
        ``cinv``, ``cvtfall``, ``csecfrl`` and ``clwprat``. See the module
        docstring for the rule between and outside ECHAM's truncations.

    """
    return resolution_defaults(
        ECHAM_CLOUD_DEFAULTS, truncation,
        nearest=ECHAM_CLOUD_NEAREST_FIELDS, fallback=63,
        table_name="ECHAM cloud defaults (mo_echam_cloud_params.f90)")


def inversion_levels_from_interfaces(a_half, b_half) -> tuple[int, int]:
    """ECHAM's ``(jbmin, jbmax)`` for a hybrid grid, as 0-based indices.

    Implements ``sucloud`` l.134-162: interface pressures at a 101320 Pa
    surface, full levels as the interface mean, heights
    ``(p_s - p) / (g · 1.25)``, and the first full level from the top whose
    height is below 2000 m (``jbmin``) and below 500 m (``jbmax``). As in the
    Fortran loop, a grid with no level below a threshold returns the last
    level.

    Args:
        a_half: interface ``a`` coefficients [Pa], top first (length
            ``nlev + 1``).
        b_half: interface ``b`` coefficients [1], top first.

    Returns:
        ``(jbmin, jbmax)`` as 0-based, top-first level indices, i.e. ECHAM's
        1-based values minus one: ECHAM6.3 L47's ``(40, 45)`` is returned as
        ``(39, 44)``.

    """
    a_half = np.asarray(a_half, dtype=np.float64)
    b_half = np.asarray(b_half, dtype=np.float64)
    p_half = a_half + b_half * INVERSION_REFERENCE_SURFACE_PRESSURE
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    height = (p_half[-1] - p_full) / (c.grav * INVERSION_REFERENCE_DENSITY)
    nlev = p_full.shape[0]

    def first_below(threshold: float) -> int:
        below = np.nonzero(height < threshold)[0]
        return int(below[0]) if below.size else nlev - 1

    return (first_below(INVERSION_SEARCH_TOP_M),
            first_below(INVERSION_SEARCH_BOTTOM_M))


def inversion_levels(coords_or_vertical) -> tuple[int, int]:
    """ECHAM's ``(jbmin, jbmax)`` for a coordinate system or vertical grid.

    Accepts a ``CoordinateSystem`` (its ``.vertical`` is used), a
    ``HybridCoordinates`` (``a`` in Pa, as jcm's ECHAM tables are) or a
    ``SigmaCoordinates`` (``a = 0``). See
    :func:`inversion_levels_from_interfaces`.
    """
    vertical = getattr(coords_or_vertical, "vertical", coords_or_vertical)
    if hasattr(vertical, "a_boundaries"):
        return inversion_levels_from_interfaces(
            vertical.a_boundaries, vertical.b_boundaries)
    sigma = np.asarray(vertical.boundaries)
    return inversion_levels_from_interfaces(np.zeros_like(sigma), sigma)
