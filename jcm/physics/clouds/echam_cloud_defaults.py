"""Per-truncation defaults of the ECHAM cloud schemes' tunables.

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

Two tables hold the values, so that ECHAM's own numbers stay on record next to
the ones jcm ships:

* :data:`ECHAM_CLOUD_DEFAULTS` is ECHAM's table, row for row. Its test pins it
  against the Fortran source, and the Fortran comparison runs on the constants
  its reference data recorded.
* :data:`JCM_CLOUD_DEFAULTS` is the table jcm reads: ECHAM's rows, with the
  five cloud-cover fields of the T63 row replaced by jcm's calibrated set
  (:data:`JCM_CALIBRATED_COVER_T63`). Every other field, and every other row,
  is ECHAM's and is uncalibrated.

:func:`echam_cloud_defaults` gives the per-truncation tunables of
:data:`JCM_CLOUD_DEFAULTS` through the generic
:func:`jcm.physics.resolution_defaults.resolution_defaults`: the table's values
at T31, T63, T127 and T255, linear interpolation in the truncation number
between them (T106 lies 43/64 of the way from the calibrated T63 row to ECHAM's
T127 row), and the nearer tabulated truncation's value for ``nex`` and
``nadd``, which are integers in ECHAM. ECHAM itself has no configuration
between its four truncations; the interpolated values are jcm's choice and are
untuned. A grid without a spectral truncation (the pySES cubed sphere) takes
the T63 row, calibrated cover fields included.
"""

from __future__ import annotations

import numpy as np

import jcm.constants as c
from jcm.physics.resolution_defaults import resolution_defaults

__all__ = [
    "CTHOMI_BELOW_TMELT",
    "ECHAM_CLOUD_DEFAULTS",
    "ECHAM_CLOUD_NEAREST_FIELDS",
    "JCM_CALIBRATED_COVER_T63",
    "JCM_CLOUD_DEFAULTS",
    "INVERSION_REFERENCE_DENSITY",
    "INVERSION_REFERENCE_SURFACE_PRESSURE",
    "INVERSION_SEARCH_BOTTOM_M",
    "INVERSION_SEARCH_TOP_M",
    "echam_cloud_defaults",
    "inversion_levels",
    "inversion_levels_from_interfaces",
]

#: ``mo_echam_cloud_params.f90`` l.198-237: ECHAM's four truncations, as ECHAM
#: has them (uncalibrated; jcm's calibrated T63 cover fields are
#: :data:`JCM_CALIBRATED_COVER_T63`, laid over this table in
#: :data:`JCM_CLOUD_DEFAULTS`).
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

#: jcm's calibrated cloud-cover fields at T63, laid over ECHAM's T63 row.
#:
#: Provenance: Stage 2b of the v3 release calibration
#: (``docs/source/design/jam_aerosol_retune.md``, "Stage 2b: cloud fraction").
#: A 25-arm Gaussian-process expected-improvement search over exactly these
#: five fields on the 2M host (30-day January and July windows, the eight-target
#: loss against the jcm-monitor climatologies), then 365-day confirmation years.
#: The set is the best of the 25 arms among those with at most one parameter
#: within 5 % of its range of a bound of the searched box (every lower-loss arm
#: has two or more): interior in ``crt``, ``nex``, ``csatsc`` and ``cinv``,
#: while ``crs`` sits on the box's lower bound (0.90) and is the one field the
#: data do not bound from below. The sweep's final best arm sat on the box edges (``nex`` 3.96 of 4,
#: ``cinv`` 0.5 of 0.5, ``crs`` 0.9) and was not adopted: an optimum on the
#: edge of the searched box is where the search ran out of room, not a value
#: the data have located.
#:
#: ``nex`` is declared INTEGER in ECHAM, but nothing in the cover needs an
#: integer. ``mo_cover.f90`` l.233 evaluates the critical relative humidity
#: ``crt + (crs - crt)·exp(1 - (p_s/p)^nex)`` with a power whose base is
#: ``p_s/p >= 1``, so a real exponent gives a continuous, differentiable
#: profile that equals ``crs`` at the surface and tends to ``crt`` aloft for
#: every ``nex > 0``. jcm already holds ``nex`` as a real, differentiable
#: leaf (:class:`jcm.physics.clouds.sundqvist.CloudParameters`).
#:
#: The same five fields are read by the cover of the 1M, the 2M and the JAM-2M
#: hosts (``SundqvistCloudFraction``), so one set serves all three. It was
#: calibrated and confirmed on the 2M host; the 1M host inherits it without a
#: calibration of its own.
JCM_CALIBRATED_COVER_T63: dict[str, float] = dict(
    crt=0.679016061, crs=0.9, nex=1.84856084, csatsc=0.948216414,
    cinv=0.213005383)

#: The table jcm reads: ECHAM's rows, T63's five cover fields calibrated. The
#: T31, T127 and T255 rows are ECHAM's values and are uncalibrated, and so are
#: the T63 fields outside :data:`JCM_CALIBRATED_COVER_T63` (``nadd``,
#: ``cvtfall``, ``csecfrl``, ``clwprat``).
JCM_CLOUD_DEFAULTS: dict[int, dict[str, float | int]] = {
    truncation: dict(row) for truncation, row in ECHAM_CLOUD_DEFAULTS.items()}
JCM_CLOUD_DEFAULTS[63].update(JCM_CALIBRATED_COVER_T63)

#: Fields that are integers in ECHAM and are therefore never interpolated:
#: between two tabulated truncations they take the nearer one's value. ``nex``
#: stays here although the calibrated T63 value is real: T64 to T94 take the
#: calibrated T63 ``nex`` and T95 to T126 take ECHAM's T127 value.
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
    """Return the cloud tunables for a truncation, interpolated between rows.

    Args:
        truncation: the triangular truncation (63 for T63), or ``None`` for a
            grid that is not spectral (which gets the T63 row, with a warning).

    Returns:
        ``{field: value}`` for ``crs``, ``crt``, ``nex``, ``nadd``, ``csatsc``,
        ``cinv``, ``cvtfall``, ``csecfrl`` and ``clwprat``, from
        :data:`JCM_CLOUD_DEFAULTS` (ECHAM's rows with jcm's calibrated T63
        cover fields). See the module docstring for the rule between and
        outside the tabulated truncations.

    """
    return resolution_defaults(
        JCM_CLOUD_DEFAULTS, truncation,
        nearest=ECHAM_CLOUD_NEAREST_FIELDS, fallback=63,
        table_name="cloud defaults (mo_echam_cloud_params.f90 and jcm's T63 calibration)")


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
