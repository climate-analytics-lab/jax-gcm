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

Three tables hold the values, so that ECHAM's own numbers stay on record next
to the ones jcm ships:

* :data:`ECHAM_CLOUD_DEFAULTS` is ECHAM's table, row for row. Its test pins it
  against the Fortran source, and the Fortran comparison runs on the constants
  its reference data recorded.
* :data:`JCM_CLOUD_DEFAULTS` and :data:`JCM_CLOUD_DEFAULTS_JAM` are the tables
  jcm reads: ECHAM's rows, with the five cloud-cover fields of the T63 row
  replaced by a calibrated set. The cover is calibrated per droplet-number
  source: :data:`JCM_CALIBRATED_COVER_T63` for the hosts whose droplet number
  is prescribed (1M, and 2M with simple-plume activation), and
  :data:`JCM_CALIBRATED_COVER_T63_JAM` for the hosts whose droplet number
  comes from activating the prognostic JAM aerosol. Every other field, and
  every other row, is ECHAM's and is uncalibrated.

:func:`echam_cloud_defaults` gives the per-truncation tunables of either table
through the generic
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
    "JCM_CALIBRATED_COVER_T63_JAM",
    "JCM_CLOUD_DEFAULTS",
    "JCM_CLOUD_DEFAULTS_JAM",
    "INVERSION_REFERENCE_DENSITY",
    "INVERSION_REFERENCE_SURFACE_PRESSURE",
    "INVERSION_SEARCH_BOTTOM_M",
    "INVERSION_SEARCH_TOP_M",
    "echam_cloud_defaults",
    "inversion_levels",
    "inversion_levels_from_interfaces",
]

#: ``mo_echam_cloud_params.f90`` l.198-237: ECHAM's four truncations, as ECHAM
#: has them (uncalibrated; jcm's calibrated T63 cover fields are laid over this
#: table in :data:`JCM_CLOUD_DEFAULTS` and :data:`JCM_CLOUD_DEFAULTS_JAM`).
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

#: jcm's calibrated cloud-cover fields at T63 for the hosts whose droplet
#: number is prescribed: the 2M host (simple-plume activation) and the 1M host.
#: Fitted on the 2M host with its ``ccraut``, against the release loss
#: (``docs/source/design/jam_aerosol_retune.md``, "The v3.1 refit"). The 1M
#: host, T106 and the cubed sphere read the set without a calibration of their
#: own (#1014).
#:
#: ``nex`` is declared INTEGER in ECHAM, but ``mo_cover.f90`` l.233 evaluates
#: ``crt + (crs - crt)·exp(1 - (p_s/p)^nex)`` with ``p_s/p >= 1``, so any real
#: ``nex > 0`` gives a continuous profile from ``crs`` at the surface to ``crt``
#: aloft; jcm holds it as a real, differentiable leaf.
JCM_CALIBRATED_COVER_T63: dict[str, float] = dict(
    crt=0.677, crs=0.908, nex=1.01, csatsc=0.737, cinv=0.275)

#: The same fields for the hosts whose droplet number comes from activating the
#: prognostic JAM aerosol (MAM4 and M7). Activated droplets make wetter,
#: brighter cloud than the 2M host's prescribed number at the same cover, so
#: the set is fitted on those hosts, jointly with their aerosol scales (design
#: page, "The v3.1 refit").
JCM_CALIBRATED_COVER_T63_JAM: dict[str, float] = dict(
    crt=0.727, crs=0.930, nex=1.14, csatsc=0.848, cinv=0.265)


def _with_calibrated_cover(cover: dict[str, float]
                           ) -> dict[int, dict[str, float | int]]:
    table = {truncation: dict(row)
             for truncation, row in ECHAM_CLOUD_DEFAULTS.items()}
    table[63].update(cover)
    return table


#: The tables jcm reads: ECHAM's rows with T63's five cover fields calibrated,
#: for the prescribed-droplet hosts and for the JAM hosts. The T31, T127 and
#: T255 rows are ECHAM's values and are uncalibrated, and so are the T63 fields
#: outside the cover (``nadd``, ``cvtfall``, ``csecfrl``, ``clwprat``).
JCM_CLOUD_DEFAULTS = _with_calibrated_cover(JCM_CALIBRATED_COVER_T63)
JCM_CLOUD_DEFAULTS_JAM = _with_calibrated_cover(JCM_CALIBRATED_COVER_T63_JAM)

#: Fields that are integers in ECHAM and are therefore never interpolated:
#: between two tabulated truncations they take the nearer one's value. ``nex``
#: is one of them although the calibrated T63 value is real, because ECHAM
#: defines it only at its own truncations: T64 to T94 take the calibrated T63
#: ``nex`` and T95 to T126 take ECHAM's T127 value.
ECHAM_CLOUD_NEAREST_FIELDS = ("nex", "nadd")

#: ``cthomi = tmelt - 35`` (``mo_echam_cloud_params.f90`` l.54), the
#: homogeneous-freezing temperature, kept as an offset from ``tmelt``.
CTHOMI_BELOW_TMELT = 35.0

#: Constants of ECHAM's inversion-level rule (``sucloud`` l.137, 147, 154, 161).
INVERSION_REFERENCE_SURFACE_PRESSURE = 101320.0   # Pa
INVERSION_REFERENCE_DENSITY = 1.25                # kg m-3
INVERSION_SEARCH_TOP_M = 2000.0                   # jbmin: first level below
INVERSION_SEARCH_BOTTOM_M = 500.0                 # jbmax: first level below


def echam_cloud_defaults(truncation: int | None, *,
                         jam_aerosol: bool = False) -> dict[str, float | int]:
    """Return the cloud tunables for a truncation, interpolated between rows.

    Args:
        truncation: the triangular truncation (63 for T63), or ``None`` for a
            grid that is not spectral (which gets the T63 row, with a warning).
        jam_aerosol: read :data:`JCM_CLOUD_DEFAULTS_JAM`, the cover calibrated
            for droplet number activated from the JAM aerosol, instead of
            :data:`JCM_CLOUD_DEFAULTS`. Only the five cover fields differ.

    Returns:
        ``{field: value}`` for ``crs``, ``crt``, ``nex``, ``nadd``, ``csatsc``,
        ``cinv``, ``cvtfall``, ``csecfrl`` and ``clwprat`` (ECHAM's rows with
        jcm's calibrated T63 cover fields). See the module docstring for the
        rule between and outside the tabulated truncations.

    """
    return resolution_defaults(
        JCM_CLOUD_DEFAULTS_JAM if jam_aerosol else JCM_CLOUD_DEFAULTS,
        truncation, nearest=ECHAM_CLOUD_NEAREST_FIELDS, fallback=63,
        table_name="cloud defaults (mo_echam_cloud_params.f90, jcm's T63 cover)")


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
