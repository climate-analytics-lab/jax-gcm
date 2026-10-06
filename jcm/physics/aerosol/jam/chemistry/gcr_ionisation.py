"""Galactic cosmic ray ionisation, host-side (jax-gcm#1017 Kazil/GCR task, Part B).

Full port of ``mo_ham_gcrion.f90`` (Kazil and Lovejoy's M7 scheme needs this
as ``pipr``) — the module HAM computes it in, ``ham_subm_interface``
(``mo_ham_subm.f90``), is on the Fortran side of the jcm/HAM boundary, so this
is **host-side infrastructure**, not a port of that interface itself:

* :func:`read_obrien_gcr_ipr` — the O'Brien solar-min/max ion-pair-production
  lookup tables, in the EXACT blocked text format ``read_obrien_gcr_ipr``
  (``mo_ham_gcrion.f90:120-206``) reads.
* :func:`solar_activity` — the cosine solar-cycle parameterisation
  (``mo_ham_gcrion.f90:526-557``), a closed form of the calendar date.
* :func:`gcr_ion_pair_rate` — ``gcr_ionization``/``gcr_ionization_profile``
  (``mo_ham_gcrion.f90:226-522``): geomagnetic latitude (via the IGRF dipole
  axis and ``mo_geopack.f90``'s ``geo2mag``), vertical cutoff rigidity, the
  O'Brien table's 2-D (cutoff rigidity, mass column density) interpolation,
  and the local-condition (pressure, temperature) rescaling.

The IGRF dipole axis (needed for geomagnetic latitude, hence cutoff rigidity)
secularly drifts: ``mo_geopack.f90``'s ``recalc`` recomputes it from the
current (year, day-of-year) every call. Only its degree-1 (dipole) terms
feed the rotation this module needs — see :data:`_IGRF_EPOCH_YEARS` and
:func:`_dipole_axis_coefficients` for the reduction. ``gcr_ion_pair_rate``
therefore needs ``year``/``day_of_year`` in addition to the
``(lat, lon, pressure, temperature, solar_activity, tables)`` the task
sketch names: the dipole axis is genuinely date-dependent (unlike
``solar_activity``, computed upstream and passed in as a scalar, these two
are NOT optional — the task's prose signature omits them, a decision
resolved here rather than silently; see the module docstring note in
``gcr_ionisation_test.py`` and the W3 report to the lead). Over a single
model run the dipole axis moves only a fraction of a degree, but jcm follows
the Fortran faithfully rather than freezing it at a reference epoch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import jcm.constants as c

# --- mo_ham_gcrion.f90::gcr_initialize ------------------------------------
_VCR_N = 15
_MCD_N = 110


class ObrienGcrTable(NamedTuple):
    """O'Brien solar-min/max galactic-cosmic-ray ion-pair-production table.

    ``vertical_cutoff_rigidity`` [GV] (length 15, ascending) and
    ``mass_column_density`` [g cm⁻²] (length 110, ascending) are the two
    lookup axes; ``ipr_solmin``/``ipr_solmax`` [ion pairs cm⁻³ s⁻¹ @ 273.15 K,
    1013.25 hPa] are shaped ``(15, 110)`` (same axis order as the Fortran's
    own ``gcr_ipr_solmin_table(vcr_n,mcd_n)``/``gcr_ipr_solmax_table``).
    """

    vertical_cutoff_rigidity: object
    mass_column_density: object
    ipr_solmin: object
    ipr_solmax: object


# ECHAM opens these names directly; README_GCR.txt (the staged distribution)
# names them differently again. Accept every name jax-gcm#1017's coordinator
# has confirmed is in play: ECHAM's own `gcr_ipr_solmin.txt`/
# `gcr_ipr_solmax.txt` (symlinked there from `hammoz/solmin.txt`/`solmax.txt`
# in the staged distribution) and README_GCR.txt's own `SOLMIN.txt`/
# `SOLMAX.txt`. SOLINT.txt (785 MV, a third O'Brien file) is NOT read by
# `read_obrien_gcr_ipr` and is intentionally not looked for here.
_SOLMIN_NAMES = ("gcr_ipr_solmin.txt", "solmin.txt", "SOLMIN.txt")
_SOLMAX_NAMES = ("gcr_ipr_solmax.txt", "solmax.txt", "SOLMAX.txt")


def _find(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of {names} found in {directory} (HAM_INPUT_DIR); "
        "read_obrien_gcr_ipr needs the O'Brien solar-min/max GCR tables."
    )


def _read_obrien_block(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One O'Brien file: 15 blocks of (header line, 'F11.6,A' vcr line, two
    skipped lines, 110 'mcd ipr' data lines), after one skipped header line
    -- the exact structure ``read_obrien_gcr_ipr`` (mo_ham_gcrion.f90:120-206)
    reads with Fortran list-directed/``F11.6`` formatted reads. Returns
    ``(vcr[15], mcd[110], ipr[15,110])``; ``mcd`` is read once (every block
    uses the same mass-column-density axis, as the Fortran's single
    ``mcd_table`` — shared across blocks — assumes) and checked identical
    across blocks.
    """
    lines = path.read_text().splitlines()
    pos = 1  # skip the file's own header line (READ(iunit,*))
    vcr = np.empty(_VCR_N, dtype=np.float64)
    mcd = None
    ipr = np.empty((_VCR_N, _MCD_N), dtype=np.float64)
    for block in range(_VCR_N):
        pos += 1  # skipped line before the vcr line
        # Fortran's F-edit descriptor accepts a full real literal on INPUT
        # (even though it only ever OUTPUTS plain fixed-point) -- the staged
        # files write the cutoff rigidity in 'D'-exponent form, e.g.
        # '1.495330D0'. Match the full literal including the exponent and
        # normalise D/d to E before float() (Python's float() does not
        # accept Fortran's D-exponent spelling).
        match = re.match(r"\s*([+-]?\d*\.?\d+(?:[DdEe][+-]?\d+)?)", lines[pos])
        if match is None:
            raise ValueError(f"{path}: expected an F11.6 cutoff-rigidity value at line {pos + 1}")
        vcr[block] = float(match.group(1).replace("D", "E").replace("d", "e"))
        pos += 1
        pos += 2  # two skipped lines after the vcr line
        block_mcd = np.empty(_MCD_N, dtype=np.float64)
        for row in range(_MCD_N):
            parts = lines[pos].split()
            block_mcd[row] = float(parts[0])
            ipr[block, row] = float(parts[1])
            pos += 1
        if mcd is None:
            mcd = block_mcd
        elif not np.array_equal(mcd, block_mcd):
            raise ValueError(
                f"{path}: mass-column-density axis differs between cutoff-rigidity "
                f"blocks (block {block}) -- read_obrien_gcr_ipr assumes one shared axis"
            )
    return vcr, mcd, ipr


def read_obrien_gcr_ipr(directory) -> ObrienGcrTable:
    """Read the O'Brien GCR ion-pair-production tables outside JIT.

    ``directory`` is ``HAM_INPUT_DIR``; see the module-level name lists for
    every accepted filename. Both axes are validated finite and strictly
    increasing (:func:`gcr_ion_pair_rate`'s bisection search assumes it).
    """
    directory = Path(directory)
    vcr_min, mcd_min, ipr_min = _read_obrien_block(_find(directory, _SOLMIN_NAMES))
    vcr_max, mcd_max, ipr_max = _read_obrien_block(_find(directory, _SOLMAX_NAMES))
    if not np.array_equal(vcr_min, vcr_max) or not np.array_equal(mcd_min, mcd_max):
        raise ValueError(
            "read_obrien_gcr_ipr: solar-min and solar-max files disagree on their "
            "cutoff-rigidity/mass-column-density axes"
        )
    for axis, label in ((vcr_min, "vertical_cutoff_rigidity"), (mcd_min, "mass_column_density")):
        if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f"read_obrien_gcr_ipr: {label} axis is non-finite or not strictly increasing")
    if not np.isfinite(ipr_min).all() or not np.isfinite(ipr_max).all():
        raise ValueError("read_obrien_gcr_ipr: non-finite ion-pair-production rate")
    return ObrienGcrTable(*(jnp.asarray(x) for x in (vcr_min, mcd_min, ipr_min, ipr_max)))


# --- mo_ham_gcrion.f90::solar_activity -------------------------------------

def solar_activity(year, tyear):
    """Cosine solar-activity parameterisation (``mo_ham_gcrion.f90:526-557``).

    +1 at solar maximum (1 January 1991), -1 at solar minimum (2 July 1996),
    an 11-year cycle. ``year`` is the calendar year; ``tyear`` is the 0-1
    fraction of THAT year elapsed -- the native ``(zdoy-1)/get_year_len(iyr)``
    with ``zdoy = get_year_day(current_date)`` (1-based, continuous: unlike
    :func:`gcr_ion_pair_rate`'s ``day_of_year``, this is NOT truncated to an
    integer day, i.e. ``solar_activity`` resolves sub-day time). jcm's own
    ``DateData.tyear()`` (``jcm/date.py``) already computes exactly this
    quantity ((0-based day-of-year + fraction-of-day) / days-in-year, with
    the ACTUAL calendar days-in-year, i.e. 365 or 366 -- matching
    ``get_year_len``, not a fixed 365.25) -- the adapter passes it straight
    through.
    """
    ztime = year + tyear
    return jnp.cos(2.0 * jnp.pi * (1991.0 - ztime) / 11.0)


# --- mo_geopack.f90: IGRF dipole axis -> geomagnetic coordinates ----------

# Degree-1 (dipole) IGRF coefficients at each 5-year epoch (mo_geopack.f90's
# g65..g05/h65..h05 DATA statements, index 2 = g(n=1,m=0) and index 3 =
# g(n=1,m=1)/h(n=1,m=1)) plus their 2005-epoch secular rates (dg05/dh05,
# same indices) used to extrapolate beyond 2005. These are the ONLY entries
# of the full IGRF g/h arrays geo2mag's rotation needs (mo_geopack.f90:
# 1147-1165): every other coefficient feeds field-MAGNITUDE calculations
# (IGRF/DIP/TRACE) this module never calls. The Schmidt normalisation loop
# that follows the epoch interpolation in `recalc` starts at n=2
# (mo_geopack.f90 CONSTRUCT_2's `DO n=2,14`), so these degree-1 raw values
# are used UNNORMALISED, exactly as read from the tables below.
#
# Plain Python tuples, NOT module-level jnp.array literals: a jnp.array
# created here would freeze at whatever float32/float64 jax_enable_x64 was
# set to the first time this module is imported (almost always float32,
# since most of jcm runs there by default) and NEVER update afterwards --
# jax.enable_x64(True) later in a caller/test would silently keep reading
# that stale float32 array by reference. (Found via this module's own
# Fortran-reference test: a ~1e-7 relative, otherwise-unexplained mismatch
# that vanished once these became plain tuples converted at call time,
# below -- the exact failure mode jcm.constants' own "derived quantities
# are properties, not module constants" rule (see CLAUDE.md) exists to
# prevent, here hitting a literal table instead of a derived constant.)
_IGRF_EPOCH_YEARS = (1965., 1970., 1975., 1980., 1985., 1990., 1995., 2000., 2005.)
_IGRF_G2 = (-30334., -30220., -30100., -29992., -29873., -29775., -29692., -29619.4, -29556.8)
_IGRF_G3 = (-2119., -2068., -2013., -1956., -1905., -1848., -1784., -1728.2, -1671.8)
_IGRF_H3 = (5776., 5737., 5675., 5604., 5500., 5406., 5306., 5186.1, 5080.0)
# 2005-epoch secular rates [per year] (dg05(2), dg05(3), dh05(3)).
_IGRF_DG2_2005, _IGRF_DG3_2005, _IGRF_DH3_2005 = 8.8, 10.8, -21.3


def _dipole_axis_coefficients(year, day_of_year):
    """``(st0, ct0, sl0, cl0)`` — ``mo_geopack.f90::recalc``'s dipole-axis
    rotation coefficients (lines 1147-1165), reduced to only the degree-1
    IGRF interpolation/extrapolation this needs. ``year``/``day_of_year``
    match ``recalc``'s own ``iyear``/``iday`` (1-based day of year); ``year``
    is clamped to [1965, 2010] exactly as the Fortran clamps (a warning
    there, silent here -- the clamp is a pre-existing IGRF-table limitation,
    not something introduced by this port).
    """
    # Converted from the plain-tuple module constants at call time, so the
    # dtype tracks whatever jax_enable_x64 is active NOW (see the tuples'
    # own comment above) rather than whatever was active at import time.
    epoch_years = jnp.asarray(_IGRF_EPOCH_YEARS, jnp.float64)
    g2_table = jnp.asarray(_IGRF_G2, jnp.float64)
    g3_table = jnp.asarray(_IGRF_G3, jnp.float64)
    h3_table = jnp.asarray(_IGRF_H3, jnp.float64)

    iy = jnp.clip(jnp.asarray(year, jnp.float64), 1965.0, 2010.0)
    t = iy + (jnp.asarray(day_of_year, jnp.float64) - 1.0) / 365.25
    bucket = jnp.clip(jnp.floor((iy - 1965.0) / 5.0), 0, 7).astype(jnp.int32)
    f2 = (t - epoch_years[bucket]) / 5.0
    f1 = 1.0 - f2
    g2_interp = g2_table[bucket] * f1 + g2_table[bucket + 1] * f2
    g3_interp = g3_table[bucket] * f1 + g3_table[bucket + 1] * f2
    h3_interp = h3_table[bucket] * f1 + h3_table[bucket + 1] * f2
    extrapolate = iy >= 2005.0
    dt = t - 2005.0
    g2 = jnp.where(extrapolate, g2_table[-1] + _IGRF_DG2_2005 * dt, g2_interp)
    g3 = jnp.where(extrapolate, g3_table[-1] + _IGRF_DG3_2005 * dt, g3_interp)
    h3 = jnp.where(extrapolate, h3_table[-1] + _IGRF_DH3_2005 * dt, h3_interp)

    g10 = -g2
    g11 = g3
    h11 = h3
    sq = g11 * g11 + h11 * h11
    sqq = jnp.sqrt(sq)
    sqr = jnp.sqrt(g10 * g10 + sq)
    sl0 = -h11 / sqq
    cl0 = -g11 / sqq
    st0 = sqq / sqr
    ct0 = g10 / sqr
    return st0, ct0, sl0, cl0


def geo2mag(lat_deg, lon_deg, year, day_of_year):
    """Geographic to geomagnetic dipole coordinates (``mo_geopack.f90::
    geo2mag``, lines 107-161), via ``sphcar``'s spherical<->Cartesian
    transform and ``geomag``'s rotation by the dipole-axis coefficients.
    Degrees in, degrees out; broadcasts over ``lat_deg``/``lon_deg``.
    """
    st0, ct0, sl0, cl0 = _dipole_axis_coefficients(year, day_of_year)
    stcl, stsl, ctsl, ctcl = st0 * cl0, st0 * sl0, ct0 * sl0, ct0 * cl0

    theta = jnp.pi * (0.5 - jnp.asarray(lat_deg, jnp.float64) / 180.0)
    phi = jnp.pi * jnp.asarray(lon_deg, jnp.float64) / 180.0
    # sphcar(r=1, theta, phi, ->, j=1).
    sq = jnp.sin(theta)
    xgeo = sq * jnp.cos(phi)
    ygeo = sq * jnp.sin(phi)
    zgeo = jnp.cos(theta)
    # geomag(xgeo, ygeo, zgeo, ->, j=1).
    xmag = xgeo * ctcl + ygeo * ctsl - zgeo * st0
    ymag = ygeo * cl0 - xgeo * sl0
    zmag = xgeo * stcl + ygeo * stsl + zgeo * ct0
    # sphcar(->, theta, phi, xmag, ymag, zmag, j=-1).
    sq2 = xmag * xmag + ymag * ymag
    on_axis = sq2 == 0.0
    sq2_safe = jnp.where(on_axis, 1.0, sq2)
    mag_phi = jnp.where(
        on_axis, 0.0,
        jnp.where(jnp.arctan2(ymag, xmag) < 0.0,
                  jnp.arctan2(ymag, xmag) + 2.0 * jnp.pi,
                  jnp.arctan2(ymag, xmag)),
    )
    mag_theta = jnp.where(
        on_axis, jnp.where(zmag < 0.0, jnp.pi, 0.0),
        jnp.arctan2(jnp.sqrt(sq2_safe), zmag),
    )
    mag_lat = 180.0 * (0.5 - mag_theta / jnp.pi)
    mag_lon = 180.0 * mag_phi / jnp.pi
    return mag_lat, mag_lon


def vertical_cutoff_rigidity(mag_lat_deg):
    """``mo_geopack.f90::vertical_cutoff_rigidity`` [GV]."""
    return 14.9 * jnp.cos(jnp.pi * jnp.asarray(mag_lat_deg, jnp.float64) / 180.0) ** 4.0


# --- mo_ham_gcrion.f90::gcr_ionization/gcr_ionization_profile --------------

def _bracket(table_axis, value):
    """0-based bracketing indices for a strictly increasing table (shared
    reasoning with :func:`jcm...aqueous` / M7-JAX's own ``_bracket`` for
    ``nucl_kazil_lovejoy``'s identically-structured bisection search --
    see that function's docstring for the proof this is the exact
    vectorized equivalent of a bisection search over a pre-clamped value).
    """
    n = table_axis.shape[0]
    i0 = jnp.clip(jnp.searchsorted(table_axis, value, side="right") - 1, 0, n - 2)
    return i0, i0 + 1


def gcr_ion_pair_rate(lat, lon, pressure, temperature, solar_activity, tables,
                       *, year, day_of_year, grav=None):
    """GCR ion-pair production rate [cm⁻³ s⁻¹] (``gcr_ionization``/
    ``gcr_ionization_profile``, mo_ham_gcrion.f90:226-522).

    Broadcasting-native: ``pressure``/``temperature`` are ``(kx, *horiz)``
    [Pa]/[K]; ``lat``/``lon`` are ``(*horiz)`` [radians, jcm's own
    convention -- converted to degrees internally for ``geo2mag``];
    ``solar_activity`` is the scalar [-1, 1] activity parameter (see
    :func:`solar_activity`); ``tables`` is an :class:`ObrienGcrTable`.
    ``year``/``day_of_year`` are the current model date (see the module
    docstring for why ``gcr_ion_pair_rate`` needs them beyond the task
    sketch's literal signature).

    ``grav`` (``None`` by default, reading jcm's own ``c.grav`` dynamically
    -- #772's anti-staleness pattern, see ``chemistry/aqueous.py``'s
    ``_zrgas``/``_avo_xtoc`` for the precedent) is the mass-column-density
    conversion's gravitational acceleration. ``mo_physical_constants.f90``
    hardcodes ``grav = 9.80665``, which differs from jcm's own ``c.grav``
    (9.81) at the ~3e-4 relative level -- utterly immaterial physically, but
    enough to fail a literal Fortran-parity test at the task's 1e-12 gate.
    An explicit ``grav=9.80665`` lets the (not yet written) reference test
    reproduce the Fortran exactly without changing jcm's own global gravity
    constant, which is used far outside this one aerosol diagnostic.
    """
    lat_deg = jnp.asarray(lat, jnp.float64) * (180.0 / jnp.pi)
    lon_deg = jnp.asarray(lon, jnp.float64) * (180.0 / jnp.pi)
    mag_lat, _ = geo2mag(lat_deg, lon_deg, year, day_of_year)
    zvcr = vertical_cutoff_rigidity(mag_lat)                      # (*horiz)

    vcr_table = tables.vertical_cutoff_rigidity
    mcd_table = tables.mass_column_density
    zvcr = jnp.clip(zvcr, vcr_table[0], vcr_table[-1])
    ivcr0, ivcr1 = _bracket(vcr_table, zvcr)                      # (*horiz)
    zv = (zvcr - vcr_table[ivcr0]) / (vcr_table[ivcr1] - vcr_table[ivcr0])

    press_hpa = jnp.asarray(pressure, jnp.float64) * 0.01         # (kx, *horiz)
    ptemp = jnp.asarray(temperature, jnp.float64)                 # (kx, *horiz)
    g = c.grav if grav is None else grav
    zmcd = 10.0 * press_hpa / g                                    # g cm-2, (kx, *horiz)
    zmcd = jnp.clip(zmcd, mcd_table[0], mcd_table[-1])
    imcd0, imcd1 = _bracket(mcd_table, zmcd)                      # (kx, *horiz)
    zw = (zmcd - mcd_table[imcd0]) / (mcd_table[imcd1] - mcd_table[imcd0])

    # Broadcast the horizontal-only cutoff-rigidity bracket/weight up to the
    # vertical axis the mass-column-density bracket already carries.
    ivcr0 = jnp.broadcast_to(ivcr0, zmcd.shape)
    ivcr1 = jnp.broadcast_to(ivcr1, zmcd.shape)
    zv = jnp.broadcast_to(zv, zmcd.shape)

    def gather(table):
        t00 = table[ivcr0, imcd0]
        t01 = table[ivcr0, imcd1]
        t10 = table[ivcr1, imcd0]
        t11 = table[ivcr1, imcd1]
        # Literal transcription of gcr_ionization_profile:499-507 (algebraically
        # standard bilinear interpolation, kept in this grouping to match the
        # Fortran's own floating-point evaluation order).
        return ((zv - 1.0) * (zw - 1.0) * t00 + (zw - zv * zw) * t01
                + (zv - zv * zw) * t10 + zv * zw * t11)

    ipr_solmin = gather(tables.ipr_solmin)
    ipr_solmax = gather(tables.ipr_solmax)

    psolact = jnp.asarray(solar_activity, jnp.float64)
    pgcripr = 0.5 * ((1.0 - psolact) * ipr_solmin + (1.0 + psolact) * ipr_solmax)
    # Local-condition rescaling (gcr_ionization_profile:520): normal
    # conditions are 273.15 K, 1013.25 hPa.
    pgcripr = pgcripr * press_hpa / 1013.25 * 273.15 / ptemp
    return pgcripr
