"""Gong (2003) and Long et al. (2011) sea-salt emission schemes.

Faithful ports of HAMMOZ ``mo_ham_m7_emi_seasalt::seasalt_emissions_gong``
(``nseasalt=6``, the default) and ``::seasalt_emissions_long``
(``nseasalt=7``), selected on :class:`SeaSaltEmissions` by ``scheme``.

Gong's source function gives a number flux per dry size bin that factorises
as ``f_i = (size-only factor) · u10**3.41``; the size-only factor is
wind-independent, so the per-mode mass and number fluxes collapse to **two
precomputed constants per mode** (accumulation, coarse) times ``u10**3.41``
times the open-water fraction. That makes the term cheap, jittable and
differentiable.

Long's source additionally depends on SST through the Sofiev et al. (2011)
correction, and that correction's *shape* varies with particle size, so
(unlike Gong) it does not factorise into wind-independent per-class
constants: every bin's correction must be evaluated against the actual SST
field, so the per-bin size/SST arrays are precomputed at construction and
the ``(nbin, ncols)`` evaluation happens at call time. Long's own AS/CS
split is HAM's fixed diameter bins (``_DBEG``/``_DEND``), not a population-
derived one: the population must therefore carry exactly two ``ss``
classes, in HAM's own accumulation/coarse order.

The source functions are HAM's; the shipped overall scale
(:data:`SEASALT_SCALE_DEFAULT`) is a calibration of 4 on top of them for the
MAM4 presets.

References:
  Gong, S. L. (2003), A parameterization of sea-salt aerosol source function
  for sub- and super-micron particles, Global Biogeochem. Cycles 17(4), 1097.
  Monahan et al. (1986), Oceanic Whitecaps.
  Long, M. S., W. D. Keene, D. J. Kieber, D. J. Erickson, H. Maring (2011), A
  sea-state based source function for size- and composition-resolved marine
  aerosol production, Atmos. Chem. Phys. 11, 1203-1216.
  Sofiev, M., J. Soares, M. Prank, G. de Leeuw, J. Kukkonen (2011), A
  regional-to-global model of emission and transport of sea salt particles in
  the atmosphere, J. Geophys. Res. 116, D21302.

"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, NamedTuple

import jax.numpy as jnp
import numpy as np
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import AerosolMode, ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm
from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
    accumulate_emission_fluxes, emission_flux_keys)
from jcm.physics.aerosol.jam.emissions.surface_wind import (
    MODEL_LEVEL_WIND_KEY, wind_10m)

# Shared bin-grid constants (mo_ham_m7_emi_seasalt.f90 module data, top).
_NBIN = 300
_DMTA = 0.100e-6        # lower dry diameter [m]
_DMTD = 1.000e-5        # upper dry diameter [m]

# Gong-scheme constants.
_DMTB_GONG = 0.221e-6   # Gong small/large particle split [m]
_PPWW = 3.41            # default wind-speed exponent

# Long-scheme constants (seasalt_emissions_long, mo_ham_m7_emi_seasalt.f90:865-1079).
_DMTB_LONG = 0.551e-6   # Long small/large particle split, dry diameter [m]
_PPWW_LONG = 3.74       # default wind-speed exponent
# p0 and the two size-formula polynomials stay fixed module constants, not
# SeaSaltParameters leaves: like Gong's own p0/p1/p2/p3 bin-shape
# coefficients (never exposed either), they parameterize the SHAPE of the
# fitted size spectrum rather than an overall calibration knob, and making
# them differentiable would let gradient descent warp that shape into one
# the Long et al. (2011)/Keene et al. (2007) lab fit no longer represents.
# wind_exponent_long is the one Long-specific constant exposed, matching
# Gong's own wind_exponent.
_LONG_P0 = 2.0e-8
# Log10(wet diameter [µm])**{3,2,1,0} polynomial coefficients for the
# small-particle (p1*) and large-particle (p2*) branches.
_LONG_P1 = (1.46e0, 1.33e0, -1.82e0, 8.83e0)
_LONG_P2 = (-1.53e0, -8.1e-2, -4.26e-1, 8.84e0)
# HAM's own fixed size-class diameter bounds [m] (crdiv-style; index 0 is the
# Aitken range, computed by the native routine but never output -- "currently
# Aitken mode particles are negelected"). index 1 = accumulation, 2 = coarse.
_DBEG = (0.050e-6, 0.100e-6, 1.000e-6)
_DEND = (0.100e-6, 1.000e-6, 1.000e-5)

# Sofiev et al. (2011) SST correction: 3 bands, each linear in SST between
# two anchor temperatures whose own (SST_corr_1, SST_corr_2) are themselves
# power laws in dry diameter [µm] -- (coefficient, exponent) pairs below, one
# per anchor. Band 3 is NOT clamped above 298.15 K (the source's own
# "limit T dependence to <25 Deg C" guard is commented out), so it
# extrapolates linearly beyond its fitted range.
#
# Literal-kind audit (mo_ham_m7_emi_seasalt.f90:1000-1046): every coefficient,
# exponent and anchor below is written as e.g. `0.13e0` or bare `278.15` --
# an "e" exponent letter (or none) without a `_dp`/kind suffix is Fortran's
# DEFAULT REAL (single precision), not double, regardless of the "e0". gfortran
# therefore rounds each to its nearest float32 bit pattern FIRST, then widens
# that rounded value to double for the arithmetic -- exactly the
# `float(np.float32(x))` promotion properties.py documents for M7's native
# `1.E-2` literal. None of 0.13/0.78/0.22/0.70/0.18/1.45/271.15/278.15/288.15/
# 298.15 are exactly representable in binary (float32 or float64), so this is
# a real ~3e-8 relative rounding on each, not a no-op; used naively as exact
# Python floats this under-corrected the AS/CS split by up to ~2.6e-6
# relative before this fix (#1017 W1 task 2 review). The window divisors
# (7./10./10.) and the dmt->µm factor (1.e06) are also bare/"e0" literals but
# ARE exactly representable in float32 (7, 10, 1e6 are small integers), so
# promoting them changes nothing and they are left as plain floats below.
def _f32(x):
    return float(np.float32(x))


_LONG_SST_ANCHORS = (_f32(271.15), _f32(278.15), _f32(288.15), _f32(298.15))
_LONG_SST_BAND_COEFFS = (
    ((_f32(0.13), _f32(-0.78)), (_f32(0.22), _f32(-0.70))),   # band 1: 271.15-278.15 K
    ((_f32(0.22), _f32(-0.70)), (_f32(0.70), _f32(-0.18))),   # band 2: 278.15-288.15 K
    ((_f32(0.70), _f32(-0.18)), (_f32(1.45), _f32(0.18))),    # band 3: >288.15 K, unclamped above
)


def _bin_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the shared ``start_emi_seasalt`` bin grid (mo_ham_m7_emi_seasalt.f90:92-166).

    ``nbin`` log-spaced bins between ``_DMTA`` and ``_DMTD``; returns dry
    diameter [m], dry radius [m], and RH=80% wet radius [µm] (``rm`` in the
    Fortran). Every scheme in this module (Gong, Long) evaluates its own
    source function on this same grid; Gong additionally uses ``bmn``
    (``start_emi_seasalt``'s Monahan-fit exponent, native to Gong/Monahan/
    Guelle only — Long does not read it), computed in :func:`gong_class_factors`.

    Vectorized as ``exp(log(dmta) + arange(nbin)*zdx)`` — mathematically the
    same grid the Fortran builds by repeated addition (``zdd = zdd + zdx``
    each iteration, mo_ham_m7_emi_seasalt.f90:137-144), but NOT bit-identical
    to it: Gong's own class split (:func:`gong_class_factors`) assigns each
    bin by nearest-log-distance to a population mode, which an ~1e-16
    relative grid perturbation essentially never flips, so this form is safe
    for Gong. Long's class split instead keys on HAM's fixed diameter
    bounds (``_DBEG``/``_DEND``) via an exact ``<=`` comparison, and ``nbin``
    is chosen so several of those bounds (notably 1 µm, the AS/CS edge) fall
    in theory exactly on a bin edge — so which side of `<=` a bin lands on
    depends on which grid construction you use. :func:`long_bin_geometry`
    therefore builds its own grid the same way the Fortran does (repeated
    addition), not this vectorized form, instead of sharing it here.
    """
    zdx = (math.log(_DMTD) - math.log(_DMTA)) / _NBIN
    dmt = np.exp(math.log(_DMTA) + np.arange(_NBIN) * zdx)   # dry diameter [m]
    rd = dmt * 0.5                                           # dry radius [m]
    rm = 1.814 * rd * 1.0e6                                  # RH=80% wet radius [µm]
    return dmt, rd, rm


def _long_bin_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long's own bin grid, built by the same repeated addition the Fortran
    uses (mo_ham_m7_emi_seasalt.f90:137-144: ``zdd = zdd + zdx`` each of
    ``nbin`` iterations), not :func:`_bin_grid`'s single-multiply vectorized
    form — see that function's docstring for why this matters here and not
    for Gong. Confirmed against the harness: bin 150 (of 300) sits within
    4e-22 m of exactly 1e-6 m (the AS/CS edge) either way, but on OPPOSITE
    sides of it between the two constructions, which moved ~1-3% of mass
    between AS and CS before this fix (see #1017 W1 task 2 final report).
    """
    zdx = (math.log(_DMTD) - math.log(_DMTA)) / _NBIN
    dmt = np.empty(_NBIN)
    zdd = 0.0
    for m in range(_NBIN):
        dmt[m] = math.exp(math.log(_DMTA) + zdd)
        zdd += zdx
    rd = dmt * 0.5
    rm = 1.814 * rd * 1.0e6
    return dmt, rd, rm


def gong_class_factors(
    classes: tuple[AerosolMode, ...], density: float
) -> dict[str, tuple[float, float]]:
    """Partition the Gong (2003) source flux across a population's ``ss`` classes.

    Returns ``{class_short: (mass_factor, number_factor)}`` — the
    wind-independent factors such that, per open-water area,
    ``mass_flux = mass_factor · u10**ppww`` [kg/m²/s] and
    ``number_flux = number_factor · u10**ppww`` [#/m²/s].

    The Gong source is evaluated on a fine dry-diameter grid, then **every bin
    is assigned to exactly one of ``classes``** — the class whose ``[dgnum_lo,
    dgnum_hi]`` contains it, or, for a bin in a gap/tail, the nearest class in
    log-diameter. That makes the split representation-agnostic (modal modes or
    sectional bins via each class's own size range) and **mass-conserving** — the
    whole 0.1–10 µm Gong spectrum is distributed with nothing dropped. (This
    replaces the previous hardcoded ``(accumulation, coarse)`` HAMMOZ-M7 diameter
    bands; the accum/coarse boundary now follows the population's mode edges
    rather than a fixed 1 µm cut.)
    """
    dmt, rd, rm = _bin_grid()
    bmn = (0.380 - np.log10(rm)) / 0.650
    small = (dmt > _DMTA) & (dmt <= _DMTB_GONG)
    bmn = np.where(small, (0.433 - np.log10(rm)) / 0.433, bmn)

    fi = np.zeros(_NBIN)        # size-only number-flux factor per bin
    for m in range(1, _NBIN):
        dr = rm[m] - rm[m - 1]
        if _DMTA < dmt[m] <= _DMTB_GONG:
            p0 = 4.7 * (1.0 + 30.0 * rm[m]) ** (-0.017 * rm[m] ** (-1.44))
            p1 = 1.373 * rm[m] ** (-p0)
            p2 = 1.0 + 0.057 * rm[m] ** 3.45
            p3 = 10.0 ** (1.607 * math.exp(-bmn[m] ** 2))
            fi[m] = p1 * p2 * p3 * dr
        elif _DMTB_GONG < dmt[m] <= _DMTD:
            p1 = 1.373 * rm[m] ** (-3)
            p2 = 1.0 + 0.057 * rm[m] ** 1.05
            p3 = 10.0 ** (1.19 * math.exp(-bmn[m] ** 2))
            fi[m] = p1 * p2 * p3 * dr

    zav = density * (4.0 / 3.0) * math.pi * rd ** 3          # particle mass [kg]

    # Assign each bin to the class containing it (log-distance 0), else nearest.
    ld = np.log(dmt)[:, None]
    lo = np.log(np.array([c.dgnum_lo for c in classes]))[None, :]
    hi = np.log(np.array([c.dgnum_hi for c in classes]))[None, :]
    dist = np.maximum(0.0, np.maximum(lo - ld, ld - hi))     # 0 if inside
    assign = np.argmin(dist, axis=1)                         # (nbin,) class idx
    return {
        c.short: (float(np.sum(fi[assign == ci] * zav[assign == ci])),
                  float(np.sum(fi[assign == ci])))
        for ci, c in enumerate(classes)
    }


#: Overall emission scale on the Gong (2003) source function. The function
#: itself is HAM's, unscaled (scale 1), and at T63 L47 it emits 2057 Tg/yr
#: (dry diameter 0.1-10 um) against the AeroCom median of 6280 (mean 16 600),
#: leaving a sea-salt burden of 6.6 mg/m2 (AeroCom mean 14.7, median 12.5). 4 is
#: the upper edge of the range [1, 4] that Stage 2c of the JAM aerosol retune
#: swept (``docs/source/design/jam_aerosol_retune.md``), with the Sundqvist T63
#: cloud-fraction set in place and the dust threshold scale as the other lever:
#: the 13 best of its 20 arms all sit at 3.996 or above, so the data constrain
#: the scale from below only, and 4 is the edge of the range rather than an
#: interior optimum. The scale acts mainly through the total AOD (ESA-CCI SU
#: v4.21), the largest of the non-dust levers, and it also moves the cloud
#: radiative terms of the loss. The cover set's larger cloud fraction removes
#: sea salt faster (a lifetime of 0.37 d, against 0.60 d with ECHAM's cover
#: parameters), which is why the scale that restores the AOD is large. A
#: 365-day ``echam-jam-t63-l47`` year at 4 emits 8475 Tg/yr (between the AeroCom
#: median and mean), with a burden of 16.4 mg/m2 (12 % above the AeroCom mean)
#: and a lifetime of 0.37 d (AeroCom 0.48). The value is a T63 L47 calibration
#: against the 10 m wind this host produces and the wet removal the cover set
#: implies; the default applies at every resolution because no other has been
#: validated. ``+physics.seasalt.scale=`` (or ``seasalt={"scale": ...}`` of
#: ``echam_physics``) overrides it.
SEASALT_SCALE_DEFAULT = 4.0


class LongBinGeometry(NamedTuple):
    """Per-bin arrays :func:`long_bin_geometry` precomputes (density- and
    bin-geometry-only; no wind/SST dependence, so these are built once at
    :class:`SeaSaltEmissions` construction and reused every call).
    """

    size_factor: jnp.ndarray   # (nbin,) 10**p(bin)*logdp(bin), wind/SST-free
    particle_mass: jnp.ndarray  # (nbin,) kg; "zav" in the Fortran
    band_coeffs: jnp.ndarray   # (3 bands, 2 anchors, nbin) SST_corr_{1,2}(bin)
    as_mask: jnp.ndarray       # (nbin,) bool: dry diameter in HAM's AS range
    cs_mask: jnp.ndarray       # (nbin,) bool: dry diameter in HAM's CS range


def long_bin_geometry(density: float) -> LongBinGeometry:
    """Precompute the Long et al. (2011) + Sofiev et al. (2011) per-bin arrays
    (``seasalt_emissions_long``, mo_ham_m7_emi_seasalt.f90:865-1079).

    Unlike :func:`gong_class_factors`, this does NOT collapse to per-class
    (mass_factor, number_factor) constants: the Sofiev SST correction's shape
    varies with bin diameter, so ``fi(bin, col)`` is genuinely 2-D in bin and
    SST and must be evaluated at call time against the real SST field (see
    :class:`SeaSaltEmissions`). What CAN be precomputed — because it depends
    only on bin geometry and the (fixed, non-differentiable) particle
    density — is bundled here: the wind/SST-free size factor, the per-bin
    particle mass, the six power-law coefficients of the SST correction (as
    functions of dry diameter only), and the fixed HAM accumulation/coarse
    bin masks.
    """
    dmt, rd, rm = _long_bin_grid()
    size_factor = np.zeros(_NBIN)
    for m in range(1, _NBIN):
        logdp = np.log10(2.0 * rm[m]) - np.log10(2.0 * rm[m - 1])
        log_wet_diam = np.log10(2.0 * rm[m])
        if _DMTA < dmt[m] <= _DMTB_LONG:
            p11, p12, p13, p14 = _LONG_P1
            p = p11 * log_wet_diam ** 3 + p12 * log_wet_diam ** 2 + p13 * log_wet_diam + p14
            size_factor[m] = 10.0 ** p * logdp
        elif _DMTB_LONG < dmt[m] <= _DMTD:
            p21, p22, p23, p24 = _LONG_P2
            p = p21 * log_wet_diam ** 3 + p22 * log_wet_diam ** 2 + p23 * log_wet_diam + p24
            size_factor[m] = 10.0 ** p * logdp
        # dmt(m) <= _DMTA cannot occur for m>=1 (dmt is strictly increasing
        # from dmt[0]==_DMTA), matching the native m=1 bin the Fortran DO
        # loop (starting at m=2, 1-based) never touches either.

    particle_mass = density * (4.0 / 3.0) * math.pi * rd ** 3   # kg; native "zav"
    dmtum = dmt * 1.0e6   # dry diameter [µm] -- the SST correction's own axis
    band_coeffs = np.array([
        [coeff * dmtum ** exponent for coeff, exponent in band]
        for band in _LONG_SST_BAND_COEFFS
    ])   # (3, 2, nbin)
    as_mask = (dmt > _DBEG[1]) & (dmt <= _DEND[1])
    cs_mask = (dmt > _DBEG[2]) & (dmt <= _DEND[2])
    return LongBinGeometry(
        size_factor=jnp.asarray(size_factor),
        particle_mass=jnp.asarray(particle_mass),
        band_coeffs=jnp.asarray(band_coeffs),
        as_mask=jnp.asarray(as_mask),
        cs_mask=jnp.asarray(cs_mask),
    )


@tree_math.struct
class SeaSaltParameters:
    """Calibratable knobs for the Gong and Long sea-salt schemes.

    ``wind_exponent_long`` defaults in the constructor itself (not only in
    ``.default()``), so existing code that builds
    ``SeaSaltParameters(scale=..., wind_exponent=...)`` — as the Gong tests
    and any pre-#1017 caller do — keeps working unchanged; it is read only
    by the Long path (``scheme="long"``), so the Gong path's own arithmetic
    is untouched either way.
    """

    scale: jnp.ndarray               # overall emission scale factor (both schemes)
    wind_exponent: jnp.ndarray       # Gong's u10 exponent (default 3.41)
    wind_exponent_long: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.asarray(_PPWW_LONG))  # Long's u10 exponent

    @classmethod
    def default(cls) -> "SeaSaltParameters":
        return cls(scale=jnp.asarray(SEASALT_SCALE_DEFAULT),
                   wind_exponent=jnp.asarray(_PPWW),
                   wind_exponent_long=jnp.asarray(_PPWW_LONG))


class SeaSaltEmissions(PhysicsTerm):
    """Wind-driven sea-salt emission into accumulation + coarse.

    ``scheme="gong"`` (default, unchanged): Gong (2003). ``scheme="long"``:
    Long et al. (2011) with the Sofiev et al. (2011) SST correction — HAM's
    own two-class (accumulation, coarse) output, so it requires the
    population to carry exactly two ``ss`` classes in that (spec) order and
    does not use :func:`gong_class_factors`'s population-size-range
    partition.
    """

    name: ClassVar[str] = "jam_seasalt_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = (
        emission_flux_keys() + (MODEL_LEVEL_WIND_KEY,))

    def __init__(
        self,
        params: SeaSaltParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
        scheme: str = "gong",
    ):
        """Precompute the per-class/per-bin geometry for sea-salt density."""
        self.params = nnx.Param(params or SeaSaltParameters.default())
        self._spec = spec or MAM4_SPEC
        self._scheme = scheme
        density = self._spec.species_props("ss").density
        if scheme == "gong":
            # Which classes carry sea salt — and their size ranges — come
            # from the population, so the term names no modes and a
            # sectional spec works unchanged.
            self._fac = gong_class_factors(self._spec.classes_for("ss"), density)
        elif scheme == "long":
            classes = self._spec.classes_for("ss")
            if len(classes) != 2:
                raise ValueError(
                    "scheme='long' ports HAM's own two-class (accumulation, "
                    "coarse) AS/CS output (mo_ham_m7_emi_seasalt.f90's fixed "
                    "dbeg/dend bins, not a population size-range partition "
                    f"like Gong's); got {len(classes)} 'ss' classes "
                    f"({[c.short for c in classes]}). Supply a population "
                    "whose 'ss' species carries exactly two classes, in "
                    "HAM's accumulation-then-coarse order."
                )
            self._long_classes = classes   # (accumulation-like, coarse-like)
            # Precomputed jnp arrays, not a differentiable leaf (bin geometry
            # fixed at construction) -- nnx.data so the Module's pytree
            # flattening treats it as data, not a static attribute (unlike
            # Gong's self._fac, which holds only plain Python floats).
            self._long = nnx.data(long_bin_geometry(density))
        else:
            raise ValueError(
                f"Unknown sea-salt scheme {scheme!r}; choose 'gong' or 'long'.")

    def _open_water_fraction(self, forcing, terrain, ncols):
        """Non-iced open-water fraction (1 − land)·(1 − sea-ice), land>0.5→0."""
        fm = getattr(terrain, "fmask", None) if terrain is not None else None
        land = (
            jnp.clip(jnp.ravel(fm), 0.0, 1.0)
            if fm is not None and fm.size == ncols
            else jnp.zeros((ncols,))
        )
        sice = getattr(forcing, "sice_am", None) if forcing is not None else None
        sea_ice = (
            jnp.clip(jnp.ravel(sice), 0.0, 1.0)
            if sice is not None and jnp.size(sice) == ncols
            else jnp.zeros((ncols,))
        )
        frac = (1.0 - land) * (1.0 - sea_ice)
        return jnp.where(land > 0.5, 0.0, jnp.clip(frac, 0.0, 1.0))

    def _long_as_cs_fluxes(self, u10, sst, seafrac):
        """Evaluate seasalt_emissions_long's AS/CS mass/number fluxes.

        Per open-water area; ``u10``, ``sst``, ``seafrac`` are ``(ncols,)``.
        Unlike Gong, the Sofiev SST correction couples bin and column (its
        shape depends on dry diameter), so ``fi`` is a genuine
        ``(nbin, ncols)`` array built here, not a precomputed per-class
        constant; see :func:`long_bin_geometry`.
        """
        g = self._long
        p = self.params.get_value()
        wind_factor = _LONG_P0 * u10 ** p.wind_exponent_long          # (ncols,)
        size_wind = g.size_factor[:, None] * wind_factor[None, :]     # (nbin, ncols)

        sst_b = sst[None, :]                                          # (1, ncols)
        t1, t2, t3, t4 = _LONG_SST_ANCHORS
        corr1_a, corr2_a = g.band_coeffs[0, 0][:, None], g.band_coeffs[0, 1][:, None]
        corr1_b, corr2_b = g.band_coeffs[1, 0][:, None], g.band_coeffs[1, 1][:, None]
        corr1_c, corr2_c = g.band_coeffs[2, 0][:, None], g.band_coeffs[2, 1][:, None]
        # The native divisors are the independent literals 7./1.e1/1.e1 (both
        # exact in float32, mo_ham_m7_emi_seasalt.f90:1009/1025/1042), NOT
        # (t2-t1) etc. computed from the now-float32-rounded anchors above --
        # those two differ by ~1e-8 relative, since e.g. float32(278.15) -
        # float32(271.15) is not exactly 7.0.
        band1 = (corr1_a * (t2 - sst_b) + corr2_a * (sst_b - t1)) / 7.0
        band2 = (corr1_b * (t3 - sst_b) + corr2_b * (sst_b - t2)) / 10.0
        band3 = (corr1_c * (t4 - sst_b) + corr2_c * (sst_b - t3)) / 10.0
        # Exactly one band applies per column (its own MERGE in the Fortran);
        # summing the three masked terms reproduces that selection.
        sst_corr = (jnp.where(sst_b <= t2, band1, 0.0)
                    + jnp.where((sst_b > t2) & (sst_b <= t3), band2, 0.0)
                    + jnp.where(sst_b > t3, band3, 0.0))

        # The native routine multiplies each bin's `fi` by `zseafrac` inside
        # the per-bin accumulation (mo_ham_m7_emi_seasalt.f90:1062-1075);
        # since zseafrac does not vary across bins, applying it once after
        # the bin sum is the same sum in exact arithmetic and differs only
        # by ~300-term floating-point reassociation (operation order).
        fi = size_wind * sst_corr                                     # (nbin, ncols)
        as_fi = jnp.where(g.as_mask[:, None], fi, 0.0)
        cs_fi = jnp.where(g.cs_mask[:, None], fi, 0.0)
        number_as = jnp.sum(as_fi, axis=0) * seafrac
        number_cs = jnp.sum(cs_fi, axis=0) * seafrac
        mass_as = jnp.sum(as_fi * g.particle_mass[:, None], axis=0) * seafrac
        mass_cs = jnp.sum(cs_fi * g.particle_mass[:, None], axis=0) * seafrac
        return mass_as, mass_cs, number_as, number_cs

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        u10, from_model_level = wind_10m(state, diagnostics)
        seafrac = self._open_water_fraction(forcing, terrain, ncols)

        inv = 1.0 / (air_density[-1] * dz[-1])              # kg/kg per kg/m²/s
        bottom = lambda flux2d: jnp.zeros((nlev, ncols)).at[-1].set(flux2d * inv)

        tracer_tends = {}
        if self._scheme == "gong":
            wind = p.scale * u10 ** p.wind_exponent * seafrac   # (ncols,)
            for short, (mass_fac, numb_fac) in self._fac.items():
                tracer_tends[mass_name("ss", short)] = bottom(mass_fac * wind)
                tracer_tends[number_name(short)] = bottom(numb_fac * wind)
        else:  # "long"
            sst = jnp.ravel(forcing.sea_surface_temperature)
            mass_as, mass_cs, number_as, number_cs = self._long_as_cs_fluxes(
                u10, sst, seafrac)
            accum_short, coarse_short = (c.short for c in self._long_classes)
            tracer_tends[mass_name("ss", accum_short)] = bottom(p.scale * mass_as)
            tracer_tends[number_name(accum_short)] = bottom(p.scale * number_as)
            tracer_tends[mass_name("ss", coarse_short)] = bottom(p.scale * mass_cs)
            tracer_tends[number_name(coarse_short)] = bottom(p.scale * number_cs)
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        # Publish this term's contribution to the AeroCom per-species
        # emission fluxes (accumulated across all emitting terms).
        diagnostics = accumulate_emission_fluxes(
            diagnostics, tracer_tends,
            diagnostics["air_density"],
            diagnostics["layer_thickness"])
        diagnostics = {**diagnostics, MODEL_LEVEL_WIND_KEY: jnp.maximum(
            diagnostics.get(MODEL_LEVEL_WIND_KEY, 0.0), from_model_level)}

        return tendency, diagnostics
