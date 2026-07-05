"""Gong (2003) sea-salt emission scheme.

Faithful port of HAMMOZ ``mo_ham_m7_emi_seasalt::seasalt_emissions_gong``
(``nseasalt=6``). The Gong (2003) source function gives a number flux per dry
size bin that factorises as ``f_i = (size-only factor) · u10**3.41``; the
size-only factor is wind-independent, so the per-mode mass and number fluxes
collapse to **two precomputed constants per mode** (accumulation, coarse)
times ``u10**3.41`` times the open-water fraction. That makes the term cheap,
jittable and differentiable.

References:
  Gong, S. L. (2003), A parameterization of sea-salt aerosol source function
  for sub- and super-micron particles, Global Biogeochem. Cycles 17(4), 1097.
  Monahan et al. (1986), Oceanic Whitecaps.

"""

from __future__ import annotations

import math
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import AerosolMode, ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

# Gong-scheme constants (mo_ham_m7_emi_seasalt.f90).
_NBIN = 300
_DMTA = 0.100e-6        # lower dry diameter [m]
_DMTD = 1.000e-5        # upper dry diameter [m]
_DMTB_GONG = 0.221e-6   # Gong small/large particle split [m]
_PPWW = 3.41            # default wind-speed exponent


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
    zdx = (math.log(_DMTD) - math.log(_DMTA)) / _NBIN
    dmt = np.exp(math.log(_DMTA) + np.arange(_NBIN) * zdx)   # dry diameter [m]
    rd = dmt * 0.5                                           # dry radius [m]
    rm = 1.814 * rd * 1.0e6                                  # RH=80% wet radius [µm]
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


@tree_math.struct
class SeaSaltParameters:
    """Calibratable knobs for the Gong sea-salt scheme."""

    scale: jnp.ndarray          # overall emission scale factor
    wind_exponent: jnp.ndarray  # u10 exponent (Gong default 3.41)

    @classmethod
    def default(cls) -> "SeaSaltParameters":
        return cls(scale=jnp.asarray(1.0), wind_exponent=jnp.asarray(_PPWW))


class SeaSaltEmissions(PhysicsTerm):
    """Wind-driven sea-salt emission (Gong 2003) into accumulation + coarse."""

    name: ClassVar[str] = "jam_seasalt_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: SeaSaltParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Precompute the Gong per-class bin factors for sea-salt density."""
        self.params = nnx.Param(params or SeaSaltParameters.default())
        self._spec = spec or MAM4_SPEC
        density = self._spec.species_props("ss").density
        # Which classes carry sea salt — and their size ranges — come from the
        # population, so the term names no modes and a sectional spec works
        # unchanged.
        self._fac = gong_class_factors(self._spec.classes_for("ss"), density)

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

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        u10 = jnp.sqrt(jnp.maximum(state.u_wind[-1] ** 2 + state.v_wind[-1] ** 2, 1.0e-30))
        seafrac = self._open_water_fraction(forcing, terrain, ncols)
        wind = p.scale * u10 ** p.wind_exponent * seafrac   # (ncols,)

        inv = 1.0 / (air_density[-1] * dz[-1])              # kg/kg per kg/m²/s
        bottom = lambda flux2d: jnp.zeros((nlev, ncols)).at[-1].set(flux2d * inv)

        tracer_tends = {}
        for short, (mass_fac, numb_fac) in self._fac.items():
            tracer_tends[mass_name("ss", short)] = bottom(mass_fac * wind)
            tracer_tends[number_name(short)] = bottom(numb_fac * wind)
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
