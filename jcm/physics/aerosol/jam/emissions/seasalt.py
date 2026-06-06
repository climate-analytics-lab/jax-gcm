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
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

# Gong-scheme constants (mo_ham_m7_emi_seasalt.f90).
_NBIN = 300
_DMTA = 0.100e-6        # lower dry diameter [m]
_DMTD = 1.000e-5        # upper dry diameter [m]
_DMTB_GONG = 0.221e-6   # Gong small/large particle split [m]
_PPWW = 3.41            # default wind-speed exponent
# Dry-diameter ranges mapping bins → (aitken, accumulation, coarse) [m].
_DBEG = (0.050e-6, 0.100e-6, 1.000e-6)
_DEND = (0.100e-6, 1.000e-6, 1.000e-5)


def gong_mode_factors(density: float) -> dict[str, float]:
    """Integrate the Gong source function over each mode's size range.

    Returns ``{"mass_acc", "numb_acc", "mass_cor", "numb_cor"}`` — the
    wind-independent factors such that, per open-water area,
    ``mass_flux = mass_<mode> · u10**ppww`` [kg/m²/s] and
    ``number_flux = numb_<mode> · u10**ppww`` [#/m²/s].
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
    acc = (dmt > _DBEG[1]) & (dmt <= _DEND[1])
    cor = (dmt > _DBEG[2]) & (dmt <= _DEND[2])
    return {
        "mass_acc": float(np.sum(fi[acc] * zav[acc])),
        "numb_acc": float(np.sum(fi[acc])),
        "mass_cor": float(np.sum(fi[cor] * zav[cor])),
        "numb_cor": float(np.sum(fi[cor])),
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
        """Precompute the Gong per-mode bin factors for sea-salt density."""
        self.params = nnx.Param(params or SeaSaltParameters.default())
        self._spec = spec or MAM4_SPEC
        density = self._spec.species_props("ss").density
        self._fac = gong_mode_factors(density)

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

        u10 = jnp.sqrt(state.u_wind[-1] ** 2 + state.v_wind[-1] ** 2)
        seafrac = self._open_water_fraction(forcing, terrain, ncols)
        wind = p.scale * u10 ** p.wind_exponent * seafrac   # (ncols,)

        inv = 1.0 / (air_density[-1] * dz[-1])              # kg/kg per kg/m²/s
        bottom = lambda flux2d: jnp.zeros((nlev, ncols)).at[-1].set(flux2d * inv)

        tracer_tends = {
            mass_name("ss", "acc"): bottom(self._fac["mass_acc"] * wind),
            number_name("acc"): bottom(self._fac["numb_acc"] * wind),
            mass_name("ss", "cor"): bottom(self._fac["mass_cor"] * wind),
            number_name("cor"): bottom(self._fac["numb_cor"] * wind),
        }
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
