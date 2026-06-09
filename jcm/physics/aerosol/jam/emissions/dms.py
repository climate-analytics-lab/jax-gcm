"""Oceanic DMS emission (Nightingale 2000 sea–air exchange).

Port of the Nightingale (2000) branch of HAMMOZ
``mo_hammoz_emi_ocean::dms_emissions``: a sea–air transfer (piston) velocity
from the 10 m wind and the DMS Schmidt number (Andreae fit), times a
prescribed seawater DMS concentration. The DMS gas oxidises to SO2 and then
sulfate; since this configuration carries no gas-phase sulfur tracer, the
emitted sulfur is added directly as primary sulfate to the Aitken/accumulation
modes (an interim — full DMS→SO2→SO4 gas chemistry is #496).

The seawater DMS field is read from ``forcing.dms_seawater`` (a prescribed
climatology, e.g. AeroCom ``conc_aerocom_DMS_sea.nc``); it falls back to zero
when absent, so the term is simply inert until the field is supplied.

Reference: Nightingale et al. (2000), Global Biogeochem. Cycles 14, 373.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.emissions.distributors import distribute_surface_flux
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

_CMH_TO_MS = 0.01 / 3600.0   # cm/h → m/s


def dms_schmidt_number(sst_celsius: jnp.ndarray) -> jnp.ndarray:
    """DMS Schmidt number (Andreae fit), SST in °C clamped to [-2, 35]."""
    t = jnp.clip(sst_celsius, -2.0, 35.0)
    return 3652.047271 - 246.99 * t + 8.536397 * t ** 2 - 0.124397 * t ** 3


def piston_velocity(u10: jnp.ndarray, schmidt: jnp.ndarray) -> jnp.ndarray:
    """Nightingale (2000) transfer velocity [m/s].

    ``kw = 0.222·u10² + 0.333·u10`` (cm/h) scaled by ``(Sc/660)^-1/2`` —
    smooth in wind (no wind-class branches), so it is cleanly differentiable.
    """
    kw_cmh = 0.222 * u10 ** 2 + 0.333 * u10
    return kw_cmh * (schmidt / 660.0) ** (-0.5) * _CMH_TO_MS


@tree_math.struct
class DmsParameters:
    """Calibratable knobs for DMS sea–air emission."""

    flux_scale: jnp.ndarray      # folds seawater-conc units → emitted SO4 mass
    aitken_fraction: jnp.ndarray  # split of emitted sulfate into the Aitken mode

    @classmethod
    def default(cls) -> "DmsParameters":
        return cls(flux_scale=jnp.asarray(1.0), aitken_fraction=jnp.asarray(0.5))


class DmsEmissions(PhysicsTerm):
    """Oceanic DMS → primary sulfate emission (Nightingale 2000)."""

    name: ClassVar[str] = "jam_dms_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: DmsParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or DmsParameters.default())
        self._spec = spec or MAM4_SPEC

    @staticmethod
    def _forcing_field(forcing, name, ncols, default):
        v = getattr(forcing, name, None) if forcing is not None else None
        if v is not None and jnp.size(v) == ncols:
            return jnp.ravel(v)
        return jnp.full((ncols,), default)

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        u10 = jnp.sqrt(state.u_wind[-1] ** 2 + state.v_wind[-1] ** 2)
        sst = self._forcing_field(
            forcing, "sea_surface_temperature", ncols, 288.0
        )
        dms_sea = self._forcing_field(forcing, "dms_seawater", ncols, 0.0)

        # Open-water fraction (1 − land); sea-ice already in the seawater field.
        fm = getattr(terrain, "fmask", None) if terrain is not None else None
        land = (
            jnp.clip(jnp.ravel(fm), 0.0, 1.0)
            if fm is not None and fm.size == ncols
            else jnp.zeros((ncols,))
        )
        seafrac = jnp.where(land > 0.5, 0.0, 1.0 - land)

        vp = piston_velocity(u10, dms_schmidt_number(sst - 273.15))
        flux_s = p.flux_scale * vp * dms_sea * seafrac        # kg-S/m²/s scale

        fluxes = [
            ("so4", "ait", flux_s * p.aitken_fraction),
            ("so4", "acc", flux_s * (1.0 - p.aitken_fraction)),
        ]
        tracer_tends = distribute_surface_flux(self._spec, fluxes, air_density, dz)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
