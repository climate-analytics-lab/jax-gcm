"""``JamEmissions`` — online natural + prescribed aerosol surface sources.

Computes wind-driven sea salt, threshold dust, ocean DMS→sulfate, and
constant volcanic-SO2 / biogenic-SOA surface mass fluxes, then distributes
them into the lowest-layer modal tracers (mass + implied number) via
``distributors.distribute_surface_flux``. Mirrors the source split of
``mo_hammoz_emissions`` and friends, simplified for the aquaplanet.

The source *magnitudes* are order-of-magnitude defaults, not calibrated to an
emission inventory (the AeroCom-budget acceptance gate needs reference data we
don't carry); they are differentiable knobs on ``EmissionParameters``.
Prescribed anthropogenic sector fluxes are a future hook (need new
``ForcingData`` fields).
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


@tree_math.struct
class EmissionParameters:
    """Differentiable source coefficients (uncalibrated order-of-magnitude)."""

    seasalt_coeff: jnp.ndarray
    seasalt_wind_exp: jnp.ndarray
    seasalt_accum_frac: jnp.ndarray
    dust_coeff: jnp.ndarray
    dust_u_threshold: jnp.ndarray
    dms_coeff: jnp.ndarray
    volcanic_so4: jnp.ndarray
    biogenic_soa: jnp.ndarray

    @classmethod
    def default(cls) -> "EmissionParameters":
        return cls(
            seasalt_coeff=jnp.asarray(1.0e-13),
            seasalt_wind_exp=jnp.asarray(3.41),
            seasalt_accum_frac=jnp.asarray(0.2),
            dust_coeff=jnp.asarray(1.0e-14),
            dust_u_threshold=jnp.asarray(6.0),
            dms_coeff=jnp.asarray(1.0e-13),
            volcanic_so4=jnp.asarray(0.0),
            biogenic_soa=jnp.asarray(0.0),
        )


class JamEmissions(PhysicsTerm):
    """Online natural + prescribed aerosol emissions term."""

    name: ClassVar[str] = "ham_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: EmissionParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or EmissionParameters.default())
        self._spec = spec or MAM4_SPEC

    def _land_fraction(self, terrain, ncols):
        fm = getattr(terrain, "fmask", None) if terrain is not None else None
        if fm is not None and fm.size == ncols:
            return jnp.clip(jnp.ravel(fm), 0.0, 1.0)
        return jnp.zeros((ncols,))

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        u10 = jnp.sqrt(state.u_wind[-1] ** 2 + state.v_wind[-1] ** 2)
        land = self._land_fraction(terrain, ncols)
        ocean = 1.0 - land

        seasalt = ocean * p.seasalt_coeff * u10 ** p.seasalt_wind_exp
        dust = land * p.dust_coeff * jnp.maximum(u10 - p.dust_u_threshold, 0.0) ** 3
        dms_so4 = ocean * p.dms_coeff * u10
        volc = jnp.full((ncols,), p.volcanic_so4)
        bio = jnp.full((ncols,), p.biogenic_soa)

        fluxes = [
            ("ss", "acc", seasalt * p.seasalt_accum_frac),
            ("ss", "cor", seasalt * (1.0 - p.seasalt_accum_frac)),
            ("du", "acc", dust * 0.1),
            ("du", "cor", dust * 0.9),
            ("so4", "ait", dms_so4 * 0.5),
            ("so4", "acc", dms_so4 * 0.5 + volc),
            ("soa", "ait", bio * 0.5),
            ("soa", "acc", bio * 0.5),
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
