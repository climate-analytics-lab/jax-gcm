"""``WetScavenging`` — in-cloud and below-cloud aerosol scavenging.

Two removal pathways for interstitial aerosol, both differentiable and built
only from diagnostics the cloud scheme already exposes (so the cloud
microphysics terms are untouched):

* **In-cloud nucleation scavenging** — the activated fraction of aerosol is in
  cloud droplets and is removed at the rate cloud condensate converts to
  precipitation. The per-level precip-formation rate is reconstructed by
  distributing the column surface precip (``CloudData.precip_rain/snow``)
  across the cloudy column weighted by in-cloud condensate.
* **Below-cloud impaction scavenging** — falling precipitation collects
  interstitial aerosol in clear air, with a size-dependent (∝ r²) collection
  efficiency so coarse particles are scavenged far faster than accumulation
  mode.

Mirrors ``mo_hammoz_wetdep``. Precip re-evaporation re-injection is deferred
(no per-level evaporation-rate diagnostic is exposed yet).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

_EPS = 1.0e-30


@tree_math.struct
class WetDepParameters:
    """Tunable scavenging knobs (differentiable)."""

    incloud_scale: jnp.ndarray     # multiplies in-cloud removal
    below_coeff: jnp.ndarray       # below-cloud Λ per mm/h of rain [1/s]
    below_radius_ref: jnp.ndarray  # reference radius for ∝r² impaction [m]

    @classmethod
    def default(cls) -> "WetDepParameters":
        return cls(
            incloud_scale=jnp.asarray(1.0),
            below_coeff=jnp.asarray(1.0e-4),
            below_radius_ref=jnp.asarray(1.0e-7),
        )


def precip_formation_rate(
    precip_col: jnp.ndarray,      # (ncols,) surface precip [kg/m²/s]
    cloud_fraction: jnp.ndarray,  # (nlev, ncols)
    qc: jnp.ndarray,              # (nlev, ncols)
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
) -> jnp.ndarray:
    """Per-level condensate→precip conversion rate [kg/kg/s].

    Distributes the column surface precip across the cloudy column weighted
    by in-cloud condensate, converting the surface mass flux to a local
    mixing-ratio sink rate.
    """
    weight = cloud_fraction * jnp.maximum(qc, 0.0)
    w_sum = jnp.sum(weight, axis=0, keepdims=True)
    frac = weight / jnp.maximum(w_sum, _EPS)
    local = precip_col[jnp.newaxis, :] / (air_density * layer_thickness)
    return local * frac


def in_cloud_rate(
    activated_fraction: jnp.ndarray,
    p_form: jnp.ndarray,
    qc: jnp.ndarray,
) -> jnp.ndarray:
    """In-cloud scavenging rate [1/s] applied to interstitial aerosol."""
    return activated_fraction * p_form / jnp.maximum(qc, _EPS)


def below_cloud_rate(
    precip_col: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    r_wet: jnp.ndarray,
    params: WetDepParameters,
) -> jnp.ndarray:
    """Below-cloud impaction scavenging rate [1/s], size-dependent (∝ r²)."""
    rain_mmph = precip_col[jnp.newaxis, :] * 3600.0  # kg/m²/s -> mm/h
    efficiency = (r_wet / params.below_radius_ref) ** 2
    return (
        params.below_coeff * rain_mmph * (1.0 - cloud_fraction) * efficiency
    )


class WetScavenging(PhysicsTerm):
    """In-cloud + below-cloud scavenging of interstitial aerosol."""

    name: ClassVar[str] = "ham_wet_deposition"
    category: ClassVar[str] = "aerosol_wetdep"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "activated_fraction", "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: WetDepParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or WetDepParameters.default())
        self._spec = spec or MAM4_SPEC

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        aer = diagnostics["_jam_state"]
        activated_fraction = diagnostics["activated_fraction"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        clouds = diagnostics.get("clouds")
        if clouds is not None:
            precip_col = clouds.precip_rain + clouds.precip_snow
            cloud_fraction = clouds.cloud_fraction
            qc = clouds.qc
        else:
            precip_col = jnp.zeros((ncols,))
            cloud_fraction = jnp.zeros((nlev, ncols))
            qc = jnp.zeros((nlev, ncols))

        p_form = precip_formation_rate(
            precip_col, cloud_fraction, qc, air_density, dz,
        )
        rate_incloud = params.incloud_scale * in_cloud_rate(
            activated_fraction, p_form, qc,
        )

        tracer_tends: dict[str, jnp.ndarray] = {}
        for i, mode in enumerate(self._spec.modes):
            rate_below = below_cloud_rate(
                precip_col, cloud_fraction, aer.r_wet[i], params,
            )
            # In-cloud only removes from activatable (soluble) modes.
            rate = rate_below + (rate_incloud if mode.can_activate else 0.0)
            names = [number_name(mode.short)] + [
                mass_name(sp, mode.short) for sp in mode.species
            ]
            for nm in names:
                q = state.tracers.get(nm)
                if q is None:
                    continue
                tracer_tends[nm] = -rate * q

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
