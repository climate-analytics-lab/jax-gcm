"""``PrescribedOxidants`` — interim oxidant fields for the JAM sulfur chemistry.

The gas-phase (DMS+OH/NO₃, SO₂+OH) and aqueous (SO₂+H₂O₂/O₃) sulfur oxidation
need OH, NO₃, O₃ and H₂O₂. A full implementation reads a monthly oxidant
climatology; **as an interim** (issue #496) these are *derived from fields jcm
already carries* so the chain runs and is testable without new input data:

* **O₃** — reused from ``diagnostics["chemistry"].ozone_vmr`` (the
  ``SimpleChemistry`` ozone field; mole-fraction = ``ozone_vmr·1e-6``, matching
  RRTMGP's convention). Falls back to a fixed tropospheric value.
* **OH** — daytime photochemical proxy ``∝ cos(zenith)·[O₃]`` (O(¹D)+H₂O source
  scales with O₃ photolysis), reusing ``diagnostics["radiation"].cos_zenith``.
* **H₂O₂** — a reservoir species (multi-day lifetime); a boundary-layer-weighted
  prescribed profile (not the instantaneous cos-zenith).
* **NO₃** — nighttime oxidant ``∝ (1−cos zenith)·[O₃]``.

Chemistry + radiation run **after** the JAM block in ``echam_physics``, so both
are read from the **previous step's carry** with a fallback (the pattern ARG
uses for ``vertical_diffusion``); on the first step the fallbacks apply.

All scales are differentiable :class:`OxidantParameters` so the interim fields
can be calibrated and later swapped for a real climatology. Output number
densities are in **molec cm⁻³** (the unit gas-kinetic rate constants expect).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

import jcm.constants as c
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

# Reference mole fractions for the proxy scalings.
_O3_REF = 4.0e-8        # 40 ppbv — typical tropospheric O₃
_P_REF = 1.0e5          # Pa — surface pressure scale for BL weighting


@tree_math.struct
class OxidantField:
    """Prescribed oxidant number densities [molec cm⁻³], ``(nlev, ncols)``."""

    oh: jnp.ndarray      # hydroxyl radical
    no3: jnp.ndarray     # nitrate radical (night)
    o3: jnp.ndarray      # ozone
    h2o2: jnp.ndarray    # hydrogen peroxide


@tree_math.struct
class OxidantParameters:
    """Tunable scales for the interim oxidant proxies (differentiable)."""

    oh_ref: jnp.ndarray        # peak daytime OH [molec cm⁻³]
    h2o2_ref_vmr: jnp.ndarray  # surface H₂O₂ mole fraction
    no3_ref_vmr: jnp.ndarray   # nighttime NO₃ mole fraction
    o3_fallback_vmr: jnp.ndarray   # O₃ mole fraction when chemistry absent
    cos_zenith_fallback: jnp.ndarray  # cos(SZA) when radiation absent

    @classmethod
    def default(cls) -> "OxidantParameters":
        return cls(
            oh_ref=jnp.asarray(2.5e6),
            h2o2_ref_vmr=jnp.asarray(5.0e-10),   # 0.5 ppbv
            no3_ref_vmr=jnp.asarray(1.0e-12),    # 1 pptv
            o3_fallback_vmr=jnp.asarray(4.0e-8),  # 40 ppbv
            cos_zenith_fallback=jnp.asarray(0.25),
        )


def air_number_density(temperature: jnp.ndarray,
                       pressure: jnp.ndarray) -> jnp.ndarray:
    """Air number density [molec cm⁻³] from the ideal gas law ``n = P/(k_B T)``."""
    return pressure / (c.ak * temperature) * 1.0e-6


def oxidant_field(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    o3_vmr_ppmv: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    params: OxidantParameters,
) -> OxidantField:
    """Build the interim oxidant number densities (see module docstring).

    Args:
        temperature: ``(nlev, ncols)`` [K].
        pressure: ``(nlev, ncols)`` [Pa].
        o3_vmr_ppmv: ozone, in the ``SimpleChemistry`` ppmv-style unit
            (mole fraction = ``·1e-6``); ``(nlev, ncols)``.
        cos_zenith: cosine solar zenith angle, broadcast to ``(nlev, ncols)``.
        params: tunable scales.

    """
    n_air = air_number_density(temperature, pressure)
    o3_molefrac = jnp.maximum(o3_vmr_ppmv * 1.0e-6, 0.0)
    n_o3 = o3_molefrac * n_air
    cosz = jnp.clip(cos_zenith, 0.0, 1.0)
    o3_ratio = jnp.clip(o3_molefrac / _O3_REF, 0.0, 5.0)
    p_weight = jnp.clip(pressure / _P_REF, 0.0, 1.0)

    # OH: daytime photochemical proxy ∝ cos(zenith)·[O₃ relative to typical].
    oh = params.oh_ref * cosz * o3_ratio

    # H₂O₂: reservoir; boundary-layer-weighted prescribed mole fraction.
    n_h2o2 = params.h2o2_ref_vmr * p_weight * n_air

    # NO₃: nighttime, O₃-scaled.
    n_no3 = params.no3_ref_vmr * (1.0 - cosz) * o3_ratio * n_air

    return OxidantField(
        oh=jnp.maximum(oh, 0.0),
        no3=jnp.maximum(n_no3, 0.0),
        o3=jnp.maximum(n_o3, 0.0),
        h2o2=jnp.maximum(n_h2o2, 0.0),
    )


class PrescribedOxidants(PhysicsTerm):
    """Write the interim ``oxidants`` diagnostic for the sulfur chemistry."""

    name: ClassVar[str] = "jam_prescribed_oxidants"
    category: ClassVar[str] = "aerosol_oxidants"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    provides: ClassVar[tuple[str, ...]] = ("oxidants",)

    def __init__(self, params: OxidantParameters | None = None):
        """Hold the (differentiable) oxidant scales."""
        self.params = nnx.Param(params or OxidantParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        temperature = state.temperature
        pressure = diagnostics["pressure_full"]

        # O₃ and solar geometry are produced *after* the aerosol block, so read
        # them from the previous-step carry; fall back on step 1 / the probe.
        chemistry = diagnostics.get("chemistry")
        o3_vmr = (
            chemistry.ozone_vmr if chemistry is not None
            else jnp.full_like(temperature, params.o3_fallback_vmr * 1.0e6)
        )
        radiation = diagnostics.get("radiation")
        cosz = (
            jnp.broadcast_to(radiation.cos_zenith, temperature.shape)
            if radiation is not None
            else jnp.full_like(temperature, params.cos_zenith_fallback)
        )

        field = oxidant_field(temperature, pressure, o3_vmr, cosz, params)
        tendency = PhysicsTendency.zeros(temperature.shape)
        return tendency, {**diagnostics, "oxidants": field}
