"""``WetScavenging`` — in-cloud and below-cloud aerosol scavenging.

Two removal pathways for interstitial aerosol, both differentiable and built
only from diagnostics the cloud scheme already exposes (so the cloud
microphysics terms are untouched):

* **In-cloud nucleation scavenging** — the activated fraction of aerosol is in
  cloud droplets and is removed at the rate cloud condensate converts to
  precipitation. As an interim, the per-level precip-formation rate is
  *reconstructed* by distributing the column surface precip
  (``CloudData.precip_rain/snow``) across the cloudy column weighted by
  in-cloud condensate; exposing the true per-level formation (and
  evaporation) rate from the cloud schemes — and adding re-evaporation
  re-injection — is tracked in #499.
* **Below-cloud impaction scavenging** — falling precipitation collects
  interstitial aerosol in clear air, with a size-dependent (∝ r²) collection
  efficiency so coarse particles are scavenged far faster than accumulation
  mode. Driven by the TOTAL precipitation (stratiform + convective): both
  fall through the same clear air below cloud.
* **Convective in-cloud scavenging** — HAMMOZ scavenges aerosol in raining
  convective columns with per-mode scavenging ratios applied to the
  convective precipitation formation. The Tiedtke port does not (yet) expose
  in-updraft condensate or the ``cloud_base``/``cloud_top`` indices (reserved
  zero-filled fields), so the removal is parameterised as first-order in the
  column convective rain intensity, applied on the convectively active
  layers — where ``ConvectionData.heating_rate > 0`` marks the condensing
  updraft — and only to activatable (soluble) modes, matching the stratiform
  in-cloud gating. Without this pathway the majority of tropical
  precipitation removed no aerosol at all, and mass lofted by the
  convectively driven circulation accumulated in the upper troposphere
  (observed after 200 days online: 43 % of the SO4 burden above 300 hPa,
  burdens 3–5× climatological anchors, AOD ~3× low).

``ConvectionData`` is read via ``diagnostics.get("convection")`` with a
zero-precip fallback (rather than a hard ``requires`` entry) so the term
still composes in setups without a convection scheme; in the ECHAM ordering
convection runs upstream of every aerosol term, so real runs always see the
current step's convective precipitation.

Mirrors ``mo_hammoz_wetdep``.
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
    conv_incloud_coeff: jnp.ndarray  # convective in-cloud Λ per mm/h [1/s]

    @classmethod
    def default(cls) -> "WetDepParameters":
        # conv_incloud_coeff = 5e-4 (1/s per mm/h): a 10 mm/h deep-convective
        # core gives Λ = 5e-3 1/s, so a 15-min step's implicit update removes
        # 1 − exp(−4.5) ≈ 99 % of soluble aerosol in the active layers —
        # matching HAMMOZ's ~0.99 in-cloud scavenging ratio per raining
        # convective pass — while light convective drizzle (0.5 mm/h) gives a
        # ~1 h removal timescale. Differentiable: a first-line calibration
        # target alongside ``incloud_scale``.
        return cls(
            incloud_scale=jnp.asarray(1.0),
            below_coeff=jnp.asarray(1.0e-4),
            below_radius_ref=jnp.asarray(1.0e-7),
            conv_incloud_coeff=jnp.asarray(5.0e-4),
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
    # Clear-sky (below-cloud) fraction, clipped to [0, 1]. The cloud scheme can
    # return cloud_fraction > 1 (e.g. where RH > 1), which would make this
    # fraction — and hence the scavenging rate — NEGATIVE. A negative rate makes
    # the implicit ``1 - exp(-rate·dt)`` removed fraction overflow to +inf,
    # NaN-ing every aerosol tracer. Scavenging rates are non-negative by
    # construction, so clip the clear fraction here.
    clear_fraction = jnp.clip(1.0 - cloud_fraction, 0.0, 1.0)
    return params.below_coeff * rain_mmph * clear_fraction * efficiency


def conv_in_cloud_rate(
    conv_precip_col: jnp.ndarray,  # (*horiz,) convective precip [kg/m²/s]
    conv_heating: jnp.ndarray,     # (nlev, *horiz) convective heating [K/s]
    params: WetDepParameters,
) -> jnp.ndarray:
    """Convective in-cloud (nucleation) scavenging rate [1/s].

    First-order in the column convective rain intensity, restricted to the
    convectively active layers (``heating_rate > 0``, i.e. where the updraft
    condenses — the only per-level footprint the Tiedtke port exposes today;
    switch to the true updraft condensate / cloud_base–cloud_top bounds when
    those diagnostics are ported). The linear-in-rain form mirrors
    ``below_cloud_rate`` but without the clear-sky factor (removal happens
    inside the convective cloud) and without the ∝r² impaction efficiency
    (nucleation scavenging is not size-selective in HAMMOZ's soluble-mode
    ratios). Non-negative by construction; bounded overall by the implicit
    exponential update in ``WetScavenging.__call__``.
    """
    rain_mmph = conv_precip_col[jnp.newaxis] * 3600.0  # kg/m²/s -> mm/h
    active = (conv_heating > 0.0).astype(conv_heating.dtype)
    return params.conv_incloud_coeff * rain_mmph * active


class WetScavenging(PhysicsTerm):
    """In-cloud + below-cloud scavenging of interstitial aerosol."""

    name: ClassVar[str] = "jam_wet_deposition"
    category: ClassVar[str] = "aerosol_wetdep"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "activated_fraction", "clouds",
        "air_density", "layer_thickness",
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
        # Timestep for the implicit (exponential) scavenging update below.
        dt = diagnostics.get("_dt_seconds", 1800.0)

        clouds = diagnostics["clouds"]
        precip_col = clouds.precip_rain + clouds.precip_snow
        cloud_fraction = clouds.cloud_fraction
        qc = clouds.qc

        # Convective precipitation (Tiedtke). Zero-precip fallback keeps the
        # term composable without a convection scheme (see module docstring).
        conv = diagnostics.get("convection")
        if conv is None:
            conv_precip = jnp.zeros_like(precip_col)
            rate_conv_incloud = jnp.zeros_like(state.temperature)
        else:
            conv_precip = conv.precip_conv
            rate_conv_incloud = conv_in_cloud_rate(
                conv_precip, conv.heating_rate, params,
            )

        p_form = precip_formation_rate(
            precip_col, cloud_fraction, qc, air_density, dz,
        )
        rate_incloud = params.incloud_scale * in_cloud_rate(
            activated_fraction, p_form, qc,
        ) + rate_conv_incloud

        # Everything that falls — stratiform and convective — washes out
        # interstitial aerosol below cloud.
        precip_total = precip_col + conv_precip

        # Build a per-tracer scavenging rate and stack with the matching
        # tracers, so the elementwise removal runs as one batched op (rather
        # than an unrolled tendency per mode×species). ``state.tracers`` is
        # empty during ``Model.get_empty_data``'s structural probe, so fall
        # back to zeros there (real runs have every declared tracer seeded).
        zeros = jnp.zeros_like(state.temperature)
        names: list[str] = []
        q_list: list[jnp.ndarray] = []
        rate_list: list[jnp.ndarray] = []
        for i, mode in enumerate(self._spec.modes):
            rate_below = below_cloud_rate(
                precip_total, cloud_fraction, aer.r_wet[i], params,
            )
            # In-cloud only removes from activatable (soluble) modes.
            rate = rate_below + (rate_incloud if mode.can_activate else 0.0)
            for nm in [number_name(mode.short)] + [
                mass_name(sp, mode.short) for sp in mode.species
            ]:
                names.append(nm)
                q_list.append(state.tracers.get(nm, zeros))
                rate_list.append(rate)

        # Implicit (exponential) scavenging over the step: q(t+dt) = q·exp(-rate·dt).
        # The first-order-decay rate is unbounded — the in-cloud rate ∝ 1/qc
        # diverges in near-clear cells and the below-cloud rate ∝ (r_wet/r_ref)²
        # is large for the coarse mode — so an explicit ``dq = -rate·q`` step
        # removes far more than the available mass when ``rate·dt ≫ 1`` (observed
        # ``rate·dt ~ 1e4`` for coarse sea salt over the high-wind Southern
        # Ocean), overshooting into a sign-flipped runaway that NaNs the model
        # in a few steps. The analytic exponential of the decay is unconditionally
        # stable and positivity-preserving for any ``rate ≥ 0`` (HAMMOZ
        # ``mo_ham_wetdep`` applies the same ``1 - exp(-Λ·Δt)`` removed fraction).
        # Emitted as a per-second tendency so the operator-split sum + dynamics
        # apply exactly ``q·(exp(-rate·dt) - 1)`` over the step.
        # Clamp the decay rate to ≥0 so the exponential update is always a
        # bounded removal (a scavenging rate is non-negative by construction).
        rate_arr = jnp.maximum(jnp.stack(rate_list), 0.0)
        removed_frac = -jnp.expm1(-rate_arr * dt)              # 1 - exp(-rate·dt) ∈ [0, 1]
        dq_stack = -(removed_frac * jnp.stack(q_list)) / dt
        tracer_tends = {nm: dq_stack[k] for k, nm in enumerate(names)}

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
