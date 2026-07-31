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
  mode. Driven by stratiform AND convective precipitation — with the
  convective contribution masked to levels at/below the convective cloud
  top (diagnosed by pressure from the heating footprint), since rain
  cannot collect aerosol above where it forms.
* **Convective in-cloud scavenging** — first-order in the column convective
  rain intensity on the convectively active layers (``heating_rate > 0``),
  soluble modes only: a proxy for HAMMOZ's per-mode convective scavenging
  ratios until the Tiedtke port exposes in-updraft condensate.

``ConvectionData`` is read via ``diagnostics.get("convection")`` with a
zero-precip fallback so the term still composes without a convection
scheme.

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
        # conv_incloud_coeff: Λ per mm/h of convective rain. 5e-4 removes
        # ~99 % of soluble aerosol per 15-min step in a 10 mm/h core.
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

    First-order in column convective rain intensity on the convectively
    active layers (``heating_rate > 0``). No clear-sky factor (removal is
    inside the cloud) and no ∝r² efficiency (nucleation, not impaction);
    bounded by the implicit exponential update in ``__call__``.
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
            conv_below = jnp.zeros_like(state.temperature)
        else:
            conv_precip = conv.precip_conv
            rate_conv_incloud = conv_in_cloud_rate(
                conv_precip, conv.heating_rate, params,
            )
            # Convective washout acts only at/below the convective cloud
            # top — rain cannot collect aerosol above where it forms. The
            # top is the lowest-pressure convectively active level
            # (orientation-agnostic); columns with no active layer get an
            # all-zero mask (min over empty set = +inf).
            p_full = diagnostics.get("pressure_full")
            if p_full is not None:
                active = conv.heating_rate > 0.0
                p_conv_top = jnp.min(
                    jnp.where(active, p_full, jnp.inf), axis=0, keepdims=True,
                )
                conv_below = (p_full >= p_conv_top).astype(p_full.dtype)
            else:
                # No pressure diagnostic: column-wide washout, not none.
                conv_below = jnp.ones_like(state.temperature)

        p_form = precip_formation_rate(
            precip_col, cloud_fraction, qc, air_density, dz,
        )
        rate_incloud = params.incloud_scale * in_cloud_rate(
            activated_fraction, p_form, qc,
        ) + rate_conv_incloud

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
            # Stratiform washout column-wide (interim, #499); convective
            # only below the convective cloud top.
            rate_below = below_cloud_rate(
                precip_col, cloud_fraction, aer.r_wet[i], params,
            ) + conv_below * below_cloud_rate(
                conv_precip, cloud_fraction, aer.r_wet[i], params,
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
