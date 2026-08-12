"""``WetScavenging`` — in-cloud and below-cloud aerosol scavenging.

Two removal pathways for aerosol, both differentiable and built
only from diagnostics the cloud scheme already exposes (so the cloud
microphysics terms are untouched). With a prognostic cloud-borne phase
(``spec.cloud_borne``, #602) the stratiform in-cloud pathway acts on the
cloud-borne tracers at the full condensate→precip conversion rate and the
interstitial tracers keep impaction + convective processing; without one,
the in-cloud pathway acts on interstitial aerosol weighted by its activated
fraction (the implicit M7/TOMAS-style treatment):

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
* **Convective in-cloud scavenging** — the convective mirror of the
  stratiform pathway: scavenging ratio × (per-layer updraft precip
  formation / in-updraft condensate), from ``ConvectionData``'s
  ``precip_formation`` (ECHAM ``pdmfup``) and ``qc_conv``/``qi_conv``;
  soluble modes only.

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
    conv_scav_ratio: jnp.ndarray   # convective in-cloud scavenging ratio [-]

    @classmethod
    def default(cls) -> "WetDepParameters":
        # conv_scav_ratio: fraction of soluble aerosol removed with the
        # condensate-to-precip conversion (HAMMOZ soluble-mode value).
        return cls(
            incloud_scale=jnp.asarray(1.0),
            below_coeff=jnp.asarray(1.0e-4),
            below_radius_ref=jnp.asarray(1.0e-7),
            conv_scav_ratio=jnp.asarray(0.99),
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
    precip_formation: jnp.ndarray,  # (nlev, *horiz) updraft precip gen [kg/m²/s]
    conv_condensate: jnp.ndarray,   # (nlev, *horiz) in-updraft qc+qi [kg/kg]
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    params: WetDepParameters,
) -> jnp.ndarray:
    """Convective in-cloud (nucleation) scavenging rate [1/s].

    The convective mirror of ``in_cloud_rate``: scavenging ratio × (local
    condensate→precip conversion rate / in-updraft condensate), with the
    per-layer formation flux converted to a mixing-ratio rate by ρ·Δz.
    Zero wherever the updraft carries no condensate.
    """
    local_form = jnp.maximum(precip_formation, 0.0) / (
        air_density * layer_thickness
    )
    qcond = jnp.maximum(conv_condensate, _EPS)
    rate = params.conv_scav_ratio * local_form / qcond
    return jnp.where(conv_condensate > 1.0e-12, rate, 0.0)


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
            conv_condensate = conv.qc_conv + conv.qi_conv
            rate_conv_incloud = conv_in_cloud_rate(
                conv.precip_formation, conv_condensate,
                air_density, dz, params,
            )
            # Convective washout acts only at/below the convective cloud
            # top — rain cannot collect aerosol above where it forms. The
            # top is the lowest-pressure level with in-updraft condensate
            # (orientation-agnostic); no convective cloud -> all-zero mask
            # (min over empty set = +inf).
            p_full = diagnostics.get("pressure_full")
            if p_full is not None:
                active = conv_condensate > 1.0e-12
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
        # The implicit stratiform rate is weighted by the cloudy area
        # fraction: the grid-mean p_form/qc ratio is the IN-CLOUD conversion
        # rate (cf cancels between them), but only cf·af of the grid-mean
        # interstitial tracer is in droplets. Without the cf factor broken
        # cloud (cf ~ 0.3) over-scavenged ~3x, and the implicit and explicit
        # cloud-borne representations would disagree at exchange equilibrium
        # for reasons that have nothing to do with the representation (#602).
        cf_clip = jnp.clip(cloud_fraction, 0.0, 1.0)
        # ``in_cloud_rate`` is linear in the activated fraction, so keep the
        # unit-fraction base and apply per-mode, per-quantity fractions
        # below: ARG's number and mass fractions differ a lot (large
        # particles activate preferentially) and vary by mode, and using
        # the aggregate number-weighted fraction for everything biases the
        # implicit representation away from the explicit one at exchange
        # equilibrium. The aggregate is kept only as a fallback for
        # standalone composition without the ARG term upstream.
        rate_ic_unit = params.incloud_scale * cf_clip * in_cloud_rate(
            jnp.ones_like(state.temperature), p_form, qc,
        )
        jam_act = diagnostics.get("_jam_activation")

        # Build a per-tracer scavenging rate and stack with the matching
        # tracers, so the elementwise removal runs as one batched op (rather
        # than an unrolled tendency per mode×species). ``state.tracers`` is
        # empty during ``Model.get_empty_data``'s structural probe, so fall
        # back to zeros there (real runs have every declared tracer seeded).
        zeros = jnp.zeros_like(state.temperature)
        names: list[str] = []
        q_list: list[jnp.ndarray] = []
        rate_list: list[jnp.ndarray] = []
        # With a prognostic cloud-borne phase (``spec.cloud_borne``, #602) the
        # stratiform in-cloud (nucleation) pathway belongs to the cloud-borne
        # tracers, which sit in the droplets by definition: they are removed
        # at the full condensate→precip conversion rate, and the interstitial
        # tracers keep only impaction and convective processing (activated
        # aerosol first transfers via ``CloudBorneExchange``, then rains
        # out). Without it, the current implicit treatment stands — the
        # interstitial tracers are scavenged by their activated fraction.
        explicit_cb = self._spec.cloud_borne
        rate_cb = params.incloud_scale * in_cloud_rate(
            jnp.ones_like(state.temperature), p_form, qc,
        )
        for i, mode in enumerate(self._spec.modes):
            # Stratiform washout column-wide (interim, #499); convective
            # only below the convective cloud top.
            rate_below = below_cloud_rate(
                precip_col, cloud_fraction, aer.r_wet[i], params,
            ) + conv_below * below_cloud_rate(
                conv_precip, cloud_fraction, aer.r_wet[i], params,
            )
            # In-cloud only removes from activatable (soluble) modes — and
            # only implicitly (via the activated fraction) when there is no
            # explicit cloud-borne phase to carry it. Convective processing
            # always acts on interstitial (updrafts ingest environment air).
            if mode.can_activate and not explicit_cb:
                if jam_act is not None:
                    frac_num = jam_act.number_frac[i]
                    frac_mass = jam_act.mass_frac[i]
                else:
                    frac_num = frac_mass = activated_fraction
                rate_num = rate_below + rate_conv_incloud + (
                    frac_num * rate_ic_unit
                )
                rate_mass = rate_below + rate_conv_incloud + (
                    frac_mass * rate_ic_unit
                )
            elif mode.can_activate:
                rate_num = rate_mass = rate_below + rate_conv_incloud
            else:
                rate_num = rate_mass = rate_below
            names.append(number_name(mode.short))
            q_list.append(state.tracers.get(number_name(mode.short), zeros))
            rate_list.append(rate_num)
            for nm in [mass_name(sp, mode.short) for sp in mode.species]:
                names.append(nm)
                q_list.append(state.tracers.get(nm, zeros))
                rate_list.append(rate_mass)
            if explicit_cb:
                # Cloud-borne aerosol is entirely in-droplet: no below-cloud
                # impaction, no activated-fraction weighting.
                for nm in [number_name(mode.short, cloud_borne=True)] + [
                    mass_name(sp, mode.short, cloud_borne=True)
                    for sp in mode.species
                ]:
                    names.append(nm)
                    q_list.append(state.tracers.get(nm, zeros))
                    rate_list.append(rate_cb)

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
        # AeroCom deposition fluxes (jax-gcm#581): this term's removal,
        # column-integrated, accumulated onto the per-step-reset keys.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            accumulate_deposition_fluxes)
        diagnostics = accumulate_deposition_fluxes(
            diagnostics, tracer_tends,
            diagnostics["air_density"], diagnostics["layer_thickness"],
            kind="wet")
        return tendency, diagnostics
