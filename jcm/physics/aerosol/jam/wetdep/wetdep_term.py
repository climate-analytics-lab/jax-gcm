"""``WetScavenging`` — in-cloud and below-cloud aerosol scavenging.

Removal pathways for aerosol, all differentiable, driven by the per-level
precipitation process rates the cloud microphysics schemes actually
integrate (``CloudData.precip_formation_rate`` / ``precip_evaporation_rate``,
#499; the carrier flux is their own cumulative ledger, which deliberately
excludes the sedimenting cloud-ice flux — see ``__call__``). With a prognostic
cloud-borne phase (``spec.cloud_borne``, #602) the stratiform in-cloud
pathway acts on the cloud-borne tracers at the full condensate→precip
conversion rate and the interstitial tracers keep impaction + convective
processing; without one, the in-cloud pathway acts on interstitial aerosol
weighted by its per-mode activated fractions (the implicit M7/TOMAS-style
treatment):

* **In-cloud nucleation scavenging** — aerosol residing in cloud droplets is
  removed at the local rate cloud condensate converts to precipitation,
  ``precip_formation_rate / (qc + qi)`` (both grid-mean, so the cloudy-area
  factor cancels inside the ratio).
* **Below-cloud impaction scavenging** — precipitation falling through a
  layer collects aerosol in its clear-air part, with a size-dependent
  (∝ r²) collection efficiency. The stratiform contribution uses the
  per-level flux entering each layer, so washout is automatically confined
  below where precip actually forms; the convective contribution uses the
  surface convective precip masked to levels at/below the convective cloud
  top (diagnosed by pressure from the heating footprint).
* **Convective in-cloud scavenging** — the convective mirror of the
  stratiform pathway: scavenging ratio × (per-layer updraft precip
  formation / in-updraft condensate), from ``ConvectionData``'s
  ``precip_formation`` (ECHAM ``pdmfup``) and ``qc_conv``/``qi_conv``;
  activatable modes only.
* **Re-evaporation re-injection** — aerosol scavenged by the stratiform
  pathways is carried in the falling precip; where that precip evaporates
  or sublimates, the same fraction of the carried aerosol returns to the
  INTERSTITIAL phase (a droplet that evaporates releases its aerosol, so
  cloud-borne-scavenged material also re-enters as interstitial). The
  aerosol-in-precip flux is integrated top to bottom per tracer, exactly
  mirroring HAMMOZ ``mo_ham_wetdep``'s re-evaporation ledger; the flux
  reaching the bottom is the net surface wet deposition. Convectively
  scavenged aerosol is deposited directly (the convection scheme exposes no
  evaporation profile yet).

``ConvectionData`` is read via ``diagnostics.get("convection")`` with a
zero-precip fallback so the term still composes without a convection
scheme. A cloud scheme that does not populate the per-level process rates
(both ECHAM schemes do) yields zero stratiform scavenging altogether — the
in-cloud rate, the impaction carrier and the re-injection ledger all derive
from those two fields.

Mirrors ``mo_hammoz_wetdep``.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm

_EPS = 1.0e-30
# Physical floors for the re-injection budget's divisions. Values below
# these are dynamically negligible (a 1e-15 1/s removal rate is a ~30 Myr
# timescale; 1e-12 kg/m²/s is ~1e-4 mm/day), and a *physical* floor — not a
# tiny epsilon — keeps the guarded-division VJPs clear of the squared-
# underflow window in float32 (the double-where NaN class).
_RATE_FLOOR = 1.0e-15   # [1/s]
_FLUX_FLOOR = 1.0e-12   # [kg/m²/s]


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


#: Physical condensate floor for the in-cloud rate [kg/kg]. A cell with less
#: than this holds no cloud worth scavenging in; it is also what keeps the
#: division's VJP clear of the float32 squared-underflow window (a 1e-30
#: epsilon guard here gave a verified NaN gradient in every clear f32 cell:
#: the cotangent forms p_form/condensate², and 1e-60 flushes to zero).
_CONDENSATE_FLOOR = 1.0e-12


def in_cloud_rate(
    activated_fraction: jnp.ndarray,
    p_form: jnp.ndarray,
    qc: jnp.ndarray,
) -> jnp.ndarray:
    """In-cloud scavenging rate [1/s] applied to in-droplet aerosol."""
    rate = activated_fraction * p_form / jnp.maximum(qc, _CONDENSATE_FLOOR)
    return jnp.where(qc > _CONDENSATE_FLOOR, rate, 0.0)


def below_cloud_rate(
    precip_flux: jnp.ndarray,     # (nlev, ncols) precip falling through [kg/m²/s]
    cloud_fraction: jnp.ndarray,  # (nlev, ncols)
    r_wet: jnp.ndarray,
    params: WetDepParameters,
) -> jnp.ndarray:
    """Below-cloud impaction scavenging rate [1/s], size-dependent (∝ r²).

    ``precip_flux`` is the local flux falling through each layer (per-level
    profile for stratiform precip; a broadcast surface value is the interim
    convective treatment).
    """
    rain_mmph = precip_flux * 3600.0  # kg/m²/s -> mm/h
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


def reinjection_budget(
    scavenged_flux: jnp.ndarray,  # (K, nlev, ncols) removal into precip [kg/m²/s]
    evap_fraction: jnp.ndarray,   # (nlev, ncols) precip fraction evaporating
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aerosol-in-precip ledger: re-injected flux per layer + surface flux.

    Integrates the scavenged-aerosol flux top to bottom: entering each layer,
    the fraction of the carrier precip that evaporates there releases the
    same fraction of the carried aerosol (``reinjected``); the layer's own
    scavenging then joins the flux. Returns ``(reinjected, surface_flux)``
    with ``reinjected`` shaped like ``scavenged_flux`` and ``surface_flux``
    ``(K, ncols)``, satisfying column conservation
    ``sum_k(scavenged - reinjected) = surface_flux`` exactly.
    """
    def step(carried, xs):
        s_k, e_k = xs                       # (K, ncols), (ncols,)
        # The layer's own scavenging joins the ledger BEFORE the release,
        # as in HAMMOZ: in a fully-evaporating (virga) layer everything —
        # including aerosol impacted out within that layer — is released,
        # so nothing rides a terminated carrier to the surface through dry
        # air.
        total = carried + s_k
        released = total * e_k[jnp.newaxis, :]
        carried = total - released
        return carried, released

    k, _, ncols = scavenged_flux.shape
    surface, released = jax.lax.scan(
        step,
        jnp.zeros((k, ncols), scavenged_flux.dtype),
        (jnp.moveaxis(scavenged_flux, 1, 0), evap_fraction),
    )
    return jnp.moveaxis(released, 0, 1), surface


class WetScavenging(PhysicsTerm):
    """In-cloud + below-cloud scavenging with re-evaporation re-injection."""

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
        dm = air_density * dz
        # Timestep for the implicit (exponential) scavenging update below.
        dt = diagnostics.get("_dt_seconds", 1800.0)

        clouds = diagnostics["clouds"]
        cloud_fraction = clouds.cloud_fraction
        # Total cloud condensate: the formation ledger spans both the rain
        # (from qc) and snow (from qc and qi) pathways, so the in-droplet
        # residence pool is liquid + ice.
        condensate = jnp.maximum(clouds.qc, 0.0) + jnp.maximum(clouds.qi, 0.0)
        # Per-level stratiform process rates from the cloud scheme (#499):
        # the true local condensate→precip conversion and the falling-precip
        # evaporation. The carrier flux for impaction and the re-evap ledger
        # is rebuilt as the cumulative (formation − evaporation) integral of
        # those same rates rather than read from ``clouds.rain_flux +
        # snow_flux``: the schemes fold the sedimenting cloud-ice flux into
        # the frozen profile at interior levels, and that ice is not a
        # scavenging carrier (its sublimation is likewise excluded from
        # ``precip_evaporation_rate``), so using the profiles directly would
        # both drive impaction with a non-carrier flux and dilute the
        # re-evaporation fraction under cirrus. Melt only moves mass between
        # rain and snow, so the summed ledger is exact for the actual
        # precip; the floor guards accumulated round-off. Note the
        # ice-sedimentation flux that reaches the surface as snow therefore
        # carries no aerosol removal at all — a real missing sink, accepted
        # with the non-carrier stance.
        p_form = jnp.maximum(clouds.precip_formation_rate, 0.0)
        p_evap = jnp.maximum(clouds.precip_evaporation_rate, 0.0)
        strat_flux = jnp.maximum(
            jnp.cumsum((p_form - p_evap) * dm, axis=0), 0.0
        )
        flux_in = jnp.concatenate(
            [jnp.zeros_like(strat_flux[:1]), strat_flux[:-1]], axis=0
        )
        # Fraction of the precip entering a layer that evaporates within it,
        # guarded on a physical flux floor (see _FLUX_FLOOR note above).
        evap_fraction = jnp.where(
            flux_in > _FLUX_FLOOR,
            jnp.clip(p_evap * dm / jnp.maximum(flux_in, _FLUX_FLOOR), 0.0, 1.0),
            0.0,
        )

        # Convective precipitation (Tiedtke). Zero-precip fallback keeps the
        # term composable without a convection scheme (see module docstring).
        conv = diagnostics.get("convection")
        if conv is None:
            conv_precip = jnp.zeros_like(state.temperature[0])
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

        # The implicit stratiform rate is weighted by the cloudy area
        # fraction: the grid-mean p_form/condensate ratio is the IN-CLOUD
        # conversion rate (cf cancels between them), but only cf·af of the
        # grid-mean interstitial tracer is in droplets. ``in_cloud_rate`` is
        # linear in the activated fraction, so keep the unit-fraction base
        # and apply per-mode, per-quantity fractions below: ARG's number and
        # mass fractions differ a lot (large particles activate
        # preferentially) and vary by mode. The aggregate fraction is kept
        # only as a fallback for standalone composition without ARG
        # upstream.
        cf_clip = jnp.clip(cloud_fraction, 0.0, 1.0)
        rate_ic_unit = params.incloud_scale * cf_clip * in_cloud_rate(
            jnp.ones_like(state.temperature), p_form, condensate,
        )
        jam_act = diagnostics.get("_jam_activation")

        # Build per-tracer scavenging rates and stack with the matching
        # tracers, so the elementwise removal runs as one batched op (rather
        # than an unrolled tendency per mode×species). Stratiform and
        # convective rates are kept separate: only stratiform-scavenged
        # aerosol enters the re-evaporation ledger (its carrier's per-level
        # evaporation is known). ``state.tracers`` is empty during
        # ``Model.get_empty_data``'s structural probe, so fall back to zeros
        # there (real runs have every declared tracer seeded).
        zeros = jnp.zeros_like(state.temperature)
        names: list[str] = []
        # Interstitial destination for each stacked tracer's re-injected
        # aerosol: itself for interstitial tracers; the interstitial partner
        # for cloud-borne ones (an evaporated droplet releases its aerosol
        # to the interstitial phase).
        reinject_to: list[str] = []
        q_list: list[jnp.ndarray] = []
        rate_strat: list[jnp.ndarray] = []
        rate_conv: list[jnp.ndarray] = []
        # With a prognostic cloud-borne phase (``spec.cloud_borne``, #602) the
        # stratiform in-cloud (nucleation) pathway belongs to the cloud-borne
        # tracers, which sit in the droplets by definition: they are removed
        # at the full condensate→precip conversion rate, and the interstitial
        # tracers keep only impaction and convective processing (activated
        # aerosol first transfers via ``CloudBorneExchange``, then rains
        # out). Without it, the implicit treatment stands — the interstitial
        # tracers are scavenged by their per-mode activated fractions.
        explicit_cb = self._spec.cloud_borne
        rate_cb = params.incloud_scale * in_cloud_rate(
            jnp.ones_like(state.temperature), p_form, condensate,
        )
        for i, mode in enumerate(self._spec.modes):
            below_strat = below_cloud_rate(
                flux_in, cloud_fraction, aer.r_wet[i], params,
            )
            below_conv = conv_below * below_cloud_rate(
                conv_precip[jnp.newaxis, :], cloud_fraction,
                aer.r_wet[i], params,
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
                strat_num = below_strat + frac_num * rate_ic_unit
                strat_mass = below_strat + frac_mass * rate_ic_unit
                conv_rate = below_conv + rate_conv_incloud
            elif mode.can_activate:
                strat_num = strat_mass = below_strat
                conv_rate = below_conv + rate_conv_incloud
            else:
                strat_num = strat_mass = below_strat
                conv_rate = below_conv
            n_nm = number_name(mode.short)
            names.append(n_nm)
            reinject_to.append(n_nm)
            q_list.append(state.tracers.get(n_nm, zeros))
            rate_strat.append(strat_num)
            rate_conv.append(conv_rate)
            for sp in mode.species:
                nm = mass_name(sp, mode.short)
                names.append(nm)
                reinject_to.append(nm)
                q_list.append(state.tracers.get(nm, zeros))
                rate_strat.append(strat_mass)
                rate_conv.append(conv_rate)
            if explicit_cb:
                # Cloud-borne aerosol is entirely in-droplet: no below-cloud
                # impaction, no activated-fraction weighting, and its
                # re-injected share returns to the interstitial partner.
                pairs = [(number_name(mode.short, cloud_borne=True),
                          number_name(mode.short))] + [
                    (mass_name(sp, mode.short, cloud_borne=True),
                     mass_name(sp, mode.short))
                    for sp in mode.species
                ]
                for nm, partner in pairs:
                    names.append(nm)
                    reinject_to.append(partner)
                    q_list.append(state.tracers.get(nm, zeros))
                    rate_strat.append(rate_cb)
                    rate_conv.append(zeros)

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
        # Clamp the decay rates to ≥0 so the exponential update is always a
        # bounded removal (a scavenging rate is non-negative by construction).
        strat_arr = jnp.maximum(jnp.stack(rate_strat), 0.0)
        conv_arr = jnp.maximum(jnp.stack(rate_conv), 0.0)
        rate_arr = strat_arr + conv_arr
        removed_frac = -jnp.expm1(-rate_arr * dt)              # 1 - exp(-rate·dt) ∈ [0, 1]
        dq_stack = -(removed_frac * jnp.stack(q_list)) / dt

        # Re-evaporation re-injection (#499): the stratiform share of each
        # tracer's removal (proportional attribution between the two rate
        # families) rides the falling precip; ``reinjection_budget``
        # releases it where the carrier evaporates. Convectively scavenged
        # aerosol deposits directly (no convective evap profile yet).
        strat_share = jnp.where(
            rate_arr > _RATE_FLOOR,
            strat_arr / jnp.maximum(rate_arr, _RATE_FLOOR),
            0.0,
        )
        scavenged_flux = -dq_stack * strat_share * dm[jnp.newaxis]
        reinjected, _surface = reinjection_budget(
            scavenged_flux, evap_fraction,
        )
        reinject_tend = reinjected / dm[jnp.newaxis]

        tracer_tends = {nm: dq_stack[k] for k, nm in enumerate(names)}
        for k, target in enumerate(reinject_to):
            tracer_tends[target] = tracer_tends[target] + reinject_tend[k]

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        # AeroCom deposition fluxes (jax-gcm#581): this term's NET removal
        # (scavenging minus re-injection), column-integrated, accumulated
        # onto the per-step-reset keys.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            accumulate_deposition_fluxes)
        diagnostics = accumulate_deposition_fluxes(
            diagnostics, tracer_tends,
            diagnostics["air_density"], diagnostics["layer_thickness"],
            kind="wet")
        return tendency, diagnostics
