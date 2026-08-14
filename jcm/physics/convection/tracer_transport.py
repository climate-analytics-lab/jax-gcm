"""``ConvectiveTracerTransport`` — bulk mass-flux tracer transport.

ECHAM transports every tracer through Tiedtke convection (the
``cuxtte``/``mo_cuascn`` xt budgeting) and CAM's ``convtran`` does the
same for constituents; jcm applied the convective mass fluxes only to
heat, moisture and momentum (#602 item 2). This term closes that gap for
an explicit tracer list using the profiles the Tiedtke term publishes
in ``ConvectionData``: the updraft mass flux at each layer's top
interface (``mass_flux_up``), the downdraft mass flux at each layer's
bottom interface (``mass_flux_down``), and the absolute per-layer
entrainment fluxes (``entrain_up``/``entrain_down``), all carrying the
scheme's rescale + cap ledger scaling, so tracer transport is
proportional to the heat and moisture transport actually applied.

The scheme is a bulk entraining/detraining plume with compensating
subsidence, plus the mirrored downdraft leg (jax-gcm#622):

* Per-layer detrainment is derived from plume continuity,
  ``D_k = max(E_raw_k − (M_k − M_{k+1}), 0)`` with the effective
  entrainment ``E_k = D_k + (M_k − M_{k+1})`` — this absorbs everything
  the mass-flux profile actually did (survival cuts, cloud-base supply:
  at cloud base the mass-flux jump appears as entrainment of that
  layer's air, which is exactly the physical picture).
* Updraft tracer concentration from an upward scan:
  ``q_up_k = (M_{k+1}·q_up_{k+1} + E_k·q_k) / (M_{k+1} + E_k)`` — a
  convex mix, so the plume concentration is bounded by the environment
  profile it entrained.
* The downdraft (ECHAM ``cudlfs``/``cuddraf``, CAM ``convtran``'s
  ``cond`` loop) is the mirror image: the same continuity derivation on
  the downdraft profile turns the level-of-free-sinking seed into
  entrainment of that layer's air and the surface taper into sub-cloud
  detrainment, and a downward scan carries the in-downdraft
  concentration. Deviation from the Fortran: ``cudlfs`` seeds the
  downdraft with a 50/50 updraft/wet-bulb-environment mix, which moves
  plume-processed air across without a matching debit in the updraft
  budget; entraining environment air instead keeps the column budget
  telescoping exactly (CAM's downdraft does the same — environment
  entrainment only, "no transformation or removal is applied in the
  downdraft").
* Environment tendency in flux form: detrainment source, entrainment
  sink, and the compensating advection between (downward at the updraft
  interface flux, upward at the downdraft interface flux). Column tracer
  mass is conserved exactly up to the scavenging sink (all flux sums
  telescope; each plume's own budget closes by construction).

In-plume scavenging (jax-gcm#621) follows CAM's ``aero_convproc``
(mirage2 form): the fraction of the plume's condensate converted to
precipitation in a layer sets a first-order removal,
``cdt = pdmfup / (M_u·q_cond)`` and removed fraction
``w·(1 − exp(−cdt))``, applied to the in-plume concentration inside the
ascent scan after entrainment mixing — so aerosol scavenged low in the
plume never detrains aloft. The updraft-area fraction cancels in
``cdt`` (CAM's Note1: the cam5 variant needs the unknown updraft
fraction; the mirage2 form does not). ``w`` is a per-tracer weight
(soluble/activatable modes only) times the differentiable
``scav_ratio``; the removed flux deposits at the surface (like CAM's
``dconudt_wetdep``; per-tracer surface fluxes are published under
``_conv_scav_flux`` for the JAM wetdep ledger, which retires its own
environment-profile convective in-cloud pathway when this one is
active — jax-gcm#621's double-counting reconciliation). Aerosol
resuspension by evaporating convective precip is not modelled (CAM's
``dcondt_prevap``) — the removed flux goes straight to the surface,
matching the existing wetdep treatment of convective scavenging.

Explicit stability: the per-column ledger is scaled so the combined
subsidence Courant number stays ≤ 1 (the scheme's own cloud-base CFL cap
makes this a no-op in practice).

Like the other cross-step consumers (``vertical_diffusion``), the
``convection`` diagnostic is read from the previous step's carry with a
no-op fallback when absent.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

#: Physical floor on plume mass flux [kg/m²/s] in divisions: below this
#: the plume carries nothing worth budgeting (~1e-4 mm/day of air), and a
#: physical floor keeps the guarded-division VJPs out of the float32
#: squared-underflow window.
_MF_FLOOR = 1.0e-10

#: In-plume condensate threshold for scavenging [kg/kg] — CAM
#: ``aero_convproc`` ``clw_cut``: below this the updraft is effectively
#: precipitation-free and the removal-rate division is unreliable.
_CLW_CUT = 1.0e-6


@tree_math.struct
class ConvTransportParameters:
    """Tunable knobs for convective tracer transport (differentiable)."""

    transport_scale: jnp.ndarray   # multiplies the mass-flux ledger
    scav_ratio: jnp.ndarray        # in-plume scavenging ratio for soluble
                                   # tracers [-] (CAM ``aqfrac`` analogue)

    @classmethod
    def default(cls) -> "ConvTransportParameters":
        # scav_ratio mirrors the JAM wetdep ``conv_scav_ratio`` default it
        # supersedes for transported species.
        return cls(
            transport_scale=jnp.asarray(1.0),
            scav_ratio=jnp.asarray(0.99),
        )


def convective_tracer_tendency(
    q: jnp.ndarray,        # (K, nlev, ncols) tracer stack
    mfu: jnp.ndarray,      # (nlev, ncols) updraft flux at layer TOP [kg/m²/s]
    entrain: jnp.ndarray,  # (nlev, ncols) per-layer updraft entrainment [kg/m²/s]
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    dt: jnp.ndarray,
    mfd: jnp.ndarray | None = None,          # (nlev, ncols) downdraft flux at
                                             # layer BOTTOM [kg/m²/s], ≤ 0
    entrain_down: jnp.ndarray | None = None,  # (nlev, ncols) per-layer
                                              # downdraft entrainment [kg/m²/s]
    scav_weights: jnp.ndarray | None = None,  # (K,) per-tracer removal weight
    precip_formation: jnp.ndarray | None = None,  # (nlev, ncols) updraft
                                                  # precip generation [kg/m²/s]
    plume_condensate: jnp.ndarray | None = None,  # (nlev, ncols) in-updraft
                                                  # qc+qi [kg/kg]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bulk-plume + subsidence tracer tendency and scavenged surface flux.

    Returns ``(dq, scav_flux)``: the environment tendency [.../s] shaped
    like ``q``, and the per-tracer scavenged surface flux
    ``(K, ncols)`` [tracer·kg/m²/s] satisfying
    ``sum_k(dq·dm) = −scav_flux`` exactly (zero without scavenging).
    """
    dm = air_density * layer_thickness
    mfu = jnp.maximum(mfu, 0.0)
    # No flux through the model top: a residual plume reaching the top
    # layer detrains there (via the continuity-derived D below), which is
    # what makes the column budget close EXACTLY rather than only when
    # the plume happens to terminate lower down.
    mfu = mfu.at[0].set(0.0)
    entrain = jnp.maximum(entrain, 0.0)

    # Continuity-derived detrainment and effective entrainment (see
    # module docstring). ``mfu_below`` is the flux entering the layer
    # from below (zero under the surface). Derived BEFORE the CFL guard:
    # the environment sink coefficient is (E_eff + mfu_below)·Δt/Δm, and
    # in a net-detrainment layer that references the flux from the layer
    # BELOW over this layer's OWN mass — a cross-level ratio a guard on
    # (mfu + E) per layer never forms. With thin layers aloft (generic in
    # hybrid coordinates) the wrong guard let the sink exceed 1 and drove
    # detrainment-layer tracers negative (adversarial review, confirmed
    # repro). All ledger arrays are positively homogeneous in the mass
    # fluxes, so scaling AFTER the derivation is exactly linear and
    # preserves continuity and column conservation.
    mfu_below = jnp.concatenate(
        [mfu[1:], jnp.zeros_like(mfu[:1])], axis=0
    )
    delta = mfu - mfu_below
    detrain = jnp.maximum(entrain - delta, 0.0)
    entrain_eff = detrain + delta                     # >= 0 by construction

    # Downdraft ledger (jax-gcm#622), mirrored: ``mfd[k]`` is the flux
    # leaving layer k through its BOTTOM interface (the downdraft scan's
    # convention — the taper halves then land in the two lowest layers
    # exactly as cuddraf's ``itopde`` split), so the flux entering from
    # above is ``mfd[k-1]``. Magnitudes throughout; the bottom layer's
    # outflow is forced to zero (no flux through the surface — a residual
    # detrains there via continuity, mirroring the model-top handling).
    if mfd is not None:
        md_out = jnp.maximum(-mfd, 0.0)
        md_out = md_out.at[-1].set(0.0)
        md_in = jnp.concatenate(
            [jnp.zeros_like(md_out[:1]), md_out[:-1]], axis=0
        )
        e_dn_raw = (
            jnp.maximum(entrain_down, 0.0)
            if entrain_down is not None else jnp.zeros_like(md_out)
        )
        delta_dn = md_out - md_in                     # E − D per layer
        detrain_dn = jnp.maximum(e_dn_raw - delta_dn, 0.0)
        entrain_dn = detrain_dn + delta_dn            # >= 0 by construction
    else:
        md_out = md_in = detrain_dn = entrain_dn = jnp.zeros_like(mfu)

    # Combined Courant guard: both legs remove environment air from layer
    # k at (E_up_eff + mfu_below) + (E_dn_eff + md_in); one shared scale
    # keeps the two circulations proportional.
    courant = jnp.max(
        (entrain_eff + mfu_below + entrain_dn + md_in) * dt / dm,
        axis=0, keepdims=True,
    )
    scale = jnp.minimum(1.0, 1.0 / jnp.maximum(courant, 1.0))
    mfu = mfu * scale
    mfu_below = mfu_below * scale
    detrain = detrain * scale
    entrain_eff = entrain_eff * scale
    md_out = md_out * scale
    md_in = md_in * scale
    detrain_dn = detrain_dn * scale
    entrain_dn = entrain_dn * scale

    # In-plume scavenging profile (jax-gcm#621, CAM aero_convproc
    # mirage2 form): cdt = precip formed / plume condensate flux is the
    # first-order removal exponent over the layer transit — the updraft
    # area fraction cancels. Gated exactly like CAM: only where the
    # plume holds condensate and precip actually forms.
    m_up = mfu_below + entrain_eff                    # post-mix plume flux
    if scav_weights is not None:
        pf = jnp.maximum(precip_formation, 0.0)
        cond_flux = m_up * jnp.maximum(plume_condensate, 0.0)
        cdt = jnp.where(
            (pf > 0.0) & (plume_condensate > _CLW_CUT) & (m_up > _MF_FLOOR),
            pf / jnp.maximum(cond_flux, _MF_FLOOR * _CLW_CUT),
            0.0,
        )
        base_frac = -jnp.expm1(-cdt)                  # ∈ [0, 1)
        w = jnp.clip(scav_weights, 0.0, 1.0)
    else:
        base_frac = jnp.zeros_like(m_up)
        w = jnp.zeros(q.shape[0], dtype=q.dtype)

    # Upward plume scan for the in-plume concentration (surface -> top),
    # removing the scavenged share right after entrainment mixing (CAM
    # applies dconudt_wetdep to conu inside the same ascent loop) so what
    # detrains aloft is the already-scavenged concentration.
    def ascend(q_up_below, xs):
        m_below_k, e_k, q_k, m_up_k, frac_k = xs
        denom = m_below_k + e_k
        q_mix = jnp.where(
            (denom > _MF_FLOOR)[jnp.newaxis],
            (m_below_k[jnp.newaxis] * q_up_below + e_k[jnp.newaxis] * q_k)
            / jnp.maximum(denom, _MF_FLOOR)[jnp.newaxis],
            q_k,
        )
        removed = w[:, jnp.newaxis] * frac_k[jnp.newaxis] * q_mix
        q_up_k = q_mix - removed
        r_k = m_up_k[jnp.newaxis] * removed           # (K, ncols) flux
        return q_up_k, (q_up_k, r_k)

    q_lev = jnp.moveaxis(q, 1, 0)                     # (nlev, K, ncols)
    _, (q_up_rev, r_rev) = jax.lax.scan(
        ascend,
        q_lev[-1],                                    # seeded, overwritten at base
        (mfu_below, entrain_eff, q_lev, m_up, base_frac),
        reverse=True,
    )
    q_up = jnp.moveaxis(q_up_rev, 0, 1)               # (K, nlev, ncols)
    scav_flux = jnp.sum(r_rev, axis=0)                # (K, ncols)

    # Downward plume scan for the in-downdraft concentration (top ->
    # surface) — cuddraf's tracer budget: mix the arriving flux with the
    # layer's entrained environment air; detrainment leaves at the mixed
    # concentration, so the plume budget closes exactly like the updraft.
    def descend(q_dn_above, xs):
        m_in_k, e_k, q_k = xs
        denom = m_in_k + e_k
        q_mix = jnp.where(
            (denom > _MF_FLOOR)[jnp.newaxis],
            (m_in_k[jnp.newaxis] * q_dn_above + e_k[jnp.newaxis] * q_k)
            / jnp.maximum(denom, _MF_FLOOR)[jnp.newaxis],
            q_k,
        )
        return q_mix, q_mix

    _, q_dn_lev = jax.lax.scan(
        descend,
        q_lev[0],                                     # seeded, overwritten at LFS
        (md_in, entrain_dn, q_lev),
    )
    q_dn = jnp.moveaxis(q_dn_lev, 0, 1)               # (K, nlev, ncols)

    # Compensating advection: environment air enters each layer from
    # above at the layer-top updraft flux and leaves to the layer below
    # at the layer-bottom flux; the downdraft drives the mirror-image
    # upward environment flow through its own interface fluxes. The top
    # layer's "from above" is itself (no flux through the model top) and
    # the bottom pads are dead (md_out[-1] = 0), which keeps the
    # telescoping sums — and hence column conservation — exact.
    q_above = jnp.concatenate([q[:, :1], q[:, :-1]], axis=1)
    q_below = jnp.concatenate([q[:, 1:], q[:, -1:]], axis=1)
    dq = (
        detrain[jnp.newaxis] * q_up
        - entrain_eff[jnp.newaxis] * q
        + mfu[jnp.newaxis] * q_above
        - mfu_below[jnp.newaxis] * q
        + detrain_dn[jnp.newaxis] * q_dn
        - entrain_dn[jnp.newaxis] * q
        + md_out[jnp.newaxis] * q_below
        - md_in[jnp.newaxis] * q
    ) / dm[jnp.newaxis]
    return dq, scav_flux


class ConvectiveTracerTransport(PhysicsTerm):
    """Updraft + downdraft + subsidence transport of an explicit tracer list."""

    name: ClassVar[str] = "convective_tracer_transport"
    category: ClassVar[str] = "tracer_transport"
    requires: ClassVar[tuple[str, ...]] = (
        "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        tracer_names: tuple[str, ...],
        params: ConvTransportParameters | None = None,
        scav_weights: tuple[float, ...] | None = None,
    ):
        """Hold the tracer list, params and per-tracer scavenging weights.

        ``scav_weights`` (aligned with ``tracer_names``) selects which
        tracers the in-plume scavenging acts on — 1 for soluble aerosol,
        0 for insoluble aerosol and gases; ``None`` disables scavenging
        entirely. The static mask multiplies the differentiable
        ``params.scav_ratio``.
        """
        if not tracer_names:
            raise ValueError(
                "ConvectiveTracerTransport needs a non-empty tracer list."
            )
        if scav_weights is not None and len(scav_weights) != len(tracer_names):
            raise ValueError(
                "scav_weights must align with tracer_names: got "
                f"{len(scav_weights)} weights for {len(tracer_names)} tracers."
            )
        self._tracer_names = tuple(tracer_names)
        self._scav_weights = (
            tuple(float(x) for x in scav_weights)
            if scav_weights is not None else None
        )
        self.params = nnx.Param(params or ConvTransportParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        conv = diagnostics.get("convection")
        zeros = jnp.zeros_like(state.temperature)
        if conv is None:
            tracer_tends = {nm: zeros for nm in self._tracer_names}
            scav_flux = jnp.zeros(
                (len(self._tracer_names),) + state.temperature.shape[1:],
                dtype=state.temperature.dtype,
            )
        else:
            dt = diagnostics.get("_dt_seconds", 1800.0)
            q = jnp.stack([
                state.tracers.get(nm, zeros) for nm in self._tracer_names
            ])
            if self._scav_weights is not None:
                scav_kwargs = dict(
                    scav_weights=(
                        jnp.asarray(self._scav_weights, dtype=q.dtype)
                        * params.scav_ratio
                    ),
                    precip_formation=conv.precip_formation,
                    plume_condensate=conv.qc_conv + conv.qi_conv,
                )
            else:
                scav_kwargs = {}
            dq, scav_flux = convective_tracer_tendency(
                q,
                params.transport_scale * conv.mass_flux_up,
                params.transport_scale * conv.entrain_up,
                diagnostics["air_density"],
                diagnostics["layer_thickness"],
                dt,
                mfd=params.transport_scale * conv.mass_flux_down,
                entrain_down=params.transport_scale * conv.entrain_down,
                **scav_kwargs,
            )
            tracer_tends = {
                nm: dq[k] for k, nm in enumerate(self._tracer_names)
            }
        if self._scav_weights is not None:
            # Per-tracer surface deposition of in-plume scavenged mass
            # [kg/m²/s], for downstream wet-deposition bookkeeping (the
            # JAM wetdep term folds these into the AeroCom ``wet_*``
            # fluxes). Underscore key: internal handoff, not a
            # user-facing output field. Published UNCONDITIONALLY (zeros
            # without a convection diagnostic): the diagnostics dict is a
            # ``lax.scan`` carry, so its key set must not depend on
            # whether the step had convection composed — the structural
            # probe runs without it and the carry structures must match.
            diagnostics = dict(diagnostics)
            diagnostics["_conv_scav_flux"] = {
                nm: scav_flux[k]
                for k, nm in enumerate(self._tracer_names)
                if self._scav_weights[k] > 0.0
            }

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
