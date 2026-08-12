"""``ConvectiveTracerTransport`` — bulk mass-flux tracer transport.

ECHAM transports every tracer through Tiedtke convection (the
``cuxtte``/``mo_cuascn`` xt budgeting) and CAM's ``convtran`` does the
same for constituents; jcm applied the convective mass fluxes only to
heat, moisture and momentum (#602 item 2). This term closes that gap for
an explicit tracer list using the profiles the Tiedtke term now publishes
in ``ConvectionData``: the updraft mass flux at each layer's top
interface (``mass_flux_up``) and the absolute per-layer entrainment flux
(``entrain_up``), both carrying the scheme's rescale + cap ledger
scaling, so tracer transport is proportional to the heat and moisture
transport actually applied.

The scheme is a bulk entraining/detraining plume with compensating
subsidence:

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
* Environment tendency in flux form: detrainment source, entrainment
  sink, and compensating subsidence between (downward advection of
  environment air at the interface updraft flux). Column tracer mass is
  conserved exactly (the subsidence fluxes telescope; the plume's own
  budget closes by construction).

Downdraft tracer transport is not included yet (``mass_flux_down`` is
published for it; the downdraft entrainment ledger is not) — the
documented follow-up, typically a ~30% correction on the updraft
circulation. Likewise not included: in-plume scavenging of soluble
aerosol (ECHAM ``cuscav`` removes a large fraction of what the updraft
carries before it detrains; here the plume is conservative and the
convective wet removal lives separately in the JAM wetdep term), so
expect a high bias in upper-tropospheric soluble aerosol in deep
convective regions until that lands. Explicit stability: the per-column ledger is scaled so the
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


@tree_math.struct
class ConvTransportParameters:
    """Tunable knob for convective tracer transport (differentiable)."""

    transport_scale: jnp.ndarray   # multiplies the mass-flux ledger

    @classmethod
    def default(cls) -> "ConvTransportParameters":
        return cls(transport_scale=jnp.asarray(1.0))


def convective_tracer_tendency(
    q: jnp.ndarray,        # (K, nlev, ncols) tracer stack
    mfu: jnp.ndarray,      # (nlev, ncols) updraft flux at layer TOP [kg/m²/s]
    entrain: jnp.ndarray,  # (nlev, ncols) per-layer entrainment flux [kg/m²/s]
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Bulk-plume + compensating-subsidence tracer tendency [.../s]."""
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
    # repro). All four ledger arrays are positively homogeneous in
    # (mfu, entrain), so scaling AFTER the derivation is exactly linear
    # and preserves continuity and column conservation.
    mfu_below = jnp.concatenate(
        [mfu[1:], jnp.zeros_like(mfu[:1])], axis=0
    )
    delta = mfu - mfu_below
    detrain = jnp.maximum(entrain - delta, 0.0)
    entrain_eff = detrain + delta                     # >= 0 by construction

    courant = jnp.max(
        (entrain_eff + mfu_below) * dt / dm, axis=0, keepdims=True,
    )
    scale = jnp.minimum(1.0, 1.0 / jnp.maximum(courant, 1.0))
    mfu = mfu * scale
    mfu_below = mfu_below * scale
    detrain = detrain * scale
    entrain_eff = entrain_eff * scale

    # Upward plume scan for the in-plume concentration (surface -> top).
    def ascend(q_up_below, xs):
        m_below_k, e_k, q_k = xs                      # (ncols,), (ncols,), (K, ncols)
        denom = m_below_k + e_k
        q_mix = jnp.where(
            (denom > _MF_FLOOR)[jnp.newaxis],
            (m_below_k[jnp.newaxis] * q_up_below + e_k[jnp.newaxis] * q_k)
            / jnp.maximum(denom, _MF_FLOOR)[jnp.newaxis],
            q_k,
        )
        return q_mix, q_mix

    q_lev = jnp.moveaxis(q, 1, 0)                     # (nlev, K, ncols)
    _, q_up_rev = jax.lax.scan(
        ascend,
        q_lev[-1],                                    # seeded, overwritten at base
        (mfu_below, entrain_eff, q_lev),
        reverse=True,
    )
    q_up = jnp.moveaxis(q_up_rev, 0, 1)               # (K, nlev, ncols)

    # Compensating subsidence: environment air enters each layer from
    # above at the layer-top updraft flux and leaves to the layer below
    # at the layer-bottom flux. The top layer's "from above" is itself
    # (no flux through the model top), which keeps the telescoping sum —
    # and hence column conservation — exact.
    q_above = jnp.concatenate([q[:, :1], q[:, :-1]], axis=1)
    dq = (
        detrain[jnp.newaxis] * q_up
        - entrain_eff[jnp.newaxis] * q
        + mfu[jnp.newaxis] * q_above
        - mfu_below[jnp.newaxis] * q
    ) / dm[jnp.newaxis]
    return dq


class ConvectiveTracerTransport(PhysicsTerm):
    """Updraft + compensating-subsidence transport of an explicit tracer list."""

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
    ):
        """Hold the tracer list and params."""
        if not tracer_names:
            raise ValueError(
                "ConvectiveTracerTransport needs a non-empty tracer list."
            )
        self._tracer_names = tuple(tracer_names)
        self.params = nnx.Param(params or ConvTransportParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        conv = diagnostics.get("convection")
        zeros = jnp.zeros_like(state.temperature)
        if conv is None:
            tracer_tends = {nm: zeros for nm in self._tracer_names}
        else:
            dt = diagnostics.get("_dt_seconds", 1800.0)
            q = jnp.stack([
                state.tracers.get(nm, zeros) for nm in self._tracer_names
            ])
            dq = convective_tracer_tendency(
                q,
                params.transport_scale * conv.mass_flux_up,
                params.transport_scale * conv.entrain_up,
                diagnostics["air_density"],
                diagnostics["layer_thickness"],
                dt,
            )
            tracer_tends = {
                nm: dq[k] for k, nm in enumerate(self._tracer_names)
            }

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
