"""``ConvectiveTracerTransport`` — bulk mass-flux tracer transport.

ECHAM transports every tracer through Tiedtke convection (the
``cuxtte``/``mo_cuascn`` xt budgeting) and CAM's ``convtran`` does the
same for constituents; jcm applied the convective mass fluxes only to
heat, moisture and momentum (#602 item 2). This term closes that gap for
an explicit tracer list using the profiles the Tiedtke term publishes
in ``ConvectionData``: the updraft and downdraft mass fluxes at each
layer's TOP interface (``mass_flux_up``/``mass_flux_down``, ECHAM's
half-level ``pmfu``/``pmfd``; no flux crosses the surface), and the
absolute per-layer entrainment fluxes (``entrain_up``/``entrain_down``),
all carrying the
scheme's rescale + cap ledger scaling, so tracer transport is
proportional to the heat and moisture transport actually applied.

The scheme is a bulk entraining/detraining plume with compensating
subsidence, plus the mirrored downdraft leg (jax-gcm#622):

* Per-layer detrainment is derived from plume continuity,
  ``D_k = max(E_raw_k − (M_k − M_{k+1}), 0)`` with the effective
  entrainment ``E_k = D_k + (M_k − M_{k+1})`` — this absorbs everything
  the mass-flux profile actually did (survival cuts, cloud-base supply:
  below cloud base the published flux carries cuflx's linear-in-pressure
  sub-cloud taper, so each sub-cloud layer supplies air in proportion to
  its mass — ECHAM's ``pmfuxt`` taper).
* Updraft tracer concentration from an upward scan in cuasc's flux form
  (mo_cuascent.f90:421-424): detrained air leaves at the incoming
  concentration and the continuing flux carries
  ``M_k·q_up_k = (M_{k+1} − D_k)·q_up_{k+1} + E_k·q_k`` — a convex mix,
  so the plume concentration is bounded by the environment profile it
  entrained.
* The downdraft (ECHAM ``cudlfs``/``cuddraf``, CAM ``convtran``'s
  ``cond`` loop) is the mirror image: the same continuity derivation on
  the downdraft profile turns the level-of-free-sinking seed into
  entrainment of that layer's air and the surface taper into sub-cloud
  detrainment, and a downward scan in cuddraf's flux form carries the
  in-downdraft concentration. Deviation from the Fortran: ``cudlfs`` seeds the
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

In-plume scavenging (jax-gcm#621) takes ECHAM-HAM's parameters, inputs
and processes (``mo_cufluxdts.f90::cuflx`` → ``cuflx_subm`` →
``mo_hammoz_wetdep.f90::wetdep_interface`` → ``mo_ham_wetdep.f90::
ham_wetdep``/``ic_scav``) inside a closed plume budget:

* **Activation.** A per-tracer fraction ``csr`` of the aerosol entering the
  cloudy plume is in the condensate — HAMMOZ's convective in-droplet
  fraction ``csr_conv`` of the tracer's mode. It is taken once, where the
  aerosol meets condensate: the whole plume at the first level holding
  condensate, the air entrained at each such level above. The ``1 − csr``
  that is not in the condensate rides the plume up.
* **Removal.** At each level the activated share loses the fraction of the
  plume condensate converted to precipitation there,
  ``ConvectionData.precip_efficiency`` — HAMMOZ's ``peff =
  pmrateprecip/pmwc`` from cuasc's condensate before and after conversion,
  per phase as ``prep_wetdep_hydro`` forms it. The removal happens inside
  the ascent scan, so the plume carries the scavenged concentration up and
  detrains it.
* **Release.** The removed aerosol falls with the precipitation, top to
  bottom; at each level the running total loses
  ``ConvectionData.precip_evap_fraction`` of itself back to the environment
  there — HAMMOZ's ``prevap``, the fraction of the falling precipitation
  that evaporates or sublimates (``ham_wetdep``'s ``zdxtevapic``). What
  reaches the surface is the wet deposition, published per tracer under
  ``_conv_scav_flux`` for the JAM ``wet_*`` ledger (``WetScavenging``
  retires its own environment-profile convective in-cloud pathway when
  this term is composed, jax-gcm#621).

HAMMOZ applies ``csr_conv·peff`` after the ascent to the unscavenged
``pxtu`` and overwrites the updraft's deviation flux with the total one
(``ham_wetdep``'s ``pmfuxt = pxtp1c·pmfu``), which drops the compensating
subsidence of every wet-deposited tracer and lets a level remove more than
the plume carries; ``xt_conv_massfix`` restores the column total but not
positivity. Here the removal acts on the plume's own concentration, so the
column budget closes exactly and positively and no mass fixer is needed
(``docs/source/science/aerosol.md`` records the comparison).

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

#: In-plume condensate below which a level holds no cloud [kg/kg] —
#: HAMMOZ ``prep_wetdep_hydro``'s ``zmin`` (mo_hammoz_wetdep.f90:409).
_COND_MIN = 1.0e-10


@tree_math.struct
class ConvTransportParameters:
    """Tunable knobs for convective tracer transport (differentiable)."""

    transport_scale: jnp.ndarray   # multiplies the mass-flux ledger
    csr_conv: jnp.ndarray          # (K,) per-tracer fraction of the aerosol
                                   # entering the cloudy plume that is in
                                   # the condensate [-] (HAMMOZ ``csr_conv``)

    @classmethod
    def default(cls, csr_conv=()) -> "ConvTransportParameters":
        """Build unit transport scale with the given per-tracer ``csr_conv``."""
        return cls(
            transport_scale=jnp.asarray(1.0),
            csr_conv=jnp.asarray(csr_conv, dtype=jnp.float32),
        )


def release_scavenged(removed: jnp.ndarray, evap_fraction: jnp.ndarray
                      ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Carry scavenged aerosol down with the precipitation, releasing it.

    HAMMOZ's re-evaporation ledger (``mo_ham_wetdep.f90::ham_wetdep``,
    lines 329-373): top to bottom, the level's removal joins the running
    deposition total, and ``prevap`` of that total returns to the
    environment at the same level. ``removed`` is the per-level removal
    flux ``(K, nlev, *horiz)`` [tracer·kg/m²/s] on the top-first axis,
    ``evap_fraction`` the per-level ``prevap`` ``(nlev, *horiz)``.

    Returns ``(released, surface)``: the per-level release flux shaped like
    ``removed`` and the flux reaching the surface ``(K, *horiz)``, with
    ``sum(removed) = sum(released) + surface`` exactly.
    """
    e = jnp.clip(evap_fraction, 0.0, 1.0)

    def fall(total, xs):
        r_k, e_k = xs
        total = total + r_k
        rel = e_k[jnp.newaxis] * total
        return total - rel, rel

    surface, released = jax.lax.scan(
        fall, jnp.zeros_like(removed[:, 0]),
        (jnp.moveaxis(removed, 1, 0), e),
    )
    return jnp.moveaxis(released, 0, 1), surface


def convective_tracer_tendency(
    q: jnp.ndarray,        # (K, nlev, ncols) tracer stack
    mfu: jnp.ndarray,      # (nlev, ncols) updraft flux at layer TOP [kg/m²/s]
    entrain: jnp.ndarray,  # (nlev, ncols) per-layer updraft entrainment [kg/m²/s]
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    dt: jnp.ndarray,
    mfd: jnp.ndarray | None = None,          # (nlev, ncols) downdraft flux at
                                             # layer TOP [kg/m²/s], ≤ 0
    entrain_down: jnp.ndarray | None = None,  # (nlev, ncols) per-layer
                                              # downdraft entrainment [kg/m²/s]
    csr_conv: jnp.ndarray | None = None,  # (K,) in-condensate fraction
    precip_efficiency: jnp.ndarray | None = None,  # (nlev, ncols) HAMMOZ
                                                   # ``peff`` [-]
    plume_condensate: jnp.ndarray | None = None,  # (nlev, ncols) in-updraft
                                                  # qc+qi [kg/kg]
    evap_fraction: jnp.ndarray | None = None,  # (nlev, ncols) HAMMOZ
                                               # ``prevap`` [-]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bulk-plume + subsidence tracer tendency and scavenged surface flux.

    Returns ``(dq, scav_flux)``: the environment tendency [.../s] shaped
    like ``q``, and the per-tracer wet deposition at the surface
    ``(K, ncols)`` [tracer·kg/m²/s] — removal minus release — satisfying
    ``sum_k(dq·dm) = −scav_flux`` exactly (zero without scavenging).
    Scavenging acts when ``csr_conv`` is given, with ``precip_efficiency``
    and ``plume_condensate``; ``evap_fraction`` (optional) releases.
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
    # ENTERING layer k through its TOP interface (ECHAM's half-level
    # ``pmfd``, the same interface as ``mfu[k]``), so the flux leaving
    # through its bottom is ``mfd[k+1]`` — zero out of the bottom layer: no
    # flux crosses the surface, and the cuddraf taper's residual at the top
    # of the lowest layer detrains there via continuity. Magnitudes
    # throughout.
    if mfd is not None:
        md_in = jnp.maximum(-mfd, 0.0)
        md_out = jnp.concatenate(
            [md_in[1:], jnp.zeros_like(md_in[:1])], axis=0
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

    # In-plume scavenging profile: HAMMOZ's per-level conversion fraction
    # of the plume condensate, on the flux that continues through the
    # layer's top interface (the flux it was diagnosed against); a layer
    # holds cloud where that plume carries condensate above ``zmin``.
    if csr_conv is not None:
        live_up = mfu > _MF_FLOOR
        base_frac = jnp.where(
            live_up, jnp.clip(precip_efficiency, 0.0, 1.0), 0.0)
        cloudy = (plume_condensate > _COND_MIN) & live_up
        w = jnp.clip(csr_conv, 0.0, 1.0)
    else:
        base_frac = jnp.zeros_like(mfu)
        cloudy = jnp.zeros(mfu.shape, dtype=bool)
        w = jnp.zeros(q.shape[0], dtype=q.dtype)

    # Upward plume scan (surface -> top). Within one layer k, crossed from
    # its bottom interface (flux ``M_{k+1}``) to its top (``M_k``), the
    # order is ECHAM-HAM's:
    #
    # 1. Entrainment and detrainment, in flux form: the layer's entrained
    #    air joins and its detrained air leaves at the INCOMING plume
    #    concentration, so the continuing flux carries
    #    ``M_k·x_k = M_{k+1}·x_{k+1} + E·q_k − D·x_{k+1}``
    #    (mo_cuascent.f90:421-424; the downdraft likewise,
    #    mo_cudescent.f90:293-296). Detrained air has seen none of this
    #    layer's conversion. Where continuity has the layer detrain more
    #    than arrived from below (the terminating layer's whole entrainment
    #    is published), the excess is entrained air leaving again at the
    #    environment's value.
    # 2. The entrained (fresh) aerosol of a layer holding condensate joins
    #    it at ``csr`` — the whole plume at the first such layer.
    # 3. The layer converts ``peff`` of the continuing plume's condensate to
    #    precipitation at its top interface (cuasc 446-462, on ``pmfu(jk)``)
    #    and the same fraction of the aerosol in the condensate leaves with
    #    it: ``zdep = pxtu·csr_conv·peff·pmfu(jk)`` (mo_ham_wetdep.f90:250,
    #    325, 545-555), i.e. removal from the CONTINUING flux ``M_k``.
    # 4. The continuing plume carries the scavenged concentration into the
    #    next layer, so removal compounds on the share in the condensate;
    #    the ``1 − csr`` share outside it rides up.
    # 5. The downdraft (below) transports and removes nothing
    #    (``ham_wetdep`` touches ``pmfuxt`` only).
    # 6. The removed aerosol falls with the precipitation and is released
    #    top to bottom by ``prevap`` (``release_scavenged``).
    #
    # The environment air a layer entrains is its own full-level value;
    # ECHAM entrains the half-level ``pxtenh(jk+1)``, which charges part of
    # the sink to the layer below and needs no separate budget. Using the
    # layer's own air keeps each layer's exchange local and positive.
    #
    # The plume carries each tracer in three pools: ``fresh`` (not yet met
    # cloud), ``act`` (in the condensate, ``csr`` of the fresh aerosol at a
    # layer holding condensate — HAMMOZ's level-independent ``csr_conv``)
    # and ``inact`` (the ``1 − csr`` share outside it, which stays out for
    # the rest of the ascent; offering the leftover to the fixed fraction
    # again at every level would put the part that stayed out of the
    # droplets back into them layer after layer, a compounding HAMMOZ does
    # not have). Per-level pools are (K, *horiz); ``w`` broadcasts against
    # them, so a bare (K, nlev) column works the same as a (K, nlev, ncols)
    # block.
    w_b = w.reshape((-1,) + (1,) * (q.ndim - 2))

    def ascend(pools_below, xs):
        act_b, inact_b, fresh_b = pools_below
        m_below_k, e_k, d_k, m_k, q_k, frac_k, cloudy_k = xs
        # Step 1: detrainment leaves at the incoming concentration, up to
        # what arrived; any excess is this layer's entrained air.
        d_in = jnp.minimum(d_k, m_below_k)
        d_env = d_k - d_in
        x_b = act_b + inact_b + fresh_b
        det_k = (d_in[jnp.newaxis] * x_b + d_env[jnp.newaxis] * q_k)
        live = (m_k > _MF_FLOOR)[jnp.newaxis]
        inv = 1.0 / jnp.maximum(m_k, _MF_FLOOR)
        keep = ((m_below_k - d_in) * inv)[jnp.newaxis]
        ent = ((e_k - d_env) * inv)[jnp.newaxis]
        # A layer the plume does not leave through its top resets the
        # plume to the local air, all of it fresh; the value only seeds
        # the next live layer.
        act = jnp.where(live, keep * act_b, 0.0)
        inact = jnp.where(live, keep * inact_b, 0.0)
        fresh = jnp.where(live, keep * fresh_b + ent * q_k, q_k)
        # Step 2.
        c = cloudy_k[jnp.newaxis]
        act = act + jnp.where(c, w_b * fresh, 0.0)
        inact = inact + jnp.where(c, (1.0 - w_b) * fresh, 0.0)
        fresh = jnp.where(c, 0.0, fresh)
        # Step 3. Scavenge only the nonnegative part: spectral ringing
        # leaves negative lobes on near-zero tracers, and removing a
        # negative concentration would INJECT plume mass and drive the
        # wet_* ledger negative. Transport of the signed value is untouched.
        removed = jnp.where(live, frac_k[jnp.newaxis] * jnp.maximum(act, 0.0),
                            0.0)
        act = act - removed
        r_k = m_k[jnp.newaxis] * removed              # (K, ncols) flux
        return (act, inact, fresh), (det_k, r_k)

    q_lev = jnp.moveaxis(q, 1, 0)                     # (nlev, K, ncols)
    zero_pool = jnp.zeros_like(q_lev[-1])
    _, (det_rev, r_rev) = jax.lax.scan(
        ascend,
        (zero_pool, zero_pool, q_lev[-1]),            # seeded, overwritten at base
        (mfu_below, entrain_eff, detrain, mfu, q_lev, base_frac, cloudy),
        reverse=True,
    )
    det_up = jnp.moveaxis(det_rev, 0, 1)              # (K, nlev, ncols)
    removed = jnp.moveaxis(r_rev, 0, 1)               # (K, nlev, ncols)
    # Step 6: the removed aerosol falls with the precipitation and returns
    # to the environment where it evaporates (HAMMOZ ``zdxtevapic``).
    if evap_fraction is not None:
        released, scav_flux = release_scavenged(removed, evap_fraction)
    else:
        released = jnp.zeros_like(removed)
        scav_flux = jnp.sum(removed, axis=1)

    # Downward plume scan (top -> surface), cuddraf's tracer budget
    # (mo_cudescent.f90:293-296): the layer's entrained air joins and its
    # detrained air leaves at the incoming downdraft concentration, the
    # excess over what arrived being entrained air leaving again, as in
    # the updraft.
    def descend(q_dn_above, xs):
        m_in_k, e_k, d_k, m_out_k, q_k = xs
        d_in = jnp.minimum(d_k, m_in_k)
        d_env = d_k - d_in
        det_k = d_in[jnp.newaxis] * q_dn_above + d_env[jnp.newaxis] * q_k
        live = (m_out_k > _MF_FLOOR)[jnp.newaxis]
        inv = 1.0 / jnp.maximum(m_out_k, _MF_FLOOR)
        q_out = jnp.where(
            live,
            ((m_in_k - d_in) * inv)[jnp.newaxis] * q_dn_above
            + ((e_k - d_env) * inv)[jnp.newaxis] * q_k,
            q_k,
        )
        return q_out, det_k

    _, det_dn_lev = jax.lax.scan(
        descend,
        q_lev[0],                                     # seeded, overwritten at LFS
        (md_in, entrain_dn, detrain_dn, md_out, q_lev),
    )
    det_dn = jnp.moveaxis(det_dn_lev, 0, 1)           # (K, nlev, ncols)

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
        det_up
        - entrain_eff[jnp.newaxis] * q
        + mfu[jnp.newaxis] * q_above
        - mfu_below[jnp.newaxis] * q
        + det_dn
        - entrain_dn[jnp.newaxis] * q
        + md_out[jnp.newaxis] * q_below
        - md_in[jnp.newaxis] * q
        + released
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
        csr_conv: tuple[float, ...] | None = None,
    ):
        """Hold the tracer list, params and per-tracer in-condensate fractions.

        ``csr_conv`` (aligned with ``tracer_names``) is the fraction of each
        tracer entering the cloudy plume that is in the condensate — HAMMOZ's
        per-mode ``csr_conv`` for aerosol, 0 for gases; ``None`` disables
        scavenging entirely. It is the default of the differentiable
        ``params.csr_conv``; tracers given a positive value publish a
        ``_conv_scav_flux`` entry.
        """
        if not tracer_names:
            raise ValueError(
                "ConvectiveTracerTransport needs a non-empty tracer list."
            )
        if csr_conv is not None and len(csr_conv) != len(tracer_names):
            raise ValueError(
                "csr_conv must align with tracer_names: got "
                f"{len(csr_conv)} fractions for {len(tracer_names)} tracers."
            )
        self._tracer_names = tuple(tracer_names)
        self._csr_conv = (
            tuple(float(x) for x in csr_conv) if csr_conv is not None else None
        )
        if params is None:
            params = ConvTransportParameters.default(
                self._csr_conv if self._csr_conv is not None
                else (0.0,) * len(tracer_names))
        elif jnp.shape(params.csr_conv) != (len(tracer_names),):
            raise ValueError(
                "params.csr_conv must hold one fraction per tracer: got shape "
                f"{jnp.shape(params.csr_conv)} for {len(tracer_names)} tracers."
            )
        self.params = nnx.Param(params)

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
            if self._csr_conv is not None:
                scav_kwargs = dict(
                    csr_conv=jnp.asarray(params.csr_conv, dtype=q.dtype),
                    precip_efficiency=conv.precip_efficiency,
                    plume_condensate=conv.qc_conv + conv.qi_conv,
                    evap_fraction=conv.precip_evap_fraction,
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
        if self._csr_conv is not None:
            # Per-tracer surface deposition of in-plume scavenged mass net
            # of its release below [kg/m²/s], for downstream wet-deposition bookkeeping (the
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
                if self._csr_conv[k] > 0.0
            }

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
