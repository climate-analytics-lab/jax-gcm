"""``CloudBorneExchange`` — interstitial ↔ cloud-borne aerosol transfer (#602).

Closes the cloud-borne population cycle for MAM-style populations
(``spec.cloud_borne``): activation is the only physical source of cloud-borne
aerosol and droplet evaporation its only return path, and before this term
neither existed — the ``mc_*``/``nc_*`` mirrors were transported at full cost
while carrying nothing.

The scheme is a bounded relaxation toward the activation-equilibrium
partition. For each mode and phase pair ``(q_int, q_cb)``, ARG's per-mode
activated fractions (``_jam_activation``, number and mass separately — large
particles activate preferentially, so the mass fraction is well above the
number fraction) define the grid-mean equilibrium cloud-borne amount

    q_cb* = f · (q_int + q_cb),   f = (1 − p_ice)·f_act + p_ice·f_ice

and the pair relaxes toward it with a tunable timescale, activation and
resuspension each getting their own knob. The in-cloud aerosol is split
between the liquid and ice phases by the ice share ``p_ice`` of the in-cloud
condensate, as HAM splits it (``mo_ham_wetdep.f90::ham_wetdep``): the liquid
part is ARG droplet activation; the ice part is HAM's rule for aerosol in ice
(``ic_scav_nuc``): one particle per ice crystal, largest mode first
(``ice_phase_fractions``). A growing or persistent cloud pulls ``q_cb`` up
toward that partition; an ice cloud with few crystals therefore holds next to
none, and its reservoir drains on the resuspension timescale. Where the cover
has gone, the downward direction is keyed
to the microphysics' condensate-evaporation ledger (#708), PER PHASE
(#1070): the droplet-held share of the reservoir releases on the LIQUID
evaporation ledger alone, and the crystal-held share on the ICE one alone
— a cell cleared by evaporation of BOTH phases resuspends everything, a
mixed-phase cell where only one phase evaporated resuspends only that
phase's share, and a cell cleared by rainout resuspends nothing (that
aerosol leaves with the precip, removed by the wetdep term running just
after this one). A phase with no process of its own this step (advected-in
``q_cb`` in clear air, or a trace the microphysics never touched) falls
back to the slow ``resuspension_timescale`` drain. Every per-step transfer
factor lies in [0, 1], so the move is unconditionally bounded by the donor
phase's content and exactly conserving (the two tendencies are equal and
opposite); because the reservoir is one scalar per pair with no memory of
which phase holds which mass, the two phase-specific rates are blended by
the reservoir's estimated droplet-held share (``LIQUID_SHARE_KEY``), which
this term updates to its POST-transfer value so wet deposition and aqueous
chemistry, reading it immediately after, see the split the transfer
actually left behind.

This is deliberately simpler than CAM's ``dropmixnuc``, which couples the
transfer to an implicit turbulent-mixing solve; jcm has no physics-side
aerosol vertical transport yet (#602 item 2), so the relaxation form is the
honest standalone treatment. Convective processing of cloud-borne aerosol
(the ``aero_convproc`` analogue) is likewise future work, and cloud-borne
aerosol does not sediment (it follows the hydrometeors — see
``sedi_term``). ECHAM-HAM's M7 and sectional schemes like TOMAS never carry
an explicit cloud-borne phase at all: for those populations
``spec.cloud_borne = False`` and this term is not composed — the harness
then scavenges interstitial aerosol by ``cf · [(1 − p_ice)·f_act·f_wat +
p_ice·f_ice_share·f_ice]``, HAM's own form, which removes the same mass at
exchange equilibrium (wet deposition removes this reservoir phase by phase,
from the droplet-held share the term records under ``LIQUID_SHARE_KEY``). The representations still
differ where the representation itself matters: the explicit phase delays
rainout by the exchange timescale, and its in-droplet mass is invisible to
the (interstitial-only) aerosol optics, whereas the implicit treatment
keeps it in ``m_*`` where the optics see it.

Boundedness: each of this term's two directions is bounded by its donor
phase in isolation. In carry mode the updates are sequential, so the
overdraw question does not arise. In tracers mode, parallel operator
splitting sums this tendency with wet/dry deposition computed from the
same start-of-step state, so a decaying, still-precipitating cloud can
transiently overdraw ``mc_*`` below zero; every consumer floors at 0 on
read (removal terms deliberately treat negative values as empty rather
than pumping them — the storage A/B measured that pumping driving the
advected mirrors net-negative), so a negative excursion is inert until
transport or the next transfer refills the cell.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.cloud_borne_store import (
    CARRY_KEY,
    LIQUID_SHARE_KEY,
    apply_updates,
    carry_mode,
)
from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    ice_phase_fractions,
    phase_split,
    post_microphysics_icnc,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.removal_split import split_view
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


#: Cloud fraction below which a box counts as clear: activation stops and the
#: cloud-borne reservoir drains. Also floors the 1/cf stretch of the activation
#: timescale, so a vanishingly thin cloud cannot make it infinite.
_MIN_CLOUD_FRACTION = 1.0e-3

#: Partition below which a pair's phase composition is undefined and the
#: condensate's own liquid share stands in.
_SHARE_FLOOR = 1.0e-12

#: Grid-mean condensate floor [kg/kg] deciding whether the cell saw any cloud
#: process this step (evaporation + surviving pool + formation); below it the
#: evaporation-ledger keying falls back to the slow timescale drain. Physical
#: floor, not an epsilon, for the same f32 VJP reason as wetdep's
#: ``_CONDENSATE_FLOOR``.
_PROCESS_FLOOR = 1.0e-12


@jax.custom_jvp
def _safe_fraction(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute ``a / (a + b)`` for nonnegative ``a``, ``b``, at any scale.

    The forward division is already safe — ``a``, ``b`` are both nonnegative,
    so ``a + b`` never cancels — but JAX's default VJP for a quotient goes
    through ``1/(a+b)**2`` computed from an UNMASKED ``a+b``, which both
    underflows float32 once ``a + b`` drops below ~1e-19 (#1070 Codex
    review, found repeatedly in this module) and divides by a literal zero
    where ``a = b = 0``. The custom JVP below masks ``a + b`` the same way
    the forward pass does and computes the two partials from that ONE
    masked denominator — ``∂f/∂a = b/denom²`` and ``∂f/∂b = −a/denom²`` —
    the ordinary quotient-rule partials, not rewritten through ``f`` or
    ``1−f``: an earlier revision of this helper computed ``∂f/∂a`` as
    ``(1−f)/denom`` (algebraically identical, since ``1−f = b/denom``), but
    ``1−f`` catastrophically cancels in float32 once ``a ≫ b`` makes ``f``
    round to exactly ``1.0``, silently zeroing a nonzero gradient (a FOURTH
    Codex-review finding on this helper). Computing ``b/denom²`` and
    ``−a/denom²`` directly from ``b``/``a`` keeps the numerator exact at
    every scale; the only thing masked is the shared, nonneg denominator,
    which the forward pass already confirms is either exactly 0 (the one
    masked case, tangent 0) or safely positive.

    Even so, ``denom**2`` must never be formed as its own intermediate
    value: ``denom`` can be safely positive and ordinary (e.g.
    ``a = b = 1e-30``, ``denom = 2e-30`` — not even subnormal) while its
    SQUARE underflows float32 to literal 0 (``4e-60`` is far below the
    smallest representable subnormal, ~1.4e-45), after which ``b/denom**2``
    divides by that literal 0 instead of by a merely-small number. A first
    revision of this JVP floored ``denom**2`` at float32's smallest normal
    value to avoid the resulting ``inf``/``nan`` — found wrong by Codex
    review: the floor silently corrupted the still-perfectly-representable
    true derivatives it was never needed for (``2.5e19`` read back as
    ``8.5e17`` at ``a = b = 1e-20``; CLAUDE.md's "no substitute caps"
    prohibition). ``(b / denom) / denom`` is algebraically identical to
    ``b / denom**2`` but never forms the squared intermediate at all: each
    division is by ``denom`` itself, which the ``positive`` mask already
    keeps a safely nonzero, ordinary value, so nothing here is capped —
    the result is the exact derivative at every grid point this module's
    own physics can reach, and genuinely overflows (rather than being
    silently wrong) only beyond it, where the TRUE derivative itself
    exceeds float32's range (e.g. ``a = b = 1e-40``, both already
    subnormal: the true ``2.5e39`` is itself unrepresentable in float32).

    A SIXTH Codex-review finding: under ``jax.jit`` (how this module always
    actually runs), XLA's algebraic simplifier recognises ``(b / denom) /
    denom`` as ``b * denom⁻¹ * denom⁻¹`` and refuses to preserve the
    sequential evaluation — it rewrites it back into ``b / denom**2``,
    re-forming the squared intermediate the sequential form exists to avoid
    and reproducing the exact ``inf`` this docstring just finished
    explaining (verified: ``a = b = 1e-20`` gives the correct ``2.5e19`` in
    eager mode but ``inf`` once compiled). The custom JVP below is unaffected
    by autodiff correctness — the math is still exact — but the *compiler*
    defeats the chosen evaluation order, so the fix has to stop the
    simplifier from seeing across it: ``jax.lax.optimization_barrier`` is a
    value-identity op XLA may not fuse, reorder, or algebraically simplify
    through, so wrapping the two first-stage quotients (``b/denom``,
    ``a/denom``) in one forces the second division to run against the
    already-computed quotient rather than against a re-derived ``denom**2``,
    in both eager and compiled execution.
    """
    denom = jnp.where(a + b > 0.0, a + b, 1.0)
    return jnp.where(a + b > 0.0, a / denom, 0.0)


@_safe_fraction.defjvp
def _safe_fraction_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    positive = a + b > 0.0
    denom = jnp.where(positive, a + b, 1.0)
    f = jnp.where(positive, a / denom, 0.0)
    # The barrier blocks XLA from re-associating these into b/denom**2 and
    # a/denom**2 under jit, which would re-underflow exactly as the
    # sequential form above was written to avoid (see docstring).
    quotient_a, quotient_b = jax.lax.optimization_barrier((b / denom, a / denom))
    df_da = jnp.where(positive, quotient_a / denom, 0.0)
    df_db = jnp.where(positive, -(quotient_b / denom), 0.0)
    tangent_out = df_da * da + df_db * db
    return f, tangent_out


@tree_math.struct
class CloudBorneExchangeParameters:
    """Tunable exchange timescales (differentiable)."""

    activation_timescale: jnp.ndarray    # τ toward the activated partition [s]
    resuspension_timescale: jnp.ndarray  # τ back to interstitial [s]

    @classmethod
    def default(cls) -> "CloudBorneExchangeParameters":
        # Droplet nucleation and evaporative release are both fast against
        # the coupling step; 900 s makes the partition track the cloud field
        # within ~a step at Δt = 1800 s without being a hard swap.
        return cls(
            activation_timescale=jnp.asarray(900.0),
            resuspension_timescale=jnp.asarray(900.0),
        )


class CloudBorneExchange(PhysicsTerm):
    """Relax the interstitial/cloud-borne partition toward ARG's equilibrium."""

    name: ClassVar[str] = "jam_cloud_borne_exchange"
    category: ClassVar[str] = "aerosol_cloud_borne"
    requires: ClassVar[tuple[str, ...]] = ("_jam_activation", "clouds")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: CloudBorneExchangeParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population (which must prognose the phase).

        Resuspension keys to the cloud scheme's condensate-evaporation
        ledger (#708) — the physical discriminator between "the cloud
        evaporated, release the aerosol" and "the cloud rained out, the
        aerosol left with it". Cover alone cannot distinguish the two: a
        fully-rained-out cell also ends with cover 0, and a timescale
        drain there would race the same step's rainout (with this term
        running before wetdep, up to ``1-exp(-dt/τ)`` ≈ 86% at Δt=1800 s
        of the reservoir would escape scavenging through exactly the
        cells with the largest removal). This term therefore requires a
        cloud scheme that publishes the ``CloudData`` ledger fields; the
        physics factory enforces that at compose time.
        """
        self.params = nnx.Param(
            params or CloudBorneExchangeParameters.default()
        )
        self._spec = spec or MAM4_SPEC
        if not self._spec.cloud_borne:
            # Composed against a population without the mirror tracers, the
            # ``mc_*``/``nc_*`` tendencies would be silently dropped by the
            # accumulator while the interstitial side still fires — venting
            # mass with no error. Fail at compose time instead.
            raise ValueError(
                "CloudBorneExchange needs a population with "
                "spec.cloud_borne=True; this spec prognoses no cloud-borne "
                "phase (use the implicit activated-fraction scavenging "
                "instead)."
            )
        if carry_mode(self._spec):
            # In carry mode the store term must run upstream each step
            # (name-set fixing + vertical mixing); requiring its key makes
            # _validate_ordering enforce that, instead of apply_updates
            # silently seeding an unmixed, unmanaged dict.
            self.requires = (*type(self).requires, CARRY_KEY)

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        act = diagnostics["_jam_activation"]
        clouds = diagnostics["clouds"]
        cf = jnp.clip(clouds.cloud_fraction, 0.0, 1.0)
        dt = diagnostics.get("_dt_seconds", 1800.0)
        # HAM's phase split of the in-cloud condensate (the process-time
        # pool, as the scavenging uses it) and the per-phase scavenged
        # fractions of this step.
        f_wat, f_ice, pice = phase_split(clouds, dt)

        # Gather every (interstitial, cloud-borne) pair with its activated
        # fraction and run the relaxation once over the whole stack (the
        # wetdep/sedimentation batching pattern). ``state.tracers`` is empty
        # during ``Model.get_empty_data``'s structural probe, so fall back to
        # zeros there. Tracers are floored at 0: spectral advection leaves
        # small negative mass/number on near-zero fields (Gibbs ringing),
        # and a negative donor would flip the transfer's sign.
        zeros = jnp.zeros_like(state.temperature)
        # Operator-split read: transfer from the interstitial mass that
        # survived this step's removal, so the credit to the carry is
        # matched by a debit that exists.
        view = split_view(self._spec, state, diagnostics)
        int_names: list[str] = []
        cb_names: list[str] = []
        q_int: list[jnp.ndarray] = []
        q_cb: list[jnp.ndarray] = []
        fracs: list[jnp.ndarray] = []
        liquid_share: list[jnp.ndarray] = []
        # In-cloud aerosol inside the ice (HAM): crystals filled one particle
        # each from the largest mode down, against the total (interstitial +
        # cloud-borne) number of each mode.
        number = [
            jnp.maximum(view.get(number_name(m.short), zeros), 0.0)
            + jnp.maximum(view.get(number_name(m.short, cloud_borne=True),
                                   zeros), 0.0)
            for m in self._spec.modes
        ]
        ice_number, ice_mass = ice_phase_fractions(
            self._spec, number, post_microphysics_icnc(state, diagnostics, dt))
        for i, mode in enumerate(self._spec.modes):
            # Activated partition of the in-cloud aerosol: the liquid share
            # by ARG droplet activation, the ice share by the crystal count.
            pairs = [(
                number_name(mode.short),
                number_name(mode.short, cloud_borne=True),
                (1.0 - pice) * act.number_frac[i], pice * ice_number[i],
            )] + [(
                mass_name(sp, mode.short),
                mass_name(sp, mode.short, cloud_borne=True),
                (1.0 - pice) * act.mass_frac[i], pice * ice_mass[i],
            ) for sp in mode.species]
            for int_nm, cb_nm, in_liquid, in_ice in pairs:
                int_names.append(int_nm)
                cb_names.append(cb_nm)
                q_int.append(jnp.maximum(view.get(int_nm, zeros), 0.0))
                q_cb.append(jnp.maximum(view.get(cb_nm, zeros), 0.0))
                frac = in_liquid + in_ice
                fracs.append(frac)
                # The droplet-held share of the reservoir, at the partition
                # it relaxes toward: what rain formation removes and aqueous
                # chemistry sees, against the crystal-held rest that only
                # snow formation removes (HAM scavenges the two phases apart).
                # It describes the whole reservoir, with no memory of the
                # phase the aerosol was taken up in: the aerosol sits in the
                # hydrometeors the cell holds now. Droplets that froze hold
                # it in the ice, droplets that evaporated release it through
                # the evaporation ledger below, and a droplet-held remainder
                # in a cell without droplets is not a state the cloud can be
                # in. HAM likewise splits its in-cloud aerosol by the
                # current ``pice`` every step — but HAM has no reservoir to
                # carry between steps at all, so for it this is the whole
                # design, not a simplification. jcm's explicit reservoir
                # DOES persist, yet this estimate is re-derived from scratch
                # every step with no memory of the PREVIOUS step's share —
                # self-consistent within a single step (the post-transfer
                # correction below), but not across steps if the condensate
                # composition swings abruptly between them. Tracked as #1102
                # (persist the share itself, or split the reservoir into
                # separate liquid-held/ice-held stores).
                has = frac > _SHARE_FLOOR
                liquid_share.append(jnp.where(
                    has, in_liquid / jnp.where(has, frac, 1.0), 1.0 - pice))

        q_int_arr = jnp.stack(q_int)
        q_cb_arr = jnp.stack(q_cb)
        # Cloud fraction sets the RATE at which the box's air is processed
        # through droplets, not a ceiling on how much aerosol can be in them.
        #
        # Putting cf in the target instead pins the reservoir at cf·f_act of
        # the total. Soluble interstitial aerosol has no stratiform in-cloud
        # sink of its own, so the grid-mean removal then collapses to
        # cf·f_act·rate_cb — algebraically the implicit (no cloud-borne phase)
        # treatment, meaning the explicit reservoir buys only a delay. Measured
        # against CAM's ``wetdepa_v2`` on this model's own condensate and
        # precip-formation fields, that left accumulation-mode sulfate removal
        # at 5-38% of CAM's (#658), and accumulation mode sits in the
        # Greenfield gap where in-cloud scavenging is its only real sink.
        #
        # CAM has no downward relaxation at all: ``raercol_cw`` falls only when
        # the cloud shrinks or disappears (``ndrop.F90:486-518, 719-721``), and
        # cloud fraction enters through the activation flux and the
        # cloud-fraction increment, so under a persistent deck the reservoir
        # fills toward the activated fraction of the total. Matching that here:
        # where there is cloud, relax toward ``f · q_total`` on a timescale
        # stretched by 1/cf — thin cloud processes the box slowly — and where
        # the cloud has gone, drain to zero on the resuspension timescale.
        #
        # ``f`` is the phase-split partition above, which matters most for
        # cold upper-level ice cloud. The reservoir is neither advected nor
        # sedimented, so filling it by droplet activation under cirrus or the
        # polar-vortex ice cloud — a few crystals per litre — would collect the
        # aerosol carried through the cloud and return it, concentrated,
        # wherever the ice evaporated: thin, sharp aerosol layers at the
        # ice-cloud level with no sink up there. HAM's crystal count leaves
        # such clouds next to empty and still fills convective anvils
        # (thousands of crystals per litre), whose snow then removes it.
        cloudy_cf = jnp.maximum(cf, _MIN_CLOUD_FRACTION)
        target = jnp.where(
            cf > _MIN_CLOUD_FRACTION,
            jnp.stack(fracs) * (q_int_arr + q_cb_arr),
            0.0,
        )
        phi_up = -jnp.expm1(
            -dt / jnp.maximum(params.activation_timescale / cloudy_cf, 1.0)
        )
        phi_slow = -jnp.expm1(
            -dt / jnp.maximum(params.resuspension_timescale, 1.0)
        )
        # Resuspension keyed to the microphysics' own evaporation ledger
        # (#708), split by PHASE (#1070). Each phase's own evaporation
        # claims only its own share of the reservoir: the droplet-held
        # share releases on the liquid ledger (zxlevap) alone, the
        # crystal-held share on the ice ledger (zxievap) alone. Before
        # #1070, a single combined E = zxlevap + zxievap over a combined
        # pool could not tell "the droplets evaporated, the crystals
        # didn't" from "both evaporated" in a mixed-phase cell that ends
        # the step fully clear, so the whole reservoir was released
        # whenever EITHER phase evaporated — over-releasing the phase that
        # instead persisted (and should either wait for its own ledger, or
        # fall back to the slow drain below).
        e_liq_gm = jnp.maximum(
            clouds.condensate_evaporation_liquid_rate, 0.0) * dt
        e_ice_gm = jnp.maximum(
            clouds.condensate_evaporation_ice_rate, 0.0) * dt
        cf_proc = jnp.clip(clouds.process_cloud_fraction, 0.0, 1.0)
        pool_liq_gm = cf_proc * jnp.maximum(clouds.incloud_liquid, 0.0)
        pool_ice_gm = cf_proc * jnp.maximum(clouds.incloud_ice, 0.0)
        # The liquid (rain formation + riming) and ice (snow formation)
        # parts are floored separately, as HAM keeps the phases apart
        # (mo_hammoz_wetdep.f90:428-435 clips the ice efficiency from
        # pmrateps alone and takes the liquid one from pmratepr + pmsnowacl;
        # ``incloud_scavenged_fractions`` does the same). The snow-formation
        # ledger is signed: its sedimentation seed is negative in a level
        # that absorbs more falling ice than it sheds (mo_cloud_micro_2m.f90
        # 2258-2265), and a joint floor would let that offset rain and
        # riming in the same cell.
        formed_liq_gm = cf_proc * dt * jnp.maximum(
            clouds.incloud_rain_formation + clouds.incloud_riming, 0.0)
        formed_ice_gm = cf_proc * dt * jnp.maximum(
            clouds.incloud_snow_formation, 0.0)
        # Per pair: the PRE-transfer estimate of the reservoir's own
        # droplet-held vs crystal-held split — the droplet-held share
        # rains out at the liquid fraction, the crystal-held share at the
        # ice fraction. (Below, this same quantity is corrected to its
        # POST-transfer value for publication.)
        share = jnp.stack(liquid_share)
        # ``live`` stays a CELL-level gate, exactly as before the #1070
        # split: it asks "did ANY process (either phase's evaporation,
        # surviving pool, or formation) touch this cell this step", which
        # is what distinguishes a cell whose ledger genuinely reflects
        # this step's microphysics from a truly-untouched one (stale
        # advected-in q_cb with no cloud process at all, which must fall
        # back to the slow drain rather than trust a 0/0 evaporation
        # ratio). Only the EVAPORATION FRACTION itself — the numerator
        # that was the #1070 bug — is split by phase: f_evap_ice is
        # genuinely 0 (not a fallback) whenever this step's ice ledger
        # recorded no evaporation, even in a cell the liquid ledger marks
        # live, and conversely for f_evap_liq.
        own_liq = e_liq_gm + pool_liq_gm + formed_liq_gm
        own_ice = e_ice_gm + pool_ice_gm + formed_ice_gm
        live = (own_liq + own_ice) > _PROCESS_FLOOR
        # The per-phase ratio's own denominator must NOT be floored at
        # ``_PROCESS_FLOOR``: that floor is the CELL-level significance
        # threshold ``live`` already uses, not a bound on how small a
        # single phase's own (evaporated + surviving) total may
        # legitimately be. Flooring it there (Codex review of this PR)
        # inflated the denominator whenever a phase's own total sat below
        # the floor while the CELL was still live through the other phase
        # or formation — e.g. both phases fully evaporate with
        # e_liq_gm = e_ice_gm = 7.5e-13 (individually below the floor, cell
        # live since their sum exceeds it) gave f_evap = 0.75 instead of
        # the correct 1.0 (pool is exactly 0: the phase's ENTIRE total
        # evaporated, however small that total was). ``denom_liq``/
        # ``denom_ice`` are sums of nonnegative terms, so the VALUE needs
        # no floor at all once the denominator is confirmed nonzero — only
        # the genuine 0/0 (a phase with neither evaporation nor pool) must
        # fall back, and ``_safe_fraction`` already reads that as 0.
        #
        # ``_safe_fraction`` (not a bare division) because a LIVE cell can
        # still have one phase's own ``e_*_gm``/pool at a scale where the
        # ratio is meaningful but the denominator is not — Codex review of
        # this PR found exactly this (``e_liq_gm = pool_liq_gm = 1e-20``
        # alongside a normal-sized ``pool_ice_gm`` keeping the cell live):
        # a bare ``a / jnp.where(..., denom, 1.0)`` gets the FORWARD value
        # right but still differentiates through JAX's default quotient
        # VJP (``1/denom**2``), underflowing float32 well before the TRUE
        # gradient — which is representable, once rewritten through the
        # fraction itself (``O(1/denom)``, see ``_safe_fraction``) — does.
        # Composing it with the ``live`` mask needs no extra double-where
        # care either: ``_safe_fraction`` is gradient-safe at every input
        # by construction, so there is no "discarded branch" left to poison
        # whatever ``jnp.where(live, ...)`` selects.
        f_evap_liq = jnp.where(live, _safe_fraction(e_liq_gm, pool_liq_gm), 0.0)
        f_evap_ice = jnp.where(live, _safe_fraction(e_ice_gm, pool_ice_gm), 0.0)
        # The ledger keying applies ONLY where the sky has cleared
        # (cf < _MIN_CLOUD_FRACTION, i.e. target = 0), and then PER PHASE:
        # a phase that rained/snowed out has f_evap ≈ 0 through its own
        # formation claim (no resuspension racing that phase's rainout), a
        # phase that evaporated releases its own share in one step, and a
        # phase with no process of its own this step (zero evaporation,
        # zero surviving pool, zero formation — e.g. there was never any
        # ice at this level) falls back to the slow drain below rather
        # than trusting a 0/0 evaporation ratio — so a cleared cell whose
        # liquid evaporated while its ice sat untouched resuspends only
        # the liquid-held share in this one step, leaving the crystal-held
        # share to drain on its own timescale (the #1070 case).
        #
        # A SEVENTH Codex-review finding, on this "untouched phase" case
        # specifically: gating that fallback by the CELL-level ``live``
        # (as an earlier revision did) is wrong whenever the OTHER phase
        # is what makes the cell live — ``cleared & live`` is then True
        # even though THIS phase's own ledger is exactly empty, so
        # ``release_ice`` (itself correctly 0, via ``f_evap_ice``'s own
        # 0/0) gets selected over ``phi_slow`` and the untouched phase
        # freezes at a literal 0 decay rate instead of draining on the
        # timescale the module docstring promises. The gate below instead
        # requires each phase's own ledger (``own_liq``/``own_ice``, same
        # terms ``live`` sums) to be positive, not merely the cell's —
        # which does not reopen the #1070 bug this guards against: a
        # phase that IS evaporating always has ``own_* > 0`` through its
        # own ``e_*_gm`` term, so it still takes the fast ``release_*``
        # path exactly as before; only a phase with truly nothing of its
        # own this step (``own_* == 0``) is redirected to ``phi_slow``,
        # regardless of what the other phase is doing.
        #
        # Where cloud PERSISTS, the downward direction is the equilibrium
        # relaxation toward the (nonzero) target and MUST keep its
        # timescale, for BOTH phases: keying it to the evaporation ledger
        # zeroes it in any non-evaporating cloudy cell, which turns the
        # reservoir into a ratchet — q_cb can rise toward the activation
        # target but never fall, loading the cloud-borne phase without
        # bound at cloud levels. The under-cloud relaxation is what bounds
        # the reservoir by the activation equilibrium. Under an ice cloud
        # with fewer crystals than particles that equilibrium is small, so
        # aerosol activated while the cloud still held liquid returns to
        # the interstitial phase on the resuspension timescale (CAM
        # likewise resuspends the cloud-borne aerosol of a shrinking
        # liquid cloud) instead of sitting in a persistent ice cloud with
        # no exit.
        cleared = cf <= _MIN_CLOUD_FRACTION
        release_liq = jnp.minimum(f_evap_liq, jnp.maximum(1.0 - f_wat, 0.0))
        release_ice = jnp.minimum(f_evap_ice, jnp.maximum(1.0 - f_ice, 0.0))
        phase_touched_liq = own_liq > 0.0
        phase_touched_ice = own_ice > 0.0
        phi_down_liq = jnp.where(
            cleared & live & phase_touched_liq, release_liq, phi_slow)
        phi_down_ice = jnp.where(
            cleared & live & phase_touched_ice, release_ice, phi_slow)
        # The reservoir is ONE scalar per (interstitial, cloud-borne) pair
        # — it carries no memory of which phase holds which mass — so the
        # rate applied to the pair as a whole blends the two phase-specific
        # rates by the reservoir's current estimated droplet-held share.
        phi_down = share * phi_down_liq + (1.0 - share) * phi_down_ice
        # phi ∈ [0, 1]: the move never overshoots the target, so neither
        # phase can go negative (|Δ| ≤ |target − q_cb| ≤ donor).
        phi = jnp.where(target > q_cb_arr, phi_up, phi_down)
        transfer = (target - q_cb_arr) * phi / dt   # [.../s], + toward cloud-borne

        # The published droplet-held share (#1070): growing a pair keeps
        # ``share`` unchanged (the activation split of the increment, which
        # is exactly what ``share`` already is), and a persisting-but-
        # shrinking pair also keeps it (both phases decay at the SAME
        # phi_slow there, so their ratio is preserved) — only a CLEARED
        # pair whose two phases released at DIFFERENT rates needs the
        # correction, which is exactly the #1070 case. The single general
        # form below is exact in every branch rather than needing to
        # special-case the cleared one: with target_liq = share·target and
        # target_ice = (1 − share)·target (the same split ``in_liquid``/
        # ``in_ice`` define for ``target`` itself), the phase-wise
        # relaxations sum EXACTLY to the scalar ``transfer`` above, since
        # phi_up applies identically to both phases when growing and
        # target_liq·phi_down_liq + target_ice·phi_down_ice = target·phi_down
        # by construction of target_liq/target_ice.
        phi_eff_liq = jnp.where(target > q_cb_arr, phi_up, phi_down_liq)
        phi_eff_ice = jnp.where(target > q_cb_arr, phi_up, phi_down_ice)
        # Retained DIMENSIONLESS fraction per phase: dividing by ``base``
        # (the pair's pre-transfer content plus what it relaxes toward —
        # the two nonnegative quantities ``liquid_after``/``ice_after`` are
        # built from) BEFORE combining with ``share`` keeps every quantity
        # in [0, 1] regardless of the reservoir's absolute scale, in both
        # the forward value and its gradient. Dividing by an ABSOLUTE mass
        # instead (two earlier revisions of this fix, both found by Codex
        # review) is wrong twice over: first, whatever floor guards a
        # literal 0/0 is itself an absolute scale, so a trace reservoir
        # below it silently keeps the stale pre-transfer share instead of
        # its own, perfectly determined one; second, even where the
        # forward value comes out right, back-propagating a division by a
        # tiny absolute mass hits JAX's ``d(a/b)/db = -a/b²`` and
        # UNDERFLOWS ``b²`` in float32 for ``b`` below ~1e-19 — NaN
        # gradients into wet deposition and aqueous chemistry, which read
        # this published share immediately after. ``r_liq``/``r_ice`` are
        # each a convex combination of (1 − phi_eff) and phi_eff weighted
        # by the pair's own pre-transfer/target split, so they are O(1)
        # and well-conditioned however small ``base`` is in absolute
        # terms; the one remaining division (by ``share · r_liq +
        # (1 − share) · r_ice``) is therefore never a tiny-absolute-mass
        # VJP, only a dimensionless retained fraction — which an EXACT
        # (not epsilon-fuzzed) zero guard below is enough to make safe.
        #
        # ``base`` itself is stop-gradiented before it divides anything.
        # It is only a NORMALIZATION choice — ``r_liq``/``r_ice``'s ratio
        # to each other (which is all ``share_after`` actually depends on;
        # ``base`` cancels out of that ratio exactly) carries no physical
        # meaning of its own, so there is nothing to differentiate through
        # it correctly even in principle. Differentiating through it
        # literally reintroduces the SAME class of bug one level up: this
        # base's own gradient (e.g. w.r.t. ``q_cb_arr``, which ``base``
        # depends on directly) hits ``d(w/base)/d(base) ∝ 1/base²`` again,
        # NaN below float32's ~1e-19 — now for a differentiation target
        # (the reservoir's own content) this PR was not asked to harden,
        # but leaving it live would just move the trap rather than close
        # it. Stopping it keeps the exact forward value (``base`` cancels
        # algebraically either way) while differentiating ``share_after``
        # fully through the physically meaningful paths — the activation/
        # resuspension timescales and each phase's own evaporation and
        # rainout ledgers, all of which reach this formula only through
        # the numerators ``r_liq``/``r_ice``, never through ``base``.
        base = q_cb_arr + target
        base_safe = jax.lax.stop_gradient(
            jnp.where(base > 0.0, base, 1.0))
        r_liq = (q_cb_arr * (1.0 - phi_eff_liq)
                 + target * phi_eff_liq) / base_safe
        r_ice = (q_cb_arr * (1.0 - phi_eff_ice)
                 + target * phi_eff_ice) / base_safe
        retained = share * r_liq + (1.0 - share) * r_ice
        # ``retained`` is a SUM of two nonnegative terms (``share``,
        # ``r_liq``, ``r_ice`` are each clipped into [0, 1] upstream), never
        # a difference, so it carries no cancellation error: it is exactly
        # 0 only when both phi_eff are exactly 1 (both phases fully
        # release), and otherwise exactly represents a real, well-defined
        # composition — including a small one. A 1e-6 RELATIVE cutoff here
        # (Codex review of this PR, round 1) wrongly discarded exactly that
        # case: a phase that releases completely (its own ``r`` term is
        # exactly 0) alongside a phase that releases 99.9999% but not quite
        # all of it (``r`` ~ 1e-6, small but exact, not noise) gave a
        # ``retained`` small enough to trip a 1e-6 threshold while
        # ``share * r_liq`` was UNAMBIGUOUSLY 0.
        #
        # An EXACT ``retained <= 0.0`` guard (the first fix) is not enough
        # either: ``share`` and ``r_liq`` can EACH independently be a tiny
        # positive float32 value (e.g. a near-fully-ice-held pair whose
        # liquid-held share nearly, but not quite, fully evaporates), whose
        # PRODUCT is smaller than either alone — round 2 of Codex review
        # measured ``retained`` as small as ~1e-22 this way without ever
        # hitting literal 0. Differentiating a division by a denominator
        # that small hits JAX's ``d(a/b)/db ∝ -a/b²``, and ``b²`` (~1e-44)
        # underflows float32 (min positive normal ~1.18e-38) well before
        # ``b`` itself reads as zero — NaN/Inf gradients into wet
        # deposition and aqueous chemistry.
        #
        # The guard below (``1e-12``) sits with wide margin on both sides:
        # above it, the LIVE division's denominator is never smaller than
        # 1e-12, whose square (~1e-24) stays safely representable, so no
        # VJP through it can underflow; below it, it is still ~5 orders of
        # magnitude above the largest "small but exact, not noise" retained
        # fraction either Codex round actually constructed (~1e-6), so it
        # cannot resurrect the round-1 bug. A pair genuinely retaining less
        # than 1e-12 of itself is, for any physical purpose, empty — falling
        # back to the stale ``share`` there is harmless because there is
        # nothing left to mischaracterize. ``jnp.where`` guards BOTH the
        # numerator and the denominator the LIVE branch sees (not only the
        # denominator, as the round-2 fix did) — the standard double-where
        # hardening (#558): masking only one operand still differentiates
        # the OTHER, unmasked one in the branch JAX computes but discards,
        # which is exactly how round 2's denominator-only mask still let a
        # live, unmasked ``share * r_liq`` numerator chase a live,
        # underflowing denominator through the "selected-away" branch.
        negligible = retained <= 1.0e-12
        safe_retained = jnp.where(negligible, 1.0, retained)
        safe_numerator = jnp.where(negligible, 0.0, share * r_liq)
        share_after = jnp.where(
            negligible, share, safe_numerator / safe_retained)

        # Cloud-borne side to the active store (carry mode integrates it
        # now, sequentially); interstitial side through the ordinary
        # tendency accumulator in both modes.
        diagnostics, tracer_tends = apply_updates(
            self._spec, diagnostics,
            {nm: transfer[k] for k, nm in enumerate(cb_names)}, dt,
        )
        diagnostics = {
            **diagnostics,
            LIQUID_SHARE_KEY: {
                nm: share_after[k] for k, nm in enumerate(cb_names)},
        }
        for k, nm in enumerate(int_names):
            tracer_tends[nm] = -transfer[k]

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
