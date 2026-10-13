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
                # current ``pice`` every step.
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
        live = (
            e_liq_gm + e_ice_gm + pool_liq_gm + pool_ice_gm
            + formed_liq_gm + formed_ice_gm
        ) > _PROCESS_FLOOR
        f_evap_liq = jnp.where(
            live,
            e_liq_gm / jnp.maximum(e_liq_gm + pool_liq_gm, _PROCESS_FLOOR),
            0.0,
        )
        f_evap_ice = jnp.where(
            live,
            e_ice_gm / jnp.maximum(e_ice_gm + pool_ice_gm, _PROCESS_FLOOR),
            0.0,
        )
        # The ledger keying applies ONLY where the sky has cleared
        # (cf < _MIN_CLOUD_FRACTION, i.e. target = 0), and then PER PHASE:
        # a phase that rained/snowed out has f_evap ≈ 0 through its own
        # formation claim (no resuspension racing that phase's rainout), a
        # phase that evaporated releases its own share in one step, and a
        # phase with no process of its own this step (zero evaporation,
        # zero surviving pool, zero formation — e.g. there was never any
        # ice at this level) also reads f_evap = 0 and so is correctly
        # retained rather than released — so a cleared cell whose liquid
        # evaporated while its ice sat untouched (persisting, or simply
        # absent) resuspends only the liquid-held share, leaving the
        # crystal-held share exactly where it was (the #1070 case).
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
        phi_down_liq = jnp.where(cleared & live, release_liq, phi_slow)
        phi_down_ice = jnp.where(cleared & live, release_ice, phi_slow)
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
        liquid_before = share * q_cb_arr
        ice_before = (1.0 - share) * q_cb_arr
        liquid_after = (liquid_before * (1.0 - phi_eff_liq)
                        + share * target * phi_eff_liq)
        ice_after = (ice_before * (1.0 - phi_eff_ice)
                     + (1.0 - share) * target * phi_eff_ice)
        total_after = liquid_after + ice_after
        # "Negligible" must be RELATIVE, with NO absolute floor anywhere in
        # the comparison: ``_SHARE_FLOOR`` bounds a dimensionless activation
        # fraction elsewhere, and reusing it as an absolute floor here (even
        # indirectly, as a floor UNDER ``q_cb_arr`` inside a ``maximum``)
        # made the correction concentration-dependent at whatever scale the
        # floor sits — first at 1e-12, then again at 1e-18 after the first
        # fix, both found by Codex review of this PR. A trace reservoir
        # could end with total_after below ANY fixed floor while still
        # carrying the SAME, perfectly determined phase split as a larger
        # one clearing it, silently discarding the correct share for the
        # stale pre-transfer value. The natural scale-free reference is
        # ``q_cb_arr + target``: the pair's own pre-transfer content plus
        # what it is relaxing toward, the only two nonnegative quantities
        # ``total_after`` is built from (see the derivation above) — it is
        # exactly 0 only when the pair has nothing before AND nothing to
        # grow toward, the one case where ``share`` is genuinely undefined
        # and harmless to leave stale. Elsewhere this correctly flags ONLY
        # float32 cancellation noise (both phases fully released, or a
        # vanishing growth increment, where ``share_after`` provably equals
        # ``share`` anyway) and never a real small-but-nonzero value.
        scale = q_cb_arr + target
        negligible = total_after <= 1.0e-6 * scale
        share_after = jnp.where(
            negligible,
            share,
            liquid_after / jnp.where(negligible, 1.0, total_after),
        )

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
