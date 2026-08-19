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

    q_cb* = cloud_fraction · f_act · (q_int + q_cb)

and the pair relaxes toward it with a tunable timescale, activation and
resuspension each getting their own knob. Both directions come out of the one
expression: a growing or persistent cloud pulls ``q_cb`` up toward the
activated partition, and a cleared sky (``cloud_fraction → 0``) returns the
whole reservoir to interstitial. The per-step transfer uses the exponential
relaxation factor ``1 − exp(−Δt/τ)``, so it is unconditionally bounded by the
donor phase's content and exactly conserving (the two tendencies are equal
and opposite).

This is deliberately simpler than CAM's ``dropmixnuc``, which couples the
transfer to an implicit turbulent-mixing solve; jcm has no physics-side
aerosol vertical transport yet (#602 item 2), so the relaxation form is the
honest standalone treatment. Convective processing of cloud-borne aerosol
(the ``aero_convproc`` analogue) is likewise future work, and cloud-borne
aerosol does not sediment (it follows the hydrometeors — see
``sedi_term``). ECHAM-HAM's M7 and sectional schemes like TOMAS never carry
an explicit cloud-borne phase at all: for those populations
``spec.cloud_borne = False`` and this term is not composed — the harness
then scavenges interstitial aerosol by ``cf · activated_fraction``, which
removes the same mass at exchange equilibrium. The representations still
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
    apply_updates,
    carry_mode,
    tracer_view,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


#: Cloud fraction below which a box counts as clear: activation stops and the
#: cloud-borne reservoir drains. Also floors the 1/cf stretch of the activation
#: timescale, so a vanishingly thin cloud cannot make it infinite.
_MIN_CLOUD_FRACTION = 1.0e-3


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
        """Hold params and the population (which must prognose the phase)."""
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

        # Gather every (interstitial, cloud-borne) pair with its activated
        # fraction and run the relaxation once over the whole stack (the
        # wetdep/sedimentation batching pattern). ``state.tracers`` is empty
        # during ``Model.get_empty_data``'s structural probe, so fall back to
        # zeros there. Tracers are floored at 0: spectral advection leaves
        # small negative mass/number on near-zero fields (Gibbs ringing),
        # and a negative donor would flip the transfer's sign.
        zeros = jnp.zeros_like(state.temperature)
        view = tracer_view(self._spec, state, diagnostics)
        int_names: list[str] = []
        cb_names: list[str] = []
        q_int: list[jnp.ndarray] = []
        q_cb: list[jnp.ndarray] = []
        fracs: list[jnp.ndarray] = []
        for i, mode in enumerate(self._spec.modes):
            pairs = [(
                number_name(mode.short),
                number_name(mode.short, cloud_borne=True),
                act.number_frac[i],
            )] + [(
                mass_name(sp, mode.short),
                mass_name(sp, mode.short, cloud_borne=True),
                act.mass_frac[i],
            ) for sp in mode.species]
            for int_nm, cb_nm, frac in pairs:
                int_names.append(int_nm)
                cb_names.append(cb_nm)
                q_int.append(jnp.maximum(view.get(int_nm, zeros), 0.0))
                q_cb.append(jnp.maximum(view.get(cb_nm, zeros), 0.0))
                fracs.append(frac)

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
        # where there is cloud, relax toward ``f_act · q_total`` on a timescale
        # stretched by 1/cf — thin cloud processes the box slowly — and where
        # the cloud has gone, drain to zero on the resuspension timescale.
        cloudy_cf = jnp.maximum(cf, _MIN_CLOUD_FRACTION)
        target = jnp.where(
            cf > _MIN_CLOUD_FRACTION,
            jnp.stack(fracs) * (q_int_arr + q_cb_arr),
            0.0,
        )
        tau = jnp.where(
            target > q_cb_arr,
            params.activation_timescale / cloudy_cf,
            params.resuspension_timescale,
        )
        # 1 − exp(−Δt/τ) ∈ [0, 1]: the move never overshoots the target, so
        # neither phase can go negative (|Δ| ≤ |target − q_cb| ≤ donor).
        phi = -jnp.expm1(-dt / jnp.maximum(tau, 1.0))
        transfer = (target - q_cb_arr) * phi / dt   # [.../s], + toward cloud-borne

        # Cloud-borne side to the active store (carry mode integrates it
        # now, sequentially); interstitial side through the ordinary
        # tendency accumulator in both modes.
        diagnostics, tracer_tends = apply_updates(
            self._spec, diagnostics,
            {nm: transfer[k] for k, nm in enumerate(cb_names)}, dt,
        )
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
