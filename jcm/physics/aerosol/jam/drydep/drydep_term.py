"""``SlinnDryDeposition`` — turbulent/Brownian dry removal at the surface.

Per-mode non-gravitational deposition velocity (resistance-in-series) applied
to the interstitial aerosol in the lowest model layer. The friction velocity is
read from the ``vertical_diffusion`` diagnostic's ``surface_friction_velocity``,
which the TTE-TKE term derives from the unified surface momentum exchange
coefficient (u*² = |U|·⟨CM·|U|⟩), so it is consistent with the surface stress
and the vdiff damping. Because that term runs *after* the aerosol block in the
default ECHAM ordering, the value comes from the previous step (or the floored
carry on step 1), so it is read with a fallback rather than declared as a hard
``requires``. The aerodynamic resistance uses a neutral log-law; a
Monin-Obukhov stability correction is a future refinement (the diagnostic does
not yet carry a usable surface ``L``).

Mirrors ``mo_hammoz_drydep``.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.jam.drydep.resistances import deposition_velocity
from jcm.physics.aerosol.jam.cloud_borne_store import (
    CARRY_KEY,
    apply_updates,
    carry_mode,
    mirror_names,
    tracer_view,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.sedimentation.sedi_term import stokes_velocity
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


@tree_math.struct
class DryDepParameters:
    """Tunable knobs for dry deposition (differentiable)."""

    z_ref: jnp.ndarray            # reference height [m]
    z0: jnp.ndarray               # roughness length [m]
    u_star_default: jnp.ndarray   # fallback friction velocity [m/s]

    @classmethod
    def default(cls) -> "DryDepParameters":
        return cls(
            z_ref=jnp.asarray(10.0),
            z0=jnp.asarray(1.0e-4),
            u_star_default=jnp.asarray(0.3),
        )


class SlinnDryDeposition(PhysicsTerm):
    """Surface dry deposition of interstitial aerosol tracers."""

    name: ClassVar[str] = "jam_dry_deposition"
    category: ClassVar[str] = "aerosol_drydep"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "air_density", "layer_thickness", "pressure_full",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: DryDepParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or DryDepParameters.default())
        self._spec = spec or MAM4_SPEC
        if carry_mode(self._spec):
            # In carry mode the store term must run upstream each step
            # (name-set fixing + vertical mixing); requiring its key makes
            # _validate_ordering enforce that, instead of apply_updates
            # silently seeding an unmixed, unmanaged dict.
            self.requires = (*type(self).requires, CARRY_KEY)

    def _u_star(self, diagnostics, ncols, params):
        if "vertical_diffusion" in diagnostics:
            return diagnostics["vertical_diffusion"].surface_friction_velocity
        return jnp.full((ncols,), params.u_star_default)

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        aer = diagnostics["_jam_state"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        pressure = diagnostics["pressure_full"]
        temperature = state.temperature
        nlev, ncols = temperature.shape

        dt = diagnostics.get("_dt_seconds", 1800.0)
        u_star = self._u_star(diagnostics, ncols, params)        # (ncols,)
        t_sfc = temperature[-1]
        p_sfc = pressure[-1]
        rho_sfc = air_density[-1]
        dz_sfc = dz[-1]

        # ``state.tracers`` is empty during ``Model.get_empty_data``'s
        # structural probe; fall back to zeros there (real runs have every
        # declared tracer seeded).
        zeros = jnp.zeros_like(state.temperature)
        view = tracer_view(self._spec, state, diagnostics)
        tracer_tends: dict[str, jnp.ndarray] = {}
        for i, mode in enumerate(self._spec.modes):
            r_sfc = aer.r_wet[i, -1]
            v_grav = stokes_velocity(r_sfc, aer.rho[i, -1], t_sfc, p_sfc)
            v_dep = deposition_velocity(
                r_sfc, v_grav, u_star, t_sfc, p_sfc, rho_sfc,
                z_ref=params.z_ref, z0=params.z0,
            )
            loss_rate = v_dep / dz_sfc  # [1/s] applied to bottom layer
            # Implicit (exponential) removal over the step, bounded to ≤100% of
            # the layer's mass: q(t+dt) = q·exp(-loss_rate·dt). An explicit
            # ``-loss_rate·q`` step overshoots into a sign-flipped runaway when
            # ``loss_rate·dt > 1`` (large deposition velocity for the coarse mode
            # over a thin surface layer) — the same instability that NaNs wet
            # deposition. ``1 - exp(-x)`` is unconditionally stable for any x ≥ 0.
            removed_frac = -jnp.expm1(-loss_rate * dt)   # ∈ [0, 1]

            names = [number_name(mode.short)] + [
                mass_name(sp, mode.short) for sp in mode.species
            ]
            # A prognostic cloud-borne phase (#602) deposits too — CAM's
            # ``aero_model_drydep`` treatment, using the mode's interstitial
            # deposition velocity (droplet-resolved velocities are a
            # refinement). Small next to wet removal, but it keeps a
            # surface-layer cloud from becoming a sink-less corner.
            if self._spec.cloud_borne:
                names += [number_name(mode.short, cloud_borne=True)] + [
                    mass_name(sp, mode.short, cloud_borne=True)
                    for sp in mode.species
                ]
            for nm in names:
                # Floored at 0: removal on a negative (ringing) value
                # would inject mass (see the wetdep note).
                q = jnp.maximum(view.get(nm, zeros), 0.0)
                tracer_tends[nm] = jnp.zeros_like(q).at[-1].set(
                    -(removed_frac * q[-1]) / dt
                )

        if carry_mode(self._spec):
            cb_updates = {
                nm: tracer_tends.pop(nm)
                for nm in mirror_names(self._spec) if nm in tracer_tends
            }
            diagnostics, passthrough = apply_updates(
                self._spec, diagnostics, cb_updates, dt,
            )
            tracer_tends.update(passthrough)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
