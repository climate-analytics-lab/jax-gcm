"""``SlinnDryDeposition`` — turbulent/Brownian dry removal at the surface.

Per-mode non-gravitational deposition velocity (resistance-in-series) applied
to the aerosol in the lowest model layer, published into the AeroCom
``dry_*`` ledger alongside gravitational settling. The friction velocity is
read from the ``vertical_diffusion`` diagnostic's ``surface_friction_velocity``,
which the TTE-TKE term derives from the unified surface momentum exchange
coefficient (u*² = |U|·⟨CM·|U|⟩), so it is consistent with the surface stress
and the vdiff damping. The default ECHAM ordering runs vertical diffusion before aerosol removal.
A fallback allows the standalone column harness without vertical diffusion. The aerodynamic resistance uses a neutral log-law; a
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
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.removal_split import split_view
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

    def _velocity(self, r, v_grav, u_star, t, p, rho, *, mode, moment, params, terrain):
        return deposition_velocity(
            r, v_grav, u_star, t, p, rho,
            geom_std_dev=mode.geom_std_dev, moment=moment,
            z_ref=params.z_ref, z0=params.z0,
        )

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
        # Operator splitting: deposit what sedimentation left, not the
        # step-start state (see ``removal_split``).
        view = split_view(self._spec, state, diagnostics)
        tracer_tends: dict[str, jnp.ndarray] = {}
        for i, mode in enumerate(self._spec.modes):
            r_sfc = aer.r_wet[i, -1]
            # Number and mass ride different moments of the mode, so both the
            # settling and the size-dependent Brownian/impaction terms differ
            # between them. CAM calls its deposition-velocity routine once per
            # moment for exactly this reason (``aero_model.F90:740-747``,
            # ``jvlc=1`` number / ``jvlc=2`` mass).
            removed_frac_by_moment = {}
            for moment in (0, 3):
                v_grav = stokes_velocity(
                    r_sfc, aer.rho[i, -1], t_sfc, p_sfc,
                    geom_std_dev=mode.geom_std_dev, moment=moment,
                    aspherical=mode.short == "cor",
                )
                v_dep = self._velocity(
                    r_sfc, v_grav, u_star, t_sfc, p_sfc, rho_sfc,
                    mode=mode, moment=moment, params=params, terrain=terrain,
                )
                loss_rate = v_dep / dz_sfc  # [1/s] applied to bottom layer
                # Implicit (exponential) removal over the step, bounded to
                # ≤100% of the layer's mass: q(t+dt) = q·exp(-loss_rate·dt).
                # An explicit ``-loss_rate·q`` step overshoots into a
                # sign-flipped runaway when ``loss_rate·dt > 1`` (large
                # deposition velocity for the coarse mode over a thin surface
                # layer) — the same instability that NaNs wet deposition.
                # ``1 - exp(-x)`` is unconditionally stable for any x ≥ 0.
                removed_frac_by_moment[moment] = -jnp.expm1(-loss_rate * dt)

            named_moments = [(0, number_name(mode.short))] + [
                (3, mass_name(sp, mode.short)) for sp in mode.species
            ]
            # A prognostic cloud-borne phase (#602) deposits too — CAM's
            # ``aero_model_drydep`` treatment, using the mode's interstitial
            # deposition velocity (CAM's droplet-resolved ``jvlc=3,4``
            # velocities are a refinement). Small next to wet removal, but it
            # keeps a surface-layer cloud from becoming a sink-less corner.
            if self._spec.cloud_borne:
                named_moments += [
                    (0, number_name(mode.short, cloud_borne=True))
                ] + [
                    (3, mass_name(sp, mode.short, cloud_borne=True))
                    for sp in mode.species
                ]
            for moment, nm in named_moments:
                # Floored at 0: removal on a negative (ringing) value
                # would inject mass (see the wetdep note).
                q = jnp.maximum(view.get(nm, zeros), 0.0)
                # The removed fraction carries the float64 parameters under
                # x64 (pySES runs float32 physics there), so the value is
                # pinned to the tracer's dtype before the scatter (#770).
                tracer_tends[nm] = jnp.zeros_like(q).at[-1].set(
                    (-(removed_frac_by_moment[moment] * q[-1]) / dt
                     ).astype(q.dtype)
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
            flux_tends = {**tracer_tends, **cb_updates}
        else:
            flux_tends = tracer_tends

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        # AeroCom deposition fluxes (jax-gcm#581): turbulent/Brownian dry
        # removal belongs in ``dry_*`` alongside gravitational settling.
        # Cloud-borne removals go into the carry, so they are folded in
        # explicitly.
        from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
            accumulate_deposition_fluxes)
        diagnostics = accumulate_deposition_fluxes(
            diagnostics, flux_tends,
            diagnostics["air_density"], diagnostics["layer_thickness"],
            kind="dry")
        # Separate the dust dry sinks so lifetime changes can be attributed
        # to settling versus surface collection rather than to their sum.
        diagnostics = {**diagnostics, "turb_dry_du": -sum([
            jnp.sum(tend * air_density * dz, axis=0)
            for name, tend in flux_tends.items()
            if name.startswith(("m_du_", "mc_du_"))
        ], jnp.zeros_like(air_density[0]))}
        return tendency, diagnostics


class CAMDryDeposition(SlinnDryDeposition):
    """CAM land-cover collection with the host's neutral surface resistance.

    All eleven source classes are prescribed, conservatively remapped to
    the model grid at construction. Aquaplanet terrain explicitly selects
    all water. This is surface collection only; settling remains separate.
    """

    def __init__(self, params: DryDepParameters | None = None, *,
                 spec: ModalAerosolSpec | None = None):
        """Hold host resistance parameters and coordinate-dependent cover."""
        super().__init__(params, spec=spec)
        self._fractions = nnx.Variable(jnp.eye(11)[6, :, None])

    def cache_coords(self, coords):
        from pathlib import Path
        import numpy as np
        import xarray as xr
        from jcm.data.regridding import build_regridder, model_grid, nearest_index
        path = Path(__file__).resolve().parents[4] / "data/bc/cam_landuse.nc"
        with xr.open_dataset(path) as ds:
            lon, lat, _ = model_grid(coords)
            if lon.ndim == lat.ndim == 1:
                remap = build_regridder(ds.lon.values, ds.lat.values,
                                       np.ones((ds.sizes["lat"], ds.sizes["lon"])), lon, lat)
                fractions = remap(ds.fraction_landuse.values).reshape(11, -1)
            else:
                # Point-grid hosts have no rectilinear cell edges. Use the
                # nearest inventory class mixture at each physical point.
                sx, sy = np.meshgrid(ds.lon.values, ds.lat.values)
                index = nearest_index(sy.ravel(), sx.ravel(),
                                      np.rad2deg(lat).ravel(), np.rad2deg(lon).ravel())
                fractions = ds.fraction_landuse.values.reshape(11, -1)[:, index]
        # CAM normalizes AFTER remapping, because the raw inventory's PFT
        # and lake/wetland/urban cover can overlap. Normalizing each source
        # cell first would change the coarse-grid area-weighted mixture.
        fractions /= fractions.sum(axis=0, keepdims=True)
        self._fractions = nnx.Variable(jnp.asarray(fractions, dtype=jnp.float32))

    def _velocity(self, r, v_grav, u_star, t, p, rho, *, mode, moment, params, terrain):
        from jcm.physics.aerosol.jam.drydep.resistances import cam_collection_velocity
        fractions = self._fractions.get_value()
        # The prescribed land-use map is the CAM input, independent of the
        # dynamical terrain. A flat all-ocean experiment must not inherit it.
        mask = getattr(terrain, "fmask", None)
        if mask is not None:
            fractions = jnp.where(jnp.all(mask == 0), jnp.eye(11)[6, :, None], fractions)
        return cam_collection_velocity(
            r, v_grav, u_star, t, p, rho, fractions,
            geom_std_dev=mode.geom_std_dev, moment=moment,
            z_ref=params.z_ref, z0=params.z0,
        )
