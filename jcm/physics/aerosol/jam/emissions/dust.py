"""Wind-erosion dust emission (Tegen et al. 2002 flux physics).

The full HAMMOZ BGC/Tegen scheme (``mo_ham_dust``) carries a 17-soil-type
database, soil size distributions and preferential-source maps to derive a
per-gridcell threshold friction velocity and sandblasting efficiency. That
soil pre-processing produces, in effect, a **dust source/erodibility field**;
here we read that field from forcing and apply the Tegen *emission physics*
to it — the saltation horizontal flux and its sandblasting conversion to a
vertical (emitted) flux, gated by friction velocity exceeding the threshold:

    G = scale · (ρ_air/g) · u*³ · (1 + u*t/u*) · (1 − (u*t/u*)²)   (u* > u*t)
    F = source · α · G

with ``u*`` the surface friction velocity (from the ``vertical_diffusion``
diagnostic), ``u*t`` the threshold, ``α`` the sandblasting efficiency, and
``source`` the prescribed erodibility (0–1). ``source`` is read from
``forcing.dust_source`` and falls back to zero, so the term is inert (e.g. on
an aquaplanet) until the field is supplied. The soil-database parameters are
folded into the calibratable ``source``/``α``/``u*t`` knobs.

References: Tegen et al. (2002), JGR 107; Marticorena & Bergametti (1995).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.constants import grav as _G
from jcm.physics.aerosol.jam.emissions.distributors import distribute_surface_flux
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.physics_term import PhysicsTendency, PhysicsTerm


@tree_math.struct
class DustParameters:
    """Calibratable knobs for the Tegen wind-erosion flux."""

    scale: jnp.ndarray             # overall horizontal-flux scale
    alpha: jnp.ndarray             # sandblasting efficiency [1/m]
    u_threshold: jnp.ndarray       # threshold friction velocity [m/s]
    accum_fraction: jnp.ndarray    # fraction of emitted dust into accum (rest coarse)
    u_star_default: jnp.ndarray    # fallback friction velocity [m/s]

    @classmethod
    def default(cls) -> "DustParameters":
        return cls(
            scale=jnp.asarray(1.0),
            alpha=jnp.asarray(1.0e-5),
            u_threshold=jnp.asarray(0.2),
            accum_fraction=jnp.asarray(0.1),
            u_star_default=jnp.asarray(0.3),
        )


def horizontal_flux(
    u_star: jnp.ndarray, u_threshold: jnp.ndarray, air_density: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """Tegen saltation horizontal flux G [kg/m/s] (zero below threshold)."""
    u = jnp.maximum(u_star, 1.0e-3)
    ratio = u_threshold / u
    g_flux = scale * (air_density / _G) * u ** 3 * (1.0 + ratio) * (1.0 - ratio ** 2)
    return jnp.where(u_star > u_threshold, g_flux, 0.0)


class DustEmissions(PhysicsTerm):
    """Wind-erosion dust emission (Tegen flux × prescribed source field)."""

    name: ClassVar[str] = "jam_dust_emissions"
    category: ClassVar[str] = "aerosol_emissions"
    requires: ClassVar[tuple[str, ...]] = ("air_density", "layer_thickness")
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: DustParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or DustParameters.default())
        self._spec = spec or MAM4_SPEC

    def _u_star(self, diagnostics, ncols, params):
        if "vertical_diffusion" in diagnostics:
            return diagnostics["vertical_diffusion"].surface_friction_velocity
        return jnp.full((ncols,), params.u_star_default)

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.get_value()
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        nlev, ncols = state.temperature.shape

        source = (
            jnp.clip(jnp.ravel(forcing.dust_source), 0.0, 1.0)
            if forcing is not None and getattr(forcing, "dust_source", None) is not None
            and jnp.size(forcing.dust_source) == ncols
            else jnp.zeros((ncols,))
        )
        u_star = self._u_star(diagnostics, ncols, p)
        g_flux = horizontal_flux(u_star, p.u_threshold, air_density[-1], p.scale)
        flux = source * p.alpha * g_flux                       # kg/m²/s

        fluxes = [
            ("du", "acc", flux * p.accum_fraction),
            ("du", "cor", flux * (1.0 - p.accum_fraction)),
        ]
        tracer_tends = distribute_surface_flux(self._spec, fluxes, air_density, dz)

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
