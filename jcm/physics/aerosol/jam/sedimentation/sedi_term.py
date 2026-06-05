"""``StokesSedimentation`` — gravitational settling of interstitial aerosol.

Stokes settling velocity per mode (with Cunningham slip correction) from the
``_jam_state`` wet radius and particle density, transported between vertical
layers by a donor-cell (first-order upwind) scheme. Mass leaving the lowest
layer is the sedimentation flux to the surface (a sink from the column).
Cloud-borne tracers do not sediment here — they follow the hydrometeors.

Mirrors ``mo_hammoz_sedimentation`` / ``mo_ham_sedimentation``.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.constants import grav as _G
from jcm.constants import m_air as _MA
from jcm.constants import r_universal as _RGAS
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


@tree_math.struct
class SedParameters:
    """Tunable knobs for sedimentation (differentiable)."""

    velocity_scale: jnp.ndarray   # multiplies the settling velocity

    @classmethod
    def default(cls) -> "SedParameters":
        return cls(velocity_scale=jnp.asarray(1.0))


def air_viscosity(temperature: jnp.ndarray) -> jnp.ndarray:
    """Dynamic viscosity of air [Pa·s] (Sutherland's law)."""
    return 1.458e-6 * temperature ** 1.5 / (temperature + 110.4)


def stokes_velocity(
    r_wet: jnp.ndarray,
    rho_p: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
) -> jnp.ndarray:
    """Stokes settling velocity [m/s] with Cunningham slip correction.

    ``v = (2 g ρ_p r² C_c) / (9 μ)``. Inputs broadcast against each other;
    ``r_wet``/``rho_p`` may carry a leading mode axis.
    """
    mu = air_viscosity(temperature)
    # Mean free path λ = (μ/p)·√(π R T / (2 M_a)).
    mfp = (mu / pressure) * jnp.sqrt(jnp.pi * _RGAS * temperature / (2.0 * _MA))
    kn = mfp / jnp.maximum(r_wet, 1.0e-10)
    cunningham = 1.0 + kn * (1.257 + 0.4 * jnp.exp(-1.1 / jnp.maximum(kn, 1e-12)))
    return (2.0 * _G * rho_p * r_wet ** 2 * cunningham) / (9.0 * mu)


def sediment_column(
    q: jnp.ndarray,        # (nlev, ncols) mixing ratio
    velocity: jnp.ndarray, # (nlev, ncols) downward settling velocity [m/s]
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Donor-cell vertical sedimentation tendency.

    Level 0 is the top of the atmosphere; settling moves mass to increasing
    level index. Returns ``(dq_dt, surface_flux)`` where ``surface_flux`` is
    the mass mixing-ratio flux out of the lowest layer [kg/m²/s].
    """
    flux = air_density * velocity * q           # downward flux at layer base
    # Flux entering each layer = flux out of the layer above (0 at the top).
    flux_in = jnp.concatenate(
        [jnp.zeros_like(flux[:1]), flux[:-1]], axis=0
    )
    dq_dt = (flux_in - flux) / (air_density * layer_thickness)
    return dq_dt, flux[-1]


class StokesSedimentation(PhysicsTerm):
    """Per-mode gravitational settling of interstitial aerosol tracers."""

    name: ClassVar[str] = "jam_sedimentation"
    category: ClassVar[str] = "aerosol_sedimentation"
    requires: ClassVar[tuple[str, ...]] = (
        "_jam_state", "air_density", "layer_thickness", "pressure_full",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: SedParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
    ):
        """Hold params and the population."""
        self.params = nnx.Param(params or SedParameters.default())
        self._spec = spec or MAM4_SPEC

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        aer = diagnostics["_jam_state"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        pressure = diagnostics["pressure_full"]
        temperature = state.temperature

        # Gather every interstitial tracer to settle and the (per-mode)
        # settling velocity that transports it, then run the donor-cell
        # transport once over the whole stack so XLA batches it (rather than
        # emitting an unrolled op per tracer). ``state.tracers`` is empty
        # during ``Model.get_empty_data``'s structural probe, so fall back to
        # zeros there (real runs have every declared tracer seeded).
        zeros = jnp.zeros_like(state.temperature)
        names: list[str] = []
        q_list: list[jnp.ndarray] = []
        v_list: list[jnp.ndarray] = []
        for i, mode in enumerate(self._spec.modes):
            if not mode.sediments:
                continue
            v = params.velocity_scale * stokes_velocity(
                aer.r_wet[i], aer.rho[i], temperature, pressure,
            )
            for nm in [number_name(mode.short)] + [
                mass_name(sp, mode.short) for sp in mode.species
            ]:
                names.append(nm)
                q_list.append(state.tracers.get(nm, zeros))
                v_list.append(v)

        q_stack = jnp.stack(q_list)            # (K, nlev, ncols)
        v_stack = jnp.stack(v_list)            # (K, nlev, ncols)
        dq_stack, _ = jax.vmap(
            sediment_column, in_axes=(0, 0, None, None),
        )(q_stack, v_stack, air_density, dz)
        tracer_tends = {nm: dq_stack[k] for k, nm in enumerate(names)}

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
