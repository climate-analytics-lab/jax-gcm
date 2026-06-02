"""``HamSedimentation`` — gravitational settling of interstitial aerosol.

Stokes settling velocity per mode (with Cunningham slip correction) from the
``_ham_state`` wet radius and particle density, transported between vertical
layers by a donor-cell (first-order upwind) scheme. Mass leaving the lowest
layer is the sedimentation flux to the surface (a sink from the column).
Cloud-borne tracers do not sediment here — they follow the hydrometeors.

Mirrors ``mo_hammoz_sedimentation`` / ``mo_ham_sedimentation``.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.aerosol.ham.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.ham.population import ModalAerosolSpec
from jcm.physics.aerosol.ham.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency

_G = 9.80665
_MA = 0.028965     # molar mass of dry air [kg/mol]
_RGAS = 8.314462


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


class HamSedimentation(PhysicsTerm):
    """Per-mode gravitational settling of interstitial aerosol tracers."""

    name: ClassVar[str] = "ham_sedimentation"
    category: ClassVar[str] = "aerosol_sedimentation"
    requires: ClassVar[tuple[str, ...]] = (
        "_ham_state", "air_density", "layer_thickness", "pressure_full",
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
        ham = diagnostics["_ham_state"]
        air_density = diagnostics["air_density"]
        dz = diagnostics["layer_thickness"]
        pressure = diagnostics["pressure_full"]
        temperature = state.temperature

        tracer_tends: dict[str, jnp.ndarray] = {}
        for i, mode in enumerate(self._spec.modes):
            if not mode.sediments:
                continue
            v = params.velocity_scale * stokes_velocity(
                ham.r_wet[i], ham.rho[i], temperature, pressure,
            )
            # Same velocity transports every interstitial tracer of the mode.
            names = [number_name(mode.short)] + [
                mass_name(sp, mode.short) for sp in mode.species
            ]
            for nm in names:
                q = state.tracers.get(nm)
                if q is None:
                    continue
                dq, _ = sediment_column(q, v, air_density, dz)
                tracer_tends[nm] = dq

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, diagnostics
