"""κ-Köhler equilibrium placeholder microphysics core.

Computes diagnostic per-mode dry/wet radius, density and hygroscopicity from
the prognostic mass + number tracers, with **zero** tendency. It is
mass-conserving and differentiable, and exposes the real MAM4 4-mode
population shape, so it exercises the whole harness (emissions, activation,
deposition, sedimentation) end-to-end while the real microphysics core
(MAM4-JAX, #490) is wired in later. No coagulation/condensation/nucleation/
ageing happens here — those belong to the real core.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from jcm.physics.aerosol.ham.ham_state import HamAerosolState
from jcm.physics.aerosol.ham.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.ham.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.ham.population import ModalAerosolSpec
from jcm.physics.aerosol.ham.tracer_layout import mass_name, number_name
from jcm.physics_interface import PhysicsTendency

#: Floors to keep radius/density/κ finite where a mode is empty.
_TINY_NUM = 1.0e-30      # kg^-1
_TINY_VOL = 1.0e-40      # m³/kg


def saturation_ratio(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    *,
    rh_max: float = 0.99,
) -> jnp.ndarray:
    """Ambient water-activity ≈ relative humidity, clipped below 1.

    Tetens saturation vapour pressure over liquid water; RH clipped to
    ``rh_max`` so the sub-saturated κ-Köhler growth factor stays finite.
    """
    t_c = temperature - 273.15
    es = 611.2 * jnp.exp(17.62 * t_c / (temperature - 30.03))  # Pa
    e = specific_humidity * pressure / (0.622 + 0.378 * specific_humidity)
    return jnp.clip(e / jnp.maximum(es, 1.0e-3), 0.0, rh_max)


def equilibrium_modal_state(
    masses: dict[str, jnp.ndarray],
    numbers: dict[str, jnp.ndarray],
    spec: ModalAerosolSpec,
    saturation: jnp.ndarray,
) -> HamAerosolState:
    """Diagnose per-mode radii/density/κ from mass+number tracers.

    Args:
        masses: ``{mass_tracer_name: (nlev, ncols)}`` interstitial dry mass
            [kg/kg], one per (mode, species).
        numbers: ``{number_tracer_name: (nlev, ncols)}`` interstitial number
            [kg^-1], one per mode.
        spec: the modal population.
        saturation: ambient water activity a_w (≈RH<1), ``(nlev, ncols)``.

    Returns:
        A :class:`HamAerosolState` with the mode axis first.

    """
    r_dry_modes, r_wet_modes, rho_modes, kappa_modes = [], [], [], []
    mass_modes, num_modes = [], []

    for mode in spec.modes:
        # Per-species volume [m³/kg] and totals.
        total_mass = jnp.zeros_like(saturation)
        total_vol = jnp.zeros_like(saturation)
        vol_kappa = jnp.zeros_like(saturation)
        for sp in mode.species:
            m = masses[mass_name(sp, mode.short)]
            props = spec.species_props(sp)
            v = m / props.density
            total_mass = total_mass + m
            total_vol = total_vol + v
            vol_kappa = vol_kappa + v * props.hygroscopicity

        number = jnp.maximum(numbers[number_name(mode.short)], _TINY_NUM)

        # Mass-mean particle density and volume-weighted κ (guarded).
        safe_vol = jnp.maximum(total_vol, _TINY_VOL)
        rho = total_mass / safe_vol
        rho = jnp.where(total_vol > _TINY_VOL, rho, props.density)
        kappa = jnp.where(total_vol > _TINY_VOL, vol_kappa / safe_vol, 0.0)

        # Number-median dry diameter from total volume of a log-normal mode:
        #   V = N (π/6) Dg³ exp(9/2 ln²σ)   →   Dg = (V / (N k))^(1/3)
        ln_sigma = jnp.log(mode.geom_std_dev)
        k = (jnp.pi / 6.0) * jnp.exp(4.5 * ln_sigma ** 2)
        dg_cubed = total_vol / (number * k)
        dg = jnp.cbrt(jnp.maximum(dg_cubed, mode.dgnum_lo ** 3))
        dg = jnp.clip(dg, mode.dgnum_lo, mode.dgnum_hi)
        # Empty mode → fall back to the mode's reference diameter.
        dg = jnp.where(total_vol > _TINY_VOL, dg, mode.dgnum)
        r_dry = 0.5 * dg

        # κ-Köhler sub-saturated growth factor (Petters & Kreidenweis 2007,
        # Kelvin term dropped — valid for accumulation/coarse sizes < ~98% RH):
        #   (r_wet/r_dry)³ = 1 + κ a_w/(1 - a_w)
        growth = jnp.cbrt(1.0 + kappa * saturation / (1.0 - saturation))
        r_wet = r_dry * growth

        r_dry_modes.append(r_dry)
        r_wet_modes.append(r_wet)
        rho_modes.append(rho)
        kappa_modes.append(kappa)
        mass_modes.append(total_mass)
        num_modes.append(numbers[number_name(mode.short)])

    stack = lambda xs: jnp.stack(xs, axis=0)
    return HamAerosolState(
        r_dry=stack(r_dry_modes),
        r_wet=stack(r_wet_modes),
        rho=stack(rho_modes),
        kappa=stack(kappa_modes),
        mass_per_mode=stack(mass_modes),
        number_per_mode=stack(num_modes),
    )


class PlaceholderMicrophysics(ModalMicrophysicsTerm):
    """Zero-tendency κ-Köhler equilibrium core on the MAM4 4-mode population."""

    name: ClassVar[str] = "ham_placeholder_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    spec: ClassVar[ModalAerosolSpec] = MAM4_SPEC

    def __init__(self, spec: ModalAerosolSpec | None = None):
        """Optionally override the population (defaults to MAM4 4-mode)."""
        # Allow injecting an alternative population (e.g. a reduced test
        # spec); defaults to the full MAM4 4-mode layout.
        if spec is not None:
            self.spec = spec
        # No tunable parameters; keep an nnx-friendly marker attribute.
        self._initialized = True

    def __call__(self, state, diagnostics, forcing, terrain):
        spec = self.spec
        zeros = jnp.zeros_like(state.temperature)
        masses = {
            mass_name(sp, mode.short): state.tracers.get(
                mass_name(sp, mode.short), zeros
            )
            for mode in spec.modes
            for sp in mode.species
        }
        numbers = {
            number_name(mode.short): state.tracers.get(
                number_name(mode.short), zeros
            )
            for mode in spec.modes
        }

        sat = saturation_ratio(
            state.temperature, state.specific_humidity,
            diagnostics["pressure_full"],
        )
        ham_state = equilibrium_modal_state(masses, numbers, spec, sat)

        # Zero tendency — equilibrium diagnostics only.
        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, "_ham_state": ham_state}
