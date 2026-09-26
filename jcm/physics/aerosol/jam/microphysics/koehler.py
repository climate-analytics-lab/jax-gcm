"""Shared κ-Köhler equilibrium diagnostics for modal aerosol populations.

Diagnoses per-mode dry/wet radius, wet density and hygroscopicity from the
prognostic mass + number tracers of any :class:`ModalAerosolSpec`. These are
*state diagnostics*, not a microphysics process: nothing here evolves the
population, so every microphysics core needs them and none of them owns them.

They lived in ``placeholder.py`` until the in-tree core
(:mod:`~jcm.physics.aerosol.jam.microphysics.modal`) also needed them, at
which point a real core importing from a module named "placeholder" would
have been backwards. ``placeholder`` re-exports both names, so the five
existing call sites are unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c

from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

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
) -> JamAerosolState:
    """Diagnose per-mode radii/density/κ from mass+number tracers.

    Args:
        masses: ``{mass_tracer_name: (nlev, ncols)}`` interstitial dry mass
            [kg/kg], one per (mode, species).
        numbers: ``{number_tracer_name: (nlev, ncols)}`` interstitial number
            [kg^-1], one per mode.
        spec: the modal population.
        saturation: ambient water activity a_w (≈RH<1), ``(nlev, ncols)``.

    Returns:
        A :class:`JamAerosolState` with the mode axis first.

    """
    r_dry_modes, r_wet_modes, rho_modes, kappa_modes = [], [], [], []
    mass_modes, num_modes = [], []

    for mode in spec.modes:
        # Density an empty mode falls back to: its first species, as CAM
        # uses ``specdens_amode(1,mode)``.
        default_density = spec.species_props(mode.species[0]).density
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

        # Volume-weighted κ and the DRY material density (guarded); the
        # wet density follows once the growth factor is known.
        safe_vol = jnp.maximum(total_vol, _TINY_VOL)
        rho_dry = jnp.where(
            total_vol > _TINY_VOL, total_mass / safe_vol, default_density)
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

        # ρ_wet = (ρ_dry + (g³-1)ρ_w)/g³ — settling and deposition use the
        # wet radius, so they need the density of that same particle
        # (CAM's ``wetdens`` from ``modal_aero_wateruptake``).
        vol_growth = growth ** 3
        rho = (rho_dry + (vol_growth - 1.0) * c.rhow) / vol_growth

        r_dry_modes.append(r_dry)
        r_wet_modes.append(r_wet)
        rho_modes.append(rho)
        kappa_modes.append(kappa)
        mass_modes.append(total_mass)
        num_modes.append(numbers[number_name(mode.short)])

    stack = lambda xs: jnp.stack(xs, axis=0)
    return JamAerosolState(
        r_dry=stack(r_dry_modes),
        r_wet=stack(r_wet_modes),
        rho=stack(rho_modes),
        kappa=stack(kappa_modes),
        mass=stack(mass_modes),
        number=stack(num_modes),
    )
