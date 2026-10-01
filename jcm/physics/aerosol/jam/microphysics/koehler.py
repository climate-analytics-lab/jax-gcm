"""Shared equilibrium water-uptake diagnostics for modal aerosol populations.

Diagnoses per-mode dry/wet radius, wet density and hygroscopicity from the
prognostic mass + number tracers of any :class:`ModalAerosolSpec`. These are
*state diagnostics*, not a microphysics process: nothing here evolves the
population, so every microphysics core needs them and none of them owns them.

They lived in ``placeholder.py`` until the in-tree core
(:mod:`~jcm.physics.aerosol.jam.microphysics.modal`) also needed them, at
which point a real core importing from a module named "placeholder" would
have been backwards. ``placeholder`` re-exports both names, so the five
existing call sites are unchanged.

**The water uptake is MAM's** (E3SM ``modal_aero_wateruptake.F90``:
``modal_aero_wateruptake_dr`` → ``_sub`` → ``modal_aero_kohler``; read
2026-10-02, decision D-A6 in the design doc), not the κ-Köhler growth factor
without a Kelvin term it was until then:

* the Köhler equilibrium ``ln s = A/r − κ r_d³/(r³ − r_d³)`` — Pruppacher and
  Klett's eq. 6-35 with the volume-mean hygroscopicity — solved for the wet
  radius of the mode's volume-mean particle, with the Kelvin coefficient
  ``A = 2 σ M_w/(R T ρ_w)`` at MAM's fixed 273 K (≈1.2 nm);
* relative humidity capped at 0.98;
* deliquescence/crystallization hysteresis: no water below the mode's
  ``rh_crystallization``, the Köhler water volume scaled linearly between
  it and ``rh_deliquescence``, the full solution above;
* the wet/dry *ratio* of that particle applied to the number-median
  diameter, and the wet density of the same particle.

Against the MAM4-JAX core on the same populations the κ-Köhler form gave an
Aitken wet radius up to 2× MAM's at mid humidity (no hysteresis) and
~20 % larger near saturation (no Kelvin term), and every uptake-dependent
process inherited the ratio; with MAM's uptake the two cores agree per
process to a few per cent (AER-12, design doc §9).
"""

from __future__ import annotations

import jax.numpy as jnp

import jcm.constants as c

from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.microphysics import numerics
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

#: MAM evaluates the Kelvin coefficient at a fixed temperature
#: (``modal_aero_kohler``: ``tair = 273``), and caps the humidity water
#: uptake sees (``modal_aero_wateruptake_dr``: ``rh = min(rh, 0.98)``).
KELVIN_TEMPERATURE = 273.0
RH_MAX_UPTAKE = 0.98
#: Molar mass of water [kg/mol] for the Kelvin coefficient.
_M_WATER = 0.018015
#: Newton iterations on the Köhler quartic. The iterate starts at the
#: no-Kelvin solution, an upper bound, on a convex increasing branch, so it
#: converges monotonically; twelve steps are far past float64 convergence.
_NEWTON_STEPS = 12

#: Floors that decide whether a mode is populated. The thresholds live in
#: ``moments`` so the microphysics core and these diagnostics agree on what
#: "empty" means; they are only ever used as ``where`` predicates here, never
#: as divisors — see ``moments.diameter_from_volume`` for why that matters.
_TINY_NUM = moments.TINY_NUMBER   # kg^-1
_TINY_VOL = moments.TINY_VOLUME   # m³/kg


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


def kelvin_coefficient() -> float:
    """Return MAM's Kelvin coefficient ``A = 2 σ M_w / (R T ρ_w)`` [m] at 273 K (≈1.2 nm)."""
    return (2.0 * c.surface_tension_water * _M_WATER
            / (c.r_universal * KELVIN_TEMPERATURE * c.rhow))


def kohler_growth_factor(r_dry, kappa, relative_humidity):
    """Return ``r_wet / r_dry`` from MAM's Köhler equilibrium, ``≥ 1``.

    Solves ``ln s = A/r − κ r_d³/(r³ − r_d³)`` (``modal_aero_kohler``'s
    quartic, ``r⁴ − (A/ln s) r³ + (κ r_d³/ln s − r_d³) r + A r_d³/ln s = 0``)
    for the smallest root at or above ``r_d``, in the dimensionless
    ``x = r/r_d`` so float32 holds it at any size: ``f(x) = x⁴ − α x³ +
    (κ/ln s − 1) x + α`` with ``α = A/(r_d ln s) < 0``. ``f(1) = κ/ln s ≤ 0``
    and ``f`` is positive at the no-Kelvin solution ``x₀³ = 1 + κ/(−ln s)``,
    convex and increasing between, so Newton from ``x₀`` converges
    monotonically onto the root. ``κ = 0`` (insoluble, or an empty mode)
    gives exactly 1 with no iteration.

    Args:
        r_dry: dry radius [m] the Kelvin term is evaluated at — MAM uses the
            mode's volume-mean particle.
        kappa: volume-weighted hygroscopicity.
        relative_humidity: ``s`` in (0, 1); callers cap it at
            :data:`RH_MAX_UPTAKE`.

    """
    s = jnp.clip(relative_humidity, 1.0e-10, RH_MAX_UPTAKE)
    ln_s = jnp.log(s)                                   # < 0
    alpha = numerics.divide(kelvin_coefficient(), r_dry * ln_s)
    linear = kappa / ln_s - 1.0                        # ≤ −1
    x = jnp.cbrt(1.0 + kappa / (-ln_s))                # no-Kelvin solution

    def f(x):
        return x ** 4 - alpha * x ** 3 + linear * x + alpha

    def df(x):
        return 4.0 * x ** 3 - 3.0 * alpha * x ** 2 + linear

    for _ in range(_NEWTON_STEPS):
        x = jnp.maximum(x - numerics.divide(f(x), df(x)), 1.0)
    return x


def uptake_growth_factor(r_dry_volume_mean, kappa, saturation, mode):
    """Return ``r_wet / r_dry`` with MAM's hysteresis (``modal_aero_wateruptake_sub``).

    Below ``mode.rh_crystallization`` the particle is dry; between it and
    ``mode.rh_deliquescence`` the Köhler water volume is scaled by
    ``(rh − rh_cryst)/(rh_deliq − rh_cryst)``; above, the Köhler solution
    stands. Continuous at both switch points.
    """
    rh = jnp.clip(saturation, 0.0, RH_MAX_UPTAKE)
    growth = kohler_growth_factor(r_dry_volume_mean, kappa, rh)
    water = growth ** 3 - 1.0                           # water volume / dry volume
    span = max(mode.rh_deliquescence - mode.rh_crystallization, 1.0e-5)
    partial = water * jnp.clip((rh - mode.rh_crystallization) / span, 0.0, 1.0)
    water = jnp.where(rh < mode.rh_crystallization, 0.0,
                      jnp.where(rh < mode.rh_deliquescence, partial, water))
    return jnp.cbrt(1.0 + water)


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

        number = numbers[number_name(mode.short)]

        # Volume-weighted κ and the DRY material density. Double-where, not
        # a floored divisor: ``mass / max(vol, tiny)`` has reverse-mode
        # cotangent ``-mass / tiny^2``, and ``1e-80`` underflows in float32,
        # so an empty mode gave NaN gradients for rho, kappa and r_wet
        # (measured 2026-10-01; finite in float64). The unselected branch
        # divides by 1.0 and the result is masked, which is finite in both.
        has_volume = total_vol > _TINY_VOL
        safe_vol = jnp.where(has_volume, total_vol, 1.0)
        # ... and through ``numerics.divide``, so the live quotient's reverse
        # pass never forms ``vol⁻²`` either: a volume of 1e-33 m³/kg (the
        # ECHAM column's top) has a square no float32 holds (AER-11).
        rho_dry = jnp.where(has_volume, numerics.divide(total_mass, safe_vol), default_density)
        kappa = jnp.where(has_volume, numerics.divide(vol_kappa, safe_vol), 0.0)

        # Number-median dry diameter from the mode's total volume: the
        # log-normal M_3 inversion, owned by ``moments`` so the microphysics
        # core and this diagnostic cannot disagree about it.
        dg = moments.diameter_from_volume(total_vol, number, mode)
        # Empty mode → fall back to the mode's reference diameter.
        dg = jnp.where(has_volume, dg, mode.dgnum)
        r_dry = 0.5 * dg

        # MAM's water uptake (module docstring): the Köhler solve with the
        # Kelvin term at the mode's VOLUME-MEAN particle, ``r_v = r_dry
        # exp(1.5 ln²σ)``, with hysteresis; the ratio applied to the
        # number-median radius, as ``dgncur_awet = dgncur_a · wetrad/dryrad``.
        r_volume_mean = r_dry * moments.moment_factor(mode, 3.0) ** (1.0 / 3.0)
        growth = uptake_growth_factor(r_volume_mean, kappa, saturation, mode)
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
