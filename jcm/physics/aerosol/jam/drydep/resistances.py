"""Resistances and deposition velocity for aerosol dry deposition.

Resistance-in-series model ``v_dep = 1/(r_a + r_b)`` for the turbulent +
Brownian/impaction removal of aerosol at the surface. Gravitational settling
is handled separately by the sedimentation term (which already fluxes mass
out of the lowest layer), so this term adds only the non-gravitational part
to avoid double counting.

* ``r_a`` aerodynamic resistance — neutral log-law (Monin-Obukhov stability
  correction is a future refinement).
* ``r_b`` quasi-laminar resistance — Slinn & Slinn (1980) over-water form
  combining Brownian diffusion (Schmidt number) and inertial impaction
  (Stokes number). Mirrors the structure of ``mo_hammoz_drydep`` /
  Ganzeveld, specialised to the aquaplanet ocean surface.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.constants import ak as _KB  # Boltzmann constant [J/K]
from jcm.constants import grav as _G
from jcm.constants import m_air as _MA
from jcm.constants import r_universal as _RGAS
from jcm.physics.aerosol.jam.sedimentation.sedi_term import moment_radius


def air_viscosity(temperature: jnp.ndarray) -> jnp.ndarray:
    """Dynamic viscosity of air [Pa·s] (Sutherland)."""
    return 1.458e-6 * temperature ** 1.5 / (temperature + 110.4)


def cunningham_slip(
    r: jnp.ndarray, temperature: jnp.ndarray, pressure: jnp.ndarray
) -> jnp.ndarray:
    """Cunningham slip-correction factor [-]."""
    mu = air_viscosity(temperature)
    mfp = (mu / pressure) * jnp.sqrt(jnp.pi * _RGAS * temperature / (2.0 * _MA))
    kn = mfp / jnp.maximum(r, 1.0e-10)
    return 1.0 + kn * (1.257 + 0.4 * jnp.exp(-1.1 / jnp.maximum(kn, 1e-12)))


def aerodynamic_resistance(
    u_star: jnp.ndarray,
    *,
    z_ref: float = 10.0,
    z0: float = 1.0e-4,
    karman: float = 0.4,
) -> jnp.ndarray:
    """Neutral aerodynamic resistance r_a [s/m]."""
    u = jnp.maximum(u_star, 1.0e-3)
    return jnp.log(z_ref / z0) / (karman * u)


def quasi_laminar_resistance(
    r_wet: jnp.ndarray,
    v_grav: jnp.ndarray,
    u_star: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    air_density: jnp.ndarray,
) -> jnp.ndarray:
    """Slinn & Slinn (1980) over-water quasi-laminar resistance r_b [s/m].

    ``1/r_b = u* (Sc^{-1/2} + 10^{-3/St})`` with Brownian Schmidt number Sc
    and inertial Stokes number St = v_grav u*² / (g ν).
    """
    mu = air_viscosity(temperature)
    nu = mu / air_density                              # kinematic viscosity
    cc = cunningham_slip(r_wet, temperature, pressure)
    diffusivity = _KB * temperature * cc / (6.0 * jnp.pi * mu * jnp.maximum(r_wet, 1e-10))
    schmidt = nu / diffusivity
    u = jnp.maximum(u_star, 1.0e-3)
    stokes = v_grav * u ** 2 / (_G * nu)
    conductance = u * (schmidt ** -0.5 + 10.0 ** (-3.0 / jnp.maximum(stokes, 1e-6)))
    return 1.0 / jnp.maximum(conductance, 1e-12)


def deposition_velocity(
    r_wet: jnp.ndarray,
    v_grav: jnp.ndarray,
    u_star: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    air_density: jnp.ndarray,
    *,
    geom_std_dev: float,
    moment: int,
    z_ref: float = 10.0,
    z0: float = 1.0e-4,
    karman: float = 0.4,
) -> jnp.ndarray:
    """Non-gravitational dry-deposition velocity 1/(r_a + r_b) [m/s].

    ``r_wet`` is the mode's number-median radius and ``moment`` selects which
    moment is being deposited (0 number, 3 mass). The scaling has to reach
    the radius and not just ``v_grav``: the Schmidt number that sets Brownian
    removal is built from the radius directly, and Brownian dominates for
    submicron modes, so scaling only the Stokes/impaction term would leave
    the mass and number velocities effectively identical.
    """
    ra = aerodynamic_resistance(u_star, z_ref=z_ref, z0=z0, karman=karman)
    rb = quasi_laminar_resistance(
        moment_radius(r_wet, geom_std_dev=geom_std_dev, moment=moment),
        v_grav, u_star, temperature, pressure, air_density,
    )
    return 1.0 / (ra + rb)
