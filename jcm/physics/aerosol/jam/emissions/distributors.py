"""Distribute surface mass fluxes into modal tracer tendencies.

A microphysics-family-aware mapping from ``(species, mode, mass_flux)`` triples
to per-tracer tendencies on the lowest model layer, including the implied
number flux (from the emitted mode's mean single-particle mass). Only the
modal path is implemented here; a sectional distributor is future work (#491).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from jcm.physics.aerosol.jam.population import AerosolMode, ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name


def particle_mean_mass(mode: AerosolMode, species_density: float) -> float:
    """Mean single-particle mass [kg] of a log-normal mode at its ref size.

    ``m_p = ρ_p (π/6) Dg³ exp(9/2 ln²σ)`` (mass-equivalent of the number
    distribution's third moment).
    """
    ln_sigma = math.log(mode.geom_std_dev)
    return (
        species_density
        * (math.pi / 6.0)
        * mode.dgnum ** 3
        * math.exp(4.5 * ln_sigma ** 2)
    )


def distribute_surface_flux(
    spec: ModalAerosolSpec,
    fluxes: list[tuple[str, str, jnp.ndarray]],
    air_density: jnp.ndarray,       # (nlev, ncols)
    layer_thickness: jnp.ndarray,   # (nlev, ncols)
) -> dict[str, jnp.ndarray]:
    """Build bottom-layer tracer tendencies from surface mass fluxes.

    Args:
        spec: the modal population.
        fluxes: list of ``(species_token, mode_short, mass_flux)`` with
            ``mass_flux`` shaped ``(ncols,)`` in kg/m²/s.
        air_density: air density [kg/m³].
        layer_thickness: geometric layer thickness [m].

    Returns:
        ``{tracer_name: (nlev, ncols) tendency}`` for mass and number,
        non-zero only on the lowest layer.

    """
    nlev, ncols = air_density.shape
    rho_sfc = air_density[-1]
    dz_sfc = layer_thickness[-1]
    inv = 1.0 / (rho_sfc * dz_sfc)

    tends: dict[str, jnp.ndarray] = {}
    number_flux = {mode.short: jnp.zeros((ncols,)) for mode in spec.modes}

    for species, mode_short, mass_flux in fluxes:
        mode = spec.mode(mode_short)
        props = spec.species_props(species)
        mname = mass_name(species, mode_short)
        dq = jnp.zeros((nlev, ncols)).at[-1].set(mass_flux * inv)
        tends[mname] = tends.get(mname, jnp.zeros((nlev, ncols))) + dq
        m_p = particle_mean_mass(mode, props.density)
        number_flux[mode_short] = number_flux[mode_short] + mass_flux / m_p

    for mode_short, n_flux in number_flux.items():
        dq = jnp.zeros((nlev, ncols)).at[-1].set(n_flux * inv)
        tends[number_name(mode_short)] = dq

    return tends
