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


def particle_mean_mass(mode: AerosolMode, species_density: float,
                       emission_diameter=None):
    """Mean single-particle mass [kg] of freshly emitted material.

    Freshly emitted particles are not at their class's equilibrium size, so a
    scheme that knows the volume-mean diameter D of the size distribution it
    emits into passes it and gets ``m_p = ρ·(π/6)·D³``; the ``None`` fallback
    is the class's own geometry, ``m_p = ρ/number_factor``. Since the emitted
    number for a given mass scales as D⁻³, the difference is large wherever
    the emitted and equilibrium sizes differ.

    The mechanism only; no scheme in this module supplies a diameter yet.
    Each emission scheme owns the size its own reference prescribes, and this
    is a HAMMOZ package: the primary-carbon and primary-sulfate sizes come
    from HAM (Stier et al. 2005) with the emissions port, not from CAM/CESM.
    """
    if emission_diameter is None:
        return species_density / mode.number_factor
    return species_density * (math.pi / 6.0) * emission_diameter ** 3


def emit_over_profile(
    flux: jnp.ndarray,             # (ncols,) surface flux [X/m²/s]
    weights: jnp.ndarray,          # (nlev, ncols) vertical weights summing to 1
    air_density: jnp.ndarray,      # (nlev, ncols) [kg/m³]
    layer_thickness: jnp.ndarray,  # (nlev, ncols) [m]
) -> jnp.ndarray:
    """Spread a surface mass/number flux over a vertical profile.

    Returns a ``(nlev, ncols)`` mixing-ratio tendency [X/kg/s]. Because
    ``weights`` sum to 1 over levels, the column-integrated emitted amount
    equals the input flux (``Σ ρ_k Δz_k · dq_k = flux``), so it is
    mass-conserving for any (differentiable) profile. Works for both mass
    [kg/kg/s] and number [kg⁻¹/s] tendencies.
    """
    return weights * flux[jnp.newaxis, :] / (air_density * layer_thickness)


def distribute_surface_flux(
    spec: ModalAerosolSpec,
    fluxes: list[tuple[str, str, jnp.ndarray]],
    air_density: jnp.ndarray,       # (nlev, ncols)
    layer_thickness: jnp.ndarray,   # (nlev, ncols)
) -> dict[str, jnp.ndarray]:
    """Build bottom-layer tracer tendencies from surface mass fluxes.

    Args:
        spec: the modal population.
        fluxes: list of ``(species_token, mode_short, mass_flux)`` or
            ``(species_token, mode_short, mass_flux, emission_diameter)`` with
            ``mass_flux`` shaped ``(ncols,)`` in kg/m²/s and the optional
            volume-mean diameter [m] of the emitted size distribution (see
            :func:`particle_mean_mass`); omitted, the class geometry is used.
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

    for species, mode_short, mass_flux, *rest in fluxes:
        mode = spec.mode(mode_short)
        props = spec.species_props(species)
        mname = mass_name(species, mode_short)
        dq = jnp.zeros((nlev, ncols)).at[-1].set(mass_flux * inv)
        tends[mname] = tends.get(mname, jnp.zeros((nlev, ncols))) + dq
        m_p = particle_mean_mass(mode, props.density, *rest)
        number_flux[mode_short] = number_flux[mode_short] + mass_flux / m_p

    for mode_short, n_flux in number_flux.items():
        dq = jnp.zeros((nlev, ncols)).at[-1].set(n_flux * inv)
        tends[number_name(mode_short)] = dq

    return tends
