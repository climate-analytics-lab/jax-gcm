"""Sea-ice diagnostic fluxes for the ECHAM multi-tile surface scheme.

Provides the bulk-flux pieces — albedo, roughness, and turbulent /
radiative surface fluxes — that the surface tile aggregator needs over
sea ice. Ice thermodynamics (heat conduction, surface melt, bottom
freeze, frazil-ice formation, snow evolution) are NOT modelled here:
the ECHAM flow prescribes ice fraction and ice surface temperature from
boundary forcing, and any prognostic ice / slab-ocean configuration
lives outside this repo.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jcm.constants import PhysicalConstants
from .surface_types import (
    SurfaceParameters, AtmosphericForcing,
    SurfaceFluxes, SurfaceTendencies
)

# Create constants instance
PHYS_CONST = PhysicalConstants()


@jax.jit
def compute_ice_albedo(
    ice_thickness: jnp.ndarray,
    snow_depth: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute sea ice albedo as a function of ice thickness and snow cover.

    Args:
        ice_thickness: Ice thickness [m] (ncol,)
        snow_depth: Snow depth [m] (ncol,)
        params: Surface parameters

    Returns:
        Tuple of (albedo_vis_direct, albedo_vis_diffuse,
                 albedo_nir_direct, albedo_nir_diffuse)

    """
    # Base ice albedo (depends on ice thickness)
    thick_ice_albedo_vis = 0.75  # Thick ice visible albedo
    thin_ice_albedo_vis = 0.50   # Thin ice visible albedo
    thick_ice_albedo_nir = 0.65  # Thick ice NIR albedo
    thin_ice_albedo_nir = 0.40   # Thin ice NIR albedo

    # Ice thickness transition
    h_transition = 0.5  # m
    ice_factor = jnp.tanh(ice_thickness / h_transition)

    # Base ice albedo
    albedo_ice_vis = thin_ice_albedo_vis + (thick_ice_albedo_vis - thin_ice_albedo_vis) * ice_factor
    albedo_ice_nir = thin_ice_albedo_nir + (thick_ice_albedo_nir - thin_ice_albedo_nir) * ice_factor

    # Snow albedo (higher than ice)
    snow_albedo_vis = 0.85
    snow_albedo_nir = 0.75

    # Snow masking factor
    snow_mask_depth = 0.01  # m (1 cm)
    snow_factor = jnp.minimum(snow_depth / snow_mask_depth, 1.0)

    # Combined albedo
    albedo_vis = albedo_ice_vis * (1.0 - snow_factor) + snow_albedo_vis * snow_factor
    albedo_nir = albedo_ice_nir * (1.0 - snow_factor) + snow_albedo_nir * snow_factor

    # Assume same for direct and diffuse
    albedo_vis_direct = albedo_vis
    albedo_vis_diffuse = albedo_vis
    albedo_nir_direct = albedo_nir
    albedo_nir_diffuse = albedo_nir

    return albedo_vis_direct, albedo_vis_diffuse, albedo_nir_direct, albedo_nir_diffuse


@jax.jit
def compute_ice_roughness(
    ice_thickness: jnp.ndarray,
    snow_depth: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """Compute sea ice surface roughness.

    Args:
        ice_thickness: Ice thickness [m] (ncol,)
        snow_depth: Snow depth [m] (ncol,)
        params: Surface parameters

    Returns:
        Ice roughness length [m] (ncol,)

    """
    # Base ice roughness
    z0_ice_base = params.z0_ice

    # Snow effect (snow is smoother)
    z0_snow = 1e-4  # m

    # Snow masking
    snow_mask_depth = 0.01  # m
    snow_factor = jnp.minimum(snow_depth / snow_mask_depth, 1.0)

    # Combined roughness
    z0_ice = z0_ice_base * (1.0 - snow_factor) + z0_snow * snow_factor

    return z0_ice


@jax.jit
def sea_ice_physics_step(
    atmospheric_state: AtmosphericForcing,
    ice_temp: jnp.ndarray,
    ice_thickness: jnp.ndarray,
    snow_depth: jnp.ndarray,
    ocean_temp: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[SurfaceFluxes, SurfaceTendencies, jnp.ndarray]:
    """Diagnostic sea-ice surface step: bulk fluxes only.

    Computes turbulent and radiative bulk fluxes over sea ice. Ice
    surface temperature, thickness, and snow depth are prescribed via
    the boundary forcing in the ECHAM flow, so this step returns zero
    tendencies for every prognostic ice / snow field. Sublimation is
    used in place of evaporation for the latent-heat flux.

    Args:
        atmospheric_state: Atmospheric forcing
        ice_temp: Ice temperature [K] (ncol, nice_layers); top layer
            is treated as the ice surface temperature for fluxes.
        ice_thickness: Ice thickness [m] (ncol, nice_layers)
        snow_depth: Snow depth [m] (ncol,)
        ocean_temp: Ocean temperature [K] (ncol,) — accepted for
            interface compatibility, unused (no ice-ocean conduction).
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        dt: Time step [s] — accepted for interface compatibility, unused.
        params: Surface parameters

    Returns:
        Tuple of (surface_fluxes, zero_tendencies, roughness_length)

    """
    ncol = ice_temp.shape[0]
    del dt, ocean_temp  # Unused — no prognostic step.

    # Surface temperature (top of ice/snow)
    surface_temp = ice_temp[:, 0]

    # Air density
    air_density = (atmospheric_state.pressure /
                  (PHYS_CONST.rd * atmospheric_state.temperature))

    # Surface saturation humidity
    e_sat = 611.0 * jnp.exp(17.27 * (surface_temp - PHYS_CONST.t0) /
                           (surface_temp - PHYS_CONST.t0 + 237.3))
    q_sat_surface = PHYS_CONST.eps * e_sat / atmospheric_state.pressure

    # Temperature and humidity differences. Positive convention: flux UP
    # from surface into the atmosphere when the surface is warmer / wetter
    # than the air. Same convention as the ocean tile and ``apply_surface``.
    delta_temp = surface_temp - atmospheric_state.temperature
    delta_humidity = q_sat_surface - atmospheric_state.humidity

    # Turbulent fluxes (latent uses ``alhs`` for sublimation over ice)
    sensible_heat = air_density * PHYS_CONST.cp * exchange_coeff_heat * delta_temp
    latent_heat = air_density * PHYS_CONST.alhs * exchange_coeff_moisture * delta_humidity

    # Momentum fluxes
    momentum_u = air_density * exchange_coeff_momentum * atmospheric_state.u_wind
    momentum_v = air_density * exchange_coeff_momentum * atmospheric_state.v_wind

    # Evaporation/sublimation
    evaporation = air_density * exchange_coeff_moisture * delta_humidity

    # Ice albedo
    albedo_vis_dir, albedo_vis_dif, albedo_nir_dir, albedo_nir_dif = compute_ice_albedo(
        jnp.sum(ice_thickness, axis=1), snow_depth, params
    )

    # Net shortwave
    albedo_mean = 0.25 * (albedo_vis_dir + albedo_vis_dif + albedo_nir_dir + albedo_nir_dif)
    shortwave_net = atmospheric_state.sw_downward * (1.0 - albedo_mean)

    # Net longwave
    lw_upward = params.emissivity * params.stefan_boltzmann * surface_temp**4
    longwave_net = atmospheric_state.lw_downward - lw_upward

    # Ice roughness
    roughness = compute_ice_roughness(
        jnp.sum(ice_thickness, axis=1), snow_depth, params
    )

    # Create single-tile flux structure (ice only)
    fluxes = SurfaceFluxes(
        sensible_heat=sensible_heat[:, None],
        latent_heat=latent_heat[:, None],
        longwave_net=longwave_net[:, None],
        shortwave_net=shortwave_net[:, None],
        ground_heat=jnp.zeros((ncol, 1)),
        momentum_u=momentum_u[:, None],
        momentum_v=momentum_v[:, None],
        evaporation=evaporation[:, None],
        transpiration=jnp.zeros((ncol, 1)),
        sensible_heat_mean=sensible_heat,
        latent_heat_mean=latent_heat,
        momentum_u_mean=momentum_u,
        momentum_v_mean=momentum_v,
        evaporation_mean=evaporation
    )

    nice_layers = ice_temp.shape[1]
    tendencies = SurfaceTendencies(
        surface_temp_tendency=jnp.zeros((ncol, 1)),
        ocean_temp_tendency=jnp.zeros(ncol),
        ice_temp_tendency=jnp.zeros((ncol, nice_layers)),
        soil_temp_tendency=jnp.zeros((ncol, 1)),
        ice_thickness_tendency=jnp.zeros((ncol, nice_layers)),
        snow_depth_tendency=jnp.zeros(ncol),
        soil_moisture_tendency=jnp.zeros((ncol, 1))
    )

    return fluxes, tendencies, roughness
