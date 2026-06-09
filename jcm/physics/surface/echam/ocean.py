"""Ocean surface diagnostic fluxes for the ECHAM multi-tile surface scheme.

Provides the bulk-flux pieces — albedo, Charnock roughness, and
turbulent / radiative surface fluxes — that the surface tile aggregator
needs over open water. Ocean temperature itself is prescribed (boundary
SST) in ECHAM's flow, so this module computes diagnostic fluxes only and
returns zero ocean-temperature / ice tendencies. Slab / mixed-layer
ocean and ocean-atmosphere coupling fluxes live outside this repo.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from .surface_types import (
    SurfaceParameters, AtmosphericForcing,
    SurfaceFluxes, SurfaceTendencies
)


@jax.jit
def compute_ocean_albedo(
    solar_zenith_angle: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ocean surface albedo as a function of solar zenith angle.

    Args:
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        params: Surface parameters

    Returns:
        Tuple of (albedo_vis_direct, albedo_vis_diffuse,
                 albedo_nir_direct, albedo_nir_diffuse)

    """
    # Fresnel reflection formula for direct beam
    # Simplified parameterization based on zenith angle
    cos_theta = jnp.cos(solar_zenith_angle)

    # Direct beam albedo (increases with zenith angle)
    albedo_direct_vis = 0.05 + 0.15 * (1.0 - cos_theta)**2
    albedo_direct_nir = 0.05 + 0.10 * (1.0 - cos_theta)**2

    # Diffuse albedo (constant)
    albedo_diffuse_vis = jnp.full_like(albedo_direct_vis, 0.06)
    albedo_diffuse_nir = jnp.full_like(albedo_direct_nir, 0.06)

    return albedo_direct_vis, albedo_diffuse_vis, albedo_direct_nir, albedo_diffuse_nir


@jax.jit
def compute_ocean_roughness(
    wind_speed: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """Compute ocean surface roughness using Charnock relation.

    Args:
        wind_speed: Wind speed [m/s] (ncol,)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        params: Surface parameters

    Returns:
        Ocean roughness length [m] (ncol,)

    """
    # Relative wind speed (wind minus ocean current)
    # For simplicity, assume wind_speed is the magnitude and ocean currents are small
    wind_rel_speed = jnp.maximum(wind_speed, params.min_wind_speed)

    # Charnock relation: z0 = alpha * u*^2 / g
    # where u* = sqrt(tau / rho_air)
    # Simplified: z0 = alpha * U^2 / g with alpha ≈ 0.018
    charnock_alpha = 0.018
    gravity = c.grav

    z0_ocean = charnock_alpha * wind_rel_speed**2 / gravity

    # Apply minimum and maximum bounds
    z0_min = 1e-5  # 0.01 mm minimum
    z0_max = 0.1   # 10 cm maximum (for very high winds)
    z0_ocean = jnp.clip(z0_ocean, z0_min, z0_max)

    return z0_ocean


@jax.jit
def compute_ocean_surface_fluxes(
    atmospheric_state: AtmosphericForcing,
    ocean_temp: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray,
    solar_zenith_angle: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[SurfaceFluxes, jnp.ndarray]:
    """Compute surface fluxes over ocean.

    Args:
        atmospheric_state: Atmospheric forcing
        ocean_temp: Ocean surface temperature [K] (ncol,)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        params: Surface parameters

    Returns:
        Tuple of (surface_fluxes, roughness_length)

    """
    ncol = ocean_temp.shape[0]

    # Air density
    air_density = (atmospheric_state.pressure /
                  (c.rd * atmospheric_state.temperature))

    # Ocean surface saturation humidity
    # Saturation vapor pressure over ocean
    T_celsius = ocean_temp - c.tmelt
    e_sat = 611.0 * jnp.exp(17.27 * T_celsius / (T_celsius + 237.3))  # Pa
    q_sat_ocean = c.eps * e_sat / atmospheric_state.pressure

    # Temperature and humidity differences
    delta_temp = ocean_temp - atmospheric_state.temperature
    delta_humidity = q_sat_ocean - atmospheric_state.humidity

    # Wind relative to ocean surface
    wind_rel_u = atmospheric_state.u_wind - ocean_u
    wind_rel_v = atmospheric_state.v_wind - ocean_v
    wind_rel_speed = jnp.sqrt(wind_rel_u**2 + wind_rel_v**2)

    # Sensible heat flux [W/m²]
    sensible_heat = air_density * c.cpd * exchange_coeff_heat * delta_temp

    # Momentum fluxes [N/m²]
    momentum_u = air_density * exchange_coeff_momentum * wind_rel_u
    momentum_v = air_density * exchange_coeff_momentum * wind_rel_v

    # Evaporation rate [kg/m²/s]
    evaporation = air_density * exchange_coeff_moisture * delta_humidity

    # Latent heat flux [W/m²]
    latent_heat = c.alhc * evaporation

    # Ocean albedo
    albedo_vis_dir, albedo_vis_dif, albedo_nir_dir, albedo_nir_dif = compute_ocean_albedo(
        solar_zenith_angle, params
    )

    # Shortwave absorption (simplified - assume 50% visible, 50% NIR)
    albedo_mean = (albedo_vis_dir + albedo_vis_dif + albedo_nir_dir + albedo_nir_dif) / 4.0
    shortwave_net = atmospheric_state.sw_downward * (1.0 - albedo_mean)

    # Longwave flux
    lw_upward = params.emissivity * params.stefan_boltzmann * ocean_temp**4
    longwave_net = atmospheric_state.lw_downward - lw_upward

    # Ocean roughness
    roughness = compute_ocean_roughness(wind_rel_speed, ocean_u, ocean_v, params)

    # Create single-tile flux structure (ocean only)
    fluxes = SurfaceFluxes(
        sensible_heat=sensible_heat[:, None],
        latent_heat=latent_heat[:, None],
        longwave_net=longwave_net[:, None],
        shortwave_net=shortwave_net[:, None],
        ground_heat=jnp.zeros((ncol, 1)),  # No ground heat flux for ocean
        momentum_u=momentum_u[:, None],
        momentum_v=momentum_v[:, None],
        evaporation=evaporation[:, None],
        transpiration=jnp.zeros((ncol, 1)),  # No transpiration for ocean
        sensible_heat_mean=sensible_heat,
        latent_heat_mean=latent_heat,
        momentum_u_mean=momentum_u,
        momentum_v_mean=momentum_v,
        evaporation_mean=evaporation
    )

    return fluxes, roughness


@jax.jit
def ocean_physics_step(
    atmospheric_state: AtmosphericForcing,
    ocean_temp: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray,
    solar_zenith_angle: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[SurfaceFluxes, SurfaceTendencies, jnp.ndarray]:
    """Diagnostic ocean surface step: bulk fluxes only, no prognostic state.

    Ocean temperature is prescribed by the boundary SST in the ECHAM flow,
    so this step computes the bulk surface fluxes via
    ``compute_ocean_surface_fluxes`` and returns zero tendencies for
    every prognostic field. Slab-ocean evolution lives outside this repo.

    Args:
        atmospheric_state: Atmospheric forcing
        ocean_temp: Ocean temperature [K] (ncol,)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        dt: Time step [s] — accepted for interface compatibility, unused.
        params: Surface parameters

    Returns:
        Tuple of (surface_fluxes, zero_tendencies, roughness_length)

    """
    ncol = ocean_temp.shape[0]
    del dt  # Unused — no prognostic step.

    # Compute surface fluxes
    surface_fluxes, roughness = compute_ocean_surface_fluxes(
        atmospheric_state, ocean_temp, ocean_u, ocean_v,
        exchange_coeff_heat, exchange_coeff_moisture, exchange_coeff_momentum,
        solar_zenith_angle, params
    )

    tendencies = SurfaceTendencies(
        surface_temp_tendency=jnp.zeros((ncol, 1)),
        ocean_temp_tendency=jnp.zeros(ncol),
        ice_temp_tendency=jnp.zeros((ncol, 1)),
        soil_temp_tendency=jnp.zeros((ncol, 1)),
        ice_thickness_tendency=jnp.zeros((ncol, 1)),
        snow_depth_tendency=jnp.zeros(ncol),
        soil_moisture_tendency=jnp.zeros((ncol, 1))
    )

    return surface_fluxes, tendencies, roughness
