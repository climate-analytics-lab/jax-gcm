"""
Ocean surface physics for ICON surface scheme.

This module implements a simple mixed-layer ocean model and ocean-atmosphere
coupling following ICON's approach.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing, 
    SurfaceFluxes, SurfaceTendencies
)

# Create constants instance
PHYS_CONST = PhysicalConstants()


@jax.jit
def compute_ocean_albedo(
    solar_zenith_angle: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute ocean surface albedo as a function of solar zenith angle.
    
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
    """
    Compute ocean surface roughness using Charnock relation.
    
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
    gravity = PHYS_CONST.grav
    
    z0_ocean = charnock_alpha * wind_rel_speed**2 / gravity
    
    # Apply minimum and maximum bounds
    z0_min = 1e-5  # 0.01 mm minimum
    z0_max = 0.1   # 10 cm maximum (for very high winds)
    z0_ocean = jnp.clip(z0_ocean, z0_min, z0_max)
    
    return z0_ocean


@jax.jit
def mixed_layer_ocean_step(
    ocean_temp: jnp.ndarray,
    surface_heat_flux: jnp.ndarray,
    shortwave_penetration: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Evolve mixed layer ocean temperature.
    
    Args:
        ocean_temp: Ocean temperature [K] (ncol,)
        surface_heat_flux: Net surface heat flux [W/m²] (ncol,)
        shortwave_penetration: Shortwave penetrating into ocean [W/m²] (ncol,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Ocean temperature tendency [K/s] (ncol,)
    """
    # Mixed layer heat capacity [J/m²/K]
    heat_capacity = params.rho_water * params.cp_water * params.ml_depth
    
    # Total heat flux into mixed layer
    total_heat_flux = surface_heat_flux + shortwave_penetration
    
    # Temperature tendency
    temp_tendency = total_heat_flux / heat_capacity
    
    return temp_tendency


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
    """
    Compute surface fluxes over ocean.
    
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
                  (PHYS_CONST.rd * atmospheric_state.temperature))
    
    # Ocean surface saturation humidity
    # Saturation vapor pressure over ocean
    T_celsius = ocean_temp - PHYS_CONST.t0
    e_sat = 611.0 * jnp.exp(17.27 * T_celsius / (T_celsius + 237.3))  # Pa
    q_sat_ocean = PHYS_CONST.eps * e_sat / atmospheric_state.pressure
    
    # Temperature and humidity differences
    delta_temp = atmospheric_state.temperature - ocean_temp
    delta_humidity = atmospheric_state.humidity - q_sat_ocean
    
    # Wind relative to ocean surface
    wind_rel_u = atmospheric_state.u_wind - ocean_u
    wind_rel_v = atmospheric_state.v_wind - ocean_v
    wind_rel_speed = jnp.sqrt(wind_rel_u**2 + wind_rel_v**2)
    
    # Sensible heat flux [W/m²]
    sensible_heat = air_density * PHYS_CONST.cp * exchange_coeff_heat * delta_temp
    
    # Latent heat flux [W/m²]
    latent_heat = air_density * PHYS_CONST.alhc * exchange_coeff_moisture * delta_humidity
    
    # Momentum fluxes [N/m²]
    momentum_u = air_density * exchange_coeff_momentum * wind_rel_u
    momentum_v = air_density * exchange_coeff_momentum * wind_rel_v
    
    # Evaporation rate [kg/m²/s]
    evaporation = air_density * exchange_coeff_moisture * delta_humidity
    
    # Ocean albedo
    albedo_vis_dir, albedo_vis_dif, albedo_nir_dir, albedo_nir_dif = compute_ocean_albedo(
        solar_zenith_angle, params
    )
    
    # Shortwave absorption (simplified - assume 50% visible, 50% NIR)
    albedo_mean = 0.5 * (albedo_vis_dir + albedo_vis_dif + albedo_nir_dir + albedo_nir_dif) / 2.0
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
def ocean_surface_temperature_step(
    ocean_temp: jnp.ndarray,
    surface_fluxes: SurfaceFluxes,
    shortwave_penetration_fraction: float = 0.7,
    dt: float = 3600.0,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Update ocean surface temperature using energy balance.
    
    Args:
        ocean_temp: Ocean temperature [K] (ncol,)
        surface_fluxes: Surface fluxes
        shortwave_penetration_fraction: Fraction of SW penetrating below surface
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Ocean temperature tendency [K/s] (ncol,)
    """
    # Net surface heat flux
    net_heat_flux = (surface_fluxes.shortwave_net[:, 0] + 
                    surface_fluxes.longwave_net[:, 0] - 
                    surface_fluxes.sensible_heat[:, 0] - 
                    surface_fluxes.latent_heat[:, 0])
    
    # Account for shortwave penetration
    surface_absorbed_sw = (surface_fluxes.shortwave_net[:, 0] * 
                          (1.0 - shortwave_penetration_fraction))
    penetrating_sw = (surface_fluxes.shortwave_net[:, 0] * 
                     shortwave_penetration_fraction)
    
    # Surface heat flux (excluding penetrating shortwave)
    surface_heat_flux = (surface_absorbed_sw + surface_fluxes.longwave_net[:, 0] - 
                        surface_fluxes.sensible_heat[:, 0] - 
                        surface_fluxes.latent_heat[:, 0])
    
    # Mixed layer temperature tendency
    temp_tendency = mixed_layer_ocean_step(
        ocean_temp, surface_heat_flux, penetrating_sw, dt, params
    )
    
    return temp_tendency


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
    """
    Complete ocean physics step.
    
    Args:
        atmospheric_state: Atmospheric forcing
        ocean_temp: Ocean temperature [K] (ncol,)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        solar_zenith_angle: Solar zenith angle [rad] (ncol,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Tuple of (surface_fluxes, tendencies, roughness_length)
    """
    ncol = ocean_temp.shape[0]
    
    # Compute surface fluxes
    surface_fluxes, roughness = compute_ocean_surface_fluxes(
        atmospheric_state, ocean_temp, ocean_u, ocean_v,
        exchange_coeff_heat, exchange_coeff_moisture, exchange_coeff_momentum,
        solar_zenith_angle, params
    )
    
    # Compute temperature tendency
    temp_tendency = ocean_surface_temperature_step(
        ocean_temp, surface_fluxes, dt=dt, params=params
    )
    
    # Create tendencies structure
    tendencies = SurfaceTendencies(
        surface_temp_tendency=temp_tendency[:, None],  # Single ocean tile
        ocean_temp_tendency=temp_tendency,
        ice_temp_tendency=jnp.zeros((ncol, 1)),  # No ice
        soil_temp_tendency=jnp.zeros((ncol, 1)),  # No soil
        ice_thickness_tendency=jnp.zeros((ncol, 1)),
        snow_depth_tendency=jnp.zeros(ncol),
        soil_moisture_tendency=jnp.zeros((ncol, 1))
    )
    
    return surface_fluxes, tendencies, roughness


@jax.jit
def compute_ocean_coupling_fluxes(
    surface_fluxes: SurfaceFluxes,
    precipitation_rate: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute fluxes for ocean-atmosphere coupling.
    
    Args:
        surface_fluxes: Surface fluxes
        precipitation_rate: Precipitation rate [kg/m²/s] (ncol,)
        params: Surface parameters
        
    Returns:
        Tuple of (heat_flux, freshwater_flux, momentum_flux_magnitude)
    """
    # Net heat flux into ocean [W/m²]
    heat_flux = (surface_fluxes.shortwave_net[:, 0] + 
                surface_fluxes.longwave_net[:, 0] - 
                surface_fluxes.sensible_heat[:, 0] - 
                surface_fluxes.latent_heat[:, 0])
    
    # Freshwater flux [kg/m²/s] (positive into ocean)
    freshwater_flux = precipitation_rate - surface_fluxes.evaporation[:, 0]
    
    # Momentum flux magnitude [N/m²]
    momentum_flux_magnitude = jnp.sqrt(
        surface_fluxes.momentum_u[:, 0]**2 + surface_fluxes.momentum_v[:, 0]**2
    )
    
    return heat_flux, freshwater_flux, momentum_flux_magnitude