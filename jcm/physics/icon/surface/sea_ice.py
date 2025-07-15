"""
Sea ice thermodynamics for ICON surface scheme.

This module implements sea ice thermodynamics following ICON's approach,
including ice growth/melt, thermal conduction, and snow processes.
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
def compute_ice_albedo(
    ice_thickness: jnp.ndarray,
    snow_depth: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute sea ice albedo as a function of ice thickness and snow cover.
    
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
    """
    Compute sea ice surface roughness.
    
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
def ice_heat_conduction(
    ice_temp: jnp.ndarray,
    ice_thickness: jnp.ndarray,
    surface_temp: jnp.ndarray,
    ocean_temp: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Compute heat conduction through sea ice.
    
    Args:
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        ice_thickness: Ice thickness [m] (ncol, nice_layers)
        surface_temp: Surface temperature [K] (ncol,)
        ocean_temp: Ocean temperature [K] (ncol,)
        params: Surface parameters
        
    Returns:
        Heat conduction flux [W/m²] (ncol,)
    """
    # Simple 1-layer ice model
    # Thermal conductivity through ice
    total_thickness = jnp.sum(ice_thickness, axis=1)
    
    # Prevent division by zero
    total_thickness = jnp.maximum(total_thickness, 0.01)
    
    # Linear temperature profile assumption
    # Heat flux = k * (T_bottom - T_top) / thickness
    heat_flux = (params.conduct_ice * (ocean_temp - surface_temp) / 
                 total_thickness)
    
    return heat_flux


@jax.jit
def ice_surface_temperature_step(
    ice_temp: jnp.ndarray,
    surface_fluxes: SurfaceFluxes,
    ice_thickness: jnp.ndarray,
    snow_depth: jnp.ndarray,
    ocean_temp: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update ice surface temperature and check for melting.
    
    Args:
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        surface_fluxes: Surface fluxes
        ice_thickness: Ice thickness [m] (ncol, nice_layers)
        snow_depth: Snow depth [m] (ncol,)
        ocean_temp: Ocean temperature [K] (ncol,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Tuple of (ice_temp_tendency, surface_temp_tendency)
    """
    ncol = ice_temp.shape[0]
    nice_layers = ice_temp.shape[1]
    
    # Surface temperature (top of ice/snow)
    surface_temp = ice_temp[:, 0]  # Top layer
    
    # Net surface energy flux
    net_flux = (surface_fluxes.shortwave_net[:, params.iice] + 
                surface_fluxes.longwave_net[:, params.iice] - 
                surface_fluxes.sensible_heat[:, params.iice] - 
                surface_fluxes.latent_heat[:, params.iice])
    
    # Heat conduction from ocean
    conduction_flux = ice_heat_conduction(
        ice_temp, ice_thickness, surface_temp, ocean_temp, params
    )
    
    # Total heat input to ice
    total_heat = net_flux + conduction_flux
    
    # Heat capacity of ice layer
    layer_thickness = ice_thickness / nice_layers
    heat_capacity = params.rho_ice * params.cp_ice * layer_thickness
    
    # Temperature tendency
    temp_tendency = total_heat[:, None] / heat_capacity
    
    # Check for melting (temperature cannot exceed 0°C)
    melting_mask = (ice_temp + temp_tendency * dt) > PHYS_CONST.tmelt
    temp_tendency = jnp.where(
        melting_mask,
        (PHYS_CONST.tmelt - ice_temp) / dt,
        temp_tendency
    )
    
    # Surface temperature tendency (same as top layer)
    surface_temp_tendency = temp_tendency[:, 0]
    
    return temp_tendency, surface_temp_tendency


@jax.jit
def ice_thickness_evolution(
    ice_thickness: jnp.ndarray,
    ice_temp: jnp.ndarray,
    surface_fluxes: SurfaceFluxes,
    ocean_temp: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Compute ice thickness evolution due to freezing/melting.
    
    Args:
        ice_thickness: Ice thickness [m] (ncol, nice_layers)
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        surface_fluxes: Surface fluxes
        ocean_temp: Ocean temperature [K] (ncol,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Ice thickness tendency [m/s] (ncol, nice_layers)
    """
    ncol = ice_thickness.shape[0]
    nice_layers = ice_thickness.shape[1]
    
    # Surface melting
    surface_temp = ice_temp[:, 0]
    surface_melt_mask = surface_temp >= PHYS_CONST.tmelt
    
    # Net surface energy (positive = melting)
    net_surface_energy = (surface_fluxes.shortwave_net[:, params.iice] + 
                         surface_fluxes.longwave_net[:, params.iice] - 
                         surface_fluxes.sensible_heat[:, params.iice] - 
                         surface_fluxes.latent_heat[:, params.iice])
    
    # Surface melt rate [m/s]
    latent_heat_fusion = 334000.0  # J/kg
    surface_melt_rate = jnp.where(
        surface_melt_mask & (net_surface_energy > 0),
        net_surface_energy / (params.rho_ice * latent_heat_fusion),
        0.0
    )
    
    # Bottom melting/freezing
    bottom_temp = ice_temp[:, -1]  # Bottom layer
    bottom_melt_mask = bottom_temp >= PHYS_CONST.tmelt
    
    # Heat conduction from ocean
    conduction_flux = ice_heat_conduction(
        ice_temp, ice_thickness, surface_temp, ocean_temp, params
    )
    
    # Bottom growth/melt rate [m/s]
    bottom_rate = jnp.where(
        bottom_melt_mask & (conduction_flux > 0),
        conduction_flux / (params.rho_ice * latent_heat_fusion),  # Melting
        jnp.where(
            ocean_temp < PHYS_CONST.tmelt,
            -jnp.abs(conduction_flux) / (params.rho_ice * latent_heat_fusion),  # Freezing
            0.0
        )
    )
    
    # Distribute thickness changes
    thickness_tendency = jnp.zeros((ncol, nice_layers))
    
    # Surface melting affects top layer
    thickness_tendency = thickness_tendency.at[:, 0].add(-surface_melt_rate)
    
    # Bottom changes affect bottom layer
    thickness_tendency = thickness_tendency.at[:, -1].add(bottom_rate)
    
    # Ensure thickness stays positive
    total_thickness = jnp.sum(ice_thickness, axis=1)
    total_tendency = jnp.sum(thickness_tendency, axis=1)
    
    # Limit melting to available ice
    # Expand condition to match thickness array shape [ncol, nice_layers]
    melting_condition = (total_thickness + total_tendency * dt) < 0.0
    melting_condition_expanded = melting_condition[:, jnp.newaxis]  # Shape [ncol, 1] for broadcasting
    
    thickness_tendency = jnp.where(
        melting_condition_expanded,
        -ice_thickness / dt,
        thickness_tendency
    )
    
    return thickness_tendency


@jax.jit
def snow_evolution(
    snow_depth: jnp.ndarray,
    surface_fluxes: SurfaceFluxes,
    precipitation_snow: jnp.ndarray,
    ice_temp: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Compute snow depth evolution on sea ice.
    
    Args:
        snow_depth: Snow depth [m] (ncol,)
        surface_fluxes: Surface fluxes
        precipitation_snow: Snow precipitation [kg/m²/s] (ncol,)
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Snow depth tendency [m/s] (ncol,)
    """
    # Snow accumulation from precipitation
    rho_snow = 300.0  # kg/m³ (fresh snow density)
    snow_accum = precipitation_snow / rho_snow
    
    # Snow melting
    surface_temp = ice_temp[:, 0]
    snow_melt_mask = surface_temp >= PHYS_CONST.tmelt
    
    # Net surface energy for melting
    net_surface_energy = (surface_fluxes.shortwave_net[:, params.iice] + 
                         surface_fluxes.longwave_net[:, params.iice] - 
                         surface_fluxes.sensible_heat[:, params.iice] - 
                         surface_fluxes.latent_heat[:, params.iice])
    
    # Snow melt rate [m/s]
    latent_heat_fusion = 334000.0  # J/kg
    snow_melt_rate = jnp.where(
        snow_melt_mask & (net_surface_energy > 0) & (snow_depth > 0),
        net_surface_energy / (rho_snow * latent_heat_fusion),
        0.0
    )
    
    # Total snow tendency
    snow_tendency = snow_accum - snow_melt_rate
    
    # Ensure snow depth stays positive
    snow_tendency = jnp.where(
        (snow_depth + snow_tendency * dt) < 0.0,
        -snow_depth / dt,
        snow_tendency
    )
    
    return snow_tendency


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
    """
    Complete sea ice physics step.
    
    Args:
        atmospheric_state: Atmospheric forcing
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        ice_thickness: Ice thickness [m] (ncol, nice_layers)
        snow_depth: Snow depth [m] (ncol,)
        ocean_temp: Ocean temperature [K] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Tuple of (surface_fluxes, tendencies, roughness_length)
    """
    ncol = ice_temp.shape[0]
    nice_layers = ice_temp.shape[1]
    
    # Surface temperature (top of ice/snow)
    surface_temp = ice_temp[:, 0]
    
    # Air density
    air_density = (atmospheric_state.pressure / 
                  (PHYS_CONST.rd * atmospheric_state.temperature))
    
    # Surface saturation humidity
    e_sat = 611.0 * jnp.exp(17.27 * (surface_temp - PHYS_CONST.t0) / 
                           (surface_temp - PHYS_CONST.t0 + 237.3))
    q_sat_surface = PHYS_CONST.eps * e_sat / atmospheric_state.pressure
    
    # Temperature and humidity differences
    delta_temp = atmospheric_state.temperature - surface_temp
    delta_humidity = atmospheric_state.humidity - q_sat_surface
    
    # Turbulent fluxes
    sensible_heat = air_density * PHYS_CONST.cp * exchange_coeff_heat * delta_temp
    latent_heat = air_density * PHYS_CONST.alhs * exchange_coeff_moisture * delta_humidity  # Sublimation
    
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
    
    # Compute tendencies
    ice_temp_tendency, surface_temp_tendency = ice_surface_temperature_step(
        ice_temp, fluxes, ice_thickness, snow_depth, ocean_temp, dt, params
    )
    
    thickness_tendency = ice_thickness_evolution(
        ice_thickness, ice_temp, fluxes, ocean_temp, dt, params
    )
    
    snow_tendency = snow_evolution(
        snow_depth, fluxes, atmospheric_state.snow_rate, ice_temp, dt, params
    )
    
    # Create tendencies structure
    tendencies = SurfaceTendencies(
        surface_temp_tendency=surface_temp_tendency[:, None],
        ocean_temp_tendency=jnp.zeros(ncol),
        ice_temp_tendency=ice_temp_tendency,
        soil_temp_tendency=jnp.zeros((ncol, 1)),
        ice_thickness_tendency=thickness_tendency,
        snow_depth_tendency=snow_tendency,
        soil_moisture_tendency=jnp.zeros((ncol, 1))
    )
    
    return fluxes, tendencies, roughness