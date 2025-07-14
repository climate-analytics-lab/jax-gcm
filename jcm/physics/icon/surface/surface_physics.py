"""
Main surface physics interface for ICON surface scheme.

This module provides the main interface for surface physics calculations,
coordinating different surface types (ocean, sea ice, land) and computing
grid-box mean fluxes.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing, 
    SurfaceFluxes, SurfaceTendencies, SurfaceResistances, SurfaceDiagnostics
)
from .turbulent_fluxes import (
    compute_bulk_richardson_number, compute_stability_functions,
    compute_exchange_coefficients, compute_turbulent_fluxes,
    compute_surface_resistances, compute_surface_diagnostics
)
from .ocean import ocean_physics_step
from .sea_ice import sea_ice_physics_step
from .land import land_surface_physics_step

# Create constants instance
PHYS_CONST = PhysicalConstants()


def initialize_surface_state(
    ncol: int,
    surface_fractions: jnp.ndarray,
    ocean_temp: jnp.ndarray,
    ice_temp: jnp.ndarray,
    soil_temp: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> SurfaceState:
    """
    Initialize surface state from basic inputs.
    
    Args:
        ncol: Number of columns
        surface_fractions: Surface type fractions [water, ice, land] (ncol, 3)
        ocean_temp: Ocean temperature [K] (ncol,)
        ice_temp: Ice temperature [K] (ncol, nice_layers)
        soil_temp: Soil temperature [K] (ncol, nsoil_layers)
        params: Surface parameters
        
    Returns:
        Initialized surface state
    """
    # Use fixed value for nsfc_type since it needs to be concrete for array creation
    nsfc_type = 3  # Always 3: water, ice, land
    nice_layers = 2  # Default ice layers
    nsoil_layers = 4  # Default soil layers
    
    # Surface temperatures (combine different surface types)
    surface_temp = jnp.zeros((ncol, nsfc_type))
    surface_temp = surface_temp.at[:, params.iwtr].set(ocean_temp)
    surface_temp = surface_temp.at[:, params.iice].set(ice_temp[:, 0])  # Top ice layer
    surface_temp = surface_temp.at[:, params.ilnd].set(soil_temp[:, 0])  # Top soil layer
    
    # Radiative temperature (area-weighted mean)
    temp_rad = jnp.sum(surface_fractions * surface_temp, axis=1)
    
    # Ocean variables
    ocean_u = jnp.zeros(ncol)
    ocean_v = jnp.zeros(ncol)
    
    # Ice variables
    ice_thickness = jnp.full((ncol, nice_layers), 2.0)  # 2m default thickness
    snow_depth = jnp.zeros(ncol)
    
    # Soil variables
    soil_moisture = jnp.full((ncol, nsoil_layers), 0.3)  # 30% default moisture
    vegetation_temp = soil_temp[:, 0]  # Same as surface
    
    # Default surface properties
    roughness_momentum = jnp.zeros((ncol, nsfc_type))
    roughness_momentum = roughness_momentum.at[:, params.iwtr].set(params.z0_water)
    roughness_momentum = roughness_momentum.at[:, params.iice].set(params.z0_ice)
    roughness_momentum = roughness_momentum.at[:, params.ilnd].set(params.z0_land)
    
    roughness_heat = roughness_momentum * 0.1  # Typical ratio
    
    # Default albedos
    albedo_vis_direct = jnp.zeros((ncol, nsfc_type))
    albedo_vis_diffuse = jnp.zeros((ncol, nsfc_type))
    albedo_nir_direct = jnp.zeros((ncol, nsfc_type))
    albedo_nir_diffuse = jnp.zeros((ncol, nsfc_type))
    
    # Ocean albedo
    albedo_vis_direct = albedo_vis_direct.at[:, params.iwtr].set(0.06)
    albedo_vis_diffuse = albedo_vis_diffuse.at[:, params.iwtr].set(0.06)
    albedo_nir_direct = albedo_nir_direct.at[:, params.iwtr].set(0.06)
    albedo_nir_diffuse = albedo_nir_diffuse.at[:, params.iwtr].set(0.06)
    
    # Ice albedo
    albedo_vis_direct = albedo_vis_direct.at[:, params.iice].set(0.75)
    albedo_vis_diffuse = albedo_vis_diffuse.at[:, params.iice].set(0.75)
    albedo_nir_direct = albedo_nir_direct.at[:, params.iice].set(0.65)
    albedo_nir_diffuse = albedo_nir_diffuse.at[:, params.iice].set(0.65)
    
    # Land albedo
    albedo_vis_direct = albedo_vis_direct.at[:, params.ilnd].set(0.15)
    albedo_vis_diffuse = albedo_vis_diffuse.at[:, params.ilnd].set(0.15)
    albedo_nir_direct = albedo_nir_direct.at[:, params.ilnd].set(0.30)
    albedo_nir_diffuse = albedo_nir_diffuse.at[:, params.ilnd].set(0.30)
    
    return SurfaceState(
        temperature=surface_temp,
        temperature_rad=temp_rad,
        fraction=surface_fractions,
        ocean_temp=ocean_temp,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        ice_thickness=ice_thickness,
        ice_temp=ice_temp,
        snow_depth=snow_depth,
        soil_temp=soil_temp,
        soil_moisture=soil_moisture,
        vegetation_temp=vegetation_temp,
        roughness_momentum=roughness_momentum,
        roughness_heat=roughness_heat,
        albedo_visible_direct=albedo_vis_direct,
        albedo_visible_diffuse=albedo_vis_diffuse,
        albedo_nir_direct=albedo_nir_direct,
        albedo_nir_diffuse=albedo_nir_diffuse
    )


@jax.jit
def surface_physics_step(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[SurfaceFluxes, SurfaceTendencies, SurfaceDiagnostics]:
    """
    Complete surface physics step for all surface types.
    
    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Tuple of (surface_fluxes, tendencies, diagnostics)
    """
    ncol, nsfc_type = surface_state.temperature.shape
    
    # Compute bulk Richardson number
    surface_humidity = jnp.full_like(surface_state.temperature, 0.01)  # Simplified
    wind_speed = jnp.sqrt(atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2)
    
    ri_bulk = compute_bulk_richardson_number(
        atmospheric_state.temperature, surface_state.temperature,
        atmospheric_state.humidity, surface_humidity, wind_speed
    )
    
    # Compute stability functions
    stability_heat, stability_momentum = compute_stability_functions(ri_bulk)
    
    # Compute exchange coefficients
    exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture = compute_exchange_coefficients(
        wind_speed, surface_state.roughness_momentum, surface_state.roughness_heat,
        stability_heat, stability_momentum, params.min_wind_speed, params.von_karman
    )
    
    # Initialize output arrays
    all_fluxes = []
    all_tendencies = []
    all_roughness = []
    
    # Process each surface type
    for isfc in range(nsfc_type):
        surface_mask = surface_state.fraction[:, isfc] > 0.01
        
        if isfc == params.iwtr:
            # Ocean physics
            fluxes_isfc, tendencies_isfc, roughness_isfc = ocean_physics_step(
                atmospheric_state,
                surface_state.ocean_temp,
                surface_state.ocean_u,
                surface_state.ocean_v,
                exchange_coeff_heat[:, isfc],
                exchange_coeff_moisture[:, isfc],
                exchange_coeff_momentum[:, isfc],
                jnp.zeros(ncol),  # Solar zenith angle (simplified)
                dt, params
            )
            
        elif isfc == params.iice:
            # Sea ice physics
            fluxes_isfc, tendencies_isfc, roughness_isfc = sea_ice_physics_step(
                atmospheric_state,
                surface_state.ice_temp,
                surface_state.ice_thickness,
                surface_state.snow_depth,
                surface_state.ocean_temp,
                exchange_coeff_heat[:, isfc],
                exchange_coeff_moisture[:, isfc],
                exchange_coeff_momentum[:, isfc],
                dt, params
            )
            
        elif isfc == params.ilnd:
            # Land physics
            vegetation_fraction = jnp.full(ncol, 0.5)  # 50% vegetation
            soil_depths = jnp.array([0.1, 0.3, 0.6, 1.0])  # Soil layer depths
            
            fluxes_isfc, tendencies_isfc, roughness_isfc = land_surface_physics_step(
                atmospheric_state,
                surface_state.soil_temp,
                surface_state.soil_moisture,
                surface_state.vegetation_temp,
                surface_state.snow_depth,
                exchange_coeff_heat[:, isfc],
                exchange_coeff_moisture[:, isfc],
                exchange_coeff_momentum[:, isfc],
                vegetation_fraction,
                soil_depths,
                dt, params
            )
            
        all_fluxes.append(fluxes_isfc)
        all_tendencies.append(tendencies_isfc)
        all_roughness.append(roughness_isfc)
    
    # Combine fluxes from all surface types
    combined_fluxes = combine_surface_fluxes(all_fluxes, surface_state.fraction)
    
    # Combine tendencies
    combined_tendencies = combine_surface_tendencies(all_tendencies, surface_state.fraction)
    
    # Compute surface resistances
    resistances = compute_surface_resistances(
        atmospheric_state, surface_state, ri_bulk, params
    )
    
    # Compute diagnostics
    diagnostics = compute_surface_diagnostics(
        atmospheric_state, surface_state, combined_fluxes, resistances, params
    )
    
    return combined_fluxes, combined_tendencies, diagnostics


@jax.jit
def combine_surface_fluxes(
    flux_list: list,
    fractions: jnp.ndarray
) -> SurfaceFluxes:
    """
    Combine fluxes from different surface types into grid-box means.
    
    Args:
        flux_list: List of SurfaceFluxes for each surface type
        fractions: Surface type fractions (ncol, nsfc_type)
        
    Returns:
        Combined surface fluxes
    """
    ncol, nsfc_type = fractions.shape
    
    # Stack fluxes from all surface types
    sensible_heat = jnp.stack([flux.sensible_heat[:, 0] for flux in flux_list], axis=1)
    latent_heat = jnp.stack([flux.latent_heat[:, 0] for flux in flux_list], axis=1)
    longwave_net = jnp.stack([flux.longwave_net[:, 0] for flux in flux_list], axis=1)
    shortwave_net = jnp.stack([flux.shortwave_net[:, 0] for flux in flux_list], axis=1)
    ground_heat = jnp.stack([flux.ground_heat[:, 0] for flux in flux_list], axis=1)
    momentum_u = jnp.stack([flux.momentum_u[:, 0] for flux in flux_list], axis=1)
    momentum_v = jnp.stack([flux.momentum_v[:, 0] for flux in flux_list], axis=1)
    evaporation = jnp.stack([flux.evaporation[:, 0] for flux in flux_list], axis=1)
    transpiration = jnp.stack([flux.transpiration[:, 0] for flux in flux_list], axis=1)
    
    # Compute grid-box means
    sensible_mean = jnp.sum(fractions * sensible_heat, axis=1)
    latent_mean = jnp.sum(fractions * latent_heat, axis=1)
    momentum_u_mean = jnp.sum(fractions * momentum_u, axis=1)
    momentum_v_mean = jnp.sum(fractions * momentum_v, axis=1)
    evaporation_mean = jnp.sum(fractions * (evaporation + transpiration), axis=1)
    
    return SurfaceFluxes(
        sensible_heat=sensible_heat,
        latent_heat=latent_heat,
        longwave_net=longwave_net,
        shortwave_net=shortwave_net,
        ground_heat=ground_heat,
        momentum_u=momentum_u,
        momentum_v=momentum_v,
        evaporation=evaporation,
        transpiration=transpiration,
        sensible_heat_mean=sensible_mean,
        latent_heat_mean=latent_mean,
        momentum_u_mean=momentum_u_mean,
        momentum_v_mean=momentum_v_mean,
        evaporation_mean=evaporation_mean
    )


@jax.jit
def combine_surface_tendencies(
    tendency_list: list,
    fractions: jnp.ndarray
) -> SurfaceTendencies:
    """
    Combine tendencies from different surface types.
    
    Args:
        tendency_list: List of SurfaceTendencies for each surface type
        fractions: Surface type fractions (ncol, nsfc_type)
        
    Returns:
        Combined surface tendencies
    """
    ncol, nsfc_type = fractions.shape
    
    # Extract tendencies (simplified - just use first valid tendency)
    ocean_temp_tendency = tendency_list[0].ocean_temp_tendency  # From ocean
    ice_temp_tendency = tendency_list[1].ice_temp_tendency if len(tendency_list) > 1 else jnp.zeros((ncol, 2))
    soil_temp_tendency = tendency_list[2].soil_temp_tendency if len(tendency_list) > 2 else jnp.zeros((ncol, 4))
    
    # Surface temperature tendency (area-weighted)
    surface_temp_tendency = jnp.stack([
        tendency.surface_temp_tendency[:, 0] for tendency in tendency_list
    ], axis=1)
    
    # Other tendencies
    ice_thickness_tendency = tendency_list[1].ice_thickness_tendency if len(tendency_list) > 1 else jnp.zeros((ncol, 2))
    snow_depth_tendency = jnp.zeros(ncol)  # Simplified
    soil_moisture_tendency = tendency_list[2].soil_moisture_tendency if len(tendency_list) > 2 else jnp.zeros((ncol, 4))
    
    return SurfaceTendencies(
        surface_temp_tendency=surface_temp_tendency,
        ocean_temp_tendency=ocean_temp_tendency,
        ice_temp_tendency=ice_temp_tendency,
        soil_temp_tendency=soil_temp_tendency,
        ice_thickness_tendency=ice_thickness_tendency,
        snow_depth_tendency=snow_depth_tendency,
        soil_moisture_tendency=soil_moisture_tendency
    )


@jax.jit
def update_surface_state(
    surface_state: SurfaceState,
    tendencies: SurfaceTendencies,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> SurfaceState:
    """
    Update surface state using computed tendencies.
    
    Args:
        surface_state: Current surface state
        tendencies: Surface tendencies
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Updated surface state
    """
    # Update temperatures
    new_surface_temp = surface_state.temperature + tendencies.surface_temp_tendency * dt
    new_ocean_temp = surface_state.ocean_temp + tendencies.ocean_temp_tendency * dt
    new_ice_temp = surface_state.ice_temp + tendencies.ice_temp_tendency * dt
    new_soil_temp = surface_state.soil_temp + tendencies.soil_temp_tendency * dt
    
    # Update other prognostic variables
    new_ice_thickness = surface_state.ice_thickness + tendencies.ice_thickness_tendency * dt
    new_snow_depth = surface_state.snow_depth + tendencies.snow_depth_tendency * dt
    new_soil_moisture = surface_state.soil_moisture + tendencies.soil_moisture_tendency * dt
    
    # Ensure physical bounds
    new_ice_thickness = jnp.maximum(new_ice_thickness, 0.0)
    new_snow_depth = jnp.maximum(new_snow_depth, 0.0)
    new_soil_moisture = jnp.clip(new_soil_moisture, 0.0, 1.0)
    
    # Update radiative temperature
    new_temp_rad = jnp.sum(surface_state.fraction * new_surface_temp, axis=1)
    
    return surface_state._replace(
        temperature=new_surface_temp,
        temperature_rad=new_temp_rad,
        ocean_temp=new_ocean_temp,
        ice_temp=new_ice_temp,
        soil_temp=new_soil_temp,
        ice_thickness=new_ice_thickness,
        snow_depth=new_snow_depth,
        soil_moisture=new_soil_moisture
    )