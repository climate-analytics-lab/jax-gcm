"""
Land surface physics for ICON surface scheme.

This module implements simplified land surface processes including
soil thermodynamics, vegetation effects, and land-atmosphere coupling.
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
def compute_land_albedo(
    vegetation_fraction: jnp.ndarray,
    soil_wetness: jnp.ndarray,
    snow_depth: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute land surface albedo based on vegetation, soil, and snow.
    
    Args:
        vegetation_fraction: Vegetation fraction [-] (ncol,)
        soil_wetness: Soil wetness [-] (ncol,)
        snow_depth: Snow depth [m] (ncol,)
        params: Surface parameters
        
    Returns:
        Tuple of (albedo_vis_direct, albedo_vis_diffuse, 
                 albedo_nir_direct, albedo_nir_diffuse)
    """
    # Vegetation albedo
    veg_albedo_vis = 0.05  # Green vegetation visible
    veg_albedo_nir = 0.45  # Green vegetation NIR
    
    # Soil albedo (depends on wetness)
    soil_albedo_dry_vis = 0.30   # Dry soil visible
    soil_albedo_wet_vis = 0.15   # Wet soil visible
    soil_albedo_dry_nir = 0.40   # Dry soil NIR
    soil_albedo_wet_nir = 0.25   # Wet soil NIR
    
    # Interpolate soil albedo based on wetness
    soil_albedo_vis = soil_albedo_dry_vis * (1.0 - soil_wetness) + soil_albedo_wet_vis * soil_wetness
    soil_albedo_nir = soil_albedo_dry_nir * (1.0 - soil_wetness) + soil_albedo_wet_nir * soil_wetness
    
    # Combine vegetation and soil
    surface_albedo_vis = veg_albedo_vis * vegetation_fraction + soil_albedo_vis * (1.0 - vegetation_fraction)
    surface_albedo_nir = veg_albedo_nir * vegetation_fraction + soil_albedo_nir * (1.0 - vegetation_fraction)
    
    # Snow albedo (high)
    snow_albedo_vis = 0.85
    snow_albedo_nir = 0.75
    
    # Snow masking (simplified)
    snow_mask_depth = 0.01  # m
    snow_factor = jnp.minimum(snow_depth / snow_mask_depth, 1.0)
    
    # Final albedo with snow
    albedo_vis = surface_albedo_vis * (1.0 - snow_factor) + snow_albedo_vis * snow_factor
    albedo_nir = surface_albedo_nir * (1.0 - snow_factor) + snow_albedo_nir * snow_factor
    
    # Assume same for direct and diffuse
    albedo_vis_direct = albedo_vis
    albedo_vis_diffuse = albedo_vis
    albedo_nir_direct = albedo_nir
    albedo_nir_diffuse = albedo_nir
    
    return albedo_vis_direct, albedo_vis_diffuse, albedo_nir_direct, albedo_nir_diffuse


@jax.jit
def compute_land_roughness(
    vegetation_fraction: jnp.ndarray,
    snow_depth: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Compute land surface roughness.
    
    Args:
        vegetation_fraction: Vegetation fraction [-] (ncol,)
        snow_depth: Snow depth [m] (ncol,)
        params: Surface parameters
        
    Returns:
        Land roughness length [m] (ncol,)
    """
    # Vegetation roughness
    z0_vegetation = 0.5  # m (forest)
    z0_bare_soil = 0.01  # m (bare soil)
    
    # Combine based on vegetation fraction
    z0_surface = z0_vegetation * vegetation_fraction + z0_bare_soil * (1.0 - vegetation_fraction)
    
    # Snow effect (reduces roughness)
    z0_snow = 1e-3  # m
    snow_mask_depth = 0.05  # m
    snow_factor = jnp.minimum(snow_depth / snow_mask_depth, 1.0)
    
    # Final roughness
    z0_land = z0_surface * (1.0 - snow_factor) + z0_snow * snow_factor
    
    return z0_land


@jax.jit
def soil_heat_conduction(
    soil_temp: jnp.ndarray,
    surface_temp: jnp.ndarray,
    soil_depths: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Compute heat conduction in soil layers.
    
    Args:
        soil_temp: Soil temperature [K] (ncol, nsoil_layers)
        surface_temp: Surface temperature [K] (ncol,)
        soil_depths: Soil layer depths [m] (nsoil_layers,)
        params: Surface parameters
        
    Returns:
        Heat conduction flux [W/m²] (ncol,)
    """
    # Soil thermal properties
    thermal_conductivity = 1.5  # W/m/K (typical soil)
    
    # Temperature gradient at surface
    # Simple finite difference between surface and first soil layer
    temp_gradient = (surface_temp - soil_temp[:, 0]) / (soil_depths[0] / 2)
    
    # Heat flux (positive into soil)
    heat_flux = thermal_conductivity * temp_gradient
    
    return heat_flux


@jax.jit
def soil_temperature_step(
    soil_temp: jnp.ndarray,
    surface_heat_flux: jnp.ndarray,
    soil_depths: jnp.ndarray,
    soil_moisture: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Update soil temperature using heat diffusion equation.
    
    Args:
        soil_temp: Soil temperature [K] (ncol, nsoil_layers)
        surface_heat_flux: Surface heat flux [W/m²] (ncol,)
        soil_depths: Soil layer depths [m] (nsoil_layers,)
        soil_moisture: Soil moisture [-] (ncol, nsoil_layers)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Soil temperature tendency [K/s] (ncol, nsoil_layers)
    """
    ncol, nsoil_layers = soil_temp.shape
    
    # Soil thermal properties
    rho_soil = 1500.0  # kg/m³
    cp_soil_dry = 800.0  # J/kg/K
    cp_water = 4200.0  # J/kg/K
    
    # Heat capacity depends on soil moisture
    cp_soil = cp_soil_dry + soil_moisture * (cp_water - cp_soil_dry)
    heat_capacity = rho_soil * cp_soil * soil_depths[None, :]
    
    # Initialize tendency
    temp_tendency = jnp.zeros_like(soil_temp)
    
    # Top layer: affected by surface heat flux
    temp_tendency = temp_tendency.at[:, 0].set(
        surface_heat_flux / heat_capacity[:, 0]
    )
    
    # Simple heat diffusion between layers
    thermal_diffusivity = 1e-6  # m²/s (typical soil)
    
    for i in range(1, nsoil_layers):
        # Heat exchange with layer above
        layer_spacing = 0.5 * (soil_depths[i] + soil_depths[i-1])
        heat_exchange = (thermal_diffusivity * 
                        (soil_temp[:, i-1] - soil_temp[:, i]) / layer_spacing)
        
        temp_tendency = temp_tendency.at[:, i].add(
            heat_exchange / heat_capacity[:, i]
        )
        temp_tendency = temp_tendency.at[:, i-1].add(
            -heat_exchange / heat_capacity[:, i-1]
        )
    
    return temp_tendency


@jax.jit
def compute_transpiration(
    vegetation_fraction: jnp.ndarray,
    soil_moisture: jnp.ndarray,
    surface_temp: jnp.ndarray,
    atmospheric_humidity: jnp.ndarray,
    net_radiation: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    air_density: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Compute transpiration from vegetation using simplified Penman-Monteith.
    
    Args:
        vegetation_fraction: Vegetation fraction [-] (ncol,)
        soil_moisture: Soil moisture [-] (ncol, nsoil_layers)
        surface_temp: Surface temperature [K] (ncol,)
        atmospheric_humidity: Atmospheric humidity [kg/kg] (ncol,)
        net_radiation: Net radiation [W/m²] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        air_density: Air density [kg/m³] (ncol,)
        params: Surface parameters
        
    Returns:
        Transpiration rate [kg/m²/s] (ncol,)
    """
    # Only transpire where there's vegetation
    veg_mask = vegetation_fraction > 0.01
    
    # Soil water availability (use root zone moisture)
    root_zone_moisture = soil_moisture[:, 0]  # Top layer
    water_stress = jnp.minimum(root_zone_moisture / 0.3, 1.0)  # Wilting point at 30%
    
    # Canopy resistance (simplified)
    canopy_resistance_min = 100.0  # s/m
    canopy_resistance = canopy_resistance_min / water_stress
    
    # Aerodynamic resistance
    aerodynamic_resistance = 1.0 / exchange_coeff_moisture
    
    # Saturation vapor pressure at surface
    e_sat = 611.0 * jnp.exp(17.27 * (surface_temp - PHYS_CONST.t0) / 
                           (surface_temp - PHYS_CONST.t0 + 237.3))
    
    # Simplified Penman-Monteith equation
    # Transpiration = (Delta * Rn + rho * cp * VPD / ra) / (Delta + gamma * (1 + rc/ra))
    # where Delta = slope of saturation vapor pressure curve, gamma = psychrometric constant
    
    # Simplified: transpiration proportional to net radiation and water availability
    potential_transpiration = jnp.maximum(net_radiation * 0.5, 0.0) / PHYS_CONST.alhc
    
    # Apply water stress and vegetation fraction
    transpiration = (potential_transpiration * water_stress * vegetation_fraction *
                    veg_mask.astype(jnp.float32))
    
    return transpiration


@jax.jit
def soil_moisture_step(
    soil_moisture: jnp.ndarray,
    precipitation_rain: jnp.ndarray,
    evaporation: jnp.ndarray,
    transpiration: jnp.ndarray,
    soil_depths: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> jnp.ndarray:
    """
    Update soil moisture including precipitation, evaporation, and transpiration.
    
    Args:
        soil_moisture: Soil moisture [-] (ncol, nsoil_layers)
        precipitation_rain: Rain rate [kg/m²/s] (ncol,)
        evaporation: Evaporation rate [kg/m²/s] (ncol,)
        transpiration: Transpiration rate [kg/m²/s] (ncol,)
        soil_depths: Soil layer depths [m] (nsoil_layers,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Soil moisture tendency [1/s] (ncol, nsoil_layers)
    """
    ncol, nsoil_layers = soil_moisture.shape
    
    # Soil properties
    rho_water = 1000.0  # kg/m³
    porosity = 0.45  # Typical soil porosity
    
    # Water capacity of each layer
    water_capacity = rho_water * porosity * soil_depths[None, :]
    
    # Initialize tendency
    moisture_tendency = jnp.zeros_like(soil_moisture)
    
    # Top layer: affected by precipitation, evaporation, and transpiration
    water_input = precipitation_rain
    water_output = evaporation + transpiration
    net_water_flux = water_input - water_output
    
    moisture_tendency = moisture_tendency.at[:, 0].set(
        net_water_flux / water_capacity[:, 0]
    )
    
    # Simple infiltration between layers
    infiltration_rate = 1e-6  # m/s (hydraulic conductivity)
    
    for i in range(1, nsoil_layers):
        # Water movement from layer above (if saturated)
        excess_water = jnp.maximum(soil_moisture[:, i-1] - 1.0, 0.0)
        infiltration = infiltration_rate * excess_water
        
        moisture_tendency = moisture_tendency.at[:, i].add(
            infiltration / water_capacity[:, i]
        )
        moisture_tendency = moisture_tendency.at[:, i-1].add(
            -infiltration / water_capacity[:, i-1]
        )
    
    # Ensure moisture stays within bounds [0, 1]
    moisture_tendency = jnp.where(
        (soil_moisture + moisture_tendency * dt) < 0.0,
        -soil_moisture / dt,
        moisture_tendency
    )
    moisture_tendency = jnp.where(
        (soil_moisture + moisture_tendency * dt) > 1.0,
        (1.0 - soil_moisture) / dt,
        moisture_tendency
    )
    
    return moisture_tendency


@jax.jit
def land_surface_physics_step(
    atmospheric_state: AtmosphericForcing,
    soil_temp: jnp.ndarray,
    soil_moisture: jnp.ndarray,
    vegetation_temp: jnp.ndarray,
    snow_depth: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray,
    vegetation_fraction: jnp.ndarray,
    soil_depths: jnp.ndarray,
    dt: float,
    params: SurfaceParameters = SurfaceParameters.default()
) -> Tuple[SurfaceFluxes, SurfaceTendencies, jnp.ndarray]:
    """
    Complete land surface physics step.
    
    Args:
        atmospheric_state: Atmospheric forcing
        soil_temp: Soil temperature [K] (ncol, nsoil_layers)
        soil_moisture: Soil moisture [-] (ncol, nsoil_layers)
        vegetation_temp: Vegetation temperature [K] (ncol,)
        snow_depth: Snow depth [m] (ncol,)
        exchange_coeff_heat: Heat exchange coefficient [m/s] (ncol,)
        exchange_coeff_moisture: Moisture exchange coefficient [m/s] (ncol,)
        exchange_coeff_momentum: Momentum exchange coefficient [m/s] (ncol,)
        vegetation_fraction: Vegetation fraction [-] (ncol,)
        soil_depths: Soil layer depths [m] (nsoil_layers,)
        dt: Time step [s]
        params: Surface parameters
        
    Returns:
        Tuple of (surface_fluxes, tendencies, roughness_length)
    """
    ncol = soil_temp.shape[0]
    nsoil_layers = soil_temp.shape[1]
    
    # Surface temperature (top soil layer)
    surface_temp = soil_temp[:, 0]
    
    # Air density
    air_density = (atmospheric_state.pressure / 
                  (PHYS_CONST.rd * atmospheric_state.temperature))
    
    # Surface humidity (soil + vegetation)
    # Soil evaporation depends on soil moisture
    soil_beta = jnp.minimum(soil_moisture[:, 0] / 0.75, 1.0)  # Evaporation efficiency
    
    # Simplified surface humidity
    e_sat = 611.0 * jnp.exp(17.27 * (surface_temp - PHYS_CONST.t0) / 
                           (surface_temp - PHYS_CONST.t0 + 237.3))
    q_sat_surface = PHYS_CONST.eps * e_sat / atmospheric_state.pressure
    q_surface = soil_beta * q_sat_surface  # Reduced by soil dryness
    
    # Temperature and humidity differences
    delta_temp = atmospheric_state.temperature - surface_temp
    delta_humidity = atmospheric_state.humidity - q_surface
    
    # Turbulent fluxes
    sensible_heat = air_density * PHYS_CONST.cp * exchange_coeff_heat * delta_temp
    
    # Evaporation from soil
    evaporation = air_density * exchange_coeff_moisture * delta_humidity
    evaporation = jnp.maximum(evaporation, 0.0)  # No condensation on land
    
    # Transpiration from vegetation
    net_radiation = atmospheric_state.sw_downward + atmospheric_state.lw_downward
    transpiration = compute_transpiration(
        vegetation_fraction, soil_moisture, surface_temp, 
        atmospheric_state.humidity, net_radiation, exchange_coeff_moisture,
        air_density, params
    )
    
    # Total latent heat flux
    latent_heat = PHYS_CONST.alhc * (evaporation + transpiration)
    
    # Momentum fluxes
    momentum_u = air_density * exchange_coeff_momentum * atmospheric_state.u_wind
    momentum_v = air_density * exchange_coeff_momentum * atmospheric_state.v_wind
    
    # Land albedo
    albedo_vis_dir, albedo_vis_dif, albedo_nir_dir, albedo_nir_dif = compute_land_albedo(
        vegetation_fraction, soil_moisture[:, 0], snow_depth, params
    )
    
    # Net shortwave
    albedo_mean = 0.25 * (albedo_vis_dir + albedo_vis_dif + albedo_nir_dir + albedo_nir_dif)
    shortwave_net = atmospheric_state.sw_downward * (1.0 - albedo_mean)
    
    # Net longwave
    lw_upward = params.emissivity * params.stefan_boltzmann * surface_temp**4
    longwave_net = atmospheric_state.lw_downward - lw_upward
    
    # Ground heat flux
    ground_heat = soil_heat_conduction(
        soil_temp, surface_temp, soil_depths, params
    )
    
    # Land roughness
    roughness = compute_land_roughness(vegetation_fraction, snow_depth, params)
    
    # Create single-tile flux structure (land only)
    fluxes = SurfaceFluxes(
        sensible_heat=sensible_heat[:, None],
        latent_heat=latent_heat[:, None],
        longwave_net=longwave_net[:, None],
        shortwave_net=shortwave_net[:, None],
        ground_heat=ground_heat[:, None],
        momentum_u=momentum_u[:, None],
        momentum_v=momentum_v[:, None],
        evaporation=evaporation[:, None],
        transpiration=transpiration[:, None],
        sensible_heat_mean=sensible_heat,
        latent_heat_mean=latent_heat,
        momentum_u_mean=momentum_u,
        momentum_v_mean=momentum_v,
        evaporation_mean=evaporation + transpiration
    )
    
    # Compute tendencies
    soil_temp_tendency = soil_temperature_step(
        soil_temp, ground_heat, soil_depths, soil_moisture, dt, params
    )
    
    soil_moisture_tendency = soil_moisture_step(
        soil_moisture, atmospheric_state.rain_rate, evaporation, 
        transpiration, soil_depths, dt, params
    )
    
    # Create tendencies structure
    tendencies = SurfaceTendencies(
        surface_temp_tendency=soil_temp_tendency[:, 0:1],  # Top layer
        ocean_temp_tendency=jnp.zeros(ncol),
        ice_temp_tendency=jnp.zeros((ncol, 1)),
        soil_temp_tendency=soil_temp_tendency,
        ice_thickness_tendency=jnp.zeros((ncol, 1)),
        snow_depth_tendency=jnp.zeros(ncol),  # Simplified
        soil_moisture_tendency=soil_moisture_tendency
    )
    
    return fluxes, tendencies, roughness