"""Turbulent flux calculations for surface-atmosphere exchange.

This module implements bulk aerodynamic formulations for computing
surface fluxes of momentum, heat, and moisture following ICON's approach.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from .surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing,
    SurfaceFluxes, SurfaceResistances, SurfaceDiagnostics
)


@jax.jit
def compute_bulk_richardson_number(
    temperature_air: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    humidity_air: jnp.ndarray,
    humidity_surface: jnp.ndarray,
    wind_speed: jnp.ndarray,
    reference_height: float = 10.0
) -> jnp.ndarray:
    """Compute bulk Richardson number for surface layer stability.
    
    Args:
        temperature_air: Air temperature [K] (ncol,)
        temperature_surface: Surface temperature [K] (ncol, nsfc_type)
        humidity_air: Air specific humidity [kg/kg] (ncol,)
        humidity_surface: Surface specific humidity [kg/kg] (ncol, nsfc_type)
        wind_speed: Wind speed [m/s] (ncol,)
        reference_height: Reference height [m]
        
    Returns:
        Bulk Richardson number [-] (ncol, nsfc_type)

    """
    # Virtual potential temperatures
    theta_v_air = temperature_air * (1.0 + 0.608 * humidity_air)
    theta_v_surface = temperature_surface * (1.0 + 0.608 * humidity_surface)
    
    # Mean virtual potential temperature
    theta_v_mean = 0.5 * (theta_v_air[:, None] + theta_v_surface)
    
    # Buoyancy term
    buoyancy = (c.grav * reference_height *
               (theta_v_air[:, None] - theta_v_surface) / theta_v_mean)
    
    # Wind shear term (prevent division by zero)
    wind_shear_squared = jnp.maximum(wind_speed[:, None]**2, 0.01)
    
    # Richardson number
    ri_bulk = buoyancy / wind_shear_squared
    
    return ri_bulk


@jax.jit
def compute_stability_functions(
    richardson_number: jnp.ndarray,
    stable_limit: float = 0.2,
    unstable_coeff: float = 16.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute stability functions following Businger-Dyer relationships.
    
    Args:
        richardson_number: Bulk Richardson number [-] (ncol, nsfc_type)
        stable_limit: Maximum Richardson number for stable conditions
        unstable_coeff: Coefficient for unstable conditions
        
    Returns:
        Tuple of (heat_stability_function, momentum_stability_function)

    """
    # Stable conditions (Ri > 0)
    stable_mask = richardson_number >= 0.0
    ri_stable = jnp.minimum(richardson_number, stable_limit)
    phi_h_stable = 1.0 + 5.0 * ri_stable
    phi_m_stable = phi_h_stable
    
    # Unstable conditions (Ri < 0). Businger-Dyer stability functions:
    #     Φ_m = (1 - 16 ζ)^(-1/4)   = (1 + 16|ζ|)^(-1/4)  for ζ < 0
    #     Φ_h = (1 - 16 ζ)^(-1/2)   = (1 + 16|ζ|)^(-1/2)  for ζ < 0
    # Both are < 1 under unstable conditions, which makes the bulk
    # exchange coefficients (κ² / (ln·Φ_m·Φ_h)·…) LARGER — the standard
    # boundary-layer enhancement of turbulent mixing under buoyant
    # convection. The original implementation here used (1 - 16|Ri|)
    # inside the parenthesis, which flipped the sign and made Φ ≫ 1
    # for unstable cells, suppressing surface fluxes by ~5× over warm
    # surfaces and creating runaway near-surface heat accumulation
    # that drove the day-7 NaN on T63L47 + real terrain.
    ri_unstable = jnp.maximum(richardson_number, -0.5)  # Limit to -0.5
    x_arg = 1.0 + unstable_coeff * jnp.abs(ri_unstable)
    x = x_arg**0.25
    phi_h_unstable = x_arg**(-0.5)
    phi_m_unstable = x**(-1)
    
    # Combine stable and unstable
    phi_h = jnp.where(stable_mask, phi_h_stable, phi_h_unstable)
    phi_m = jnp.where(stable_mask, phi_m_stable, phi_m_unstable)
    
    return phi_h, phi_m


@jax.jit
def compute_exchange_coefficients(
    wind_speed: jnp.ndarray,
    roughness_momentum: jnp.ndarray,
    roughness_heat: jnp.ndarray,
    stability_heat: jnp.ndarray,
    stability_momentum: jnp.ndarray,
    min_wind_speed: float,
    von_karman: float,
    reference_height: float = 10.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute bulk exchange coefficients for momentum, heat, and moisture.
    
    Args:
        wind_speed: Wind speed [m/s] (ncol,)
        roughness_momentum: Momentum roughness length [m] (ncol, nsfc_type)
        roughness_heat: Heat roughness length [m] (ncol, nsfc_type)
        stability_heat: Heat stability function [-] (ncol, nsfc_type)
        stability_momentum: Momentum stability function [-] (ncol, nsfc_type)
        reference_height: Reference height [m]
        params: Surface parameters
        
    Returns:
        Tuple of (momentum_coeff, heat_coeff, moisture_coeff) [m/s]

    """    
    # Ensure minimum wind speed
    wind_speed_safe = jnp.maximum(wind_speed, min_wind_speed)

    # Logarithmic terms — guard against non-positive arguments
    ln_z_z0m = jnp.log(jnp.maximum(reference_height, 0.1)
                       / jnp.maximum(roughness_momentum, 1e-5))
    ln_z_z0h = jnp.log(jnp.maximum(reference_height, 0.1)
                       / jnp.maximum(roughness_heat, 1e-5))
    
    # Exchange coefficients with stability correction
    cd = (von_karman**2 / 
          (ln_z_z0m * stability_momentum)**2)
    ch = (von_karman**2 / 
          (ln_z_z0m * ln_z_z0h * stability_momentum * stability_heat))
    
    # Convert to exchange coefficients [m/s]
    momentum_coeff = cd * wind_speed_safe[:, None]
    heat_coeff = ch * wind_speed_safe[:, None]
    moisture_coeff = heat_coeff  # Assume same as heat
    
    return momentum_coeff, heat_coeff, moisture_coeff


@jax.jit
def compute_surface_humidity(
    temperature_surface: jnp.ndarray,
    pressure: jnp.ndarray,
    surface_type_indices: jnp.ndarray = None
) -> jnp.ndarray:
    """Compute saturation humidity at the surface.
    
    Args:
        temperature_surface: Surface temperature [K] (ncol, nsfc_type)
        pressure: Surface pressure [Pa] (ncol,)
        surface_type_indices: Surface type for each tile (ncol, nsfc_type)
        
    Returns:
        Surface saturation humidity [kg/kg] (ncol, nsfc_type)

    """
    # Saturation vapor pressure (simplified Clausius-Clapeyron)
    # e_sat = e0 * exp(L/Rv * (1/T0 - 1/T))
    e0 = 611.0  # Pa
    L_over_Rv = c.alhc / c.rv  # K
    T0 = c.tmelt  # K
    
    # Wide math-safety clip only — prevents divide-by-zero and exp overflow,
    # NOT a physical-range bound
    T_safe = jnp.clip(temperature_surface, 50.0, 500.0)
    exponent = jnp.clip(L_over_Rv * (1.0/T0 - 1.0/T_safe), -50.0, 50.0)
    e_sat = e0 * jnp.exp(exponent)

    # Saturation mixing ratio — cap e_sat < 0.99 * pressure
    epsilon = c.eps
    p_safe = jnp.maximum(pressure[:, None], 1.0)
    e_sat = jnp.minimum(e_sat, 0.99 * p_safe)
    q_sat = epsilon * e_sat / jnp.maximum(p_safe - (1.0 - epsilon) * e_sat, 1.0)

    # Ensure reasonable bounds
    q_sat = jnp.clip(q_sat, 0.0, 0.1)  # Max 100 g/kg

    return q_sat


@jax.jit
def compute_turbulent_fluxes(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    exchange_coeffs_momentum: jnp.ndarray,
    exchange_coeffs_heat: jnp.ndarray,
    exchange_coeffs_moisture: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> SurfaceFluxes:
    """Compute turbulent surface fluxes using bulk aerodynamic formulas.
    
    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        exchange_coeffs_momentum: Momentum exchange coefficients [m/s]
        exchange_coeffs_heat: Heat exchange coefficients [m/s]
        exchange_coeffs_moisture: Moisture exchange coefficients [m/s]
        params: Surface parameters
        
    Returns:
        Surface fluxes

    """
    ncol = atmospheric_state.temperature.shape[0]
    nsfc_type = surface_state.temperature.shape[1]
    
    # Air density
    air_density = (atmospheric_state.pressure /
                  (c.rd * atmospheric_state.temperature))

    # Surface saturation humidity
    q_surface = compute_surface_humidity(
        surface_state.temperature, atmospheric_state.pressure
    )
    
    # Wind speed
    wind_speed = jnp.sqrt(atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2)
    wind_speed = jnp.maximum(wind_speed, params.min_wind_speed)
    
    # Temperature and humidity differences
    delta_temp = (atmospheric_state.temperature[:, None] - 
                 surface_state.temperature)
    delta_humidity = (atmospheric_state.humidity[:, None] - q_surface)
    
    # Sensible heat flux [W/m²]
    sensible_heat = (air_density[:, None] * c.cpd *
                    exchange_coeffs_heat * delta_temp)
    
    # Latent heat flux [W/m²]
    # Use appropriate latent heat (condensation vs sublimation)
    latent_heat_coeff = jnp.where(
        surface_state.temperature > c.tmelt,
        c.alhc,  # Condensation
        c.alhs   # Sublimation
    )
    latent_heat = (air_density[:, None] * latent_heat_coeff * 
                  exchange_coeffs_moisture * delta_humidity)
    
    # Momentum fluxes [N/m²]
    momentum_u = (air_density[:, None] * exchange_coeffs_momentum * 
                 atmospheric_state.u_wind[:, None])
    momentum_v = (air_density[:, None] * exchange_coeffs_momentum * 
                 atmospheric_state.v_wind[:, None])
    
    # Evaporation rate [kg/m²/s]
    evaporation = air_density[:, None] * exchange_coeffs_moisture * delta_humidity
    transpiration = jnp.zeros_like(evaporation)  # Will be computed by land model
    
    # Net radiation fluxes [W/m²]
    # Shortwave
    shortwave_net = atmospheric_state.sw_downward[:, None] * (
        1.0 - (surface_state.albedo_visible_direct + 
               surface_state.albedo_visible_diffuse + 
               surface_state.albedo_nir_direct + 
               surface_state.albedo_nir_diffuse) / 4.0
    )
    
    # Longwave
    lw_upward = (params.emissivity * params.stefan_boltzmann * 
                surface_state.temperature**4)
    longwave_net = atmospheric_state.lw_downward[:, None] - lw_upward
    
    # Ground heat flux (placeholder - needs soil model)
    ground_heat = jnp.zeros((ncol, nsfc_type))
    
    # Compute grid-box means
    fractions = surface_state.fraction
    sensible_mean = jnp.sum(fractions * sensible_heat, axis=1)
    latent_mean = jnp.sum(fractions * latent_heat, axis=1)
    momentum_u_mean = jnp.sum(fractions * momentum_u, axis=1)
    momentum_v_mean = jnp.sum(fractions * momentum_v, axis=1)
    evaporation_mean = jnp.sum(fractions * evaporation, axis=1)
    
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
def compute_surface_resistances(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    richardson_number: jnp.ndarray,
    params: SurfaceParameters = SurfaceParameters.default()
) -> SurfaceResistances:
    """Compute surface resistances for heat, moisture, and momentum transfer.
    
    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        richardson_number: Bulk Richardson number
        params: Surface parameters
        
    Returns:
        Surface resistances

    """
    ncol, nsfc_type = surface_state.temperature.shape
    
    # Wind speed
    wind_speed = jnp.sqrt(atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2)
    wind_speed = jnp.maximum(wind_speed, params.min_wind_speed)
    
    # Stability functions
    stability_heat, stability_momentum = compute_stability_functions(richardson_number)
    
    # Exchange coefficients
    momentum_coeff, heat_coeff, moisture_coeff = compute_exchange_coefficients(
        wind_speed, surface_state.roughness_momentum, surface_state.roughness_heat,
        stability_heat, stability_momentum, params.min_wind_speed, params.von_karman,
    )
    
    # Convert to resistances [s/m]
    aerodynamic_momentum = 1.0 / jnp.maximum(momentum_coeff, 1e-6)
    aerodynamic_heat = 1.0 / jnp.maximum(heat_coeff, 1e-6)
    aerodynamic_moisture = 1.0 / jnp.maximum(moisture_coeff, 1e-6)
    
    # Surface resistances (simplified)
    surface_moisture = jnp.where(
        jnp.arange(nsfc_type)[None, :] == params.iwtr,  # Water
        0.0,  # No surface resistance over water
        100.0  # Simple surface resistance for land/ice
    ) * jnp.ones((ncol, nsfc_type))
    
    # Canopy and soil resistances (placeholders for land model)
    canopy_resistance = jnp.full(ncol, 100.0)  # s/m
    soil_resistance = jnp.full(ncol, 200.0)   # s/m
    
    return SurfaceResistances(
        aerodynamic_heat=aerodynamic_heat,
        aerodynamic_moisture=aerodynamic_moisture,
        aerodynamic_momentum=aerodynamic_momentum,
        surface_moisture=surface_moisture,
        canopy_resistance=canopy_resistance,
        soil_resistance=soil_resistance,
        stability_heat=stability_heat,
        stability_momentum=stability_momentum
    )


@jax.jit
def compute_surface_diagnostics(
    atmospheric_state: AtmosphericForcing,
    surface_state: SurfaceState,
    surface_fluxes: SurfaceFluxes,
    resistances: SurfaceResistances,
    params: SurfaceParameters = SurfaceParameters.default()
) -> SurfaceDiagnostics:
    """Compute standard surface diagnostics (2m temperature, 10m wind, etc.).
    
    Args:
        atmospheric_state: Atmospheric forcing
        surface_state: Surface state
        surface_fluxes: Surface fluxes
        resistances: Surface resistances
        params: Surface parameters
        
    Returns:
        Surface diagnostics

    """
    ncol, nsfc_type = surface_state.temperature.shape
    
    # Reference heights
    z_2m = 2.0    # 2m for temperature/humidity
    z_10m = 10.0  # 10m for wind
    
    # Grid-box mean surface temperature
    temp_surface_mean = jnp.sum(surface_state.fraction * surface_state.temperature, axis=1)
    
    # Simple linear interpolation for 2m temperature
    # T_2m = T_surface + (T_air - T_surface) * (z_2m / z_ref)
    temp_2m = temp_surface_mean + (atmospheric_state.temperature - temp_surface_mean) * (z_2m / z_10m)
    
    # 2m humidity (similar approach)
    q_surface_mean = compute_surface_humidity(
        temp_surface_mean[:, None], atmospheric_state.pressure
    )[:, 0]
    humidity_2m = q_surface_mean + (atmospheric_state.humidity - q_surface_mean) * (z_2m / z_10m)
    
    # 2m dew point (simplified)
    dewpoint_2m = temp_2m - 20.0 * (1.0 - atmospheric_state.humidity / 0.01)
    
    # 10m wind (use atmospheric wind as approximation)
    wind_speed_10m = jnp.sqrt(atmospheric_state.u_wind**2 + atmospheric_state.v_wind**2)
    u_wind_10m = atmospheric_state.u_wind
    v_wind_10m = atmospheric_state.v_wind
    
    # Friction velocity
    momentum_flux_magnitude = jnp.sqrt(
        surface_fluxes.momentum_u_mean**2 + surface_fluxes.momentum_v_mean**2
    )
    air_density = atmospheric_state.pressure / (c.rd * atmospheric_state.temperature)
    friction_velocity = jnp.sqrt(momentum_flux_magnitude / air_density)
    
    # Richardson number (grid-box mean)
    ri_mean = jnp.sum(surface_state.fraction * compute_bulk_richardson_number(
        atmospheric_state.temperature, surface_state.temperature,
        atmospheric_state.humidity, 
        compute_surface_humidity(surface_state.temperature, atmospheric_state.pressure),
        wind_speed_10m
    ), axis=1)
    
    # Energy balance
    net_radiation = surface_fluxes.shortwave_net + surface_fluxes.longwave_net
    net_radiation_mean = jnp.sum(surface_state.fraction * net_radiation, axis=1)
    
    energy_balance_residual = (net_radiation_mean - surface_fluxes.sensible_heat_mean - 
                              surface_fluxes.latent_heat_mean)
    
    # Tile-specific diagnostics
    temp_2m_tile = jnp.zeros((ncol, nsfc_type))
    humidity_2m_tile = jnp.zeros((ncol, nsfc_type))
    wind_speed_10m_tile = jnp.zeros((ncol, nsfc_type))
    
    for isfc in range(nsfc_type):
        temp_2m_tile = temp_2m_tile.at[:, isfc].set(
            surface_state.temperature[:, isfc] + 
            (atmospheric_state.temperature - surface_state.temperature[:, isfc]) * (z_2m / z_10m)
        )
        q_sfc = compute_surface_humidity(
            surface_state.temperature[:, isfc:isfc+1], atmospheric_state.pressure
        )[:, 0]
        humidity_2m_tile = humidity_2m_tile.at[:, isfc].set(
            q_sfc + (atmospheric_state.humidity - q_sfc) * (z_2m / z_10m)
        )
        wind_speed_10m_tile = wind_speed_10m_tile.at[:, isfc].set(wind_speed_10m)
    
    return SurfaceDiagnostics(
        temperature_2m=temp_2m,
        humidity_2m=humidity_2m,
        dewpoint_2m=dewpoint_2m,
        wind_speed_10m=wind_speed_10m,
        u_wind_10m=u_wind_10m,
        v_wind_10m=v_wind_10m,
        friction_velocity=friction_velocity,
        richardson_number=ri_mean,
        surface_layer_height=jnp.full(ncol, 100.0),  # Simplified
        net_radiation=net_radiation_mean,
        radiation_balance=net_radiation_mean,
        energy_balance_residual=energy_balance_residual,
        temperature_2m_tile=temp_2m_tile,
        humidity_2m_tile=humidity_2m_tile,
        wind_speed_10m_tile=wind_speed_10m_tile
    )