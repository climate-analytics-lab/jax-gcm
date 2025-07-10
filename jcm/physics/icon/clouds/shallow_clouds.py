"""
Shallow cloud scheme for ICON physics

This module implements a simplified cloud scheme focusing on:
- Cloud fraction diagnosis based on relative humidity
- Cloud water and ice content
- Basic condensation/evaporation processes

Based on the Lohmann and Roeckner (1996) scheme used in ICON/ECHAM.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass

from ..constants.physical_constants import (
    tmelt, alhc, alhs, rd, rv, cp, eps, grav
)


@dataclass(frozen=True)
class CloudConfig:
    """Configuration parameters for shallow cloud scheme"""
    
    # Cloud fraction parameters
    crt: float = 0.9           # Critical relative humidity at surface  
    crs: float = 0.7           # Critical relative humidity at TOA  
    nex: float = 4.0           # Exponent for RH threshold profile
    csatsc: float = 0.97       # Saturation factor for stratocumulus
    
    # Microphysics parameters
    ccraut: float = 0.0005     # Autoconversion threshold (kg/kg)
    ceffmin: float = 10.0      # Minimum cloud droplet radius (microns)
    ceffmax: float = 150.0     # Maximum cloud droplet radius (microns)
    
    # Numerical parameters
    epsilon: float = 1.0e-12   # Small number for numerical stability
    
    # Cloud ice temperature thresholds
    t_ice: float = 238.15      # Temperature below which all cloud is ice (K)
    t_mix_min: float = 238.15  # Lower bound of mixed phase (K)
    t_mix_max: float = 273.15  # Upper bound of mixed phase (K)


class CloudState(NamedTuple):
    """Cloud state variables"""
    
    cloud_fraction: jnp.ndarray     # Cloud fraction [0-1]
    cloud_water: jnp.ndarray        # Cloud liquid water content (kg/kg)
    cloud_ice: jnp.ndarray          # Cloud ice content (kg/kg)
    rel_humidity: jnp.ndarray       # Relative humidity [0-1]
    
    # Diagnostics
    total_cloud_cover: jnp.ndarray  # Column total cloud cover
    
    
class CloudTendencies(NamedTuple):
    """Tendencies from cloud processes"""
    
    dtedt: jnp.ndarray         # Temperature tendency (K/s)
    dqdt: jnp.ndarray          # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray         # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray         # Cloud ice tendency (kg/kg/s)
    
    # Surface precipitation fluxes
    rain_flux: jnp.ndarray     # Surface rain flux (kg/m²/s)
    snow_flux: jnp.ndarray     # Surface snow flux (kg/m²/s)


def saturation_vapor_pressure_water(temperature: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate saturation vapor pressure over water using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)
    """
    t_celsius = temperature - tmelt
    return 610.78 * jnp.exp(17.27 * t_celsius / (t_celsius + 237.3))


def saturation_vapor_pressure_ice(temperature: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate saturation vapor pressure over ice using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)
    """
    t_celsius = temperature - tmelt
    return 610.78 * jnp.exp(21.87 * t_celsius / (t_celsius + 265.5))


def saturation_specific_humidity(
    pressure: jnp.ndarray, 
    temperature: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate saturation specific humidity
    
    Args:
        pressure: Pressure (Pa)
        temperature: Temperature (K)
        
    Returns:
        Saturation specific humidity (kg/kg)
    """
    # Use appropriate saturation vapor pressure based on temperature
    es_water = saturation_vapor_pressure_water(temperature)
    es_ice = saturation_vapor_pressure_ice(temperature)
    
    # Blend between ice and water saturation in mixed phase region
    # Linear interpolation between t_ice and tmelt
    weight = jnp.clip((temperature - 238.15) / (tmelt - 238.15), 0.0, 1.0)
    es = weight * es_water + (1.0 - weight) * es_ice
    
    # Convert to saturation specific humidity
    qs = eps * es / (pressure - es * (1.0 - eps))
    return jnp.maximum(qs, 0.0)


def calculate_cloud_fraction(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    surface_pressure: float,
    config: CloudConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate cloud fraction using relative humidity scheme
    
    Based on Lohmann and Roeckner (1996) diagnostic cloud scheme.
    
    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        pressure: Pressure (Pa)
        surface_pressure: Surface pressure (Pa)
        config: Cloud configuration
        
    Returns:
        Tuple of (cloud_fraction, relative_humidity)
    """
    # Calculate saturation specific humidity
    qs = saturation_specific_humidity(pressure, temperature)
    
    # Calculate relative humidity
    rel_humidity = specific_humidity / (qs + config.epsilon)
    rel_humidity = jnp.clip(rel_humidity, 0.0, 1.0)
    
    # Calculate critical relative humidity threshold
    # Varies from crt at surface to crs at TOA
    # Following Lohmann & Roeckner (1996) formulation
    sigma = pressure / surface_pressure  # Normalized pressure (1 at surface, 0 at TOA)
    # RHc = crt at surface (sigma=1) and crs at TOA (sigma→0)
    # Using exponential interpolation: at sigma=1, exp(0)=1 so rhc=crt
    # as sigma→0, exp(-nex)→0 so rhc→crs
    rhc = config.crs + (config.crt - config.crs) * jnp.exp(
        -config.nex * (1.0 - sigma)
    )
    
    # Calculate cloud fraction using quadratic relationship
    # b_0 = (RH - RH_crit) / (1 - RH_crit)
    b0 = (rel_humidity - rhc) / (1.0 - rhc + config.epsilon)
    b0 = jnp.clip(b0, 0.0, 1.0)
    
    # Cloud fraction: cc = 1 - sqrt(1 - b0)
    cloud_fraction = 1.0 - jnp.sqrt(1.0 - b0)
    
    # Apply minimum cloud fraction threshold
    cloud_fraction = jnp.where(cloud_fraction < 0.01, 0.0, cloud_fraction)
    
    return cloud_fraction, rel_humidity


def partition_cloud_phase(
    temperature: jnp.ndarray,
    total_cloud_water: jnp.ndarray,
    config: CloudConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Partition cloud water between liquid and ice phases
    
    Args:
        temperature: Temperature (K)
        total_cloud_water: Total cloud condensate (kg/kg)
        config: Cloud configuration
        
    Returns:
        Tuple of (cloud_liquid, cloud_ice)
    """
    # Calculate ice fraction based on temperature
    # All ice below t_ice, all liquid above tmelt
    # Linear transition in between
    ice_frac = jnp.clip(
        (config.t_mix_max - temperature) / (config.t_mix_max - config.t_mix_min),
        0.0, 1.0
    )
    
    # Partition cloud water
    cloud_ice = ice_frac * total_cloud_water
    cloud_liquid = (1.0 - ice_frac) * total_cloud_water
    
    return cloud_liquid, cloud_ice


def condensation_evaporation(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    pressure: jnp.ndarray,
    dt: float,
    config: CloudConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate condensation/evaporation tendencies
    
    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        cloud_water: Cloud liquid water (kg/kg)
        cloud_ice: Cloud ice (kg/kg)
        cloud_fraction: Cloud fraction [0-1]
        pressure: Pressure (Pa)
        dt: Time step (s)
        config: Cloud configuration
        
    Returns:
        Tuple of (dT/dt, dq/dt, dqc/dt, dqi/dt)
    """
    # Calculate saturation specific humidity
    qs = saturation_specific_humidity(pressure, temperature)
    
    # In-cloud specific humidity and saturation
    # Assume saturation inside clouds
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0
    )
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / cloud_fraction,
        0.0
    )
    
    # Calculate condensation/evaporation
    # Positive for condensation, negative for evaporation
    q_excess = specific_humidity - qs
    
    # Condensation/evaporation rate (instantaneous adjustment)
    # Positive q_excess -> condensation, negative -> evaporation
    cond_evap_rate = q_excess / dt
    
    # Limit evaporation to available cloud water/ice
    total_cloud = cloud_water + cloud_ice
    max_evap_rate = -total_cloud / dt
    
    # Apply limits
    cond_evap = jnp.where(
        cond_evap_rate < 0,  # Evaporation
        jnp.maximum(cond_evap_rate, max_evap_rate),
        cond_evap_rate  # Condensation
    )
    
    # Specific humidity tendency (opposite sign)
    dqdt = -cond_evap
    
    # Partition between liquid and ice based on temperature
    weight_liquid = jnp.clip(
        (temperature - config.t_mix_min) / (config.t_mix_max - config.t_mix_min),
        0.0, 1.0
    )
    
    # Tendencies for cloud water and ice
    # For condensation (positive cond_evap), partition between liquid and ice
    # For evaporation (negative cond_evap), remove proportionally from existing phases
    dqcdt = jnp.where(
        cond_evap > 0,  # Condensation
        weight_liquid * cond_evap,
        cond_evap * cloud_water / (total_cloud + config.epsilon)  # Proportional evaporation
    )
    dqidt = jnp.where(
        cond_evap > 0,  # Condensation  
        (1.0 - weight_liquid) * cond_evap,
        cond_evap * cloud_ice / (total_cloud + config.epsilon)  # Proportional evaporation
    )
    
    # Temperature tendency from latent heat
    # Use appropriate latent heat based on phase
    L = jnp.where(
        cond_evap > 0,  # Condensation - use weighted latent heat
        weight_liquid * alhc + (1.0 - weight_liquid) * alhs,
        # Evaporation - use weighted latent heat based on what's evaporating
        (cloud_water * alhc + cloud_ice * alhs) / (total_cloud + config.epsilon)
    )
    # Positive cond_evap (condensation) releases heat -> positive temperature tendency
    # Negative cond_evap (evaporation) consumes heat -> negative temperature tendency
    dtedt = L * cond_evap / cp
    
    return dtedt, dqdt, dqcdt, dqidt


def shallow_cloud_scheme(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray, 
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    surface_pressure: float,
    dt: float,
    config: Optional[CloudConfig] = None
) -> Tuple[CloudTendencies, CloudState]:
    """
    Main shallow cloud scheme
    
    Args:
        temperature: Temperature (K) [nlev] or scalar
        specific_humidity: Specific humidity (kg/kg) [nlev] or scalar
        pressure: Pressure (Pa) [nlev] or scalar
        cloud_water: Cloud liquid water (kg/kg) [nlev] or scalar
        cloud_ice: Cloud ice (kg/kg) [nlev] or scalar
        surface_pressure: Surface pressure (Pa)
        dt: Time step (s)
        config: Cloud configuration
        
    Returns:
        Tuple of (tendencies, cloud_state)
    """
    if config is None:
        config = CloudConfig()
    
    # Ensure all inputs are arrays
    temperature = jnp.atleast_1d(temperature)
    specific_humidity = jnp.atleast_1d(specific_humidity)
    pressure = jnp.atleast_1d(pressure)
    cloud_water = jnp.atleast_1d(cloud_water)
    cloud_ice = jnp.atleast_1d(cloud_ice)
    
    nlev = temperature.shape[0]
    
    # Calculate cloud fraction and relative humidity
    cloud_fraction, rel_humidity = calculate_cloud_fraction(
        temperature, specific_humidity, pressure, surface_pressure, config
    )
    
    # Calculate condensation/evaporation
    dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
        temperature, specific_humidity, cloud_water, cloud_ice,
        cloud_fraction, pressure, dt, config
    )
    
    # Simple precipitation formation (autoconversion)
    # Convert cloud water to rain if it exceeds threshold
    rain_rate = jnp.where(
        cloud_water > config.ccraut,
        0.001 * (cloud_water - config.ccraut) / dt,  # Simple rate
        0.0
    )
    snow_rate = jnp.where(
        cloud_ice > config.ccraut * 0.1,  # Lower threshold for ice
        0.01 * cloud_ice / dt,  # Faster rate for ice
        0.0
    )
    
    # Update tendencies for precipitation
    dqcdt = dqcdt - rain_rate
    dqidt = dqidt - snow_rate
    
    # Calculate total cloud cover (maximum overlap assumption)
    total_cloud_cover = jnp.max(cloud_fraction)
    
    # Surface precipitation fluxes (simplified - no vertical integration)
    # In reality, would need to integrate through column accounting for evaporation
    rain_flux = jnp.sum(rain_rate) * 1000.0  # Convert to kg/m²/s (approximate)
    snow_flux = jnp.sum(snow_rate) * 1000.0
    
    # Create output structures
    tendencies = CloudTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dqcdt=dqcdt,
        dqidt=dqidt,
        rain_flux=jnp.array(rain_flux),
        snow_flux=jnp.array(snow_flux)
    )
    
    state = CloudState(
        cloud_fraction=cloud_fraction,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        rel_humidity=rel_humidity,
        total_cloud_cover=jnp.array(total_cloud_cover)
    )
    
    return tendencies, state