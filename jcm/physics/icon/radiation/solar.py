"""
Solar radiation calculations

This module handles solar geometry and top-of-atmosphere radiation
using the jax-solar package.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple
from functools import partial

# Note: In production, we would import jax_solar
# For now, we'll implement basic solar calculations
# import jax_solar
# import jax_datetime as jdt


@jax.jit
def solar_declination(day_of_year: float) -> float:
    """
    Calculate solar declination angle.
    
    Args:
        day_of_year: Day of year (1-365)
        
    Returns:
        Solar declination in radians
    """
    # Approximation for solar declination
    angle = 2.0 * jnp.pi * (day_of_year - 1) / 365.0
    declination = 0.006918 - 0.399912 * jnp.cos(angle) + 0.070257 * jnp.sin(angle) \
                  - 0.006758 * jnp.cos(2*angle) + 0.000907 * jnp.sin(2*angle) \
                  - 0.002697 * jnp.cos(3*angle) + 0.00148 * jnp.sin(3*angle)
    return declination


@jax.jit
def hour_angle(longitude: jnp.ndarray, hour_utc: float) -> jnp.ndarray:
    """
    Calculate solar hour angle.
    
    Args:
        longitude: Longitude in degrees
        hour_utc: Hour of day in UTC (0-24)
        
    Returns:
        Hour angle in radians
    """
    # Solar noon is at 12:00 UTC at Greenwich
    # Hour angle is 15 degrees per hour from solar noon
    solar_time = hour_utc + longitude / 15.0  # Convert longitude to hours
    hour_angle_deg = 15.0 * (solar_time - 12.0)
    return jnp.deg2rad(hour_angle_deg)


@jax.jit
def cosine_solar_zenith_angle(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    day_of_year: float,
    hour_utc: float
) -> jnp.ndarray:
    """
    Calculate cosine of solar zenith angle.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees  
        day_of_year: Day of year (1-365)
        hour_utc: Hour of day in UTC (0-24)
        
    Returns:
        Cosine of solar zenith angle
    """
    # Convert to radians
    lat_rad = jnp.deg2rad(latitude)
    
    # Solar calculations
    decl = solar_declination(day_of_year)
    h_angle = hour_angle(longitude, hour_utc)
    
    # Cosine of zenith angle
    cos_zenith = (jnp.sin(lat_rad) * jnp.sin(decl) + 
                  jnp.cos(lat_rad) * jnp.cos(decl) * jnp.cos(h_angle))
    
    # Ensure non-negative (sun above horizon)
    return jnp.maximum(0.0, cos_zenith)


@jax.jit
def top_of_atmosphere_flux(
    cos_zenith: jnp.ndarray,
    solar_constant: float = 1361.0,
    earth_sun_distance_factor: float = 1.0
) -> jnp.ndarray:
    """
    Calculate top-of-atmosphere solar flux.
    
    Args:
        cos_zenith: Cosine of solar zenith angle
        solar_constant: Solar constant (W/m²)
        earth_sun_distance_factor: Correction for Earth-Sun distance
        
    Returns:
        TOA solar flux (W/m²)
    """
    return solar_constant * earth_sun_distance_factor * cos_zenith


@jax.jit
def earth_sun_distance_factor(day_of_year: float) -> float:
    """
    Calculate Earth-Sun distance correction factor.
    
    Args:
        day_of_year: Day of year (1-365)
        
    Returns:
        Distance correction factor (dimensionless)
    """
    # Approximation for Earth-Sun distance variation
    angle = 2.0 * jnp.pi * (day_of_year - 1) / 365.0
    
    # Eccentricity correction
    distance_factor = 1.000110 + 0.034221 * jnp.cos(angle) + 0.001280 * jnp.sin(angle) \
                     + 0.000719 * jnp.cos(2*angle) + 0.000077 * jnp.sin(2*angle)
    
    return distance_factor


@jax.jit
def daylight_fraction(
    latitude: jnp.ndarray,
    day_of_year: float,
    timestep_hours: float = 1.0
) -> jnp.ndarray:
    """
    Calculate fraction of timestep with daylight.
    
    For instantaneous calculations, returns 1 where sun is up, 0 otherwise.
    For longer timesteps, returns fraction of timestep with daylight.
    
    Args:
        latitude: Latitude in degrees
        day_of_year: Day of year (1-365)
        timestep_hours: Length of timestep in hours
        
    Returns:
        Daylight fraction (0-1)
    """
    # For now, simple implementation
    # In full implementation, would calculate sunrise/sunset times
    
    # Convert to radians
    lat_rad = jnp.deg2rad(latitude)
    decl = solar_declination(day_of_year)
    
    # Calculate sunrise/sunset hour angle
    cos_hour_angle_sunset = -jnp.tan(lat_rad) * jnp.tan(decl)
    
    # Check for polar day/night
    polar_day = cos_hour_angle_sunset < -1.0
    polar_night = cos_hour_angle_sunset > 1.0
    
    # Normal case
    hour_angle_sunset = jnp.arccos(jnp.clip(cos_hour_angle_sunset, -1.0, 1.0))
    daylight_hours = 2.0 * hour_angle_sunset * 12.0 / jnp.pi
    
    # Combine cases
    daylight_hours = jnp.where(polar_day, 24.0, daylight_hours)
    daylight_hours = jnp.where(polar_night, 0.0, daylight_hours)
    
    # For instantaneous calculation
    return jnp.where(
        timestep_hours <= 1.0,
        jnp.where(daylight_hours > 0, 1.0, 0.0),
        jnp.clip(daylight_hours / 24.0, 0.0, 1.0)
    )


# Simplified interface that would use jax_solar in production
def calculate_solar_radiation(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    time_info: dict,
    solar_constant: float = 1361.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate solar radiation parameters.
    
    In production, this would use jax_solar for accurate calculations.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        time_info: Dictionary with 'day_of_year' and 'hour_utc'
        solar_constant: Solar constant (W/m²)
        
    Returns:
        Tuple of (toa_flux, cos_zenith, daylight_fraction)
    """
    day_of_year = time_info['day_of_year']
    hour_utc = time_info['hour_utc']
    
    # Calculate solar geometry
    cos_zenith = cosine_solar_zenith_angle(latitude, longitude, day_of_year, hour_utc)
    daylight_frac = daylight_fraction(latitude, day_of_year)
    
    # Earth-Sun distance correction
    distance_factor = earth_sun_distance_factor(day_of_year)
    
    # TOA flux
    toa_flux = top_of_atmosphere_flux(cos_zenith, solar_constant, distance_factor)
    
    return toa_flux, cos_zenith, daylight_frac


# Test functions
def test_solar_calculations():
    """Test solar calculation functions"""
    
    # Test declination
    # Summer solstice (day 172)
    decl_summer = solar_declination(172.0)
    assert jnp.abs(decl_summer - 0.4091) < 0.01  # ~23.4 degrees
    
    # Winter solstice (day 355)
    decl_winter = solar_declination(355.0)
    assert jnp.abs(decl_winter + 0.4091) < 0.01  # ~-23.4 degrees
    
    # Test zenith angle
    # Noon at equator on equinox
    cos_z = cosine_solar_zenith_angle(0.0, 0.0, 80.0, 12.0)  # Spring equinox
    assert jnp.abs(cos_z - 1.0) < 0.01  # Sun directly overhead
    
    # Test TOA flux
    flux = top_of_atmosphere_flux(1.0, 1361.0)
    assert jnp.abs(flux - 1361.0) < 0.1
    
    print("Solar calculations tests passed!")


if __name__ == "__main__":
    test_solar_calculations()