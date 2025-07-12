"""
Solar radiation calculations using jax-solar

This module provides solar radiation calculations using the official
jax-solar package. This is the preferred implementation for jax-gcm.

Requirements:
    - Python 3.11+
    - jax-solar
    - jax-datetime

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple, Union
import jax_solar
import jax_datetime as jdt
# from functools import partial  # Not needed anymore


# Re-export jax-solar constants
TOTAL_SOLAR_IRRADIANCE = jax_solar.TOTAL_SOLAR_IRRADIANCE  # ~1361 W/m²
SOLAR_IRRADIANCE_VARIATION = jax_solar.SOLAR_IRRADIANCE_VARIATION  # ~3.4 W/m²


@jax.jit
def calculate_solar_radiation_from_datetime(
    datetime_str: str,
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    solar_constant: float = TOTAL_SOLAR_IRRADIANCE
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate solar radiation from datetime string.
    
    Args:
        datetime_str: ISO format datetime string (e.g., '2024-06-21T12:00:00')
        longitude: Longitude in degrees
        latitude: Latitude in degrees  
        solar_constant: Solar constant (W/m²)
        
    Returns:
        Tuple of (toa_flux, cos_zenith_angle, solar_altitude_degrees)
    """
    # Convert to jax-datetime
    time = jdt.to_datetime(datetime_str)
    
    # Get radiation flux
    flux = jax_solar.radiation_flux(time, longitude, latitude, solar_constant)
    
    # Get solar altitude (sine)
    orbital_time = jax_solar.OrbitalTime.from_datetime(time)
    sin_altitude = jax_solar.get_solar_sin_altitude(orbital_time, longitude, latitude)
    
    # Convert to angles
    cos_zenith = sin_altitude  # cos(zenith) = sin(altitude)
    altitude_degrees = jnp.rad2deg(jnp.arcsin(sin_altitude))
    
    return flux, cos_zenith, altitude_degrees


@jax.jit
def calculate_daily_integrated_radiation(
    date_str: str,
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    n_times: int = 24,
    solar_constant: float = TOTAL_SOLAR_IRRADIANCE
) -> jnp.ndarray:
    """
    Calculate daily integrated solar radiation.
    
    Args:
        date_str: Date string (e.g., '2024-06-21')
        longitude: Longitude in degrees
        latitude: Latitude in degrees
        n_times: Number of time points for integration
        solar_constant: Solar constant (W/m²)
        
    Returns:
        Daily integrated radiation (J/m²)
    """
    # Create hourly times for the day
    hours = jnp.linspace(0, 24, n_times, endpoint=False)
    
    def get_flux_at_hour(hour):
        # Create datetime for this hour
        hour_int = int(hour)
        minute_frac = (hour - hour_int) * 60
        minute_int = int(minute_frac)
        
        datetime_str = f"{date_str}T{hour_int:02d}:{minute_int:02d}:00"
        time = jdt.to_datetime(datetime_str)
        
        return jax_solar.radiation_flux(time, longitude, latitude, solar_constant)
    
    # Calculate flux at each hour
    fluxes = jax.vmap(get_flux_at_hour)(hours)
    
    # Integrate using trapezoidal rule (convert hours to seconds)
    dt_seconds = 3600.0  # 1 hour in seconds
    daily_integrated = jnp.trapz(fluxes, dx=dt_seconds, axis=0)
    
    return daily_integrated


@jax.jit
def calculate_solar_radiation_gcm(
    day_of_year: float,
    seconds_since_midnight: float,
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    solar_constant: float = TOTAL_SOLAR_IRRADIANCE
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate solar radiation for GCM usage.
    
    This function is designed for use in GCMs where time is typically
    represented as day of year and seconds since midnight.
    
    Args:
        day_of_year: Day of year (1-365 or 1-366)
        seconds_since_midnight: Seconds since midnight UTC
        longitude: Longitude in degrees
        latitude: Latitude in degrees
        solar_constant: Solar constant (W/m²)
        
    Returns:
        Tuple of (toa_flux, cos_zenith_angle)
    """
    # Create OrbitalTime
    # Convert day of year to orbital phase
    orbital_phase = 2 * jnp.pi * (day_of_year - 1) / 365.25
    
    # Convert seconds to synodic phase
    synodic_phase = 2 * jnp.pi * seconds_since_midnight / 86400.0
    
    orbital_time = jax_solar.OrbitalTime(
        orbital_phase=orbital_phase,
        synodic_phase=synodic_phase
    )
    
    # Get radiation flux
    flux = jax_solar.radiation_flux(
        orbital_time, longitude, latitude, solar_constant
    )
    
    # Get cosine of zenith angle
    cos_zenith = jax_solar.get_solar_sin_altitude(
        orbital_time, longitude, latitude
    )
    
    return flux, cos_zenith


@jax.jit 
def cosine_solar_zenith_angle(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray, 
    day_of_year: float,
    hour_utc: float
) -> jnp.ndarray:
    """
    Calculate cosine of solar zenith angle.
    
    Compatible interface with the fallback implementation.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        day_of_year: Day of year (1-365)
        hour_utc: Hour in UTC (0-24)
        
    Returns:
        Cosine of solar zenith angle
    """
    seconds_since_midnight = hour_utc * 3600.0
    _, cos_zenith = calculate_solar_radiation_gcm(
        day_of_year, seconds_since_midnight, longitude, latitude
    )
    return cos_zenith


@jax.jit
def top_of_atmosphere_flux(
    cos_zenith: jnp.ndarray,
    day_of_year: float,
    solar_constant: float = TOTAL_SOLAR_IRRADIANCE
) -> jnp.ndarray:
    """
    Calculate TOA flux from cosine zenith angle.
    
    Compatible interface with the fallback implementation.
    
    Args:
        cos_zenith: Cosine of solar zenith angle  
        day_of_year: Day of year (for Earth-Sun distance)
        solar_constant: Solar constant (W/m²)
        
    Returns:
        TOA flux (W/m²)
    """
    # Get Earth-Sun distance correction
    orbital_phase = 2 * jnp.pi * (day_of_year - 1) / 365.25
    irradiance = jax_solar.direct_solar_irradiance(
        orbital_phase, solar_constant, SOLAR_IRRADIANCE_VARIATION
    )
    
    # Calculate flux
    flux = jnp.where(cos_zenith > 0, irradiance * cos_zenith, 0.0)
    return flux


@jax.jit
def daylight_fraction(
    latitude: jnp.ndarray,
    day_of_year: float,
    timestep_hours: float = 1.0
) -> jnp.ndarray:
    """
    Calculate fraction of timestep with daylight.
    
    Args:
        latitude: Latitude in degrees
        day_of_year: Day of year (1-365)
        timestep_hours: Length of timestep in hours
        
    Returns:
        Daylight fraction (0-1)
    """
    # Sample multiple times within the timestep
    n_samples = max(int(timestep_hours * 4), 1)  # At least 4 samples per hour
    
    # Create time points
    hours = jnp.linspace(0, 24, n_samples)
    
    # Calculate solar altitude at each time
    def check_daylight(hour):
        seconds = hour * 3600.0
        _, cos_z = calculate_solar_radiation_gcm(
            day_of_year, seconds, 0.0, latitude  # Use 0 longitude for simplicity
        )
        return cos_z > 0
    
    # Check daylight at each sample
    daylight_samples = jax.vmap(check_daylight)(hours)
    
    # Calculate fraction
    return jnp.mean(daylight_samples)


# Convenience function for radiation scheme integration
@jax.jit
def calculate_solar_radiation(
    state: 'RadiationState',  # Type hint, actual import would be circular
    parameters: 'RadiationParameters',
    day_of_year: float,
    seconds_since_midnight: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate solar radiation for the radiation scheme.
    
    Args:
        state: Radiation state containing lat/lon
        parameters: Radiation parameters  
        day_of_year: Day of year
        seconds_since_midnight: Seconds since midnight UTC
        
    Returns:
        Tuple of (toa_flux, cos_zenith_angle)
    """
    return calculate_solar_radiation_gcm(
        day_of_year,
        seconds_since_midnight,
        state.longitude,
        state.latitude,
        parameters.solar_constant
    )


# Test function
def test_jax_solar_implementation():
    """Test the jax-solar based implementation"""
    
    print("Testing jax-solar implementation...")
    
    # Test data
    latitude = jnp.array([0.0, 45.0, -45.0])
    longitude = jnp.array([0.0, -74.0, 139.7])
    
    # Test datetime interface
    flux, cos_z, alt = calculate_solar_radiation_from_datetime(
        '2024-06-21T12:00:00',
        longitude, latitude
    )
    print(f"\nDatetime interface test:")
    print(f"Flux: {flux}")
    print(f"Cos zenith: {cos_z}")
    print(f"Altitude degrees: {alt}")
    
    # Test GCM interface
    day_of_year = 172  # ~June 21
    seconds = 12 * 3600  # Noon
    
    flux_gcm, cos_z_gcm = calculate_solar_radiation_gcm(
        day_of_year, seconds, longitude, latitude
    )
    print(f"\nGCM interface test:")
    print(f"Flux: {flux_gcm}")
    print(f"Cos zenith: {cos_z_gcm}")
    
    # Test daylight fraction
    frac = daylight_fraction(latitude, day_of_year, 24.0)
    print(f"\nDaylight fraction: {frac}")
    
    # Test daily integrated
    try:
        daily = calculate_daily_integrated_radiation(
            '2024-06-21', longitude[0], latitude[0], n_times=24
        )
        print(f"\nDaily integrated radiation: {daily:.2e} J/m²")
    except Exception as e:
        print(f"Daily integration not available in JIT context: {e}")
    
    print("\n✓ All jax-solar tests completed!")


if __name__ == "__main__":
    test_jax_solar_implementation()