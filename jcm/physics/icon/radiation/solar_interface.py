"""
Solar radiation interface using jax-solar

This module provides solar radiation calculations using the jax-solar package.
jax-solar is now a required dependency.

Date: 2025-01-16
"""

import jax.numpy as jnp
import jax
from typing import Union, Optional, Tuple

# Import jax-solar (required dependency)
import jax_solar


# Constants
TOTAL_SOLAR_IRRADIANCE = 1361.0  # W/m^2
SOLAR_IRRADIANCE_VARIATION = 3.4  # W/m^2
DAYS_PER_YEAR = 365.25
SECONDS_PER_DAY = 86400.0
SOLAR_CONSTANT = TOTAL_SOLAR_IRRADIANCE  # Alias for compatibility


# Use jax-solar's OrbitalTime directly
OrbitalTime = jax_solar.OrbitalTime

# Helper function to create OrbitalTime from day/hour
def orbital_time_from_day_hour(day_of_year: float, hour_utc: float):
    """Create OrbitalTime from day of year and hour UTC"""
    orbital_phase = 2 * jnp.pi * (day_of_year - 1) / 365.25
    synodic_phase = 2 * jnp.pi * hour_utc / 24.0
    return OrbitalTime(orbital_phase=orbital_phase, synodic_phase=synodic_phase)


@jax.jit
def get_declination(orbital_phase: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Returns angle between the Earth-Sun line and the Earth equatorial plane.
    
    Args:
        orbital_phase: Phase in Earth's orbit (radians)
        
    Returns:
        Solar declination (radians)
    """
    return jax_solar.get_declination(orbital_phase)


@jax.jit
def get_hour_angle(
    synodic_phase: Union[float, jnp.ndarray],
    longitude: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns the hour angle.
    
    Args:
        synodic_phase: Phase in Earth's rotation (radians)
        longitude: Longitude (degrees)
        
    Returns:
        Hour angle (radians)
    """
    # Create a minimal OrbitalTime object with just synodic_phase
    # For simplicity, set orbital_phase to 0
    orbital_time = jax_solar.OrbitalTime(orbital_phase=0.0, synodic_phase=synodic_phase)
    return jax_solar.get_hour_angle(orbital_time, longitude)


def get_solar_sin_altitude(
    orbital_time: OrbitalTime,
    longitude: jnp.ndarray,
    latitude: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns sine of the solar altitude angle.
    
    This is equivalent to the cosine of the solar zenith angle.
    
    Args:
        orbital_time: OrbitalTime object
        longitude: Longitude (degrees)
        latitude: Latitude (degrees)
        
    Returns:
        Sine of solar altitude (= cosine of zenith angle)
    """
    return jax_solar.get_solar_sin_altitude(orbital_time, longitude, latitude)


@jax.jit
def direct_solar_irradiance(
    orbital_phase: Union[float, jnp.ndarray],
    mean_irradiance: float = TOTAL_SOLAR_IRRADIANCE,
    variation: float = SOLAR_IRRADIANCE_VARIATION
) -> jnp.ndarray:
    """
    Returns direct solar irradiance accounting for Earth-Sun distance.
    
    Args:
        orbital_phase: Phase in Earth's orbit (radians)
        mean_irradiance: Mean solar irradiance (W/m^2)
        variation: Annual variation in irradiance (W/m^2)
        
    Returns:
        Direct solar irradiance (W/m^2)
    """
    return jax_solar.direct_solar_irradiance(orbital_phase, mean_irradiance, variation)


def radiation_flux(
    time: Union[OrbitalTime, Tuple[float, float]],
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    mean_irradiance: float = TOTAL_SOLAR_IRRADIANCE,
    variation: float = SOLAR_IRRADIANCE_VARIATION
) -> jnp.ndarray:
    """
    Returns TOA incident radiation flux.
    
    Args:
        time: OrbitalTime object or tuple of (day_of_year, hour_utc)
        longitude: Longitude (degrees)
        latitude: Latitude (degrees)
        mean_irradiance: Mean solar irradiance (W/m^2)
        variation: Annual variation in irradiance (W/m^2)
        
    Returns:
        TOA incident radiation flux (W/m^2)
    """
    return jax_solar.radiation_flux(time, longitude, latitude, mean_irradiance, variation)


def normalized_radiation_flux(
    time: Union[OrbitalTime, Tuple[float, float]],
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    mean_irradiance: float = TOTAL_SOLAR_IRRADIANCE,
    variation: float = SOLAR_IRRADIANCE_VARIATION
) -> jnp.ndarray:
    """
    Like radiation_flux(), but normalized to between 0 and 1.
    
    The normalization is relative to the mean solar irradiance.
    
    Args:
        time: OrbitalTime object or tuple of (day_of_year, hour_utc)
        longitude: Longitude (degrees)
        latitude: Latitude (degrees)
        mean_irradiance: Mean solar irradiance (W/m^2)
        variation: Annual variation in irradiance (W/m^2)
        
    Returns:
        Normalized TOA incident radiation flux (0-1)
    """
    return jax_solar.normalized_radiation_flux(time, longitude, latitude, mean_irradiance, variation)


# Convenience functions that match our original interface
def calculate_toa_radiation(
    day_of_year: float,
    hour_utc: float,
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    solar_constant: float = TOTAL_SOLAR_IRRADIANCE
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate top-of-atmosphere radiation.
    
    Args:
        day_of_year: Day of year (1-365)
        hour_utc: Hour in UTC (0-24)
        longitude: Longitude (degrees)
        latitude: Latitude (degrees)
        solar_constant: Solar constant (W/m^2)
        
    Returns:
        Tuple of (flux, cos_zenith, hour_angle)
    """
    # Create orbital time
    orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
    
    # Get flux
    flux = radiation_flux(orbital_time, longitude, latitude, solar_constant)
    
    # Get cosine of zenith angle
    cos_zenith = get_solar_sin_altitude(orbital_time, longitude, latitude)
    
    # Get hour angle
    ha = get_hour_angle(orbital_time.synodic_phase, longitude)
    
    return flux, cos_zenith, ha


# Export the appropriate implementation status
def get_implementation_info():
    """Get information about which implementation is being used"""
    return "Using jax-solar implementation"


# Test the implementation
def test_solar_interface():
    """Test the solar interface implementation"""
    print(get_implementation_info())
    
    # Test data
    day_of_year = 172.0  # Summer solstice
    hour_utc = 12.0      # Noon UTC
    longitude = jnp.array([0.0, -74.0, 139.7])  # Greenwich, NYC, Tokyo
    latitude = jnp.array([51.5, 40.7, 35.7])    # London, NYC, Tokyo
    
    # Test using convenience function
    flux, cos_zenith, hour_angle = calculate_toa_radiation(
        day_of_year, hour_utc, longitude, latitude
    )
    
    print(f"\nTOA radiation flux: {flux}")
    print(f"Cosine zenith angle: {cos_zenith}")
    print(f"Hour angle: {hour_angle}")
    
    # Test using OrbitalTime interface
    orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
    flux2 = radiation_flux(orbital_time, longitude, latitude)
    normalized = normalized_radiation_flux(orbital_time, longitude, latitude)
    
    print(f"\nUsing OrbitalTime interface:")
    print(f"Flux: {flux2}")
    print(f"Normalized flux: {normalized}")
    
    # Test individual functions
    declination = get_declination(orbital_time.orbital_phase)
    print(f"\nSolar declination: {jnp.rad2deg(declination):.2f} degrees")
    
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    print(f"Sine of solar altitude: {sin_altitude}")
    
    assert jnp.allclose(flux, flux2)
    assert jnp.allclose(cos_zenith, sin_altitude)
    assert jnp.all(normalized >= 0) and jnp.all(normalized <= 1)
    
    print("\n✓ All solar interface tests passed!")


if __name__ == "__main__":
    test_solar_interface()