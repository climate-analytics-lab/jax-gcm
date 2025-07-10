"""
Solar radiation interface with jax-solar compatibility

This module provides a compatible interface with jax-solar package.
When jax-solar is available, it uses the official implementation.
Otherwise, it falls back to our JAX-based implementation.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Union, Optional, Tuple
import warnings

# Try to import jax-solar
try:
    import jax_solar
    HAS_JAX_SOLAR = True
except ImportError:
    HAS_JAX_SOLAR = False
    warnings.warn(
        "jax-solar not available. Using fallback implementation. "
        "Install jax-solar for the official implementation: "
        "pip install jax-solar (requires Python 3.11+)"
    )

# Import our fallback implementation
from .solar import (
    solar_declination as _solar_declination,
    hour_angle as _hour_angle,
    cosine_solar_zenith_angle as _cosine_solar_zenith_angle,
    earth_sun_distance_factor as _earth_sun_distance,
    top_of_atmosphere_flux as _top_of_atmosphere_flux
)


# Constants
TOTAL_SOLAR_IRRADIANCE = 1361.0  # W/m^2
SOLAR_IRRADIANCE_VARIATION = 3.4  # W/m^2
DAYS_PER_YEAR = 365.25
SECONDS_PER_DAY = 86400.0
SOLAR_CONSTANT = TOTAL_SOLAR_IRRADIANCE  # Alias for compatibility


class OrbitalTime:
    """Compatible implementation of jax-solar OrbitalTime"""
    
    def __init__(self, orbital_phase: float, synodic_phase: float):
        self.orbital_phase = orbital_phase
        self.synodic_phase = synodic_phase
    
    @classmethod
    def from_day_of_year_hour(cls, day_of_year: float, hour_utc: float):
        """Create from day of year and hour"""
        orbital_phase = 2 * jnp.pi * (day_of_year - 1) / DAYS_PER_YEAR
        synodic_phase = 2 * jnp.pi * hour_utc / 24.0
        return cls(orbital_phase, synodic_phase)
    
    @classmethod
    def from_datetime(cls, when, origin='1970-01-01T00:00:00', days_per_year: float = DAYS_PER_YEAR):
        """Create from datetime (simplified version)"""
        # This is a simplified version - in practice would need proper datetime parsing
        # For now, just raise NotImplementedError
        raise NotImplementedError(
            "datetime parsing not implemented in fallback. "
            "Use from_day_of_year_hour instead."
        )


@jax.jit
def get_declination(orbital_phase: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Returns angle between the Earth-Sun line and the Earth equatorial plane.
    
    Args:
        orbital_phase: Phase in Earth's orbit (radians)
        
    Returns:
        Solar declination (radians)
    """
    if HAS_JAX_SOLAR:
        return jax_solar.get_declination(orbital_phase)
    else:
        # Convert orbital phase to day of year
        day_of_year = orbital_phase * DAYS_PER_YEAR / (2 * jnp.pi) + 1
        return _solar_declination(day_of_year)


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
    if HAS_JAX_SOLAR:
        return jax_solar.get_hour_angle(synodic_phase, longitude)
    else:
        # Convert synodic phase to hour
        hour_utc = synodic_phase * 24.0 / (2 * jnp.pi)
        return _hour_angle(longitude, hour_utc)


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
    if HAS_JAX_SOLAR:
        return jax_solar.get_solar_sin_altitude(orbital_time, longitude, latitude)
    else:
        # Get declination and hour angle
        declination = get_declination(orbital_time.orbital_phase)
        ha = get_hour_angle(orbital_time.synodic_phase, longitude)
        
        # Convert latitude to radians
        lat_rad = jnp.deg2rad(latitude)
        
        # Calculate sine of altitude (= cosine of zenith)
        sin_alt = (jnp.sin(lat_rad) * jnp.sin(declination) +
                   jnp.cos(lat_rad) * jnp.cos(declination) * jnp.cos(ha))
        
        return sin_alt


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
    if HAS_JAX_SOLAR:
        return jax_solar.direct_solar_irradiance(orbital_phase, mean_irradiance, variation)
    else:
        # Convert orbital phase to day of year
        day_of_year = orbital_phase * DAYS_PER_YEAR / (2 * jnp.pi) + 1
        
        # Get Earth-Sun distance factor
        distance_factor = _earth_sun_distance(day_of_year)
        
        # Calculate direct irradiance
        # The variation is approximately mean_irradiance * (1/0.983^2 - 1/1.017^2)
        eccentricity_effect = distance_factor**2
        
        return mean_irradiance * eccentricity_effect


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
    if HAS_JAX_SOLAR:
        return jax_solar.radiation_flux(time, longitude, latitude, mean_irradiance, variation)
    else:
        # Handle different time input formats
        if isinstance(time, tuple):
            day_of_year, hour_utc = time
            orbital_time = OrbitalTime.from_day_of_year_hour(day_of_year, hour_utc)
        else:
            orbital_time = time
        
        # Get solar altitude sine (= zenith cosine)
        sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
        
        # Get direct solar irradiance
        irradiance = direct_solar_irradiance(orbital_time.orbital_phase, mean_irradiance, variation)
        
        # Calculate flux (set to zero when sun below horizon)
        flux = jnp.where(sin_altitude > 0, irradiance * sin_altitude, 0.0)
        
        return flux


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
    if HAS_JAX_SOLAR:
        return jax_solar.normalized_radiation_flux(time, longitude, latitude, mean_irradiance, variation)
    else:
        flux = radiation_flux(time, longitude, latitude, mean_irradiance, variation)
        return flux / mean_irradiance


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
    orbital_time = OrbitalTime.from_day_of_year_hour(day_of_year, hour_utc)
    
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
    if HAS_JAX_SOLAR:
        return "Using jax-solar implementation"
    else:
        return "Using fallback JAX implementation (jax-solar not available)"


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
    orbital_time = OrbitalTime.from_day_of_year_hour(day_of_year, hour_utc)
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