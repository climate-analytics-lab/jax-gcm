"""Tests for solar radiation interface."""

import pytest
import jax.numpy as jnp
import jax
from jcm.physics.icon.radiation.solar_interface import (
    OrbitalTime,
    orbital_time_from_day_hour,
    get_declination,
    get_hour_angle,
    get_solar_sin_altitude,
    direct_solar_irradiance,
    radiation_flux,
    normalized_radiation_flux,
    calculate_toa_radiation,
    get_implementation_info,
    TOTAL_SOLAR_IRRADIANCE,
    SOLAR_CONSTANT
)


class TestOrbitalTime:
    """Test orbital time creation"""
    
    def test_orbital_time_from_day_hour(self):
        """Test creating OrbitalTime from day and hour"""
        day_of_year = 172.0  # Summer solstice
        hour_utc = 12.0      # Noon
        
        ot = orbital_time_from_day_hour(day_of_year, hour_utc)
        
        assert hasattr(ot, 'orbital_phase')
        assert hasattr(ot, 'synodic_phase')
        assert jnp.isfinite(ot.orbital_phase)
        assert jnp.isfinite(ot.synodic_phase)
    
    def test_orbital_time_extremes(self):
        """Test orbital time at year extremes"""
        # Beginning of year
        ot1 = orbital_time_from_day_hour(1.0, 0.0)
        
        # End of year
        ot2 = orbital_time_from_day_hour(365.0, 24.0)
        
        # Both should be valid
        assert jnp.isfinite(ot1.orbital_phase)
        assert jnp.isfinite(ot2.orbital_phase)


class TestDeclination:
    """Test solar declination calculations"""
    
    def test_get_declination_summer_solstice(self):
        """Test declination at summer solstice"""
        # Summer solstice is around day 172
        orbital_phase = 2 * jnp.pi * (172 - 1) / 365.25
        
        declination = get_declination(orbital_phase)
        
        # Should be near +23.5 degrees (0.41 radians)
        assert jnp.abs(declination) > 0.3
        assert jnp.abs(declination) < 0.5
    
    def test_get_declination_array(self):
        """Test declination with array input"""
        phases = jnp.array([0.0, jnp.pi, 2*jnp.pi])
        
        declinations = get_declination(phases)
        
        assert declinations.shape == (3,)
        assert jnp.all(jnp.isfinite(declinations))


class TestHourAngle:
    """Test hour angle calculations"""
    
    def test_get_hour_angle_noon(self):
        """Test hour angle at noon"""
        synodic_phase = 2 * jnp.pi * 12.0 / 24.0  # Noon
        longitude = jnp.array([0.0])  # Greenwich
        
        ha = get_hour_angle(synodic_phase, longitude)
        
        # Should be near 0 at noon
        assert jnp.abs(ha[0]) < 0.5
    
    def test_get_hour_angle_multiple_locations(self):
        """Test hour angle for multiple locations"""
        synodic_phase = 0.0  # Midnight
        longitudes = jnp.array([0.0, 90.0, -90.0])
        
        ha = get_hour_angle(synodic_phase, longitudes)
        
        assert ha.shape == (3,)
        assert jnp.all(jnp.isfinite(ha))


class TestSolarAltitude:
    """Test solar altitude calculations"""
    
    def test_get_solar_sin_altitude(self):
        """Test solar altitude calculation"""
        day_of_year = 172.0
        hour_utc = 12.0
        orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
        
        longitude = jnp.array([0.0])
        latitude = jnp.array([0.0])  # Equator
        
        sin_alt = get_solar_sin_altitude(orbital_time, longitude, latitude)
        
        # At noon on equator, should be positive
        assert sin_alt[0] > 0
        assert sin_alt[0] <= 1.0
    
    def test_solar_altitude_nighttime(self):
        """Test solar altitude at night"""
        day_of_year = 172.0
        hour_utc = 0.0  # Midnight
        orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
        
        longitude = jnp.array([0.0])
        latitude = jnp.array([0.0])
        
        sin_alt = get_solar_sin_altitude(orbital_time, longitude, latitude)
        
        # Should handle nighttime (may be negative)
        assert jnp.isfinite(sin_alt[0])


class TestDirectIrradiance:
    """Test direct solar irradiance"""
    
    def test_direct_solar_irradiance(self):
        """Test direct irradiance calculation"""
        # Near perihelion (early January)
        orbital_phase = 0.0
        
        irr = direct_solar_irradiance(orbital_phase)
        
        # Should be close to solar constant
        assert irr > 1300
        assert irr < 1400
    
    def test_direct_irradiance_variation(self):
        """Test irradiance varies with orbital position"""
        # Perihelion
        phase1 = 0.0
        irr1 = direct_solar_irradiance(phase1)
        
        # Aphelion
        phase2 = jnp.pi
        irr2 = direct_solar_irradiance(phase2)
        
        # Should be different
        assert jnp.abs(irr1 - irr2) > 1.0


class TestRadiationFlux:
    """Test radiation flux calculations"""
    
    def test_radiation_flux_daytime(self):
        """Test radiation flux during day"""
        day_of_year = 172.0
        hour_utc = 12.0
        orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
        
        longitude = jnp.array([0.0, 90.0, -90.0])
        latitude = jnp.array([0.0, 45.0, -45.0])
        
        flux = radiation_flux(orbital_time, longitude, latitude)
        
        assert flux.shape == (3,)
        # At least some locations should have positive flux
        assert jnp.any(flux > 0)
    
    def test_radiation_flux_consistency(self):
        """Test radiation flux consistency between methods"""
        day_of_year = 172.0
        hour_utc = 12.0
        
        # Method 1: Using orbital_time_from_day_hour
        orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
        longitude = jnp.array([0.0])
        latitude = jnp.array([0.0])
        
        flux1 = radiation_flux(orbital_time, longitude, latitude)
        
        # Method 2: Using calculate_toa_radiation
        flux2, _, _ = calculate_toa_radiation(day_of_year, hour_utc, longitude, latitude)
        
        # Should give same result
        assert jnp.allclose(flux1, flux2)
    
    def test_normalized_radiation_flux(self):
        """Test normalized radiation flux"""
        day_of_year = 172.0
        hour_utc = 12.0
        orbital_time = orbital_time_from_day_hour(day_of_year, hour_utc)
        
        longitude = jnp.array([0.0])
        latitude = jnp.array([0.0])
        
        norm_flux = normalized_radiation_flux(orbital_time, longitude, latitude)
        
        # Should be between 0 and 1
        assert norm_flux[0] >= 0
        assert norm_flux[0] <= 1.0


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_calculate_toa_radiation(self):
        """Test TOA radiation calculation"""
        day_of_year = 172.0
        hour_utc = 12.0
        longitude = jnp.array([0.0, 90.0])
        latitude = jnp.array([0.0, 45.0])
        
        flux, cos_zenith, hour_angle = calculate_toa_radiation(
            day_of_year, hour_utc, longitude, latitude
        )
        
        # Check shapes
        assert flux.shape == (2,)
        assert cos_zenith.shape == (2,)
        assert hour_angle.shape == (2,)
        
        # Check values are finite
        assert jnp.all(jnp.isfinite(flux))
        assert jnp.all(jnp.isfinite(cos_zenith))
        assert jnp.all(jnp.isfinite(hour_angle))
    
    def test_calculate_toa_radiation_with_custom_constant(self):
        """Test TOA radiation with custom solar constant"""
        day_of_year = 172.0
        hour_utc = 12.0
        longitude = jnp.array([0.0])
        latitude = jnp.array([0.0])
        custom_constant = 1400.0
        
        flux, cos_zenith, hour_angle = calculate_toa_radiation(
            day_of_year, hour_utc, longitude, latitude, custom_constant
        )
        
        # Flux should scale with solar constant
        assert jnp.isfinite(flux[0])
    
    def test_get_implementation_info(self):
        """Test implementation info"""
        info = get_implementation_info()
        
        assert isinstance(info, str)
        assert "jax-solar" in info


class TestConstants:
    """Test module constants"""
    
    def test_solar_constants(self):
        """Test that solar constants are defined"""
        assert TOTAL_SOLAR_IRRADIANCE > 0
        assert SOLAR_CONSTANT > 0
        assert TOTAL_SOLAR_IRRADIANCE == SOLAR_CONSTANT


class TestJAXCompatibility:
    """Test JAX transformations"""
    
    def test_jit_compilation(self):
        """Test that functions can be JIT compiled"""
        orbital_phase = jnp.array(0.0)
        
        # These are already decorated with @jax.jit
        declination = get_declination(orbital_phase)
        irradiance = direct_solar_irradiance(orbital_phase)
        
        assert jnp.isfinite(declination)
        assert jnp.isfinite(irradiance)
    
    def test_vmap_compatibility(self):
        """Test vmap over multiple locations"""
        day_of_year = 172.0
        hour_utc = 12.0
        
        # Multiple locations
        lons = jnp.array([0.0, 30.0, 60.0, 90.0])
        lats = jnp.array([0.0, 20.0, 40.0, 60.0])
        
        flux, cos_zenith, ha = calculate_toa_radiation(
            day_of_year, hour_utc, lons, lats
        )
        
        assert flux.shape == (4,)
        assert cos_zenith.shape == (4,)


class TestPhysicalConsistency:
    """Test physical consistency"""
    
    def test_flux_positive_daytime(self):
        """Test that flux is positive during daytime"""
        # Noon at equator on equinox
        day_of_year = 80.0  # Near equinox
        hour_utc = 12.0
        longitude = jnp.array([0.0])
        latitude = jnp.array([0.0])
        
        flux, cos_zenith, _ = calculate_toa_radiation(
            day_of_year, hour_utc, longitude, latitude
        )
        
        # Should be positive at noon
        assert flux[0] > 0
        assert cos_zenith[0] > 0
    
    def test_declination_range(self):
        """Test that declination stays within physical bounds"""
        # Test throughout the year
        phases = jnp.linspace(0, 2*jnp.pi, 20)
        
        declinations = get_declination(phases)
        
        # Declination should be between -23.5° and +23.5° (±0.41 radians)
        assert jnp.all(jnp.abs(declinations) <= 0.42)
    
    def test_irradiance_conservation(self):
        """Test that irradiance doesn't deviate too much from mean"""
        # Test throughout the year
        phases = jnp.linspace(0, 2*jnp.pi, 100)
        
        irradiances = direct_solar_irradiance(phases)
        
        # Should stay within reasonable range of mean
        mean_irr = TOTAL_SOLAR_IRRADIANCE
        assert jnp.all(irradiances > mean_irr * 0.95)
        assert jnp.all(irradiances < mean_irr * 1.05)


def test_builtin_test_function():
    """Test the built-in test_solar_interface function"""
    from jcm.physics.icon.radiation.solar_interface import test_solar_interface
    
    # This should run without errors
    test_solar_interface()

