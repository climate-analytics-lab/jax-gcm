"""
Tests for radiation scheme comparison between ICON and RRTMGP.

Tests heating tendency comparison with identical atmospheric states.
Note: ICON radiation scheme may have issues on Apple Silicon due to complex number operations.
"""

import jax.numpy as jnp
import pytest
from jcm.physics.icon.radiation.radiation_scheme import radiation_scheme as radiation_scheme_icon
from jcm.physics.icon.radiation.radiation_scheme_rrtmgp import radiation_scheme_rrtmgp
from jcm.physics.icon.radiation.radiation_types import RadiationParameters
from jcm.physics.icon.radiation.radiation_scheme_test import (
    create_test_atmosphere,
    create_default_aerosol_data
)

def create_identical_atmosphere_state(nlev=10, lat=0.0, lon=0.0, day=172.0, time_hours=12.0):
    """Create identical atmospheric state for both radiation schemes with configurable conditions."""
    
    # Create clear-sky test atmosphere to avoid cloud optics issues
    atm = create_test_atmosphere(nlev=nlev)
    parameters = RadiationParameters.default()
    aerosol_data = create_default_aerosol_data(nlev=nlev, parameters=parameters)
    
    # Clear-sky conditions (no clouds to avoid complex number issues in cloud_optics)
    clear_sky_cloud_water = jnp.zeros_like(atm['cloud_water'])
    clear_sky_cloud_ice = jnp.zeros_like(atm['cloud_ice'])
    clear_sky_cloud_fraction = jnp.zeros_like(atm['cloud_fraction'])
    
    # Common inputs for both schemes
    common_inputs = {
        'temperature': atm['temperature'],
        'specific_humidity': atm['specific_humidity'],
        'pressure_levels': atm['pressure_levels'],
        'layer_thickness': atm['layer_thickness'],
        'air_density': atm['air_density'],
        'cloud_water': clear_sky_cloud_water,
        'cloud_ice': clear_sky_cloud_ice,
        'cloud_fraction': clear_sky_cloud_fraction,
        'day_of_year': day,
        'seconds_since_midnight': time_hours * 3600.0,
        'latitude': lat,
        'longitude': lon,
        'parameters': parameters,
        'aerosol_data': aerosol_data
    }
    
    return common_inputs


def test_icon_radiation_scheme():
    """Test ICON radiation scheme produces valid results."""
    
    inputs = create_identical_atmosphere_state(nlev=10)
    
    # Test ICON radiation scheme
    tendencies, diagnostics = radiation_scheme_icon(**inputs)
    
    # Basic validation
    assert tendencies.temperature_tendency.shape == (10,)
    assert not jnp.any(jnp.isnan(tendencies.temperature_tendency))
    assert jnp.all(jnp.isfinite(tendencies.temperature_tendency))

def test_rrtmgp_radiation_scheme():
    """Test RRTMGP radiation scheme produces valid results."""
    
    inputs = create_identical_atmosphere_state(nlev=10)
    
    # Test RRTMGP radiation scheme
    tendencies, diagnostics = radiation_scheme_rrtmgp(**inputs)
    
    # Basic validation
    assert tendencies.temperature_tendency.shape == (10,)
    assert not jnp.any(jnp.isnan(tendencies.temperature_tendency))
    assert jnp.all(jnp.isfinite(tendencies.temperature_tendency))

def test_rrtmgp_heating_rates():
    """Test RRTMGP radiation scheme heating rates in detail."""
    
    nlev = 10
    inputs = create_identical_atmosphere_state(nlev=nlev)
    
    # Run RRTMGP radiation scheme  
    rrtmgp_tendencies, rrtmgp_diagnostics = radiation_scheme_rrtmgp(**inputs)
    
    # Validation checks
    assert jnp.all(jnp.isfinite(rrtmgp_tendencies.temperature_tendency)), "RRTMGP heating rates contain non-finite values"
    assert not jnp.any(jnp.isnan(rrtmgp_tendencies.temperature_tendency)), "RRTMGP heating rates contain NaN"
    
    # Should have reasonable magnitudes (not zero everywhere)
    assert jnp.abs(jnp.mean(rrtmgp_tendencies.temperature_tendency)) > 1e-8, "RRTMGP heating rates too small"


def test_heating_tendency_comparison():
    """Compare heating tendencies between ICON and RRTMGP with identical atmospheric states."""
    
    nlev = 10
    inputs = create_identical_atmosphere_state(nlev=nlev)
    
    # Run ICON radiation scheme
    icon_tendencies, icon_diagnostics = radiation_scheme_icon(**inputs)
    
    # Run RRTMGP radiation scheme  
    rrtmgp_tendencies, rrtmgp_diagnostics = radiation_scheme_rrtmgp(**inputs)
    
    # Validation checks for both schemes
    assert jnp.all(jnp.isfinite(icon_tendencies.temperature_tendency)), "ICON heating rates contain non-finite values"
    assert jnp.all(jnp.isfinite(rrtmgp_tendencies.temperature_tendency)), "RRTMGP heating rates contain non-finite values"
    
    assert not jnp.any(jnp.isnan(icon_tendencies.temperature_tendency)), "ICON heating rates contain NaN"
    assert not jnp.any(jnp.isnan(rrtmgp_tendencies.temperature_tendency)), "RRTMGP heating rates contain NaN"
    
    # Both schemes should have reasonable magnitudes (not zero everywhere)
    assert jnp.abs(jnp.mean(icon_tendencies.temperature_tendency)) > 1e-8, "ICON heating rates too small"
    assert jnp.abs(jnp.mean(rrtmgp_tendencies.temperature_tendency)) > 1e-8, "RRTMGP heating rates too small"
    
    atol = 1e-2  # Absolute tolerance
    rtol = 0.1   # Relative tolerance
    
    # Assert heating tendencies are approximately equal
    assert icon_tendencies.temperature_tendency == pytest.approx(
        rrtmgp_tendencies.temperature_tendency, abs=atol, rel=rtol
    ), "ICON and RRTMGP heating rates differ beyond tolerance"
    
    # Also check individual heating components
    assert icon_tendencies.longwave_heating == pytest.approx(
        rrtmgp_tendencies.longwave_heating, abs=atol, rel=rtol
    ), "ICON and RRTMGP longwave heating rates differ beyond tolerance"
    
    assert icon_tendencies.shortwave_heating == pytest.approx(
        rrtmgp_tendencies.shortwave_heating, abs=atol, rel=rtol
    ), "ICON and RRTMGP shortwave heating rates differ beyond tolerance"


@pytest.mark.parametrize("lat,lon,day,time_hours", [
    (0.0, 0.0, 172.0, 12.0),      # Equator, noon, summer solstice
    (45.0, 0.0, 172.0, 12.0),     # Mid-latitude, noon, summer solstice
    (90.0, 0.0, 172.0, 12.0),     # North pole, noon, summer solstice
    (0.0, 0.0, 1.0, 12.0),        # Equator, noon, winter (Jan 1)
    (0.0, 0.0, 355.0, 12.0),      # Equator, noon, winter (Dec 21)
    (0.0, 0.0, 172.0, 6.0),       # Equator, sunrise, summer
    (0.0, 0.0, 172.0, 18.0),      # Equator, sunset, summer
    (0.0, 180.0, 172.0, 12.0),    # Equator, different longitude
    (-45.0, 90.0, 172.0, 12.0),   # Southern hemisphere, summer
])
def test_multiple_conditions(lat, lon, day, time_hours):
    """Test radiation schemes under various geographic and temporal conditions."""
    
    nlev = 10
    inputs = create_identical_atmosphere_state(nlev=nlev, lat=lat, lon=lon, day=day, time_hours=time_hours)
    
    # Run both radiation schemes
    icon_tendencies, icon_diagnostics = radiation_scheme_icon(**inputs)
    rrtmgp_tendencies, rrtmgp_diagnostics = radiation_scheme_rrtmgp(**inputs)
    
    # Basic validation
    assert jnp.all(jnp.isfinite(icon_tendencies.temperature_tendency)), f"ICON NaN/Inf at lat={lat}, lon={lon}, day={day}, time={time_hours}"
    assert jnp.all(jnp.isfinite(rrtmgp_tendencies.temperature_tendency)), f"RRTMGP NaN/Inf at lat={lat}, lon={lon}, day={day}, time={time_hours}"
    
    # Tolerance check (may need looser tolerances for extreme conditions)
    atol = 1e-2
    rtol = 0.1
    
    assert icon_tendencies.temperature_tendency == pytest.approx(
        rrtmgp_tendencies.temperature_tendency, abs=atol, rel=rtol
    ), f"Heating rates differ at lat={lat}, lon={lon}, day={day}, time={time_hours}"


if __name__ == "__main__":
    test_icon_radiation_scheme()
    test_rrtmgp_radiation_scheme()
    test_heating_tendency_comparison()