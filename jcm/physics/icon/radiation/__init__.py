"""
ICON radiation scheme for JAX-GCM

This module implements radiation physics following the ICON model approach.
It includes both shortwave (solar) and longwave (thermal) radiation.

The implementation is fully compatible with JAX transformations (jit, vmap, grad).

For solar calculations, we support two implementations:
1. jax-solar (preferred, requires Python 3.11+)
2. Fallback JAX implementation

Date: 2025-01-10
"""

# Import solar interface using jax-solar
from .solar_interface import (
    calculate_toa_radiation,
    radiation_flux,
    normalized_radiation_flux,
    TOTAL_SOLAR_IRRADIANCE,
    SOLAR_IRRADIANCE_VARIATION,
    get_implementation_info,
    get_declination
)

_SOLAR_IMPLEMENTATION = get_implementation_info()

# Create compatible functions for backward compatibility
def calculate_solar_radiation_gcm(
    day_of_year, seconds_since_midnight, longitude, latitude, solar_constant=TOTAL_SOLAR_IRRADIANCE
):
    """Calculate solar radiation for GCM (backward compatible interface)"""
    hour_utc = seconds_since_midnight / 3600.0
    flux, cos_zenith, _ = calculate_toa_radiation(
        day_of_year, hour_utc, longitude, latitude, solar_constant
    )
    return flux, cos_zenith

def cosine_solar_zenith_angle(day_of_year, seconds_since_midnight, longitude, latitude):
    """Calculate cosine of solar zenith angle (backward compatible)"""
    hour_utc = seconds_since_midnight / 3600.0
    _, cos_zenith, _ = calculate_toa_radiation(day_of_year, hour_utc, longitude, latitude)
    return cos_zenith

def top_of_atmosphere_flux(day_of_year, seconds_since_midnight, longitude, latitude):
    """Calculate top-of-atmosphere flux (backward compatible)"""
    hour_utc = seconds_since_midnight / 3600.0
    flux, _, _ = calculate_toa_radiation(day_of_year, hour_utc, longitude, latitude)
    return flux

def daylight_fraction(day_of_year, longitude, latitude):
    """Calculate daylight fraction"""
    import jax.numpy as jnp
    
    # Calculate solar declination
    orbital_phase = 2 * jnp.pi * (day_of_year - 1) / 365.25
    declination = get_declination(orbital_phase)
    
    # Convert latitude to radians  
    lat_rad = jnp.deg2rad(latitude)
    
    # Calculate hour angle at sunrise/sunset
    # cos(hour_angle) = -tan(lat) * tan(declination)
    cos_ha = -jnp.tan(lat_rad) * jnp.tan(declination)
    
    # Handle polar day/night cases
    cos_ha = jnp.clip(cos_ha, -1.0, 1.0)
    
    # Hour angle at sunrise (radians)
    ha_sunrise = jnp.arccos(cos_ha)
    
    # Daylight fraction = daylight hours / 24 hours
    daylight_hours = 2 * ha_sunrise * 12 / jnp.pi  # Convert radians to hours
    fraction = daylight_hours / 24.0
    
    return fraction

# Import radiation types
from .radiation_types import (
    RadiationParameters,
    RadiationState,
    RadiationFluxes,
    RadiationTendencies,
    OpticalProperties
)

# Import radiation components
from .gas_optics import (
    gas_optical_depth_lw,
    gas_optical_depth_sw,
    rayleigh_optical_depth
)

from .cloud_optics import (
    cloud_optics,
    effective_radius_liquid,
    effective_radius_ice
)

from .planck import (
    planck_bands_lw,
    total_thermal_emission,
    planck_function_wavenumber
)

from .two_stream import (
    longwave_fluxes,
    shortwave_fluxes,
    flux_to_heating_rate
)

# Version info
__version__ = "0.1.0"


def get_solar_implementation():
    """Return which solar implementation is being used"""
    return _SOLAR_IMPLEMENTATION


# Import main radiation interface
from .radiation_scheme import (
    radiation_scheme,
    prepare_radiation_state
)
from .radiation_scheme_rrtmgp import (
    radiation_scheme_rrtmgp
)
from ..icon_physics_data import RadiationData

__all__ = [
    # Main interface
    "radiation_scheme",
    "radiation_scheme_rrtmgp",
    "RadiationData",
    "prepare_radiation_state",
    
    # Solar functions
    "calculate_solar_radiation_gcm",
    "cosine_solar_zenith_angle", 
    "top_of_atmosphere_flux",
    "daylight_fraction",
    "TOTAL_SOLAR_IRRADIANCE",
    "SOLAR_IRRADIANCE_VARIATION",
    "get_solar_implementation",
    
    # Types
    "RadiationParameters",
    "RadiationState",
    "RadiationFluxes", 
    "RadiationTendencies",
    "OpticalProperties",
    
    # Gas optics
    "gas_optical_depth_lw",
    "gas_optical_depth_sw",
    "rayleigh_optical_depth",
    
    # Cloud optics
    "cloud_optics",
    "effective_radius_liquid",
    "effective_radius_ice",
    
    # Planck functions
    "planck_bands_lw",
    "total_thermal_emission",
    "planck_function_wavenumber",
    
    # Radiative transfer
    "longwave_fluxes",
    "shortwave_fluxes",
    "flux_to_heating_rate",
]