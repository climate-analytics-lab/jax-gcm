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

# Try to import the jax-solar based implementation
try:
    from .solar_jax import (
        calculate_solar_radiation_gcm,
        cosine_solar_zenith_angle,
        top_of_atmosphere_flux,
        daylight_fraction,
        TOTAL_SOLAR_IRRADIANCE,
        SOLAR_IRRADIANCE_VARIATION
    )
    _SOLAR_IMPLEMENTATION = "jax-solar"
except ImportError:
    # Fall back to our implementation via solar_interface
    from .solar_interface import (
        calculate_toa_radiation,
        radiation_flux,
        normalized_radiation_flux,
        TOTAL_SOLAR_IRRADIANCE,
        SOLAR_IRRADIANCE_VARIATION,
        get_implementation_info
    )
    # Import directly from solar module for compatibility
    from .solar import (
        cosine_solar_zenith_angle,
        daylight_fraction,
        top_of_atmosphere_flux
    )
    _SOLAR_IMPLEMENTATION = get_implementation_info()
    
    # Create compatible calculate_solar_radiation_gcm
    def calculate_solar_radiation_gcm(
        day_of_year, seconds_since_midnight, longitude, latitude, solar_constant=TOTAL_SOLAR_IRRADIANCE
    ):
        hour_utc = seconds_since_midnight / 3600.0
        flux, cos_zenith, _ = calculate_toa_radiation(
            day_of_year, hour_utc, longitude, latitude, solar_constant
        )
        return flux, cos_zenith

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
    planck_bands,
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


__all__ = [
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
    "planck_bands",
    "total_thermal_emission",
    "planck_function_wavenumber",
    
    # Radiative transfer
    "longwave_fluxes",
    "shortwave_fluxes",
    "flux_to_heating_rate",
]