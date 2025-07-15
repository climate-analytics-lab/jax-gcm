"""
Simple boundary conditions for ICON physics

This module provides basic boundary conditions and external forcings including:
- Solar irradiance variations
- Greenhouse gas concentrations
- Basic land surface properties
- Sea surface temperature

Date: 2025-01-15
"""

import jax.numpy as jnp
import jax
from typing import NamedTuple, Tuple, Optional
import tree_math

from ..constants.physical_constants import sbc, solc


@tree_math.struct
class BoundaryConditionParameters:
    """Configuration parameters for boundary conditions"""
    
    # Solar parameters
    solar_constant: float              # Solar constant (W/m²)
    solar_variability: float           # Solar variability amplitude (fraction)
    solar_cycle_period: float          # Solar cycle period (years)
    
    # Greenhouse gas parameters
    co2_reference: float               # Reference CO2 concentration (ppmv)
    ch4_reference: float               # Reference CH4 concentration (ppbv)
    n2o_reference: float               # Reference N2O concentration (ppbv)
    
    # Land surface parameters
    land_albedo_vis: float             # Land albedo visible
    land_albedo_nir: float             # Land albedo near-infrared
    land_emissivity: float             # Land emissivity
    
    # Ocean parameters
    ocean_albedo_vis: float            # Ocean albedo visible
    ocean_albedo_nir: float            # Ocean albedo near-infrared
    ocean_emissivity: float            # Ocean emissivity
    
    # Sea ice parameters
    seaice_albedo_vis: float           # Sea ice albedo visible
    seaice_albedo_nir: float           # Sea ice albedo near-infrared
    seaice_emissivity: float           # Sea ice emissivity
    
    @classmethod
    def default(cls) -> 'BoundaryConditionParameters':
        """Return default boundary condition parameters"""
        return cls(
            solar_constant=jnp.array(solc),         # W/m²
            solar_variability=jnp.array(0.001),    # 0.1% variation
            solar_cycle_period=jnp.array(11.0),    # 11 years
            co2_reference=jnp.array(420.0),        # ppmv
            ch4_reference=jnp.array(1900.0),       # ppbv
            n2o_reference=jnp.array(335.0),        # ppbv
            land_albedo_vis=jnp.array(0.15),       # Typical land albedo
            land_albedo_nir=jnp.array(0.25),
            land_emissivity=jnp.array(0.95),
            ocean_albedo_vis=jnp.array(0.05),      # Dark ocean
            ocean_albedo_nir=jnp.array(0.05),
            ocean_emissivity=jnp.array(0.98),
            seaice_albedo_vis=jnp.array(0.80),     # Bright ice
            seaice_albedo_nir=jnp.array(0.70),
            seaice_emissivity=jnp.array(0.95)
        )


class BoundaryConditionState(NamedTuple):
    """Boundary condition state variables"""
    
    # Solar forcing
    solar_irradiance: jnp.ndarray       # Top-of-atmosphere solar irradiance (W/m²)
    solar_zenith_angle: jnp.ndarray     # Solar zenith angle (radians)
    day_of_year: jnp.ndarray           # Day of year (1-365)
    
    # Greenhouse gas concentrations
    co2_concentration: jnp.ndarray      # CO2 concentration (ppmv)
    ch4_concentration: jnp.ndarray      # CH4 concentration (ppbv)
    n2o_concentration: jnp.ndarray      # N2O concentration (ppbv)
    
    # Surface properties
    surface_albedo_vis: jnp.ndarray     # Surface albedo visible [ncols]
    surface_albedo_nir: jnp.ndarray     # Surface albedo near-infrared [ncols]
    surface_emissivity: jnp.ndarray     # Surface emissivity [ncols]
    
    # Sea surface temperature (if available)
    sea_surface_temperature: jnp.ndarray  # SST (K) [ncols]
    sea_ice_fraction: jnp.ndarray       # Sea ice fraction (0-1) [ncols]


def compute_solar_zenith_angle(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    day_of_year: float,
    time_of_day: float
) -> jnp.ndarray:
    """
    Compute solar zenith angle
    
    Args:
        latitude: Latitude (radians) [ncols]
        longitude: Longitude (radians) [ncols]
        day_of_year: Day of year (1-365)
        time_of_day: Time of day (hours, 0-24)
        
    Returns:
        Solar zenith angle (radians) [ncols]
    """
    # Solar declination (simplified)
    declination = 23.45 * jnp.pi / 180.0 * jnp.sin(2.0 * jnp.pi * (day_of_year - 81.0) / 365.0)
    
    # Hour angle
    hour_angle = (time_of_day - 12.0) * jnp.pi / 12.0
    
    # Solar zenith angle
    cos_zenith = (jnp.sin(latitude) * jnp.sin(declination) + 
                  jnp.cos(latitude) * jnp.cos(declination) * jnp.cos(hour_angle))
    
    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, -1.0, 1.0))
    
    return zenith_angle


def compute_solar_irradiance(
    solar_zenith_angle: jnp.ndarray,
    day_of_year: float,
    year: float,
    config: BoundaryConditionParameters
) -> jnp.ndarray:
    """
    Compute top-of-atmosphere solar irradiance
    
    Args:
        solar_zenith_angle: Solar zenith angle (radians) [ncols]
        day_of_year: Day of year (1-365)
        year: Year (for solar variability)
        config: Boundary condition configuration
        
    Returns:
        Solar irradiance (W/m²) [ncols]
    """
    # Earth-Sun distance variation
    earth_sun_distance = 1.0 - 0.0167 * jnp.cos(2.0 * jnp.pi * (day_of_year - 3.0) / 365.0)
    
    # Solar variability (simplified 11-year cycle)
    solar_variation = 1.0 + config.solar_variability * jnp.sin(2.0 * jnp.pi * year / config.solar_cycle_period)
    
    # Solar irradiance
    solar_irradiance = (config.solar_constant * solar_variation / earth_sun_distance**2 * 
                       jnp.maximum(jnp.cos(solar_zenith_angle), 0.0))
    
    return solar_irradiance


def compute_surface_properties(
    land_fraction: jnp.ndarray,
    sea_ice_fraction: jnp.ndarray,
    config: BoundaryConditionParameters
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute surface optical properties based on surface type
    
    Args:
        land_fraction: Land fraction (0-1) [ncols]
        sea_ice_fraction: Sea ice fraction (0-1) [ncols]
        config: Boundary condition configuration
        
    Returns:
        Tuple of (albedo_vis, albedo_nir, emissivity) [ncols]
    """
    # Ocean fraction
    ocean_fraction = 1.0 - land_fraction - sea_ice_fraction
    ocean_fraction = jnp.maximum(ocean_fraction, 0.0)
    
    # Weighted average of surface properties
    albedo_vis = (land_fraction * config.land_albedo_vis +
                  ocean_fraction * config.ocean_albedo_vis +
                  sea_ice_fraction * config.seaice_albedo_vis)
    
    albedo_nir = (land_fraction * config.land_albedo_nir +
                  ocean_fraction * config.ocean_albedo_nir +
                  sea_ice_fraction * config.seaice_albedo_nir)
    
    emissivity = (land_fraction * config.land_emissivity +
                  ocean_fraction * config.ocean_emissivity +
                  sea_ice_fraction * config.seaice_emissivity)
    
    return albedo_vis, albedo_nir, emissivity


def simple_boundary_conditions(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    land_fraction: jnp.ndarray,
    day_of_year: float,
    time_of_day: float,
    year: float,
    sea_surface_temperature: Optional[jnp.ndarray] = None,
    sea_ice_fraction: Optional[jnp.ndarray] = None,
    config: Optional[BoundaryConditionParameters] = None
) -> BoundaryConditionState:
    """
    Compute simple boundary conditions
    
    Args:
        latitude: Latitude (radians) [ncols]
        longitude: Longitude (radians) [ncols]
        land_fraction: Land fraction (0-1) [ncols]
        day_of_year: Day of year (1-365)
        time_of_day: Time of day (hours, 0-24)
        year: Year (for solar variability)
        sea_surface_temperature: SST (K) [ncols], optional
        sea_ice_fraction: Sea ice fraction (0-1) [ncols], optional
        config: Boundary condition configuration
        
    Returns:
        Boundary condition state
    """
    if config is None:
        config = BoundaryConditionParameters.default()
    
    ncols = latitude.shape[0]
    
    # Default sea ice fraction if not provided
    if sea_ice_fraction is None:
        sea_ice_fraction = jnp.zeros(ncols)
    
    # Default SST if not provided (climatological)
    if sea_surface_temperature is None:
        # Simple SST based on latitude (warmer at equator)
        sea_surface_temperature = 288.0 - 30.0 * jnp.abs(latitude) / (jnp.pi / 2.0)
        sea_surface_temperature = jnp.maximum(sea_surface_temperature, 271.0)  # Freezing point
    
    # Compute solar zenith angle
    solar_zenith_angle = compute_solar_zenith_angle(
        latitude, longitude, day_of_year, time_of_day
    )
    
    # Compute solar irradiance
    solar_irradiance = compute_solar_irradiance(
        solar_zenith_angle, day_of_year, year, config
    )
    
    # Compute surface properties
    albedo_vis, albedo_nir, emissivity = compute_surface_properties(
        land_fraction, sea_ice_fraction, config
    )
    
    # Greenhouse gas concentrations (constant for now)
    co2_concentration = jnp.ones(ncols) * config.co2_reference
    ch4_concentration = jnp.ones(ncols) * config.ch4_reference
    n2o_concentration = jnp.ones(ncols) * config.n2o_reference
    
    return BoundaryConditionState(
        solar_irradiance=solar_irradiance,
        solar_zenith_angle=solar_zenith_angle,
        day_of_year=jnp.ones(ncols) * day_of_year,
        co2_concentration=co2_concentration,
        ch4_concentration=ch4_concentration,
        n2o_concentration=n2o_concentration,
        surface_albedo_vis=albedo_vis,
        surface_albedo_nir=albedo_nir,
        surface_emissivity=emissivity,
        sea_surface_temperature=sea_surface_temperature,
        sea_ice_fraction=sea_ice_fraction
    )


def create_idealized_boundary_conditions(
    ncols: int,
    day_of_year: float = 180.0,
    time_of_day: float = 12.0,
    year: float = 2020.0,
    config: Optional[BoundaryConditionParameters] = None
) -> BoundaryConditionState:
    """
    Create idealized boundary conditions for testing
    
    Args:
        ncols: Number of columns
        day_of_year: Day of year (1-365)
        time_of_day: Time of day (hours, 0-24)
        year: Year
        config: Boundary condition configuration
        
    Returns:
        Idealized boundary condition state
    """
    if config is None:
        config = BoundaryConditionParameters.default()
    
    # Create simple idealized geography
    latitude = jnp.linspace(-jnp.pi/2, jnp.pi/2, ncols)
    longitude = jnp.zeros(ncols)
    land_fraction = jnp.where(jnp.abs(latitude) < jnp.pi/4, 0.3, 0.1)  # More land in tropics
    
    return simple_boundary_conditions(
        latitude, longitude, land_fraction, day_of_year, time_of_day, year,
        config=config
    )