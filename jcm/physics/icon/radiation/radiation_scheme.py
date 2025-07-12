"""
Main radiation scheme interface for ICON physics

This module provides the main entry point for radiation calculations,
coordinating shortwave and longwave radiation computations.

Date: 2025-01-10
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, NamedTuple
from functools import partial

from .radiation_types import (
    RadiationParameters, 
    RadiationState,
    RadiationTendencies,
    OpticalProperties
)
from ..icon_physics_data import RadiationData

from . import (
    calculate_solar_radiation_gcm,
    gas_optical_depth_lw,
    gas_optical_depth_sw,
    cloud_optics,
    planck_bands,
    longwave_fluxes,
    shortwave_fluxes,
    flux_to_heating_rate,
)
from ..unit_conversions import (
    convert_surface_pressure,
    calculate_pressure_levels,
    geopotential_to_height
)
from ..constants import physical_constants


def prepare_radiation_state(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    height_levels: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    cos_zenith: float,
    ozone_vmr: Optional[jnp.ndarray] = None
) -> RadiationState:
    """
    Prepare radiation state from physics state variables.
    
    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure_levels: Pressure (Pa) [nlev]
        height_levels: Height (m) [nlev]
        cloud_water: Cloud water content (kg/kg) [nlev]
        cloud_ice: Cloud ice content (kg/kg) [nlev]
        cloud_fraction: Cloud fraction (0-1) [nlev]
        cos_zenith: Cosine of solar zenith angle
        ozone_vmr: Ozone volume mixing ratio [nlev]
        
    Returns:
        RadiationState ready for radiation calculations
    """
    nlev = temperature.shape[0]
    
    # Convert specific humidity to volume mixing ratio
    # q/(1-q) * Md/Mv where Md/Mv = 29/18 = 1.608
    h2o_vmr = specific_humidity / (1 - specific_humidity) * 1.608
    
    # Default ozone profile if not provided (simplified)
    if ozone_vmr is None:
        # Simple ozone profile peaking in stratosphere
        p_mb = pressure_levels / 100.0  # Convert to mb
        ozone_vmr = jnp.where(
            p_mb < 100,  # Stratosphere
            5e-6 * jnp.exp(-((jnp.log(p_mb) - jnp.log(30)) ** 2) / 2),
            1e-6  # Troposphere
        )
    
    # Calculate air density
    air_density = pressure_levels / (physical_constants.rd * temperature)
    
    # Layer thickness from height levels
    layer_thickness = jnp.zeros(nlev)
    layer_thickness = layer_thickness.at[:-1].set(height_levels[1:] - height_levels[:-1])  # Positive thickness
    layer_thickness = layer_thickness.at[-1].set(layer_thickness[-2])  # Assume same as layer above
    
    # Convert cloud water/ice from kg/kg to kg/m²
    # cloud_path = mixing_ratio * air_density * layer_thickness
    cloud_water_path = cloud_water * air_density * layer_thickness
    cloud_ice_path = cloud_ice * air_density * layer_thickness
    
    # Interface pressures
    pressure_interfaces = jnp.zeros(nlev + 1)
    pressure_interfaces = pressure_interfaces.at[1:-1].set(
        0.5 * (pressure_levels[:-1] + pressure_levels[1:])
    )
    # TOA should be lower pressure than first layer
    pressure_interfaces = pressure_interfaces.at[0].set(pressure_levels[0] * 0.1)  # Much lower for TOA
    # Surface should be higher pressure than last layer  
    pressure_interfaces = pressure_interfaces.at[-1].set(pressure_levels[-1] * 1.1)  # Slight increase for surface
    
    return RadiationState(
        cos_zenith=jnp.array([cos_zenith]),
        daylight_fraction=jnp.array([jnp.where(cos_zenith > 0, 1.0, 0.0)]),
        temperature=temperature,
        pressure=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        h2o_vmr=h2o_vmr,
        o3_vmr=ozone_vmr,
        cloud_fraction=cloud_fraction,
        cloud_water_path=cloud_water_path,
        cloud_ice_path=cloud_ice_path,
        surface_temperature=jnp.array([temperature[-1]]),  # Bottom level temperature
        surface_albedo_vis=jnp.array([0.15]),  # Visible albedo
        surface_albedo_nir=jnp.array([0.15]),  # Near-IR albedo
        surface_emissivity=jnp.array([0.98])
    )


def radiation_scheme(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    surface_pressure: jnp.ndarray,
    geopotential: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    day_of_year: float,
    seconds_since_midnight: float,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6
) -> Tuple[RadiationTendencies, RadiationData]:
    """
    Main radiation scheme entry point.
    
    This function coordinates the calculation of shortwave and longwave
    radiation, returning heating tendencies and diagnostic fluxes.
    
    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        surface_pressure: Surface pressure (normalized)
        geopotential: Geopotential (m²/s²) [nlev]
        cloud_water: Cloud water content (kg/kg) [nlev]
        cloud_ice: Cloud ice content (kg/kg) [nlev]
        cloud_fraction: Cloud fraction (0-1) [nlev]
        day_of_year: Day of year (1-365)
        seconds_since_midnight: Seconds since midnight UTC
        latitude: Latitude (degrees)
        longitude: Longitude (degrees)
        parameters: Radiation parameters (uses defaults if None)
        ozone_vmr: Ozone volume mixing ratio [nlev]
        co2_vmr: CO2 volume mixing ratio
        
    Returns:
        Tuple of (radiation tendencies, radiation diagnostics)
    """

    nlev = temperature.shape[0]
    
    # Create simple sigma levels for pressure calculation
    # Note: surface_pressure is a scalar, need to make it an array for calculate_pressure_levels
    sigma_levels = jnp.linspace(0.0, 1.0, nlev)
    surface_pressure_array = jnp.array([surface_pressure])  # Don't convert yet, let calculate_pressure_levels do it
    pressure_levels = calculate_pressure_levels(surface_pressure_array, sigma_levels)[:, 0]  # Take single column
    height_levels = geopotential_to_height(geopotential)
    
    # 1. Solar radiation calculations
    toa_flux, cos_zenith = calculate_solar_radiation_gcm(
        day_of_year=day_of_year,
        seconds_since_midnight=seconds_since_midnight,
        longitude=longitude,
        latitude=latitude,
        solar_constant=parameters.solar_constant
    )
    
    # Prepare radiation state
    rad_state = prepare_radiation_state(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        height_levels=height_levels,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        cos_zenith=cos_zenith,
        ozone_vmr=ozone_vmr
    )
    
    # Calculate layer properties
    air_density = pressure_levels / (physical_constants.rd * temperature)
    layer_thickness = jnp.zeros(nlev)
    layer_thickness = layer_thickness.at[:-1].set(height_levels[1:] - height_levels[:-1])  # Positive thickness
    layer_thickness = layer_thickness.at[-1].set(layer_thickness[-2])
    
    layer_properties = {
        'thickness': layer_thickness,
        'density': air_density
    }
    
    # 2. Full radiation calculation using ICON radiation components
    
    # Calculate gas optical depths
    gas_tau_lw = gas_optical_depth_lw(
        temperature=temperature,
        pressure=pressure_levels,
        h2o_vmr=rad_state.h2o_vmr,
        o3_vmr=rad_state.o3_vmr,
        co2_vmr=co2_vmr,
        layer_thickness=layer_properties['thickness'],
        air_density=layer_properties['density']
        # n_bands will use default value of 3
    )
    
    gas_tau_sw = gas_optical_depth_sw(
        pressure=pressure_levels,
        h2o_vmr=rad_state.h2o_vmr,
        o3_vmr=rad_state.o3_vmr,
        layer_thickness=layer_properties['thickness'],
        air_density=layer_properties['density'],
        cos_zenith=cos_zenith
        # n_bands will use default value of 2
    )
    
    # Calculate cloud optical properties
    cloud_sw_optics, cloud_lw_optics = cloud_optics(
        cloud_water_path=rad_state.cloud_water_path,
        cloud_ice_path=rad_state.cloud_ice_path,
        temperature=temperature,
        n_sw_bands=parameters.n_sw_bands,
        n_lw_bands=parameters.n_lw_bands
    )
    
    # Combine gas and cloud optical depths
    from .radiation_types import OpticalProperties
    
    total_tau_lw = gas_tau_lw + cloud_lw_optics.optical_depth
    lw_optics = OpticalProperties(
        optical_depth=total_tau_lw,
        single_scatter_albedo=cloud_lw_optics.single_scatter_albedo,
        asymmetry_factor=cloud_lw_optics.asymmetry_factor
    )
    
    total_tau_sw = gas_tau_sw + cloud_sw_optics.optical_depth
    sw_optics = OpticalProperties(
        optical_depth=total_tau_sw,
        single_scatter_albedo=cloud_sw_optics.single_scatter_albedo,
        asymmetry_factor=cloud_sw_optics.asymmetry_factor
    )
    
    # Calculate Planck functions for longwave
    lw_band_limits = parameters.lw_band_limits
    planck_layers = planck_bands(temperature, lw_band_limits, parameters.n_lw_bands)
    planck_interfaces = planck_bands(
        jnp.linspace(temperature[0], temperature[-1], nlev + 1),
        lw_band_limits, parameters.n_lw_bands
    )
    
    # Surface properties
    surface_temp = temperature[-1]
    surface_planck = planck_bands(jnp.array([surface_temp]), lw_band_limits, parameters.n_lw_bands)[0]
    
    # Calculate longwave fluxes
    flux_up_lw, flux_down_lw = longwave_fluxes(
        lw_optics, planck_layers, planck_interfaces,
        rad_state.surface_emissivity[0], surface_planck, parameters.n_lw_bands
    )
    
    # Calculate shortwave fluxes (only if sun is up)
    # TOA flux per band (assume equal distribution)
    # Create fixed-size array and mask for dynamic n_sw_bands
    max_sw_bands = 10
    toa_flux_bands_all = jnp.ones(max_sw_bands) * toa_flux / jnp.maximum(parameters.n_sw_bands, 1.0)
    sw_band_mask = jnp.arange(max_sw_bands) < parameters.n_sw_bands
    toa_flux_bands = jnp.where(sw_band_mask, toa_flux_bands_all, 0.0)
    
    flux_up_sw, flux_down_sw, flux_direct_sw, flux_diffuse_sw = shortwave_fluxes(
        sw_optics, cos_zenith, toa_flux_bands,
        jnp.array([rad_state.surface_albedo_vis[0], rad_state.surface_albedo_nir[0]]),
        parameters.n_sw_bands
    )
    
    # Zero out fluxes if sun is not up
    daylight_factor = jnp.where(cos_zenith > 0, 1.0, 0.0)
    flux_up_sw = flux_up_sw * daylight_factor
    flux_down_sw = flux_down_sw * daylight_factor
    
    # Convert fluxes to heating rates
    lw_heating_rate = flux_to_heating_rate(
        jnp.sum(flux_up_lw, axis=1), jnp.sum(flux_down_lw, axis=1),
        rad_state.pressure_interfaces
    )
    
    sw_heating_rate = flux_to_heating_rate(
        jnp.sum(flux_up_sw, axis=1), jnp.sum(flux_down_sw, axis=1),
        rad_state.pressure_interfaces
    )
    
    # Ensure SW heating is zero when no sunlight (avoid NaN propagation)
    sw_heating_rate = jnp.where(cos_zenith > 0, sw_heating_rate, 0.0)
    
    total_heating = lw_heating_rate + sw_heating_rate
    
    # Extract diagnostic fluxes
    olr = jnp.sum(flux_up_lw[0, :])  # TOA upward LW
    toa_sw_down = jnp.sum(flux_down_sw[0, :])
    toa_sw_up = jnp.sum(flux_up_sw[0, :])
    surface_sw_down = jnp.sum(flux_down_sw[-1, :])
    surface_lw_down = jnp.sum(flux_down_lw[-1, :])
    
    # Create output structures
    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating_rate,
        shortwave_heating=sw_heating_rate
    )
    
    diagnostics = RadiationData(
        sw_flux_up=flux_up_sw,
        sw_flux_down=flux_down_sw,
        lw_flux_up=flux_up_lw,
        lw_flux_down=flux_down_lw,
        sw_heating_rate=sw_heating_rate,
        lw_heating_rate=lw_heating_rate,
        toa_sw_down=toa_sw_down,
        toa_sw_up=toa_sw_up,
        toa_lw_up=olr,
        surface_sw_down=surface_sw_down,
        surface_sw_up=jnp.sum(flux_up_sw[-1, :]),  # Use calculated upward flux
        surface_lw_down=surface_lw_down,
        surface_lw_up=jnp.sum(flux_up_lw[-1, :]),  # Use calculated upward flux
    )
    
    return tendencies, diagnostics


# Convenience function for single column
def radiation_column(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    surface_pressure: float,
    geopotential: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    day_of_year: float,
    seconds_since_midnight: float,
    latitude: float,
    longitude: float,
    parameters: Optional[RadiationParameters] = None
) -> Tuple[RadiationTendencies, RadiationData]:
    """Single column radiation calculation"""
    return radiation_scheme(
        temperature=temperature,
        specific_humidity=specific_humidity,
        surface_pressure=surface_pressure,
        geopotential=geopotential,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        day_of_year=day_of_year,
        seconds_since_midnight=seconds_since_midnight,
        latitude=latitude,
        longitude=longitude,
        parameters=parameters
    )