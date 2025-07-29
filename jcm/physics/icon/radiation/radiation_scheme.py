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
    planck_bands_lw,
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


def combine_optical_properties(
    gas_optical_depth: jnp.ndarray,
    cloud_optics: OpticalProperties,
    aerosol_optical_depth: Optional[jnp.ndarray] = None,
    aerosol_ssa: Optional[jnp.ndarray] = None,
    aerosol_asymmetry: Optional[jnp.ndarray] = None
) -> OpticalProperties:
    """
    Combine gas, cloud, and aerosol optical properties.
    
    Args:
        gas_optical_depth: Gas optical depth [nlev, nbands]
        cloud_optics: Cloud optical properties
        aerosol_optical_depth: Aerosol optical depth [nlev, nbands] 
        aerosol_ssa: Aerosol single scatter albedo [nlev, nbands]
        aerosol_asymmetry: Aerosol asymmetry factor [nlev, nbands]
        
    Returns:
        Combined optical properties
    """
    # Start with gas + cloud
    total_tau = gas_optical_depth + cloud_optics.optical_depth
    
    # If no aerosols, return gas + cloud
    if aerosol_optical_depth is None:
        return OpticalProperties(
            optical_depth=total_tau,
            single_scatter_albedo=cloud_optics.single_scatter_albedo,
            asymmetry_factor=cloud_optics.asymmetry_factor
        )
    
    # Ensure aerosol properties have the right shape for the current band structure
    nlev, nbands = total_tau.shape
    if aerosol_optical_depth.shape != (nlev, nbands):
        # If aerosol data doesn't match band structure, skip aerosol effects
        return OpticalProperties(
            optical_depth=total_tau,
            single_scatter_albedo=cloud_optics.single_scatter_albedo,
            asymmetry_factor=cloud_optics.asymmetry_factor
        )
    
    # Add aerosol optical depth
    total_tau_with_aerosol = total_tau + aerosol_optical_depth
    
    # Combine single scattering albedo (weighted by scattering optical depth)
    cloud_scattering = cloud_optics.optical_depth * cloud_optics.single_scatter_albedo
    aerosol_scattering = aerosol_optical_depth * aerosol_ssa
    total_scattering = cloud_scattering + aerosol_scattering
    
    combined_ssa = jnp.where(
        total_tau_with_aerosol > 0,
        total_scattering / total_tau_with_aerosol,
        0.0
    )
    
    # Combine asymmetry factor (weighted by scattering optical depth)
    cloud_g_weighted = cloud_scattering * cloud_optics.asymmetry_factor
    aerosol_g_weighted = aerosol_scattering * aerosol_asymmetry
    
    combined_g = jnp.where(
        total_scattering > 0,
        (cloud_g_weighted + aerosol_g_weighted) / total_scattering,
        0.0
    )
    
    return OpticalProperties(
        optical_depth=total_tau_with_aerosol,
        single_scatter_albedo=combined_ssa,
        asymmetry_factor=combined_g
    )


def prepare_radiation_state(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    cos_zenith: float,
    ozone_vmr: Optional[jnp.ndarray] = None,
    aerosol_optical_depth: Optional[jnp.ndarray] = None,
    aerosol_ssa: Optional[jnp.ndarray] = None,
    aerosol_asymmetry: Optional[jnp.ndarray] = None
) -> RadiationState:
    """
    Prepare radiation state from physics state variables.
    
    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure_levels: Pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cloud_water: Cloud water content (kg/kg) [nlev]
        cloud_ice: Cloud ice content (kg/kg) [nlev]
        cloud_fraction: Cloud fraction (0-1) [nlev]
        cos_zenith: Cosine of solar zenith angle
        ozone_vmr: Ozone volume mixing ratio [nlev]
        aerosol_optical_depth: Aerosol optical depth [nlev, nbands]
        aerosol_ssa: Aerosol single scatter albedo [nlev, nbands] 
        aerosol_asymmetry: Aerosol asymmetry factor [nlev, nbands]
        
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
        cos_zenith=cos_zenith[jnp.newaxis],
        daylight_fraction=jnp.where(cos_zenith > 0, 1.0, 0.0)[jnp.newaxis],
        temperature=temperature,
        pressure=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        h2o_vmr=h2o_vmr,
        o3_vmr=ozone_vmr,
        cloud_fraction=cloud_fraction,
        cloud_water_path=cloud_water_path,
        cloud_ice_path=cloud_ice_path,
        surface_temperature=temperature[-1:],  # Bottom level temperature
        surface_albedo_vis=jnp.array([0.15]),  # Visible albedo
        surface_albedo_nir=jnp.array([0.15]),  # Near-IR albedo
        surface_emissivity=jnp.array([0.98]),
        aerosol_optical_depth=aerosol_optical_depth,
        aerosol_ssa=aerosol_ssa,
        aerosol_asymmetry=aerosol_asymmetry
    )



def radiation_scheme(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    day_of_year: float,
    seconds_since_midnight: float,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,  # AerosolData from physics_data
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6
) -> Tuple[RadiationTendencies, RadiationData]:
    """
    Radiation scheme wrapper that extracts aerosol data and includes aerosol effects.
    
    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure_levels: Pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cloud_water: Cloud water content (kg/kg) [nlev]
        cloud_ice: Cloud ice content (kg/kg) [nlev]
        cloud_fraction: Cloud fraction (0-1) [nlev]
        day_of_year: Day of year (1-365)
        seconds_since_midnight: Seconds since midnight UTC
        latitude: Latitude (degrees)
        longitude: Longitude (degrees)
        parameters: Radiation parameters
        aerosol_data: AerosolData containing optical properties
        ozone_vmr: Ozone volume mixing ratio [nlev]
        co2_vmr: CO2 volume mixing ratio
        
    Returns:
        Tuple of (radiation tendencies, radiation diagnostics)
    """
    nlev = temperature.shape[0]
    
    # For now, assume aerosol properties are spectrally uniform
    # and expand to radiation band structure
    n_sw_bands = parameters.n_sw_bands
    n_lw_bands = parameters.n_lw_bands
    
    # Expand aerosol profiles to radiation bands
    # Handle both 1D (single column from vmap) and 2D (full grid) aerosol data
    if aerosol_data.aod_profile.ndim == 1:
        # 1D case: single column from vmap
        aerosol_aod_col = aerosol_data.aod_profile[:, None]  # Make it [nlev, 1]
        aerosol_ssa_col = aerosol_data.ssa_profile[:, None]
        aerosol_asy_col = aerosol_data.asy_profile[:, None]
    else:
        # 2D case: take first column
        aerosol_aod_col = aerosol_data.aod_profile[:, 0:1]
        aerosol_ssa_col = aerosol_data.ssa_profile[:, 0:1]
        aerosol_asy_col = aerosol_data.asy_profile[:, 0:1]
    
    # SW bands - use fixed default values (2 SW bands, 3 LW bands)
    # This is standard for ICON radiation and avoids tracer issues
    default_n_sw_bands = 2
    default_n_lw_bands = 3
    
    aerosol_tau_sw = jnp.tile(aerosol_aod_col, (1, default_n_sw_bands))
    aerosol_ssa_sw = jnp.tile(aerosol_ssa_col, (1, default_n_sw_bands))
    aerosol_asy_sw = jnp.tile(aerosol_asy_col, (1, default_n_sw_bands))
    
    # LW bands (pure absorption for aerosols)
    aerosol_tau_lw = jnp.tile(aerosol_aod_col, (1, default_n_lw_bands))
    aerosol_ssa_lw = jnp.zeros((nlev, default_n_lw_bands))  # Pure absorption
    aerosol_asy_lw = jnp.zeros((nlev, default_n_lw_bands))
    
    # For SW use scattering properties, for LW use absorption
    aerosol_optical_depth = jnp.concatenate([aerosol_tau_sw, aerosol_tau_lw], axis=1)
    aerosol_ssa = jnp.concatenate([aerosol_ssa_sw, aerosol_ssa_lw], axis=1)
    aerosol_asymmetry = jnp.concatenate([aerosol_asy_sw, aerosol_asy_lw], axis=1)
    
    # Cloud droplet number concentration factor
    # Handle both 1D (single column from vmap) and 2D (full grid) cases
    if aerosol_data.cdnc_factor.ndim == 0:
        # 0D case: scalar from vmap
        cdnc_factor = aerosol_data.cdnc_factor
    else:
        # 1D case: take first element
        cdnc_factor = aerosol_data.cdnc_factor[0] 
    
    # Now perform the actual radiation calculation
    
    # Solar radiation calculations
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
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        cos_zenith=cos_zenith,
        ozone_vmr=ozone_vmr,
        aerosol_optical_depth=aerosol_optical_depth,
        aerosol_ssa=aerosol_ssa,
        aerosol_asymmetry=aerosol_asymmetry
    )
        
    # Calculate gas optical depths
    gas_tau_lw = gas_optical_depth_lw(
        temperature=temperature,
        pressure=pressure_levels,
        h2o_vmr=rad_state.h2o_vmr,
        o3_vmr=rad_state.o3_vmr,
        co2_vmr=co2_vmr,
        layer_thickness=layer_thickness,
        air_density=air_density,
    )
    
    gas_tau_sw = gas_optical_depth_sw(
        temperature=temperature,
        pressure=pressure_levels,
        h2o_vmr=rad_state.h2o_vmr,
        o3_vmr=rad_state.o3_vmr,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cos_zenith=cos_zenith
    )
    
    # Calculate cloud optical properties
    cloud_sw_optics, cloud_lw_optics = cloud_optics(
        cloud_water_path=rad_state.cloud_water_path,
        cloud_ice_path=rad_state.cloud_ice_path,
        temperature=temperature,
        cdnc_factor=cdnc_factor
    )
    
    # Combine gas, cloud, and aerosol optical depths
    sw_optics = combine_optical_properties(
        gas_tau_sw,
        cloud_sw_optics,
        aerosol_optical_depth[:, :default_n_sw_bands],
        aerosol_ssa[:, :default_n_sw_bands],
        aerosol_asymmetry[:, :default_n_sw_bands]
    )
    
    lw_optics = combine_optical_properties(
        gas_tau_lw,
        cloud_lw_optics,
        aerosol_optical_depth[:, default_n_sw_bands:],
        aerosol_ssa[:, default_n_sw_bands:],
        aerosol_asymmetry[:, default_n_sw_bands:]
    )
    
    # Calculate Planck functions for longwave
    lw_band_limits = parameters.lw_band_limits
    planck_layers = planck_bands_lw(temperature, lw_band_limits)
    planck_interfaces = planck_bands_lw(
        jnp.linspace(temperature[0], temperature[-1], nlev + 1),
        lw_band_limits
    )
    
    # Surface properties
    surface_planck = planck_bands_lw(temperature[-1:], lw_band_limits)[0]
    
    # Calculate longwave fluxes
    flux_up_lw, flux_down_lw = longwave_fluxes(
        lw_optics, planck_layers, planck_interfaces,
        rad_state.surface_emissivity[0], surface_planck
    )
    
    # Calculate shortwave fluxes
    max_sw_bands = 10
    toa_flux_bands_all = jnp.ones(max_sw_bands) * toa_flux / jnp.maximum(default_n_sw_bands, 1.0)
    sw_band_mask = jnp.arange(max_sw_bands) < default_n_sw_bands
    toa_flux_bands = jnp.where(sw_band_mask, toa_flux_bands_all, 0.0)
    
    flux_up_sw, flux_down_sw, flux_direct_sw, flux_diffuse_sw = shortwave_fluxes(
        sw_optics, cos_zenith, toa_flux_bands,
        jnp.array([rad_state.surface_albedo_vis[0], rad_state.surface_albedo_nir[0]]),
        default_n_sw_bands
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
    
    # Ensure SW heating is zero when no sunlight
    sw_heating_rate = jnp.where(cos_zenith > 0, sw_heating_rate, 0.0)
    
    total_heating = lw_heating_rate + sw_heating_rate
    
    # Extract diagnostic fluxes
    olr = jnp.sum(flux_up_lw[0, :])
    toa_sw_down = jnp.sum(flux_down_sw[0, :])
    toa_sw_up = jnp.sum(flux_up_sw[0, :])
    surface_sw_down = jnp.sum(flux_down_sw[-1, :])
    surface_sw_up = jnp.sum(flux_up_sw[-1, :])
    surface_lw_down = jnp.sum(flux_down_lw[-1, :])
    surface_lw_up = jnp.sum(flux_up_lw[-1, :])
    
    # Create output structures
    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating_rate,
        shortwave_heating=sw_heating_rate
    )
    
    diagnostics = RadiationData(
        cos_zenith=cos_zenith[jnp.newaxis],
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
        surface_sw_up=surface_sw_up,
        surface_lw_down=surface_lw_down,
        surface_lw_up=surface_lw_up,
    )
    
    return tendencies, diagnostics