"""
Type definitions and parameters for radiation calculations

This module defines the data structures and configuration parameters
used throughout the radiation scheme.

Date: 2025-01-10
"""

import jax.numpy as jnp
from typing import NamedTuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class RadiationParameters:
    """Configuration parameters for radiation scheme"""
    
    # Time stepping
    dt_rad: float = 3600.0           # Radiation time step (s)
    
    # Solar parameters  
    solar_constant: float = 1361.0    # Solar constant (W/m²)
    
    # Spectral bands
    n_sw_bands: int = 2              # Number of shortwave bands
    n_lw_bands: int = 3              # Number of longwave bands
    
    # Band limits (wavenumber in cm⁻¹)
    lw_band_limits: tuple = ((10, 350), (350, 500), (500, 2500))  # LW bands
    sw_band_limits: tuple = ((4000, 14500), (14500, 50000))       # SW bands
    
    # Gas concentrations (volume mixing ratios)
    co2_vmr: float = 400e-6          # CO2 volume mixing ratio
    ch4_vmr: float = 1.8e-6          # CH4 volume mixing ratio
    n2o_vmr: float = 0.32e-6         # N2O volume mixing ratio
    
    # Surface properties
    surface_albedo_vis: float = 0.15  # Visible band albedo
    surface_albedo_nir: float = 0.15  # Near-IR band albedo
    surface_emissivity: float = 0.98  # Longwave emissivity
    
    # Numerical parameters
    min_cos_zenith: float = 0.035    # Minimum cosine solar zenith angle (~88 deg)
    flux_epsilon: float = 1e-6       # Small value for flux calculations
    
    # Cloud optics parameters
    cld_tau_min: float = 1e-6        # Minimum cloud optical depth
    cld_frac_min: float = 1e-3       # Minimum cloud fraction


class RadiationState(NamedTuple):
    """State variables for radiation calculations"""
    
    # Solar geometry
    cos_zenith: jnp.ndarray          # Cosine of solar zenith angle
    daylight_fraction: jnp.ndarray   # Fraction of grid cell in daylight
    
    # Atmospheric profiles
    temperature: jnp.ndarray         # Temperature (K) [nlev]
    pressure: jnp.ndarray            # Pressure (Pa) [nlev]
    pressure_interfaces: jnp.ndarray # Pressure at interfaces (Pa) [nlev+1]
    
    # Gas mixing ratios
    h2o_vmr: jnp.ndarray            # Water vapor volume mixing ratio [nlev]
    o3_vmr: jnp.ndarray             # Ozone volume mixing ratio [nlev]
    
    # Cloud properties
    cloud_fraction: jnp.ndarray      # Cloud fraction [nlev]
    cloud_water_path: jnp.ndarray    # Cloud water path (kg/m²) [nlev]
    cloud_ice_path: jnp.ndarray      # Cloud ice path (kg/m²) [nlev]
    
    # Surface properties
    surface_temperature: jnp.ndarray # Surface temperature (K)
    surface_albedo_vis: jnp.ndarray  # Visible surface albedo
    surface_albedo_nir: jnp.ndarray  # Near-IR surface albedo
    surface_emissivity: jnp.ndarray  # Surface emissivity


class RadiationFluxes(NamedTuple):
    """Radiation fluxes at interfaces"""
    
    # Shortwave fluxes (W/m²) [nlev+1]
    sw_down: jnp.ndarray             # Downward shortwave flux
    sw_up: jnp.ndarray               # Upward shortwave flux
    sw_down_clear: jnp.ndarray       # Clear-sky downward shortwave
    sw_up_clear: jnp.ndarray         # Clear-sky upward shortwave
    
    # Longwave fluxes (W/m²) [nlev+1]
    lw_down: jnp.ndarray             # Downward longwave flux
    lw_up: jnp.ndarray               # Upward longwave flux
    lw_down_clear: jnp.ndarray       # Clear-sky downward longwave
    lw_up_clear: jnp.ndarray         # Clear-sky upward longwave
    
    # Surface components (W/m²)
    sw_down_vis_dir: jnp.ndarray     # Direct visible at surface
    sw_down_vis_dif: jnp.ndarray     # Diffuse visible at surface
    sw_down_nir_dir: jnp.ndarray     # Direct near-IR at surface
    sw_down_nir_dif: jnp.ndarray     # Diffuse near-IR at surface


class RadiationTendencies(NamedTuple):
    """Tendencies from radiation"""
    
    temperature_tendency: jnp.ndarray # Temperature tendency (K/s) [nlev]
    longwave_heating: jnp.ndarray     # Longwave heating rate (K/s) [nlev]
    shortwave_heating: jnp.ndarray    # Shortwave heating rate (K/s) [nlev]


class OpticalProperties(NamedTuple):
    """Optical properties for radiative transfer"""
    
    optical_depth: jnp.ndarray       # Layer optical depth [nlev, nbands]
    single_scatter_albedo: jnp.ndarray  # Single scattering albedo [nlev, nbands]
    asymmetry_factor: jnp.ndarray    # Asymmetry factor [nlev, nbands]


@dataclass(frozen=True)
class SpectralBands:
    """Definition of spectral bands"""
    
    # Shortwave bands (wavelength in micrometers)
    sw_band_limits: tuple = (
        (0.2, 0.7),    # Visible
        (0.7, 4.0),    # Near-IR
    )
    
    # Longwave bands (wavenumber in cm⁻¹)
    lw_band_limits: tuple = (
        (10, 350),     # Window region
        (350, 500),    # CO2 band
        (500, 2500),   # H2O bands
    )
    
    # Band weights for solar spectrum
    sw_solar_fraction: tuple = (0.5, 0.5)  # Simplified equal split