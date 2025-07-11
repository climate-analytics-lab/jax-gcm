"""
Data structures and types for surface physics.

This module defines the key data structures used in surface processes,
following the ICON model structure for land-atmosphere exchange.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp


class SurfaceParameters(NamedTuple):
    """Parameters for surface physics schemes."""
    
    # Surface types
    nsfc_type: int = 3         # Number of surface types (water, ice, land)
    iwtr: int = 0             # Index for water surface
    iice: int = 1             # Index for ice surface  
    ilnd: int = 2             # Index for land surface
    
    # Mixed layer ocean parameters
    ml_depth: float = 50.0     # Mixed layer depth [m]
    rho_water: float = 1025.0  # Water density [kg/m³]
    cp_water: float = 3994.0   # Specific heat of water [J/kg/K]
    
    # Sea ice parameters
    rho_ice: float = 917.0     # Ice density [kg/m³]
    cp_ice: float = 2106.0     # Specific heat of ice [J/kg/K]
    conduct_ice: float = 2.2   # Ice thermal conductivity [W/m/K]
    
    # Surface radiation parameters
    emissivity: float = 0.99   # Surface emissivity [-]
    stefan_boltzmann: float = 5.67e-8  # Stefan-Boltzmann constant [W/m²/K⁴]
    
    # Roughness lengths [m]
    z0_water: float = 1e-4     # Water surface
    z0_ice: float = 1e-3       # Ice surface
    z0_land: float = 0.1       # Land surface (can be spatially varying)
    
    # Turbulent exchange parameters
    von_karman: float = 0.4    # von Kármán constant [-]
    min_wind_speed: float = 1.0  # Minimum wind speed for flux calculations [m/s]


class SurfaceState(NamedTuple):
    """Surface state variables."""
    
    # Surface temperatures
    temperature: jnp.ndarray       # Surface temperature [K] (ncol, nsfc_type)
    temperature_rad: jnp.ndarray   # Radiative temperature [K] (ncol,)
    
    # Surface fractions
    fraction: jnp.ndarray          # Surface type fraction [-] (ncol, nsfc_type)
    
    # Ocean/lake variables
    ocean_temp: jnp.ndarray        # Ocean temperature [K] (ncol,)
    ocean_u: jnp.ndarray           # Ocean u-velocity [m/s] (ncol,)
    ocean_v: jnp.ndarray           # Ocean v-velocity [m/s] (ncol,)
    
    # Sea ice variables
    ice_thickness: jnp.ndarray     # Ice thickness [m] (ncol, nice_layers)
    ice_temp: jnp.ndarray          # Ice temperature [K] (ncol, nice_layers)
    snow_depth: jnp.ndarray        # Snow depth [m] (ncol,)
    
    # Land surface variables
    soil_temp: jnp.ndarray         # Soil temperature [K] (ncol, nsoil_layers)
    soil_moisture: jnp.ndarray     # Soil moisture [-] (ncol, nsoil_layers)
    vegetation_temp: jnp.ndarray   # Vegetation temperature [K] (ncol,)
    
    # Surface properties
    roughness_momentum: jnp.ndarray  # Momentum roughness [m] (ncol, nsfc_type)
    roughness_heat: jnp.ndarray      # Heat roughness [m] (ncol, nsfc_type)
    albedo_visible_direct: jnp.ndarray    # Visible direct albedo [-] (ncol, nsfc_type)
    albedo_visible_diffuse: jnp.ndarray   # Visible diffuse albedo [-] (ncol, nsfc_type)
    albedo_nir_direct: jnp.ndarray        # NIR direct albedo [-] (ncol, nsfc_type)
    albedo_nir_diffuse: jnp.ndarray       # NIR diffuse albedo [-] (ncol, nsfc_type)


class AtmosphericForcing(NamedTuple):
    """Atmospheric forcing at the surface."""
    
    # State at lowest atmospheric level
    temperature: jnp.ndarray       # Air temperature [K] (ncol,)
    humidity: jnp.ndarray          # Specific humidity [kg/kg] (ncol,)
    u_wind: jnp.ndarray           # Zonal wind [m/s] (ncol,)
    v_wind: jnp.ndarray           # Meridional wind [m/s] (ncol,)
    pressure: jnp.ndarray         # Pressure [Pa] (ncol,)
    
    # Radiation fluxes
    sw_downward: jnp.ndarray      # Downward shortwave [W/m²] (ncol,)
    lw_downward: jnp.ndarray      # Downward longwave [W/m²] (ncol,)
    
    # Precipitation
    rain_rate: jnp.ndarray        # Rain rate [kg/m²/s] (ncol,)
    snow_rate: jnp.ndarray        # Snow rate [kg/m²/s] (ncol,)
    
    # Exchange coefficients from turbulence scheme
    exchange_coeff_heat: jnp.ndarray     # Heat exchange [m²/s] (ncol, nsfc_type)
    exchange_coeff_moisture: jnp.ndarray # Moisture exchange [m²/s] (ncol, nsfc_type)
    exchange_coeff_momentum: jnp.ndarray # Momentum exchange [m²/s] (ncol, nsfc_type)


class SurfaceFluxes(NamedTuple):
    """Surface fluxes computed by surface physics."""
    
    # Energy fluxes [W/m²]
    sensible_heat: jnp.ndarray     # Sensible heat flux (ncol, nsfc_type)
    latent_heat: jnp.ndarray       # Latent heat flux (ncol, nsfc_type)
    longwave_net: jnp.ndarray      # Net longwave flux (ncol, nsfc_type)
    shortwave_net: jnp.ndarray     # Net shortwave flux (ncol, nsfc_type)
    ground_heat: jnp.ndarray       # Ground heat flux (ncol, nsfc_type)
    
    # Momentum fluxes [N/m²]
    momentum_u: jnp.ndarray        # U-momentum flux (ncol, nsfc_type)
    momentum_v: jnp.ndarray        # V-momentum flux (ncol, nsfc_type)
    
    # Mass fluxes [kg/m²/s]
    evaporation: jnp.ndarray       # Evaporation rate (ncol, nsfc_type)
    transpiration: jnp.ndarray     # Transpiration rate (ncol, nsfc_type)
    
    # Grid-box mean fluxes [various units]
    sensible_heat_mean: jnp.ndarray    # Grid-box mean sensible heat [W/m²] (ncol,)
    latent_heat_mean: jnp.ndarray      # Grid-box mean latent heat [W/m²] (ncol,)
    momentum_u_mean: jnp.ndarray       # Grid-box mean u-momentum [N/m²] (ncol,)
    momentum_v_mean: jnp.ndarray       # Grid-box mean v-momentum [N/m²] (ncol,)
    evaporation_mean: jnp.ndarray      # Grid-box mean evaporation [kg/m²/s] (ncol,)


class SurfaceTendencies(NamedTuple):
    """Tendencies for surface prognostic variables."""
    
    # Temperature tendencies [K/s]
    surface_temp_tendency: jnp.ndarray     # Surface temperature (ncol, nsfc_type)
    ocean_temp_tendency: jnp.ndarray       # Ocean temperature (ncol,)
    ice_temp_tendency: jnp.ndarray         # Ice temperature (ncol, nice_layers)
    soil_temp_tendency: jnp.ndarray        # Soil temperature (ncol, nsoil_layers)
    
    # Mass tendencies
    ice_thickness_tendency: jnp.ndarray    # Ice thickness [m/s] (ncol, nice_layers)
    snow_depth_tendency: jnp.ndarray       # Snow depth [m/s] (ncol,)
    soil_moisture_tendency: jnp.ndarray    # Soil moisture [1/s] (ncol, nsoil_layers)


class SurfaceDiagnostics(NamedTuple):
    """Diagnostic variables from surface physics."""
    
    # Standard diagnostics
    temperature_2m: jnp.ndarray        # 2m temperature [K] (ncol,)
    humidity_2m: jnp.ndarray           # 2m specific humidity [kg/kg] (ncol,)
    dewpoint_2m: jnp.ndarray           # 2m dew point [K] (ncol,)
    wind_speed_10m: jnp.ndarray        # 10m wind speed [m/s] (ncol,)
    u_wind_10m: jnp.ndarray            # 10m u-wind [m/s] (ncol,)
    v_wind_10m: jnp.ndarray            # 10m v-wind [m/s] (ncol,)
    
    # Surface layer properties
    friction_velocity: jnp.ndarray     # Friction velocity [m/s] (ncol,)
    richardson_number: jnp.ndarray     # Surface Richardson number [-] (ncol,)
    surface_layer_height: jnp.ndarray # Surface layer height [m] (ncol,)
    
    # Energy balance components [W/m²]
    net_radiation: jnp.ndarray         # Net radiation (ncol,)
    radiation_balance: jnp.ndarray     # Radiation balance (ncol,)
    energy_balance_residual: jnp.ndarray  # Energy balance residual (ncol,)
    
    # Tile-specific diagnostics
    temperature_2m_tile: jnp.ndarray   # 2m temperature per tile [K] (ncol, nsfc_type)
    humidity_2m_tile: jnp.ndarray      # 2m humidity per tile [kg/kg] (ncol, nsfc_type)
    wind_speed_10m_tile: jnp.ndarray   # 10m wind per tile [m/s] (ncol, nsfc_type)


class SurfaceResistances(NamedTuple):
    """Surface resistances for flux calculations."""
    
    # Aerodynamic resistances [s/m]
    aerodynamic_heat: jnp.ndarray      # Heat transfer (ncol, nsfc_type)
    aerodynamic_moisture: jnp.ndarray  # Moisture transfer (ncol, nsfc_type)
    aerodynamic_momentum: jnp.ndarray  # Momentum transfer (ncol, nsfc_type)
    
    # Surface resistances [s/m]
    surface_moisture: jnp.ndarray      # Surface moisture resistance (ncol, nsfc_type)
    canopy_resistance: jnp.ndarray     # Canopy resistance (ncol,)
    soil_resistance: jnp.ndarray       # Soil resistance (ncol,)
    
    # Stability corrections [-]
    stability_heat: jnp.ndarray        # Heat stability function (ncol, nsfc_type)
    stability_momentum: jnp.ndarray    # Momentum stability function (ncol, nsfc_type)