"""
Data structures and types for surface physics.

This module defines the key data structures used in surface processes,
following the ICON model structure for land-atmosphere exchange.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp
import tree_math


@tree_math.struct
class SurfaceParameters:
    """Parameters for surface physics schemes."""
    
    # Surface types
    nsfc_type: int         # Number of surface types (water, ice, land)
    iwtr: int              # Index for water surface
    iice: int              # Index for ice surface  
    ilnd: int              # Index for land surface
    
    # Mixed layer ocean parameters
    ml_depth: float        # Mixed layer depth [m]
    rho_water: float       # Water density [kg/m³]
    cp_water: float        # Specific heat of water [J/kg/K]
    
    # Sea ice parameters
    rho_ice: float         # Ice density [kg/m³]
    cp_ice: float          # Specific heat of ice [J/kg/K]
    conduct_ice: float     # Ice thermal conductivity [W/m/K]
    
    # Surface radiation parameters
    emissivity: float      # Surface emissivity [-]
    stefan_boltzmann: float  # Stefan-Boltzmann constant [W/m²/K⁴]
    
    # Roughness lengths [m]
    z0_water: float        # Water surface
    z0_ice: float          # Ice surface
    z0_land: float         # Land surface (can be spatially varying)
    
    # Turbulent exchange parameters
    von_karman: float      # von Kármán constant [-]
    min_wind_speed: float  # Minimum wind speed for flux calculations [m/s]

    @classmethod
    def default(cls,  nsfc_type=3, iwtr=0, iice=1, ilnd=2,
                 ml_depth=50.0, rho_water=1025.0, cp_water=3994.0,
                 rho_ice=917.0, cp_ice=2106.0, conduct_ice=2.2,
                 emissivity=0.99, stefan_boltzmann=5.67e-8,
                 z0_water=1e-4, z0_ice=1e-3, z0_land=0.1,
                 von_karman=0.4, min_wind_speed=1.0) -> 'SurfaceParameters':
        """Return default surface parameters"""
        return cls(
            nsfc_type=jnp.array(nsfc_type),
            iwtr=jnp.array(iwtr),
            iice=jnp.array(iice),
            ilnd=jnp.array(ilnd),
            ml_depth=jnp.array(ml_depth),
            rho_water=jnp.array(rho_water),
            cp_water=jnp.array(cp_water),
            rho_ice=jnp.array(rho_ice),
            cp_ice=jnp.array(cp_ice),
            conduct_ice=jnp.array(conduct_ice),
            emissivity=jnp.array(emissivity),
            stefan_boltzmann=jnp.array(stefan_boltzmann),
            z0_water=jnp.array(z0_water),
            z0_ice=jnp.array(z0_ice),
            z0_land=jnp.array(z0_land),
            von_karman=jnp.array(von_karman),
            min_wind_speed=jnp.array(min_wind_speed)
        )


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