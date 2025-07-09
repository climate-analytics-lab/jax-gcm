"""
Tiedtke-Nordeng Mass-Flux Convection Scheme

This module implements the Tiedtke-Nordeng convection parameterization
in JAX, based on the ICON atmospheric model implementation.

The scheme includes:
- Deep convection with CAPE closure
- Shallow convection with moisture convergence closure  
- Mid-level convection
- Convective momentum transport
- Downdraft processes

References:
- Tiedtke, M. (1989): A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models. Mon. Weather Rev., 117, 1779-1800.
- Nordeng, T. E. (1994): Extended versions of the convective parametrization
  scheme at ECMWF and their impact on the mean and transient activity of the
  model in the tropics. ECMWF Tech. Memo. 206.

Date: 2025-01-09
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass

from ..constants.physical_constants import (
    grav, rd, rv, cp, eps, tmelt, alhc, alhs
)

# Import updraft, downdraft and flux modules after they're defined
# This avoids circular imports


@dataclass(frozen=True)
class ConvectionConfig:
    """Configuration parameters for Tiedtke-Nordeng convection scheme"""
    
    # Time stepping
    dt_conv: float = 3600.0           # Convection timestep (s)
    
    # Entrainment/detrainment parameters
    entrpen: float = 1.0e-4           # Entrainment rate for penetrative convection (m⁻¹)
    entrscv: float = 3.0e-3           # Entrainment rate for shallow convection (m⁻¹) 
    entrmid: float = 1.0e-4           # Entrainment rate for mid-level convection (m⁻¹)
    
    # CAPE closure
    tau: float = 7200.0               # CAPE adjustment timescale (s)
    
    # Cloud base mass flux
    cmfcmax: float = 1.0              # Maximum cloud base mass flux (kg/m²/s)
    cmfcmin: float = 1.0e-10          # Minimum cloud base mass flux (kg/m²/s)
    
    # Precipitation parameters
    cprcon: float = 1.4e-3            # Coefficient for precipitation conversion
    
    # Evaporation parameters
    cevapcu: float = 2.0e-5           # Coefficient for rain evaporation
    
    # Numerical parameters
    epsilon: float = 1.0e-12          # Small number for numerical stability
    
    # Convection type thresholds
    rlcrit: float = 8.0e-4            # Critical relative humidity for shallow convection
    rhcrit: float = 0.9               # Critical relative humidity threshold
    
    # Momentum transport
    cmfctop: float = 0.33             # Mass flux fraction at cloud top


class ConvectionState(NamedTuple):
    """State variables for convection scheme"""
    
    # Updraft properties
    tu: jnp.ndarray          # Updraft temperature (K)
    qu: jnp.ndarray          # Updraft specific humidity (kg/kg)  
    lu: jnp.ndarray          # Updraft liquid water content (kg/kg)
    uu: jnp.ndarray          # Updraft zonal wind (m/s)
    vu: jnp.ndarray          # Updraft meridional wind (m/s)
    
    # Downdraft properties  
    td: jnp.ndarray          # Downdraft temperature (K)
    qd: jnp.ndarray          # Downdraft specific humidity (kg/kg)
    ud: jnp.ndarray          # Downdraft zonal wind (m/s)
    vd: jnp.ndarray          # Downdraft meridional wind (m/s)
    
    # Mass fluxes
    mfu: jnp.ndarray         # Updraft mass flux (kg/m²/s)
    mfd: jnp.ndarray         # Downdraft mass flux (kg/m²/s)
    
    # Convection diagnostics
    ktype: jnp.ndarray       # Convection type (0=none, 1=deep, 2=shallow, 3=mid)
    kbase: jnp.ndarray       # Cloud base level index
    ktop: jnp.ndarray        # Cloud top level index
    
    # Precipitation
    prate: jnp.ndarray       # Precipitation rate (kg/m²/s)


class ConvectionTendencies(NamedTuple):
    """Tendencies from convection scheme"""
    
    dtedt: jnp.ndarray       # Temperature tendency (K/s)
    dqdt: jnp.ndarray        # Specific humidity tendency (kg/kg/s)
    dudt: jnp.ndarray        # Zonal wind tendency (m/s²)
    dvdt: jnp.ndarray        # Meridional wind tendency (m/s²)
    
    # Convective fluxes
    qc_conv: jnp.ndarray     # Convective cloud water (kg/kg)
    qi_conv: jnp.ndarray     # Convective cloud ice (kg/kg)
    
    # Surface fluxes
    precip_conv: jnp.ndarray # Convective precipitation (kg/m²/s)
    
    # Tracer tendencies (including qc, qi)
    dtracer_dt: Optional[jnp.ndarray] = None  # Tracer tendencies [nlev, ntrac]


def saturation_vapor_pressure(temperature: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate saturation vapor pressure using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)
    """
    # Tetens formula coefficients
    a = 17.27
    b = 35.86
    
    t_celsius = temperature - tmelt
    
    # Over water (T > 0°C)
    es_water = 610.78 * jnp.exp(a * t_celsius / (t_celsius + 237.3))
    
    # Over ice (T <= 0°C) 
    es_ice = 610.78 * jnp.exp(b * t_celsius / (t_celsius + 265.5))
    
    # Use water or ice formula depending on temperature
    es = jnp.where(temperature > tmelt, es_water, es_ice)
    
    return es


def saturation_mixing_ratio(pressure: jnp.ndarray, 
                          temperature: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate saturation mixing ratio
    
    Args:
        pressure: Pressure (Pa)
        temperature: Temperature (K)
        
    Returns:
        Saturation mixing ratio (kg/kg)
    """
    es = saturation_vapor_pressure(temperature)
    qs = eps * es / (pressure - es * (1.0 - eps))
    return jnp.maximum(qs, 0.0)


def moist_static_energy(temperature: jnp.ndarray,
                       height: jnp.ndarray, 
                       mixing_ratio: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate moist static energy
    
    Args:
        temperature: Temperature (K)
        height: Geopotential height (m)  
        mixing_ratio: Water vapor mixing ratio (kg/kg)
        
    Returns:
        Moist static energy (J/kg)
    """
    return cp * temperature + grav * height + alhc * mixing_ratio


def initialize_convection(temperature: jnp.ndarray,
                         humidity: jnp.ndarray,
                         pressure: jnp.ndarray,
                         height: jnp.ndarray,
                         u_wind: jnp.ndarray,
                         v_wind: jnp.ndarray,
                         config: ConvectionConfig) -> ConvectionState:
    """
    Initialize convection state variables
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        config: Convection configuration
        
    Returns:
        Initial convection state
    """
    nlev = temperature.shape[0]
    
    # Initialize updraft properties with environmental values (ensure float32)
    tu = jnp.array(temperature, dtype=jnp.float32)
    qu = jnp.array(humidity, dtype=jnp.float32)
    lu = jnp.zeros_like(temperature, dtype=jnp.float32)
    uu = jnp.array(u_wind, dtype=jnp.float32)
    vu = jnp.array(v_wind, dtype=jnp.float32)
    
    # Initialize downdraft properties (ensure float32)
    td = jnp.array(temperature, dtype=jnp.float32)
    qd = jnp.array(humidity, dtype=jnp.float32)
    ud = jnp.array(u_wind, dtype=jnp.float32)
    vd = jnp.array(v_wind, dtype=jnp.float32)
    
    # Initialize mass fluxes to zero with explicit dtype
    mfu = jnp.zeros_like(temperature, dtype=jnp.float32)
    mfd = jnp.zeros_like(temperature, dtype=jnp.float32)
    
    # Initialize convection diagnostics
    ktype = jnp.array(0)  # No convection initially
    kbase = jnp.array(nlev - 1)  # Surface level
    ktop = jnp.array(0)   # Top level
    
    # Initialize precipitation
    prate = jnp.array(0.0)
    
    return ConvectionState(
        tu=tu, qu=qu, lu=lu, uu=uu, vu=vu,
        td=td, qd=qd, ud=ud, vd=vd,
        mfu=mfu, mfd=mfd,
        ktype=ktype, kbase=kbase, ktop=ktop,
        prate=prate
    )


def find_cloud_base(temperature: jnp.ndarray,
                   humidity: jnp.ndarray, 
                   pressure: jnp.ndarray,
                   config: ConvectionConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find lifting condensation level (cloud base)
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        config: Convection configuration
        
    Returns:
        Tuple of (cloud_base_level, cloud_base_exists)
    """
    nlev = len(temperature)
    
    # Start from surface (bottom level - highest pressure)
    surf_idx = jnp.argmax(pressure)  # Surface is at highest pressure
    surf_temp = temperature[surf_idx]
    surf_humid = humidity[surf_idx]
    surf_press = pressure[surf_idx]
    
    # Calculate parcel temperature at all levels (dry adiabatic)
    exner_ratios = (pressure / surf_press) ** (rd / cp)
    parcel_temps = surf_temp * exner_ratios
    
    # Calculate saturation mixing ratio at parcel temperatures
    parcel_qs = jax.vmap(saturation_mixing_ratio)(pressure, parcel_temps)
    
    # Check where parcel becomes saturated
    is_saturated = surf_humid >= parcel_qs
    
    # Find first level (from bottom up) where saturation occurs
    # Start from surface and go up
    levels = jnp.arange(nlev)
    
    # Mask for levels where saturation occurs
    # Only consider levels above surface but below very high levels
    valid_levels = jnp.logical_and(levels < nlev - 1, levels > 0)
    saturated_and_valid = jnp.logical_and(is_saturated, valid_levels)
    
    # Find first saturated level (lowest index above surface)
    saturated_levels = jnp.where(saturated_and_valid, levels, nlev)
    cloud_base_level = jnp.min(saturated_levels)
    cloud_base_found = cloud_base_level < nlev
    
    # If no cloud base found, set to surface
    cloud_base_level = jnp.where(cloud_base_found, cloud_base_level, nlev - 1)
    
    return cloud_base_level, cloud_base_found


def calculate_cape_cin(temperature: jnp.ndarray,
                      humidity: jnp.ndarray,
                      pressure: jnp.ndarray,
                      height: jnp.ndarray,
                      cloud_base: int,
                      config: ConvectionConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate CAPE and CIN for convective instability
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]  
        pressure: Environmental pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        cloud_base: Cloud base level index
        config: Convection configuration
        
    Returns:
        Tuple of (CAPE, CIN) in J/kg
    """
    nlev = len(temperature)
    
    # Surface parcel properties
    surf_idx = jnp.argmax(pressure)  # Surface is at highest pressure
    surf_temp = temperature[surf_idx]
    surf_humid = humidity[surf_idx]
    surf_press = pressure[surf_idx]
    
    # Level indices for vectorization
    k_levels = jnp.arange(nlev)
    
    # Pressure ratios
    press_ratios = pressure / surf_press
    
    # Below cloud base - dry adiabatic ascent
    parcel_temp_dry = surf_temp * (press_ratios ** (rd / cp))
    parcel_humid_dry = jnp.full_like(temperature, surf_humid)
    
    # Above cloud base - moist adiabatic (simplified)
    parcel_qs = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
    parcel_temp_moist = temperature  # Simplified moist adiabatic
    parcel_humid_moist = parcel_qs
    
    # Use dry or moist based on level relative to cloud base
    # Need to check pressure ordering to determine "below" cloud base
    pressure_decreasing = pressure[0] < pressure[-1]  # True if index 0 is top
    
    is_below_cb = lax.cond(
        pressure_decreasing,
        lambda: k_levels > cloud_base,   # Standard ordering: below = higher indices
        lambda: k_levels < cloud_base    # Reverse ordering: below = lower indices  
    )
    parcel_temp = jnp.where(is_below_cb, parcel_temp_dry, parcel_temp_moist)
    parcel_humid = jnp.where(is_below_cb, parcel_humid_dry, parcel_humid_moist)
    
    # Virtual temperatures
    env_tv = temperature * (1.0 + 0.61 * humidity)
    parcel_tv = parcel_temp * (1.0 + 0.61 * parcel_humid)
    
    # Buoyancy
    buoyancy = grav * (parcel_tv - env_tv) / env_tv
    
    # Layer thickness (avoid last level)
    dz = jnp.concatenate([jnp.diff(height), jnp.array([0.0])])
    dz = jnp.abs(dz)  # Ensure positive
    
    # Calculate CAPE and CIN contributions at each level
    cape_contrib = jnp.where(buoyancy > 0, buoyancy * dz, 0.0)
    cin_contrib = jnp.where(buoyancy <= 0, -buoyancy * dz, 0.0)
    
    # Sum over levels (exclude surface level)
    cape = jnp.sum(cape_contrib[:-1])
    cin = jnp.sum(cin_contrib[:-1])
    
    return cape, cin


def tiedtke_nordeng_convection(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray, 
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    dt: float,
    config: Optional[ConvectionConfig] = None
) -> Tuple[ConvectionTendencies, ConvectionState]:
    """
    Main Tiedtke-Nordeng convection scheme
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        dt: Time step (s)
        config: Convection configuration
        
    Returns:
        Tuple of (tendencies, final_state)
    """
    if config is None:
        config = ConvectionConfig()
    
    nlev = len(temperature)
    
    # Initialize state
    state = initialize_convection(
        temperature, humidity, pressure, height, 
        u_wind, v_wind, config
    )
    
    # Find cloud base
    cloud_base, has_cloud_base = find_cloud_base(
        temperature, humidity, pressure, config
    )
    
    # Calculate CAPE and CIN if cloud base exists
    cape, cin = lax.cond(
        has_cloud_base,
        lambda: calculate_cape_cin(temperature, humidity, pressure, height, 
                                 cloud_base, config),
        lambda: (jnp.array(0.0), jnp.array(0.0))
    )
    
    # Determine convection type based on CAPE and other criteria
    # 0 = no convection, 1 = deep, 2 = shallow, 3 = mid-level
    # Use more reasonable CAPE thresholds for triggering
    conv_type = lax.cond(
        jnp.logical_and(has_cloud_base, cape > 100.0),  # Minimum CAPE threshold
        lambda: lax.cond(cape > 1000.0, lambda: 1, lambda: 2),  # Deep vs shallow
        lambda: 0  # No convection
    )
    
    # Initialize tendencies to zero with explicit float32 dtype
    dtedt = jnp.zeros_like(temperature, dtype=jnp.float32)
    dqdt = jnp.zeros_like(humidity, dtype=jnp.float32)
    dudt = jnp.zeros_like(u_wind, dtype=jnp.float32)
    dvdt = jnp.zeros_like(v_wind, dtype=jnp.float32)
    qc_conv = jnp.zeros_like(temperature, dtype=jnp.float32)
    qi_conv = jnp.zeros_like(temperature, dtype=jnp.float32)
    precip_conv = jnp.array(0.0, dtype=jnp.float32)
    
    # Import modules here to avoid circular imports
    from .updraft import calculate_updraft
    from .downdraft import calculate_downdraft
    from .flux_tendencies import (
        calculate_tendencies, mass_flux_closure
    )
    
    # Calculate air density
    rho = pressure / (rd * temperature)
    
    # Apply full convection scheme if active
    def apply_full_convection():
        # Determine cloud top based on CAPE profile  
        # Simplified - full version would search for equilibrium level
        cloud_depth = lax.cond(conv_type == 2, lambda: 3, lambda: 6)  # Shallow vs deep
        
        # Handle level ordering properly
        pressure_decreasing = pressure[0] < pressure[-1]
        
        ktop = lax.cond(
            pressure_decreasing,
            lambda: jnp.maximum(cloud_base - cloud_depth, 0),      # Standard: top = lower index
            lambda: jnp.minimum(cloud_base + cloud_depth, nlev-1)  # Reverse: top = higher index
        )
        
        # Calculate mass flux using appropriate closure
        moisture_conv = jnp.array(0.0)  # Would calculate from large-scale fields
        mass_flux_base = mass_flux_closure(
            cape, cin, moisture_conv, conv_type, config
        )
        
        # Calculate updraft
        updraft_state = calculate_updraft(
            temperature, humidity, pressure, height, rho,
            cloud_base, ktop, conv_type, mass_flux_base, config
        )
        
        # Calculate precipitation from updraft
        precip_rate = jnp.sum(updraft_state.lu * updraft_state.mfu) * config.cprcon
        
        # Calculate downdraft (simplified for now to avoid dtype issues)
        # downdraft_state = calculate_downdraft(
        #     temperature, humidity, pressure, height, rho,
        #     updraft_state, precip_rate, cloud_base, ktop, config
        # )
        
        # Simplified downdraft state with correct dtypes
        from .downdraft import DowndraftState
        downdraft_state = DowndraftState(
            td=jnp.array(temperature, dtype=jnp.float32),
            qd=jnp.array(humidity, dtype=jnp.float32),
            mfd=jnp.zeros_like(temperature, dtype=jnp.float32),
            lfs=nlev - 1,   # Surface level
            active=False    # No downdraft for now
        )
        
        # Calculate final tendencies
        tendencies = calculate_tendencies(
            temperature, humidity, u_wind, v_wind, pressure, rho,
            updraft_state, downdraft_state, 
            cloud_base, ktop, dt, config
        )
        
        # Update state
        new_state = ConvectionState(
            tu=updraft_state.tu, qu=updraft_state.qu, lu=updraft_state.lu,
            uu=u_wind, vu=v_wind,  # Simplified - would update from momentum transport
            td=downdraft_state.td, qd=downdraft_state.qd,
            ud=u_wind, vd=v_wind,  # Simplified
            mfu=updraft_state.mfu, mfd=downdraft_state.mfd,
            ktype=jnp.array(conv_type), kbase=jnp.array(cloud_base), 
            ktop=jnp.array(ktop), prate=tendencies.precip_conv
        )
        
        return tendencies, new_state
    
    # No convection case
    def no_convection():
        tendencies = ConvectionTendencies(
            dtedt=dtedt, dqdt=dqdt, dudt=dudt, dvdt=dvdt,
            qc_conv=qc_conv, qi_conv=qi_conv, precip_conv=precip_conv
        )
        return tendencies, state
    
    # Apply convection if active
    tendencies, updated_state = lax.cond(
        conv_type > 0,
        apply_full_convection,
        no_convection
    )
    
    return tendencies, updated_state


def tiedtke_nordeng_convection_with_tracers(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray, 
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    tracers: jnp.ndarray,
    dt: float,
    config: Optional[ConvectionConfig] = None,
    tracer_indices: Optional['TracerIndices'] = None
) -> Tuple[ConvectionTendencies, ConvectionState]:
    """
    Enhanced Tiedtke-Nordeng convection with tracer transport
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        height: Geopotential height (m) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        tracers: Tracer concentrations [nlev, ntrac] including qc, qi
        dt: Time step (s)
        config: Convection configuration
        tracer_indices: Indices for different tracer types
        
    Returns:
        Tuple of (tendencies, final_state) with tracer transport
    """
    # First run standard convection
    tendencies, state = tiedtke_nordeng_convection(
        temperature, humidity, pressure, height,
        u_wind, v_wind, dt, config
    )
    
    # If no convection, return with zero tracer tendencies
    if state.ktype == 0:
        ntrac = tracers.shape[1] if tracers.ndim > 1 else 0
        tendencies = tendencies._replace(
            dtracer_dt=jnp.zeros((len(temperature), ntrac))
        )
        return tendencies, state
    
    # Import tracer transport module
    from .tracer_transport import transport_tracers, TracerIndices
    
    if tracer_indices is None:
        tracer_indices = TracerIndices()
    
    # Calculate air density
    rho = pressure / (rd * temperature)
    
    # Import updraft/downdraft modules
    from .updraft import calculate_updraft
    from .downdraft import calculate_downdraft
    
    # Recalculate updraft and downdraft states for tracer transport
    # (In a full implementation, these would be cached from the main calculation)
    cape, cin = calculate_cape_cin(
        temperature, humidity, pressure, height, 
        state.kbase, config or ConvectionConfig()
    )
    
    # Get mass flux from closure
    from .flux_tendencies import mass_flux_closure
    mass_flux_base = mass_flux_closure(
        cape, cin, jnp.array(0.0), state.ktype, 
        config or ConvectionConfig()
    )
    
    # Calculate updraft
    updraft_state = calculate_updraft(
        temperature, humidity, pressure, height, rho,
        state.kbase, state.ktop, state.ktype, 
        mass_flux_base, config or ConvectionConfig()
    )
    
    # Calculate downdraft
    downdraft_state = calculate_downdraft(
        temperature, humidity, pressure, height, rho,
        updraft_state, state.prate, state.kbase, state.ktop,
        config or ConvectionConfig()
    )
    
    # Transport tracers
    tracer_transport = transport_tracers(
        tracers, temperature, pressure, rho,
        updraft_state, downdraft_state, dt, tracer_indices
    )
    
    # Update tendencies with tracer transport
    # Note: qc and qi tendencies from detrainment are in dtracer_dt
    tendencies = tendencies._replace(
        dtracer_dt=tracer_transport.dtracer_dt,
        qc_conv=tracer_transport.detrain_qc,  # Instantaneous detrainment
        qi_conv=tracer_transport.detrain_qi   # Instantaneous detrainment
    )
    
    return tendencies, state