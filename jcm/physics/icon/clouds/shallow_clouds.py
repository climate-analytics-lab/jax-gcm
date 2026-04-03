"""Shallow cloud scheme for ICON physics

This module implements a simplified cloud scheme focusing on:
- Cloud fraction diagnosis based on relative humidity
- Cloud water and ice content
- Basic condensation/evaporation processes

Based on the Lohmann and Roeckner (1996) scheme used in ICON/ECHAM.

Date: 2025-01-10
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
import tree_math

from ..constants.physical_constants import (
    tmelt, alhc, alhs, cp, eps, grav
)


@tree_math.struct
class CloudParameters:
    """Configuration parameters for shallow cloud scheme"""
    
    # Cloud fraction parameters
    crt: float           # Critical relative humidity at surface  
    crs: float           # Critical relative humidity at TOA  
    nex: float           # Exponent for RH threshold profile
    csatsc: float        # Saturation factor for stratocumulus
    
    # Microphysics parameters
    ccraut: float        # Beheng autoconversion rate scaling (dimensionless, ECHAM default 4)
    ccsaut: float        # Ice autoconversion rate (1/s)
    ceffmin: float       # Minimum cloud droplet radius (microns)
    ceffmax: float       # Maximum cloud droplet radius (microns)

    # Numerical parameters
    epsilon: float       # Small number for numerical stability
    
    # Cloud ice temperature thresholds
    t_ice: float         # Temperature below which all cloud is ice (K)
    t_mix_min: float     # Lower bound of mixed phase (K)
    t_mix_max: float     # Upper bound of mixed phase (K)

    @classmethod
    def default(cls, crt=0.9, crs=0.7, nex=4.0,
                 csatsc=0.97, ccraut=4.0, ccsaut=0.001,
                 ceffmin=10.0,
                 ceffmax=150.0, epsilon=1.0e-12,
                 t_ice=238.15, t_mix_min=238.15, t_mix_max=273.15) -> 'CloudParameters':
        """Return default cloud parameters"""
        return cls(
            crt=jnp.array(crt),
            crs=jnp.array(crs),
            nex=jnp.array(nex),
            csatsc=jnp.array(csatsc),
            ccraut=jnp.array(ccraut),
            ccsaut=jnp.array(ccsaut),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            epsilon=jnp.array(epsilon),
            t_ice=jnp.array(t_ice),
            t_mix_min=jnp.array(t_mix_min),
            t_mix_max=jnp.array(t_mix_max)
        )


class CloudState(NamedTuple):
    """Cloud state variables"""
    
    cloud_fraction: jnp.ndarray     # Cloud fraction [0-1]
    cloud_water: jnp.ndarray        # Cloud liquid water content (kg/kg)
    cloud_ice: jnp.ndarray          # Cloud ice content (kg/kg)
    rel_humidity: jnp.ndarray       # Relative humidity [0-1]
    
    # Diagnostics
    total_cloud_cover: jnp.ndarray  # Column total cloud cover
    
    
class CloudTendencies(NamedTuple):
    """Tendencies from cloud processes"""
    
    dtedt: jnp.ndarray         # Temperature tendency (K/s)
    dqdt: jnp.ndarray          # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray         # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray         # Cloud ice tendency (kg/kg/s)
    
    # Surface precipitation fluxes
    rain_flux: jnp.ndarray     # Surface rain flux (kg/m²/s)
    snow_flux: jnp.ndarray     # Surface snow flux (kg/m²/s)


def saturation_vapor_pressure_water(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation vapor pressure over water using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)

    """
    t_celsius = temperature - tmelt
    return 610.78 * jnp.exp(17.27 * t_celsius / (t_celsius + 237.3))


def saturation_vapor_pressure_ice(temperature: jnp.ndarray) -> jnp.ndarray:
    """Calculate saturation vapor pressure over ice using Tetens formula
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Saturation vapor pressure (Pa)

    """
    t_celsius = temperature - tmelt
    return 610.78 * jnp.exp(21.87 * t_celsius / (t_celsius + 265.5))


def saturation_specific_humidity(
    pressure: jnp.ndarray, 
    temperature: jnp.ndarray
) -> jnp.ndarray:
    """Calculate saturation specific humidity
    
    Args:
        pressure: Pressure (Pa)
        temperature: Temperature (K)
        
    Returns:
        Saturation specific humidity (kg/kg)

    """
    # Use appropriate saturation vapor pressure based on temperature
    es_water = saturation_vapor_pressure_water(temperature)
    es_ice = saturation_vapor_pressure_ice(temperature)
    
    # Blend between ice and water saturation in mixed phase region
    # Linear interpolation between t_ice and tmelt
    weight = jnp.clip((temperature - 238.15) / (tmelt - 238.15), 0.0, 1.0)
    es = weight * es_water + (1.0 - weight) * es_ice
    
    # Convert to saturation specific humidity
    qs = eps * es / (pressure - es * (1.0 - eps))
    return jnp.maximum(qs, 0.0)


def calculate_cloud_fraction(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    surface_pressure: float,
    config: CloudParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate cloud fraction using relative humidity scheme
    
    Based on Lohmann and Roeckner (1996) diagnostic cloud scheme.
    
    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        pressure: Pressure (Pa)
        surface_pressure: Surface pressure (Pa)
        config: Cloud configuration
        
    Returns:
        Tuple of (cloud_fraction, relative_humidity)

    """
    # Calculate saturation specific humidity
    qs = saturation_specific_humidity(pressure, temperature)
    
    # Calculate relative humidity
    rel_humidity = specific_humidity / (qs + config.epsilon)
    rel_humidity = jnp.clip(rel_humidity, 0.0, 1.0)
    
    # Calculate critical relative humidity threshold
    # Varies from crt at surface to crs at TOA
    # Following Lohmann & Roeckner (1996) formulation
    sigma = pressure / surface_pressure  # Normalized pressure (1 at surface, 0 at TOA)
    # RHc = crt at surface (sigma=1) and crs at TOA (sigma→0)
    # Using exponential interpolation: at sigma=1, exp(0)=1 so rhc=crt
    # as sigma→0, exp(-nex)→0 so rhc→crs
    rhc = config.crs + (config.crt - config.crs) * jnp.exp(
        -config.nex * (1.0 - sigma)
    )
    
    # Calculate cloud fraction using quadratic relationship
    # b_0 = (RH - RH_crit) / (1 - RH_crit)
    b0 = (rel_humidity - rhc) / (1.0 - rhc + config.epsilon)
    b0 = jnp.clip(b0, 0.0, 1.0)
    
    # Cloud fraction: cc = 1 - sqrt(1 - b0)
    cloud_fraction = 1.0 - jnp.sqrt(1.0 - b0)
    
    # Apply minimum cloud fraction threshold
    cloud_fraction = jnp.where(cloud_fraction < 0.01, 0.0, cloud_fraction)
    
    return cloud_fraction, rel_humidity


def partition_cloud_phase(
    temperature: jnp.ndarray,
    total_cloud_water: jnp.ndarray,
    config: CloudParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Partition cloud water between liquid and ice phases
    
    Args:
        temperature: Temperature (K)
        total_cloud_water: Total cloud condensate (kg/kg)
        config: Cloud configuration
        
    Returns:
        Tuple of (cloud_liquid, cloud_ice)

    """
    # Calculate ice fraction based on temperature
    # All ice below t_ice, all liquid above tmelt
    # Linear transition in between
    ice_frac = jnp.clip(
        (config.t_mix_max - temperature) / (config.t_mix_max - config.t_mix_min),
        0.0, 1.0
    )
    
    # Partition cloud water
    cloud_ice = ice_frac * total_cloud_water
    cloud_liquid = (1.0 - ice_frac) * total_cloud_water
    
    return cloud_liquid, cloud_ice


def condensation_evaporation(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    pressure: jnp.ndarray,
    dt: float,
    config: CloudParameters
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate condensation/evaporation tendencies
    
    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        cloud_water: Cloud liquid water (kg/kg)
        cloud_ice: Cloud ice (kg/kg)
        cloud_fraction: Cloud fraction [0-1]
        pressure: Pressure (Pa)
        dt: Time step (s)
        config: Cloud configuration
        
    Returns:
        Tuple of (dT/dt, dq/dt, dqc/dt, dqi/dt)

    """
    # Calculate saturation specific humidity
    qs = saturation_specific_humidity(pressure, temperature)
    
    # Calculate condensation/evaporation
    # Positive for condensation, negative for evaporation
    q_excess = specific_humidity - qs
    
    # Condensation/evaporation rate (instantaneous adjustment)
    # Positive q_excess -> condensation, negative -> evaporation
    cond_evap_rate = q_excess / dt
    
    # Limit evaporation to available cloud water/ice
    total_cloud = cloud_water + cloud_ice
    max_evap_rate = -total_cloud / dt
    
    # Apply limits
    cond_evap = jnp.where(
        cond_evap_rate < 0,  # Evaporation
        jnp.maximum(cond_evap_rate, max_evap_rate),
        cond_evap_rate  # Condensation
    )
    
    # Specific humidity tendency (opposite sign)
    dqdt = -cond_evap
    
    # Partition between liquid and ice based on temperature
    weight_liquid = jnp.clip(
        (temperature - config.t_mix_min) / (config.t_mix_max - config.t_mix_min),
        0.0, 1.0
    )
    
    # Tendencies for cloud water and ice
    # For condensation (positive cond_evap), partition between liquid and ice
    # For evaporation (negative cond_evap), remove proportionally from existing phases
    dqcdt = jnp.where(
        cond_evap > 0,  # Condensation
        weight_liquid * cond_evap,
        cond_evap * cloud_water / (total_cloud + config.epsilon)  # Proportional evaporation
    )
    dqidt = jnp.where(
        cond_evap > 0,  # Condensation  
        (1.0 - weight_liquid) * cond_evap,
        cond_evap * cloud_ice / (total_cloud + config.epsilon)  # Proportional evaporation
    )
    
    # Temperature tendency from latent heat
    # Use appropriate latent heat based on phase
    L = jnp.where(
        cond_evap > 0,  # Condensation - use weighted latent heat
        weight_liquid * alhc + (1.0 - weight_liquid) * alhs,
        # Evaporation - use weighted latent heat based on what's evaporating
        (cloud_water * alhc + cloud_ice * alhs) / (total_cloud + config.epsilon)
    )
    # Positive cond_evap (condensation) releases heat -> positive temperature tendency
    # Negative cond_evap (evaporation) consumes heat -> negative temperature tendency
    dtedt = L * cond_evap / cp
    
    return dtedt, dqdt, dqcdt, dqidt


def shallow_cloud_scheme(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    surface_pressure: float,
    cdnc: jnp.ndarray,
    dt: float,
    config: Optional[CloudParameters] = None
) -> Tuple[CloudTendencies, CloudState]:
    """Run shallow cloud scheme

    Args:
        temperature: Temperature (K) [nlev] or scalar
        specific_humidity: Specific humidity (kg/kg) [nlev] or scalar
        pressure: Pressure (Pa) [nlev] or scalar
        cloud_water: Cloud liquid water (kg/kg) [nlev] or scalar
        cloud_ice: Cloud ice (kg/kg) [nlev] or scalar
        surface_pressure: Surface pressure (Pa)
        cdnc: Cloud droplet number concentration (1/m³) [nlev]
        dt: Time step (s)
        config: Cloud configuration

    Returns:
        Tuple of (tendencies, cloud_state)

    """
    if config is None:
        config = CloudParameters.default()
    
    # Ensure all inputs are arrays
    temperature = jnp.atleast_1d(temperature)
    specific_humidity = jnp.atleast_1d(specific_humidity)
    pressure = jnp.atleast_1d(pressure)
    cloud_water = jnp.atleast_1d(cloud_water)
    cloud_ice = jnp.atleast_1d(cloud_ice)
        
    # Calculate cloud fraction and relative humidity
    cloud_fraction, rel_humidity = calculate_cloud_fraction(
        temperature, specific_humidity, pressure, surface_pressure, config
    )
    
    # Calculate condensation/evaporation
    dtedt, dqdt, dqcdt, dqidt = condensation_evaporation(
        temperature, specific_humidity, cloud_water, cloud_ice,
        cloud_fraction, pressure, dt, config
    )
    
    # --- Within-timestep condensation → autoconversion → precipitation ---
    # Following ECHAM mo_cloud.f90: condensation updates in-cloud water
    # (zxlb += zcnd), then autoconversion acts on the updated values.
    # Use the condensation computed by condensation_evaporation above,
    # applied within this timestep to get the updated cloud water.
    updated_cloud_water = jnp.maximum(cloud_water + dqcdt * dt, 0.0)
    updated_cloud_ice = jnp.maximum(cloud_ice + dqidt * dt, 0.0)

    # Air density from ideal gas law (needed for Beheng autoconversion)
    from ..constants.physical_constants import rd
    rho = pressure / (rd * temperature)

    # Warm-phase autoconversion: Beheng (1994), following ECHAM mo_cloud.f90 lines 834-862
    # ECHAM operates on IN-CLOUD water (zxlb = grid-mean / cloud_fraction).
    # Analytic solution to dqc/dt = -gamma * qc^(1 + exm1)
    # gamma = ccraut * 1.2e27 / rho * cdnc^(-3.3) * (rho * 1e-3)^4.7
    exm1 = 3.7  # 4.7 - 1.0
    zexp = -1.0 / exm1
    # ECHAM converts cdnc from 1/m³ to 1/cm³ (pacdnc*1e-6) for the Beheng formula
    cdnc_cgs = jnp.maximum(cdnc, 1.0) * 1e-6  # 1/m³ → 1/cm³, floor to avoid zero
    gamma = (config.ccraut * 1.2e27) / rho * cdnc_cgs**(-3.3) * (rho * 1e-3)**4.7

    # Convert to in-cloud values (ECHAM: zxlb = grid-mean * zclcauxi)
    cf_safe = jnp.maximum(cloud_fraction, config.epsilon)
    qc_incloud = jnp.maximum(updated_cloud_water / cf_safe, 0.0)
    qi_incloud = jnp.maximum(updated_cloud_ice / cf_safe, 0.0)

    denom = 1.0 + gamma * dt * exm1 * qc_incloud**exm1
    rain_auto_incloud = qc_incloud * (1.0 - denom**zexp)
    rain_auto_incloud = jnp.maximum(rain_auto_incloud, 0.0)
    # Convert back to grid-mean (ECHAM: zrpr = zclcaux * zraut)
    rain_auto = cloud_fraction * rain_auto_incloud

    # Ice autoconversion: Levkov et al. (1992), following ECHAM mo_cloud.f90 lines 912-913
    snow_auto_incloud = qi_incloud * (1.0 - 1.0 / (1.0 + config.ccsaut * dt * qi_incloud))
    snow_auto_incloud = jnp.maximum(snow_auto_incloud, 0.0)
    snow_auto = cloud_fraction * snow_auto_incloud

    # Update cloud water/ice tendencies to include precipitation loss
    dqcdt = dqcdt - rain_auto / dt
    dqidt = dqidt - snow_auto / dt

    # Column-integrated precipitation flux using dp/g (kg/m²/s)
    dp = jnp.abs(jnp.diff(pressure, prepend=0.0))
    rain_flux = jnp.sum(rain_auto * dp / dt) / grav
    snow_flux = jnp.sum(snow_auto * dp / dt) / grav

    # Total cloud cover (maximum overlap assumption)
    total_cloud_cover = jnp.max(cloud_fraction)
    
    # Create output structures
    tendencies = CloudTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dqcdt=dqcdt,
        dqidt=dqidt,
        rain_flux=jnp.array(rain_flux),
        snow_flux=jnp.array(snow_flux)
    )
    
    state = CloudState(
        cloud_fraction=cloud_fraction,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        rel_humidity=rel_humidity,
        total_cloud_cover=jnp.array(total_cloud_cover)
    )
    
    return tendencies, state