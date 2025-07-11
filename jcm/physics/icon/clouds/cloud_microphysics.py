"""
Cloud microphysics scheme for ICON physics

This module implements comprehensive cloud microphysics including:
- Autoconversion of cloud water to rain (Khairoutdinov and Kogan, 2000)
- Accretion of cloud droplets by rain
- Autoconversion of cloud ice to snow 
- Aggregation of ice crystals and accretion by snow
- Melting of snow and freezing of rain
- Sedimentation of cloud ice and snow
- Evaporation of rain and sublimation of snow

Based on the ECHAM6/ICON microphysics as described in:
- Lohmann and Roeckner (1996)
- Levkov et al. (1992) for ice phase
- Beheng (1994) for warm phase

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple, Optional
import tree_math

from ..constants.physical_constants import (
    tmelt, alhf, alhc, alhs, rd, rv, cp, grav, rhow, eps
)


@tree_math.struct
class MicrophysicsParameters:
    """Configuration parameters for cloud microphysics"""
    
    # Autoconversion parameters
    ccraut: float        # Critical cloud water for autoconversion (kg/kg)
    ccracl: float        # Accretion coefficient (cloud to rain)
    cauloc: float        # Cloud droplet dispersion parameter
    ceffmin: float       # Minimum cloud droplet radius (microns)
    ceffmax: float       # Maximum cloud droplet radius (microns)
    
    # Ice microphysics parameters
    cn0s: float          # Snow particle number density (1/m^3)
    crhosno: float       # Snow density (kg/m^3)
    cvtfall: float       # Terminal velocity factor for ice
    cthomi: float        # Homogeneous ice nucleation temperature (K)
    csecfrl: float       # Critical ice fraction for Bergeron-Findeisen
    
    # Collection efficiencies
    ccollec: float       # Collection efficiency rain/cloud
    ccollei: float       # Collection efficiency snow/ice
    
    # Time scale parameters
    tau_melt: float      # Melting time scale (s)
    tau_freeze: float    # Freezing time scale (s)
    
    # Evaporation/sublimation parameters
    cevaprain: float     # Rain evaporation coefficient
    cevapsnow: float     # Snow sublimation coefficient
    
    # Sedimentation parameters
    vt_ice: float        # Ice crystal fall speed (m/s)
    vt_snow_a: float     # Snow fall speed coefficient a
    vt_snow_b: float     # Snow fall speed exponent b
    vt_rain_a: float     # Rain fall speed coefficient a
    vt_rain_b: float     # Rain fall speed exponent b
    
    # Numerical parameters
    epsilon: float       # Small number for numerical stability
    dt_sedi: float       # Sub-timestep for sedimentation (s)

    @classmethod
    def default(cls, ccraut=5.0e-4, ccracl=6.0, cauloc=1.0, ceffmin=10.0, ceffmax=150.0, cn0s=3.0e6,
                 crhosno=100.0, cvtfall=3.29, cthomi=233.15, csecfrl=0.1, ccollec=0.7,
                 ccollei=0.3, tau_melt=100.0, tau_freeze=100.0, cevaprain=1.0e-3,
                 cevapsnow=5.0e-4, vt_ice=0.1, vt_snow_a=8.8, vt_snow_b=0.15,
                 vt_rain_a=386.0, vt_rain_b=0.67, epsilon=1.0e-12, dt_sedi=10.0) -> 'MicrophysicsParameters':
        """Return default microphysics parameters"""
        return cls(
            ccraut=jnp.array(ccraut),
            ccracl=jnp.array(ccracl),
            cauloc=jnp.array(cauloc),
            ceffmin=jnp.array(ceffmin),
            ceffmax=jnp.array(ceffmax),
            cn0s=jnp.array(cn0s),
            crhosno=jnp.array(crhosno),
            cvtfall=jnp.array(cvtfall),
            cthomi=jnp.array(cthomi),
            csecfrl=jnp.array(csecfrl),
            ccollec=jnp.array(ccollec),
            ccollei=jnp.array(ccollei),
            tau_melt=jnp.array(tau_melt),
            tau_freeze=jnp.array(tau_freeze),
            cevaprain=jnp.array(cevaprain),
            cevapsnow=jnp.array(cevapsnow),
            vt_ice=jnp.array(vt_ice),
            vt_snow_a=jnp.array(vt_snow_a),
            vt_snow_b=jnp.array(vt_snow_b),
            vt_rain_a=jnp.array(vt_rain_a),
            vt_rain_b=jnp.array(vt_rain_b),
            epsilon=jnp.array(epsilon),
            dt_sedi=jnp.array(dt_sedi)
        )


class MicrophysicsState(NamedTuple):
    """Microphysics state variables and diagnostics"""
    
    # Precipitation fluxes (kg/m²/s)
    rain_flux: jnp.ndarray      # Rain flux at each level
    snow_flux: jnp.ndarray      # Snow flux at each level
    
    # In-cloud values
    qc_in_cloud: jnp.ndarray    # In-cloud liquid water (kg/kg)
    qi_in_cloud: jnp.ndarray    # In-cloud ice (kg/kg)
    
    # Process rates (kg/kg/s)
    autoconv_rate: jnp.ndarray  # Autoconversion rate
    accretion_rate: jnp.ndarray # Accretion rate
    melting_rate: jnp.ndarray   # Melting rate
    freezing_rate: jnp.ndarray  # Freezing rate
    
    # Precipitation at surface
    precip_rain: jnp.ndarray    # Surface rain (kg/m²/s)
    precip_snow: jnp.ndarray    # Surface snow (kg/m²/s)


class MicrophysicsTendencies(NamedTuple):
    """Tendencies from microphysics processes"""
    
    dtedt: jnp.ndarray          # Temperature tendency (K/s)
    dqdt: jnp.ndarray           # Specific humidity tendency (kg/kg/s)
    dqcdt: jnp.ndarray          # Cloud water tendency (kg/kg/s)
    dqidt: jnp.ndarray          # Cloud ice tendency (kg/kg/s)
    dqrdt: jnp.ndarray          # Rain water tendency (kg/kg/s)
    dqsdt: jnp.ndarray          # Snow tendency (kg/kg/s)


def cloud_droplet_radius(
    cloud_water: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """
    Calculate effective cloud droplet radius
    
    Args:
        cloud_water: Cloud liquid water content (kg/kg)
        air_density: Air density (kg/m³)
        droplet_number: Droplet number concentration (1/kg)
        config: Microphysics configuration
        
    Returns:
        Effective radius (m)
    """
    # Convert mixing ratio to mass concentration
    cloud_water_density = cloud_water * air_density  # kg/m³
    
    # Convert droplet number from per kg to per m³
    droplet_density = droplet_number * air_density  # 1/m³
    
    # Volume of single droplet
    volume_per_droplet = cloud_water_density / (droplet_density + config.epsilon) / rhow  # m³
    
    # Volume mean radius
    radius = (3.0 * volume_per_droplet / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    
    # Apply limits
    radius = jnp.clip(radius, config.ceffmin * 1e-6, config.ceffmax * 1e-6)
    
    return radius


def autoconversion_kk2000(
    cloud_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """
    Autoconversion of cloud water to rain (Khairoutdinov and Kogan, 2000)
    
    This parameterization is more sophisticated than simple threshold-based
    schemes and depends on both cloud water content and droplet concentration.
    
    Args:
        cloud_water: Cloud water mixing ratio (kg/kg) 
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        droplet_number: Cloud droplet number concentration (1/kg)
        dt: Time step (s)
        config: Microphysics configuration
        
    Returns:
        Autoconversion rate (kg/kg/s)
    """
    # In-cloud values
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0
    )
    
    # Convert mixing ratios to g/m³ for KK2000 formula
    qc_gm3 = qc_in_cloud * air_density * 1000.0  # g/m³
    nc_cm3 = droplet_number * air_density * 1e-6  # 1/cm³
    
    # KK2000 autoconversion rate formula
    # P_aut = 1350 * qc^2.47 * (Nc * 1e-6)^-1.79
    # Factor of 1e-3 converts from g/m³/s to kg/m³/s
    autoconv_rate = jnp.where(
        qc_in_cloud > config.ccraut,
        1350.0 * qc_gm3**2.47 * (nc_cm3 + config.epsilon)**(-1.79) * 1e-3 / air_density,
        0.0
    )
    
    # Convert to grid-mean tendency
    autoconv_rate = autoconv_rate * cloud_fraction
    
    # Limit to available cloud water
    max_rate = cloud_water / dt
    autoconv_rate = jnp.minimum(autoconv_rate, max_rate)
    
    return autoconv_rate


def accretion_rain_cloud(
    cloud_water: jnp.ndarray,
    rain_water: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """
    Accretion of cloud droplets by rain
    
    Args:
        cloud_water: Cloud water mixing ratio (kg/kg)
        rain_water: Rain water mixing ratio (kg/kg)
        cloud_fraction: Cloud fraction (0-1)
        air_density: Air density (kg/m³)
        config: Microphysics configuration
        
    Returns:
        Accretion rate (kg/kg/s)
    """
    # In-cloud values
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0
    )
    
    # Accretion rate following Beheng (1994)
    # Uses collection efficiency and geometric sweep-out
    accretion_rate = config.ccracl * config.ccollec * qc_in_cloud * rain_water * air_density**0.5
    
    # Convert to grid-mean
    accretion_rate = accretion_rate * cloud_fraction
    
    return accretion_rate


def ice_autoconversion(
    cloud_ice: jnp.ndarray,
    temperature: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """
    Autoconversion of cloud ice to snow through aggregation
    
    Args:
        cloud_ice: Cloud ice mixing ratio (kg/kg)
        temperature: Temperature (K)
        cloud_fraction: Cloud fraction (0-1)
        dt: Time step (s)
        config: Microphysics configuration
        
    Returns:
        Ice autoconversion rate (kg/kg/s)
    """
    # Temperature-dependent aggregation efficiency
    # Maximum near -15°C (258K)
    t_celsius = temperature - tmelt
    agg_efficiency = jnp.exp(-0.05 * jnp.abs(t_celsius + 15.0))
    
    # Critical ice content for autoconversion (fixed)
    qi_crit = 0.3e-3  # kg/kg
    
    # In-cloud ice
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / cloud_fraction,
        0.0
    )
    
    # Autoconversion rate with temperature-dependent efficiency
    autoconv_rate = jnp.where(
        qi_in_cloud > qi_crit,
        agg_efficiency * 0.001 * (qi_in_cloud - qi_crit) / dt,
        0.0
    )
    
    # Convert to grid mean
    autoconv_rate = autoconv_rate * cloud_fraction
    
    # Limit to available ice
    max_rate = cloud_ice / dt
    autoconv_rate = jnp.minimum(autoconv_rate, max_rate)
    
    return autoconv_rate


def snow_accretion(
    target: jnp.ndarray,
    snow: jnp.ndarray, 
    temperature: jnp.ndarray,
    air_density: jnp.ndarray,
    is_liquid: bool,
    config: MicrophysicsParameters
) -> jnp.ndarray:
    """
    Accretion of cloud water/ice by falling snow
    
    Args:
        target: Target species mixing ratio (cloud water or ice) (kg/kg)
        snow: Snow mixing ratio (kg/kg)
        temperature: Temperature (K)
        air_density: Air density (kg/m³)
        is_liquid: True for riming (liquid), False for aggregation (ice)
        config: Microphysics configuration
        
    Returns:
        Accretion rate (kg/kg/s)
    """
    # Collection efficiency
    efficiency = config.ccollec if is_liquid else config.ccollei
    
    # Temperature factor for aggregation (ice only)
    if not is_liquid:
        t_celsius = temperature - tmelt
        temp_factor = jnp.exp(-0.03 * jnp.abs(t_celsius + 15.0))
        efficiency = efficiency * temp_factor
    
    # Snow fall velocity
    snow_gm3 = snow * air_density * 1000.0  # g/m³
    vt_snow = config.vt_snow_a * snow_gm3**config.vt_snow_b
    
    # Accretion rate
    accretion_rate = efficiency * target * snow * vt_snow / (air_density**0.5)
    
    return accretion_rate


def melting_freezing(
    temperature: jnp.ndarray,
    snow: jnp.ndarray,
    rain: jnp.ndarray,
    dt: float,
    config: MicrophysicsParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate melting of snow and freezing of rain
    
    Args:
        temperature: Temperature (K)
        snow: Snow mixing ratio (kg/kg)
        rain: Rain mixing ratio (kg/kg)  
        dt: Time step (s)
        config: Microphysics configuration
        
    Returns:
        Tuple of (melting_rate, freezing_rate) in kg/kg/s
    """
    # Temperature departure from freezing
    dt_freeze = tmelt - temperature
    
    # Melting rate (T > 0°C)
    melt_rate = jnp.where(
        temperature > tmelt,
        snow * (temperature - tmelt) / (config.tau_melt * 10.0),  # Scaled by temp
        0.0
    )
    melt_rate = jnp.minimum(melt_rate, snow / dt)
    
    # Freezing rate (T < 0°C)  
    # Heterogeneous freezing increases rapidly below -5°C
    freeze_efficiency = jnp.where(
        dt_freeze > 5.0,
        1.0 - jnp.exp(-0.5 * (dt_freeze - 5.0)),
        0.0
    )
    
    freeze_rate = freeze_efficiency * rain / config.tau_freeze
    freeze_rate = jnp.minimum(freeze_rate, rain / dt)
    
    return melt_rate, freeze_rate


def evaporation_sublimation(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    rain: jnp.ndarray,
    snow: jnp.ndarray,
    air_density: jnp.ndarray,
    config: MicrophysicsParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate evaporation of rain and sublimation of snow
    
    Args:
        temperature: Temperature (K)
        specific_humidity: Specific humidity (kg/kg)
        pressure: Pressure (Pa)
        rain: Rain mixing ratio (kg/kg)
        snow: Snow mixing ratio (kg/kg)
        air_density: Air density (kg/m³)
        config: Microphysics configuration
        
    Returns:
        Tuple of (rain_evap_rate, snow_sublim_rate) in kg/kg/s
    """
    from .shallow_clouds import saturation_specific_humidity
    
    # Saturation specific humidity
    qs = saturation_specific_humidity(pressure, temperature)
    
    # Subsaturation
    subsaturation = jnp.maximum(0.0, (qs - specific_humidity) / qs)
    
    # Rain evaporation
    rain_gm3 = rain * air_density * 1000.0
    rain_evap = jnp.where(
        rain > config.epsilon,
        config.cevaprain * subsaturation * rain_gm3**0.5 / air_density,
        0.0
    )
    
    # Snow sublimation
    snow_gm3 = snow * air_density * 1000.0  
    snow_sublim = jnp.where(
        snow > config.epsilon,
        config.cevapsnow * subsaturation * snow_gm3**0.5 / air_density,
        0.0
    )
    
    return rain_evap, snow_sublim


def sedimentation_flux(
    hydrometeor: jnp.ndarray,
    air_density: jnp.ndarray,
    dz: jnp.ndarray,
    terminal_velocity: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate sedimentation flux and tendency for a hydrometeor
    
    Uses upwind differencing with flux limiter to maintain stability.
    JAX-compatible implementation without loops.
    
    Args:
        hydrometeor: Hydrometeor mixing ratio (kg/kg) [nlev]
        air_density: Air density (kg/m³) [nlev]
        dz: Layer thickness (m) [nlev]
        terminal_velocity: Fall velocity (m/s) [nlev]
        dt: Time step (s)
        
    Returns:
        Tuple of (flux [nlev+1], tendency [nlev])
    """
    nlev = hydrometeor.shape[0]
    
    # Mass content (kg/m³)
    mass_content = hydrometeor * air_density
    
    # Calculate fluxes at each interface (upwind)
    # Flux from level k to k+1
    flux_unlimited = mass_content * terminal_velocity
    
    # CFL limiter to prevent overshooting
    max_flux = mass_content * dz / dt
    flux_limited = jnp.minimum(flux_unlimited, max_flux)
    
    # Build interface fluxes
    # flux[0] = 0 (top), flux[k+1] = flux from level k
    flux = jnp.concatenate([jnp.zeros(1), flux_limited])
    
    # Tendency from flux divergence
    # (flux_in - flux_out) / (dz * rho)
    flux_in = flux[:-1]  # Flux from above
    flux_out = flux[1:]  # Flux to below
    tendency = (flux_in - flux_out) / (dz * air_density)
    
    return flux, tendency


def cloud_microphysics(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    rain_water: jnp.ndarray,
    snow: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    air_density: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    droplet_number: jnp.ndarray,
    dt: float,
    config: Optional[MicrophysicsParameters] = None
) -> Tuple[MicrophysicsTendencies, MicrophysicsState]:
    """
    Main cloud microphysics scheme
    
    Computes tendencies from all microphysical processes including:
    - Autoconversion and accretion
    - Melting and freezing
    - Evaporation and sublimation
    - Sedimentation
    
    Args:
        temperature: Temperature (K) [nlev]
        specific_humidity: Specific humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        cloud_water: Cloud liquid water (kg/kg) [nlev]
        cloud_ice: Cloud ice (kg/kg) [nlev]
        rain_water: Rain water (kg/kg) [nlev]
        snow: Snow (kg/kg) [nlev]
        cloud_fraction: Cloud fraction [nlev]
        air_density: Air density (kg/m³) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        droplet_number: Droplet number concentration (1/kg) [nlev]
        dt: Time step (s)
        config: Microphysics configuration
        
    Returns:
        Tuple of (tendencies, state)
    """
    if config is None:
        config = MicrophysicsParameters.default()
    
    # Ensure all inputs are arrays
    temperature = jnp.atleast_1d(temperature)
    nlev = temperature.shape[0]
    
    # Initialize tendencies
    dtedt = jnp.zeros(nlev)
    dqdt = jnp.zeros(nlev)
    dqcdt = jnp.zeros(nlev)
    dqidt = jnp.zeros(nlev)
    dqrdt = jnp.zeros(nlev)
    dqsdt = jnp.zeros(nlev)
    
    # Calculate in-cloud values
    qc_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_water / cloud_fraction,
        0.0
    )
    qi_in_cloud = jnp.where(
        cloud_fraction > config.epsilon,
        cloud_ice / cloud_fraction,
        0.0
    )
    
    # 1. Autoconversion processes
    qc_auto = autoconversion_kk2000(
        cloud_water, cloud_fraction, air_density, droplet_number, dt, config
    )
    qi_auto = ice_autoconversion(
        cloud_ice, temperature, cloud_fraction, dt, config
    )
    
    # 2. Accretion processes
    qc_accr = accretion_rain_cloud(
        cloud_water, rain_water, cloud_fraction, air_density, config
    )
    qc_rime = snow_accretion(
        cloud_water, snow, temperature, air_density, True, config
    )
    qi_aggr = snow_accretion(
        cloud_ice, snow, temperature, air_density, False, config
    )
    
    # 3. Melting and freezing
    snow_melt, rain_freeze = melting_freezing(
        temperature, snow, rain_water, dt, config
    )
    
    # 4. Evaporation and sublimation
    rain_evap, snow_sublim = evaporation_sublimation(
        temperature, specific_humidity, pressure,
        rain_water, snow, air_density, config
    )
    
    # 5. Update tendencies from microphysical processes
    # Cloud water: loses to autoconversion, accretion, riming
    dqcdt = -(qc_auto + qc_accr + qc_rime)
    
    # Cloud ice: loses to autoconversion and aggregation
    dqidt = -(qi_auto + qi_aggr)
    
    # Rain: gains from warm processes and melting, loses to evaporation and freezing
    dqrdt = qc_auto + qc_accr + snow_melt - rain_evap - rain_freeze
    
    # Snow: gains from cold processes and freezing, loses to melting and sublimation
    dqsdt = qi_auto + qi_aggr + qc_rime + rain_freeze - snow_melt - snow_sublim
    
    # Humidity: gains from evaporation/sublimation
    dqdt = rain_evap + snow_sublim
    
    # Temperature: latent heat effects
    dtedt = (
        - alhc / cp * (rain_evap - qc_auto - qc_accr)  # Liquid phase changes
        - alhs / cp * (snow_sublim - qi_auto - qi_aggr - qc_rime)  # Ice phase changes  
        - alhf / cp * (snow_melt - rain_freeze)  # Melting/freezing
    )
    
    # 6. Sedimentation (using simple approach for now)
    # Calculate terminal velocities
    rain_gm3 = rain_water * air_density * 1000.0
    vt_rain = config.vt_rain_a * rain_gm3**config.vt_rain_b * 1e-3  # m/s
    
    snow_gm3 = snow * air_density * 1000.0
    vt_snow = config.vt_snow_a * snow_gm3**config.vt_snow_b * 1e-3  # m/s
    
    vt_ice = jnp.ones(nlev) * config.vt_ice
    
    # Simple sedimentation tendencies (more sophisticated version would use flux form)
    # For now, just remove with time scale based on fall speed and layer thickness
    ice_sedi = cloud_ice * vt_ice / (layer_thickness + config.epsilon)
    rain_sedi = rain_water * vt_rain / (layer_thickness + config.epsilon) 
    snow_sedi = snow * vt_snow / (layer_thickness + config.epsilon)
    
    # Update tendencies
    dqidt = dqidt - ice_sedi
    dqrdt = dqrdt - rain_sedi
    dqsdt = dqsdt - snow_sedi
    
    # Calculate precipitation fluxes (simplified - just from lowest level)
    rain_flux = jnp.zeros(nlev)
    snow_flux = jnp.zeros(nlev)
    rain_flux = rain_flux.at[-1].set(rain_sedi[-1] * air_density[-1] * layer_thickness[-1])
    snow_flux = snow_flux.at[-1].set(snow_sedi[-1] * air_density[-1] * layer_thickness[-1])
    
    # Surface precipitation
    precip_rain = rain_flux[-1]
    precip_snow = snow_flux[-1]
    
    # Create output structures
    tendencies = MicrophysicsTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dqcdt=dqcdt,
        dqidt=dqidt,
        dqrdt=dqrdt,
        dqsdt=dqsdt
    )
    
    state = MicrophysicsState(
        rain_flux=rain_flux,
        snow_flux=snow_flux,
        qc_in_cloud=qc_in_cloud,
        qi_in_cloud=qi_in_cloud,
        autoconv_rate=qc_auto,
        accretion_rate=qc_accr,
        melting_rate=snow_melt,
        freezing_rate=rain_freeze,
        precip_rain=jnp.array(precip_rain),
        precip_snow=jnp.array(precip_snow)
    )
    
    return tendencies, state