"""
Downdraft calculations for Tiedtke-Nordeng convection scheme

This module implements the downdraft calculations including:
- Level of free sinking (LFS) determination
- Downdraft entrainment and detrainment
- Evaporative cooling
- Moist descent

Based on ICON mo_cudescent.f90

Date: 2025-01-09
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import NamedTuple, Tuple
from functools import partial

from ..constants.physical_constants import (
    grav, rd, rv, cp, eps, tmelt, alhc, alhs
)
from .tiedtke_nordeng import (
    ConvectionConfig, saturation_mixing_ratio
)
from .updraft import saturation_adjustment


class DowndraftState(NamedTuple):
    """State variables for downdraft calculation"""
    td: jnp.ndarray      # Downdraft temperature (K)
    qd: jnp.ndarray      # Downdraft specific humidity (kg/kg)
    mfd: jnp.ndarray     # Downdraft mass flux (kg/m²/s) - negative values
    lfs: int             # Level of free sinking
    active: bool         # Whether downdraft is active


def wetbulb_temperature(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate wet-bulb temperature and humidity
    
    Simplified version - full implementation would iterate
    
    Args:
        temperature: Environmental temperature (K)
        humidity: Environmental humidity (kg/kg)
        pressure: Pressure (Pa)
        
    Returns:
        Tuple of (wetbulb_temp, wetbulb_humidity)
    """
    # Get saturation values
    qs = saturation_mixing_ratio(pressure, temperature)
    
    # If already saturated, wet-bulb equals dry-bulb
    is_saturated = humidity >= qs
    
    def calculate_wetbulb():
        # Simplified: assume wet-bulb is slightly cooler
        # Full version would iterate to find equilibrium
        cooling = (qs - humidity) * alhc / cp
        twb = temperature - 0.3 * cooling  # Damping factor
        qwb = saturation_mixing_ratio(pressure, twb)
        return twb, qwb
    
    def already_saturated():
        return temperature, humidity
    
    return lax.cond(is_saturated, already_saturated, calculate_wetbulb)


def find_lfs(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    updraft_temp: jnp.ndarray,
    updraft_humid: jnp.ndarray,
    updraft_mf: jnp.ndarray,
    precip_rate: jnp.ndarray,
    kbase: int,
    ktop: int,
    config: ConvectionConfig
) -> Tuple[int, bool]:
    """
    Find level of free sinking for downdraft initiation
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        updraft_temp: Updraft temperature (K) [nlev]
        updraft_humid: Updraft humidity (kg/kg) [nlev]
        updraft_mf: Updraft mass flux (kg/m²/s) [nlev]
        precip_rate: Precipitation rate (kg/m²/s)
        kbase: Cloud base level
        ktop: Cloud top level
        config: Convection configuration
        
    Returns:
        Tuple of (lfs_level, found_lfs)
    """
    nlev = len(temperature)
    
    # Scan from cloud top down to find LFS
    def check_lfs(k):
        # Skip if outside cloud
        if k < ktop or k > kbase:
            return False, 0.0
            
        # Calculate wet-bulb values for environment
        twb, qwb = wetbulb_temperature(temperature[k], humidity[k], pressure[k])
        
        # Mix 50% cloud air with 50% environmental air at wet-bulb
        t_mix = 0.5 * (updraft_temp[k] + twb)
        q_mix = 0.5 * (updraft_humid[k] + qwb)
        
        # Calculate buoyancy
        vt_mix = t_mix * (1.0 + 0.608 * q_mix)
        vt_env = temperature[k] * (1.0 + 0.608 * humidity[k])
        buoyancy = (vt_mix - vt_env) / vt_env
        
        # Condensation in downdraft
        condensation = humidity[k] - qwb
        
        # Minimum mass flux threshold
        min_flux = config.cmfcmin * updraft_mf[kbase]
        
        # Check LFS criteria:
        # 1. Negative buoyancy
        # 2. Sufficient precipitation to maintain downdraft
        is_lfs = jnp.logical_and(
            buoyancy < 0.0,
            precip_rate > 10.0 * min_flux * condensation
        )
        
        return is_lfs, buoyancy
    
    # Find first level that satisfies LFS criteria
    lfs_found = False
    lfs_level = ktop
    
    for k in range(ktop, kbase + 1):
        is_lfs, buoy = check_lfs(k)
        if is_lfs and not lfs_found:
            lfs_level = k
            lfs_found = True
            break
    
    return lfs_level, lfs_found


def downdraft_step(
    carry: DowndraftState,
    level_inputs: Tuple
) -> Tuple[DowndraftState, DowndraftState]:
    """
    Single step of downdraft calculation for use with lax.scan
    
    Args:
        carry: Current downdraft state
        level_inputs: Environment variables at current level
        
    Returns:
        Tuple of (updated_carry, output_state)
    """
    k, env_temp, env_q, pressure, dz, rho, precip, config = level_inputs
    
    # Skip if downdraft not active or above LFS
    skip = jnp.logical_or(~carry.active, k < carry.lfs)
    
    def compute_downdraft():
        # Entrainment rate for downdrafts
        entr = config.entrscv * 0.5  # Reduced entrainment for downdrafts
        
        # Mass flux change due to entrainment (no detrainment in downdraft)
        dmf_entr = entr * jnp.abs(carry.mfd[k-1]) * dz
        
        # Update mass flux (more negative)
        mfd_new = carry.mfd[k-1] - dmf_entr
        
        # Mix in environmental air
        if_mfd = 1.0 / jnp.maximum(jnp.abs(mfd_new), 1e-10)
        
        # Temperature after mixing
        temp_mix = carry.td[k-1] * jnp.abs(carry.mfd[k-1]) + env_temp * dmf_entr
        temp_mix = temp_mix * if_mfd
        
        # Humidity after mixing  
        q_mix = carry.qd[k-1] * jnp.abs(carry.mfd[k-1]) + env_q * dmf_entr
        q_mix = q_mix * if_mfd
        
        # Evaporative cooling from precipitation
        # Amount of rain that can evaporate
        qs = saturation_mixing_ratio(pressure, temp_mix)
        evap_potential = jnp.maximum(qs - q_mix, 0.0)
        
        # Actual evaporation limited by available precipitation
        evap_rate = jnp.minimum(
            config.cevapcu * evap_potential * jnp.abs(mfd_new),
            precip
        )
        
        # Update temperature and humidity due to evaporation
        td_new = temp_mix - alhc * evap_rate / (cp * jnp.abs(mfd_new))
        qd_new = q_mix + evap_rate / jnp.abs(mfd_new)
        
        # Ensure physical bounds
        td_new = jnp.clip(td_new, 100.0, 400.0)
        qd_new = jnp.maximum(qd_new, 0.0)
        
        # Check buoyancy - stop downdraft if becomes positively buoyant
        vt_down = td_new * (1.0 + 0.608 * qd_new)
        vt_env = env_temp * (1.0 + 0.608 * env_q)
        buoyancy = (vt_down - vt_env) / vt_env
        
        # Continue only if negatively buoyant
        mfd_new = lax.cond(
            buoyancy < 0.0,
            lambda: mfd_new,
            lambda: 0.0
        )
        
        # Update state
        new_state = carry._replace(
            td=carry.td.at[k].set(td_new),
            qd=carry.qd.at[k].set(qd_new),
            mfd=carry.mfd.at[k].set(mfd_new),
            active=jnp.abs(mfd_new) > config.cmfcmin
        )
        
        return new_state
    
    # Skip calculation if appropriate
    updated_state = lax.cond(skip, lambda: carry, compute_downdraft)
    
    return updated_state, updated_state


def calculate_downdraft(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    rho: jnp.ndarray,
    updraft_state,  # UpdatedraftState from updraft.py
    precip_rate: jnp.ndarray,
    kbase: int,
    ktop: int,
    config: ConvectionConfig
) -> DowndraftState:
    """
    Calculate full downdraft profile
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev]
        height: Height (m) [nlev]
        rho: Air density (kg/m³) [nlev]
        updraft_state: Computed updraft state
        precip_rate: Column precipitation rate (kg/m²/s)
        kbase: Cloud base level
        ktop: Cloud top level
        config: Convection configuration
        
    Returns:
        DowndraftState with computed profiles
    """
    nlev = len(temperature)
    
    # Find level of free sinking
    lfs, has_lfs = find_lfs(
        temperature, humidity, pressure,
        updraft_state.tu, updraft_state.qu, updraft_state.mfu,
        precip_rate, kbase, ktop, config
    )
    
    # Initialize downdraft state
    td_init = temperature.copy()
    qd_init = humidity.copy()
    mfd_init = jnp.zeros(nlev)
    
    # If LFS found, initialize downdraft there
    if has_lfs:
        # Mix cloud and environmental air at LFS
        twb, qwb = wetbulb_temperature(
            temperature[lfs], humidity[lfs], pressure[lfs]
        )
        td_init = td_init.at[lfs].set(0.5 * (updraft_state.tu[lfs] + twb))
        qd_init = qd_init.at[lfs].set(0.5 * (updraft_state.qu[lfs] + qwb))
        
        # Initial downdraft mass flux (fraction of updraft mass flux)
        mfd_init = mfd_init.at[lfs].set(
            -config.cmfctop * updraft_state.mfu[kbase]
        )
    
    initial_state = DowndraftState(
        td=td_init,
        qd=qd_init,
        mfd=mfd_init,
        lfs=lfs,
        active=has_lfs
    )
    
    # Calculate layer thicknesses
    dz = jnp.diff(height)
    dz = jnp.concatenate([dz, jnp.array([dz[-1]])])
    
    # Prepare inputs for scan
    k_levels = jnp.arange(nlev)
    level_inputs = (
        k_levels, temperature, humidity, pressure, 
        dz, rho, jnp.full(nlev, precip_rate), config
    )
    
    # Use scan to compute downdraft from LFS downward
    final_state, all_states = lax.scan(
        partial(downdraft_step),
        initial_state,
        level_inputs
    )
    
    return final_state