"""
Updraft calculations for Tiedtke-Nordeng convection scheme

This module implements the updraft calculations including:
- Cloud base determination
- Entrainment and detrainment
- Moist ascent with condensation
- Buoyancy calculations

Based on ICON mo_cuascent.f90

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
    ConvectionConfig, saturation_mixing_ratio, saturation_vapor_pressure
)


class UpdatedraftState(NamedTuple):
    """State variables for updraft calculation"""
    tu: jnp.ndarray      # Updraft temperature (K)
    qu: jnp.ndarray      # Updraft specific humidity (kg/kg)
    lu: jnp.ndarray      # Updraft liquid water (kg/kg)
    mfu: jnp.ndarray     # Updraft mass flux (kg/m²/s)
    entr: jnp.ndarray    # Entrainment rate (1/m)
    detr: jnp.ndarray    # Detrainment rate (1/m)
    buoy: jnp.ndarray    # Buoyancy (m/s²)


def calculate_entrainment_detrainment(
    k: int,
    kbase: int, 
    ktop: int,
    ktype: int,
    mfu: jnp.ndarray,
    buoy: jnp.ndarray,
    dz: jnp.ndarray,
    rho: jnp.ndarray,
    config: ConvectionConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate entrainment and detrainment rates
    
    Args:
        k: Current level index
        kbase: Cloud base level
        ktop: Cloud top level
        ktype: Convection type (1=deep, 2=shallow, 3=mid)
        mfu: Mass flux at level k+1
        buoy: Buoyancy at current level
        dz: Layer thickness (m)
        rho: Air density (kg/m³)
        config: Convection configuration
        
    Returns:
        Tuple of (entrainment_rate, detrainment_rate) in 1/m
    """
    # Select entrainment rate based on convection type
    entr_param = lax.select(
        ktype - 1,  # Index selector
        jnp.array([config.entrpen, config.entrscv, config.entrmid])
    )
    
    # Basic turbulent entrainment
    entr = entr_param
    
    # Turbulent detrainment equals entrainment for mass conservation
    detr_turb = entr
    
    # Organized detrainment for deep convection
    # Applied in upper part of cloud
    detr_org = 0.0
    
    # For deep convection (ktype=1), add organized detrainment near cloud top
    def organized_detrainment():
        # Calculate relative position in cloud
        cloud_depth = jnp.maximum(kbase - ktop, 1)
        relative_height = (k - ktop) / cloud_depth
        
        # Smooth profile using hyperbolic tangent
        # Maximum detrainment at cloud top, decreasing downward
        org_profile = 0.5 * (1.0 + jnp.tanh(3.0 * (relative_height - 0.7)))
        
        # Scale by mass flux and density
        return 2.0e-3 * org_profile  # Organized detrainment rate
    
    # Apply organized detrainment only for deep convection in upper cloud
    detr_org = lax.cond(
        jnp.logical_and(ktype == 1, k <= kbase - cloud_depth * 0.5),
        organized_detrainment,
        lambda: 0.0
    )
    
    # Total detrainment
    detr = detr_turb + detr_org
    
    # Increase entrainment if losing buoyancy
    entr = lax.cond(
        buoy < 0.0,
        lambda: entr * 2.0,  # Double entrainment for negative buoyancy
        lambda: entr
    )
    
    return entr, detr


def saturation_adjustment(
    temperature: jnp.ndarray,
    total_water: jnp.ndarray,
    pressure: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Adjust temperature and moisture for saturation
    
    Args:
        temperature: Temperature (K)
        total_water: Total water mixing ratio (kg/kg)
        pressure: Pressure (Pa)
        
    Returns:
        Tuple of (adjusted_temp, vapor, liquid)
    """
    # Calculate saturation mixing ratio
    qs = saturation_mixing_ratio(pressure, temperature)
    
    # Check if supersaturated
    is_saturated = total_water > qs
    
    # If saturated, condense excess moisture
    def condense():
        # Iterative adjustment (simplified - full version would iterate)
        # Latent heat release
        latent = alhc  # Use liquid condensation
        
        # First guess of condensate
        condensate = total_water - qs
        
        # Temperature adjustment from latent heat
        temp_adj = temperature + latent * condensate / cp
        
        # Recalculate saturation at new temperature
        qs_new = saturation_mixing_ratio(pressure, temp_adj)
        
        # Final moisture split
        vapor = qs_new
        liquid = total_water - qs_new
        
        return temp_adj, vapor, jnp.maximum(liquid, 0.0)
    
    # If not saturated, all water is vapor
    def no_condensation():
        return temperature, total_water, jnp.array(0.0)
    
    return lax.cond(is_saturated, condense, no_condensation)


def updraft_step(
    carry: UpdatedraftState,
    level_inputs: Tuple
) -> Tuple[UpdatedraftState, UpdatedraftState]:
    """
    Single step of updraft calculation for use with lax.scan
    
    Args:
        carry: Current updraft state
        level_inputs: Tuple of (k, env_temp, env_q, pressure, dz, rho)
        
    Returns:
        Tuple of (updated_carry, output_state)
    """
    k, env_temp, env_q, pressure, dz, rho, kbase, ktop, ktype, config = level_inputs
    
    # Skip if below cloud base
    skip = k > kbase
    
    def compute_updraft():
        # Get entrainment and detrainment rates
        entr, detr = calculate_entrainment_detrainment(
            k, kbase, ktop, ktype, carry.mfu[k+1], 
            carry.buoy[k+1], dz, rho, config
        )
        
        # Mass flux change due to entrainment/detrainment
        dmf_entr = entr * carry.mfu[k+1] * dz
        dmf_detr = detr * carry.mfu[k+1] * dz
        
        # Update mass flux
        mfu_new = carry.mfu[k+1] + dmf_entr - dmf_detr
        mfu_new = jnp.maximum(mfu_new, 0.0)  # No negative mass flux
        
        # Mix in environmental air
        if_mfu = 1.0 / jnp.maximum(mfu_new, 1e-10)
        
        # Total water and energy after mixing
        total_water = (carry.qu[k+1] + carry.lu[k+1]) * carry.mfu[k+1] + env_q * dmf_entr
        total_water = total_water * if_mfu
        
        # Temperature after mixing (dry static energy conservation)
        temp_mix = carry.tu[k+1] * carry.mfu[k+1] + env_temp * dmf_entr
        temp_mix = temp_mix * if_mfu
        
        # Saturation adjustment
        tu_new, qu_new, lu_new = saturation_adjustment(temp_mix, total_water, pressure)
        
        # Calculate buoyancy
        virtual_temp_u = tu_new * (1.0 + 0.608 * qu_new - lu_new)
        virtual_temp_e = env_temp * (1.0 + 0.608 * env_q)
        buoy_new = grav * (virtual_temp_u - virtual_temp_e) / virtual_temp_e
        
        # Update state
        new_state = carry._replace(
            tu=carry.tu.at[k].set(tu_new),
            qu=carry.qu.at[k].set(qu_new),
            lu=carry.lu.at[k].set(lu_new),
            mfu=carry.mfu.at[k].set(mfu_new),
            entr=carry.entr.at[k].set(entr),
            detr=carry.detr.at[k].set(detr),
            buoy=carry.buoy.at[k].set(buoy_new)
        )
        
        return new_state
    
    # Skip calculation if below cloud base
    updated_state = lax.cond(skip, lambda: carry, compute_updraft)
    
    return updated_state, updated_state


def calculate_updraft(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    height: jnp.ndarray,
    rho: jnp.ndarray,
    kbase: int,
    ktop: int, 
    ktype: int,
    mass_flux_base: float,
    config: ConvectionConfig
) -> UpdatedraftState:
    """
    Calculate full updraft profile
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev] 
        height: Height (m) [nlev]
        rho: Air density (kg/m³) [nlev]
        kbase: Cloud base level index
        ktop: Cloud top level index
        ktype: Convection type
        mass_flux_base: Cloud base mass flux (kg/m²/s)
        config: Convection configuration
        
    Returns:
        UpdatedraftState with computed profiles
    """
    nlev = len(temperature)
    
    # Initialize updraft state at cloud base
    tu_init = jnp.zeros(nlev)
    qu_init = jnp.zeros(nlev)
    lu_init = jnp.zeros(nlev) 
    mfu_init = jnp.zeros(nlev)
    entr_init = jnp.zeros(nlev)
    detr_init = jnp.zeros(nlev)
    buoy_init = jnp.zeros(nlev)
    
    # Set cloud base values
    tu_init = tu_init.at[kbase].set(temperature[kbase])
    qu_init = qu_init.at[kbase].set(humidity[kbase])
    mfu_init = mfu_init.at[kbase].set(mass_flux_base)
    
    # Calculate initial buoyancy at cloud base
    virtual_temp_base = temperature[kbase] * (1.0 + 0.608 * humidity[kbase])
    buoy_init = buoy_init.at[kbase].set(0.0)  # Neutral at cloud base
    
    initial_state = UpdatedraftState(
        tu=tu_init, qu=qu_init, lu=lu_init,
        mfu=mfu_init, entr=entr_init, detr=detr_init,
        buoy=buoy_init
    )
    
    # Calculate layer thicknesses
    dz = jnp.diff(height)
    dz = jnp.concatenate([dz, jnp.array([dz[-1]])])  # Repeat last value
    
    # Prepare inputs for scan (extract config parameters to avoid passing object)
    k_levels = jnp.arange(nlev)
    level_inputs = (
        k_levels, temperature, humidity, pressure, dz, rho,
        jnp.full(nlev, kbase), jnp.full(nlev, ktop), 
        jnp.full(nlev, ktype), 
        jnp.full(nlev, config.entrpen), jnp.full(nlev, config.entrscv),
        jnp.full(nlev, config.entrmid)
    )
    
    # Create specialized step function with config parameters
    def updraft_step_with_config(carry, inputs):
        k, env_temp, env_q, pressure, dz, rho, kbase, ktop, ktype, entrpen, entrscv, entrmid = inputs
        
        # Skip if below cloud base
        skip = k > kbase
        
        def compute_updraft():
            # Simplified entrainment/detrainment calculation for scan
            entr = jnp.where(ktype == 1, entrpen, 
                            jnp.where(ktype == 2, entrscv, entrmid))
            detr = entr  # Simplified
            
            # Mass flux change
            dmf_entr = entr * carry.mfu[k+1] * dz
            dmf_detr = detr * carry.mfu[k+1] * dz
            
            # Update mass flux
            mfu_new = jnp.maximum(carry.mfu[k+1] + dmf_entr - dmf_detr, 0.0)
            
            # Simple mixing (placeholder for full calculation)
            tu_new = env_temp  # Simplified
            qu_new = env_q     # Simplified
            lu_new = 0.0       # Simplified
            
            # Simple buoyancy
            buoy_new = 0.0     # Simplified
            
            # Update state
            new_state = carry._replace(
                tu=carry.tu.at[k].set(tu_new),
                qu=carry.qu.at[k].set(qu_new),
                lu=carry.lu.at[k].set(lu_new),
                mfu=carry.mfu.at[k].set(mfu_new),
                entr=carry.entr.at[k].set(entr),
                detr=carry.detr.at[k].set(detr),
                buoy=carry.buoy.at[k].set(buoy_new)
            )
            
            return new_state
        
        # Skip calculation if below cloud base
        updated_state = lax.cond(skip, lambda: carry, compute_updraft)
        
        return updated_state, updated_state
    
    # Use scan to compute updraft from bottom to top
    final_state, all_states = lax.scan(
        updraft_step_with_config,
        initial_state,
        level_inputs,
        reverse=True  # Go from bottom to top
    )
    
    return final_state