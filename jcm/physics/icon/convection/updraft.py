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
    ConvectionParameters, saturation_mixing_ratio, saturation_vapor_pressure
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
    env_temp: jnp.ndarray,
    env_humidity: jnp.ndarray,
    updraft_temp: jnp.ndarray,
    updraft_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    config: ConvectionParameters
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate enhanced entrainment and detrainment rates
    
    Args:
        k: Current level index
        kbase: Cloud base level
        ktop: Cloud top level
        ktype: Convection type (1=deep, 2=shallow, 3=mid)
        mfu: Mass flux at level k+1
        buoy: Buoyancy at current level
        dz: Layer thickness (m)
        rho: Air density (kg/m³)
        env_temp: Environmental temperature (K)
        env_humidity: Environmental specific humidity (kg/kg)
        updraft_temp: Updraft temperature (K)
        updraft_humidity: Updraft specific humidity (kg/kg)
        pressure: Pressure (Pa)
        config: Convection configuration
        
    Returns:
        Tuple of (entrainment_rate, detrainment_rate) in 1/m
    """
    # Base entrainment rate based on convection type
    entr_base = lax.select(
        ktype - 1,  # Index selector
        jnp.array([config.entrpen, config.entrscv, config.entrmid])
    )
    
    # Environmental humidity dependence (dry air increases entrainment)
    from .tiedtke_nordeng import saturation_mixing_ratio
    qs_env = saturation_mixing_ratio(pressure, env_temp)
    relative_humidity = jnp.clip(env_humidity / qs_env, 0.0, 1.0)
    
    # Entrainment increases as RH decreases (more dry air entrainment)
    humidity_factor = 1.0 + 2.0 * (1.0 - relative_humidity)**2
    
    # Buoyancy dependence - enhanced entrainment for negative buoyancy
    buoyancy_factor = lax.cond(
        buoy < 0.0,
        lambda: 1.0 + 3.0 * jnp.abs(buoy),  # Increase based on negative buoyancy magnitude
        lambda: 1.0
    )
    
    # Thermal contrast dependence - more entrainment with larger temperature differences
    temp_contrast = jnp.abs(updraft_temp - env_temp)
    thermal_factor = 1.0 + 0.1 * temp_contrast  # Modest enhancement
    
    # Combined entrainment rate
    entr = entr_base * humidity_factor * buoyancy_factor * thermal_factor
    
    # Limit maximum entrainment to prevent instability
    entr = jnp.clip(entr, 0.0, 0.01)  # Max 1% per meter
    
    # Turbulent detrainment for mass conservation
    detr_turb = entr * 0.5  # Partial compensation
    
    # Enhanced organized detrainment for deep convection
    cloud_depth = jnp.maximum(kbase - ktop, 1)
    
    def enhanced_organized_detrainment():
        # Calculate relative position in cloud (0 = cloud top, 1 = cloud base)
        relative_height = (kbase - k) / cloud_depth
        
        # Enhanced profile with stronger detrainment in upper levels
        # Peak detrainment around 0.8-0.9 relative height (near cloud top)
        peak_position = 0.85
        width = 0.3
        
        # Gaussian-like profile centered near cloud top
        org_profile = jnp.exp(-0.5 * ((relative_height - peak_position) / width)**2)
        
        # Scale based on mass flux strength and cloud depth
        detr_strength = 0.003 * jnp.sqrt(cloud_depth / 10.0)  # Stronger for deeper clouds
        
        return detr_strength * org_profile
    
    # Apply enhanced organized detrainment for deep convection
    detr_org = lax.cond(
        jnp.logical_and(ktype == 1, k <= kbase),  # Throughout deep cloud
        enhanced_organized_detrainment,
        lambda: 0.0
    )
    
    # Total detrainment
    detr = detr_turb + detr_org
    
    # Minimum detrainment to ensure some mixing
    detr = jnp.maximum(detr, 0.0001)
    
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
        # Safe array indexing - handle potential out-of-bounds access
        next_level = jnp.minimum(k + 1, len(carry.mfu) - 1)
        
        # Get entrainment and detrainment rates
        entr, detr = calculate_entrainment_detrainment(
            k, kbase, ktop, ktype, carry.mfu[next_level], 
            carry.buoy[next_level], dz, rho,
            env_temp, env_q, carry.tu[k], carry.qu[k], pressure, config
        )
        
        # Mass flux change due to entrainment/detrainment
        dmf_entr = entr * carry.mfu[next_level] * dz
        dmf_detr = detr * carry.mfu[next_level] * dz
        
        # Update mass flux
        mfu_new = carry.mfu[next_level] + dmf_entr - dmf_detr
        mfu_new = jnp.maximum(mfu_new, 0.0)  # No negative mass flux
        
        # Mix in environmental air
        if_mfu = 1.0 / jnp.maximum(mfu_new, 1e-10)
        
        # Total water and energy after mixing
        total_water = (carry.qu[next_level] + carry.lu[next_level]) * carry.mfu[next_level] + env_q * dmf_entr
        total_water = total_water * if_mfu
        
        # Temperature after mixing (dry static energy conservation)
        temp_mix = carry.tu[next_level] * carry.mfu[next_level] + env_temp * dmf_entr
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
    layer_thickness: jnp.ndarray,
    rho: jnp.ndarray,
    kbase: int,
    ktop: int, 
    ktype: int,
    mass_flux_base: float,
    config: ConvectionParameters
) -> UpdatedraftState:
    """
    Calculate full updraft profile
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        pressure: Pressure (Pa) [nlev] 
        layer_thickness: Layer thickness (m) [nlev]
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
    
    # Prepare inputs for scan (extract config parameters to avoid passing object)
    k_levels = jnp.arange(nlev)
    level_inputs = (
        k_levels, temperature, humidity, pressure, layer_thickness, rho,
        jnp.full(nlev, kbase), jnp.full(nlev, ktop), 
        jnp.full(nlev, ktype), 
        jnp.full(nlev, config.entrpen), jnp.full(nlev, config.entrscv),
        jnp.full(nlev, config.entrmid)
    )
    
    # Create specialized step function with config parameters
    def updraft_step_with_config(carry, inputs):
        k, env_temp, env_q, pressure, dz, rho, kbase, ktop, ktype, entrpen, entrscv, entrmid = inputs
        
        # Skip if outside cloud layer or at cloud base (boundary condition)
        # Cloud base is the boundary condition, so we don't compute it
        # We only compute levels between cloud base and cloud top (exclusive of base)
        
        # For standard ordering (pressure decreasing): ktop < kbase, compute ktop to kbase-1
        # For reverse ordering: ktop > kbase, compute kbase+1 to ktop
        in_cloud_interior = jnp.logical_and(
            jnp.minimum(ktop, kbase) < k,
            k < jnp.maximum(ktop, kbase)
        )
        
        # Also include cloud top in the calculation
        at_cloud_top = (k == ktop)
        
        # Process cloud interior and cloud top, but not cloud base
        should_compute = jnp.logical_or(in_cloud_interior, at_cloud_top)
        skip = jnp.logical_not(should_compute)
        
        def compute_updraft():
            # Entrainment/detrainment calculation with proper physics
            entr = jnp.where(ktype == 1, entrpen, 
                            jnp.where(ktype == 2, entrscv, entrmid))
            detr = entr  # Simplified - could be enhanced with organized detrainment
            
            # Safe array indexing - clamp k+1 to valid range
            next_level = jnp.minimum(k + 1, nlev - 1)
            
            # Mass flux change
            dmf_entr = entr * carry.mfu[next_level] * dz
            dmf_detr = detr * carry.mfu[next_level] * dz
            
            # Update mass flux
            mfu_new = jnp.maximum(carry.mfu[next_level] + dmf_entr - dmf_detr, 0.0)
            
            # Proper mixing with entrainment
            # Avoid division by zero
            if_mfu = 1.0 / jnp.maximum(mfu_new, 1e-10)
            
            # Total water and energy after mixing
            total_water = (carry.qu[next_level] + carry.lu[next_level]) * carry.mfu[next_level] + env_q * dmf_entr
            total_water = total_water * if_mfu
            
            # Temperature after mixing (dry static energy conservation)
            temp_mix = carry.tu[next_level] * carry.mfu[next_level] + env_temp * dmf_entr
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
    
    # Use scan to compute updraft from bottom to top
    final_state, all_states = lax.scan(
        updraft_step_with_config,
        initial_state,
        level_inputs,
        reverse=True  # Go from bottom to top
    )
    
    return final_state