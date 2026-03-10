"""Updraft calculations for Tiedtke-Nordeng convection scheme

This module implements the updraft calculations including:
- Cloud base determination
- Entrainment and detrainment
- Moist ascent with condensation
- Buoyancy calculations

Based on ICON mo_cuascent.f90

Date: 2025-01-09
"""

import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple

from ..constants.physical_constants import (
    grav, cp, alhc
)
from .tiedtke_nordeng import (
    ConvectionParameters, saturation_mixing_ratio
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


def saturation_adjustment(
    temperature: jnp.ndarray,
    total_water: jnp.ndarray,
    pressure: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Adjust temperature and moisture for saturation
    
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
    """Calculate full updraft profile
    
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
        in_cloud_interior = jnp.logical_and(
            jnp.minimum(ktop, kbase) < k,
            k < jnp.maximum(ktop, kbase)
        )
        at_cloud_top = (k == ktop)
        should_compute = jnp.logical_or(in_cloud_interior, at_cloud_top)
        skip = jnp.logical_not(should_compute)

        def compute_updraft():
            # Base entrainment rate by convection type
            entr_base = jnp.where(ktype == 1, entrpen,
                                  jnp.where(ktype == 2, entrscv, entrmid))

            # Humidity-dependent entrainment: drier environment entrains more
            qs_env = saturation_mixing_ratio(pressure, env_temp)
            rh = jnp.clip(env_q / jnp.maximum(qs_env, 1e-10), 0.0, 1.0)
            humidity_factor = 1.0 + 2.0 * (1.0 - rh) ** 2

            entr = jnp.clip(entr_base * humidity_factor, 0.0, 0.01)

            # Turbulent detrainment: fraction of entrainment
            detr_turb = 0.5 * entr

            # Organized detrainment for deep convection (Fortran tan() profile).
            # The ICON cuentr subroutine uses a tan-based profile that produces
            # sharp detrainment near cloud top, unlike a symmetric Gaussian.
            cloud_depth = jnp.maximum(kbase - ktop, 1.0)
            # Fractional distance from base (0 at base, 1 at top)
            frac_height = jnp.clip((kbase - k) / cloud_depth, 0.0, 1.0)
            # tan() profile: gentle in lower cloud, sharp increase near top
            # Argument mapped to (-pi/4, pi/2) so tan ranges from ~-1 to inf
            tan_arg = jnp.pi * (0.75 * frac_height - 0.25)
            org_profile = jnp.maximum(jnp.tan(tan_arg), 0.0)
            # Normalize: peak value of tan(pi/2 * 0.75 - pi/4) is bounded
            # Scale strength with cloud depth
            detr_strength = 0.003 * jnp.sqrt(cloud_depth / 10.0)
            detr_org = jnp.where(ktype == 1, detr_strength * org_profile, 0.0)

            detr = detr_turb + detr_org
            
            # Safe array indexing - clamp k+1 to valid range
            next_level = jnp.minimum(k + 1, nlev - 1)
            
            # Mass flux change
            dmf_entr = entr * carry.mfu[next_level] * dz
            dmf_detr = detr * carry.mfu[next_level] * dz
            
            # Update mass flux
            mfu_new = jnp.maximum(carry.mfu[next_level] + dmf_entr - dmf_detr, 0.0)
            
            # Proper mixing with entrainment
            # When mass flux is negligible, use environmental values instead of dividing by tiny numbers
            mfu_threshold = 1e-6  # kg/m²/s - below this, updraft is negligible

            def compute_updraft_properties():
                # Avoid division by zero
                if_mfu = 1.0 / jnp.maximum(mfu_new, 1e-10)

                # Total water and energy after mixing
                total_water = (carry.qu[next_level] + carry.lu[next_level]) * carry.mfu[next_level] + env_q * dmf_entr
                total_water = total_water * if_mfu

                # Temperature after mixing (dry static energy conservation)
                temp_mix = carry.tu[next_level] * carry.mfu[next_level] + env_temp * dmf_entr
                temp_mix = temp_mix * if_mfu

                # Saturation adjustment
                return saturation_adjustment(temp_mix, total_water, pressure)

            def use_environmental_values():
                # When updraft mass flux is negligible, use environmental values
                return env_temp, env_q, jnp.array(0.0)

            # Use environmental values when mass flux is too small
            tu_new, qu_new, lu_new = lax.cond(
                mfu_new > mfu_threshold,
                compute_updraft_properties,
                use_environmental_values
            )
            
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