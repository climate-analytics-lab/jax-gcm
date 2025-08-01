"""
Flux calculations and tendency updates for Tiedtke-Nordeng convection

This module implements:
- Final mass flux adjustments
- Temperature and moisture tendency calculations
- Momentum transport
- Precipitation and cloud water/ice partitioning

Based on ICON mo_cufluxdts.f90

Date: 2025-01-09
"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import Tuple

from ..constants.physical_constants import (
    grav, cp, alhc, alhs, tmelt
)
from .tiedtke_nordeng import ConvectionParameters, ConvectionTendencies
from .updraft import UpdatedraftState
from .downdraft import DowndraftState


def calculate_precipitation_rate(
    updraft_state: UpdatedraftState,
    kbase: int,
    dt: float,
    config: ConvectionParameters
) -> jnp.ndarray:
    """
    Calculate surface precipitation rate from convection
    
    Args:
        updraft_state: Updraft calculation results
        kbase: Cloud base level
        dt: Time step (s)
        config: Convection configuration
        
    Returns:
        Surface precipitation rate (kg/m²/s)
    """
    # Integrate liquid water flux through cloud
    nlev = len(updraft_state.mfu)
    
    # Precipitation conversion efficiency
    precip_eff = config.cprcon
    
    # Calculate precipitation production at each level using JAX-compatible operations
    k_levels = jnp.arange(nlev)
    
    # Mask for levels at or below cloud base
    cloud_mask = k_levels >= kbase  # Note: k >= kbase means level at or below cloud base
    
    # Liquid water flux for all levels
    lw_flux = updraft_state.mfu * updraft_state.lu
    
    # Convert fraction to precipitation (only below cloud base)
    precip_prod = jnp.where(cloud_mask, precip_eff * lw_flux, 0.0)
    
    # Surface precipitation is integral of production
    precip_rate = jnp.sum(precip_prod)
    
    return precip_rate


def calculate_cloud_water_ice(
    temperature: jnp.ndarray,
    updraft_lw: jnp.ndarray,
    updraft_mf: jnp.ndarray,
    downdraft_mf: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Partition cloud condensate into liquid and ice
    
    Args:
        temperature: Temperature profile (K)
        updraft_lw: Updraft liquid water (kg/kg)
        updraft_mf: Updraft mass flux (kg/m²/s)
        downdraft_mf: Downdraft mass flux (kg/m²/s)
        
    Returns:
        Tuple of (cloud_water, cloud_ice) in kg/kg
    """
    # Temperature thresholds for ice formation
    t_ice = tmelt - 40.0  # All ice below this
    t_water = tmelt       # All water above this
    
    # Linear transition between water and ice
    ice_frac = jnp.clip((t_water - temperature) / (t_water - t_ice), 0.0, 1.0)
    water_frac = 1.0 - ice_frac
    
    # Net vertical mass flux
    net_mf = updraft_mf + downdraft_mf  # downdraft is negative
    
    # Cloud fraction estimate (simplified)
    cloud_frac = jnp.clip(net_mf / 0.1, 0.0, 1.0)  # 0.1 kg/m²/s for full cloud
    
    # In-cloud condensate
    in_cloud_lw = updraft_lw * updraft_mf / jnp.maximum(net_mf, 1e-10)
    
    # Grid-mean cloud water and ice
    cloud_water = cloud_frac * in_cloud_lw * water_frac
    cloud_ice = cloud_frac * in_cloud_lw * ice_frac
    
    return cloud_water, cloud_ice


def calculate_tendencies(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    pressure: jnp.ndarray,
    rho: jnp.ndarray,
    updraft_state: UpdatedraftState,
    downdraft_state: DowndraftState,
    kbase: int,
    ktop: int,
    dt: float,
    config: ConvectionParameters
) -> ConvectionTendencies:
    """
    Calculate final tendencies from convective fluxes
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental humidity (kg/kg) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        pressure: Pressure (Pa) [nlev]
        rho: Air density (kg/m³) [nlev]
        updraft_state: Computed updraft state
        downdraft_state: Computed downdraft state
        kbase: Cloud base level
        ktop: Cloud top level
        dt: Time step (s)
        config: Convection configuration
        
    Returns:
        ConvectionTendencies with all tendency terms
    """
    nlev = len(temperature)
    
    # Calculate mass flux divergence at each level using JAX-compatible operations
    
    # Layer thickness (pressure)
    dp = jnp.diff(pressure, axis=0)
    
    diff_updraft = jnp.diff(updraft_state.mfu, axis=0)
    diff_downdraft = jnp.diff(downdraft_state.mfd, axis=0)

    # FIXME: check signs (downdraft)
    mass_flux_div = - diff_updraft - diff_downdraft
    factor = mass_flux_div / (rho[:-1] * dp) * grav # Factor for tendency calculation
    
    temp_flux_div = - jnp.diff(updraft_state.tu * updraft_state.mfu + downdraft_state.td * downdraft_state.mfd, axis=0) # FIXME this is blowing up level 1
    q_flux_div = - jnp.diff(updraft_state.qu * updraft_state.mfu + downdraft_state.qd * downdraft_state.mfd, axis=0)
    lh_source = - alhc * jnp.diff(updraft_state.lu * updraft_state.mfu, axis=0) # Include latent heat from condensation/evaporation

    dtedt_k_levels = (temp_flux_div + lh_source/cp) * factor
    dqdt_k_levels = q_flux_div * factor

    # Downdraft momentum transport (assumes downdraft winds ~ environmental winds)
    u_downdraft_flux = - jnp.diff(u_wind * downdraft_state.mfd, axis=0)
    v_downdraft_flux = - jnp.diff(v_wind * downdraft_state.mfd, axis=0)

    # Enhanced momentum tendencies
    def calculate_momentum_transport():
        # Simplified momentum transport using environmental winds
        # Updrafts and downdrafts carry momentum similar to their source levels
        
        # Updraft momentum transport (assumes updraft winds ~ cloud base winds)
        u_cloud_base = u_wind[kbase, None]
        v_cloud_base = v_wind[kbase, None]
        u_updraft_flux = - diff_updraft * u_cloud_base
        v_updraft_flux = - diff_updraft * v_cloud_base
        
        # Total momentum flux divergence
        u_total_flux = u_updraft_flux + u_downdraft_flux
        v_total_flux = v_updraft_flux + v_downdraft_flux
        
        # Momentum tendency from mass flux transport
        dudt_transport = u_total_flux * factor
        dvdt_transport = v_total_flux * factor

        # Add pressure gradient force effect (simplified)
        # Vertical momentum mixing tends to accelerate flow toward cloud base winds
        pgf_efficiency = 0.3  # Moderate coupling strength
        dudt_pgf = pgf_efficiency * (u_cloud_base - u_wind[:-1]) * mass_flux_div * factor
        dvdt_pgf = pgf_efficiency * (v_cloud_base - v_wind[:-1]) * mass_flux_div * factor

        return dudt_transport + dudt_pgf, dvdt_transport + dvdt_pgf
    
    dudt_k_levels, dvdt_k_levels = lax.cond(
        config.cmfctop > 0,
        calculate_momentum_transport,
        lambda: (jnp.zeros(nlev-1), jnp.zeros(nlev-1)),
    )
    
    # Make tendency arrays
    dtedt = jnp.zeros(nlev).at[:-1].set(dtedt_k_levels)
    dqdt = jnp.zeros(nlev).at[:-1].set(dqdt_k_levels)
    dudt = jnp.zeros(nlev).at[:-1].set(dudt_k_levels)
    dvdt = jnp.zeros(nlev).at[:-1].set(dvdt_k_levels)

    # Calculate precipitation rate
    precip_rate = calculate_precipitation_rate(
        updraft_state, kbase, dt, config
    )
    
    # Partition cloud condensate
    qc_conv, qi_conv = calculate_cloud_water_ice(
        temperature, updraft_state.lu, 
        updraft_state.mfu, downdraft_state.mfd
    )
    
    # Apply time step
    dtedt = dtedt / dt
    dqdt = dqdt / dt
    dudt = dudt / dt
    dvdt = dvdt / dt
    
    # Calculate fixed qc/qi tendencies (simplified approach)
    # These represent the tendency of cloud water and ice from convective transport
    nlev = len(temperature)
    dqc_dt = jnp.zeros(nlev)  # Cloud water tendency from convection
    dqi_dt = jnp.zeros(nlev)  # Cloud ice tendency from convection
    
    # For levels with convective activity, add some tendency
    # This is a simplified approach - more sophisticated transport would be needed
    conv_levels = (jnp.arange(nlev) >= kbase) & (jnp.arange(nlev) <= ktop)

    # Simple cloud water/ice production based on updraft liquid water
    dqc_dt = jnp.where(conv_levels, qc_conv * 0.1 / dt, 0.0)
    dqi_dt = jnp.where(conv_levels, qi_conv * 0.1 / dt, 0.0)
    
    return ConvectionTendencies(
        dtedt=dtedt,
        dqdt=dqdt,
        dudt=dudt,
        dvdt=dvdt,
        qc_conv=qc_conv,
        qi_conv=qi_conv,
        precip_conv=precip_rate,
        dqc_dt=dqc_dt,
        dqi_dt=dqi_dt
    )


def mass_flux_closure(
    cape: jnp.ndarray,
    cin: jnp.ndarray,
    moisture_conv: jnp.ndarray,
    ktype: int,
    config: ConvectionParameters
) -> jnp.ndarray:
    """
    Determine cloud base mass flux using appropriate closure
    
    Args:
        cape: Convective available potential energy (J/kg)
        cin: Convective inhibition (J/kg)
        moisture_conv: Low-level moisture convergence (kg/m²/s)
        ktype: Convection type (1=deep, 2=shallow, 3=mid)
        config: Convection configuration
        
    Returns:
        Cloud base mass flux (kg/m²/s)
    """
    # Deep convection: CAPE closure
    def deep_closure():
        # Timescale for CAPE removal
        tau = config.tau
        
        # Mass flux to remove CAPE over timescale
        # Simplified - full version would iterate
        mf_cape = cape / (grav * tau)
        
        # Apply limits
        return jnp.clip(mf_cape, config.cmfcmin, config.cmfcmax)
    
    # Shallow convection: moisture convergence closure
    def shallow_closure():
        # Balance low-level moisture convergence
        # For shallow convection, also use CAPE but with different scaling
        # If no moisture convergence, use CAPE-based trigger for shallow convection
        cape_flux = cape / (grav * config.tau * 10.0)  # Weaker than deep convection
        moisture_flux = moisture_conv * 0.1  # Efficiency factor
        
        # Use the larger of the two triggers
        base_flux = jnp.maximum(cape_flux, moisture_flux)
        
        return jnp.clip(
            base_flux,
            config.cmfcmin * 10.0,  # Minimum for shallow convection
            config.cmfcmax * 0.3    # Lower limit for shallow
        )
    
    # Mid-level convection: hybrid closure
    def mid_closure():
        # Combination of CAPE and moisture
        return 0.5 * (deep_closure() + shallow_closure())
    
    # Select closure based on convection type using clipped index
    # Ensure index is in valid range [0, 2] for switch
    switch_index = jnp.clip(ktype - 1, 0, 2)
    
    return lax.switch(
        switch_index,
        [deep_closure, shallow_closure, mid_closure],
    )