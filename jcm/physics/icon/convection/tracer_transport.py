"""
Tracer transport for Tiedtke-Nordeng convection scheme

This module handles the convective transport of cloud water, cloud ice,
and additional tracers (chemical species, aerosols, etc.)

Date: 2025-01-09
"""

import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial
from jax import lax

from .updraft import UpdatedraftState
from .downdraft import DowndraftState
from ..constants.physical_constants import tmelt


class TracerIndices(NamedTuple):
    """Indices for different tracer types"""
    iqv: int = 0    # Water vapor (specific humidity)
    iqc: int = 1    # Cloud liquid water
    iqi: int = 2    # Cloud ice
    iqt: int = 3    # Start of additional tracers
    

class TracerTransport(NamedTuple):
    """Tracer transport state and tendencies"""
    # Tracer concentrations in updraft/downdraft
    tracer_u: jnp.ndarray   # [nlev, ntrac] - updraft tracer concentrations
    tracer_d: jnp.ndarray   # [nlev, ntrac] - downdraft tracer concentrations
    
    # Tracer mass fluxes
    mfuxt: jnp.ndarray      # [nlev, ntrac] - updraft tracer mass flux
    mfdxt: jnp.ndarray      # [nlev, ntrac] - downdraft tracer mass flux
    
    # Tendencies
    dtracer_dt: jnp.ndarray # [nlev, ntrac] - tracer tendencies
    
    # Cloud detrainment (special handling for qc and qi)
    detrain_qc: jnp.ndarray # [nlev] - detrained cloud water
    detrain_qi: jnp.ndarray # [nlev] - detrained cloud ice


def partition_cloud_detrainment(
    temperature: jnp.ndarray,
    detrain_total: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Partition detrained condensate into liquid and ice
    
    Args:
        temperature: Temperature at each level (K)
        detrain_total: Total detrained condensate (kg/kg)
        
    Returns:
        Tuple of (detrain_liquid, detrain_ice)
    """
    # Temperature thresholds
    t_ice = tmelt - 35.0   # All ice below this
    t_water = tmelt        # All water above this
    
    # Linear transition between water and ice
    ice_frac = jnp.clip((t_water - temperature) / (t_water - t_ice), 0.0, 1.0)
    water_frac = 1.0 - ice_frac
    
    detrain_qc = detrain_total * water_frac
    detrain_qi = detrain_total * ice_frac
    
    return detrain_qc, detrain_qi


def transport_tracers(
    tracers: jnp.ndarray,
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    rho: jnp.ndarray,
    updraft_state: UpdatedraftState,
    downdraft_state: DowndraftState,
    dt: float,
    indices: Optional[TracerIndices] = None
) -> TracerTransport:
    """
    Calculate tracer transport by convection
    
    Args:
        tracers: Environmental tracer concentrations [nlev, ntrac]
        temperature: Temperature profile (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        rho: Air density (kg/m³) [nlev]
        updraft_state: Computed updraft state
        downdraft_state: Computed downdraft state
        dt: Time step (s)
        indices: Tracer index definitions
        
    Returns:
        TracerTransport with all transport terms
    """
    if indices is None:
        indices = TracerIndices()
    
    nlev, ntrac = tracers.shape
    
    # Initialize tracer arrays
    tracer_u = jnp.zeros_like(tracers)
    tracer_d = jnp.zeros_like(tracers)
    mfuxt = jnp.zeros_like(tracers)
    mfdxt = jnp.zeros_like(tracers)
    dtracer_dt = jnp.zeros_like(tracers)
    
    # Special handling for water vapor (already computed in main scheme)
    tracer_u = tracer_u.at[:, indices.iqv].set(updraft_state.qu)
    tracer_d = tracer_d.at[:, indices.iqv].set(downdraft_state.qd)
    
    # Transport additional tracers (chemical species, aerosols, etc.)
    if ntrac > indices.iqt:
        # Simple approach: tracers follow air parcels
        for itrac in range(indices.iqt, ntrac):
            # In updraft: mix with environmental air based on entrainment
            for k in range(nlev):
                if updraft_state.mfu[k] > 0:
                    # Entrainment dilutes tracer concentration
                    tracer_u = tracer_u.at[k, itrac].set(tracers[k, itrac])
                    
            # In downdraft: similar mixing
            for k in range(nlev):
                if downdraft_state.mfd[k] < 0:
                    tracer_d = tracer_d.at[k, itrac].set(tracers[k, itrac])
    
    # Calculate mass fluxes for all tracers
    for itrac in range(ntrac):
        mfuxt = mfuxt.at[:, itrac].set(updraft_state.mfu * tracer_u[:, itrac])
        mfdxt = mfdxt.at[:, itrac].set(downdraft_state.mfd * tracer_d[:, itrac])
    
    # Calculate tendencies from mass flux divergence
    for k in range(nlev-1):
        dp = pressure[k+1] - pressure[k]
        factor = 1.0 / (rho[k] * dp) * 9.81
        
        for itrac in range(indices.iqt, ntrac):
            # Mass flux divergence
            flux_div = (mfuxt[k, itrac] - mfuxt[k+1, itrac] +
                       mfdxt[k, itrac] - mfdxt[k+1, itrac])
            
            dtracer_dt = dtracer_dt.at[k, itrac].set(flux_div * factor / dt)
    
    # Handle cloud water and ice detrainment
    # Calculate detrainment from updraft liquid water
    detrain_total = jnp.zeros(nlev)
    for k in range(nlev):
        if updraft_state.mfu[k] > 0 and updraft_state.detr[k] > 0:
            # Detrain fraction of cloud water
            detrain_total = detrain_total.at[k].set(
                updraft_state.lu[k] * updraft_state.detr[k] * 
                updraft_state.mfu[k] / rho[k]
            )
    
    # Partition into liquid and ice
    detrain_qc, detrain_qi = partition_cloud_detrainment(temperature, detrain_total)
    
    # Add detrainment as source terms for cloud water and ice
    dtracer_dt = dtracer_dt.at[:, indices.iqc].set(detrain_qc / dt)
    dtracer_dt = dtracer_dt.at[:, indices.iqi].set(detrain_qi / dt)
    
    return TracerTransport(
        tracer_u=tracer_u,
        tracer_d=tracer_d,
        mfuxt=mfuxt,
        mfdxt=mfdxt,
        dtracer_dt=dtracer_dt,
        detrain_qc=detrain_qc,
        detrain_qi=detrain_qi
    )


def initialize_tracers(nlev: int, include_chemistry: bool = False) -> Tuple[jnp.ndarray, TracerIndices]:
    """
    Initialize tracer array with appropriate species
    
    Args:
        nlev: Number of vertical levels
        include_chemistry: Whether to include chemical tracers
        
    Returns:
        Tuple of (tracer_array, indices)
    """
    # Basic tracers: qv, qc, qi
    basic_tracers = 3
    
    # Add chemical species if requested
    if include_chemistry:
        # Example chemical species
        chem_species = 5  # e.g., O3, CO, NOx, SO2, DMS
        ntrac = basic_tracers + chem_species
    else:
        ntrac = basic_tracers
    
    # Initialize with small positive values
    tracers = jnp.ones((nlev, ntrac)) * 1e-12
    
    # Set realistic initial values for water species
    tracers = tracers.at[:, 0].set(1e-3)  # qv: 1 g/kg
    tracers = tracers.at[:, 1].set(0.0)   # qc: initially no clouds
    tracers = tracers.at[:, 2].set(0.0)   # qi: initially no ice
    
    indices = TracerIndices()
    
    return tracers, indices