"""
Two-stream radiative transfer solver

This module implements the two-stream approximation for radiative
transfer through a multi-layer atmosphere.

The implementation uses the Eddington approximation with the
adding method for combining layers.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple, Optional
# from functools import partial  # Not needed anymore

from .radiation_types import OpticalProperties


@jax.jit
def two_stream_coefficients(
    tau: jnp.ndarray, # FIXME: remove this if unused
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    mu0: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate two-stream coefficients.
    
    Using Eddington approximation (Meador & Weaver 1980).
    
    Args:
        tau: Optical depth
        ssa: Single scattering albedo
        g: Asymmetry factor
        mu0: Cosine of solar zenith angle (for SW only)
        
    Returns:
        Tuple of (gamma1, gamma2, gamma3, gamma4)
    """
    # Eddington approximation coefficients
    gamma1 = (7.0 - ssa * (4.0 + 3.0 * g)) / 4.0
    gamma2 = -(1.0 - ssa * (4.0 - 3.0 * g)) / 4.0
    
    if mu0 is not None:
        # Shortwave with solar angle
        gamma3 = (2.0 - 3.0 * g * mu0) / 4.0
        gamma4 = 1.0 - gamma3
    else:
        # Longwave (no direct beam)
        gamma3 = jnp.zeros_like(tau)
        gamma4 = jnp.ones_like(tau)
    
    return gamma1, gamma2, gamma3, gamma4


@jax.jit
def layer_reflectance_transmittance(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    mu0: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate layer reflectance and transmittance.
    
    Args:
        tau: Optical depth
        ssa: Single scattering albedo  
        g: Asymmetry factor
        mu0: Cosine of solar zenith angle (for SW)
        
    Returns:
        Tuple of (R_dif, T_dif, R_dir, T_dir)
        For LW, only diffuse components are used
    """
    # Get two-stream coefficients
    gamma1, gamma2, gamma3, gamma4 = two_stream_coefficients(tau, ssa, g, mu0)
    
    # Calculate lambda (eigenvalue)
    lambda_val = jnp.sqrt(gamma1**2 - gamma2**2)
    
    # For normal optical depths, calculate exponentials
    lambda_tau = lambda_val * tau
    
    # Handle large optical depths consistently - use lambda_tau threshold
    # Use 88 as threshold since exp(90) = inf, so be conservative
    large_tau = lambda_tau >= 88
    
    # exp_plus = jnp.where(large_tau, jnp.inf, jnp.exp(lambda_tau))
    # exp_minus = jnp.where(large_tau, 0.0, jnp.exp(-lambda_tau))
    
    # Helper terms
    term1 = 1.0 / (lambda_val + gamma1)
    term2 = 1.0 / (lambda_val - gamma1)
    
    # For large optical depths, avoid NaN by using safe values
    # Use finite values instead of inf for subsequent calculations
    exp_plus_safe = jnp.where(large_tau, 1.0, jnp.exp(lambda_tau))
    exp_minus_safe = jnp.where(large_tau, 0.0, jnp.exp(-lambda_tau))
    
    denom = exp_plus_safe - gamma2**2 / gamma1**2 * exp_minus_safe
    # Ensure denominator is never zero
    denom = jnp.where(jnp.abs(denom) < 1e-10, 1e-10, denom)
    
    R_dif_normal = gamma2 * (exp_plus_safe - exp_minus_safe) / denom
    T_dif_normal = (1.0 - R_dif_normal * gamma2 / gamma1) * exp_minus_safe
    
    # For large optical depths, use asymptotic behavior
    # Pure absorption case: R=0, T=0
    # Scattering case: R approaches gamma2/gamma1 (but clipped to physical bounds)
    R_dif_asymptotic = jnp.where(
        ssa > 0.001,  # If there's significant scattering
        jnp.clip(gamma2 / gamma1, 0.0, 1.0),
        0.0  # Pure absorption case
    )
    T_dif_asymptotic = 0.0  # No transmission for large tau
    
    # Choose based on optical depth
    R_dif = jnp.where(large_tau, R_dif_asymptotic, R_dif_normal)
    T_dif = jnp.where(large_tau, T_dif_asymptotic, T_dif_normal)
    
    # Ensure physical bounds
    R_dif = jnp.clip(R_dif, 0.0, 1.0)
    T_dif = jnp.clip(T_dif, 0.0, 1.0)
    
    if mu0 is not None:
        # Direct beam transmittance (Beer's law)
        T_dir = jnp.exp(-tau / mu0)
        T_dir = jnp.where(tau / mu0 > 100, 0.0, T_dir)
        
        # Direct to diffuse reflectance
        # From single scattering of direct beam
        R_dir = ssa * gamma3 * (1.0 - T_dir) / (1.0 - ssa * gamma4)
        R_dir = jnp.clip(R_dir, 0.0, 1.0)
    else:
        # Longwave - no direct beam
        T_dir = jnp.zeros_like(tau)
        R_dir = jnp.zeros_like(tau)
    
    return R_dif, T_dif, R_dir, T_dir


@jax.jit
def adding_method(
    R1: jnp.ndarray,
    T1: jnp.ndarray,
    R2: jnp.ndarray,
    T2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Combine two layers using adding method.
    
    Args:
        R1, T1: Reflectance and transmittance of upper layer
        R2, T2: Reflectance and transmittance of lower layer
        
    Returns:
        Combined reflectance and transmittance
    """
    # Denominator with numerical stability
    denom = 1.0 - R1 * R2
    denom = jnp.where(denom < 1e-10, 1e-10, denom)
    
    # Combined reflectance
    # R = R1 + T1² * R2 / (1 - R1 * R2)
    R_combined = R1 + T1**2 * R2 / denom
    
    # Combined transmittance  
    # T = T1 * T2 / (1 - R1 * R2)
    T_combined = T1 * T2 / denom
    
    # Ensure physical bounds
    R_combined = jnp.clip(R_combined, 0.0, 1.0)
    T_combined = jnp.clip(T_combined, 0.0, 1.0)
    
    return R_combined, T_combined


def longwave_fluxes_single_band(
    tau: jnp.ndarray,
    ssa: jnp.ndarray, 
    g: jnp.ndarray,
    planck_layer: jnp.ndarray,
    planck_interfaces: jnp.ndarray,
    surface_emissivity: float,
    surface_planck: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate LW fluxes for a single band."""
    nlev = tau.shape[0]
    
    # Calculate layer properties (no direct beam for LW)
    R_dif, T_dif, _, _ = layer_reflectance_transmittance(tau, ssa, g, mu0=None)
    
    # Simplified approach: use source function method
    # Source = Planck * (1 - transmittance)
    source = planck_layer * (1.0 - T_dif)
    
    # Initialize arrays
    flux_up = jnp.zeros(nlev + 1)
    flux_down = jnp.zeros(nlev + 1)
    
    # Surface emission
    flux_up = flux_up.at[nlev].set(surface_emissivity * surface_planck)
    
    # Upward flux calculation using recurrence relation
    def upward_step(carry, x):
        flux_below = carry
        lev, R, T, S = x
        flux_above = R * flux_below + T * flux_below + S
        return flux_above, flux_above
        
    # Scan from bottom to top
    indices = jnp.arange(nlev - 1, -1, -1)
    _, flux_up_levels = jax.lax.scan(
        upward_step,
        flux_up[nlev],
        (indices, R_dif[::-1], T_dif[::-1], source[::-1])
    )
    flux_up = flux_up.at[:-1].set(flux_up_levels[::-1])
    
    # Downward flux from top
    def downward_step(carry, x):
        flux_above = carry
        lev, R, T, S = x
        flux_below = T * flux_above + S
        return flux_below, flux_below
        
    # Scan from top to bottom  
    indices = jnp.arange(nlev)
    _, flux_down_levels = jax.lax.scan(
        downward_step,
        0.0,  # No downward LW at TOA
        (indices, R_dif, T_dif, source)
    )
    flux_down = flux_down.at[1:].set(flux_down_levels)
    
    return flux_up, flux_down


@jax.jit
def longwave_fluxes(
    optical_properties: OpticalProperties,
    planck_layers: jnp.ndarray,
    planck_interfaces: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    surface_planck: jnp.ndarray,
    n_bands: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate longwave fluxes using two-stream method.
    
    Args:
        optical_properties: Layer optical properties
        planck_layers: Planck function at layer centers [nlev, n_bands]
        planck_interfaces: Planck function at interfaces [nlev+1, n_bands]
        surface_emissivity: Surface emissivity
        surface_planck: Surface Planck emission [n_bands]
        n_bands: Number of spectral bands
        
    Returns:
        Tuple of (upward_flux, downward_flux) at interfaces [nlev+1, n_bands]
    """
    # Process all bands using vmap
    def process_band(band_idx):
        flux_up_band, flux_down_band = longwave_fluxes_single_band(
            optical_properties.optical_depth[:, band_idx],
            optical_properties.single_scatter_albedo[:, band_idx],
            optical_properties.asymmetry_factor[:, band_idx],
            planck_layers[:, band_idx],
            planck_interfaces[:, band_idx],
            surface_emissivity,
            surface_planck[band_idx]
        )
        return flux_up_band, flux_down_band
    
    # Apply to all bands - use fixed size for JAX compatibility
    max_bands = 10
    band_indices = jnp.arange(max_bands)
    band_mask = band_indices < n_bands
    flux_up_all, flux_down_all = jax.vmap(process_band)(band_indices)
    # Mask inactive bands (keep full size)
    flux_up = jnp.where(band_mask[:, None], flux_up_all, 0.0)
    flux_down = jnp.where(band_mask[:, None], flux_down_all, 0.0)
    
    # Transpose to get [nlev+1, n_bands] shape
    flux_up = flux_up.T
    flux_down = flux_down.T
    
    # Convert from radiance to flux (multiply by π)
    flux_up *= jnp.pi
    flux_down *= jnp.pi
    
    return flux_up, flux_down


def shortwave_fluxes_single_band(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    cos_zenith: float,
    toa_flux: float,
    surface_albedo: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate SW fluxes for a single band."""
    nlev = tau.shape[0]
    
    # Calculate layer properties with solar angle
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(tau, ssa, g, cos_zenith)
    
    # Direct beam transmission through atmosphere
    # Calculate cumulative direct transmission from TOA
    direct_trans = jnp.cumprod(T_dir, axis=0)
    
    # Add TOA transmission
    direct_trans_full = jnp.concatenate([jnp.array([1.0]), direct_trans])
    
    # Direct flux at each level
    flux_direct = toa_flux * direct_trans_full
    
    # Diffuse radiation calculation
    # Source from direct beam scattering
    source_diffuse = toa_flux * R_dir * direct_trans_full[:-1]
    
    # Initialize diffuse fluxes
    flux_down_dif = jnp.zeros(nlev + 1)
    flux_up_dif = jnp.zeros(nlev + 1)
    
    # Surface reflection of direct beam
    flux_up_dif = flux_up_dif.at[nlev].set(surface_albedo * flux_direct[nlev])
    
    # Upward diffuse calculation
    def upward_diffuse_step(carry, x):
        flux_below = carry
        R, T, S = x
        flux_above = T * flux_below + S
        return flux_above, flux_above
    
    _, flux_up_levels = jax.lax.scan(
        upward_diffuse_step,
        flux_up_dif[nlev],
        (R_dif[::-1], T_dif[::-1], source_diffuse[::-1])
    )
    flux_up_dif = flux_up_dif.at[:-1].set(flux_up_levels[::-1])
    
    # Downward diffuse calculation
    def downward_diffuse_step(carry, x):
        flux_above = carry
        R, T, S, flux_up = x
        flux_below = T * flux_above + R * flux_up + S # FIXME: verify this logic
        return flux_below, flux_below
    
    _, flux_down_levels = jax.lax.scan(
        downward_diffuse_step,
        0.0,  # No diffuse at TOA
        (R_dif, T_dif, source_diffuse, flux_up_dif[:-1])
    )
    flux_down_dif = flux_down_dif.at[1:].set(flux_down_levels)
    
    # Total fluxes
    flux_down_total = flux_direct + flux_down_dif
    flux_up_total = flux_up_dif  # No upward direct
    
    return flux_up_total, flux_down_total, flux_direct, flux_down_dif


@jax.jit
def shortwave_fluxes(
    optical_properties: OpticalProperties,
    cos_zenith: float,
    toa_flux: jnp.ndarray,
    surface_albedo: jnp.ndarray,
    n_bands: int = 2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate shortwave fluxes using two-stream method.
    
    Args:
        optical_properties: Layer optical properties
        cos_zenith: Cosine of solar zenith angle
        toa_flux: TOA incident flux [n_bands]
        surface_albedo: Surface albedo [n_bands]
        n_bands: Number of spectral bands
        
    Returns:
        Tuple of (up_flux, down_flux, down_direct, down_diffuse)
        All at interfaces [nlev+1, n_bands]
    """
    # Process all bands using vmap
    def process_band(band_idx):
        flux_up, flux_down, flux_dir, flux_dif = shortwave_fluxes_single_band(
            optical_properties.optical_depth[:, band_idx],
            optical_properties.single_scatter_albedo[:, band_idx],
            optical_properties.asymmetry_factor[:, band_idx],
            cos_zenith,
            toa_flux[band_idx],
            surface_albedo[band_idx]
        )
        return flux_up, flux_down, flux_dir, flux_dif
    
    # Apply to all bands - use fixed size for JAX compatibility
    max_bands = 10
    band_indices = jnp.arange(max_bands)
    band_mask = band_indices < n_bands
    flux_up_all, flux_down_all, flux_direct_all, flux_diffuse_all = jax.vmap(process_band)(band_indices)
    # Mask inactive bands (keep full size)
    flux_up = jnp.where(band_mask[:, None], flux_up_all, 0.0)
    flux_down = jnp.where(band_mask[:, None], flux_down_all, 0.0)
    flux_direct = jnp.where(band_mask[:, None], flux_direct_all, 0.0)
    flux_diffuse = jnp.where(band_mask[:, None], flux_diffuse_all, 0.0)
    
    # Transpose to get [nlev+1, n_bands] shape
    flux_up = flux_up.T
    flux_down = flux_down.T
    flux_direct = flux_direct.T
    flux_diffuse = flux_diffuse.T
    
    return flux_up, flux_down, flux_direct, flux_diffuse


@jax.jit
def flux_to_heating_rate(
    flux_up: jnp.ndarray,
    flux_down: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    g: float = 9.81,  # m/s^2
    cp: float = 1004.0  # J/kg/K
) -> jnp.ndarray:
    """
    Convert flux divergence to heating rate.
    
    dT/dt = -g/cp * dF/dp
    
    Args:
        flux_up: Upward flux at interfaces [nlev+1]
        flux_down: Downward flux at interfaces [nlev+1]
        pressure_interfaces: Pressure at interfaces [nlev+1]
        cp: Specific heat capacity
        
    Returns:
        Heating rate (K/s) [nlev]
    """
    # Net flux at interfaces
    net_flux_down = flux_down - flux_up
    
    # Flux divergence in layers
    flux_div = jnp.diff(net_flux_down, axis=0)
    
    # Pressure thickness
    dp = jnp.diff(pressure_interfaces, axis=0)
    
    # Heating rate
    heating = (g / cp) * (-flux_div) / dp
    
    return heating


