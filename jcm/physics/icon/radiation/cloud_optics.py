"""
Cloud optical properties for radiation

This module calculates optical properties of clouds including
optical depth, single scattering albedo, and asymmetry parameter
for both liquid and ice clouds.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple
from functools import partial

from .radiation_types import OpticalProperties


@jax.jit
def effective_radius_liquid(
    temperature: jnp.ndarray,
    land_fraction: float = 0.5
) -> jnp.ndarray:
    """
    Calculate effective radius for liquid cloud droplets.
    
    Simple parameterization based on temperature and surface type.
    
    Args:
        temperature: Temperature (K)
        land_fraction: Fraction of land (0=ocean, 1=land)
        
    Returns:
        Effective radius (microns)
    """
    # Different values over land and ocean
    r_eff_ocean = 14.0  # microns
    r_eff_land = 8.0    # microns
    
    # Weighted average
    r_eff = r_eff_ocean * (1 - land_fraction) + r_eff_land * land_fraction
    
    # Temperature dependence (smaller droplets in colder clouds)
    t_factor = jnp.clip((temperature - 253.0) / 20.0, 0.5, 1.5)
    
    return r_eff * t_factor


@jax.jit
def effective_radius_ice(
    temperature: jnp.ndarray,
    ice_water_content: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate effective radius for ice crystals.
    
    Based on temperature and ice water content.
    
    Args:
        temperature: Temperature (K)
        ice_water_content: Ice water content (kg/m³)
        
    Returns:
        Effective radius (microns)
    """
    # Base radius depends on temperature
    # Colder = smaller crystals
    t_celsius = temperature - 273.15
    r_base = 20.0 + 1.5 * jnp.clip(t_celsius + 40.0, 0.0, 40.0)
    
    # Adjust for ice content (higher content = larger crystals)
    iwc_factor = jnp.clip(ice_water_content * 1e4, 0.5, 2.0)
    
    return r_base * iwc_factor


@jax.jit
def liquid_cloud_optics_sw(
    cloud_water_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate shortwave optical properties for liquid clouds.
    
    Based on parameterizations from Slingo (1989).
    
    Args:
        cloud_water_path: Cloud water path (kg/m²)
        effective_radius: Droplet effective radius (microns)
        band: Spectral band (0=vis, 1=nir)
        
    Returns:
        Tuple of (optical_depth, single_scatter_albedo, asymmetry_factor)
    """
    # Convert to g/m²
    cwp = cloud_water_path * 1000.0
    
    # Optical depth parameterization
    a_tau = jnp.where(band == 0, 2.21, 2.17)
    b_tau = jnp.where(band == 0, -0.023, -0.020)
    
    tau = cwp * (a_tau + b_tau * effective_radius)
    
    # Single scattering albedo
    ssa = jnp.where(
        band == 0,
        1.0 - 5e-7 * effective_radius**2,  # Visible
        1.0 - 0.06 - 2e-5 * effective_radius**2  # Near-IR
    )
    
    ssa = jnp.clip(ssa, 0.5, 0.99999)
    
    # Asymmetry factor
    g = jnp.where(
        band == 0,
        0.85 + 0.0015 * effective_radius,  # Visible
        0.80 + 0.002 * effective_radius   # Near-IR
    )
    
    g = jnp.clip(g, 0.7, 0.95)
    
    return tau, ssa, g


@jax.jit
def ice_cloud_optics_sw(
    cloud_ice_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate shortwave optical properties for ice clouds.
    
    Simplified parameterization.
    
    Args:
        cloud_ice_path: Cloud ice path (kg/m²)
        effective_radius: Ice crystal effective radius (microns)
        band: Spectral band (0=vis, 1=nir)
        
    Returns:
        Tuple of (optical_depth, single_scatter_albedo, asymmetry_factor)
    """
    # Convert to g/m²
    cip = cloud_ice_path * 1000.0
    
    # Optical depth - ice less efficient than liquid
    tau = cip * jnp.where(
        band == 0,
        1.5 / effective_radius,  # Visible
        1.3 / effective_radius   # Near-IR
    )
    
    # Single scattering albedo - ice more absorbing
    ssa = jnp.where(
        band == 0,
        0.999 - 1e-5 * effective_radius,  # Visible
        0.95 - 0.001 * effective_radius   # Near-IR
    )
    
    ssa = jnp.clip(ssa, 0.5, 0.99999)
    
    # Asymmetry factor - ice crystals more forward scattering
    g = 0.75 + 0.09 * jnp.log(effective_radius / 50.0)
    g = jnp.clip(g, 0.7, 0.95)
    
    return tau, ssa, g


@jax.jit
def liquid_cloud_optics_lw(
    cloud_water_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate longwave optical properties for liquid clouds.
    
    Longwave assumes pure absorption (no scattering).
    
    Args:
        cloud_water_path: Cloud water path (kg/m²)
        effective_radius: Droplet effective radius (microns)
        band: Spectral band
        
    Returns:
        Optical depth (absorption)
    """
    # Absorption coefficient depends on band
    k_abs = jnp.where(
        band == 0,
        50.0,   # Window region (m²/kg)
        130.0   # Water vapor bands (m²/kg)
    )
    
    # Weak dependence on droplet size
    size_factor = jnp.sqrt(10.0 / effective_radius)
    
    tau = k_abs * cloud_water_path * size_factor
    
    return tau


@jax.jit
def ice_cloud_optics_lw(
    cloud_ice_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate longwave optical properties for ice clouds.
    
    Args:
        cloud_ice_path: Cloud ice path (kg/m²)
        effective_radius: Ice crystal effective radius (microns)
        band: Spectral band
        
    Returns:
        Optical depth (absorption)
    """
    # Ice absorption coefficient
    k_abs = jnp.where(
        band == 0,
        20.0,   # Window region (m²/kg)
        65.0    # Water vapor bands (m²/kg)
    )
    
    # Size dependence
    size_factor = jnp.sqrt(30.0 / effective_radius)
    
    tau = k_abs * cloud_ice_path * size_factor
    
    return tau


@partial(jax.jit, static_argnames=['n_sw_bands', 'n_lw_bands'])
def cloud_optics(
    cloud_water_path: jnp.ndarray,
    cloud_ice_path: jnp.ndarray,
    temperature: jnp.ndarray,
    n_sw_bands: int = 2,
    n_lw_bands: int = 3,
    land_fraction: float = 0.5
) -> Tuple[OpticalProperties, OpticalProperties]:
    """
    Calculate complete cloud optical properties.
    
    Args:
        cloud_water_path: Cloud water path (kg/m²) [nlev]
        cloud_ice_path: Cloud ice path (kg/m²) [nlev]
        temperature: Temperature (K) [nlev]
        n_sw_bands: Number of SW bands
        n_lw_bands: Number of LW bands
        land_fraction: Land fraction for droplet size
        
    Returns:
        Tuple of (sw_optics, lw_optics)
    """
    nlev = temperature.shape[0]
    
    # Calculate effective radii
    r_eff_liq = effective_radius_liquid(temperature, land_fraction)
    r_eff_ice = effective_radius_ice(
        temperature,
        cloud_ice_path / jnp.maximum(1.0, cloud_water_path + cloud_ice_path)
    )
    
    # Calculate SW properties for all bands
    def calculate_sw_band(band):
        # Liquid clouds
        tau_liq, ssa_liq, g_liq = liquid_cloud_optics_sw(
            cloud_water_path, r_eff_liq, band
        )
        
        # Ice clouds
        tau_ice, ssa_ice, g_ice = ice_cloud_optics_sw(
            cloud_ice_path, r_eff_ice, band
        )
        
        # Combine (additive optical depth)
        tau_total = tau_liq + tau_ice
        
        # Combined single scattering albedo (weighted by tau)
        ssa_combined = jnp.where(
            tau_total > 0,
            (tau_liq * ssa_liq + tau_ice * ssa_ice) / tau_total,
            1.0
        )
        
        # Combined asymmetry factor (weighted by tau*ssa)
        g_combined = jnp.where(
            tau_total * ssa_combined > 0,
            (tau_liq * ssa_liq * g_liq + tau_ice * ssa_ice * g_ice) / 
            (tau_total * ssa_combined),
            0.0
        )
        
        return tau_total, ssa_combined, g_combined
    
    # Apply to all SW bands
    sw_bands = jnp.arange(n_sw_bands)
    tau_sw, ssa_sw, g_sw = jax.vmap(calculate_sw_band)(sw_bands)
    tau_sw = tau_sw.T
    ssa_sw = ssa_sw.T
    g_sw = g_sw.T
    
    # Calculate LW properties for all bands
    def calculate_lw_band(band):
        tau_liq = liquid_cloud_optics_lw(cloud_water_path, r_eff_liq, band)
        tau_ice = ice_cloud_optics_lw(cloud_ice_path, r_eff_ice, band)
        return tau_liq + tau_ice
    
    # Apply to all LW bands
    lw_bands = jnp.arange(n_lw_bands)
    tau_lw = jax.vmap(calculate_lw_band)(lw_bands).T
    
    # Create optical properties
    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=ssa_sw,
        asymmetry_factor=g_sw
    )
    
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.zeros((nlev, n_lw_bands)),  # Pure absorption
        asymmetry_factor=jnp.zeros((nlev, n_lw_bands))
    )
    
    return sw_optics, lw_optics


@jax.jit
def cloud_overlap_factor(
    cloud_fraction_above: jnp.ndarray,
    cloud_fraction_current: jnp.ndarray,
    overlap_param: float = 0.5
) -> jnp.ndarray:
    """
    Calculate cloud overlap factor.
    
    Maximum-random overlap approximation.
    
    Args:
        cloud_fraction_above: Cloud fraction in layer above
        cloud_fraction_current: Cloud fraction in current layer
        overlap_param: Overlap parameter (0=random, 1=maximum)
        
    Returns:
        Overlap factor
    """
    # Maximum overlap
    max_overlap = jnp.minimum(cloud_fraction_above, cloud_fraction_current)
    
    # Random overlap
    random_overlap = cloud_fraction_above * cloud_fraction_current
    
    # Combined
    overlap = overlap_param * max_overlap + (1 - overlap_param) * random_overlap
    
    return overlap


# Test functions
def test_cloud_optics():
    """Test cloud optics calculations"""
    
    # Test data
    nlev = 10
    temperature = jnp.linspace(250, 290, nlev)
    cwp = jnp.where(temperature > 273, 0.1, 0.0)  # Liquid above freezing
    cip = jnp.where(temperature <= 273, 0.05, 0.0)  # Ice below freezing
    
    # Calculate optics
    sw_optics, lw_optics = cloud_optics(cwp, cip, temperature, 2, 3)
    
    # Check shapes
    assert sw_optics.optical_depth.shape == (nlev, 2)
    assert lw_optics.optical_depth.shape == (nlev, 3)
    
    # Check values
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo <= 1)
    assert jnp.all(sw_optics.asymmetry_factor >= 0)
    assert jnp.all(sw_optics.asymmetry_factor <= 1)
    
    print("Cloud optics tests passed!")


if __name__ == "__main__":
    test_cloud_optics()