"""
Gas optics for radiation calculations

This module computes absorption coefficients and optical depths
for atmospheric gases in both shortwave and longwave spectral regions.

Simplified implementation with parameterized absorption.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple
# from functools import partial  # Not needed anymore

from .radiation_types import OpticalProperties


@jax.jit
def water_vapor_continuum(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate water vapor continuum absorption.
    
    Simplified parameterization of H2O continuum absorption.
    
    Args:
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        h2o_vmr: Water vapor volume mixing ratio [nlev]
        band: Spectral band index
        
    Returns:
        Absorption coefficient (m²/kg)
    """
    # Reference temperature and pressure
    T_ref = 296.0  # K
    P_ref = 101325.0  # Pa
    
    # Convert VMR to mass mixing ratio
    # q = vmr * (M_h2o / M_air) ≈ vmr * 0.622
    h2o_mmr = h2o_vmr * 0.622
    
    # Temperature dependence
    T_factor = jnp.exp(1800.0 * (1.0/temperature - 1.0/T_ref))
    
    # Pressure scaling
    P_factor = pressure / P_ref
    
    # Band-dependent coefficients (simplified)
    k_ref = jnp.where(
        band == 0,
        0.01,  # Window region
        jnp.where(
            band == 1,
            0.1,   # Weak absorption
            1.0    # Strong absorption
        )
    )
    
    # Absorption coefficient
    k_abs = k_ref * T_factor * P_factor * h2o_mmr
    
    return k_abs


@jax.jit
def co2_absorption(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    co2_vmr: float,
    band: int
) -> jnp.ndarray:
    """
    Calculate CO2 absorption.
    
    Simplified parameterization for CO2 15-micron band.
    
    Args:
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        co2_vmr: CO2 volume mixing ratio (constant)
        band: Spectral band index
        
    Returns:
        Absorption coefficient (m²/kg)
    """
    # Only significant in band 1 (CO2 band around 667 cm⁻¹)
    return jnp.where(
        band != 1,
        jnp.zeros_like(temperature),
        # Calculate absorption for band 1
        _calculate_co2_band1(temperature, pressure, co2_vmr)
    )


def _calculate_co2_band1(temperature, pressure, co2_vmr):
    """Helper function to calculate CO2 absorption in band 1"""
    # Reference conditions
    T_ref = 296.0
    P_ref = 101325.0
    
    # Temperature dependence (simplified)
    T_factor = (T_ref / temperature) ** 0.5
    
    # Pressure broadening
    P_factor = (pressure / P_ref) ** 0.75
    
    # Reference absorption
    k_ref = 0.1  # Simplified
    
    # CO2 mass mixing ratio
    co2_mmr = co2_vmr * (44.0 / 29.0)  # M_CO2 / M_air
    
    return k_ref * T_factor * P_factor * co2_mmr


@jax.jit
def ozone_absorption_sw(
    o3_vmr: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate ozone absorption in shortwave.
    
    Simplified parameterization for UV/visible absorption.
    
    Args:
        o3_vmr: Ozone volume mixing ratio [nlev]
        band: Spectral band index (0=vis, 1=nir)
        
    Returns:
        Absorption coefficient (m²/kg)
    """
    # Ozone absorbs mainly in UV/visible (band 0)
    k_o3 = 100.0  # m²/kg (very simplified)
    o3_mmr = o3_vmr * (48.0 / 29.0)  # M_O3 / M_air
    
    return jnp.where(
        band == 0,
        k_o3 * o3_mmr,
        jnp.zeros_like(o3_vmr)
    )


@jax.jit
def ozone_absorption_lw(
    temperature: jnp.ndarray,
    o3_vmr: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate ozone absorption in longwave.
    
    Simplified parameterization for 9.6 micron band.
    
    Args:
        temperature: Temperature (K) [nlev]
        o3_vmr: Ozone volume mixing ratio [nlev]
        band: Spectral band index
        
    Returns:
        Absorption coefficient (m²/kg)
    """
    # Ozone 9.6 micron band (around 1042 cm⁻¹) - mainly in band 2
    T_ref = 296.0
    T_factor = jnp.sqrt(T_ref / temperature)
    k_o3 = 50.0  # Simplified
    o3_mmr = o3_vmr * (48.0 / 29.0)
    
    return jnp.where(
        band == 2,
        k_o3 * T_factor * o3_mmr,
        jnp.zeros_like(temperature)
    )


@jax.jit
def gas_optical_depth_lw(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    o3_vmr: jnp.ndarray,
    co2_vmr: float,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    n_bands: int = 3
) -> jnp.ndarray:
    """
    Calculate longwave gas optical depths.
    
    Args:
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        h2o_vmr: Water vapor VMR [nlev]
        o3_vmr: Ozone VMR [nlev]
        co2_vmr: CO2 VMR (constant)
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        n_bands: Number of LW bands
        
    Returns:
        Optical depth [nlev, n_bands]
    """
    nlev = temperature.shape[0]
    
    # Calculate absorption for all bands using vmap
    def single_band_absorption(band):
        # Water vapor absorption
        k_h2o = water_vapor_continuum(temperature, pressure, h2o_vmr, band)
        
        # CO2 absorption
        k_co2 = co2_absorption(temperature, pressure, co2_vmr, band)
        
        # Ozone absorption
        k_o3 = ozone_absorption_lw(temperature, o3_vmr, band)
        
        # Total absorption coefficient
        k_total = k_h2o + k_co2 + k_o3
        
        # Optical depth = absorption * density * path length
        return k_total * air_density * layer_thickness
    
    # Apply to all bands - use fixed range for JAX compatibility
    max_bands = 10
    bands = jnp.arange(max_bands)
    band_mask = bands < n_bands
    tau_all = jax.vmap(single_band_absorption)(bands)
    # Return full size array with inactive bands masked to zero
    tau = jnp.where(band_mask[:, None], tau_all, 0.0).T
    
    return tau


@jax.jit
def gas_optical_depth_sw(
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    o3_vmr: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    n_bands: int = 2
) -> jnp.ndarray:
    """
    Calculate shortwave gas optical depths.
    
    Args:
        pressure: Pressure (Pa) [nlev]
        h2o_vmr: Water vapor VMR [nlev]
        o3_vmr: Ozone VMR [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cos_zenith: Cosine of solar zenith angle
        n_bands: Number of SW bands
        
    Returns:
        Optical depth [nlev, n_bands]
    """
    nlev = pressure.shape[0]
    
    # Path length correction for solar angle
    sec_zenith = 1.0 / jnp.maximum(cos_zenith, 0.01)
    
    # Calculate absorption for all bands
    def single_band_absorption(band):
        # Water vapor absorption (simplified - mainly NIR)
        h2o_mmr = h2o_vmr * 0.622
        k_h2o = jnp.where(
            band == 1,  # NIR band
            0.01 * h2o_mmr,  # Very simplified
            0.0
        )
        
        # Ozone absorption
        k_o3 = ozone_absorption_sw(o3_vmr, band)
        
        # Total absorption
        k_total = k_h2o + k_o3
        
        # Optical depth with slant path correction
        return k_total * air_density * layer_thickness * sec_zenith
    
    # Apply to all bands - use fixed range for JAX compatibility
    max_bands = 10
    bands = jnp.arange(max_bands)
    band_mask = bands < n_bands
    tau_all = jax.vmap(single_band_absorption)(bands)
    # Return full size array with inactive bands masked to zero
    tau = jnp.where(band_mask[:, None], tau_all, 0.0).T
    
    return tau


@jax.jit
def rayleigh_optical_depth(
    pressure: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    wavelength: float = 0.55  # microns
) -> jnp.ndarray:
    """
    Calculate Rayleigh scattering optical depth.
    
    Args:
        pressure: Pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        wavelength: Wavelength in microns
        
    Returns:
        Rayleigh optical depth [nlev]
    """
    # Rayleigh scattering coefficient
    # τ_Ray = 0.008569 * λ^(-4) * (1 + 0.0113 * λ^(-2) + 0.00013 * λ^(-4))
    
    lambda_inv4 = wavelength ** (-4)
    tau_ray_sea_level = 0.008569 * lambda_inv4 * (
        1.0 + 0.0113 * wavelength**(-2) + 0.00013 * lambda_inv4
    )
    
    # Scale by pressure
    P_sea_level = 101325.0  # Pa
    tau = tau_ray_sea_level * (pressure / P_sea_level) * (layer_thickness / 8000.0)
    
    return tau


def create_gas_optics(
    state,
    layer_properties,
    config,
    cos_zenith
) -> Tuple[OpticalProperties, OpticalProperties]:
    """
    Create gas optical properties for SW and LW.
    
    Args:
        state: Atmospheric state
        layer_properties: Layer thickness and density
        config: Radiation configuration
        cos_zenith: Cosine solar zenith angle
        
    Returns:
        Tuple of (sw_optics, lw_optics)
    """
    # Longwave optical depths
    tau_lw = gas_optical_depth_lw(
        state.temperature,
        state.pressure,
        state.h2o_vmr,
        state.o3_vmr,
        config.co2_vmr,
        layer_properties['thickness'],
        layer_properties['density'],
        config.n_lw_bands
    )
    
    # Shortwave optical depths
    tau_sw = gas_optical_depth_sw(
        state.pressure,
        state.h2o_vmr,
        state.o3_vmr,
        layer_properties['thickness'],
        layer_properties['density'],
        cos_zenith,
        config.n_sw_bands
    )
    
    # Add Rayleigh scattering to visible band
    tau_rayleigh = rayleigh_optical_depth(
        state.pressure,
        layer_properties['thickness'],
        0.55  # Visible wavelength
    )
    tau_sw = tau_sw.at[:, 0].add(tau_rayleigh)
    
    # Gas absorption has no scattering (ssa=0) except Rayleigh
    nlev = state.temperature.shape[0]
    
    # Longwave: pure absorption
    # Create fixed-size arrays for JAX compatibility
    max_bands = 10
    lw_band_mask = jnp.arange(max_bands) < config.n_lw_bands
    lw_ssa_all = jnp.zeros((nlev, max_bands))
    lw_g_all = jnp.zeros((nlev, max_bands))
    
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.where(lw_band_mask[None, :], lw_ssa_all, 0.0),
        asymmetry_factor=jnp.where(lw_band_mask[None, :], lw_g_all, 0.0)
    )
    
    # Shortwave: Rayleigh scattering in visible
    sw_band_mask = jnp.arange(max_bands) < config.n_sw_bands
    sw_ssa_all = jnp.zeros((nlev, max_bands))
    sw_ssa_all = sw_ssa_all.at[:, 0].set(
        tau_rayleigh / jnp.maximum(tau_sw[:, 0], 1e-10)
    )
    sw_g_all = jnp.zeros((nlev, max_bands))
    
    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=jnp.where(sw_band_mask[None, :], sw_ssa_all, 0.0),
        asymmetry_factor=jnp.where(sw_band_mask[None, :], sw_g_all, 0.0)  # Rayleigh: g=0
    )
    
    return sw_optics, lw_optics


# Test function
def test_gas_optics():
    """Test gas optics calculations"""
    nlev = 20
    
    # Create test atmosphere
    pressure = jnp.linspace(100000, 10000, nlev)
    temperature = jnp.linspace(288, 220, nlev)
    h2o_vmr = jnp.linspace(0.01, 1e-6, nlev)
    o3_vmr = jnp.ones(nlev) * 5e-6
    thickness = jnp.ones(nlev) * 500.0
    density = pressure / (287.0 * temperature)
    
    # Test LW optical depth
    tau_lw = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr,
        400e-6, thickness, density, 3
    )
    
    assert tau_lw.shape == (nlev, 3)
    assert jnp.all(tau_lw >= 0)
    assert jnp.all(jnp.isfinite(tau_lw))
    
    print("Gas optics tests passed!")


if __name__ == "__main__":
    test_gas_optics()