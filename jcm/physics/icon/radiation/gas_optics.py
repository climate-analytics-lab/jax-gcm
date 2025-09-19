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
    
    # Enhanced band-dependent coefficients based on MT_CKD continuum model
    # Updated for 8 longwave bands with improved spectral resolution
    
    # Self-broadening coefficients (H2O-H2O interactions)
    k_self = jnp.where(
        band == 0, 0.005,  # Far-IR window (10-200 cm⁻¹)
        jnp.where(
            band == 1, 0.15,   # H2O rotation band (200-280 cm⁻¹)
            jnp.where(
                band == 2, 0.08,   # CO2 bending + H2O (280-400 cm⁻¹)
                jnp.where(
                    band == 3, 0.12,   # CO2 v2 + H2O (400-540 cm⁻¹)
                    jnp.where(
                        band == 4, 0.25,   # H2O continuum (540-800 cm⁻¹)
                        jnp.where(
                            band == 5, 0.18,   # H2O + O3 (800-1000 cm⁻¹)
                            jnp.where(
                                band == 6, 0.22,   # O3 + H2O (1000-1200 cm⁻¹)
                                0.35                # H2O bands (1200-2600 cm⁻¹)
                            )
                        )
                    )
                )
            )
        )
    )
    
    # Foreign-broadening coefficients (H2O-N2, H2O-O2 interactions)
    k_foreign = jnp.where(
        band == 0, 0.001,  # Far-IR window
        jnp.where(
            band == 1, 0.035,  # H2O rotation band
            jnp.where(
                band == 2, 0.018,  # CO2 bending + H2O
                jnp.where(
                    band == 3, 0.028,  # CO2 v2 + H2O
                    jnp.where(
                        band == 4, 0.055,  # H2O continuum
                        jnp.where(
                            band == 5, 0.042,  # H2O + O3
                            jnp.where(
                                band == 6, 0.048,  # O3 + H2O
                                0.08                # H2O bands
                            )
                        )
                    )
                )
            )
        )
    )
    
    # Total continuum absorption (self + foreign contributions)
    # Self-broadening scales with H2O partial pressure
    # Foreign-broadening scales with total pressure
    h2o_partial_pressure = pressure * h2o_vmr
    dry_air_pressure = pressure * (1.0 - h2o_vmr)
    
    k_ref = (k_self * h2o_partial_pressure/P_ref + 
             k_foreign * dry_air_pressure/P_ref)
    
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
    # CO2 absorption in multiple bands
    # Main CO2 band (667 cm⁻¹) is in band 2 (280-400 cm⁻¹)
    # Some absorption also in band 3 (400-540 cm⁻¹)
    return jnp.where(
        band == 2,
        _calculate_co2_band1(temperature, pressure, co2_vmr),  # Main CO2 band
        jnp.where(
            band == 3,
            _calculate_co2_band1(temperature, pressure, co2_vmr) * 0.3,  # Secondary CO2 band
            jnp.zeros_like(temperature)
        )
    )


def _calculate_co2_band1(temperature, pressure, co2_vmr):
    """
    Enhanced CO2 absorption calculation with improved temperature/pressure dependence.
    
    Based on HITRAN line data parameterization for the 15 μm CO2 band.
    """
    # Reference conditions
    T_ref = 296.0
    P_ref = 101325.0
    
    # Enhanced temperature dependence for CO2 line strength
    # Based on HITRAN formula: S(T) = S_ref * (T_ref/T) * exp(-E_low/k*(1/T - 1/T_ref))
    # where E_low is the lower state energy
    E_low_k = 960.0  # Lower state energy / Boltzmann constant (K) for 15 μm band
    
    T_factor = (T_ref / temperature) * jnp.exp(-E_low_k * (1.0/temperature - 1.0/T_ref))
    
    # Improved pressure broadening with temperature dependence
    # γ(T,P) = γ_ref * (T_ref/T)^n * P/P_ref
    n_temp = 0.69  # Temperature exponent for CO2 line widths
    P_factor = (pressure / P_ref) * (T_ref / temperature)**n_temp
    
    # Enhanced absorption coefficient based on spectroscopic data
    # Includes both line absorption and continuum effects
    k_ref = 0.15  # Increased from 0.1 to better match observations
    
    # CO2 mass mixing ratio
    co2_mmr = co2_vmr * (44.0 / 29.0)  # M_CO2 / M_air
    
    # Add saturation effects for high CO2 concentrations
    # Prevents unrealistic absorption at very high CO2 levels
    saturation_factor = 1.0 / (1.0 + 0.1 * co2_mmr * P_factor)
    
    return k_ref * T_factor * P_factor * co2_mmr * saturation_factor


@jax.jit
def ozone_absorption_sw(
    o3_vmr: jnp.ndarray,
    temperature: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Enhanced ozone absorption in shortwave with temperature-dependent UV cross-sections.
    
    Based on Hartley-Huggins bands and Chappuis band parameterizations.
    
    Args:
        o3_vmr: Ozone volume mixing ratio [nlev]
        temperature: Temperature [K] [nlev]
        band: Spectral band index (0=vis/UV, 1=nir)
        
    Returns:
        Absorption coefficient (m²/kg)
    """
    # Reference temperature
    T_ref = 273.15
    
    # Enhanced band-dependent absorption cross-sections using JAX-compatible conditionals
    # Based on UV-visible spectroscopy data
    
    # Constants
    N_A = 6.022e23  # molecules/mol
    M_O3 = 48.0e-3  # kg/mol
    
    # UV/visible band (200-700 nm) - Hartley-Huggins-Chappuis bands
    sigma_ref_uv_vis = 1.2e-21  # cm²/molecule at 273K for UV/Vis peak
    a_uv_vis = -3.5e-4  # Linear temperature coefficient (K⁻¹)
    b_uv_vis = 1.0e-6   # Quadratic temperature coefficient (K⁻²)
    
    dT = temperature - T_ref
    temp_factor_uv_vis = 1.0 + a_uv_vis * dT + b_uv_vis * dT**2
    k_o3_uv_vis = sigma_ref_uv_vis * N_A / M_O3 * temp_factor_uv_vis * 1e-4
    
    # Near-infrared band (700-4000 nm)
    sigma_ref_nir = 4.5e-23  # cm²/molecule (much weaker than UV/Vis)
    temp_factor_nir = 1.0 + 1.5e-4 * (temperature - T_ref)
    k_o3_nir = sigma_ref_nir * N_A / M_O3 * temp_factor_nir * 1e-4

    k_o3_by_band = jnp.array([
        k_o3_uv_vis * 1.5,  # UV-C/B - highest O3 absorption
        k_o3_uv_vis,        # UV-A - strong O3 absorption
        k_o3_uv_vis * 0.8,  # Blue - moderate O3 absorption
        k_o3_uv_vis * 0.3,  # Green-Red - weak O3 absorption
        k_o3_nir,           # Near-IR 1 - very weak
        k_o3_nir * 0.5      # Near-IR 2 - minimal
    ])

    k_o3 = k_o3_by_band[band]

    # Convert VMR to mass mixing ratio
    o3_mmr = o3_vmr * (48.0 / 29.0)  # M_O3 / M_air
    
    return k_o3 * o3_mmr


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
    # Ozone 9.6 micron band (around 1042 cm⁻¹) - mainly in band 6 (1000-1200 cm⁻¹)
    # Some contribution also in band 5 (800-1000 cm⁻¹)
    T_ref = 296.0
    T_factor = jnp.sqrt(T_ref / temperature)
    k_o3_main = 50.0  # Main 9.6 μm band
    k_o3_secondary = 15.0  # Secondary bands
    o3_mmr = o3_vmr * (48.0 / 29.0)
    
    return jnp.where(
        band == 6,
        k_o3_main * T_factor * o3_mmr,      # Main 9.6 μm band
        jnp.where(
            band == 5,
            k_o3_secondary * T_factor * o3_mmr,  # Secondary band
            jnp.zeros_like(temperature)
        )
    )


@jax.jit
def gas_optical_depth_lw(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    o3_vmr: jnp.ndarray,
    co2_vmr: float,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray
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
    
    # Apply to all LW bands - use fixed shape
    from .constants import N_LW_BANDS
    tau = jnp.zeros((nlev, N_LW_BANDS))
    for band in range(N_LW_BANDS):
        tau = tau.at[:, band].set(single_band_absorption(band))
    
    return tau


@jax.jit
def gas_optical_depth_sw(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    o3_vmr: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cos_zenith: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate shortwave gas optical depths with enhanced temperature dependence.
    
    Args:
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        h2o_vmr: Water vapor VMR [nlev]
        o3_vmr: Ozone VMR [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cos_zenith: Cosine of solar zenith angle
        
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
        
        # Ozone absorption with temperature dependence
        k_o3 = ozone_absorption_sw(o3_vmr, temperature, band)
        
        # Total absorption
        k_total = k_h2o + k_o3
        
        # Optical depth with slant path correction
        return k_total * air_density * layer_thickness * sec_zenith
    
    # Apply to all SW bands - use fixed shape
    from .constants import N_SW_BANDS
    tau = jnp.zeros((nlev, N_SW_BANDS))
    for band in range(N_SW_BANDS):
        tau = tau.at[:, band].set(single_band_absorption(band))
    
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
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    o3_vmr: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cos_zenith,
    config,
) -> Tuple[OpticalProperties, OpticalProperties]:
    """
    Create gas optical properties for SW and LW.
    
    Args:
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        h2o_vmr: Water vapor volume mixing ratio [nlev]
        o3_vmr: Ozone volume mixing ratio [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        air_density: Air density (kg/m³) [nlev]
        cos_zenith: Cosine solar zenith angle
        config: Radiation configuration
        
    Returns:
        Tuple of (sw_optics, lw_optics)
    """
    # Longwave optical depths
    tau_lw = gas_optical_depth_lw(
        temperature,
        pressure,
        h2o_vmr,
        o3_vmr,
        config.co2_vmr,
        layer_thickness,
        air_density,
    )
    
    # Shortwave optical depths
    tau_sw = gas_optical_depth_sw(
        pressure,
        temperature,
        h2o_vmr,
        o3_vmr,
        layer_thickness,
        air_density,
        cos_zenith
    )
    
    # Add Rayleigh scattering to visible band
    tau_rayleigh = rayleigh_optical_depth(
        pressure,
        layer_thickness,
        0.55  # Visible wavelength
    )
    tau_sw = tau_sw.at[:, 0].add(tau_rayleigh)
    
    # Gas absorption has no scattering (ssa=0) except Rayleigh
    nlev = temperature.shape[0]
    
    # Longwave: pure absorption
    # Create fixed-size arrays for JAX compatibility
    max_bands = 10
    lw_band_mask = jnp.arange(max_bands) < config.n_lw_bands
    lw_ssa_all = jnp.zeros((nlev, max_bands))
    lw_g_all = jnp.zeros((nlev, max_bands))
    
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.where(lw_band_mask[jnp.newaxis], lw_ssa_all, 0.0),
        asymmetry_factor=jnp.where(lw_band_mask[jnp.newaxis], lw_g_all, 0.0)
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
        single_scatter_albedo=jnp.where(sw_band_mask[jnp.newaxis], sw_ssa_all, 0.0),
        asymmetry_factor=jnp.where(sw_band_mask[jnp.newaxis], sw_g_all, 0.0)  # Rayleigh: g=0
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
        400e-6, thickness, density
    )
    
    from .constants import N_LW_BANDS
    assert tau_lw.shape == (nlev, N_LW_BANDS)
    assert jnp.all(tau_lw >= 0)
    assert jnp.all(jnp.isfinite(tau_lw))
    
    print("Gas optics tests passed!")


if __name__ == "__main__":
    test_gas_optics()