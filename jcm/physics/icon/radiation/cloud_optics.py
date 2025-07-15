"""
Enhanced cloud optical properties with Mie scattering

This module calculates optical properties of clouds using proper Mie scattering
theory for liquid droplets and improved parameterizations for ice crystals.
Includes wavelength-dependent optical properties across multiple spectral bands.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from typing import Tuple
# from functools import partial  # Not needed anymore

from .radiation_types import OpticalProperties
from .constants import SW_BAND_LIMITS, LW_BAND_LIMITS


# Physical constants for Mie scattering
WATER_REFRACTIVE_INDEX = {
    # Real part of refractive index for water at different wavelengths (μm)
    # Based on Hale & Querry (1973) data
    0.25: 1.362,  # UV
    0.31: 1.349,  # UV-A
    0.38: 1.340,  # Blue
    0.55: 1.333,  # Green
    0.94: 1.327,  # Near-IR 1
    2.5: 1.312,   # Near-IR 2
}

WATER_ABSORPTION_COEFF = {
    # Imaginary part of refractive index (absorption) for water
    0.25: 1.1e-9,   # UV - very low absorption
    0.31: 1.8e-9,   # UV-A
    0.38: 2.5e-9,   # Blue
    0.55: 5.7e-9,   # Green
    0.94: 2.89e-6,  # Near-IR 1 - higher absorption
    2.5: 1.38e-4,   # Near-IR 2 - strong absorption
}


def get_band_wavelength(band: int, is_sw: bool = True) -> float:
    """
    Get representative wavelength for a spectral band.
    
    Args:
        band: Band index
        is_sw: True for shortwave, False for longwave
        
    Returns:
        Representative wavelength in micrometers
    """
    # Define all SW wavelengths
    sw_wavelengths = jnp.array([
        0.245,  # Band 0: UV-C/B (0.20-0.29 μm)
        0.305,  # Band 1: UV-A (0.29-0.32 μm)
        0.38,   # Band 2: Blue (0.32-0.44 μm)
        0.565,  # Band 3: Green-Red (0.44-0.69 μm)
        0.94,   # Band 4: Near-IR 1 (0.69-1.19 μm)
        2.595,  # Band 5: Near-IR 2 (1.19-4.00 μm)
    ])
    
    # Define all LW wavelengths (converted from wavenumber)
    lw_wavelengths = jnp.array([
        95.2,   # Band 0: Far-IR window (10-200 cm⁻¹)
        35.7,   # Band 1: H2O rotation (200-280 cm⁻¹)
        29.4,   # Band 2: CO2 bending (280-400 cm⁻¹)
        21.3,   # Band 3: CO2 v2 (400-540 cm⁻¹)
        14.9,   # Band 4: H2O continuum (540-800 cm⁻¹)
        11.1,   # Band 5: H2O + O3 (800-1000 cm⁻¹)
        9.1,    # Band 6: O3 + H2O (1000-1200 cm⁻¹)
        5.26,   # Band 7: H2O bands (1200-2600 cm⁻¹)
    ])
    
    # Get wavelength using JAX-compatible conditional
    sw_wl = jnp.where(
        band < len(sw_wavelengths),
        sw_wavelengths[band],
        0.55  # Default visible
    )
    
    lw_wl = jnp.where(
        band < len(lw_wavelengths),
        lw_wavelengths[band],
        10.0  # Default LW
    )
    
    return jnp.where(is_sw, sw_wl, lw_wl)


@jax.jit
def interpolate_refractive_index(wavelength: float) -> Tuple[float, float]:
    """
    Interpolate refractive index of water at given wavelength.
    
    Args:
        wavelength: Wavelength in micrometers
        
    Returns:
        Tuple of (real_part, imaginary_part)
    """
    # Available wavelengths and values
    wl_points = jnp.array([0.25, 0.31, 0.38, 0.55, 0.94, 2.5])
    n_real = jnp.array([1.362, 1.349, 1.340, 1.333, 1.327, 1.312])
    n_imag = jnp.array([1.1e-9, 1.8e-9, 2.5e-9, 5.7e-9, 2.89e-6, 1.38e-4])
    
    # Interpolate (linear in log space for imaginary part)
    real_part = jnp.interp(wavelength, wl_points, n_real)
    imag_part = jnp.exp(jnp.interp(wavelength, wl_points, jnp.log(n_imag)))
    
    return real_part, imag_part


@jax.jit
def mie_scattering_water(
    wavelength: float,
    radius: float,
    n_real: float,
    n_imag: float
) -> Tuple[float, float, float]:
    """
    Calculate Mie scattering properties for water droplets.
    
    Approximations based on Wiscombe (1980) and Bohren & Huffman (1983).
    
    Args:
        wavelength: Wavelength in micrometers
        radius: Droplet radius in micrometers
        n_real: Real part of refractive index
        n_imag: Imaginary part of refractive index
        
    Returns:
        Tuple of (extinction_efficiency, single_scatter_albedo, asymmetry_factor)
    """
    # Size parameter
    x = 2.0 * jnp.pi * radius / wavelength
    
    # Complex refractive index
    m = n_real + 1j * n_imag
    
    # Mie theory approximations for different size parameter regimes
    
    # Small particle limit (x << 1) - Rayleigh scattering
    def rayleigh_regime():
        # Rayleigh scattering approximation
        m_sq = m * jnp.conj(m)
        alpha = (m_sq - 1.0) / (m_sq + 2.0)
        
        q_sca = (8.0/3.0) * x**4 * jnp.real(alpha * jnp.conj(alpha))
        q_abs = 4.0 * x * jnp.imag(alpha)
        q_ext = q_sca + q_abs
        
        ssa = q_sca / jnp.maximum(q_ext, 1e-10)
        g = 0.0  # Rayleigh scattering is isotropic
        
        return q_ext, ssa, g
    
    # Large particle limit (x >> 1) - Geometric optics
    def geometric_regime():
        # Geometric optics approximation
        # Fresnel reflection coefficient at normal incidence
        rho = jnp.abs((m - 1.0) / (m + 1.0))**2
        
        # Total extinction efficiency approaches 2 for large particles
        q_ext = 2.0
        
        # Absorption efficiency - includes both volume absorption and surface effects
        # For water droplets, absorption is primarily due to imaginary part of refractive index
        # Volume absorption: 1 - exp(-4πni*r/λ) for ray passing through center
        # But need to account for ray distribution and internal reflections
        
        # Simplified absorption efficiency for large particles
        # Account for the fact that not all rays pass through the center
        absorption_path_factor = 1.33  # Average enhancement factor for non-central rays
        q_abs = 1.0 - jnp.exp(-4.0 * jnp.pi * n_imag * radius * absorption_path_factor / wavelength)
        
        # For very weakly absorbing particles (like water in visible)
        # absorption efficiency is much smaller than extinction
        q_abs = jnp.minimum(q_abs, 0.1)  # Limit absorption for realistic water droplets
        
        # Scattering efficiency
        q_sca = q_ext - q_abs
        
        ssa = q_sca / jnp.maximum(q_ext, 1e-10)
        g = 0.85  # Forward scattering for large particles
        
        return q_ext, ssa, g
    
    # Intermediate regime - approximate Mie solution
    def intermediate_regime():
        # Simplified Mie approximation based on van de Hulst (1957)
        m_sq = m * jnp.conj(m)
        
        # Extinction efficiency
        q_ext = 2.0 - (4.0/x) * jnp.sin(x) + (4.0/x**2) * (1.0 - jnp.cos(x))
        
        # Absorption efficiency
        q_abs = 1.0 - jnp.exp(-4.0 * jnp.pi * n_imag * radius / wavelength)
        
        # Scattering efficiency
        q_sca = q_ext - q_abs
        
        ssa = q_sca / jnp.maximum(q_ext, 1e-10)
        
        # Asymmetry factor approximation
        g = 0.5 + 0.35 * jnp.tanh(0.1 * (x - 10.0))
        
        return q_ext, ssa, g
    
    # Choose regime based on size parameter
    # Get results for all regimes
    q_ext_ray, ssa_ray, g_ray = rayleigh_regime()
    q_ext_geo, ssa_geo, g_geo = geometric_regime()
    q_ext_int, ssa_int, g_int = intermediate_regime()
    
    # Select based on size parameter
    q_ext = jnp.where(
        x < 0.3,
        q_ext_ray,
        jnp.where(x > 50.0, q_ext_geo, q_ext_int)
    )
    
    ssa = jnp.where(
        x < 0.3,
        ssa_ray,
        jnp.where(x > 50.0, ssa_geo, ssa_int)
    )
    
    g = jnp.where(
        x < 0.3,
        g_ray,
        jnp.where(x > 50.0, g_geo, g_int)
    )
    
    # Ensure physical bounds
    ssa = jnp.clip(ssa, 0.0, 1.0)
    g = jnp.clip(g, 0.0, 1.0)
    q_ext = jnp.maximum(q_ext, 0.0)
    
    return q_ext, ssa, g


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
    Calculate shortwave optical properties for liquid clouds using Mie scattering.
    
    Enhanced implementation with proper Mie scattering calculations.
    
    Args:
        cloud_water_path: Cloud water path (kg/m²)
        effective_radius: Droplet effective radius (microns)
        band: Spectral band index
        
    Returns:
        Tuple of (optical_depth, single_scatter_albedo, asymmetry_factor)
    """
    # Get wavelength for this band
    wavelength = get_band_wavelength(band, is_sw=True)
    
    # Get refractive index for water at this wavelength
    n_real, n_imag = interpolate_refractive_index(wavelength)
    
    # Calculate Mie scattering properties
    q_ext, ssa, g = mie_scattering_water(wavelength, effective_radius, n_real, n_imag)
    
    # Calculate optical depth from extinction efficiency
    # tau = N * sigma_ext * path_length
    # where N = number density, sigma_ext = extinction cross section
    
    # Water density = 1000 kg/m³
    # Droplet volume = (4/3) * π * r³
    # Number density = cwp / (droplet_mass) = cwp / (density * volume)
    
    droplet_volume = (4.0/3.0) * jnp.pi * (effective_radius * 1e-6)**3  # m³
    droplet_mass = 1000.0 * droplet_volume  # kg
    
    # Number of droplets per m² column
    number_density = cloud_water_path / droplet_mass  # droplets/m²
    
    # Extinction cross section
    sigma_ext = jnp.pi * (effective_radius * 1e-6)**2 * q_ext  # m²
    
    # Optical depth
    tau = number_density * sigma_ext
    
    # Ensure reasonable bounds
    tau = jnp.maximum(tau, 0.0)
    ssa = jnp.clip(ssa, 0.01, 0.99999)
    g = jnp.clip(g, 0.0, 0.99)
    
    return tau, ssa, g


@jax.jit
def ice_cloud_optics_sw(
    cloud_ice_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate shortwave optical properties for ice clouds.
    
    Enhanced parameterization based on ice crystal habits and database calculations.
    Based on Yang et al. (2013) and Baum et al. (2014).
    
    Args:
        cloud_ice_path: Cloud ice path (kg/m²)
        effective_radius: Ice crystal effective radius (microns)
        band: Spectral band index
        
    Returns:
        Tuple of (optical_depth, single_scatter_albedo, asymmetry_factor)
    """
    # Get wavelength for this band
    wavelength = get_band_wavelength(band, is_sw=True)
    
    # Ice crystal optical properties depend on wavelength and size
    # Based on comprehensive database calculations for various ice habits
    
    # Extinction efficiency for ice crystals (wavelength-dependent)
    # Fit to Yang et al. (2013) database results
    q_ext_base = jnp.where(
        wavelength < 0.7,
        2.2 - 0.1 * jnp.log(wavelength / 0.5),  # UV-Visible: higher extinction efficiency
        1.8 - 0.15 * jnp.log(wavelength / 0.7)  # Near-IR: lower extinction efficiency, size-dependent
    )
    
    # Size dependence - larger crystals are more efficient
    size_factor = 1.0 + 0.1 * jnp.log(effective_radius / 30.0)
    q_ext = q_ext_base * jnp.clip(size_factor, 0.7, 2.0)
    
    # Single scattering albedo - ice absorption increases with wavelength
    # Based on refractive index of ice (Warren & Brandt, 2008)
    ssa = jnp.where(
        wavelength < 0.4,
        0.9999 - 1e-6 * effective_radius,  # UV: very high scattering
        jnp.where(
            wavelength < 1.0,
            0.999 - 1e-5 * effective_radius - 0.001 * (wavelength - 0.4),  # Visible: high scattering
            0.98 - 0.1 * (wavelength - 1.0) - 1e-4 * effective_radius  # Near-IR: increased absorption
        )
    )
    
    ssa = jnp.clip(ssa, 0.5, 0.99999)
    
    # Asymmetry factor - ice crystals have strong forward scattering
    # Depends on crystal habit and size
    # Ice crystals are generally more forward-scattering than water droplets
    g_base = 0.82 + 0.12 * jnp.tanh((effective_radius - 50.0) / 20.0)
    
    # Wavelength dependence - more forward scattering at shorter wavelengths
    wavelength_factor = 1.0 + 0.03 * jnp.exp(-(wavelength - 0.55)**2 / 0.1)
    g = g_base * wavelength_factor
    
    g = jnp.clip(g, 0.7, 0.95)
    
    # Calculate optical depth
    # Ice density ≈ 917 kg/m³
    crystal_volume = (4.0/3.0) * jnp.pi * (effective_radius * 1e-6)**3  # m³
    crystal_mass = 917.0 * crystal_volume  # kg
    
    # Number of crystals per m² column
    number_density = cloud_ice_path / crystal_mass  # crystals/m²
    
    # Extinction cross section
    sigma_ext = jnp.pi * (effective_radius * 1e-6)**2 * q_ext  # m²
    
    # Optical depth
    tau = number_density * sigma_ext
    
    # Ensure reasonable bounds
    tau = jnp.maximum(tau, 0.0)
    
    return tau, ssa, g


@jax.jit
def liquid_cloud_optics_lw(
    cloud_water_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate longwave optical properties for liquid clouds.
    
    Enhanced implementation with improved spectral dependence.
    Longwave assumes pure absorption (no scattering).
    
    Args:
        cloud_water_path: Cloud water path (kg/m²)
        effective_radius: Droplet effective radius (microns)
        band: Spectral band index (0-7)
        
    Returns:
        Optical depth (absorption)
    """
    # Get wavelength for this band
    wavelength = get_band_wavelength(band, is_sw=False)
    
    # Enhanced absorption coefficient depends on band
    # Based on water absorption spectrum in IR
    k_abs = jnp.where(
        band == 0, 25.0,   # Far-IR window (10-200 cm⁻¹)
        jnp.where(
            band == 1, 180.0,  # H2O rotation band (200-280 cm⁻¹) - high absorption
            jnp.where(
                band == 2, 90.0,   # CO2 bending + H2O (280-400 cm⁻¹)
                jnp.where(
                    band == 3, 120.0,  # CO2 v2 + H2O (400-540 cm⁻¹)
                    jnp.where(
                        band == 4, 160.0,  # H2O continuum (540-800 cm⁻¹) - very high
                        jnp.where(
                            band == 5, 140.0,  # H2O + O3 (800-1000 cm⁻¹)
                            jnp.where(
                                band == 6, 100.0,  # O3 + H2O (1000-1200 cm⁻¹)
                                200.0               # H2O bands (1200-2600 cm⁻¹) - strongest
                            )
                        )
                    )
                )
            )
        )
    )
    
    # Size dependence - smaller droplets have slightly higher absorption per unit mass
    size_factor = jnp.sqrt(12.0 / effective_radius)
    
    # Wavelength dependence - longer wavelengths have higher absorption
    wavelength_factor = 1.0 + 0.1 * jnp.log(wavelength / 10.0)
    
    tau = k_abs * cloud_water_path * size_factor * wavelength_factor
    
    return jnp.maximum(tau, 0.0)


@jax.jit
def ice_cloud_optics_lw(
    cloud_ice_path: jnp.ndarray,
    effective_radius: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """
    Calculate longwave optical properties for ice clouds.
    
    Enhanced implementation with improved spectral dependence.
    
    Args:
        cloud_ice_path: Cloud ice path (kg/m²)
        effective_radius: Ice crystal effective radius (microns)
        band: Spectral band index (0-7)
        
    Returns:
        Optical depth (absorption)
    """
    # Get wavelength for this band
    wavelength = get_band_wavelength(band, is_sw=False)
    
    # Ice absorption coefficient depends on band
    # Ice is generally less absorbing than liquid water
    k_abs = jnp.where(
        band == 0, 12.0,   # Far-IR window (10-200 cm⁻¹)
        jnp.where(
            band == 1, 85.0,   # H2O rotation band (200-280 cm⁻¹) - moderate absorption
            jnp.where(
                band == 2, 45.0,   # CO2 bending + H2O (280-400 cm⁻¹)
                jnp.where(
                    band == 3, 60.0,   # CO2 v2 + H2O (400-540 cm⁻¹)
                    jnp.where(
                        band == 4, 90.0,   # H2O continuum (540-800 cm⁻¹) - higher
                        jnp.where(
                            band == 5, 75.0,   # H2O + O3 (800-1000 cm⁻¹)
                            jnp.where(
                                band == 6, 55.0,   # O3 + H2O (1000-1200 cm⁻¹)
                                110.0               # H2O bands (1200-2600 cm⁻¹) - strongest
                            )
                        )
                    )
                )
            )
        )
    )
    
    # Size dependence - larger crystals have different absorption characteristics
    size_factor = jnp.sqrt(35.0 / effective_radius)
    
    # Wavelength dependence - similar to liquid but weaker
    wavelength_factor = 1.0 + 0.05 * jnp.log(wavelength / 10.0)
    
    tau = k_abs * cloud_ice_path * size_factor * wavelength_factor
    
    return jnp.maximum(tau, 0.0)


@jax.jit
def cloud_optics(
    cloud_water_path: jnp.ndarray,
    cloud_ice_path: jnp.ndarray,
    temperature: jnp.ndarray,
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
    
    # Apply to all SW bands - use fixed shape
    from .constants import N_SW_BANDS
    tau_sw = jnp.zeros((nlev, N_SW_BANDS))
    ssa_sw = jnp.zeros((nlev, N_SW_BANDS))
    g_sw = jnp.zeros((nlev, N_SW_BANDS))
    
    for band in range(N_SW_BANDS):
        tau_band, ssa_band, g_band = calculate_sw_band(band)
        tau_sw = tau_sw.at[:, band].set(tau_band)
        ssa_sw = ssa_sw.at[:, band].set(ssa_band)
        g_sw = g_sw.at[:, band].set(g_band)
    
    # Calculate LW properties for all bands
    def calculate_lw_band(band):
        tau_liq = liquid_cloud_optics_lw(cloud_water_path, r_eff_liq, band)
        tau_ice = ice_cloud_optics_lw(cloud_ice_path, r_eff_ice, band)
        return tau_liq + tau_ice
    
    # Apply to all LW bands - use fixed shape
    from .constants import N_LW_BANDS
    tau_lw = jnp.zeros((nlev, N_LW_BANDS))
    
    for band in range(N_LW_BANDS):
        tau_lw = tau_lw.at[:, band].set(calculate_lw_band(band))
    
    # Create optical properties
    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=ssa_sw,
        asymmetry_factor=g_sw
    )
    
    # LW properties (pure absorption)
    lw_ssa = jnp.zeros((nlev, N_LW_BANDS))
    lw_g = jnp.zeros((nlev, N_LW_BANDS))
    
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=lw_ssa,  # Pure absorption
        asymmetry_factor=lw_g
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
    sw_optics, lw_optics = cloud_optics(cwp, cip, temperature)
    
    # Check shapes
    from .constants import N_SW_BANDS, N_LW_BANDS
    assert sw_optics.optical_depth.shape == (nlev, N_SW_BANDS)
    assert lw_optics.optical_depth.shape == (nlev, N_LW_BANDS)
    
    # Check values
    assert jnp.all(sw_optics.optical_depth >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo >= 0)
    assert jnp.all(sw_optics.single_scatter_albedo <= 1)
    assert jnp.all(sw_optics.asymmetry_factor >= 0)
    assert jnp.all(sw_optics.asymmetry_factor <= 1)
    
    print("Cloud optics tests passed!")


if __name__ == "__main__":
    test_cloud_optics()