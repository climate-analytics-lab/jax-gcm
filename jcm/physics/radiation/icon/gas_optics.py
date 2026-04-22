"""Gas optics for radiation calculations

This module computes absorption coefficients and optical depths
for atmospheric gases in both shortwave and longwave spectral regions.

Simplified implementation with parameterized absorption.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax


@jax.jit
def water_vapor_continuum(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    h2o_vmr: jnp.ndarray,
    band: int
) -> jnp.ndarray:
    """Calculate water vapor continuum absorption.
    
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

    # Clip to physical range so CFL violations (T<<100 or T>>400) cannot
    # produce NaN via division or exp overflow
    # Wide safety clip — prevents exp overflow in T_factor calculations at
    # pathological values, not a physical-range bound
    temperature = jnp.clip(temperature, 50.0, 500.0)
    pressure = jnp.maximum(pressure, 1.0)
    h2o_vmr = jnp.clip(h2o_vmr, 0.0, 0.99)

    # Convert VMR to mass mixing ratio
    # q = vmr * (M_h2o / M_air) ≈ vmr * 0.622
    h2o_mmr = h2o_vmr * 0.622

    # Temperature dependence (cap exponent to avoid overflow in AD)
    T_factor = jnp.exp(jnp.clip(1800.0 * (1.0/temperature - 1.0/T_ref), -50.0, 50.0))

    # Pressure scaling
    P_factor = pressure / P_ref
    
    # Band-dependent coefficients for 3-band LW structure:
    #   Band 0: 10-350 cm⁻¹  (far-IR + H2O rotation)
    #   Band 1: 350-500 cm⁻¹ (CO2 15μm + H2O window)
    #   Band 2: 500-2500 cm⁻¹ (H2O continuum + O3 9.6μm)

    # Self-broadening coefficients (H2O-H2O interactions)
    # 2x scaling on the continuum bands (far-IR + window + bands) to bring
    # SFC LW down closer to Earth's ~340 W/m² (was 185 W/m² at 1x). Combined
    # with hybrid coords + 0.5x diffusion + dt=3min for stability under the
    # stronger surface forcing that this implies.
    k_self = jnp.where(
        band == 0, 0.20,   # Far-IR + rotation: strong H2O absorption
        jnp.where(
            band == 1, 0.20,   # CO2 + H2O window: moderate
            0.60               # H2O continuum + bands: strongest
        )
    )

    # Foreign-broadening coefficients (H2O-N2/O2 interactions)
    k_foreign = jnp.where(
        band == 0, 0.040,  # Far-IR + rotation
        jnp.where(
            band == 1, 0.050,  # CO2 window
            0.130              # H2O continuum + bands
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
    """Calculate CO2 absorption.
    
    Simplified parameterization for CO2 15-micron band.
    
    Args:
        temperature: Temperature (K) [nlev]
        pressure: Pressure (Pa) [nlev]
        co2_vmr: CO2 volume mixing ratio (constant)
        band: Spectral band index
        
    Returns:
        Absorption coefficient (m²/kg)

    """
    # CO2 15μm band (667 cm⁻¹) falls in band 1 (350-500 cm⁻¹).
    # Some weak CO2 absorption extends into band 2 (500-2500 cm⁻¹).
    return jnp.where(
        band == 1,
        _calculate_co2_band1(temperature, pressure, co2_vmr),
        jnp.where(
            band == 2,
            _calculate_co2_band1(temperature, pressure, co2_vmr) * 0.15,
            jnp.zeros_like(temperature)
        )
    )


def _calculate_co2_band1(temperature, pressure, co2_vmr):
    """Enhanced CO2 absorption calculation with improved temperature/pressure dependence.
    
    Based on HITRAN line data parameterization for the 15 μm CO2 band.
    """
    # Reference conditions
    T_ref = 296.0
    P_ref = 101325.0

    # Clip to physical range
    # Wide safety clip — prevents exp overflow in T_factor calculations at
    # pathological values, not a physical-range bound
    temperature = jnp.clip(temperature, 50.0, 500.0)
    pressure = jnp.maximum(pressure, 1.0)

    # Enhanced temperature dependence for CO2 line strength
    # Based on HITRAN formula: S(T) = S_ref * (T_ref/T) * exp(-E_low/k*(1/T - 1/T_ref))
    # where E_low is the lower state energy
    E_low_k = 960.0  # Lower state energy / Boltzmann constant (K) for 15 μm band

    T_factor = (T_ref / temperature) * jnp.exp(
        jnp.clip(-E_low_k * (1.0/temperature - 1.0/T_ref), -50.0, 50.0)
    )

    # Improved pressure broadening with temperature dependence
    # γ(T,P) = γ_ref * (T_ref/T)^n * P/P_ref
    n_temp = 0.69  # Temperature exponent for CO2 line widths
    P_factor = (pressure / P_ref) * (T_ref / temperature)**n_temp
    
    # Enhanced absorption coefficient based on spectroscopic data
    # Includes both line absorption and continuum effects
    # 2x scaling — see comment on k_self above
    k_ref = 0.30
    
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
    """Enhanced ozone absorption in shortwave with temperature-dependent UV cross-sections.
    
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

    # Clip to physical range
    # Wide safety clip — prevents exp overflow in T_factor calculations at
    # pathological values, not a physical-range bound
    temperature = jnp.clip(temperature, 50.0, 500.0)
    o3_vmr = jnp.clip(o3_vmr, 0.0, 1.0)

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

    # 2 SW bands: 0 = UV+visible (4000-14500 cm⁻¹), 1 = near-IR (14500-50000 cm⁻¹)
    k_o3_by_band = jnp.array([
        k_o3_uv_vis,        # UV + visible: strong O3 (Hartley-Huggins-Chappuis)
        k_o3_nir * 0.5,     # Near-IR: very weak
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
    """Calculate ozone absorption in longwave.
    
    Simplified parameterization for 9.6 micron band.
    
    Args:
        temperature: Temperature (K) [nlev]
        o3_vmr: Ozone volume mixing ratio [nlev]
        band: Spectral band index
        
    Returns:
        Absorption coefficient (m²/kg)

    """
    # Ozone 9.6μm band (1042 cm⁻¹) falls entirely in band 2 (500-2500 cm⁻¹)
    T_ref = 296.0
    # Wide safety clip — prevents exp overflow in T_factor calculations at
    # pathological values, not a physical-range bound
    temperature = jnp.clip(temperature, 50.0, 500.0)
    T_factor = jnp.sqrt(T_ref / temperature)
    k_o3_main = 50.0
    o3_mmr = jnp.clip(o3_vmr, 0.0, 1.0) * (48.0 / 29.0)

    return jnp.where(
        band == 2,
        k_o3_main * T_factor * o3_mmr,
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
    air_density: jnp.ndarray
) -> jnp.ndarray:
    """Calculate longwave gas optical depths.
    
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

    # Apply to all LW bands via vmap (was a Python ``for band in range(N_LW_BANDS)``
    # loop that staged N_LW_BANDS separate ``.at[:, band].set(...)`` updates into
    # XLA, producing a long unrolled dependency chain).
    from .constants import N_LW_BANDS
    bands = jnp.arange(N_LW_BANDS)
    tau_per_band = jax.vmap(single_band_absorption)(bands)  # (N_LW_BANDS, nlev)
    return tau_per_band.T  # (nlev, N_LW_BANDS)


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
    """Calculate shortwave gas optical depths with enhanced temperature dependence.
    
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

    # Apply to all SW bands via vmap (same motivation as ``gas_optical_depth_lw``).
    from .constants import N_SW_BANDS
    bands = jnp.arange(N_SW_BANDS)
    tau_per_band = jax.vmap(single_band_absorption)(bands)  # (N_SW_BANDS, nlev)
    return tau_per_band.T  # (nlev, N_SW_BANDS)


@jax.jit
def rayleigh_optical_depth(
    pressure: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    wavelength: float = 0.55  # microns
) -> jnp.ndarray:
    """Calculate Rayleigh scattering optical depth.
    
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


