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
from functools import partial

from .radiation_types import OpticalProperties


@jax.jit
def two_stream_coefficients(
    tau: jnp.ndarray,
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
    
    # Exponential terms
    exp_plus = jnp.exp(lambda_val * tau)
    exp_minus = jnp.exp(-lambda_val * tau)
    
    # Avoid numerical issues for large optical depths
    exp_minus = jnp.where(tau > 100, 0.0, exp_minus)
    
    # Helper terms
    term1 = 1.0 / (lambda_val + gamma1)
    term2 = 1.0 / (lambda_val - gamma1)
    
    # Diffuse reflectance and transmittance
    R_dif = gamma2 * (exp_plus - exp_minus) / (exp_plus - gamma2**2 / gamma1**2 * exp_minus)
    T_dif = (1.0 - R_dif * gamma2 / gamma1) * exp_minus
    
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


@partial(jax.jit, static_argnames=['n_bands'])
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
    
    # Apply to all bands
    band_indices = jnp.arange(n_bands)
    flux_up, flux_down = jax.vmap(process_band)(band_indices)
    
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
    def direct_transmission(carry, T):
        trans_above = carry
        trans_below = trans_above * T
        return trans_below, trans_below
    
    _, direct_trans = jax.lax.scan(
        direct_transmission,
        1.0,  # Full transmission at TOA
        T_dir
    )
    
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
        flux_above = R * flux_below + S
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
        flux_below = T * flux_above + R * flux_up + S
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


@partial(jax.jit, static_argnames=['n_bands'])
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
    
    # Apply to all bands
    band_indices = jnp.arange(n_bands)
    flux_up, flux_down, flux_direct, flux_diffuse = jax.vmap(process_band)(band_indices)
    
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
    net_flux = flux_up - flux_down
    
    # Flux divergence in layers
    flux_div = net_flux[1:] - net_flux[:-1]
    
    # Pressure thickness
    dp = pressure_interfaces[1:] - pressure_interfaces[:-1]
    
    # Heating rate
    g = 9.81
    heating = -g / cp * flux_div / dp
    
    return heating


# Test functions
def test_two_stream_coefficients():
    """Test two-stream coefficient calculations"""
    tau = jnp.array([0.1, 0.5, 1.0])
    ssa = jnp.array([0.9, 0.8, 0.7])
    g = jnp.array([0.85, 0.85, 0.85])
    
    # Test LW (no solar angle)
    gamma1, gamma2, gamma3, gamma4 = two_stream_coefficients(tau, ssa, g, mu0=None)
    assert gamma1.shape == tau.shape
    assert jnp.all(gamma3 == 0)  # No direct beam
    assert jnp.all(gamma4 == 1)
    
    # Test SW
    mu0 = 0.5
    gamma1, gamma2, gamma3, gamma4 = two_stream_coefficients(tau, ssa, g, mu0)
    assert jnp.all(gamma3 > 0)
    assert jnp.all(gamma3 + gamma4 == 1.0)
    
    print("✓ Two-stream coefficients test passed")


def test_layer_properties():
    """Test layer reflectance and transmittance"""
    tau = jnp.array([0.1, 1.0, 10.0])
    ssa = jnp.array([0.9, 0.9, 0.9])
    g = jnp.array([0.85, 0.85, 0.85])
    
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(tau, ssa, g, mu0=0.5)
    
    # Physical constraints
    assert jnp.all(R_dif >= 0) and jnp.all(R_dif <= 1)
    assert jnp.all(T_dif >= 0) and jnp.all(T_dif <= 1)
    assert jnp.all(R_dif + T_dif <= 1)  # Energy conservation
    
    # Larger optical depth = less transmission
    assert T_dif[0] > T_dif[1] > T_dif[2]
    
    print("✓ Layer properties test passed")


def test_adding_method():
    """Test adding method for combining layers"""
    R1 = jnp.array(0.2)
    T1 = jnp.array(0.7)
    R2 = jnp.array(0.3)
    T2 = jnp.array(0.6)
    
    R_combined, T_combined = adding_method(R1, T1, R2, T2)
    
    # Should have more reflection than either layer alone
    assert R_combined > R1
    assert R_combined > R2
    
    # Transmission should be reasonable
    assert 0 <= T_combined <= 1
    assert T_combined > 0  # Some transmission through both layers
    
    print("✓ Adding method test passed")


def test_heating_rate():
    """Test flux to heating rate conversion"""
    nlev = 10
    flux_up = jnp.linspace(100, 300, nlev + 1)
    flux_down = jnp.linspace(400, 200, nlev + 1)
    pressure = jnp.linspace(100000, 10000, nlev + 1)
    
    heating = flux_to_heating_rate(flux_up, flux_down, pressure)
    
    assert heating.shape == (nlev,)
    # Net flux divergence should give heating/cooling
    assert jnp.any(heating != 0)
    
    print("✓ Heating rate test passed")


def test_two_stream_integration():
    """Integration test for two-stream solver"""
    from .radiation_types import OpticalProperties
    from .planck import planck_bands
    
    nlev = 20
    n_lw_bands = 3
    n_sw_bands = 2
    
    # Create test optical properties
    tau_lw = jnp.ones((nlev, n_lw_bands)) * 0.5
    tau_sw = jnp.ones((nlev, n_sw_bands)) * 0.3
    
    lw_optics = OpticalProperties(
        optical_depth=tau_lw,
        single_scatter_albedo=jnp.zeros((nlev, n_lw_bands)),
        asymmetry_factor=jnp.zeros((nlev, n_lw_bands))
    )
    
    sw_optics = OpticalProperties(
        optical_depth=tau_sw,
        single_scatter_albedo=jnp.ones((nlev, n_sw_bands)) * 0.9,
        asymmetry_factor=jnp.ones((nlev, n_sw_bands)) * 0.85
    )
    
    # Temperature profile
    temperature = jnp.linspace(250, 290, nlev)
    
    # Planck functions (simplified)
    lw_bands = ((10, 350), (350, 500), (500, 2500))
    planck_layer = planck_bands(temperature, lw_bands, n_lw_bands)
    planck_interface = planck_bands(
        jnp.linspace(250, 290, nlev + 1), lw_bands, n_lw_bands
    )
    
    # Surface properties
    surface_emissivity = 0.98
    surface_temp = 290.0
    surface_planck = planck_bands(jnp.array([surface_temp]), lw_bands, n_lw_bands)[0]
    
    # Test LW
    flux_up_lw, flux_down_lw = longwave_fluxes(
        lw_optics, planck_layer, planck_interface,
        surface_emissivity, surface_planck, n_lw_bands
    )
    
    assert flux_up_lw.shape == (nlev + 1, n_lw_bands)
    assert flux_down_lw.shape == (nlev + 1, n_lw_bands)
    assert jnp.all(flux_up_lw >= 0)
    
    # Test SW
    cos_zenith = 0.5
    toa_flux = jnp.array([500.0, 500.0])  # W/m²
    surface_albedo = jnp.array([0.15, 0.15])
    
    flux_up_sw, flux_down_sw, flux_dir, flux_dif = shortwave_fluxes(
        sw_optics, cos_zenith, toa_flux, surface_albedo, n_sw_bands
    )
    
    assert flux_up_sw.shape == (nlev + 1, n_sw_bands)
    assert jnp.all(flux_down_sw >= flux_up_sw)  # Net downward in SW
    
    print("✓ Two-stream integration test passed")


def test_two_stream():
    """Run all two-stream tests"""
    test_two_stream_coefficients()
    test_layer_properties()
    test_adding_method()
    test_heating_rate()
    test_two_stream_integration()
    print("\nAll two-stream solver tests passed!")
    

if __name__ == "__main__":
    test_two_stream()