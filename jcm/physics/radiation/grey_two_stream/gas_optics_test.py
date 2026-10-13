"""Unit tests for gas optics calculations

Tests gas absorption coefficients and optical depths for atmospheric gases
including H2O, CO2, and O3 in both longwave and shortwave bands.

Date: 2025-01-10
"""

import pytest
import jax.numpy as jnp
from jcm.physics.radiation.grey_two_stream.gas_optics import (
    water_vapor_continuum,
    co2_absorption,
    ozone_absorption_sw,
    ozone_absorption_lw,
    gas_optical_depth_lw,
    gas_optical_depth_sw
)
from jcm.physics.radiation.cloud_optics import (
    get_band_wavelength,
    surface_albedo_by_sw_band,
)
from jcm.physics.radiation.constants import N_SW_BANDS


def test_grey_sw_band_inputs_are_all_aligned_to_band_order():
    """Every grey SW band-indexed input agrees on which band is near-IR.

    End-to-end guard against the #678 family trap: the SW band order lives in
    ``SW_BAND_LIMITS`` (band 0 = near-IR, band 1 = UV/visible), and four
    band-indexed inputs must all follow it -- the cloud-optics wavelength, the
    surface albedo, ozone absorption (strong in UV/visible), and water-vapour
    absorption (near-IR). A future reorder that touches only some of them
    (as the original fix did) fails this test.

    For each band it pins the whole tuple: wavelength band ⇒ albedo component ⇒
    dominant gas absorber.
    """
    assert N_SW_BANDS == 2
    temperature = jnp.array([250.0])
    o3 = jnp.array([1e-6])
    # Distinct albedos so the ordering is observable.
    albedo = surface_albedo_by_sw_band(jnp.array(0.10), jnp.array(0.90))
    # Water-vapour-only SW optical depth per band.
    h2o_tau = gas_optical_depth_sw(
        temperature, jnp.array([5.0e4]), jnp.array([0.01]),
        jnp.zeros(1), jnp.array([1000.0]), jnp.array([0.7]), jnp.array(0.5),
    )[0]
    ozone_k = jnp.array(
        [float(ozone_absorption_sw(o3, temperature, b)[0])
         for b in range(N_SW_BANDS)]
    )

    near_ir_bands = [b for b in range(N_SW_BANDS)
                     if float(get_band_wavelength(b, True)) > 0.7]
    vis_bands = [b for b in range(N_SW_BANDS)
                 if float(get_band_wavelength(b, True)) <= 0.7]
    assert len(near_ir_bands) == 1 and len(vis_bands) == 1
    nir, vis = near_ir_bands[0], vis_bands[0]

    # Near-IR band: near-IR albedo, water-vapour absorption, weak ozone.
    assert float(albedo[nir]) == pytest.approx(0.90)
    assert float(h2o_tau[nir]) > 0.0
    # UV/visible band: visible albedo, no water vapour, strong ozone.
    assert float(albedo[vis]) == pytest.approx(0.10)
    assert float(h2o_tau[vis]) == 0.0
    assert float(ozone_k[vis]) > float(ozone_k[nir])


def test_water_vapor_continuum():
    """Test water vapor continuum absorption"""
    nlev = 10
    temperature = jnp.linspace(288.0, 220.0, nlev)
    pressure = jnp.linspace(100000.0, 10000.0, nlev)
    h2o_vmr = jnp.linspace(0.01, 0.001, nlev)
    
    # Test different bands
    for band in range(3):
        k_abs = water_vapor_continuum(temperature, pressure, h2o_vmr, band)
        
        # Should return valid absorption coefficients
        assert k_abs.shape == (nlev,)
        assert jnp.all(k_abs >= 0)  # Absorption should be non-negative
        assert not jnp.any(jnp.isnan(k_abs))
        
    # Band 2 should have higher absorption than band 0
    k_band0 = water_vapor_continuum(temperature, pressure, h2o_vmr, 0)
    k_band2 = water_vapor_continuum(temperature, pressure, h2o_vmr, 2)
    assert jnp.all(k_band2 > k_band0)


def test_co2_absorption():
    """Test CO2 absorption"""
    nlev = 5
    temperature = jnp.linspace(288.0, 220.0, nlev)
    pressure = jnp.linspace(100000.0, 10000.0, nlev)
    co2_vmr = 400e-6
    
    # Test different bands
    for band in range(3):
        k_abs = co2_absorption(temperature, pressure, co2_vmr, band)
        
        # Should return valid absorption coefficients
        assert k_abs.shape == (nlev,)
        assert jnp.all(k_abs >= 0)
        assert not jnp.any(jnp.isnan(k_abs))
    
    # Band 1 (350-500 cm⁻¹) contains the CO2 15 μm band (667 cm⁻¹ actually
    # lies just above but the coarse 3-band LW structure places the bulk of
    # the CO2 absorption here); band 2 has a reduced tail (15% of band 1).
    k_band0 = co2_absorption(temperature, pressure, co2_vmr, 0)
    k_band1 = co2_absorption(temperature, pressure, co2_vmr, 1)
    k_band2 = co2_absorption(temperature, pressure, co2_vmr, 2)
    k_band3 = co2_absorption(temperature, pressure, co2_vmr, 3)
    assert jnp.all(k_band1 >= k_band0)
    assert jnp.all(k_band1 >= k_band2)
    assert jnp.all(k_band1 >= k_band3)


def test_ozone_absorption_sw():
    """Test ozone shortwave absorption"""
    nlev = 5
    o3_vmr = jnp.ones(nlev) * 1e-6
    temperature = jnp.ones(nlev) * 273.15  # Standard temperature
    
    # Band order follows SW_BAND_LIMITS: band 0 = near-IR, band 1 = UV/visible
    # (#678). Ozone absorbs in the UV/visible far more than the near-IR.
    k_nir = ozone_absorption_sw(o3_vmr, temperature, 0)
    k_vis = ozone_absorption_sw(o3_vmr, temperature, 1)

    assert jnp.all(k_vis > 0)
    assert jnp.all(k_nir > 0)  # near-IR band still has some absorption
    assert jnp.all(k_vis > k_nir)  # UV/visible (band 1) should be stronger

    # Should be proportional to O3 VMR
    o3_vmr_double = o3_vmr * 2
    k_vis_double = ozone_absorption_sw(o3_vmr_double, temperature, 1)
    assert jnp.allclose(k_vis_double, k_vis * 2, rtol=1e-10)


def test_ozone_absorption_lw():
    """Test ozone longwave absorption"""
    nlev = 5
    temperature = jnp.linspace(288.0, 220.0, nlev)
    o3_vmr = jnp.ones(nlev) * 1e-6
    
    # Test different bands — O3 9.6 μm band (1042 cm⁻¹) now falls in band 2
    # (500-2500 cm⁻¹) under the coarse 3-band LW structure.
    k_band0 = ozone_absorption_lw(temperature, o3_vmr, 0)
    k_band1 = ozone_absorption_lw(temperature, o3_vmr, 1)
    k_band2 = ozone_absorption_lw(temperature, o3_vmr, 2)

    assert jnp.all(k_band0 == 0)
    assert jnp.all(k_band1 == 0)
    assert jnp.all(k_band2 > 0)  # Main O3 9.6 μm band

    # Should have temperature dependence
    temp_low = jnp.ones(nlev) * 200.0
    temp_high = jnp.ones(nlev) * 300.0
    k_low = ozone_absorption_lw(temp_low, o3_vmr, 2)
    k_high = ozone_absorption_lw(temp_high, o3_vmr, 2)
    assert not jnp.allclose(k_low, k_high)


def test_gas_optical_depth_lw():
    """Test longwave gas optical depth calculation"""
    nlev = 10
    temperature = jnp.linspace(288.0, 220.0, nlev)
    pressure = jnp.linspace(100000.0, 10000.0, nlev)
    h2o_vmr = jnp.linspace(0.01, 0.001, nlev)
    o3_vmr = jnp.ones(nlev) * 1e-6
    co2_vmr = 400e-6
    
    # Layer properties
    air_density = pressure / (287.0 * temperature)
    layer_thickness = jnp.ones(nlev) * 2000.0  # 2 km layers
    
    tau = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, co2_vmr,
        layer_thickness, air_density
    )
    
    # Check output shape
    from jcm.physics.radiation.constants import N_LW_BANDS
    assert tau.shape == (nlev, N_LW_BANDS)
    
    # Optical depth should be non-negative
    assert jnp.all(tau >= 0)
    
    # Should not have NaN values
    assert not jnp.any(jnp.isnan(tau))
    
    # Should have reasonable optical depth values
    # Note: actual relationship depends on specific atmospheric profile
    assert jnp.any(tau > 0)  # Should have some absorption


def test_gas_optical_depth_sw():
    """Test shortwave gas optical depth calculation"""
    nlev = 10
    pressure = jnp.linspace(100000.0, 10000.0, nlev)
    h2o_vmr = jnp.linspace(0.01, 0.001, nlev)
    o3_vmr = jnp.ones(nlev) * 1e-6
    
    # Layer properties
    temperature = jnp.linspace(288.0, 220.0, nlev)
    air_density = pressure / (287.0 * temperature)
    layer_thickness = jnp.ones(nlev) * 2000.0
    cos_zenith = 0.5
    
    tau = gas_optical_depth_sw(
        temperature, pressure, h2o_vmr, o3_vmr, layer_thickness,
        air_density, cos_zenith
    )

    # Check output shape
    from jcm.physics.radiation.constants import N_SW_BANDS
    assert tau.shape == (nlev, N_SW_BANDS)

    # Optical depth should be non-negative
    assert jnp.all(tau >= 0)

    # Should not have NaN values
    assert not jnp.any(jnp.isnan(tau))

    # Both bands should have some absorption (O3 dominates band 0, H2O dominates band 1)
    assert jnp.sum(tau[:, 0]) > 0
    assert jnp.sum(tau[:, 1]) > 0


def test_gas_optical_depth_lw_realistic():
    """Test LW optical depth with realistic atmospheric profile"""
    nlev = 20
    
    # Realistic atmospheric profile
    pressure = jnp.logspace(jnp.log10(100000), jnp.log10(1000), nlev)  # Pa
    temperature = 288.0 - 6.5e-3 * jnp.logspace(jnp.log10(1), jnp.log10(15000), nlev)  # Standard lapse rate
    h2o_vmr = 0.01 * jnp.exp(-jnp.linspace(0, 10, nlev))  # Exponential decrease
    o3_vmr = jnp.where(
        pressure < 20000,  # Stratosphere
        3e-6, 1e-6  # Higher O3 in stratosphere
    )
    
    # Calculate air density and layer thickness
    air_density = pressure / (287.0 * temperature)
    height = -287.0 * temperature * jnp.log(pressure / 100000.0) / 9.81
    layer_thickness = jnp.concatenate([
        jnp.abs(jnp.diff(height)),  # Ensure positive thickness
        jnp.array([jnp.abs(height[-1] - height[-2])])
    ])
    
    tau = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, 400e-6,
        layer_thickness, air_density
    )
    
    # Should produce reasonable optical depths
    assert jnp.all(tau >= 0)
    assert jnp.all(tau < 100)  # Not unreasonably large
    assert not jnp.any(jnp.isnan(tau))
    
    # Water vapor should dominate in lower atmosphere
    assert jnp.max(tau) > 1.0  # Should have some significant absorption


def test_gas_optical_depth_zero_vmr():
    """Test optical depth with zero gas concentrations"""
    nlev = 5
    temperature = jnp.ones(nlev) * 288.0
    pressure = jnp.ones(nlev) * 50000.0
    h2o_vmr = jnp.zeros(nlev)
    o3_vmr = jnp.zeros(nlev)
    co2_vmr = 0.0
    
    air_density = jnp.ones(nlev) * 0.6
    layer_thickness = jnp.ones(nlev) * 1000.0
    
    tau_lw = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, co2_vmr,
        layer_thickness, air_density
    )
    
    tau_sw = gas_optical_depth_sw(
        pressure, temperature, h2o_vmr, o3_vmr, layer_thickness,
        air_density, 0.5
    )
    
    # With zero concentrations, should get zero optical depth
    assert jnp.allclose(tau_lw, 0.0)
    assert jnp.allclose(tau_sw, 0.0)


def test_gas_optical_depth_scaling():
    """Test optical depth scaling with layer thickness and density"""
    nlev = 5
    temperature = jnp.ones(nlev) * 288.0
    pressure = jnp.ones(nlev) * 50000.0
    h2o_vmr = jnp.ones(nlev) * 0.01
    o3_vmr = jnp.ones(nlev) * 1e-6
    co2_vmr = 400e-6
    
    # Base case
    air_density = jnp.ones(nlev) * 0.6
    layer_thickness = jnp.ones(nlev) * 1000.0
    
    tau_base = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, co2_vmr,
        layer_thickness, air_density
    )
    
    # Double layer thickness
    tau_thick = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, co2_vmr,
        layer_thickness * 2, air_density
    )
    
    # Double air density
    tau_dense = gas_optical_depth_lw(
        temperature, pressure, h2o_vmr, o3_vmr, co2_vmr,
        layer_thickness, air_density * 2
    )
    
    # Optical depth should scale linearly with thickness and density
    assert jnp.allclose(tau_thick, tau_base * 2, rtol=1e-10)
    assert jnp.allclose(tau_dense, tau_base * 2, rtol=1e-10)