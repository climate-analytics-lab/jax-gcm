"""
Unit tests for Tiedtke-Nordeng convection scheme

This file provides comprehensive unit tests that can be run with pytest.

Date: 2025-01-09
"""

import pytest
import jax.numpy as jnp
import jax
import numpy as np

# Import convection modules
from jcm.physics.icon.convection.tiedtke_nordeng import (
    ConvectionParameters,
    saturation_mixing_ratio,
    find_cloud_base,
    calculate_cape_cin
)
from jcm.physics.icon.convection.tracer_transport import (
    TracerIndices,
    initialize_tracers
)


def create_realistic_atmosphere(nlev=20, unstable=True):
    """Create a realistic atmospheric profile for testing"""
    # Pressure levels (Pa) - from surface (1000 hPa) to top (~200 hPa)
    pressure = jnp.linspace(1e5, 2e4, nlev)
    
    # Height (m) - increases with decreasing pressure
    height = jnp.linspace(0, 12000, nlev)
    
    if unstable:
        # Unstable profile - warm at surface with normal lapse rate
        temperature = 300.0 - 6.5e-3 * height
        surface_humidity = 0.012  # 12 g/kg
    else:
        # Stable profile - cooler surface, weaker lapse rate
        temperature = 285.0 - 5.0e-3 * height
        surface_humidity = 0.003  # 3 g/kg (dry)
    
    # Humidity profile limited by saturation
    humidity_profile = surface_humidity * jnp.exp(-height / 2000.0)
    qs_profile = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
    humidity = jnp.minimum(humidity_profile, 0.9 * qs_profile)
    
    # Simple wind profile
    u_wind = jnp.full(nlev, 10.0)
    v_wind = jnp.zeros(nlev)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'height': height,
        'u_wind': u_wind,
        'v_wind': v_wind
    }


class TestSaturationFunctions:
    """Test saturation calculations"""
    
    def test_saturation_mixing_ratio_basic(self):
        """Test basic saturation mixing ratio calculation"""
        temp = 300.0  # K
        press = 1e5   # Pa
        
        qs = saturation_mixing_ratio(press, temp)
        
        # Should be reasonable tropical value
        assert 0.01 < qs < 0.05, f"Unrealistic saturation mixing ratio: {qs}"
        assert isinstance(qs, jnp.ndarray), "Should return JAX array"
    
    def test_saturation_temperature_dependence(self):
        """Test that saturation increases with temperature"""
        temps = jnp.array([273.15, 290.0, 300.0, 310.0])
        pressure = 1e5
        
        qs_array = jax.vmap(saturation_mixing_ratio, in_axes=(None, 0))(pressure, temps)
        
        # Should increase monotonically with temperature
        assert jnp.all(jnp.diff(qs_array) > 0), "Saturation should increase with temperature"
    
    def test_saturation_pressure_dependence(self):
        """Test that saturation increases with decreasing pressure"""
        pressures = jnp.array([1e5, 8e4, 6e4, 4e4])
        temperature = 300.0
        
        qs_array = jax.vmap(saturation_mixing_ratio, in_axes=(0, None))(pressures, temperature)
        
        # Should increase with decreasing pressure (at constant temperature)
        assert jnp.all(jnp.diff(qs_array) > 0), "Saturation should increase with decreasing pressure"
    
    def test_saturation_edge_cases(self):
        """Test edge cases for saturation calculation"""
        # Very cold temperature
        qs_cold = saturation_mixing_ratio(1e5, 200.0)
        assert qs_cold > 0, "Should have positive saturation even at low temperatures"
        assert qs_cold < 1e-3, "Should be very small at cold temperatures"
        
        # Very hot temperature
        qs_hot = saturation_mixing_ratio(1e5, 350.0)
        assert qs_hot > 0.1, "Should be large at high temperatures"
        assert qs_hot < 1.0, "Should be physically reasonable"


class TestCloudBase:
    """Test cloud base detection"""
    
    def test_cloud_base_unstable(self):
        """Test cloud base detection in unstable atmosphere"""
        atm = create_realistic_atmosphere(unstable=True)
        config = ConvectionParameters.default()
        
        cloud_base, has_cloud_base = find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        assert has_cloud_base, "Should find cloud base in unstable atmosphere"
        
        # Cloud base should be at reasonable height
        cb_height = atm['height'][cloud_base]
        assert 500 < cb_height < 4000, f"Unrealistic cloud base height: {cb_height}"
        
        # Cloud base should be above surface
        surf_idx = jnp.argmax(atm['pressure'])
        assert cloud_base > surf_idx, "Cloud base should be above surface"
    
    def test_cloud_base_stable(self):
        """Test cloud base detection in stable atmosphere"""
        atm = create_realistic_atmosphere(unstable=False)
        config = ConvectionParameters.default()
        
        cloud_base, has_cloud_base = find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        # May or may not find cloud base in stable atmosphere
        if has_cloud_base:
            cb_height = atm['height'][cloud_base]
            assert cb_height > 0, "Cloud base should be above surface"
    
    def test_cloud_base_jax_compatibility(self):
        """Test that cloud base detection works with JAX transformations"""
        atm = create_realistic_atmosphere(unstable=True)
        config = ConvectionParameters.default()
        
        # Test JIT compilation
        jitted_find_cloud_base = jax.jit(find_cloud_base)
        cloud_base, has_cloud_base = jitted_find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        assert isinstance(cloud_base, jnp.ndarray), "Should return JAX arrays"
        assert isinstance(has_cloud_base, jnp.ndarray), "Should return JAX arrays"


class TestCAPE:
    """Test CAPE/CIN calculations"""
    
    def test_cape_basic(self):
        """Test basic CAPE calculation"""
        atm = create_realistic_atmosphere(unstable=True)
        config = ConvectionParameters.default()
        
        # Find cloud base first
        cloud_base, has_cloud_base = find_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        
        if has_cloud_base:
            cape, cin = calculate_cape_cin(
                atm['temperature'], atm['humidity'], atm['pressure'],
                atm['height'], cloud_base, config
            )
            
            # CAPE should be non-negative
            assert cape >= 0, f"CAPE should be non-negative: {cape}"
            assert cin >= 0, f"CIN should be non-negative: {cin}"
            
            # Should be reasonable values
            assert cape < 10000, f"CAPE too high: {cape}"
            assert cin < 5000, f"CIN too high: {cin}"
    
    def test_cape_stable_vs_unstable(self):
        """Test that CAPE is higher in unstable atmosphere"""
        config = ConvectionParameters.default()
        
        # Unstable atmosphere
        atm_unstable = create_realistic_atmosphere(unstable=True)
        cb_unstable, has_cb_unstable = find_cloud_base(
            atm_unstable['temperature'], atm_unstable['humidity'], 
            atm_unstable['pressure'], config
        )
        
        # Stable atmosphere
        atm_stable = create_realistic_atmosphere(unstable=False)
        cb_stable, has_cb_stable = find_cloud_base(
            atm_stable['temperature'], atm_stable['humidity'],
            atm_stable['pressure'], config
        )
        
        if has_cb_unstable and has_cb_stable:
            cape_unstable, _ = calculate_cape_cin(
                atm_unstable['temperature'], atm_unstable['humidity'],
                atm_unstable['pressure'], atm_unstable['height'],
                cb_unstable, config
            )
            
            cape_stable, _ = calculate_cape_cin(
                atm_stable['temperature'], atm_stable['humidity'],
                atm_stable['pressure'], atm_stable['height'],
                cb_stable, config
            )
            
            # Unstable should have higher CAPE
            assert cape_unstable >= cape_stable, "Unstable atmosphere should have higher CAPE"


class TestJAXCompatibility:
    """Test JAX transformations"""
    
    def test_jit_compilation(self):
        """Test JIT compilation of key functions"""
        atm = create_realistic_atmosphere()
        config = ConvectionParameters.default()
        
        # Test JIT on saturation function
        jitted_saturation = jax.jit(saturation_mixing_ratio)
        qs = jitted_saturation(1e5, 300.0)
        assert qs > 0, "JIT compilation should work"
        
        # Test JIT on cloud base
        jitted_cloud_base = jax.jit(find_cloud_base)
        cb, has_cb = jitted_cloud_base(
            atm['temperature'], atm['humidity'], atm['pressure'], config
        )
        assert isinstance(cb, jnp.ndarray), "Should return JAX array"
    
    def test_vectorization(self):
        """Test vectorization with vmap"""
        temperatures = jnp.array([280.0, 290.0, 300.0, 310.0])
        pressure = 1e5
        
        # Vectorize over temperature
        vmap_saturation = jax.vmap(saturation_mixing_ratio, in_axes=(None, 0))
        qs_vec = vmap_saturation(pressure, temperatures)
        
        assert qs_vec.shape == temperatures.shape, "Should maintain shape"
        assert jnp.all(qs_vec > 0), "All values should be positive"
    
    def test_gradients(self):
        """Test gradient computation"""
        def loss_fn(temp):
            return saturation_mixing_ratio(1e5, temp)
        
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(300.0)
        
        assert gradient > 0, "Gradient should be positive (qs increases with T)"
        assert jnp.isfinite(gradient), "Gradient should be finite"


class TestTracerTransport:
    """Test tracer transport functionality"""
    
    def test_tracer_initialization(self):
        """Test tracer initialization"""
        nlev = 20
        
        # Basic tracers only
        tracers_basic, indices_basic = initialize_tracers(nlev, include_chemistry=False)
        assert tracers_basic.shape == (nlev, 3), "Should have 3 basic tracers"
        
        # With chemistry
        tracers_chem, indices_chem = initialize_tracers(nlev, include_chemistry=True)
        assert tracers_chem.shape[1] > 3, "Should have additional chemical tracers"
        
        # Check indices
        assert indices_basic.iqv == 0, "Water vapor should be index 0"
        assert indices_basic.iqc == 1, "Cloud water should be index 1"
        assert indices_basic.iqi == 2, "Cloud ice should be index 2"
        assert indices_basic.iqt == 3, "Additional tracers should start at index 3"
    
    def test_tracer_indices(self):
        """Test tracer indices structure"""
        indices = TracerIndices()
        
        assert hasattr(indices, 'iqv'), "Should have water vapor index"
        assert hasattr(indices, 'iqc'), "Should have cloud water index"
        assert hasattr(indices, 'iqi'), "Should have cloud ice index"
        assert hasattr(indices, 'iqt'), "Should have additional tracer start index"
        
        # Check ordering
        assert indices.iqv < indices.iqc < indices.iqi < indices.iqt, "Indices should be ordered"


class TestConfiguration:
    """Test configuration parameters"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ConvectionParameters.default()
        
        # Check that all required parameters are present
        assert hasattr(config, 'tau'), "Should have CAPE timescale"
        assert hasattr(config, 'entrpen'), "Should have entrainment parameters"
        assert hasattr(config, 'cmfcmax'), "Should have mass flux limits"
        
        # Check reasonable values
        assert float(config.tau) > 0, "CAPE timescale should be positive"
        assert float(config.cmfcmax) > float(config.cmfcmin), "Max mass flux should exceed min"
        assert 0 < float(config.entrpen) < 1, "Entrainment rate should be reasonable"
    
    def test_config_modification(self):
        """Test configuration modification"""
        # Create configs with different tau values
        config1 = ConvectionParameters.default(tau=3600.0)
        config2 = ConvectionParameters.default(tau=7200.0)
        
        assert float(config1.tau) != float(config2.tau), "Should allow parameter modification"
        # Note: when creating parameters directly, need to set all fields


class TestPhysicalConsistency:
    """Test physical consistency of calculations"""
    
    def test_humidity_consistency(self):
        """Test that humidity profiles are physically consistent"""
        atm = create_realistic_atmosphere()
        
        # Check humidity range
        assert jnp.all(atm['humidity'] >= 0), "Humidity should be non-negative"
        assert jnp.all(atm['humidity'] < 0.1), "Humidity should be reasonable (< 100 g/kg)"
        
        # Check relative humidity
        qs_profile = jax.vmap(saturation_mixing_ratio)(atm['pressure'], atm['temperature'])
        rel_humidity = atm['humidity'] / qs_profile
        assert jnp.all(rel_humidity <= 1.0), "Should not exceed saturation"
    
    def test_temperature_consistency(self):
        """Test that temperature profiles are physically consistent"""
        atm = create_realistic_atmosphere()
        
        # Check temperature range
        assert jnp.all(atm['temperature'] > 150), "Temperature should be reasonable (> 150K)"
        assert jnp.all(atm['temperature'] < 350), "Temperature should be reasonable (< 350K)"
        
        # Check lapse rate
        temp_gradient = jnp.mean(jnp.diff(atm['temperature']) / jnp.diff(atm['height']))
        assert -0.015 < temp_gradient < 0, "Lapse rate should be reasonable"
    
    def test_pressure_consistency(self):
        """Test that pressure profiles are physically consistent"""
        atm = create_realistic_atmosphere()
        
        # Pressure should decrease with height
        assert jnp.all(jnp.diff(atm['pressure']) < 0), "Pressure should decrease with height"
        
        # Reasonable pressure range
        assert jnp.min(atm['pressure']) > 1e4, "Minimum pressure should be reasonable"
        assert jnp.max(atm['pressure']) <= 1.1e5, "Maximum pressure should be reasonable"


# Pytest fixtures for common test data
@pytest.fixture
def unstable_atmosphere():
    """Fixture providing unstable atmospheric profile"""
    return create_realistic_atmosphere(unstable=True)


@pytest.fixture  
def stable_atmosphere():
    """Fixture providing stable atmospheric profile"""
    return create_realistic_atmosphere(unstable=False)


@pytest.fixture
def default_config():
    """Fixture providing default convection configuration"""
    return ConvectionParameters.default()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])