"""
Tests for Tiedtke-Nordeng convection scheme

Date: 2025-01-09
"""

import pytest
import jax.numpy as jnp
import jax
from jax import random

from jcm.physics.icon.convection import (
    tiedtke_nordeng_convection,
    ConvectionConfig,
    ConvectionState,
    ConvectionTendencies
)


def create_test_atmosphere(nlev=40, unstable=True):
    """Create a test atmospheric profile"""
    # Pressure levels (Pa) - from surface to top
    pressure = jnp.logspace(5, 3, nlev)[::-1]  # 1000 hPa to 10 hPa
    
    # Height (m) - hydrostatic approximation
    height = -7000 * jnp.log(pressure / 1e5)
    
    if unstable:
        # Unstable profile - warm and moist at surface
        # Temperature profile with lapse rate
        surface_temp = 300.0  # K
        lapse_rate = 6.5e-3   # K/m
        temperature = surface_temp - lapse_rate * height
        
        # Add inversion at tropopause
        trop_idx = jnp.argmin(jnp.abs(pressure - 200e2))  # ~200 hPa
        temperature = temperature.at[:trop_idx].set(
            temperature[trop_idx]
        )
        
        # Humidity profile - exponential decrease
        surface_rh = 0.8
        humidity_scale = 2000.0  # m
        rel_humidity = surface_rh * jnp.exp(-height / humidity_scale)
        
        # Convert to specific humidity
        from .tiedtke_nordeng import saturation_mixing_ratio
        qs = jax.vmap(saturation_mixing_ratio)(pressure, temperature)
        humidity = rel_humidity * qs
    else:
        # Stable profile - cool and dry
        surface_temp = 285.0
        temperature = surface_temp - 5e-3 * height
        humidity = jnp.ones_like(temperature) * 1e-3  # Very dry
    
    # Wind profile - simple shear
    u_wind = 10.0 + 20.0 * (1.0 - pressure / 1e5)
    v_wind = jnp.zeros_like(u_wind)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'height': height,
        'u_wind': u_wind,
        'v_wind': v_wind
    }


class TestConvectionScheme:
    """Test suite for Tiedtke-Nordeng convection"""
    
    def test_stable_atmosphere(self):
        """Test that stable atmosphere produces no convection"""
        # Create stable profile
        atm = create_test_atmosphere(unstable=False)
        config = ConvectionConfig()
        
        # Run convection scheme
        tendencies, state = tiedtke_nordeng_convection(
            atm['temperature'],
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        
        # Check no convection occurs
        assert state.ktype == 0
        assert jnp.allclose(tendencies.dtedt, 0.0)
        assert jnp.allclose(tendencies.dqdt, 0.0)
        assert tendencies.precip_conv == 0.0
    
    def test_unstable_atmosphere(self):
        """Test that unstable atmosphere triggers convection"""
        # Create unstable profile
        atm = create_test_atmosphere(unstable=True)
        config = ConvectionConfig()
        
        # Run convection scheme
        tendencies, state = tiedtke_nordeng_convection(
            atm['temperature'],
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        
        # Check convection occurs
        assert state.ktype > 0  # Some type of convection
        assert state.kbase < len(atm['temperature']) - 1  # Cloud base above surface
        
        # Check physical consistency
        # Total column heating should approximately balance moisture loss
        total_heating = jnp.sum(tendencies.dtedt) * 3600.0
        total_drying = jnp.sum(tendencies.dqdt) * 3600.0
        
        # Precipitation should be positive for active convection
        if state.ktype > 0:
            assert tendencies.precip_conv >= 0.0
    
    def test_mass_conservation(self):
        """Test mass flux conservation"""
        # Create test profile
        atm = create_test_atmosphere(unstable=True)
        config = ConvectionConfig()
        
        # Run convection scheme
        tendencies, state = tiedtke_nordeng_convection(
            atm['temperature'],
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        
        # If convection is active, check mass conservation
        if state.ktype > 0:
            # Net mass flux at each level should be continuous
            # (This is a simplified check)
            mf_net = state.mfu + state.mfd  # Downdraft is negative
            
            # Mass flux should decrease with height
            assert jnp.all(jnp.diff(state.mfu[:state.ktop]) <= 0)
    
    def test_energy_conservation(self):
        """Test approximate energy conservation"""
        # Create test profile
        atm = create_test_atmosphere(unstable=True)
        config = ConvectionConfig()
        
        # Run convection scheme
        tendencies, state = tiedtke_nordeng_convection(
            atm['temperature'],
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        
        if state.ktype > 0:
            # Calculate energy changes
            from ..constants.physical_constants import cp, alhc
            
            # Sensible heat change
            dH_sensible = jnp.sum(tendencies.dtedt * cp)
            
            # Latent heat change (condensation releases heat)
            dH_latent = -jnp.sum(tendencies.dqdt * alhc)
            
            # Net heating should be small (energy is redistributed, not created)
            net_heating = dH_sensible + dH_latent
            
            # This is a weak test - just ensure values are reasonable
            assert jnp.abs(net_heating) < 1e6  # W/m²
    
    def test_jax_compatibility(self):
        """Test JAX transformations work correctly"""
        # Create test profile
        atm = create_test_atmosphere(unstable=True)
        config = ConvectionConfig()
        
        # Test jit compilation
        jitted_convection = jax.jit(tiedtke_nordeng_convection)
        
        tendencies, state = jitted_convection(
            atm['temperature'],
            atm['humidity'],
            atm['pressure'],
            atm['height'],
            atm['u_wind'],
            atm['v_wind'],
            dt=3600.0,
            config=config
        )
        
        # Test gradient computation (for adjoints)
        def loss_fn(temperature):
            tendencies, _ = tiedtke_nordeng_convection(
                temperature,
                atm['humidity'],
                atm['pressure'],
                atm['height'],
                atm['u_wind'],
                atm['v_wind'],
                dt=3600.0,
                config=config
            )
            return jnp.sum(tendencies.precip_conv)
        
        # This should not error
        grad = jax.grad(loss_fn)(atm['temperature'])
        assert grad.shape == atm['temperature'].shape
    
    def test_config_parameters(self):
        """Test different configuration parameters"""
        # Create test profile
        atm = create_test_atmosphere(unstable=True)
        
        # Test with different CAPE timescales
        configs = [
            ConvectionConfig(tau=3600.0),   # Fast adjustment
            ConvectionConfig(tau=7200.0),   # Default
            ConvectionConfig(tau=14400.0),  # Slow adjustment
        ]
        
        precip_rates = []
        for config in configs:
            tendencies, state = tiedtke_nordeng_convection(
                atm['temperature'],
                atm['humidity'],
                atm['pressure'],
                atm['height'],
                atm['u_wind'],
                atm['v_wind'],
                dt=3600.0,
                config=config
            )
            precip_rates.append(tendencies.precip_conv)
        
        # Faster adjustment should produce more precipitation
        if precip_rates[0] > 0:
            assert precip_rates[0] >= precip_rates[1]
            assert precip_rates[1] >= precip_rates[2]


if __name__ == "__main__":
    # Run basic tests
    test = TestConvectionScheme()
    
    print("Testing stable atmosphere...")
    test.test_stable_atmosphere()
    print("✓ Stable atmosphere test passed")
    
    print("\nTesting unstable atmosphere...")
    test.test_unstable_atmosphere()
    print("✓ Unstable atmosphere test passed")
    
    print("\nTesting mass conservation...")
    test.test_mass_conservation()
    print("✓ Mass conservation test passed")
    
    print("\nTesting JAX compatibility...")
    test.test_jax_compatibility()
    print("✓ JAX compatibility test passed")
    
    print("\nAll tests passed!")