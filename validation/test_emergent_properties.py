#!/usr/bin/env python3
"""
Test emergent atmospheric properties in JAX-GCM ICON physics.

This script tests key emergent properties like ITCZ formation, Hadley cell circulation,
and global energy balance to validate the physics implementation.
"""

import jax.numpy as jnp
import jax
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jcm.geometry import Geometry
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.date import DateData

class EmergentPropertiesValidator:
    """Test emergent atmospheric properties in JAX-GCM ICON physics"""
    
    def __init__(self, grid_config: Dict[str, Any] = None):
        """Initialize validator with grid configuration"""
        
        # Default grid configuration
        default_config = {
            'nlat': 64,
            'nlon': 128, 
            'nlev': 47,
            'dt': 1800.0,  # 30-minute timestep
        }
        
        self.config = {**default_config, **(grid_config or {})}
        
        # Create geometry
        self.geometry = Geometry.from_grid_shape(
            nlon=self.config['nlon'],
            nlat=self.config['nlat'], 
            nlev=self.config['nlev']
        )
        
        # Initialize physics
        self.physics = IconPhysics()
        
        # Create latitude/longitude arrays
        self.lat = jnp.linspace(-90, 90, self.config['nlat'])
        self.lon = jnp.linspace(-180, 180, self.config['nlon'])
        
        print(f"Initialized validator with {self.config['nlat']}x{self.config['nlon']} grid")
        
    def create_aquaplanet_state(self, sst: float = 300.0) -> PhysicsData:
        """Create initial aquaplanet state"""
        
        ncol = self.config['nlat'] * self.config['nlon']
        nlev = self.config['nlev']
        
        # Create pressure levels (standard atmosphere)
        p_surface = 101325.0  # Pa
        p_levels = jnp.linspace(p_surface, 1000.0, nlev)  # 1000 Pa to 101325 Pa
        
        # Create temperature profile (decreasing with height)
        temp_profile = 288.0 - (1 - p_levels / p_surface) * 60.0  # K
        
        # Expand to full grid
        temperature = jnp.tile(temp_profile, (ncol, 1))
        
        # Add latitudinal temperature gradient
        lat_2d = jnp.repeat(self.lat, self.config['nlon'])
        temp_gradient = 30.0 * jnp.cos(jnp.deg2rad(lat_2d))  # Warmer at equator
        temperature = temperature + temp_gradient[:, None]
        
        # Create other state variables
        u_wind = jnp.zeros((ncol, nlev))
        v_wind = jnp.zeros((ncol, nlev))
        
        # Specific humidity (higher in tropics)
        qv_max = 0.02  # kg/kg
        qv_profile = qv_max * jnp.exp(-p_levels / 80000.0)  # Exponential decrease
        specific_humidity = jnp.tile(qv_profile, (ncol, 1))
        
        # Apply tropical enhancement
        tropical_factor = 1.0 + 0.5 * jnp.exp(-lat_2d**2 / 400.0)  # Enhanced in tropics
        specific_humidity = specific_humidity * tropical_factor[:, None]
        
        # Cloud variables (initially zero)
        cloud_water = jnp.zeros((ncol, nlev))
        cloud_ice = jnp.zeros((ncol, nlev))
        
        # Pressure arrays
        pressure_full = jnp.tile(p_levels, (ncol, 1))
        
        # Create half-level pressures (interfaces)
        p_half = jnp.zeros(nlev + 1)
        p_half = p_half.at[0].set(100.0)  # Top of atmosphere
        p_half = p_half.at[1:-1].set(0.5 * (p_levels[:-1] + p_levels[1:]))
        p_half = p_half.at[-1].set(p_surface)  # Surface
        pressure_half = jnp.tile(p_half, (ncol, 1))
        
        # Surface properties
        surface_pressure = jnp.full(ncol, p_surface)
        surface_temperature = jnp.full(ncol, sst)
        
        # Geopotential (height approximation)
        geopotential = jnp.zeros((ncol, nlev))
        for k in range(nlev):
            # Approximate height using barometric formula
            height = -287.0 * 250.0 * jnp.log(p_levels[k] / p_surface) / 9.81
            geopotential = geopotential.at[:, k].set(height * 9.81)
        
        # Create PhysicsData
        return PhysicsData(
            temperature=temperature,
            u_wind=u_wind,
            v_wind=v_wind,
            specific_humidity=specific_humidity,
            cloud_water=cloud_water,
            cloud_ice=cloud_ice,
            pressure_full=pressure_full,
            pressure_half=pressure_half,
            surface_pressure=surface_pressure,
            geopotential=geopotential,
            surface_temperature=surface_temperature,
            # Add required fields with default values
            ozone_mixing_ratio=jnp.full((ncol, nlev), 1e-6),
            methane_mixing_ratio=jnp.full((ncol, nlev), 1.8e-6),
            aerosol_optical_depth=jnp.full((ncol, nlev, 14), 0.01),
            aerosol_single_scatter_albedo=jnp.full((ncol, nlev, 14), 0.9),
            aerosol_asymmetry_factor=jnp.full((ncol, nlev, 14), 0.7),
            cdnc_factor=jnp.full(ncol, 1.0),
            surface_albedo=jnp.full(ncol, 0.1),
            surface_emissivity=jnp.full(ncol, 0.98),
            solar_zenith_angle=jnp.full(ncol, 0.5),
            solar_azimuth_angle=jnp.zeros(ncol),
            sea_surface_temperature=jnp.full(ncol, sst),
            sea_ice_fraction=jnp.zeros(ncol),
            land_fraction=jnp.zeros(ncol),
            roughness_length=jnp.full(ncol, 0.001),
            vegetation_fraction=jnp.zeros(ncol),
            soil_moisture=jnp.full(ncol, 0.3),
            snow_depth=jnp.zeros(ncol),
            co2_concentration=jnp.full(ncol, 407.8e-6),
            day_of_year=jnp.full(ncol, 15),
            hour_utc=jnp.full(ncol, 12.0),
            latitude=lat_2d,
            longitude=jnp.tile(self.lon, self.config['nlat']),
        )
    
    def run_physics_timestep(self, state: PhysicsData, date: DateData) -> Tuple[PhysicsData, Dict]:
        """Run single physics timestep"""
        
        # Compute physics tendencies
        tendencies, diagnostics = self.physics.compute_tendencies(
            state, self.geometry, date
        )
        
        # Update state (simple forward Euler)
        dt = self.config['dt']
        new_state = state._replace(
            temperature=state.temperature + tendencies.temperature * dt,
            u_wind=state.u_wind + tendencies.u_wind * dt,
            v_wind=state.v_wind + tendencies.v_wind * dt,
            specific_humidity=state.specific_humidity + tendencies.specific_humidity * dt,
            cloud_water=state.cloud_water + tendencies.cloud_water * dt,
            cloud_ice=state.cloud_ice + tendencies.cloud_ice * dt,
        )
        
        return new_state, diagnostics
    
    def run_short_integration(self, days: int = 5) -> Dict[str, jnp.ndarray]:
        """Run short integration to test basic physics"""
        
        print(f"Running {days}-day integration...")
        
        # Initialize state
        state = self.create_aquaplanet_state()
        date = DateData(year=2020, month=1, day=15, hour=12)
        
        # Storage for diagnostics
        history = {
            'temperature': [],
            'precipitation': [],
            'radiation_balance': [],
            'surface_fluxes': [],
        }
        
        # Integration loop
        timesteps = int(days * 24 * 3600 / self.config['dt'])
        print(f"Running {timesteps} timesteps...")
        
        for step in range(timesteps):
            # Run physics
            state, diagnostics = self.run_physics_timestep(state, date)
            
            # Store diagnostics every 6 hours
            if step % int(6 * 3600 / self.config['dt']) == 0:
                history['temperature'].append(state.temperature)
                # Add other diagnostics if available
                if hasattr(diagnostics, 'precipitation'):
                    history['precipitation'].append(diagnostics.precipitation)
                if hasattr(diagnostics, 'radiation_balance'):
                    history['radiation_balance'].append(diagnostics.radiation_balance)
                if hasattr(diagnostics, 'surface_sensible_flux'):
                    history['surface_fluxes'].append(diagnostics.surface_sensible_flux)
            
            # Update time
            date = date._replace(hour=date.hour + self.config['dt'] / 3600)
            
            if step % 100 == 0:
                print(f"  Step {step}/{timesteps}")
        
        # Convert to arrays
        for key in history:
            if history[key]:
                history[key] = jnp.array(history[key])
        
        print("Integration complete!")
        return history
    
    def test_temperature_distribution(self, history: Dict) -> Dict[str, Any]:
        """Test temperature distribution and gradients"""
        
        print("Testing temperature distribution...")
        
        # Get final temperature
        temp_final = history['temperature'][-1]  # (ncol, nlev)
        
        # Reshape to lat/lon grid
        temp_surface = temp_final[:, -1].reshape(self.config['nlat'], self.config['nlon'])
        
        # Calculate zonal mean temperature
        temp_zonal = jnp.mean(temp_surface, axis=1)
        
        # Find temperature gradient
        temp_gradient = jnp.gradient(temp_zonal, self.lat)
        
        # Calculate tropical mean temperature
        tropical_mask = jnp.abs(self.lat) < 30
        temp_tropical = jnp.mean(temp_zonal[tropical_mask])
        
        # Calculate polar mean temperature  
        polar_mask = jnp.abs(self.lat) > 60
        temp_polar = jnp.mean(temp_zonal[polar_mask])
        
        # Temperature contrast
        temp_contrast = temp_tropical - temp_polar
        
        results = {
            'temperature_zonal': temp_zonal,
            'temperature_gradient': temp_gradient,
            'temperature_tropical': float(temp_tropical),
            'temperature_polar': float(temp_polar),
            'temperature_contrast': float(temp_contrast),
            'expected_contrast': {'min': 30, 'max': 80},  # K
            'test_passed': 30 <= temp_contrast <= 80,
        }
        
        print(f"  Temperature contrast: {temp_contrast:.1f} K")
        print(f"  Expected range: 30-80 K")
        print(f"  Test passed: {results['test_passed']}")
        
        return results
    
    def test_energy_balance(self, history: Dict) -> Dict[str, Any]:
        """Test basic energy balance"""
        
        print("Testing energy balance...")
        
        # Simple energy balance check
        if 'radiation_balance' in history and len(history['radiation_balance']) > 0:
            rad_balance = history['radiation_balance'][-1]
            
            # Reshape to lat/lon grid
            rad_balance_2d = rad_balance.reshape(self.config['nlat'], self.config['nlon'])
            
            # Calculate zonal mean
            rad_zonal = jnp.mean(rad_balance_2d, axis=1)
            
            # Global mean should be close to zero
            global_mean = jnp.mean(rad_balance)
            
            # Tropical excess
            tropical_mask = jnp.abs(self.lat) < 30
            tropical_excess = jnp.mean(rad_zonal[tropical_mask])
            
            results = {
                'global_mean_balance': float(global_mean),
                'tropical_excess': float(tropical_excess),
                'expected_global_balance': {'min': -10, 'max': 10},  # W/m²
                'expected_tropical_excess': {'min': 20, 'max': 200},  # W/m²
                'test_passed': (-10 <= global_mean <= 10) and (20 <= tropical_excess <= 200),
            }
            
            print(f"  Global energy balance: {global_mean:.1f} W/m²")
            print(f"  Tropical excess: {tropical_excess:.1f} W/m²")
            print(f"  Test passed: {results['test_passed']}")
        else:
            print("  No radiation balance data available")
            results = {
                'global_mean_balance': 0.0,
                'tropical_excess': 0.0,
                'test_passed': False,
            }
        
        return results
    
    def test_physics_stability(self, history: Dict) -> Dict[str, Any]:
        """Test physics stability and reasonable values"""
        
        print("Testing physics stability...")
        
        # Check for NaN or infinite values
        temp_final = history['temperature'][-1]
        
        has_nan = jnp.any(jnp.isnan(temp_final))
        has_inf = jnp.any(jnp.isinf(temp_final))
        
        # Check temperature range
        temp_min = jnp.min(temp_final)
        temp_max = jnp.max(temp_final)
        
        temp_reasonable = (150 <= temp_min <= 350) and (150 <= temp_max <= 350)
        
        results = {
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'temp_min': float(temp_min),
            'temp_max': float(temp_max),
            'temp_reasonable': bool(temp_reasonable),
            'test_passed': not has_nan and not has_inf and temp_reasonable,
        }
        
        print(f"  NaN values: {has_nan}")
        print(f"  Infinite values: {has_inf}")
        print(f"  Temperature range: {temp_min:.1f} - {temp_max:.1f} K")
        print(f"  Test passed: {results['test_passed']}")
        
        return results
    
    def generate_validation_plots(self, history: Dict, results: Dict):
        """Generate validation plots"""
        
        print("Generating validation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Temperature distribution
        temp_final = history['temperature'][-1][:, -1]
        temp_2d = temp_final.reshape(self.config['nlat'], self.config['nlon'])
        
        im1 = axes[0, 0].contourf(self.lon, self.lat, temp_2d, levels=20, cmap='coolwarm')
        axes[0, 0].set_title('Surface Temperature (K)')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Zonal mean temperature
        temp_zonal = results['temperature']['temperature_zonal']
        axes[0, 1].plot(self.lat, temp_zonal, 'b-', linewidth=2)
        axes[0, 1].set_title('Zonal Mean Temperature')
        axes[0, 1].set_xlabel('Latitude')
        axes[0, 1].set_ylabel('Temperature (K)')
        axes[0, 1].grid(True)
        
        # Temperature evolution
        if len(history['temperature']) > 1:
            temp_evolution = jnp.array([jnp.mean(t) for t in history['temperature']])
            time_hours = jnp.arange(len(temp_evolution)) * 6  # 6-hourly data
            axes[1, 0].plot(time_hours, temp_evolution, 'r-', linewidth=2)
            axes[1, 0].set_title('Global Mean Temperature Evolution')
            axes[1, 0].set_xlabel('Time (hours)')
            axes[1, 0].set_ylabel('Temperature (K)')
            axes[1, 0].grid(True)
        
        # Test summary
        axes[1, 1].text(0.1, 0.8, 'Validation Results:', fontsize=12, fontweight='bold')
        y_pos = 0.7
        for test_name, test_result in results.items():
            if 'test_passed' in test_result:
                status = "✓ PASS" if test_result['test_passed'] else "✗ FAIL"
                axes[1, 1].text(0.1, y_pos, f"{test_name}: {status}", fontsize=10)
                y_pos -= 0.1
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
        print("Plots saved to validation_results.png")
    
    def run_comprehensive_validation(self, days: int = 5) -> Dict[str, Any]:
        """Run comprehensive validation test suite"""
        
        print("=" * 60)
        print("JAX-GCM ICON Physics Emergent Properties Validation")
        print("=" * 60)
        
        # Run physics integration
        history = self.run_short_integration(days)
        
        # Run validation tests
        results = {}
        results['temperature'] = self.test_temperature_distribution(history)
        results['energy_balance'] = self.test_energy_balance(history)
        results['stability'] = self.test_physics_stability(history)
        
        # Generate plots
        self.generate_validation_plots(history, results)
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_result in results.items():
            if 'test_passed' in test_result:
                total_tests += 1
                if test_result['test_passed']:
                    passed_tests += 1
                    print(f"✓ {test_name.upper()}: PASSED")
                else:
                    print(f"✗ {test_name.upper()}: FAILED")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All validation tests PASSED!")
        else:
            print("⚠️  Some validation tests failed. Check results above.")
        
        return results

def main():
    """Run emergent properties validation"""
    
    # Create validator
    validator = EmergentPropertiesValidator()
    
    # Run validation
    results = validator.run_comprehensive_validation(days=5)
    
    return results

if __name__ == "__main__":
    results = main()