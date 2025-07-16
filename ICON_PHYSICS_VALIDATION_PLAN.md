# ICON Physics Validation Plan

**Date**: January 2025  
**Purpose**: Validate JAX-GCM ICON physics implementation against original ICON code  
**Goal**: Establish scientific trust and accuracy for production use

## Overview

This validation plan provides two complementary approaches to verify the JAX-GCM ICON physics implementation:

1. **Direct Comparison**: Reference test cases from original ICON code
2. **Emergent Properties**: Test key atmospheric phenomena and climate features

## Approach 1: Direct Comparison with ICON Reference Data

### **Test Case Configuration for HPC ICON Runs**

#### **1. Single Column Tests (Highest Priority)**

**Configuration:**
- **Model**: ICON-A (atmosphere-only)
- **Grid**: Single column (lat/lon specified)
- **Vertical**: 47 levels (standard ICON-A configuration)
- **Physics**: ECHAM physics package
- **Time**: 10-day runs with hourly output

**Test Cases:**
```bash
# Test Case 1: Tropical Convection
# Location: 0°N, 0°E (Equatorial Pacific)
# Season: January (Northern winter)
# Purpose: Test convection scheme, cloud physics, radiation

# Test Case 2: Mid-latitude Storm
# Location: 45°N, 0°E (Western Europe)  
# Season: January (Northern winter)
# Purpose: Test vertical diffusion, surface physics, frontal systems

# Test Case 3: Polar Conditions
# Location: 80°N, 0°E (Arctic)
# Season: January (Polar night)
# Purpose: Test radiation under extreme conditions, surface physics

# Test Case 4: Subtropical Anticyclone
# Location: 30°N, 30°W (Azores High region)
# Season: July (Northern summer)
# Purpose: Test subsidence, clear-sky radiation, surface fluxes
```

**Required Output Variables:**
```fortran
! Atmospheric State
temperature(:,:)     ! Temperature [K]
u_wind(:,:)         ! Zonal wind [m/s] 
v_wind(:,:)         ! Meridional wind [m/s]
pressure(:,:)       ! Pressure [Pa]
specific_humidity(:,:)  ! Water vapor [kg/kg]
cloud_water(:,:)    ! Cloud liquid water [kg/kg]
cloud_ice(:,:)      ! Cloud ice [kg/kg]

! Physics Tendencies (per timestep)
temp_tendency_rad(:,:)    ! Radiation heating [K/s]
temp_tendency_conv(:,:)   ! Convective heating [K/s]
temp_tendency_vdiff(:,:)  ! Vertical diffusion [K/s]
temp_tendency_total(:,:)  ! Total temperature tendency [K/s]

u_tendency_conv(:,:)      ! Convective momentum [m/s²]
u_tendency_vdiff(:,:)     ! Turbulent momentum [m/s²]
u_tendency_total(:,:)     ! Total u tendency [m/s²]

qv_tendency_conv(:,:)     ! Convective moisture [kg/kg/s]
qv_tendency_vdiff(:,:)    ! Turbulent moisture [kg/kg/s]
qv_tendency_total(:,:)    ! Total moisture tendency [kg/kg/s]

! Radiation Diagnostics
sw_flux_up(:,:)          ! Shortwave upward flux [W/m²]
sw_flux_down(:,:)        ! Shortwave downward flux [W/m²]
lw_flux_up(:,:)          ! Longwave upward flux [W/m²]
lw_flux_down(:,:)        ! Longwave downward flux [W/m²]
heating_rate_sw(:,:)     ! SW heating rate [K/s]
heating_rate_lw(:,:)     ! LW heating rate [K/s]

! Surface Fluxes
surface_sensible_flux    ! Sensible heat flux [W/m²]
surface_latent_flux      ! Latent heat flux [W/m²]
surface_momentum_flux_u  ! Surface u-momentum flux [N/m²]
surface_momentum_flux_v  ! Surface v-momentum flux [N/m²]

! Convection Diagnostics
convective_precip        ! Convective precipitation [kg/m²/s]
cloud_base_height        ! Cloud base height [m]
cloud_top_height         ! Cloud top height [m]
cape                     ! Convective available potential energy [J/kg]
cin                      ! Convective inhibition [J/kg]

! Vertical Diffusion Diagnostics
tke(:,:)                 ! Turbulent kinetic energy [m²/s²]
mixing_length(:,:)       ! Mixing length [m]
richardson_number(:,:)   ! Richardson number [-]
boundary_layer_height    ! PBL height [m]
```

**ICON Namelist Configuration:**
```fortran
&run_nml
 num_lev        = 47           ! 47 vertical levels
 dtime          = 600          ! 10-minute timestep
 ndays          = 10           ! 10-day simulation
 output_interval = 3600        ! Hourly output
/

&echam_phy_nml
 echam_phy_tc%dt_rad = 3600    ! Radiation every hour
 echam_phy_tc%dt_conv = 600    ! Convection every timestep
 echam_phy_tc%dt_vdf = 600     ! Vertical diffusion every timestep
 echam_phy_tc%dt_sfc = 600     ! Surface physics every timestep
/

&radiation_nml
 irad_o3        = 3            ! Ozone climatology
 irad_aero      = 2            ! Kinne aerosol climatology
 irad_co2       = 2            ! CO2 concentration
/
```

#### **2. Idealized Global Tests (Medium Priority)**

**Configuration:**
- **Model**: ICON-A global
- **Grid**: R2B4 (160 km resolution)
- **Vertical**: 47 levels
- **Physics**: ECHAM physics
- **Time**: 30-day runs

**Test Cases:**
```bash
# Test Case 5: Aquaplanet
# Configuration: Uniform SST (300K), no topography
# Purpose: Test global circulation, ITCZ formation, Hadley cells

# Test Case 6: Held-Suarez
# Configuration: Idealized forcing, no surface physics
# Purpose: Test fundamental atmospheric dynamics

# Test Case 7: Radiative-Convective Equilibrium
# Configuration: Fixed SST, no rotation, small domain
# Purpose: Test convection-radiation interaction
```

### **Reference Data Processing**

Create standardized output format for comparison:
```python
# Expected file structure
reference_data/
├── single_column/
│   ├── tropical_convection/
│   │   ├── initial_conditions.nc
│   │   ├── boundary_conditions.nc
│   │   ├── hourly_output.nc
│   │   └── physics_tendencies.nc
│   ├── midlatitude_storm/
│   ├── polar_conditions/
│   └── subtropical_anticyclone/
├── global_idealized/
│   ├── aquaplanet/
│   ├── held_suarez/
│   └── radiative_convective_equilibrium/
└── metadata/
    ├── grid_coordinates.nc
    ├── vertical_levels.nc
    └── physical_constants.nc
```

## Approach 2: Emergent Properties Testing

### **Key Atmospheric Phenomena to Test**

#### **1. Tropical Circulation (Highest Priority)**

**Inter-Tropical Convergence Zone (ITCZ)**
```python
def test_itcz_formation():
    """Test ITCZ formation and characteristics"""
    # Expected properties:
    # - Latitude: 5°S to 10°N depending on season
    # - Precipitation: >8 mm/day in core
    # - Width: 200-800 km
    # - Seasonal migration: ~10° latitude
    
    # Test metrics:
    # 1. Precipitation centroid latitude
    # 2. Precipitation intensity maximum
    # 3. ITCZ width (half-width at half-maximum)
    # 4. Seasonal cycle amplitude
    
    # Validation data: GPCP, CMAP, TRMM observations
```

**Hadley Cell Circulation**
```python
def test_hadley_cell():
    """Test Hadley cell strength and structure"""
    # Expected properties:
    # - Ascending branch: 0-10°N/S
    # - Descending branch: 20-30°N/S  
    # - Upper-level poleward flow
    # - Surface trade winds
    
    # Test metrics:
    # 1. Meridional overturning streamfunction
    # 2. Trade wind strength and direction
    # 3. Subtropical high pressure systems
    # 4. Cross-equatorial flow
```

**Walker Circulation**
```python
def test_walker_circulation():
    """Test Pacific Walker circulation"""
    # Expected properties:
    # - Ascending over Maritime Continent
    # - Descending over Eastern Pacific
    # - Upper-level westerlies
    # - Surface easterlies (trade winds)
    
    # Test metrics:
    # 1. Zonal overturning circulation
    # 2. Sea level pressure gradient
    # 3. Precipitation dipole pattern
    # 4. Upper-level wind patterns
```

#### **2. Mid-latitude Dynamics**

**Jet Stream Structure**
```python
def test_jet_streams():
    """Test jet stream formation and characteristics"""
    # Expected properties:
    # - Location: 30-60°N/S
    # - Strength: 20-40 m/s in winter
    # - Seasonal variation
    # - Meridional meandering
    
    # Test metrics:
    # 1. Jet latitude and strength
    # 2. Jet width and vertical structure
    # 3. Seasonal cycle
    # 4. Eddy momentum flux convergence
```

**Storm Track Activity**
```python
def test_storm_tracks():
    """Test extratropical storm track activity"""
    # Expected properties:
    # - North Atlantic: 45-65°N
    # - North Pacific: 40-60°N
    # - Southern Ocean: 50-70°S
    # - Seasonal migration
    
    # Test metrics:
    # 1. Eddy kinetic energy
    # 2. Cyclone frequency and intensity
    # 3. Temperature variance
    # 4. Precipitation patterns
```

#### **3. Energy Balance and Climate**

**Global Energy Budget**
```python
def test_global_energy_budget():
    """Test global energy balance"""
    # Expected properties:
    # - TOA net radiation: ~0 W/m² globally
    # - Tropical excess: ~100 W/m²
    # - Polar deficit: ~100 W/m²
    # - Seasonal variations
    
    # Test metrics:
    # 1. TOA radiation balance
    # 2. Surface energy balance
    # 3. Atmospheric energy transport
    # 4. Meridional energy transport
```

**Seasonal Cycle**
```python
def test_seasonal_cycle():
    """Test seasonal climate variations"""
    # Expected properties:
    # - Temperature amplitude: max at continental interiors
    # - Monsoon systems: seasonal wind reversal
    # - ITCZ migration: follows maximum heating
    # - Polar day/night cycles
    
    # Test metrics:
    # 1. Temperature seasonal amplitude
    # 2. Precipitation seasonal cycle
    # 3. Monsoon onset/retreat timing
    # 4. Snow/ice seasonal cycle
```

#### **4. Hydrological Cycle**

**Global Precipitation Patterns**
```python
def test_precipitation_patterns():
    """Test global precipitation distribution"""
    # Expected properties:
    # - Tropical maximum: ITCZ, monsoons
    # - Subtropical minima: deserts
    # - Mid-latitude maxima: storm tracks
    # - Orographic effects
    
    # Test metrics:
    # 1. Zonal mean precipitation
    # 2. Regional precipitation patterns
    # 3. Seasonal variations
    # 4. Extreme precipitation events
```

**Water Vapor Distribution**
```python
def test_water_vapor():
    """Test atmospheric water vapor patterns"""
    # Expected properties:
    # - Exponential decrease with height
    # - Tropical maximum
    # - Seasonal/diurnal cycles
    # - Clausius-Clapeyron scaling
    
    # Test metrics:
    # 1. Specific humidity profiles
    # 2. Relative humidity patterns
    # 3. Water vapor transport
    # 4. Precipitation efficiency
```

### **Validation Test Implementation**

#### **1. Single Column Validation Framework**

```python
# File: validation/single_column_validation.py

import jax.numpy as jnp
import xarray as xr
from jcm.physics.icon import IconPhysics
from jcm.geometry import Geometry

class SingleColumnValidator:
    """Validate single column physics against ICON reference data"""
    
    def __init__(self, reference_data_path: str):
        self.reference_data = xr.open_dataset(reference_data_path)
        self.physics = IconPhysics()
        
    def load_initial_conditions(self):
        """Load ICON initial conditions"""
        return {
            'temperature': self.reference_data.temperature.isel(time=0),
            'u_wind': self.reference_data.u_wind.isel(time=0),
            'v_wind': self.reference_data.v_wind.isel(time=0),
            'specific_humidity': self.reference_data.specific_humidity.isel(time=0),
            'pressure': self.reference_data.pressure.isel(time=0),
            'cloud_water': self.reference_data.cloud_water.isel(time=0),
            'cloud_ice': self.reference_data.cloud_ice.isel(time=0),
        }
    
    def run_physics_step(self, state, dt=600.0):
        """Run single physics timestep"""
        # Create geometry for single column
        geometry = Geometry.from_grid_shape(nlon=1, nlat=1, nlev=47)
        
        # Run physics
        tendencies, diagnostics = self.physics.compute_tendencies(
            state, geometry, dt
        )
        
        return tendencies, diagnostics
    
    def compare_tendencies(self, jax_tendencies, icon_tendencies):
        """Compare physics tendencies"""
        metrics = {}
        
        # Temperature tendencies
        metrics['temp_rmse'] = jnp.sqrt(jnp.mean(
            (jax_tendencies.temperature - icon_tendencies.temp_tendency_total)**2
        ))
        metrics['temp_correlation'] = jnp.corrcoef(
            jax_tendencies.temperature.flatten(),
            icon_tendencies.temp_tendency_total.flatten()
        )[0, 1]
        
        # Add more tendency comparisons...
        
        return metrics
    
    def validate_physics_components(self):
        """Validate individual physics components"""
        results = {}
        
        # Test radiation
        results['radiation'] = self.validate_radiation()
        
        # Test convection  
        results['convection'] = self.validate_convection()
        
        # Test vertical diffusion
        results['vertical_diffusion'] = self.validate_vertical_diffusion()
        
        # Test surface physics
        results['surface'] = self.validate_surface_physics()
        
        return results
        
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report = {
            'test_case': self.reference_data.attrs['test_case'],
            'location': self.reference_data.attrs['location'],
            'period': self.reference_data.attrs['period'],
            'metrics': self.compare_tendencies(),
            'component_validation': self.validate_physics_components(),
            'plots': self.generate_comparison_plots(),
            'summary': self.generate_summary_statistics()
        }
        
        return report
```

#### **2. Emergent Properties Test Framework**

```python
# File: validation/emergent_properties_validation.py

import jax.numpy as jnp
import xarray as xr
from jcm.model import Model
from jcm.physics.icon import IconPhysics

class EmergentPropertiesValidator:
    """Test emergent atmospheric properties"""
    
    def __init__(self, model_config):
        self.model = Model(
            physics=IconPhysics(),
            **model_config
        )
        
    def run_aquaplanet_simulation(self, duration_days=100):
        """Run aquaplanet simulation for ITCZ testing"""
        # Uniform SST configuration
        sst = jnp.full((self.model.geometry.nlon, self.model.geometry.nlat), 300.0)
        
        # Run simulation
        state_history = self.model.run(
            duration=duration_days * 24 * 3600,
            sst=sst,
            save_frequency=6*3600  # 6-hourly output
        )
        
        return state_history
    
    def test_itcz_formation(self, state_history):
        """Test ITCZ formation and characteristics"""
        # Calculate precipitation
        precip = state_history.precipitation.mean(dim=['time', 'lon'])
        
        # Find ITCZ latitude (precipitation maximum)
        itcz_lat = state_history.lat[jnp.argmax(precip)]
        
        # Calculate ITCZ width (half-width at half-maximum)
        precip_max = jnp.max(precip)
        half_max_indices = jnp.where(precip > precip_max / 2)[0]
        itcz_width = (state_history.lat[half_max_indices[-1]] - 
                     state_history.lat[half_max_indices[0]])
        
        # Calculate ITCZ intensity
        itcz_intensity = jnp.max(precip) * 86400  # Convert to mm/day
        
        return {
            'itcz_latitude': float(itcz_lat),
            'itcz_width': float(itcz_width), 
            'itcz_intensity': float(itcz_intensity),
            'expected_latitude': {'min': -5, 'max': 10},
            'expected_width': {'min': 2, 'max': 8},  # degrees
            'expected_intensity': {'min': 8, 'max': 20},  # mm/day
        }
    
    def test_hadley_cell_circulation(self, state_history):
        """Test Hadley cell structure"""
        # Calculate meridional overturning streamfunction
        v_wind = state_history.v_wind.mean(dim=['time', 'lon'])
        
        # Integrate to get streamfunction
        # ψ = ∫(v * cos(φ)) dp
        cos_lat = jnp.cos(jnp.deg2rad(state_history.lat))
        dp = jnp.diff(state_history.pressure, axis=0)
        
        # Streamfunction calculation (simplified)
        psi = jnp.cumsum(v_wind * cos_lat[None, :] * dp[:, None], axis=0)
        
        # Find Hadley cell center
        psi_tropical = psi[:, jnp.abs(state_history.lat) < 30]
        hadley_strength = jnp.max(psi_tropical) - jnp.min(psi_tropical)
        
        return {
            'hadley_cell_strength': float(hadley_strength),
            'expected_strength': {'min': 1e11, 'max': 3e11},  # kg/s
        }
    
    def test_energy_balance(self, state_history):
        """Test global energy balance"""
        # TOA radiation balance
        sw_down_toa = state_history.sw_flux_down_toa.mean(dim='time')
        sw_up_toa = state_history.sw_flux_up_toa.mean(dim='time')
        lw_up_toa = state_history.lw_flux_up_toa.mean(dim='time')
        
        # Global mean energy balance
        global_mean_balance = (sw_down_toa - sw_up_toa - lw_up_toa).mean()
        
        # Tropical energy excess
        tropical_mask = jnp.abs(state_history.lat) < 30
        tropical_balance = (sw_down_toa - sw_up_toa - lw_up_toa)[tropical_mask].mean()
        
        return {
            'global_energy_balance': float(global_mean_balance),
            'tropical_energy_excess': float(tropical_balance),
            'expected_global_balance': {'min': -2, 'max': 2},  # W/m²
            'expected_tropical_excess': {'min': 50, 'max': 150},  # W/m²
        }
    
    def run_comprehensive_validation(self):
        """Run all emergent property tests"""
        # Run simulation
        state_history = self.run_aquaplanet_simulation()
        
        # Test all properties
        results = {
            'itcz': self.test_itcz_formation(state_history),
            'hadley_cell': self.test_hadley_cell_circulation(state_history),
            'energy_balance': self.test_energy_balance(state_history),
        }
        
        # Generate pass/fail assessment
        results['validation_summary'] = self.assess_validation_results(results)
        
        return results
    
    def assess_validation_results(self, results):
        """Assess whether validation tests pass"""
        summary = {}
        
        for test_name, test_results in results.items():
            if 'expected_' in str(test_results):
                # Check if values are within expected ranges
                passed = True
                for key, value in test_results.items():
                    if key.startswith('expected_'):
                        actual_key = key.replace('expected_', '')
                        if actual_key in test_results:
                            actual_value = test_results[actual_key]
                            expected_range = value
                            if not (expected_range['min'] <= actual_value <= expected_range['max']):
                                passed = False
                                
                summary[test_name] = {
                    'passed': passed,
                    'results': test_results
                }
        
        return summary
```

#### **3. Validation Test Suite**

```python
# File: validation/run_validation_suite.py

import argparse
from single_column_validation import SingleColumnValidator
from emergent_properties_validation import EmergentPropertiesValidator

def main():
    parser = argparse.ArgumentParser(description='Run ICON physics validation')
    parser.add_argument('--test-type', choices=['single_column', 'emergent', 'both'], 
                       default='both', help='Type of validation to run')
    parser.add_argument('--reference-data', type=str, 
                       help='Path to ICON reference data')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.test_type in ['single_column', 'both']:
        print("Running single column validation...")
        validator = SingleColumnValidator(args.reference_data)
        results['single_column'] = validator.generate_validation_report()
        
    if args.test_type in ['emergent', 'both']:
        print("Running emergent properties validation...")
        validator = EmergentPropertiesValidator(model_config={
            'grid_resolution': 'R2B4',
            'vertical_levels': 47,
            'time_step': 600,
        })
        results['emergent'] = validator.run_comprehensive_validation()
    
    # Generate combined report
    generate_validation_report(results, args.output_dir)
    
    print(f"Validation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

## Priority Implementation Plan

### **Phase 1: Single Column Validation (2-3 weeks)**
1. ✅ **Week 1**: Generate ICON reference data for 4 test cases
2. **Week 2**: Implement single column validation framework
3. **Week 3**: Run validation tests and generate reports

### **Phase 2: Emergent Properties Testing (2-3 weeks)**  
1. **Week 1**: Implement ITCZ and Hadley cell tests
2. **Week 2**: Implement energy balance and circulation tests
3. **Week 3**: Run comprehensive validation suite

### **Phase 3: Full Climate Validation (4-6 weeks)**
1. **Weeks 1-2**: Multi-year climate simulations
2. **Weeks 3-4**: Seasonal cycle and variability analysis
3. **Weeks 5-6**: Extreme weather and climate sensitivity tests

## Success Criteria

### **Validation Thresholds**
- **Single Column**: Physics tendencies within 20% RMSE of ICON
- **ITCZ**: Latitude within 5°, intensity within 50% of observations
- **Energy Balance**: Global balance within 5 W/m², tropical excess within 30%
- **Circulation**: Hadley cell strength within 50% of reanalysis

### **Trust Metrics**
- **Level 1**: Basic physics correctness (single column)
- **Level 2**: Emergent phenomena formation (ITCZ, jets)
- **Level 3**: Quantitative climate accuracy (energy balance)
- **Level 4**: Long-term stability and variability

This comprehensive validation plan will establish scientific trust in the JAX-GCM ICON physics implementation and provide confidence for production climate modeling applications.