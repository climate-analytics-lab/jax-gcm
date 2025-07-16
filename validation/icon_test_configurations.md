# ICON HPC Test Configurations

## Quick Reference for HPC ICON Runs

### **Single Column Test Cases**

#### **Test Case 1: Tropical Convection**
```bash
# Location: Equatorial Pacific (0°N, 180°E)
# Period: 10 days starting January 1
# Purpose: Deep convection, cloud physics, radiation-convection interaction

export ICON_TEST_CASE="tropical_convection"
export ICON_LAT=0.0
export ICON_LON=180.0
export ICON_START_DATE="2020-01-01T00:00:00Z"
export ICON_END_DATE="2020-01-11T00:00:00Z"
export ICON_OUTPUT_FREQ="PT1H"  # Hourly output
```

#### **Test Case 2: Mid-latitude Winter Storm**
```bash
# Location: Western Europe (50°N, 0°E)
# Period: 10 days starting January 15
# Purpose: Frontal systems, vertical mixing, surface fluxes

export ICON_TEST_CASE="midlat_winter"
export ICON_LAT=50.0
export ICON_LON=0.0
export ICON_START_DATE="2020-01-15T00:00:00Z"
export ICON_END_DATE="2020-01-25T00:00:00Z"
export ICON_OUTPUT_FREQ="PT1H"
```

#### **Test Case 3: Arctic Polar Night**
```bash
# Location: Central Arctic (85°N, 0°E)
# Period: 10 days starting January 1
# Purpose: Extreme radiation conditions, surface physics, stable boundary layer

export ICON_TEST_CASE="arctic_polar"
export ICON_LAT=85.0
export ICON_LON=0.0
export ICON_START_DATE="2020-01-01T00:00:00Z"
export ICON_END_DATE="2020-01-11T00:00:00Z"
export ICON_OUTPUT_FREQ="PT1H"
```

#### **Test Case 4: Subtropical Clear Sky**
```bash
# Location: Azores High (30°N, 30°W)
# Period: 10 days starting July 15
# Purpose: Subsidence, clear-sky radiation, surface energy balance

export ICON_TEST_CASE="subtropical_clear"
export ICON_LAT=30.0
export ICON_LON=-30.0
export ICON_START_DATE="2020-07-15T00:00:00Z"
export ICON_END_DATE="2020-07-25T00:00:00Z"
export ICON_OUTPUT_FREQ="PT1H"
```

### **ICON Namelist Template**

```fortran
! File: icon_scm_template.nml

&parallel_nml
 nproma          = 1
 p_test_run      = .FALSE.
 num_io_procs    = 1
 num_restart_procs = 0
 num_prefetch_proc = 0
/

&grid_nml
 dynamics_grid_filename = "scm_grid.nc"
 radiation_grid_filename = "scm_grid.nc"
 dynamics_parent_grid_id = 0
 lredgrid_phys    = .FALSE.
 lfeedback        = .FALSE.
 l_limited_area   = .FALSE.
 ifeedback_type   = 2
 start_time       = 0.0
/

&run_nml
 num_lev         = 47              ! 47 vertical levels
 lvert_nest      = .FALSE.
 dtime           = 600.0           ! 10-minute time step
 ldynamics       = .TRUE.
 ltestcase       = .TRUE.
 ltransport      = .TRUE.
 iforcing        = 3               ! NWP forcing
 ltimer          = .FALSE.
 timers_level    = 10
 check_uuid_gracefully = .TRUE.
 output          = "nml"
/

&dynamics_nml
 iequations      = 3               ! NH equations
 idiv_method     = 1
 divavg_cntrwgt  = 0.50
 lcoriolis       = .TRUE.
/

&transport_nml
 ctracer_list    = "hus","clw","cli"
 ivadv_tracer    = 3,3,3
 itype_hlimit    = 3,4,4
 ihadv_tracer    = 52,2,2
/

&nwp_phy_nml
 inwp_gscp       = 1               ! Grid-scale precipitation
 inwp_convection = 1               ! Tiedtke convection
 inwp_radiation  = 1               ! ECHAM radiation
 inwp_cldcover   = 1               ! Cloud cover
 inwp_turb       = 1               ! Turbulence
 inwp_satad      = 1               ! Saturation adjustment
 inwp_sso        = 0               ! No SSO for single column
 inwp_gwd        = 0               ! No GWD for single column
 inwp_surface    = 1               ! Surface physics
 latm_above_top  = .TRUE.
 efdt_min_raylfric = 7200.0
 itype_z0        = 2
 icapdcycl       = 3
 icpl_aero_conv  = 1
 icpl_aero_gscp  = 1
 icpl_o3_tp      = 1
 dt_rad          = 3600.0          ! Radiation every hour
 dt_conv         = 600.0           ! Convection every timestep
 dt_sso          = 600.0
 dt_gwd          = 600.0
/

&radiation_nml
 irad_o3         = 3               ! Ozone climatology
 irad_aero       = 2               ! Kinne aerosol
 irad_co2        = 2               ! CO2 concentration
 irad_ch4        = 2               ! CH4 concentration
 irad_n2o        = 2               ! N2O concentration
 irad_o2         = 2               ! O2 concentration
 irad_cfc11      = 2               ! CFC11 concentration
 irad_cfc12      = 2               ! CFC12 concentration
 vmr_co2         = 407.8e-06       ! CO2 VMR
 vmr_ch4         = 1.85e-06        ! CH4 VMR
 vmr_n2o         = 323.0e-09       ! N2O VMR
 vmr_o2          = 0.20946         ! O2 VMR
 vmr_cfc11       = 232.0e-12       ! CFC11 VMR
 vmr_cfc12       = 519.0e-12       ! CFC12 VMR
 ecrad_data_path = ""
/

&ls_forcing_nml
 is_subsidence_moment = .FALSE.
 is_subsidence_heat   = .FALSE.
 is_advection         = .FALSE.
 is_geowind           = .FALSE.
 is_rad_forcing       = .FALSE.
 is_nudging           = .FALSE.
/

&scm_nml
 i_scm_netcdf     = 1              ! NetCDF output
 lscm_read_tke    = .FALSE.
 lscm_read_z0     = .FALSE.
 dt_relax         = 21600.0
 scm_sfc_temp     = 288.0          ! Surface temperature
 scm_sfc_qv       = 0.01           ! Surface humidity
 scm_sfc_mom      = 0.1            ! Surface momentum
 scm_sfc_sensible = 0.0            ! Surface sensible heat flux
 scm_sfc_latent   = 0.0            ! Surface latent heat flux
/

&output_nml
 output_filename  = "icon_scm_output"
 filename_format  = "<output_filename>_<datetime2>"
 filetype         = 4              ! NetCDF
 dom              = 1
 output_bounds    = 0., 10., 3600. ! 0-10 days, hourly
 steps_per_file   = 240            ! 10 days * 24 hours
 mode             = 1              ! Forecast mode
 include_last     = .TRUE.
 output_grid      = .TRUE.
 ml_varlist       = 'group:atm_ml_vars','group:rad_vars','group:phys_tend_vars'
 pl_varlist       = 'group:atm_pl_vars'
 hl_varlist       = 'group:atm_hl_vars'
/

&extpar_nml
 itopo            = 1              ! Topography
 n_iter_smooth_topo = 1
 hgtdiff_max_smooth_topo = 750.0
 heightdiff_threshold = 3000.0
 itype_vegetation_cycle = 1
 itype_lwemiss    = 1
/
```

### **Required Output Variables**

```fortran
! In output_nml section, add specific variable lists:

&output_nml
 output_filename  = "physics_tendencies"
 ml_varlist       = 'temp','u','v','w','pres','rho',
                    'qv','qc','qi','qr','qs','qg',
                    'tke','clc','clw','cli',
                    'ddt_temp_radsw','ddt_temp_radlw','ddt_temp_turb',
                    'ddt_temp_conv','ddt_temp_gscp','ddt_temp_sso',
                    'ddt_u_turb','ddt_u_conv','ddt_u_sso',
                    'ddt_v_turb','ddt_v_conv','ddt_v_sso',
                    'ddt_qv_turb','ddt_qv_conv','ddt_qv_gscp',
                    'ddt_qc_turb','ddt_qc_conv','ddt_qc_gscp',
                    'ddt_qi_turb','ddt_qi_conv','ddt_qi_gscp'
/

&output_nml
 output_filename  = "radiation_diagnostics"
 ml_varlist       = 'lwflxall','lwflx_up','lwflx_dn',
                    'swflxall','swflx_up','swflx_dn',
                    'lwflx_up_sfc','lwflx_dn_sfc',
                    'swflx_up_sfc','swflx_dn_sfc',
                    'lwflx_up_toa','lwflx_dn_toa',
                    'swflx_up_toa','swflx_dn_toa',
                    'sod_t','sod_lwflx','sod_swflx'
/

&output_nml
 output_filename  = "surface_fluxes"
 ml_varlist       = 'shfl_s','lhfl_s','umfl_s','vmfl_s',
                    'sensiflx','latentflx','u_10m','v_10m',
                    't_2m','td_2m','rh_2m','sp_10m',
                    'runoff_s','runoff_g','rstom',
                    'snow_melt','h_snow','w_snow',
                    't_s','t_seasfc','fr_seaice'
/

&output_nml
 output_filename  = "convection_diagnostics"
 ml_varlist       = 'rain_con','snow_con','prr_con','prs_con',
                    'bas_con','top_con','mflx_con','cape_con',
                    'cin_con','cape_ml','cin_ml','hbas_con',
                    'htop_con','totte','qconv','ttend_conv',
                    'qtend_conv','utend_conv','vtend_conv'
/

&output_nml
 output_filename  = "turbulence_diagnostics"
 ml_varlist       = 'tke','tkvm','tkvh','rcld','edr',
                    'bruvais','ri','mixlen','z0','gz0',
                    'tcm','tch','tfm','tfh','tfv',
                    'shfl_s','lhfl_s','umfl_s','vmfl_s'
/
```

### **Job Script Template**

```bash
#!/bin/bash
#SBATCH --job-name=icon_scm_validation
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --output=icon_scm_%j.out
#SBATCH --error=icon_scm_%j.err

# Load modules
module load intel/2021.4.0
module load netcdf/4.7.4
module load hdf5/1.10.7

# Set up environment
export ICON_BASE_PATH="/path/to/icon"
export ICON_GRID_PATH="${ICON_BASE_PATH}/grids"
export ICON_DATA_PATH="${ICON_BASE_PATH}/data"

# Create single column grid
python3 create_scm_grid.py --lat=${ICON_LAT} --lon=${ICON_LON} --output=scm_grid.nc

# Set up input data
ln -sf ${ICON_DATA_PATH}/external/echam_rad_data ./
ln -sf ${ICON_DATA_PATH}/external/rrtmgp ./

# Run ICON
${ICON_BASE_PATH}/bin/icon icon_scm_${ICON_TEST_CASE}.nml

# Post-process output
python3 postprocess_scm_output.py --case=${ICON_TEST_CASE}

echo "ICON single column run completed for ${ICON_TEST_CASE}"
```

### **Data Processing Script**

```python
#!/usr/bin/env python3
# File: postprocess_scm_output.py

import xarray as xr
import numpy as np
import argparse
import os

def process_icon_output(case_name):
    """Process ICON single column output for JAX-GCM validation"""
    
    # Load all output files
    ds_phys = xr.open_dataset(f"physics_tendencies_{case_name}.nc")
    ds_rad = xr.open_dataset(f"radiation_diagnostics_{case_name}.nc")
    ds_sfc = xr.open_dataset(f"surface_fluxes_{case_name}.nc")
    ds_conv = xr.open_dataset(f"convection_diagnostics_{case_name}.nc")
    ds_turb = xr.open_dataset(f"turbulence_diagnostics_{case_name}.nc")
    
    # Combine into single validation dataset
    ds_validation = xr.Dataset()
    
    # Primary state variables
    ds_validation['temperature'] = ds_phys.temp
    ds_validation['u_wind'] = ds_phys.u
    ds_validation['v_wind'] = ds_phys.v
    ds_validation['pressure'] = ds_phys.pres
    ds_validation['specific_humidity'] = ds_phys.qv
    ds_validation['cloud_water'] = ds_phys.qc
    ds_validation['cloud_ice'] = ds_phys.qi
    
    # Physics tendencies
    ds_validation['temp_tendency_rad'] = ds_phys.ddt_temp_radsw + ds_phys.ddt_temp_radlw
    ds_validation['temp_tendency_conv'] = ds_phys.ddt_temp_conv
    ds_validation['temp_tendency_vdiff'] = ds_phys.ddt_temp_turb
    ds_validation['temp_tendency_total'] = (ds_phys.ddt_temp_radsw + ds_phys.ddt_temp_radlw + 
                                           ds_phys.ddt_temp_conv + ds_phys.ddt_temp_turb)
    
    # Radiation diagnostics
    ds_validation['sw_flux_up'] = ds_rad.swflx_up
    ds_validation['sw_flux_down'] = ds_rad.swflx_dn
    ds_validation['lw_flux_up'] = ds_rad.lwflx_up
    ds_validation['lw_flux_down'] = ds_rad.lwflx_dn
    
    # Surface fluxes
    ds_validation['surface_sensible_flux'] = ds_sfc.shfl_s
    ds_validation['surface_latent_flux'] = ds_sfc.lhfl_s
    ds_validation['surface_momentum_flux_u'] = ds_sfc.umfl_s
    ds_validation['surface_momentum_flux_v'] = ds_sfc.vmfl_s
    
    # Convection diagnostics
    ds_validation['convective_precip'] = ds_conv.rain_con
    ds_validation['cape'] = ds_conv.cape_con
    ds_validation['cin'] = ds_conv.cin_con
    ds_validation['cloud_base_height'] = ds_conv.hbas_con
    ds_validation['cloud_top_height'] = ds_conv.htop_con
    
    # Turbulence diagnostics
    ds_validation['tke'] = ds_turb.tke
    ds_validation['mixing_length'] = ds_turb.mixlen
    ds_validation['richardson_number'] = ds_turb.ri
    
    # Add metadata
    ds_validation.attrs['case_name'] = case_name
    ds_validation.attrs['model'] = 'ICON'
    ds_validation.attrs['physics'] = 'ECHAM'
    ds_validation.attrs['created'] = pd.Timestamp.now().isoformat()
    
    # Save processed data
    output_file = f"icon_validation_data_{case_name}.nc"
    ds_validation.to_netcdf(output_file)
    print(f"Processed validation data saved to {output_file}")
    
    return ds_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ICON SCM output for validation')
    parser.add_argument('--case', required=True, help='Test case name')
    args = parser.parse_args()
    
    process_icon_output(args.case)
```

### **Quick Start Commands**

```bash
# 1. Set up test case
export ICON_TEST_CASE="tropical_convection"
export ICON_LAT=0.0
export ICON_LON=180.0

# 2. Copy and modify namelist
cp icon_scm_template.nml icon_scm_${ICON_TEST_CASE}.nml

# 3. Submit job
sbatch icon_scm_job.sh

# 4. After completion, check output
ls -la icon_validation_data_${ICON_TEST_CASE}.nc
```

### **Expected Output Files**

- `icon_validation_data_tropical_convection.nc` - Main validation dataset
- `icon_validation_data_midlat_winter.nc` - Winter storm case
- `icon_validation_data_arctic_polar.nc` - Arctic case  
- `icon_validation_data_subtropical_clear.nc` - Clear sky case

Each file contains standardized variables for direct comparison with JAX-GCM output.

## Notes

1. **Grid Creation**: Single column grids need to be created for each lat/lon point
2. **Input Data**: Ensure all required external data (radiation tables, etc.) are available
3. **Vertical Levels**: Use standard ICON-A 47-level configuration for consistency
4. **Time Step**: 10-minute time step matches typical ICON-A configuration
5. **Output Frequency**: Hourly output provides good temporal resolution for validation

This configuration should produce comprehensive reference data for validating the JAX-GCM ICON physics implementation.