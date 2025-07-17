#!/bin/bash
#SBATCH --job-name=icon_tropical_convection
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --output=tropical_convection_%j.out
#SBATCH --error=tropical_convection_%j.err

# Test Case: Tropical Convection
# Location: 0°N, 180°E (Equatorial Pacific)
# Period: 10 days starting January 1, 2020
# Purpose: Deep convection, cloud physics, radiation-convection interaction

echo "Starting ICON tropical convection test case..."
echo "Test case: Tropical Convection"
echo "Location: 0°N, 180°E"
echo "Period: 10 days starting January 1, 2020"
echo "Started at: $(date)"

# Load required modules (adjust for your HPC system)
module load intel/2021.4.0 || module load gcc/11.2.0
module load netcdf/4.7.4
module load hdf5/1.10.7
module load openmpi/4.1.1

# Set up environment variables
export ICON_BASE_PATH="${ICON_BASE_PATH:-/path/to/icon}"
export ICON_GRID_PATH="${ICON_BASE_PATH}/grids"
export ICON_DATA_PATH="${ICON_BASE_PATH}/data"
export ICON_EXEC="${ICON_BASE_PATH}/bin/icon"

# Test case parameters
export ICON_LAT=0.0
export ICON_LON=180.0
export ICON_CASE="tropical_convection"

# Create single column grid
echo "Creating single column grid..."
python3 << EOF
import numpy as np
import netCDF4 as nc

# Create single column grid at 0°N, 180°E
lat = ${ICON_LAT}
lon = ${ICON_LON}

# Create minimal grid file
with nc.Dataset('scm_grid_tropical.nc', 'w') as f:
    # Dimensions
    f.createDimension('cell', 1)
    f.createDimension('vertex', 3)
    f.createDimension('edge', 3)
    
    # Cell centers
    clat = f.createVariable('clat', 'f8', ('cell',))
    clon = f.createVariable('clon', 'f8', ('cell',))
    clat[:] = np.deg2rad(lat)
    clon[:] = np.deg2rad(lon)
    
    # Cell areas (dummy values)
    cell_area = f.createVariable('cell_area', 'f8', ('cell',))
    cell_area[:] = 1.0
    
    # Add required attributes
    f.setncattr('grid_id', 'scm_tropical')
    f.setncattr('grid_level', 0)

print(f"Created single column grid at {lat}°N, {lon}°E")
EOF

# Set up input data links
echo "Setting up input data..."
if [ -d "${ICON_DATA_PATH}/external/echam_rad_data" ]; then
    ln -sf ${ICON_DATA_PATH}/external/echam_rad_data ./
fi
if [ -d "${ICON_DATA_PATH}/external/rrtmgp" ]; then
    ln -sf ${ICON_DATA_PATH}/external/rrtmgp ./
fi

# Check if ICON executable exists
if [ ! -x "${ICON_EXEC}" ]; then
    echo "ERROR: ICON executable not found at ${ICON_EXEC}"
    echo "Please set ICON_BASE_PATH correctly"
    exit 1
fi

# Run ICON
echo "Running ICON..."
${ICON_EXEC} tropical_convection.nml

# Check if run completed successfully
if [ $? -eq 0 ]; then
    echo "ICON run completed successfully!"
    
    # Post-process output
    echo "Post-processing output..."
    python3 << EOF
import xarray as xr
import numpy as np
import sys

try:
    # Load all output files
    print("Loading output files...")
    ds_phys = xr.open_dataset("tropical_convection_physics_20200101T000000Z.nc")
    ds_rad = xr.open_dataset("tropical_convection_radiation_20200101T000000Z.nc")
    ds_sfc = xr.open_dataset("tropical_convection_surface_20200101T000000Z.nc")
    ds_conv = xr.open_dataset("tropical_convection_convection_20200101T000000Z.nc")
    ds_turb = xr.open_dataset("tropical_convection_turbulence_20200101T000000Z.nc")
    
    # Combine into validation dataset
    ds_validation = xr.Dataset()
    
    # Copy key variables
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
    
    # Radiation fluxes
    ds_validation['sw_flux_up'] = ds_rad.swflx_up
    ds_validation['sw_flux_down'] = ds_rad.swflx_dn
    ds_validation['lw_flux_up'] = ds_rad.lwflx_up
    ds_validation['lw_flux_down'] = ds_rad.lwflx_dn
    
    # Surface fluxes
    ds_validation['surface_sensible_flux'] = ds_sfc.shfl_s
    ds_validation['surface_latent_flux'] = ds_sfc.lhfl_s
    
    # Convection diagnostics
    ds_validation['convective_precip'] = ds_conv.rain_con
    ds_validation['cape'] = ds_conv.cape_con
    ds_validation['cin'] = ds_conv.cin_con
    
    # Turbulence diagnostics
    ds_validation['tke'] = ds_turb.tke
    ds_validation['richardson_number'] = ds_turb.ri
    ds_validation['mixing_length'] = ds_turb.mixlen
    
    # Add metadata
    ds_validation.attrs['case_name'] = 'tropical_convection'
    ds_validation.attrs['location'] = '0°N, 180°E'
    ds_validation.attrs['model'] = 'ICON'
    ds_validation.attrs['physics'] = 'ECHAM'
    
    # Save processed data
    output_file = "icon_validation_tropical_convection.nc"
    ds_validation.to_netcdf(output_file)
    print(f"Validation data saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Temperature range: {ds_validation.temperature.min().values:.1f} - {ds_validation.temperature.max().values:.1f} K")
    print(f"Surface temperature: {ds_validation.temperature.isel(height=-1).mean().values:.1f} K")
    print(f"Total precipitation: {ds_validation.convective_precip.sum().values:.2f} mm")
    print(f"Mean CAPE: {ds_validation.cape.mean().values:.1f} J/kg")
    
except Exception as e:
    print(f"Error processing output: {e}")
    sys.exit(1)
EOF
    
    echo "Post-processing completed!"
    echo "Output file: icon_validation_tropical_convection.nc"
    
else
    echo "ERROR: ICON run failed!"
    echo "Check output files for details"
    exit 1
fi

echo "Tropical convection test case completed at: $(date)"
echo "Files created:"
ls -la icon_validation_tropical_convection.nc tropical_convection_*.nc 2>/dev/null || echo "No output files found"