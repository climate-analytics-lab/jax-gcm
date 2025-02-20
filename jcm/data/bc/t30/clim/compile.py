import xarray as xr
import os
from pathlib import Path

# Set the input directory path
input_dir = Path('')
output_file = 'boundaries.nc'

def merge_netcdf_files(input_dir, output_file):
    """
    Merge all NetCDF files in the input directory into a single file.
    
    Args:
        input_dir (Path): Directory containing NetCDF files
        output_file (str): Name of the output merged file
    """
    # Get all .nc files in the directory
    nc_files = list(input_dir.glob('*.nc'))
    
    if not nc_files:
        raise ValueError(f"No NetCDF files found in {input_dir}")
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # Read all datasets
    datasets = []
    for file in nc_files:
        print(f"Reading {file.name}")
        ds = xr.open_dataset(file)
        datasets.append(ds)
    
    # Concatenate all datasets
    print("Merging datasets...")
    merged_ds = xr.concat(datasets, dim='time')
    
    # Sort by time dimension if it exists
    if 'time' in merged_ds.dims:
        merged_ds = merged_ds.sortby('time')
    
    # Save merged dataset
    print(f"Saving merged dataset to {output_file}")
    merged_ds.to_netcdf(output_file)
    
    # Close all datasets
    for ds in datasets:
        ds.close()
    
    print("Done!")

if __name__ == "__main__":
    try:
        merge_netcdf_files(input_dir, output_file)
    except Exception as e:
        print(f"Error: {e}")
