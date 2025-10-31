import xarray as xr
import sys
from pathlib import Path
import numpy as np
from jcm.physics.speedy.physical_constants import sd2sc, swcap, swwil

# Set the input directory path
input_dir = Path(__file__).parent / 't30/clim'
output_file, orography_file = Path(__file__).parent / 't30/clim/boundaries.nc', Path(__file__).parent / 't30/clim/orography.nc'
file_names = ['land.nc', 'sea_ice.nc', 'sea_surface_temperature.nc', 'snow.nc', 'soil.nc', 'surface.nc']

def process_boundaries(ds):
    """
    Convert compiled speedy.f90 boundary conditions to format expected by jcm.

    Args:
        ds (xarray.Dataset): Dataset containing boundary conditions.
    """
    # Reorder coordinates to match jcm expectations
    ds = ds.sortby('lat', ascending=True)
    for var in ds.data_vars:
        if 'time' in ds[var].dims:
            ds[var] = ds[var].transpose('lon', 'lat', 'time')
        else:
            ds[var] = ds[var].transpose('lon', 'lat')

    # Fill nan temperatures with mean, other nans with 0
    for var in ds.data_vars:
        arr = ds[var]
        mask = arr == 9.96921e36
        fill_val = arr.where(~mask).mean() if var in ['stl', 'sst'] else 0.
        ds[var] = arr.where(~mask, fill_val)

    # Compute soil moisture variable used by jcm
    def compute_soilw_am(veg_high, veg_low, swl1, swl2):
        assert np.all(0.0 <= veg_high) and np.all(veg_high <= 1.0)
        assert np.all(0.0 <= veg_low) and np.all(veg_low <= 1.0)
        veg = veg_high + 0.8 * veg_low
        idep2 = 3
        rsw = 1.0 / (swcap + idep2 * (swcap - swwil))
        swl2_raw = veg[:, :, np.newaxis] * idep2 * (swl2 - swwil)
        soilw_raw = rsw * (swl1 + np.maximum(0.0, swl2_raw))
        return np.minimum(1.0, soilw_raw)

    soilw_am = compute_soilw_am(ds.vegh.values, ds.vegl.values, ds.swl1.values, ds.swl2.values)
    ds['soilw_am'] = xr.DataArray(soilw_am, dims=ds['swl1'].dims, coords=ds['swl1'].coords)

    ds['snowc'] = ds['snowd'] / sd2sc # Convert snow depth to snow cover
    
    da_orog = ds['orog']
    return ds.drop_vars({'swl1', 'swl2', 'swl3', 'vegh', 'vegl', 'orog', 'snowd'}), da_orog

def main(argv=None):
    """
    Main entrypoint for compile CLI and for importable use.

    Args:
        argv (list|None): list of command-line args (not including program name).
                          If None, uses sys.argv[1:].
    Returns:
        int: exit code (0 = success)
    """
    if argv is None:
        argv = sys.argv[1:]

    try:
        print("Compiling dataset...")
        merged_ds = xr.open_mfdataset([input_dir / fname for fname in file_names], combine='by_coords')

        print("Processing dataset...")
        processed_ds, da_orog = process_boundaries(merged_ds)
        
        print(f"Saving processed dataset to {output_file}")
        processed_ds.to_netcdf(output_file)
        processed_ds.close()

        print(f"Saving orography to {orography_file}")
        da_orog.to_netcdf(orography_file)
        da_orog.close()
        
        print("Done!")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())