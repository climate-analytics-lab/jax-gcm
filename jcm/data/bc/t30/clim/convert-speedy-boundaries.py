import xarray as xr
import jax.numpy as jnp

ds = xr.open_dataset('./boundaries.nc')

from jcm.physics.speedy.params import Parameters

if not 'soilw_am' in ds.data_vars:
    veg_high = ds.vegh
    veg_low = ds.vegl
    assert jnp.all(0.0 <= veg_high.values)
    assert jnp.all(1.0 >= veg_high.values)
    assert jnp.all(0.0 <= veg_low.values)
    assert jnp.all(1.0 >= veg_low.values)
    veg = veg_high + 0.8 * veg_low

    p = Parameters.default()
    idep2 = 3
    rsw = 1.0 / (p.land_model.swcap + idep2 * (p.land_model.swcap - p.land_model.swwil))

    swl2_raw = veg.values[:, :, None] * idep2 * (ds.swl2.values - p.land_model.swwil)
    soilw_raw = rsw * (ds.swl1.values + jnp.maximum(0.0, swl2_raw))
    soilw_am = jnp.minimum(1.0, soilw_raw)
    
    ds['soilw_am'] = 0*ds.swl1 + soilw_am

ds = ds.drop_vars([v for v in ('swl1', 'swl2', 'swl3', 'vegh', 'vegl') if v in ds.data_vars])

ds.to_netcdf('./boundaries-converted.nc')

ds.close()
import os
os.rename('./boundaries-converted.nc', './boundaries.nc')