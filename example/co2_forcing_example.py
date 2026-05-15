import jax
print("Devices: ", jax.devices())

import jax.numpy as jnp  # noqa: E402

import jcm  # noqa: E402
print(f"Note: jcm.__file__ = {str(jcm.__file__)}")

from jcm.model import Model  # noqa: E402
from jcm.forcing import default_forcing  # noqa: E402
from jcm.physics.speedy.speedy_coords import get_speedy_coords  # noqa: E402

coords = get_speedy_coords()


# Default forcing with reference CO2 absorptivity (6.0)
forcing_default = default_forcing(coords.horizontal).copy(ablco2=jnp.asarray(6.0))

# Elevated CO2 — double the reference value
forcing_high_co2 = default_forcing(coords.horizontal).copy(ablco2=jnp.asarray(12.0))

model = Model(coords=coords)
total_time = 360.0
print("Run with default CO2")
preds_default = model.run(
    forcing=forcing_default,
    save_interval=1.0,
    total_time=total_time,
)

print("Run with elevated CO2")
preds_high = model.run(
    forcing=forcing_high_co2,
    save_interval=1.0,
    total_time=total_time,
)

# Check ablco2 appears in output and differs between runs
ds_default = preds_default.to_xarray()
ds_high    = preds_high.to_xarray()

print("ablco2 default:", ds_default["mod_radcon.ablco2"].values)
print("ablco2 high:   ", ds_high["mod_radcon.ablco2"].values)

# Temperature should differ due to changed CO2 absorptivity
temp_diff = (ds_high["temperature"] - ds_default["temperature"]).values
print("max |ΔT|:", abs(temp_diff).max())


ds_default.to_netcdf("atm_default.nc")
ds_high.to_netcdf("atm_doubleco2.nc")
