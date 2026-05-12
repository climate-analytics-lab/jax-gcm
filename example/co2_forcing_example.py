import jax.numpy as jnp
from jcm.model import Model
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords

coords = get_speedy_coords()

# Default forcing with reference CO2 absorptivity (6.0)
forcing_default = ForcingData.zeros(
    nodal_shape=coords.horizontal.nodal_shape,
    ablco2=6.0,
)

# Elevated CO2 — double the reference value
forcing_high_co2 = forcing_default.copy(ablco2=jnp.asarray(12.0))

model = Model(coords=coords)

# Run with default CO2
preds_default = model.run(
    forcing=forcing_default,
    save_interval=1.0,
    total_time=2.0,
)

# Run with elevated CO2
preds_high = model.run(
    forcing=forcing_high_co2,
    save_interval=1.0,
    total_time=2.0,
)

# Check ablco2 appears in output and differs between runs
ds_default = preds_default.to_xarray()
ds_high    = preds_high.to_xarray()

print("ablco2 default:", ds_default["mod_radcon.ablco2"].values)
print("ablco2 high:   ", ds_high["mod_radcon.ablco2"].values)

# Temperature should differ due to changed CO2 absorptivity
temp_diff = (ds_high["temperature"] - ds_default["temperature"]).values
print("max |ΔT|:", abs(temp_diff).max())
