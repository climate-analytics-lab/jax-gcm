import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

print("Loading data...")
data = dict(
    default = xr.load_dataset("atm_default.nc"),
    double_co2 = xr.load_dataset("atm_doubleco2.nc"),
)


print("Create figure...")
fig, ax = plt.subplots(1, 1)

for casename, ds in data.items():
    da_avg = ds["temperature"].isel(level=0).weighted(np.cos(ds.coords["lat"]*np.pi/180.0)).mean(dim=["lat", "lon"])
    print(da_avg)
    ax.plot(da_avg.coords["time"], da_avg, label=casename)

ax.legend()

print("Showing data")
plt.show()



