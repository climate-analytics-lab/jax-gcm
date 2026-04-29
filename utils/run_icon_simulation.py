"""Plot helpers for ICON output netCDFs.

The model-construction surface that this script used to carry has moved
into ``jcm.runners`` (composable from Hydra config groups under
``jcm/config/``). Use the CLI directly for new runs::

    python -m jcm.main physics=icon grid=icon_t85_l47_hybrid \
        run.total_time=30 run.save_interval=1 run.output=run.nc

This file now only owns the post-run climatology / diurnal-cycle plots,
which haven't been pulled into the package proper because they pull in
matplotlib.
"""

from __future__ import annotations

import argparse


def plot_climatology(ds, output_prefix: str = "icon_t85_47lev") -> None:
    """Plot time-mean precipitation, net TOA radiation and surface T."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if "lon" in ds.coords:
        lon = ds.lon.values
        lat = ds.lat.values
    else:
        lon = np.arange(ds.sizes.get("longitude", ds.sizes.get("lon", 1)))
        lat = np.arange(ds.sizes.get("latitude", ds.sizes.get("lat", 1)))

    def _to_latlon(field):
        v = field.values.squeeze()
        if v.ndim == 2 and v.shape == (len(lon), len(lat)):
            return v.T
        return v

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": None})

    ax = axes[0]
    precip_vars = [
        v for v in ("convection.precip_conv", "clouds.precip_rain", "clouds.precip_snow")
        if v in ds
    ]
    if precip_vars:
        total_precip = sum(ds[v].mean(dim="time") for v in precip_vars)
        precip_mm = total_precip * 86400
        im = ax.pcolormesh(lon, lat, _to_latlon(precip_mm), cmap="Blues", shading="nearest")
        plt.colorbar(im, ax=ax, label="mm/day")
        ax.set_title(f"Precipitation ({', '.join(precip_vars)})")
    else:
        ax.text(
            0.5, 0.5, "No precip variables found\n" + str(list(ds.data_vars)[:10]),
            transform=ax.transAxes, ha="center", fontsize=8,
        )
        ax.set_title("Precipitation (not found)")

    ax = axes[1]
    sw_down_key = next((v for v in ds.data_vars if "toa_sw_down" in v), None)
    sw_up_key = next((v for v in ds.data_vars if "toa_sw_up" in v), None)
    lw_up_key = next((v for v in ds.data_vars if "toa_lw_up" in v), None)
    if sw_down_key and sw_up_key and lw_up_key:
        net_toa = (ds[sw_down_key] - ds[sw_up_key] - ds[lw_up_key]).mean(dim="time")
        im = ax.pcolormesh(
            lon, lat, _to_latlon(net_toa), cmap="RdBu_r",
            shading="nearest", vmin=-150, vmax=150,
        )
        plt.colorbar(im, ax=ax, label="W/m²")
        ax.set_title("Net TOA Radiation")
    else:
        available = [v for v in ds.data_vars if "toa" in v or "radiation" in v.lower()]
        ax.text(
            0.5, 0.5, f"TOA vars found: {available}", transform=ax.transAxes,
            ha="center", fontsize=8, wrap=True,
        )
        ax.set_title("Net TOA (not found)")

    ax = axes[2]
    sfc_temp_key = next(
        (v for v in ds.data_vars
         if "surface_temperature" in v and "tendency" not in v),
        None,
    )
    if sfc_temp_key:
        sfc_t = ds[sfc_temp_key].mean(dim="time")
        im = ax.pcolormesh(lon, lat, _to_latlon(sfc_t), cmap="RdYlBu_r", shading="nearest")
        plt.colorbar(im, ax=ax, label="K")
        ax.set_title("Surface Temperature")
    else:
        available = [v for v in ds.data_vars if "surface" in v or "temp" in v.lower()]
        ax.text(
            0.5, 0.5, f"Surface temp vars: {available}", transform=ax.transAxes,
            ha="center", fontsize=8, wrap=True,
        )
        ax.set_title("Surface Temp (not found)")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    out_path = f"{output_prefix}_climatology.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved climatology plot to {out_path}")
    plt.close()


def plot_diurnal_radiation(ds, output_prefix: str = "icon_diurnal") -> None:
    """Plot global-mean and equatorial-Hovmöller diurnal radiation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def find(*keys):
        for k in keys:
            for v in ds.data_vars:
                if all(p in v for p in k.split("+")):
                    return v
        return None

    sw_down = find("toa_sw_down")
    sw_up = find("toa_sw_up")
    lw_up = find("toa_lw_up")
    sfc_lw_down = find("surface+lw+down", "sfc_lw_down", "lw_down")
    sfc_sw_down = find("surface+sw+down", "sfc_sw_down", "sw_down")

    time = ds.time.values
    lon = ds.lon.values if "lon" in ds.coords else np.arange(ds.sizes.get("lon", 1))
    lat = ds.lat.values if "lat" in ds.coords else np.arange(ds.sizes.get("lat", 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for name, var in (
        ("TOA SW down", sw_down), ("TOA SW up", sw_up), ("TOA LW up", lw_up),
        ("SFC LW down", sfc_lw_down), ("SFC SW down", sfc_sw_down),
    ):
        if var is None:
            continue
        arr = ds[var]
        lat_rad = np.deg2rad(lat) if lat.max() > 3.2 else lat
        w = np.cos(lat_rad)
        dims = arr.dims
        lat_dim = next((d for d in dims if "lat" in d), None)
        lon_dim = next((d for d in dims if "lon" in d), None)
        if lat_dim and lon_dim:
            gm = (arr * w).sum(dim=lat_dim) / w.sum() if lat_dim else arr
            gm = gm.mean(dim=lon_dim) if lon_dim in gm.dims else gm
        else:
            gm = arr.mean()
        ax.plot(time, np.asarray(gm).squeeze(), label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel("W/m²")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Global-mean radiative fluxes (diurnal cycle)")

    ax = axes[1]
    if sw_down is not None:
        arr = ds[sw_down]
        lat_dim = next((d for d in arr.dims if "lat" in d), None)
        if lat_dim is not None:
            ieq = int(np.argmin(np.abs(lat)))
            arr_eq = arr.isel({lat_dim: ieq}).squeeze()
            lon_dim = next((d for d in arr_eq.dims if "lon" in d), None)
            data = (
                arr_eq.transpose("time", lon_dim).values if lon_dim
                else arr_eq.values
            )
            im = ax.pcolormesh(lon, time, data, cmap="inferno", shading="nearest")
            plt.colorbar(im, ax=ax, label="W/m²")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Time")
            ax.set_title("TOA SW down at equator (Hovmöller)")

    plt.tight_layout()
    out = f"{output_prefix}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved diurnal plot to {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot helpers for ICON netCDFs")
    parser.add_argument("input", help="Input netCDF (from ``python -m jcm.main``)")
    parser.add_argument("--mode", choices=("climatology", "diurnal"),
                        default="climatology")
    parser.add_argument("--output_prefix", default="icon")
    args = parser.parse_args()

    import xarray as xr
    ds = xr.open_dataset(args.input)
    if args.mode == "climatology":
        plot_climatology(ds, output_prefix=args.output_prefix)
    else:
        plot_diurnal_radiation(ds, output_prefix=args.output_prefix)


if __name__ == "__main__":
    main()
