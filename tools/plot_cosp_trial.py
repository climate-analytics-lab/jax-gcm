"""Quick-look analysis of the 2M + CloudSat-COSP trial run.

Reads the chunked netCDF output, checks the cosp_* diagnostics, and plots
the warm-rain occurrence and warm-rain fraction maps plus the CloudSat
precipitation-class cover. Levels/variables are selected by coordinate
values via xarray, never by blind positional indexing (CLAUDE.md).

Usage: python tools/plot_cosp_trial.py PREFIX (e.g. .../cosp2m_260710_105513)
"""

import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main(prefix):
    files = sorted(glob.glob(f"{prefix}_day*.nc"))
    if not files:
        raise SystemExit(f"no output files match {prefix}_day*.nc")
    print("files:", [f.split("/")[-1] for f in files])
    ds = xr.open_mfdataset(files, combine="by_coords")

    cosp_vars = sorted(v for v in ds.data_vars if str(v).startswith("cosp"))
    print("\ncosp variables:", cosp_vars)
    for v in cosp_vars:
        da = ds[v]
        print(f"  {v:28s} {tuple(da.dims)} min={float(da.min()):.4g} "
              f"max={float(da.max()):.4g} mean={float(da.mean()):.4g} "
              f"NaN%={float(np.isnan(da).mean()) * 100:.2f}")

    # Use the last saved interval (most spun-up); fields are interval means
    # when output_averages=true.
    last = ds.isel(time=-1) if "time" in ds.dims else ds
    warm = last["cosp_warm_rain"] + last["cosp_warm_drizzle"]
    cold = last["cosp_cold_rain"] + last["cosp_cold_drizzle"]
    total = warm + cold
    fwarm = xr.where(total > 0, warm / total, np.nan)

    frac_raining = float((total > 0).mean())
    print(f"\ngridboxes with any (driz+)rain occurrence: {frac_raining:.1%}")
    print(f"global mean warm occurrence: {float(warm.mean()):.4f}")
    print(f"global mean cold occurrence: {float(cold.mean()):.4f}")
    print(f"area-mean f_warm (where defined): {float(fwarm.mean()):.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             constrained_layout=True)
    for ax, (da, title, kw) in zip(axes.flat, [
        (warm, "warm rain+drizzle occurrence", dict(vmin=0, cmap="Blues")),
        (cold, "cold rain+drizzle occurrence", dict(vmin=0, cmap="Purples")),
        (fwarm, "warm-rain fraction f_warm", dict(vmin=0, vmax=1, cmap="RdYlBu_r")),
        (last["cosp_pia"], "path-integrated attenuation (dB)", dict(cmap="viridis")),
    ]):
        da.plot(ax=ax, x="lon", **kw)
        ax.set_title(title)
    fig.suptitle(f"CloudSat-COSP trial, last save interval — {prefix.split('/')[-1]}")
    out = f"{prefix}_cosp_quicklook.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # CloudSat precipitation-class cover (class axis expanded to .0..9 keys).
    cover_keys = [v for v in cosp_vars if "precip_cover" in str(v)]
    if cover_keys:
        fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        names = ["no precip", "rain poss", "rain prob", "rain cert",
                 "snow poss", "snow cert", "mixed poss", "mixed cert",
                 "heavy rain", "default"]
        means = [float(last[k].mean()) for k in sorted(
            cover_keys, key=lambda k: int(str(k).split(".")[-1]))]
        ax.bar(names, means)
        ax.set_ylabel("global mean cover fraction")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("CloudSat 2C-PRECIP-COLUMN class cover")
        fig2.savefig(f"{prefix}_cosp_precip_classes.png", dpi=130)
        print("wrote", f"{prefix}_cosp_precip_classes.png")


if __name__ == "__main__":
    main(sys.argv[1])
